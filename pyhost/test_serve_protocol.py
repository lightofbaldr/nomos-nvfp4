from __future__ import annotations

import os
import json
import threading
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from serve_protocol import (
    AtemServeProtocol,
    GemmaServeProtocol,
    load_serve_protocol,
    normalize_atem_messages,
    parse_atem_reply,
)


MUSE_TOKENIZER = Path(os.environ.get(
    "MUSE_TOKENIZER",
    os.path.expanduser("~/nomos_data/muse-glimmer-tokenizer"),
))
GEMMA_TOKENIZER = Path(os.environ.get(
    "GEMMA_TOKENIZER",
    os.path.expanduser("~/models/gemma-4-31b-it"),
))


@pytest.fixture(scope="module")
def muse_tokenizer():
    if not MUSE_TOKENIZER.is_dir():
        pytest.skip(f"Muse tokenizer absent: {MUSE_TOKENIZER}")
    return AutoTokenizer.from_pretrained(MUSE_TOKENIZER)


def test_muse_prompt_ends_at_native_assistant_generation_prefix(muse_tokenizer):
    protocol = load_serve_protocol(muse_tokenizer, 2)
    prompt = muse_tokenizer.apply_chat_template(
        protocol.normalize([{"role": "user", "content": "What is the capital of France?"}]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt = protocol.finish_prompt(prompt, thinking=True)

    assert isinstance(protocol, AtemServeProtocol)
    assert prompt.endswith("<|eot|><|start|>assistant")
    assert "<|channel>thought" not in prompt
    assert '<|start|>assistant to=self<|message|>' not in prompt


def test_muse_stop_set_ends_turn_but_not_reasoning_transition(muse_tokenizer):
    protocol = load_serve_protocol(muse_tokenizer, 2)
    assert protocol.eot_id == 200008
    assert protocol.eot_id in protocol.stop_ids
    assert muse_tokenizer.eos_token_id in protocol.stop_ids
    assert protocol.eom_id == 200007
    assert protocol.eom_id not in protocol.stop_ids


def test_atem_reply_splits_self_from_user():
    raw = (
        " to=self<|message|>The question asks for France's capital.<|eom|>"
        "<|start|>assistant to=user<|message|>Paris.<|eot|>"
    )
    parsed = parse_atem_reply(raw)
    assert parsed.reasoning == "The question asks for France's capital."
    assert parsed.content == "Paris."
    assert parsed.tool_call is None


def test_atem_stream_splits_self_from_user(muse_tokenizer):
    protocol = load_serve_protocol(muse_tokenizer, 2)
    raw = (
        " to=self<|message|>Reason briefly.<|eom|>"
        "<|start|>assistant to=user<|message|>Paris.<|eot|>"
    )
    parser = protocol.stream_parser(muse_tokenizer)
    events = []
    for token in muse_tokenizer(raw, add_special_tokens=False).input_ids:
        events.extend(parser.feed(token))
    parsed, residual = parser.finish()

    assert "".join(delta for field, delta in events if field == "reasoning_content") == "Reason briefly."
    assert "".join(delta for field, delta in events if field == "content") == "Paris."
    assert parsed.reasoning == "Reason briefly."
    assert parsed.content == "Paris."
    assert residual == []
    assert parser.saw_stop


def test_atem_tool_call_and_openai_history_conversion():
    raw = (
        " to=self<|message|>Use the weather tool.<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="days">3</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    parsed = parse_atem_reply(raw)
    assert parsed.tool_call == ("weather.get", {"city": "Paris", "days": 3})

    history = normalize_atem_messages([{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather.get", "arguments": '{"city":"Paris","days":3}'},
        }],
    }])
    assert history[0]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Paris", "days": 3,
    }


def test_gemma_prompt_and_stops_are_unchanged():
    if not GEMMA_TOKENIZER.is_dir():
        pytest.skip(f"Gemma tokenizer absent: {GEMMA_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_TOKENIZER)
    protocol = load_serve_protocol(tokenizer, 1)
    prompt = tokenizer.apply_chat_template(
        protocol.normalize([{"role": "user", "content": "What is the capital of France?"}]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt = protocol.finish_prompt(prompt, thinking=True)

    assert isinstance(protocol, GemmaServeProtocol)
    assert prompt.endswith("<|turn>model\n<|channel>thought\n")
    assert protocol.channel_end_id == 101
    assert protocol.tool_open_id == 48
    assert protocol.tool_close_id == 49
    assert protocol.stop_ids == [1, 49, 106]


class _FakeMuseLM:
    def __init__(self, tokenizer, raw: str):
        self.output = tokenizer(raw, add_special_tokens=False).input_ids
        self.prompt_ids = None

    def generate(self, prompt_ids, **kwargs):
        self.prompt_ids = prompt_ids
        return list(self.output)

    def generate_stream(self, prompt_ids, **kwargs):
        self.prompt_ids = prompt_ids
        yield from self.output

    def cache_len(self):
        return 0


def _fake_muse_engine(tokenizer, raw: str):
    protocol = load_serve_protocol(tokenizer, 2)
    return type("FakeEngine", (), {
        "tok": tokenizer,
        "protocol": protocol,
        "model_name": protocol.default_model_name,
        "lm": _FakeMuseLM(tokenizer, raw),
        "stop": protocol.stop_ids,
        "lock": threading.Lock(),
        "spec_loaded": False,
        "spec_vb": 8,
    })()


def test_nonstream_endpoint_separates_reasoning_from_user_content(muse_tokenizer):
    from serve import ChatReq, chat

    raw = (
        " to=self<|message|>Recall the capital.<|eom|>"
        "<|start|>assistant to=user<|message|>Paris.<|eot|>"
    )
    engine = _fake_muse_engine(muse_tokenizer, raw)
    # chat() reads the app's already-initialized engine; no GPU/kernel is touched here.
    import serve
    previous = getattr(serve.app.state, "engine", None)
    serve.app.state.engine = engine
    try:
        response = chat(ChatReq(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            temperature=0,
            max_tokens=1200,
        ), None)
    finally:
        if previous is None:
            del serve.app.state.engine
        else:
            serve.app.state.engine = previous

    message = response["choices"][0]["message"]
    assert message["content"] == "Paris."
    assert message["reasoning_content"] == "Recall the capital."
    assert response["choices"][0]["finish_reason"] == "stop"
    prompt = muse_tokenizer.decode(engine.lm.prompt_ids, skip_special_tokens=False)
    assert prompt.endswith("<|eot|><|start|>assistant")
    assert "<|channel>thought" not in prompt


def test_stream_endpoint_routes_reasoning_and_answer_separately(muse_tokenizer):
    from serve import ChatReq, _build_inputs, _stream

    raw = (
        " to=self<|message|>Recall the capital.<|eom|>"
        "<|start|>assistant to=user<|message|>Paris.<|eot|>"
    )
    engine = _fake_muse_engine(muse_tokenizer, raw)
    req = ChatReq(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0,
        max_tokens=1200,
        stream=True,
    )
    ids, grammar, thinking = _build_inputs(req, engine)
    chunks = list(_stream(req, engine, ids, grammar, thinking))
    payloads = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks
                if chunk.startswith("data: {")]
    deltas = [p["choices"][0]["delta"] for p in payloads]

    assert "".join(d.get("reasoning_content", "") for d in deltas) == "Recall the capital."
    assert "".join(d.get("content", "") for d in deltas) == "Paris."
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_muse_reasoning_stays_high_when_disable_is_requested(muse_tokenizer):
    from serve import ChatReq, _build_inputs

    engine = _fake_muse_engine(
        muse_tokenizer,
        " to=self<|message|>Reason.<|eom|>"
        "<|start|>assistant to=user<|message|>Paris.<|eot|>",
    )
    req = ChatReq(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        enable_thinking=False,
    )
    ids, grammar, thinking = _build_inputs(req, engine)
    rendered = muse_tokenizer.decode(ids, skip_special_tokens=False)

    assert thinking is True
    assert grammar is None
    assert "Reasoning strength: high." in rendered
    assert rendered.endswith("<|eot|><|start|>assistant")


def test_health_and_models_report_compiled_muse_profile(muse_tokenizer):
    import serve

    engine = _fake_muse_engine(muse_tokenizer, " to=user<|message|>Paris.<|eot|>")
    previous = getattr(serve.app.state, "engine", None)
    serve.app.state.engine = engine
    try:
        assert serve.health()["model"] == "nomos-muse-glimmer-30b"
        assert serve.models()["data"][0]["id"] == "nomos-muse-glimmer-30b"
    finally:
        if previous is None:
            del serve.app.state.engine
        else:
            serve.app.state.engine = previous
