"""nomos pyhost serve — OpenAI-compatible HTTP face over the Mojo logits engine.

v0 of the serve that REPLACES the Go nomos-api. FastAPI app; the kernel is a pure logits
engine and ALL policy (sampling, stop, grammar, tool parsing) lives here in Python.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (non-streaming; tools + grammar-constrained tool calls)

Serial by design (Nomos is serial — one in-process engine guarded by a lock). Run on the box
that holds the .so, either:
  uvicorn pyhost.serve:app --host 0.0.0.0 --port 8090
or just:
  python pyhost/serve.py
with env: NOMOS_WEIGHT_NVFP4=0 NOMOS_PRECISION_BITS=4 NOMOS_KV_SWA=0 NOMOS_MAX_SEQ=32768
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import KernelLM, H          # noqa: E402
from grammar import ToolGrammar        # noqa: E402
from agent import parse_call           # noqa: E402  (Gemma <|"|>-arg parser → (name, dict))
from serve_protocol import load_serve_protocol  # noqa: E402
# Prompt-length ceiling. With NOMOS_BATCHED_PREFILL=1 the engine fully materialises the causal
# attention scores — lib/engine_prefill.mojo:145, NH * S * S * 4 bytes, plus a bf16 copy at :150
# (NH * S * S * 2). That is QUADRATIC in prompt length: ~192 bytes * S^2, i.e. 768 MiB at 2k
# tokens but 4.7 GiB at 5k. Engine + drafter leave only ~3.7 GiB free on the RTX PRO 4000, so an ordinary
# OpenWebUI prompt (history + tool definitions) walks off the cliff. The failure is NOT
# recoverable: cudaMalloc failing inside the kernel is a Mojo ABORT that kills the daemon
# outright — observed repeatedly on 2026-08-05, the exact allocation being
#   32 heads * 5063^2 * 4 = 3_281_148_032 bytes.
# Refusing the request is strictly better than losing the engine, so cap here and return 413.
#
# Neither kernel escape hatch is available under the champion config:
#   - long_swa (:144) skips the big buffer but requires kv_swa, which the champion env pins to 0
#   - NOMOS_PREFILL_CHUNK (RUNBOOK 3c) caps the S-scaled allocations, but on this tree it
#     produces <pad> garbage on a long prompt AND poisons the engine for every request after
#     it (reproduced 2026-08-05) — so it is not a usable workaround today.
# Turning batched prefill OFF also bounds the memory, but prefill then costs a full forward
# pass per prompt token (a 576-token prompt did not finish in 10 minutes) — not viable.
# The real fix is tiled/flash-style scores that never materialise S^2; RUNBOOK 3c already
# queues it as the context-ceiling build.
MAX_PROMPT_TOKENS = int(os.environ.get("NOMOS_MAX_PROMPT_TOKENS", "2900"))

class Engine:
    """The serial logits engine + tokenizer, loaded once at app startup.

    Held on app.state.engine. A single lock serializes generation because the kernel keeps
    one in-process KV cache — concurrent decode would corrupt it.
    """

    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
        # Daemon mode (NOMOS_ENGINE_SOCKET set): talk to the persistent engine daemon over a
        # socket — this process never loads the .so/model, so restarting the serve is instant
        # and never touches the GPU. Else load the kernel in-process (KernelLM).
        sock = os.environ.get("NOMOS_ENGINE_SOCKET")
        if sock:
            from kernel_client import KernelClient
            self.lm = KernelClient(sock)
        else:
            self.lm = KernelLM(vocab_size=len(self.tok))
        # The compiled profile, not an HTTP model string or a vocab-size heuristic,
        # selects the chat protocol. Marker IDs themselves come from the tokenizer and
        # are validated as single control tokens by load_serve_protocol().
        self.profile_id = self.lm.model_id()
        self.protocol = load_serve_protocol(self.tok, self.profile_id)
        self.model_name = self.protocol.default_model_name
        print(f"[serve] profile={self.profile_id} protocol={self.protocol.kind} "
              f"model={self.model_name}", flush=True)
        self.spec_loaded = False
        self.spec_vb = int(os.environ.get("SPEC_VB", "8"))
        # Per-model serve default (models.py env; per-request repetition_penalty overrides).
        self.rep_penalty_default = float(os.environ.get("NOMOS_SERVE_REP_PENALTY", "1.0"))
        if self.rep_penalty_default != 1.0:
            print(f"[serve] default repetition penalty {self.rep_penalty_default}", flush=True)
        dflash_dir = os.environ.get("DFLASH_DIR")
        if os.environ.get("NOMOS_SERVE_SPEC") == "1" and dflash_dir:
            try:
                self.lm.dflash_load(dflash_dir)
                self.spec_loaded = True
                print(f"[serve] DFlash speculative decode enabled (vb={self.spec_vb})",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                # Keep the endpoint available on base greedy if a configured drafter cannot
                # load. The startup log makes the degraded mode explicit.
                print(f"[serve] DFlash load failed; using base decode: {e}", flush=True)
        self.stop = self.protocol.stop_ids
        self.lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine()  # blocks until weights are resident; readiness gates on it
    yield
    # Free device memory on clean exit (GB10 unified-pool leak mitigation): a clean SIGTERM
    # now releases the cudaMalloc'd buffers. The leak only bites on SIGKILL-during-load.
    try:
        app.state.engine.lm.shutdown()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="nomos-pyhost", version="0.1.0", lifespan=lifespan)


# ── OpenAI request shape (lenient — we ignore fields we don't yet honor) ──────
class ChatReq(BaseModel):
    messages: list[dict[str, Any]]
    model: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    temperature: float = 0.0
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    # None = use the model's serve default (Engine.rep_penalty_default); an explicit value
    # always wins, including 1.0 to switch the model default off for this request.
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = False
    enable_thinking: bool = True


def _rep_penalty(req: "ChatReq", eng) -> float:
    return req.repetition_penalty if req.repetition_penalty is not None else eng.rep_penalty_default


def _visible(text: str, thinking: bool = False) -> str:
    """Strip thinking channel + turn markers → user-visible content. When `thinking` is on but the
    reasoning was truncated before its close marker, there is no answer yet → return empty."""
    if thinking and "</think>" not in text and "<channel|>" not in text:
        # thinking was on but generation was truncated mid-reasoning (no close marker) — there is no
        # answer yet; the whole output is unclosed reasoning and belongs in reasoning_content, not here.
        return ""
    if "<channel|>" in text:                 # gemma thought channel
        text = text.split("<channel|>")[-1]
    if "</think>" in text:                    # qwen3 reasoning: content is after the think block
        text = text.split("</think>")[-1]
    for marker in ("<end_of_turn>", "<turn|>", "<|im_end|>", "<|im_start|>", "<|endoftext|>", "<think>", "</think>"):
        text = text.replace(marker, "")
    return text.strip()


def parse_qwen_tool(text: str):
    """Parse a qwen3 Hermes tool call: <tool_call>{"name":..,"arguments":{..}}</tool_call>.
    Returns (name, args_dict) for the first well-formed call, else None."""
    import re
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    name = obj.get("name")
    if not name:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return name, (args if isinstance(args, dict) else {})


def _thought(text: str, thinking: bool = False) -> str:
    """The thinking channel's content — the exact complement of _visible().

    rsplit mirrors _visible()'s split(...)[-1]: whatever _visible keeps as content, this
    returns as reasoning, so between them no generated token is discarded. When `thinking`
    is on and the channel was never closed (truncated mid-reasoning), the whole output is the
    thought and _visible returns empty; with thinking off, unclosed text stays ordinary content."""
    if "</think>" in text:                    # qwen3 reasoning block (generation starts inside <think>)
        head = text.rsplit("</think>", 1)[0]
        for marker in ("<think>", "<end_of_turn>", "<turn|>"):
            head = head.replace(marker, "")
        return head.strip()
    if thinking and "<channel|>" not in text:  # unclosed reasoning (truncated): the whole output IS the thought
        for marker in ("<think>", "<end_of_turn>", "<turn|>"):
            text = text.replace(marker, "")
        return text.strip()
    if "<channel|>" not in text:
        return ""
    head = text.rsplit("<channel|>", 1)[0]
    for marker in ("<|channel>thought", "<end_of_turn>", "<turn|>"):
        head = head.replace(marker, "")
    return head.strip()


@app.get("/health")
def health():
    """Readiness that proves the ENGINE answers, not merely that the serve process exists.

    hasattr(app.state, "engine") reported ready:true through an entire outage in which the
    daemon had aborted on cudaMalloc and every request was failing (2026-08-05) — and again
    while the engine was emitting <pad>. So probe the daemon with a cheap cache_len round
    trip and return 503 when it does not answer.

    Never probe while a generation holds the lock: the client socket carries one
    request/response conversation at a time, so an overlapping call would corrupt the very
    stream this endpoint is meant to vouch for. A busy engine is by definition a live one,
    which is why a failed lock acquire reports healthy without touching the socket.
    """
    eng: Engine | None = getattr(app.state, "engine", None)
    if eng is None:
        return JSONResponse(status_code=503,
                            content={"status": "starting", "model": "loading", "ready": False})
    if not eng.lock.acquire(blocking=False):
        return {"status": "ok", "model": eng.model_name, "ready": True,
                "engine": "busy", "spec": eng.spec_loaded}
    try:
        eng.lm.cache_len()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503,
                            content={"status": "error", "model": eng.model_name, "ready": False,
                                     "detail": f"engine unreachable: {e}"})
    finally:
        eng.lock.release()
    return {"status": "ok", "model": eng.model_name, "ready": True,
            "engine": "ok", "spec": eng.spec_loaded}


@app.get("/v1/models")
def models() -> dict:
    eng: Engine = app.state.engine
    return {"object": "list",
            "data": [{"id": eng.model_name, "object": "model", "owned_by": "light-of-baldr"}]}


def _build_inputs(req: ChatReq, eng: "Engine"):
    """Render the request → (prompt token ids, grammar processor or None, thinking_on)."""
    use_tools = bool(req.tools) and req.tool_choice != "none"
    tool_names = [t["function"]["name"] for t in (req.tools or []) if "function" in t]
    tc = req.tool_choice
    # tool_choice "required" or a named-function dict forces a tool call (pydantic-ai output_type).
    force_tool = use_tools and (tc == "required" or isinstance(tc, dict))
    # On Gemma, forcing disables the thinking channel: otherwise the force depends on the model
    # emitting <channel|> before grammar can require <|tool_call>. Muse never takes that shortcut:
    # reasoning is a locked quality contract, so ATEM always generates to=self before handing off.
    # The protocol owns this distinction; separating reasoning from answer content never disables
    # generation.
    thinking = eng.protocol.resolve_thinking(req.enable_thinking, force_tool)
    try:
        msgs = eng.protocol.normalize(req.messages)
        s = eng.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
            tools=req.tools if use_tools else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    s = eng.protocol.finish_prompt(s, thinking)
    ids = list(np.asarray(eng.tok(s, add_special_tokens=False).input_ids, dtype=np.int64))
    if os.environ.get("NOMOS_DEBUG_RAW"):
        print(f"[dbg] ---- {len(req.messages)} msgs use_tools={use_tools} force={force_tool} "
              f"thinking={thinking} tc={tc!r} ntok={len(ids)} ----", flush=True)
        print(f"[dbg] RENDERED PROMPT >>>\n{s}\n<<< END PROMPT", flush=True)
    gram = (ToolGrammar(eng.tok, tool_names, force=force_tool, thinking=thinking)
            if use_tools and eng.protocol.uses_gemma_grammar else None)
    return ids, gram, thinking


def _use_spec(req: ChatReq, eng: "Engine", gram) -> bool:
    """Whether this request may use DFlash speculative decode.

    DFlash is lossless only for unmodified greedy logits, so sampling and grammar processors
    stay on base decode. Shared by the streaming and non-streaming paths so they cannot
    drift: this rule previously lived only inside _stream(), and the non-streaming path
    silently ran base decode as a result (~19.5 tok/s against streaming's ~40 on identical
    settings, which reads as a kernel regression rather than a missing branch).
    """
    return bool(eng.spec_loaded and req.temperature <= 0 and gram is None)


def _sse(cid: str, created: int, model: str, delta: dict, finish=None) -> str:
    """One OpenAI chat.completion.chunk as an SSE `data:` line."""
    payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
               "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return "data: " + json.dumps(payload) + "\n\n"


def _stream(req: ChatReq, eng: "Engine", ids: list, gram, thinking: bool = True):
    """SSE generator. Streams the Gemma thinking channel as `reasoning_content` and visible
    content token-by-token; a tool call is buffered (grammar-completed) then emitted as one
    tool_calls delta. Holds the engine lock for the whole generation (serial kernel)."""
    created = int(time.time())
    cid, model = f"chatcmpl-{created}", (req.model or eng.model_name)
    with eng.lock:
        if gram:
            gram.reset()
        yield _sse(cid, created, model, {"role": "assistant"})
        # Thinking opener is pre-filled when enable_thinking → generation starts INSIDE the
        # thought channel; suppress until <channel|>. Else it starts at the answer.
        phase = "think" if thinking else "emit"
        visible_ids: list[int] = []
        emitted = ""
        think_ids: list[int] = []
        think_emitted = ""
        tool_ids: list[int] = []
        broke_stop = False
        # All downstream thinking/tool/visibility handling is shared between both decoders.
        if _use_spec(req, eng, gram):
            token_source = eng.lm.spec_generate_stream(
                ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop, vb=eng.spec_vb)
        else:
            token_source = eng.lm.generate_stream(
                ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop,
                temperature=req.temperature, top_p=req.top_p,
                top_k=req.top_k, processor=gram,
                rep_penalty=_rep_penalty(req, eng))
        if eng.protocol.kind == "atem":
            parser = eng.protocol.stream_parser(eng.tok)
            for tok in token_source:
                for field, delta in parser.feed(tok):
                    if delta:
                        yield _sse(cid, created, model, {field: delta})
                if parser.saw_stop:
                    break
            parsed, residual = parser.finish()
            for field, delta in residual:
                if delta:
                    yield _sse(cid, created, model, {field: delta})
            if os.environ.get("NOMOS_DEBUG_RAW"):
                raw = eng.tok.decode(parser.raw_ids, skip_special_tokens=False)
                print(f"[dbg] ATEM stream raw={raw[:400]!r} parsed={parsed}", flush=True)
            if parsed.tool_call:
                name, args = parsed.tool_call
                yield _sse(cid, created, model, {"tool_calls": [{
                    "index": 0, "id": f"call_{created}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]})
                finish = "tool_calls"
            else:
                finish = "stop" if parser.saw_stop else "length"
            yield _sse(cid, created, model, {}, finish=finish)
            yield "data: [DONE]\n\n"
            return
        for tok in token_source:
            if phase == "think":
                if tok == eng.protocol.channel_end_id:
                    phase = "emit"
                    continue
                # Stream the thought channel instead of dropping it. It goes out as
                # `reasoning_content`, NOT `content`: OpenWebUI renders that delta as a
                # collapsible <think> block, while grammar/tool-call consumers that only read
                # `content` are unaffected. Dropping it also made any request whose thinking
                # outran max_tokens return a completely empty body.
                think_ids.append(tok)
                full = eng.tok.decode(think_ids, skip_special_tokens=True)
                if len(full) > len(think_emitted):
                    yield _sse(cid, created, model, {"reasoning_content": full[len(think_emitted):]})
                    think_emitted = full
                continue
            if phase == "emit":
                if tok == eng.protocol.tool_open_id:
                    phase, tool_ids = "tool", [tok]
                    continue
                if tok in eng.stop:
                    broke_stop = True
                    break
                visible_ids.append(tok)
                full = eng.tok.decode(visible_ids, skip_special_tokens=True)
                if len(full) > len(emitted):
                    yield _sse(cid, created, model, {"content": full[len(emitted):]})
                    emitted = full
            else:  # tool
                tool_ids.append(tok)
                if tok == eng.protocol.tool_close_id:
                    break
        if os.environ.get("NOMOS_DEBUG_RAW"):
            print(f"[dbg] stream-end phase={phase} nvis={len(visible_ids)} "
                  f"nthink={len(think_ids)} ntool={len(tool_ids)} "
                  f"emitted={emitted[:180]!r}", flush=True)
        if phase == "tool":
            _raw_tool = eng.tok.decode(tool_ids, skip_special_tokens=False)
            call = parse_call(_raw_tool)
            if os.environ.get("NOMOS_DEBUG_RAW"):
                print(f"[dbg] tool raw: {_raw_tool!r}", flush=True)
                print(f"[dbg] parsed: {call}", flush=True)
            if call:
                name, args = call
                yield _sse(cid, created, model, {"tool_calls": [{
                    "index": 0, "id": f"call_{created}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]})
            finish = "tool_calls"
        else:
            finish = "stop" if broke_stop else "length"
        yield _sse(cid, created, model, {}, finish=finish)
    yield "data: [DONE]\n\n"


def _next_or_none(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


async def _astream(req: ChatReq, eng: "Engine", ids: list, gram, thinking: bool, request: Request):
    """Drive _stream() while watching for client disconnect.

    Starlette iterates a sync generator in a threadpool and, when the client goes away,
    simply STOPS pulling from it — it never closes it. _stream holds `eng.lock` for the whole
    generation, so an abandoned generator parked at a yield holds that lock forever and every
    later request blocks on it: the serve wedges with the GPU at 0% (observed 2026-08-05,
    /health reporting engine="busy" with nothing running). Hitting stop in OpenWebUI does
    exactly this.

    Closing the generator in `finally` throws GeneratorExit into it, which unwinds
    `with eng.lock` and frees the engine. The close is shielded so cancellation — the very
    thing that gets us here — cannot skip the cleanup.
    """
    gen = _stream(req, eng, ids, gram, thinking)
    try:
        while True:
            chunk = await anyio.to_thread.run_sync(_next_or_none, gen)
            if chunk is None:
                return
            yield chunk
            if await request.is_disconnected():
                return
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(gen.close)


@app.post("/v1/chat/completions")
def chat(req: ChatReq, request: Request):
    eng: Engine = app.state.engine
    ids, gram, thinking = _build_inputs(req, eng)
    # One line per request naming the decoder actually chosen, and the three inputs that
    # choose it. Whether spec decode was engaged was twice inferred from a stack trace and
    # once inferred wrongly; base decode is a plausible-looking number, so a silent fallback
    # reads as a kernel regression. Cheap at one line per chat request.
    use_tools = bool(req.tools) and req.tool_choice != "none"
    print(f"[serve] req stream={req.stream} temp={req.temperature} tools={use_tools} "
          f"protocol={eng.protocol.kind} grammar={gram is not None} "
          f"spec_loaded={eng.spec_loaded} -> decoder="
          f"{'DFLASH-SPEC' if _use_spec(req, eng, gram) else 'base'} "
          f"prompt_tokens={len(ids)} max_tokens={req.max_tokens}", flush=True)
    if len(ids) > MAX_PROMPT_TOKENS:
        # Refuse rather than let batched prefill take the daemon down with it. See
        # MAX_PROMPT_TOKENS: the allocation failure is fatal, not catchable.
        raise HTTPException(status_code=413, detail=(
            f"prompt is {len(ids)} tokens; this engine accepts at most {MAX_PROMPT_TOKENS}. "
            f"Two different limits sit behind this, depending on how the engine was started. "
            f"UNCHUNKED: batched prefill materialises the S*S attention scores (~192 bytes * "
            f"S^2 — 4.7 GiB at 5k tokens) and would abort the engine on cudaMalloc. "
            f"CHUNKED (NOMOS_PREFILL_CHUNK set): prefill scratch is capped per chunk, and the "
            f"binding limit is instead NOMOS_MAX_SEQ, which must cover prompt + generated "
            f"tokens — past it the kernel silently stops before writing any logits and leaves "
            f"the engine returning NaN until restart. Shorten the conversation, or raise "
            f"NOMOS_MAX_PROMPT_TOKENS (and NOMOS_MAX_SEQ with it) if the card has room."))
    if req.stream:
        return StreamingResponse(_astream(req, eng, ids, gram, thinking, request),
                                 media_type="text/event-stream")
    with eng.lock:
        if gram:
            gram.reset()
        t0 = time.time()
        if _use_spec(req, eng, gram):
            # Same token convention as generate(): the stop token is emitted, then the
            # generator returns — so finish_reason/decoding below are unaffected.
            out = list(eng.lm.spec_generate_stream(
                ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop, vb=eng.spec_vb))
        else:
            out = eng.lm.generate(ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop,
                                  temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                                  processor=gram, rep_penalty=_rep_penalty(req, eng))
        dt = time.time() - t0
    raw = eng.tok.decode(out, skip_special_tokens=False)

    parsed_atem = eng.protocol.parse_reply(raw) if eng.protocol.kind == "atem" else None
    if parsed_atem is not None:
        call = parsed_atem.tool_call
    elif eng.protocol.kind == "qwen":
        call = parse_qwen_tool(raw)
    elif gram is not None:
        call = parse_call(raw)
    else:
        call = None
    if os.environ.get("NOMOS_DEBUG_RAW"):
        print(f"[dbg] nonstream raw={raw[:400]!r}  parsed={call}", flush=True)
    if call:
        name, args = call
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{int(t0 * 1000)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}
        finish = "tool_calls"
    else:
        if parsed_atem is not None:
            message = {"role": "assistant", "content": parsed_atem.content}
            # Muse reasoning is mandatory generation state. Keep it out of content, but expose
            # it through the same reasoning_content field as Gemma's thinking channel.
            thought = parsed_atem.reasoning
        else:
            message = {"role": "assistant", "content": _visible(raw, thinking)}
            thought = _thought(raw, thinking)
        if thought:  # surfaced alongside content, never in place of it
            message["reasoning_content"] = thought
        finish = "stop" if (out and out[-1] in eng.stop) else "length"

    return {
        "id": f"chatcmpl-{int(t0 * 1000)}",
        "object": "chat.completion",
        "created": int(t0),
        "model": req.model or eng.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                  "total_tokens": len(ids) + len(out)},
        "timing": {"gen_tokens": len(out), "seconds": round(dt, 2),
                   "tok_per_s": round(len(out) / dt, 2) if dt else 0},
        "debug_raw": raw if os.environ.get("NOMOS_DEBUG_RAW") else None,
        "debug_prompt_tail": eng.tok.decode(ids[-40:], skip_special_tokens=False)
                             if os.environ.get("NOMOS_DEBUG_RAW") else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("NOMOS_HOST", "0.0.0.0"),
                port=int(os.environ.get("NOMOS_PORT", "8090")))
