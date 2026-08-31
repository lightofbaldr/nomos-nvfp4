"""Per-profile chat protocol support for the OpenAI-compatible Python serve.

The kernel profile selects the protocol; the tokenizer supplies the actual marker IDs.
Keeping those two sources independent is intentional: a Muse tokenizer next to a
Gemma-built kernel (or vice versa) is a front-door error, not something to guess through.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GEMMA_PROTOCOL = 1
ATEM_PROTOCOL = 2
QWEN_PROTOCOL = 3
PROFILES_DIR = Path(__file__).resolve().parent.parent / "lib" / "model_profiles"
_ALIAS_RE = re.compile(r'^alias\s+(\w+)\s*=\s*(?:"([^"]*)"|([^#\s]+))')


@dataclass(frozen=True)
class ParsedReply:
    content: str = ""
    reasoning: str = ""
    tool_call: tuple[str, dict[str, Any]] | None = None


def _profile_serve_spec(profile_id: int) -> tuple[int, str]:
    """Read serve policy from the tracked model profile matching the kernel ID."""
    for path in sorted(PROFILES_DIR.glob("*.mojo")):
        values: dict[str, str] = {}
        for line in path.read_text().splitlines():
            match = _ALIAS_RE.match(line.strip())
            if match:
                values[match.group(1)] = match.group(2) or match.group(3)
        try:
            candidate_id = int(values["MODEL_ID"])
        except (KeyError, ValueError):
            continue
        if candidate_id != profile_id:
            continue
        try:
            return int(values["SERVE_PROTOCOL"]), values["SERVE_MODEL_NAME"]
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"model profile {path.name} has no valid SERVE_PROTOCOL/SERVE_MODEL_NAME"
            ) from exc
    raise RuntimeError(f"no tracked model profile matches compiled profile id {profile_id}")


def _token_id(tokenizer, marker: str, *, required: bool = True) -> int | None:
    """Resolve a control marker and reject the tokenizer's unknown-token sentinel."""
    tid = tokenizer.convert_tokens_to_ids(marker)
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(tid, int) and tid >= 0 and tid != unk:
        # Some tokenizers return the first piece for an ordinary string. Control markers
        # must be one token; accepting a multi-piece lookalike makes stop handling unsafe.
        pieces = tokenizer(marker, add_special_tokens=False).input_ids
        if pieces == [tid]:
            return tid
    if required:
        raise RuntimeError(f"tokenizer is missing required control token {marker!r}")
    return None


class GemmaServeProtocol:
    """Gemma-4 channel protocol. This preserves the pre-profile serve behaviour."""

    kind = "gemma"
    uses_gemma_grammar = True

    def __init__(self, tokenizer, model_name: str) -> None:
        self.default_model_name = model_name
        self.channel_end_id = _token_id(tokenizer, "<channel|>")
        self.tool_open_id = _token_id(tokenizer, "<|tool_call>")
        self.tool_close_id = _token_id(tokenizer, "<tool_call|>")
        stop = {tokenizer.eos_token_id, self.tool_close_id}
        for marker in ("<turn|>", "<end_of_turn>"):
            tid = _token_id(tokenizer, marker, required=False)
            if tid is not None:
                stop.add(tid)
        self.stop_ids = sorted(stop)

    @staticmethod
    def normalize(messages: list[dict]) -> list[dict]:
        """Fold OpenAI roles into the historical Gemma user/assistant stream."""
        out: list[dict] = []
        pending_sys = ""
        for m in messages:
            role, content = m.get("role"), m.get("content") or ""
            if role == "system":
                pending_sys += content + "\n\n"
            elif role == "tool":
                name = m.get("name") or m.get("tool_call_id") or "tool"
                out.append({"role": "user", "content": f"[Result of {name}]\n{content}"})
            elif role == "assistant":
                txt = content
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    raw = fn.get("arguments", "")
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    except json.JSONDecodeError:
                        args = {}
                    body = ",".join(f'{k}:<|"|>{v}<|"|>' for k, v in args.items())
                    txt += f"<|tool_call>call:{fn.get('name', '')}{{{body}}}<tool_call|>"
                out.append({"role": "assistant", "content": txt})
            else:
                if pending_sys:
                    content, pending_sys = f"[Instructions: {pending_sys.strip()}]\n\n{content}", ""
                out.append({"role": "user", "content": content})
        if pending_sys:
            out.append({"role": "user", "content": f"[Instructions: {pending_sys.strip()}]"})
        return out

    @staticmethod
    def finish_prompt(rendered: str, thinking: bool) -> str:
        # Gemma generation is prefilled inside its thought channel. Muse must never see
        # this suffix; its generation prompt deliberately stops after the assistant role.
        return rendered + ("<|channel>thought\n" if thinking else "")

    @staticmethod
    def resolve_thinking(requested: bool, force_tool: bool) -> bool:
        return requested and not force_tool


def normalize_atem_messages(messages: list[dict]) -> list[dict]:
    """Keep native ATEM roles and make OpenAI tool arguments Jinja-compatible.

    Muse's template requires ``tool_call.function.arguments`` to be a mapping, whereas
    OpenAI history sends it back as a JSON string. Everything else is already represented
    natively by the shipped template and should not be folded into user text.
    """
    out: list[dict] = []
    for message in messages:
        m = dict(message)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tool_call in m["tool_calls"]:
                tc = dict(tool_call)
                fn = dict(tc.get("function") or {})
                raw = fn.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"ATEM tool history has invalid JSON arguments for "
                            f"{fn.get('name', '(unnamed)')}: {exc}"
                        ) from exc
                    if not isinstance(parsed, dict):
                        raise ValueError("ATEM tool-call arguments must decode to an object")
                    fn["arguments"] = parsed
                elif raw is None:
                    fn["arguments"] = {}
                elif not isinstance(raw, dict):
                    raise ValueError("ATEM tool-call arguments must be an object")
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


_ATEM_CHANNEL_RE = re.compile(
    r"(?:<\|start\|>)?assistant(?:\s+to=([^\s<]+))?<\|message\|>"
    r"(.*?)(?=<\|eom\|>|<\|eot\|>|<\|end_of_text\|>|\Z)",
    re.DOTALL,
)
_ATEM_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>', re.DOTALL
)
_ATEM_PARAM_RE = re.compile(
    r'<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL
)


def _atem_scalar(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_atem_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse the last complete ATEM invocation from a tool-recipient channel."""
    match = None
    for match in _ATEM_INVOKE_RE.finditer(text):
        pass
    if match is None:
        return None
    name, body = match.group(1), match.group(2)
    args = {m.group(1): _atem_scalar(m.group(2)) for m in _ATEM_PARAM_RE.finditer(body)}
    return name, args


def _clean_atem_content(text: str) -> str:
    for marker in ("<|end|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"):
        text = text.replace(marker, "")
    return text.strip()


def parse_atem_reply(raw: str) -> ParsedReply:
    """Split generated ATEM messages by recipient.

    The chat template has already emitted ``<|start|>assistant``. Consequently the
    first generated channel begins with `` to=self<|message|>`` (or ``to=user``),
    while later channels include their own full start marker after ``<|eom|>``.
    """
    scan = raw if raw.startswith("<|start|>assistant") else "<|start|>assistant" + raw
    reasoning: list[str] = []
    visible: list[str] = []
    tool_call = None
    found = False
    for match in _ATEM_CHANNEL_RE.finditer(scan):
        found = True
        recipient = match.group(1) or "user"
        body = _clean_atem_content(match.group(2))
        if recipient == "self":
            if body:
                reasoning.append(body)
        elif recipient == "user":
            if body:
                visible.append(body)
        else:
            call = parse_atem_call(body)
            if call is not None:
                tool_call = call
    if not found:
        # A plain-text continuation is still a user answer. This fallback prevents a
        # malformed envelope from turning an otherwise useful response into an empty one.
        visible.append(_clean_atem_content(raw))
    return ParsedReply(
        content="\n".join(x for x in visible if x).strip(),
        reasoning="\n".join(x for x in reasoning if x).strip(),
        tool_call=tool_call,
    )


class AtemStreamParser:
    """Recipient-aware incremental parser for Muse SSE responses."""

    def __init__(self, tokenizer, protocol: "AtemServeProtocol") -> None:
        self.tokenizer = tokenizer
        self.protocol = protocol
        self.raw_ids: list[int] = []
        self.header_ids: list[int] = []
        self.body_ids: list[int] = []
        self.recipient: str | None = None
        self.body_emitted = ""
        self.content_emitted = ""
        self.reasoning_emitted = ""
        self.saw_stop = False

    def _begin_body(self) -> None:
        header = self.tokenizer.decode(self.header_ids, skip_special_tokens=False)
        match = re.search(r"(?:^|\s)to=([^\s<]+)", header)
        self.recipient = match.group(1) if match else "user"
        self.body_ids.clear()
        self.body_emitted = ""

    def _reset_header(self) -> None:
        self.header_ids.clear()
        self.body_ids.clear()
        self.body_emitted = ""
        self.recipient = None

    def feed(self, token: int) -> list[tuple[str, str]]:
        self.raw_ids.append(token)
        if self.recipient is None:
            if token == self.protocol.message_id:
                self._begin_body()
            elif token in self.protocol.stop_ids:
                self.saw_stop = True
            else:
                self.header_ids.append(token)
            return []

        if token == self.protocol.eom_id:
            self._reset_header()
            return []
        if token in self.protocol.stop_ids:
            self.saw_stop = True
            return []

        self.body_ids.append(token)
        if self.recipient not in ("self", "user"):
            return []
        full = self.tokenizer.decode(self.body_ids, skip_special_tokens=True)
        if len(full) <= len(self.body_emitted):
            return []
        delta = full[len(self.body_emitted):]
        self.body_emitted = full
        if self.recipient == "self":
            self.reasoning_emitted += delta
            return [("reasoning_content", delta)]
        self.content_emitted += delta
        return [("content", delta)]

    def finish(self) -> tuple[ParsedReply, list[tuple[str, str]]]:
        raw = self.tokenizer.decode(self.raw_ids, skip_special_tokens=False)
        parsed = parse_atem_reply(raw)
        residual: list[tuple[str, str]] = []
        # If the model omitted/garbled an envelope, parse_atem_reply's plain-text fallback
        # still gives the caller the answer. Emit only the suffix not already streamed.
        if parsed.reasoning.startswith(self.reasoning_emitted):
            tail = parsed.reasoning[len(self.reasoning_emitted):]
            if tail:
                residual.append(("reasoning_content", tail))
        if parsed.content.startswith(self.content_emitted):
            tail = parsed.content[len(self.content_emitted):]
            if tail:
                residual.append(("content", tail))
        return parsed, residual


class AtemServeProtocol:
    """Muse-Glimmer ATEM protocol selected by the compiled Muse profile."""

    kind = "atem"
    uses_gemma_grammar = False

    def __init__(self, tokenizer, model_name: str) -> None:
        self.default_model_name = model_name
        self.start_id = _token_id(tokenizer, "<|start|>")
        self.message_id = _token_id(tokenizer, "<|message|>")
        self.eom_id = _token_id(tokenizer, "<|eom|>")
        self.eot_id = _token_id(tokenizer, "<|eot|>")
        self.stop_ids = sorted({tokenizer.eos_token_id, self.eot_id})
        if self.eom_id in self.stop_ids:
            raise RuntimeError("ATEM <|eom|> must transition channels, not stop generation")

    @staticmethod
    def normalize(messages: list[dict]) -> list[dict]:
        return normalize_atem_messages(messages)

    @staticmethod
    def finish_prompt(rendered: str, thinking: bool) -> str:
        del thinking
        return rendered

    @staticmethod
    def resolve_thinking(requested: bool, force_tool: bool) -> bool:
        # Glimmer quality is trained/evaluated with reasoning ON. This is generation
        # policy, independent of routing to=self through a separate response field.
        del requested, force_tool
        return True

    @staticmethod
    def parse_reply(raw: str) -> ParsedReply:
        return parse_atem_reply(raw)

    def stream_parser(self, tokenizer) -> AtemStreamParser:
        return AtemStreamParser(tokenizer, self)


class QwenServeProtocol:
    """Qwen3.5 ChatML protocol (<|im_start|>/<|im_end|>). Standard chat — no gemma thought channel,
    no grammar. Routes through the default (non-atem) streaming path; the gemma-specific channel/tool
    marker ids are set to an unmatchable sentinel (-1; real token ids are >= 0) so that path emits
    plain text until a stop id. Tool-call / <think> parsing is a follow-up; base chat is exact here."""

    kind = "qwen"
    uses_gemma_grammar = False

    def __init__(self, tokenizer, model_name: str) -> None:
        self.default_model_name = model_name
        self.channel_end_id = -1
        self.tool_open_id = -1
        self.tool_close_id = -1
        stop = {tokenizer.eos_token_id}
        for marker in ("<|im_end|>", "<|endoftext|>"):
            tid = _token_id(tokenizer, marker, required=False)
            if tid is not None:
                stop.add(tid)
        self.stop_ids = sorted(t for t in stop if t is not None)
        # ChatML models differ on reasoning: qwen3's template branches on enable_thinking and
        # pre-fills a <think> opener; olmo3's ignores the flag and answers plainly. Starting the
        # stream in the think phase against a template that never opens a think block suppresses
        # the ENTIRE reply (no </think> ever arrives), so thinking is only honoured when the
        # template can actually render it.
        self.supports_thinking = "enable_thinking" in (getattr(tokenizer, "chat_template", None) or "")

    @staticmethod
    def normalize(messages: list[dict]) -> list[dict]:
        # Qwen's chat template consumes OpenAI roles (system/user/assistant/tool) directly.
        return messages

    @staticmethod
    def finish_prompt(rendered: str, thinking: bool) -> str:
        del thinking  # qwen thinking is driven by the template's enable_thinking flag, not a suffix
        return rendered

    def resolve_thinking(self, requested: bool, force_tool: bool) -> bool:
        del force_tool
        return requested and self.supports_thinking


def load_serve_protocol(tokenizer, profile_id: int):
    """Construct the protocol named by the kernel's compiled profile.

    Profile identity comes from ``nomos_model_id()`` (directly or through the daemon),
    never from a filename, model request string, or vocab-size heuristic.
    """
    protocol_id, model_name = _profile_serve_spec(profile_id)
    if protocol_id == GEMMA_PROTOCOL:
        return GemmaServeProtocol(tokenizer, model_name)
    if protocol_id == QWEN_PROTOCOL:
        return QwenServeProtocol(tokenizer, model_name)
    if protocol_id == ATEM_PROTOCOL:
        return AtemServeProtocol(tokenizer, model_name)
    raise RuntimeError(
        f"compiled profile id {profile_id} selects unsupported serve protocol {protocol_id}"
    )
