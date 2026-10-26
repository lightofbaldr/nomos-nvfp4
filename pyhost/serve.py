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
from contextlib import ExitStack, asynccontextmanager
from typing import Any

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import KernelLM, H, SESSION_OPEN_UNSUPPORTED, _store_requested   # noqa: E402
from grammar import ToolGrammar        # noqa: E402
from agent import parse_call           # noqa: E402  (Gemma <|"|>-arg parser → (name, dict))
from serve_protocol import atem_exact_fields, load_serve_protocol  # noqa: E402
# Prompt-length ceiling. With NOMOS_BATCHED_PREFILL=1 the engine fully materialises the causal
# attention scores — lib/engine_prefill.mojo:145, NH * S * S * 4 bytes, plus a bf16 copy at :150
# (NH * S * S * 2). That is QUADRATIC in prompt length: ~192 bytes * S^2, i.e. 768 MiB at 2k
# tokens but 4.7 GiB at 5k. Engine + drafter leave only ~3.7 GiB free on Gold, so an ordinary
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
# This cap now applies only when NOMOS_PREFILL_CHUNK is unset (_fit_request below); chunked, NOMOS_MAX_SEQ binds.
# (The NOMOS_PREFILL_CHUNK bullet above predates the #74 capacity guard: see RUNBOOK 3b, 3c and 3d.)
MAX_PROMPT_TOKENS = int(os.environ.get("NOMOS_MAX_PROMPT_TOKENS", "2900"))
MAX_SEQ = int(os.environ.get("NOMOS_MAX_SEQ", "4096"))
PREFILL_CHUNK = int(float(os.environ.get("NOMOS_PREFILL_CHUNK", "0") or 0))


def _fit_request(n_prompt: int, max_tokens: int) -> int:
    """How many tokens a request may generate. With NOMOS_PREFILL_CHUNK set, prefill memory is capped per chunk and the
    binding limit is the KV capacity: prompt + generated must fit in NOMOS_MAX_SEQ (engine_decode's #74 capacity guard
    refuses anything past it). An unchunked engine keeps the O(S^2) prompt cap explained above."""
    if PREFILL_CHUNK <= 0 and n_prompt > MAX_PROMPT_TOKENS:
        raise HTTPException(status_code=413, detail=(
            f"prompt is {n_prompt} tokens; this engine accepts at most {MAX_PROMPT_TOKENS} with unchunked prefill: "
            f"batched prefill materialises the S*S attention scores (~192 bytes * S^2, 4.7 GiB at 5k tokens) and "
            f"would abort the engine on cudaMalloc. Set NOMOS_PREFILL_CHUNK (RUNBOOK 3c) to serve long "
            f"conversations, or shorten this one."))
    room = MAX_SEQ - n_prompt - 1
    if room < 1:
        raise HTTPException(status_code=413, detail=(
            f"prompt is {n_prompt} tokens; NOMOS_MAX_SEQ={MAX_SEQ} leaves no room to generate. "
            f"Shorten the conversation or raise NOMOS_MAX_SEQ."))
    return min(max_tokens, room)


_KVSTORE_STATUS = {1: "open", 0: "off", -2: "disabled", SESSION_OPEN_UNSUPPORTED: "off (unsupported)",
                   -10: "refused (-10: this model profile is not Muse)", -11: "refused (-11: NOMOS_KV_SWA=1)",
                   -12: "refused (-12: unquantized KV)"}


def _open_kvstore(lm) -> dict:
    """Open the KV session store at startup, not at the first chat request (ruling R79), so its directory lock, its startup
    scan and any refusal happen at launch, where the operator looks, and /health can report it. Returns /health's
    `kvstore`: its status ("open", "off", "disabled": another process holds the directory or a setting is invalid, and
    daemon.log says which; "off (unsupported)": a kernel that cannot open it at launch) and the store's canonical
    directory while it is open. Never asks the engine when NOMOS_KVSTORE is off, as the launcher forces it for every model
    but Muse: an engine of another model, or an older daemon, then sees no new request."""
    if not _store_requested():
        return {"status": "off", "dir": None}
    status, store_dir = lm.session_open()
    return {"status": _KVSTORE_STATUS.get(status, f"error ({status})"), "dir": store_dir or None}


IMAGE_TOKEN_ID = 258880  # Gemma-4 <image_soft_token> — vision features splice at these positions
# Untrusted request input drives image loading — secure by default: inline base64 only, byte- and
# pixel-capped; local paths (file-read/path-traversal) and remote fetch (SSRF) are opt-in via env.
_MAX_IMG_BYTES = int(os.environ.get("NOMOS_VISION_MAX_BYTES", 12 * 1024 * 1024))    # 12 MiB
_MAX_IMG_PIXELS = int(os.environ.get("NOMOS_VISION_MAX_PIXELS", 40 * 1024 * 1024))  # 40 MP


class Engine:
    """The serial logits engine + tokenizer, loaded once at app startup.

    Held on app.state.engine. A single lock serializes generation because the kernel keeps
    one in-process KV cache — concurrent decode would corrupt it.
    """

    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
        # Vision is OPT-IN: only when NOMOS_VISION_SO is set do we load the image processor. A text
        # request never touches it — the processor is consulted only when a request carries an image.
        self.proc = None
        if os.environ.get("NOMOS_VISION_SO"):
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(
                os.environ.get("NOMOS_PROC_DIR", os.path.expanduser("~/nomos_data/gemma-4-31b-it-bf16")))
            print(f"  [vision] image processor loaded (image_token_id={IMAGE_TOKEN_ID})", flush=True)
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
        # What this serve admits (_fit_request). The daemon reads the same variables; a serve started without the launcher
        # can differ from it, and this line is where that shows.
        print(f"[serve] NOMOS_MAX_SEQ={MAX_SEQ} NOMOS_PREFILL_CHUNK={PREFILL_CHUNK} "
              f"NOMOS_MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}", flush=True)
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
        # After the drafter decision: the store's cache key includes the drafter, so it is final only now.
        self.kvstore = _open_kvstore(self.lm)
        print(f"[serve] kvstore {self.kvstore['status']}" + (f" dir={self.kvstore['dir']}" if self.kvstore["dir"] else ""),
              flush=True)
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

    `kvstore` is the KV session store as the serve opened it at startup (_open_kvstore): its status and, while it is
    open, its canonical directory (tools/kvstore_dirs.sh reads it to know which directory is live).
    """
    eng: Engine | None = getattr(app.state, "engine", None)
    if eng is None:
        return JSONResponse(status_code=503,
                            content={"status": "starting", "model": "loading", "ready": False})
    if not eng.lock.acquire(blocking=False):
        return {"status": "ok", "model": eng.model_name, "ready": True,
                "engine": "busy", "spec": eng.spec_loaded, "kvstore": eng.kvstore}
    try:
        eng.lm.cache_len()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503,
                            content={"status": "error", "model": eng.model_name, "ready": False,
                                     "detail": f"engine unreachable: {e}", "kvstore": eng.kvstore})
    finally:
        eng.lock.release()
    return {"status": "ok", "model": eng.model_name, "ready": True,
            "engine": "ok", "spec": eng.spec_loaded, "kvstore": eng.kvstore}


_KVSTORE_STATS = ["prepares", "prompt_tokens", "reused_tokens", "restores_ram", "restores_nvme", "forks",
                  "ram_bytes", "nvme_bytes", "ram_evictions", "nvme_evictions"]


_STATS_WAIT_S = 60.0   # how long GET /v1/kvstore/stats waits for a running generation before it answers 503 busy


@app.get("/v1/kvstore/stats")
async def kvstore_stats():
    """KV session store counters (docs/KV_SESSION_STORE_DESIGN.md §3.1). Waits for a running generation:
    the engine socket carries one conversation at a time.

    Async on purpose (ledger R43). A sync `def` runs on anyio's 40-thread pool and would keep its thread through the
    whole wait for eng.lock, while the generation that holds the lock needs a pool thread for every chunk it streams
    (_astream): about 40 overlapping polls would freeze it, and the GPU would idle with the lock held. So the wait
    polls acquire(blocking=False) on the event loop and holds no thread; a thread is taken only for the stats call
    itself, once the lock is ours."""
    eng: Engine | None = getattr(app.state, "engine", None)
    if eng is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    deadline = time.monotonic() + _STATS_WAIT_S
    while not eng.lock.acquire(blocking=False):
        if time.monotonic() >= deadline:
            return JSONResponse(status_code=503, content={"status": "busy"})
        await anyio.sleep(0.05)
    try:
        # The thread is using the engine socket: abandon_on_cancel stays False (anyio's default), so a cancelled
        # request waits for the call to finish and only then releases the lock.
        return dict(zip(_KVSTORE_STATS, await anyio.to_thread.run_sync(eng.lm.session_stats)))
    finally:
        eng.lock.release()


@app.get("/v1/models")
def models() -> dict:
    eng: Engine = app.state.engine
    return {"object": "list",
            "data": [{"id": eng.model_name, "object": "model", "owned_by": "light-of-baldr"}]}


def _decode_image(raw: bytes):
    """Bytes → RGB PIL image, size- and pixel-bounded (rejects oversized inputs + decode bombs)."""
    import io
    from PIL import Image
    if len(raw) > _MAX_IMG_BYTES:
        raise HTTPException(status_code=413, detail=f"image exceeds NOMOS_VISION_MAX_BYTES ({_MAX_IMG_BYTES})")
    Image.MAX_IMAGE_PIXELS = _MAX_IMG_PIXELS
    try:
        im = Image.open(io.BytesIO(raw)); im.load()      # force decode now, bounded by MAX_IMAGE_PIXELS
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc
    return im.convert("RGB")


def _load_image(part: dict):
    """One OpenAI image content part → PIL.Image. data: URLs (base64) always; local paths and http(s)
    URLs only when explicitly enabled (file-read / SSRF surface). None if not an image part."""
    import base64
    url = None
    if part.get("type") == "image_url":
        u = part.get("image_url"); url = u.get("url") if isinstance(u, dict) else u
    elif part.get("type") == "image":
        url = part.get("image") or part.get("url")
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        return _decode_image(base64.b64decode(url.split(",", 1)[1]))
    if url.startswith(("http://", "https://")):
        if not os.environ.get("NOMOS_VISION_ALLOW_URL"):
            raise HTTPException(status_code=400, detail="remote image URLs disabled (SSRF); set NOMOS_VISION_ALLOW_URL=1")
        import urllib.request
        r = urllib.request.Request(url, headers={"User-Agent": "nomos"})
        with urllib.request.urlopen(r, timeout=15) as resp:  # noqa: S310
            return _decode_image(resp.read(_MAX_IMG_BYTES + 1))
    if not os.environ.get("NOMOS_VISION_ALLOW_LOCAL"):
        raise HTTPException(status_code=400, detail="local image paths disabled; set NOMOS_VISION_ALLOW_LOCAL=1")
    with open(url, "rb") as f:
        return _decode_image(f.read(_MAX_IMG_BYTES + 1))


def _messages_have_image(messages: list[dict]) -> bool:
    return any(isinstance(m.get("content"), list) and
               any(isinstance(p, dict) and p.get("type") in ("image_url", "image")
                   for p in m["content"]) for m in messages)


def _to_processor_messages(messages: list[dict]):
    """OpenAI messages (content may be a multi-part list) → Gemma processor chat format,
    image parts materialized as PIL images; non-image parts fold to text."""
    out = []
    for m in messages:
        role, c = m.get("role", "user"), m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if not isinstance(p, dict):
                    parts.append({"type": "text", "text": str(p)}); continue
                if p.get("type") in ("image_url", "image"):
                    img = _load_image(p)
                    if img is not None:
                        parts.append({"type": "image", "image": img})
                else:
                    parts.append({"type": "text", "text": p.get("text", "")})
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": c or ""})
    return out


def _build_vision_inputs(req: ChatReq, eng: "Engine"):
    """Image request → (ids, image_features, nvis). Uses the HF processor to interleave image
    placeholder tokens + produce pixel_values, then the kernel vision encoder for features."""
    msgs = _to_processor_messages(req.messages)
    inp = eng.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="np")
    ids = list(np.asarray(inp["input_ids"][0], dtype=np.int64))
    feats, nvis = eng.lm.encode_image(np.asarray(inp["pixel_values"]),
                                      np.asarray(inp["image_position_ids"]))
    if os.environ.get("NOMOS_DEBUG_RAW"):
        n_img = int((np.asarray(ids) == IMAGE_TOKEN_ID).sum())
        print(f"[dbg] VISION ntok={len(ids)} img_placeholders={n_img} nvis={nvis}", flush=True)
    return ids, feats, nvis


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


def _stream(req: ChatReq, eng: "Engine", ids: list, gram, thinking: bool = True,
            mm_feats=None, mm_nvis: int = 0):
    """SSE generator. Streams the Gemma thinking channel as `reasoning_content` and visible
    content token-by-token; a tool call is buffered (grammar-completed) then emitted as one
    tool_calls delta. Holds the engine lock for the whole generation (serial kernel)."""
    created = int(time.time())
    cid, model = f"chatcmpl-{created}", (req.model or eng.model_name)
    with eng.lock, ExitStack() as turn:
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
        if mm_feats is not None:
            # Vision: base decode with spliced image features (spec decode is not mm-compatible).
            token_source = eng.lm.generate_stream(
                ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop,
                temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                processor=gram, mm_features=mm_feats, mm_nvis=mm_nvis,
                rep_penalty=_rep_penalty(req, eng))
        elif _use_spec(req, eng, gram):
            token_source = eng.lm.spec_generate_stream(
                ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop, vb=eng.spec_vb)
        else:
            token_source = eng.lm.generate_stream(
                ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop,
                temperature=req.temperature, top_p=req.top_p,
                top_k=req.top_k, processor=gram,
                rep_penalty=_rep_penalty(req, eng))
        # A token source left suspended (the loop broke at the stop token, or the client went away) is finalised
        # only when this frame dies, after the lock is released. Its `finally` commits the turn to the KV session
        # store, and that commit must not run unlocked, racing the next request's prepare on the same engine
        # (or the same daemon socket). The ExitStack unwinds before the lock does.
        turn.callback(token_source.close)
        if eng.protocol.kind == "atem":
            parser = eng.protocol.stream_parser(eng.tok)
            for tok in token_source:
                for field, delta in parser.feed(tok):
                    if delta:
                        yield _sse(cid, created, model, {field: delta})
                if parser.saw_stop:
                    break
            if parser.saw_stop:
                # After the stop token a source has nothing left to hand out: resume it once, so it ends on its own (from
                # the daemon, its end frame) and a turn that ends normally keeps its daemon connection. Closing it instead
                # would close a speculative stream mid-way, which drops the socket, and the next request reconnects. The
                # ExitStack's close stays for a client that goes away mid-stream, and for a drain that fails: the reply is
                # complete by now.
                try:
                    next(token_source, None)
                except Exception as e:  # noqa: BLE001
                    print(f"[serve] the token source failed after the stop token: {e}", flush=True)
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


async def _astream(req: ChatReq, eng: "Engine", ids: list, gram, thinking: bool, request: Request,
                   mm_feats=None, mm_nvis: int = 0):
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
    gen = _stream(req, eng, ids, gram, thinking, mm_feats, mm_nvis)
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
    # Vision is opt-in per request: only a request carrying an image (and only when the processor
    # is loaded) takes the multimodal path. Everything else is the unchanged text path.
    mm_feats = None; mm_nvis = 0
    if eng.proc is not None and _messages_have_image(req.messages):
        ids, mm_feats, mm_nvis = _build_vision_inputs(req, eng)
        gram, thinking = None, False
    else:
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
    req.max_tokens = _fit_request(len(ids), req.max_tokens)
    if req.stream:
        return StreamingResponse(_astream(req, eng, ids, gram, thinking, request, mm_feats, mm_nvis),
                                 media_type="text/event-stream")
    with eng.lock:
        if gram:
            gram.reset()
        t0 = time.time()
        if mm_feats is not None:
            # Vision: base decode with spliced image features (spec decode is not mm-compatible).
            out = eng.lm.generate(ids, max_new_tokens=req.max_tokens, stop_ids=eng.stop,
                                  temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                                  processor=None, mm_features=mm_feats, mm_nvis=mm_nvis,
                                  rep_penalty=_rep_penalty(req, eng))
        elif _use_spec(req, eng, gram):
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

    # The fields a streaming client would join, unstripped (Task 10b): the history it sends back must re-render to
    # the token ids the KV holds. parse_atem_reply strips each body, so only its tool call is used.
    if eng.protocol.kind == "atem":
        reasoning, content, parsed_atem = atem_exact_fields(eng.tok, eng.protocol, out)
    else:
        reasoning = content = parsed_atem = None
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
            message = {"role": "assistant", "content": content}
            # Muse reasoning is mandatory generation state. Keep it out of content, but expose
            # it through the same reasoning_content field as Gemma's thinking channel.
            thought = reasoning
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
