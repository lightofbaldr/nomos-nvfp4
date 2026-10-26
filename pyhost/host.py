"""nomos pyhost — the Python decode/sampling/grammar host over the Mojo logits engine.

The kernel is a pure logits engine: prefill(ids)->logits, step(tok)->logits, prefill_cont
(KV reuse), KV cache in-kernel. ALL policy — sampling, stop, grammar — lives here in Python.

Two backends share the same decode logic via _GenMixin:
  * KernelLM     — in-process FFI (loads the .so + model in THIS process).
  * KernelClient — talks to a persistent engine daemon over a UNIX socket (kernel_client.py),
                   so the serve restarts without touching the GPU (no reload, no GB10 leak).
The mixin only uses backend primitives (prefill/step/prefill_cont/session_prepare/session_commit), so both work.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from typing import Callable, Iterable

import numpy as np

# Reuse the proven .so load + engine init from the parity harness for the spike;
# importing it has no side effects (the .so loads only on H.load_lib()).
_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, _TOOLS)
import fp4_parity_harness as H  # noqa: E402

VOCAB = 262144  # maximum supported vocab / historical Gemma wire bound
# session_open()'s status for a kernel without nomos_session_open (any kernel built before ruling R79, the production
# OLMo .so among them). It is not one of the kernel's own codes (1, 0, -1, -2, -10, -11, -12).
SESSION_OPEN_UNSUPPORTED = -3


def _configured_vocab_size() -> int:
    """Read the tokenizer/model vocab when the launcher supplied TOK_DIR.

    Both supported tokenizer configs nest the text-model geometry under text_config.
    Falling back to the historical maximum keeps standalone harnesses with no tokenizer
    configured byte-for-byte compatible with the old host.
    """
    path = os.path.join(H.TOK_DIR, "config.json") if H.TOK_DIR else ""
    try:
        with open(path) as f:
            config = json.load(f)
        size = config.get("text_config", {}).get("vocab_size", config.get("vocab_size"))
        if isinstance(size, int) and 0 < size <= VOCAB:
            return size
    except (OSError, ValueError, TypeError):
        pass
    return VOCAB


def _store_requested() -> bool:
    """Whether NOMOS_KVSTORE asks for the KV session store, read the way the kernel reads it (kv_session.store_status
    over gemma4_engine._env_float): the leading decimal of the value, ASCII digits only, on above 0.5. Unset, empty and
    non-numeric are off. One corner differs: the kernel sums in Float32 and this compares in float64, so 0.50000001 is
    off there and on here."""
    m = re.match(r"-?(\d+(\.\d*)?|\.\d+)", os.environ.get("NOMOS_KVSTORE", ""), re.ASCII)
    return m is not None and float(m.group()) > 0.5


LogitsProcessor = Callable[[int, list[int], np.ndarray], np.ndarray]


def _pick(logits: np.ndarray, temperature: float, top_p: float = 1.0, top_k: int = 0) -> int:
    """Greedy when temperature<=0, else temperature + nucleus (top-p) + top-k sampling.
    _pick itself applies no repetition penalty: on softcapped logits (Gemma-4) a penalty
    diverges from stock HF, so penalty is a PER-MODEL serve default (see models.py /
    _apply_rep_penalty), off unless the model's registry entry or the request asks."""
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_k and 0 < top_k < p.size:
        thresh = np.partition(p, -top_k)[-top_k]
        p[p < thresh] = 0.0
    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        keep = np.zeros(p.size, dtype=bool)
        keep[order[:cut]] = True
        p[~keep] = 0.0
    s = p.sum()
    if s <= 0.0:
        return int(np.argmax(logits))
    p /= s
    return int(np.random.choice(p.size, p=p))


def _apply_rep_penalty(logits: np.ndarray, out: list[int], penalty: float,
                       stop: set, window: int = 64) -> np.ndarray:
    """Windowed repetition penalty, mirroring the kernel's apply_rep_penalty exactly
    (lib/sampling.mojo): last `window` GENERATED tokens, once per unique token (HF
    convention — divide positive logits, multiply negative), stop/EOS-class exempt.
    Copy-on-write so the backend's logits buffer is never mutated."""
    if penalty == 1.0 or not out:
        return logits
    lg = logits.copy()
    for t in set(out[-window:]) - stop:
        if 0 <= t < lg.size:
            lg[t] = lg[t] / penalty if lg[t] > 0 else lg[t] * penalty
    # Same-token loop safety net, ported with the penalty from engine_decode: four identical
    # trailing tokens = a degenerate loop the penalty failed to break (measured: 1.15 alone
    # leaves a residual quote-runaway rate on olmo) — force a switch. Fires only in a state
    # that is already degenerate, so it cannot alter healthy generation.
    if len(out) >= 4 and out[-1] == out[-2] == out[-3] == out[-4] and 0 <= out[-1] < lg.size:
        lg[out[-1]] = np.float32(-1e9)
    return lg


class _GenMixin:
    """Decode logic shared by both backends. Requires the backend to provide:
    prefill(ids)->logits, prefill_cont(suffix)->logits, step(token)->logits,
    session_prepare(ids, spec)->cached prefix length, session_commit(), and for multimodal turns
    prefill_mm(ids, features, nvis)->logits."""

    def _start_turn(self, prompt_ids, mm_features=None, mm_nvis=0, spec=False):
        """Prefill a turn through the KV session store: reuse the longest cached prefix, prefill the rest.
        Multimodal turns bypass the store (M-RoPE state is not saved)."""
        if mm_features is not None:
            return self.prefill_mm(prompt_ids, mm_features, mm_nvis)
        c = self.session_prepare(prompt_ids, spec=spec)
        return self.prefill(prompt_ids) if c == 0 else self.prefill_cont(prompt_ids[c:])

    def _end_turn(self):
        """Persist the turn in the KV session store. Never raises: it runs in `finally` blocks, where an
        exception would replace the one in flight, and a failed persist must not fail a reply already sent."""
        try:
            self.session_commit()
        except Exception as e:  # noqa: BLE001 (e.g. the daemon socket dropped mid-turn)
            print(f"[kvstore] commit failed: {e}", flush=True)

    def generate(self, prompt_ids, max_new_tokens=1024, stop_ids=(), temperature=0.0,
                 top_p=1.0, top_k=0, processor: LogitsProcessor | None = None,
                 mm_features=None, mm_nvis: int = 0, rep_penalty: float = 1.0) -> list[int]:
        stop = set(int(s) for s in stop_ids)
        # Multimodal: image features (if present) splice at image-token positions on the first
        # prefill via nomos_prefill_mm; decode is identical to text after that.
        logits = self._start_turn(prompt_ids, mm_features, mm_nvis)
        out: list[int] = []
        try:
            for i in range(max_new_tokens):
                lg = _apply_rep_penalty(logits, out, rep_penalty, stop)
                lg = lg if processor is None else processor(i, out, lg)
                tok = _pick(lg, temperature, top_p, top_k)
                out.append(tok)
                if tok in stop:
                    break
                logits = self.step(tok)
        finally:
            if mm_features is None:
                self._end_turn()   # persist the turn even if the caller went away (Review Focus 1)
        return out

    def generate_stream(self, prompt_ids, max_new_tokens=1024, stop_ids=(), temperature=0.0,
                        top_p=1.0, top_k=0, processor: LogitsProcessor | None = None,
                        mm_features=None, mm_nvis: int = 0, rep_penalty: float = 1.0):
        """Yields each token id AS produced. The stop token IS yielded, then return."""
        stop = set(int(s) for s in stop_ids)
        logits = self._start_turn(prompt_ids, mm_features, mm_nvis)
        out: list[int] = []
        try:
            for i in range(max_new_tokens):
                lg = _apply_rep_penalty(logits, out, rep_penalty, stop)
                lg = lg if processor is None else processor(i, out, lg)
                tok = _pick(lg, temperature, top_p, top_k)
                out.append(tok)
                yield tok
                if tok in stop:
                    return
                logits = self.step(tok)
        finally:
            if mm_features is None:
                self._end_turn()


class KernelLM(_GenMixin):
    """In-process backend: loads the .so + model in THIS process via ctypes FFI."""

    def __init__(self, vocab_size: int | None = None) -> None:
        self.lib = H.load_lib()
        self.lib.nomos_model_id.argtypes = []
        self.lib.nomos_model_id.restype = ctypes.c_int32
        self.lib.nomos_prefill.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
        self.lib.nomos_prefill.restype = ctypes.c_int32
        self.lib.nomos_decode_step.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
        self.lib.nomos_decode_step.restype = ctypes.c_int32
        self.lib.nomos_kv_cache_len.argtypes = [ctypes.c_int64]
        self.lib.nomos_kv_cache_len.restype = ctypes.c_int32
        self.lib.nomos_kv_set_len.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.lib.nomos_kv_set_len.restype = ctypes.c_int32
        self.lib.nomos_prefill_cont.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
        self.lib.nomos_prefill_cont.restype = ctypes.c_int32
        try:
            self.lib.nomos_shutdown.argtypes = [ctypes.c_int64]
            self.lib.nomos_shutdown.restype = ctypes.c_int32
        except AttributeError:
            pass
        # KV session store (docs/KV_SESSION_STORE_DESIGN.md): every turn goes prepare -> prefill -> commit. A kernel built
        # before the store has none of these exports (ledger R40). It runs with the store off, as it always did, and a
        # NOMOS_KVSTORE that asks for one is refused here, before the engine loads: the operator wants a store this
        # kernel lacks. Only the complete set of four enables the store.
        session_ffi = (("nomos_session_status", [ctypes.c_int64]),
                       ("nomos_session_prepare", [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]),
                       ("nomos_session_commit", [ctypes.c_int64]),
                       ("nomos_session_stats", [ctypes.c_int64, ctypes.c_int64]))
        self._session_ok = all(hasattr(self.lib, name) for name, _ in session_ffi)
        if self._session_ok:
            for name, args in session_ffi:
                getattr(self.lib, name).argtypes = args
                getattr(self.lib, name).restype = ctypes.c_int32
        elif _store_requested():
            raise RuntimeError(
                f"NOMOS_KVSTORE={os.environ['NOMOS_KVSTORE']} but this kernel build ({H.SO_PATH}) has no KV session "
                "store: rebuild it with refresh_build.sh, or set NOMOS_KVSTORE=0")
        # nomos_session_open (ruling R79) came later than the four: without it the store is created by the first request,
        # as before, and session_open() reports SESSION_OPEN_UNSUPPORTED.
        self._session_open_ok = self._session_ok and hasattr(self.lib, "nomos_session_open")
        if self._session_open_ok:
            self.lib.nomos_session_open.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]
            self.lib.nomos_session_open.restype = ctypes.c_int32
        # DFlash speculative-decode FFI (optional — a base-only .so still loads for
        # base decode; spec is enabled by dflash_load()).
        self._dflash_loaded = False
        self._dflash_dir = ""
        try:
            self.lib.nomos_dflash_load.argtypes = [ctypes.c_int64, ctypes.c_char_p]
            self.lib.nomos_dflash_load.restype = ctypes.c_int32
            self.lib.nomos_dflash_reset.argtypes = [ctypes.c_int64]
            self.lib.nomos_dflash_reset.restype = ctypes.c_int32
            self.lib.nomos_dflash_draft_block.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64]
            self.lib.nomos_dflash_draft_block.restype = ctypes.c_int32
            self.lib.nomos_dflash_verify_fused.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64]
            self.lib.nomos_dflash_verify_fused.restype = ctypes.c_int32
            self.lib.nomos_dflash_append_verify_context.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]
            self.lib.nomos_dflash_append_verify_context.restype = ctypes.c_int32
            self.lib.nomos_dflash_cache_len.argtypes = [ctypes.c_int64]
            self.lib.nomos_dflash_cache_len.restype = ctypes.c_int32
            self.lib.nomos_dflash_set_len.argtypes = [ctypes.c_int64, ctypes.c_int32]
            self.lib.nomos_dflash_set_len.restype = ctypes.c_int32
            self._dflash_avail = True
        except AttributeError:
            self._dflash_avail = False
        self.h = H.init_engine(self.lib)
        st = self.session_status()
        if st < 0:
            self.shutdown()   # the engine is loaded and the caller never gets this object to free it (GB10's unified pool)
            raise RuntimeError(f"NOMOS_KVSTORE=1 refused by the kernel ({st}): "
                               + {-10: "this model profile is not supported (Glimmer only)",
                                  -11: "NOMOS_KV_SWA=1 is not supported",
                                  -12: "unquantized KV is not supported"}.get(st, "unknown"))
        # The compiled profile is the vocab AUTHORITY: the .so emits exactly this many logits per
        # step, so the buffer is sized from the profile id — never from a tokenizer length (which
        # may be padded or short). Fail closed if a caller-supplied (tokenizer) vocab can't fit the
        # profile: that means TOK_DIR and the .so disagree ("what runs != what's configured").
        try:
            from models import PROFILE_VOCAB
        except ImportError:
            PROFILE_VOCAB = {}
        _profile_vocab = PROFILE_VOCAB.get(int(self.lib.nomos_model_id()))
        if _profile_vocab is not None:
            if vocab_size is not None and vocab_size > _profile_vocab:
                raise ValueError(
                    f"tokenizer vocab {vocab_size} exceeds compiled profile "
                    f"{int(self.lib.nomos_model_id())} vocab {_profile_vocab} — mispointed TOK_DIR vs .so?")
            self.vocab_size = _profile_vocab
        else:
            self.vocab_size = vocab_size if vocab_size is not None else _configured_vocab_size()
        if not 0 < self.vocab_size <= VOCAB:
            raise ValueError(f"invalid host vocab size {self.vocab_size} (max {VOCAB})")
        self._logits = np.zeros(self.vocab_size, dtype=np.float32)
        self.vlib = None  # vision backend; loaded only when NOMOS_VISION_SO is set (opt-in)
        self._load_vision()

    # ── Vision (opt-in): a SEPARATE .so encodes images → features; the TEXT .so splices them ──
    # Fully gated by NOMOS_VISION_SO. Without it, everything is byte-identical text-only. Gemma-4
    # path (absolute positions → decode loop unchanged). image_token_id lives in serve.py.
    def _load_vision(self) -> None:
        vso = os.environ.get("NOMOS_VISION_SO")
        if not vso:
            return
        if not hasattr(self.lib, "nomos_prefill_mm"):
            raise RuntimeError("NOMOS_VISION_SO set but text .so has no nomos_prefill_mm — "
                               "point SO_PATH at the mm-capable build (libnomos_kernel-gemma4.so)")
        self.lib.nomos_prefill_mm.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
                                              ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
        self.lib.nomos_prefill_mm.restype = ctypes.c_int32
        v = ctypes.CDLL(vso)
        v.nomos_gemma4_vision_init.argtypes = [ctypes.c_char_p]; v.nomos_gemma4_vision_init.restype = ctypes.c_int64
        v.nomos_gemma4_vision_free.argtypes = [ctypes.c_int64]
        v.nomos_gemma4_vision_frontend.restype = ctypes.c_int32
        v.nomos_gemma4_vision_frontend.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                                   ctypes.c_int32, ctypes.c_int64]
        v.nomos_gemma4_vision_run.restype = ctypes.c_int32
        v.nomos_gemma4_vision_run.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                              ctypes.c_int32] + [ctypes.c_int64] * 5
        blobs = os.environ["NOMOS_VISION_BLOBS"]
        self.vh = v.nomos_gemma4_vision_init(blobs.encode())
        if self.vh <= 0:
            raise RuntimeError(f"nomos_gemma4_vision_init failed for {blobs}")
        self.vlib = v
        self._vhid = 1152  # gemma4 vision hidden
        print(f"  [vision] gemma4 encoder loaded: {vso} blobs={blobs} vh={self.vh}", flush=True)

    def encode_image(self, pixel_values: np.ndarray, image_position_ids: np.ndarray):
        """gemma4 vision encoder: pixels[S,768] + positions[S,2] → (features[nvis*5376], nvis).
        Dynamic pooling → nvis is image-dependent (returned via nvis_out)."""
        pix = np.ascontiguousarray(pixel_values.reshape(-1, 768).astype(np.float32))
        S = pix.shape[0]
        pos = np.ascontiguousarray(image_position_ids.reshape(S, 2).astype(np.int32))
        H_ = self._vhid
        po = np.zeros(S * H_, np.float32)
        scratch = [np.zeros(S * H_, np.float32) for _ in range(3)]  # L0/L13/L26 (unused here)
        feats = np.zeros(((S + 8) // 9 + 4) * 5376, np.float32)
        nvis = ctypes.c_int32(0)
        rc = self.vlib.nomos_gemma4_vision_frontend(self.vh, pix.ctypes.data, pos.ctypes.data,
                                                    S, po.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"gemma4_vision_frontend rc={rc}")
        rc = self.vlib.nomos_gemma4_vision_run(
            self.vh, po.ctypes.data, pos.ctypes.data, S,
            scratch[0].ctypes.data, scratch[1].ctypes.data, scratch[2].ctypes.data,
            feats.ctypes.data, ctypes.cast(ctypes.byref(nvis), ctypes.c_void_p).value)
        if rc != 0:
            raise RuntimeError(f"gemma4_vision_run rc={rc}")
        n = nvis.value
        return np.ascontiguousarray(feats[:n * 5376].astype(np.float32)), n

    def prefill_mm(self, ids, features: np.ndarray, nvis: int) -> np.ndarray:
        """Prefill with image features spliced at image-token positions (nomos_prefill_mm)."""
        a = np.asarray(ids, dtype=np.int32)
        f = np.ascontiguousarray(features.astype(np.float32))
        rc = self.lib.nomos_prefill_mm(self.h, a.ctypes.data, len(a), f.ctypes.data,
                                       int(nvis), self._logits.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"nomos_prefill_mm rc={rc}")
        return self._logits

    def model_id(self) -> int:
        """Return the model profile compiled into this kernel."""
        return int(self.lib.nomos_model_id())

    def dflash_load(self, dflash_dir: str) -> None:
        """Load the DFlash block drafter for speculative decoding. Idempotent.

        Idempotent at the CUDA level, not merely in the flag. nomos_dflash_load allocates
        ~3.3 GB of drafter weights per call and does NOT free a previous load, while the
        engine leaves only ~5 GB free. The serve calls this on every startup against a
        long-lived daemon, so re-loading walked the daemon into
            ABORT: cudaMalloc failed: bytes=3281148032 rc=2
        and killed it on the 4th serve restart (2026-08-05). Skipping the redundant FFI call
        makes serve restarts free. A DIFFERENT directory still reloads — that is the only way
        to swap drafters, and it is rare enough to accept the allocation.
        """
        if not self._dflash_avail:
            raise RuntimeError("this libnomos_kernel.so has no DFlash FFI — rebuild with the drafter")
        if not dflash_dir.endswith("/"):
            dflash_dir += "/"          # the loader requires a trailing slash
        if self._dflash_loaded and self._dflash_dir == dflash_dir:
            return
        rc = self.lib.nomos_dflash_load(self.h, dflash_dir.encode())
        if rc != 0:
            raise RuntimeError(f"nomos_dflash_load rc={rc} (dir={dflash_dir})")
        self._dflash_loaded = True
        self._dflash_dir = dflash_dir

    def spec_generate_stream(self, prompt_ids, max_new_tokens=1024, stop_ids=(), vb=8, trace=None):
        """DFlash block speculative decode — greedy, lossless-by-construction.

        Yields accepted token ids AS produced (in bursts of 1..vb per cycle). The
        accepted stream is bit-identical to greedy base decode: each cycle drafts a
        block, the target verifies all rows in one fused pass, we accept the longest
        draft prefix matching the target's own argmax, then emit the target's
        correction token — so verify==decode by the arithmetic. Falls back to an exact
        M=1 step for the final token. Greedy only (spec losslessness is defined for
        greedy); callers wanting sampling use generate_stream.

        Turns go through the KV session store (spec=True), which keeps the drafter's
        context aligned with the KV; `trace`, if given, receives the drafts accepted
        per cycle. A turn that ends on a stop token or a closed generator ends as the
        base path's does: every token it handed out is in the KV and the drafter but
        the last, so a stop or a close inside a burst leaves nothing the client did not
        see. A turn whose budget ends at the exact M=1 step leaves the last two tokens
        without rows (ledger R54 drops that step's row to keep the drafter level).
        """
        if not self._dflash_loaded:
            raise RuntimeError("DFlash drafter not loaded — call dflash_load(dir) first")
        stop = set(int(s) for s in stop_ids)
        K = vb - 1
        out15 = np.zeros(15, dtype=np.int32)
        vlogits = np.zeros((vb, self.vocab_size), dtype=np.float32)
        c = self.session_prepare(prompt_ids, spec=True)
        if c == 0:
            self.lib.nomos_dflash_reset(self.h)
            self.prefill(prompt_ids)                   # resets KV + drafter, fills both and self._logits
        else:
            self.prefill_cont(prompt_ids[c:])          # the store kept the drafter aligned; this extends both
        if self.dflash_cache_len() != self.cache_len():
            # The drafter trails the KV. This happens when the store restored a prefix but the drafter could not hold
            # this turn's continuation (ledger R50, until Task 15 sizes it by NOMOS_MAX_SEQ), or when a continuation
            # skipped the batched capture (ledger R52). Speculation would stay lossless but accept almost nothing,
            # and a fresh prefill would redo the whole context. So this turn decodes on the base path instead:
            # greedy, the same tokens spec would emit. The target KV the store restored is kept (ledger R53).
            print("[kvstore] drafter behind the KV; this turn decodes without speculation", flush=True)
            n = 0
            try:
                tok = int(np.argmax(self._logits))
                while True:
                    yield tok; n += 1
                    if tok in stop or n >= max_new_tokens:
                        return
                    self.step(tok)
                    tok = int(np.argmax(self._logits))
            finally:
                self._end_turn()
        P = self.cache_len()                           # the KV's length once the prompt is in: the turn's rows are counted from it
        y = 0                                          # tokens handed out, counted BEFORE each yield: close() lands on one
        try:
            seed = int(np.argmax(self._logits))
            y += 1; yield seed
            if seed in stop:
                return
            L = P
            while y < max_new_tokens:
                r = max_new_tokens - y
                k = min(K, r - 1) if r > 1 else 0
                if k <= 0:                                 # 1 token budget left -> exact M=1 step
                    self.step(seed)
                    tok = int(np.argmax(self._logits))
                    # The step wrote seed's KV row, a row the drafter never gets (the decode path captures no taps).
                    # Drop it, so the turn ends with KV and drafter both at L. A turn that ends one row apart makes the
                    # store cap the next speculative request at the drafter's length, find a gap in the session's drafter
                    # rows and fork, and a fork rewrites the whole conversation. The next turn prefills seed again; the
                    # cap would have made it do so anyway. Best effort, as in the settle, which makes the same cut again
                    # in the finally below: a failed cut costs a fork next turn and must not cost this reply its last token.
                    try:
                        self.set_cache_len(L)
                    except Exception as e:  # noqa: BLE001
                        print(f"[kvstore] could not drop the exact step's row: {e}", flush=True)
                    y += 1; yield tok
                    return
                vb_eff = k + 1
                nd = self.lib.nomos_dflash_draft_block(self.h, seed, L, out15.ctypes.data)
                if nd != 15:
                    raise RuntimeError(f"nomos_dflash_draft_block rc={nd}")
                drafts = [int(x) for x in out15[:k]]
                rows = np.array([seed] + drafts, dtype=np.int32)
                rc = self.lib.nomos_dflash_verify_fused(self.h, rows.ctypes.data, vb_eff, L, vlogits.ctypes.data)
                if rc != 0:
                    raise RuntimeError(f"nomos_dflash_verify_fused rc={rc}")
                tgt = [int(np.argmax(vlogits[i])) for i in range(vb_eff)]
                num_acc = 0
                while num_acc < k and drafts[num_acc] == tgt[num_acc]:
                    num_acc += 1
                if trace is not None:
                    trace.append(num_acc)
                nxt = tgt[num_acc]
                self.lib.nomos_kv_set_len(self.h, L + 1 + num_acc)
                newcl = self.lib.nomos_dflash_append_verify_context(self.h, 0, num_acc + 1)
                if newcl <= 0:
                    raise RuntimeError(f"nomos_dflash_append_verify_context rc={newcl}")
                for tok in drafts[:num_acc] + [nxt]:       # emit accepted prefix + correction
                    y += 1; yield tok
                    if tok in stop or y >= max_new_tokens:
                        return
                L = L + 1 + num_acc
                seed = nxt
        finally:
            self._settle_spec_turn(P + max(y - 1, 0))     # ledger R55, R56
            self._end_turn()

    def _settle_spec_turn(self, keep: int) -> None:
        """End a speculative turn as the base path ends one: every token handed out has its KV row, except the last (a stop
        token, or the token whose yield close() interrupted, is never stepped). A burst's accepted drafts are in the KV and the
        drafter before the first of them is yielded, so a turn that ends inside a burst (a stop token among its drafts, a client
        that went away) holds rows the client never saw. The history it sends back then diverges from the KV inside the burst,
        and the store forks, which rewrites the whole conversation. Cut both back to `keep`, or to the shorter of the two if
        the drafter trails. The drafter goes first: if a cut fails, a drafter left behind the KV costs a re-prefill next turn,
        but one left ahead of it would make the next continuation skip feeding it. At a budget end there is nothing to cut.
        Never raises: it runs in a `finally`, where an exception would replace the one in flight."""
        try:
            kv, dr = self.cache_len(), self.dflash_cache_len()
            t = min(kv, dr, keep)
            if dr > t:
                self.dflash_set_len(t)
            if kv > t:
                self.set_cache_len(t)
        except Exception as e:  # noqa: BLE001 (a failed cut costs a fork next turn, and must not fail a reply already sent)
            print(f"[kvstore] settle failed: {e}", flush=True)

    def prefill(self, ids: list[int]) -> np.ndarray:
        a = np.asarray(ids, dtype=np.int32)
        rc = self.lib.nomos_prefill(self.h, a.ctypes.data, len(a), self._logits.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"nomos_prefill rc={rc}")
        return self._logits

    def step(self, token: int) -> np.ndarray:
        rc = self.lib.nomos_decode_step(self.h, int(token), self._logits.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"nomos_decode_step rc={rc}")
        return self._logits

    def cache_len(self) -> int:
        return int(self.lib.nomos_kv_cache_len(self.h))

    def dflash_cache_len(self) -> int:
        return int(self.lib.nomos_dflash_cache_len(self.h))

    def dflash_set_len(self, n: int) -> None:
        rc = self.lib.nomos_dflash_set_len(self.h, int(n))
        if rc != 0:
            raise RuntimeError(f"nomos_dflash_set_len rc={rc}")

    def set_cache_len(self, n: int) -> None:
        rc = self.lib.nomos_kv_set_len(self.h, int(n))
        if rc != 0:
            raise RuntimeError(f"nomos_kv_set_len rc={rc}")

    def prefill_cont(self, suffix_ids: list[int]) -> np.ndarray:
        a = np.asarray(suffix_ids, dtype=np.int32)
        rc = self.lib.nomos_prefill_cont(self.h, a.ctypes.data, len(a), self._logits.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"nomos_prefill_cont rc={rc}")
        return self._logits

    # The KV session store. A kernel built before it (self._session_ok False) runs with the store off: status 0,
    # prepare 0 (the caller prefills the whole prompt, as it always did), commit 0, stats all zeros.
    def session_status(self) -> int:
        """1 enabled and supported, 0 off (NOMOS_KVSTORE unset or 0, or a kernel without the store), negative refused:
        -10 model profile, -11 NOMOS_KV_SWA=1, -12 unquantized KV."""
        return int(self.lib.nomos_session_status(self.h)) if self._session_ok else 0

    def session_open(self) -> tuple[int, str]:
        """Create the KV session store now (ruling R79), not at the first request: (status, directory). Status 1 = open,
        with the directory resolved to its canonical path (so a trailing slash, a relative or a symlinked
        NOMOS_KVSTORE_DIR all name it one way); 0 = off; -2 = DISABLED for this process (the reason is in this process's
        log); -10 / -11 / -12 = refused; SESSION_OPEN_UNSUPPORTED = this kernel cannot open it at launch."""
        if not self._session_open_ok:
            return SESSION_OPEN_UNSUPPORTED, ""
        buf = ctypes.create_string_buffer(4096)
        st = int(self.lib.nomos_session_open(self.h, ctypes.addressof(buf), len(buf)))
        d = buf.value.decode("utf-8", "surrogateescape") if st == 1 else ""
        return st, (os.path.realpath(d) if d else "")

    def session_prepare(self, ids, spec=False) -> int:
        if not self._session_ok:
            return 0
        a = np.ascontiguousarray(ids, dtype=np.int32)
        c = self.lib.nomos_session_prepare(self.h, a.ctypes.data, len(a), 1 if spec else 0)
        if c < 0:
            raise RuntimeError(f"nomos_session_prepare rc={c}")
        return int(c)

    def session_commit(self) -> int:
        if not self._session_ok:
            return 0
        rc = self.lib.nomos_session_commit(self.h)
        if rc < 0:
            print(f"[kvstore] commit rc={rc}", flush=True)   # a failed persist must not fail the reply
        return int(rc)

    def session_stats(self) -> list[int]:
        out = np.zeros(10, dtype=np.int64)
        if self._session_ok:
            self.lib.nomos_session_stats(self.h, out.ctypes.data)
        return out.tolist()

    def shutdown(self) -> None:
        """Free device memory cleanly (GB10 unified-pool leak mitigation)."""
        try:
            if getattr(self, "h", 0):
                self.lib.nomos_shutdown(self.h)
                self.h = 0
        except Exception:  # noqa: BLE001
            pass


# ── self-test: clean generation with eos-stop, no grammar ────────────────────
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lm = KernelLM()
    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")]
    msgs = [{"role": "user", "content": "Write a Python function is_prime(n). Think step by step."}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = list(np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int64))
    out = lm.generate(ids, max_new_tokens=200, stop_ids=stop_ids)
    print(f"[host] {len(out)} tokens:", repr(tok.decode(out, skip_special_tokens=False)[:300]), flush=True)
