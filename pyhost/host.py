"""nomos pyhost — the Python decode/sampling/grammar host over the Mojo logits engine.

The kernel is a pure logits engine: prefill(ids)->logits, step(tok)->logits, prefill_cont
(KV reuse), KV cache in-kernel. ALL policy — sampling, stop, grammar — lives here in Python.

Two backends share the same decode logic via _GenMixin:
  * KernelLM     — in-process FFI (loads the .so + model in THIS process).
  * KernelClient — talks to a persistent engine daemon over a UNIX socket (kernel_client.py),
                   so the serve restarts without touching the GPU (no reload, no GB10 leak).
The mixin only uses backend primitives (prefill/step/prefill_cont/set_cache_len), so both work.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Callable, Iterable

import numpy as np

# Reuse the proven .so load + engine init from the parity harness for the spike;
# importing it has no side effects (the .so loads only on H.load_lib()).
_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, _TOOLS)
import fp4_parity_harness as H  # noqa: E402

VOCAB = 262144

LogitsProcessor = Callable[[int, list[int], np.ndarray], np.ndarray]


def _pick(logits: np.ndarray, temperature: float, top_p: float = 1.0, top_k: int = 0) -> int:
    """Greedy when temperature<=0, else temperature + nucleus (top-p) + top-k sampling.
    No repetition penalty by design — applying one on softcapped logits diverges from
    stock HF Gemma-4 (a prior band-aid we removed). Quality = nucleus, not penalties."""
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


class _GenMixin:
    """Decode logic shared by both backends. Requires the backend to provide:
    prefill(ids)->logits, step(token)->logits, set_cache_len(n), prefill_cont(suffix)->logits."""

    def generate(self, prompt_ids, max_new_tokens=1024, stop_ids=(), temperature=0.0,
                 top_p=1.0, top_k=0, processor: LogitsProcessor | None = None) -> list[int]:
        stop = set(int(s) for s in stop_ids)
        logits = self.prefill(prompt_ids)
        out: list[int] = []
        for i in range(max_new_tokens):
            lg = logits if processor is None else processor(i, out, logits)
            tok = _pick(lg, temperature, top_p, top_k)
            out.append(tok)
            if tok in stop:
                break
            logits = self.step(tok)
        return out

    def generate_stream(self, prompt_ids, max_new_tokens=1024, stop_ids=(), temperature=0.0,
                        top_p=1.0, top_k=0, processor: LogitsProcessor | None = None):
        """Yields each token id AS produced. The stop token IS yielded, then return."""
        stop = set(int(s) for s in stop_ids)
        logits = self.prefill(prompt_ids)
        out: list[int] = []
        for i in range(max_new_tokens):
            lg = logits if processor is None else processor(i, out, logits)
            tok = _pick(lg, temperature, top_p, top_k)
            out.append(tok)
            yield tok
            if tok in stop:
                return
            logits = self.step(tok)

    def prefill_reuse(self, new_ids: list[int], cached_ids: list[int], min_reuse: int = 16):
        """Prefill new_ids reusing the longest common prefix already in the KV cache.
        Returns (logits, reused_prefix_len). Fresh prefill when little to reuse."""
        p = 0
        cap = min(len(new_ids), len(cached_ids))
        while p < cap and new_ids[p] == cached_ids[p]:
            p += 1
        if p < min_reuse or p >= len(new_ids):
            return self.prefill(new_ids), 0
        self.set_cache_len(p)
        return self.prefill_cont(new_ids[p:]), p


class KernelLM(_GenMixin):
    """In-process backend: loads the .so + model in THIS process via ctypes FFI."""

    def __init__(self) -> None:
        self.lib = H.load_lib()
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
            self._dflash_avail = True
        except AttributeError:
            self._dflash_avail = False
        self.h = H.init_engine(self.lib)
        self._logits = np.zeros(VOCAB, dtype=np.float32)

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

    def spec_generate_stream(self, prompt_ids, max_new_tokens=1024, stop_ids=(), vb=8):
        """DFlash block speculative decode — greedy, lossless-by-construction.

        Yields accepted token ids AS produced (in bursts of 1..vb per cycle). The
        accepted stream is bit-identical to greedy base decode: each cycle drafts a
        block, the target verifies all rows in one fused pass, we accept the longest
        draft prefix matching the target's own argmax, then emit the target's
        correction token — so verify==decode by the arithmetic. Falls back to an exact
        M=1 step for the final token. Greedy only (spec losslessness is defined for
        greedy); callers wanting sampling use generate_stream.
        """
        if not self._dflash_loaded:
            raise RuntimeError("DFlash drafter not loaded — call dflash_load(dir) first")
        stop = set(int(s) for s in stop_ids)
        K = vb - 1
        out15 = np.zeros(15, dtype=np.int32)
        vlogits = np.zeros((vb, VOCAB), dtype=np.float32)
        self.lib.nomos_dflash_reset(self.h)
        self.prefill(prompt_ids)                       # resets KV, fills self._logits
        seed = int(np.argmax(self._logits))
        n = 0
        yield seed; n += 1
        if seed in stop:
            return
        L = self.cache_len()
        while n < max_new_tokens:
            r = max_new_tokens - n
            k = min(K, r - 1) if r > 1 else 0
            if k <= 0:                                 # 1 token budget left -> exact M=1 step
                self.step(seed)
                tok = int(np.argmax(self._logits))
                yield tok; n += 1
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
            nxt = tgt[num_acc]
            self.lib.nomos_kv_set_len(self.h, L + 1 + num_acc)
            newcl = self.lib.nomos_dflash_append_verify_context(self.h, 0, num_acc + 1)
            if newcl <= 0:
                raise RuntimeError(f"nomos_dflash_append_verify_context rc={newcl}")
            for tok in drafts[:num_acc] + [nxt]:       # emit accepted prefix + correction
                yield tok; n += 1
                if tok in stop or n >= max_new_tokens:
                    return
            L = L + 1 + num_acc
            seed = nxt

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
