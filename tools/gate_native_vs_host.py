#!/usr/bin/env python3
"""PREMISE GATE (Phase 0 of docs/the native-hotpath notes) — nano's acceptance arbiter.

Does the pure-Mojo `nomos_generate` (one FFI call/request: Mojo loop + on-device sampling +
stop) produce the SAME greedy tokens as the current Python per-token decode loop
(`KernelLM.generate`, which drives `nomos_decode_step` + numpy argmax)? If yes, re-pointing the
serve at nomos_generate is semantically inert and the plan is de-risked BEFORE any host rewrite.

Self-contained: binds nomos_generate itself; does NOT depend on any host.py change. Run on the
box that has the built .so + model (discrete Blackwell). Greedy only (deterministic → exact token compare).

KNOWN, EXPLAINABLE divergence source (classified, not hidden): the single-step logits path
(`nomos_decode_step`→logits_out) returns raw post-softcap logits BEFORE the engine's special-token
masking (engine_decode.mojo:598-613), while the full-gen path (nomos_generate) masks specials
before sampling. So any greedy divergence should land ONLY on a masked-special id (pad/unused/
sentinel), NEVER on a real word token. This gate flags exactly that: PASS if identical OR every
divergence is a masked-special; FAIL if any divergence is a real token (that would be a genuine
semantics gap to fix before re-pointing).

Run:  pixi run python tools/gate_native_vs_host.py
"""
import ctypes
import os
import sys

import numpy as np

# robust path: gate lives in tools/, deps are pyhost/host.py + tools/fp4_parity_harness.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pyhost"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from host import VOCAB, KernelLM  # reuse the proven .so load + engine init
import fp4_parity_harness as H

MAX_NEW = 200
STOP_IDS = [1, 106]  # eos, <turn|>  (the serve default; nomos_generate stops on these internally)

PROMPTS = [
    "Write a Python function is_prime(n). Think step by step.",
    "What is 27 times 14?",
    "Explain what a mutex is in two sentences.",
    "Write a haiku about winter.",
    "def fibonacci(n):",
    "List three primorial numbers.",
    "What is the capital of France?",
    "Reverse the string 'kernel' character by character.",
]


def _masked_specials() -> set[int]:
    """The vocab ids engine_decode.mojo:598-613 forces to -inf before sampling."""
    keep = {46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105}
    s = {0, 2, 3, 4, 5, 255999, 256000}
    s |= {i for i in range(6, 106) if i not in keep}
    s |= set(range(256001, VOCAB))
    return s


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    MASKED = _masked_specials()

    lm = KernelLM()
    # bind the pure-Mojo full-generation entrypoint
    lm.lib.nomos_generate.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int64, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lm.lib.nomos_generate.restype = ctypes.c_int32

    def native(ids: list[int]) -> list[int]:
        a = np.asarray(ids, dtype=np.int32)
        out = np.zeros(MAX_NEW, dtype=np.int32)
        n = lm.lib.nomos_generate(lm.h, a.ctypes.data, len(a), MAX_NEW,
                                  out.ctypes.data, MAX_NEW,
                                  ctypes.c_float(0.0), ctypes.c_float(1.0), ctypes.c_float(0.0))
        if n < 0:
            raise RuntimeError(f"nomos_generate rc={n}")
        return out[:n].tolist()

    n_pass = n_fail = 0
    for pi, p in enumerate(PROMPTS):
        # raw tokenization (no chat-template dep — the premise = same-input agreement between the two
        # decode paths, which is format-agnostic; both paths get identical ids)
        ids = list(np.asarray(tok(p, add_special_tokens=True).input_ids, dtype=np.int64))

        host_toks = lm.generate(ids, max_new_tokens=MAX_NEW, stop_ids=STOP_IDS, temperature=0.0)
        # fresh KV for the native call: reset the engine cache to the prompt each time
        lm.set_cache_len(0)
        nat_toks = native(ids)

        if host_toks == nat_toks:
            print(f"[{pi}] PASS  ({len(host_toks)} toks identical)  {p[:40]!r}")
            n_pass += 1
            lm.set_cache_len(0)
            continue

        # classify first divergence
        L = min(len(host_toks), len(nat_toks))
        d = next((i for i in range(L) if host_toks[i] != nat_toks[i]), L)
        ht = host_toks[d] if d < len(host_toks) else None
        nt = nat_toks[d] if d < len(nat_toks) else None
        masked = (ht in MASKED) or (nt in MASKED)
        tag = "MASKED-SPECIAL (explainable)" if masked else "REAL-TOKEN DIVERGENCE (must fix)"
        if masked:
            n_pass += 1
        else:
            n_fail += 1
        print(f"[{pi}] DIVERGE @tok{d}  host={ht}({tok.decode([ht]) if ht is not None else '-'!r}) "
              f"native={nt}({tok.decode([nt]) if nt is not None else '-'!r})  "
              f"lens h={len(host_toks)} n={len(nat_toks)}  -> {tag}  {p[:40]!r}")
        lm.set_cache_len(0)

    print(f"\nGATE: {n_pass}/{len(PROMPTS)} acceptable, {n_fail} real-token failures")
    print("PASS" if n_fail == 0 else "FAIL — real-token divergence, do NOT re-point serve yet")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
