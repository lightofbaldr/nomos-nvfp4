#!/usr/bin/env python3
"""Clean decode-tok/s probe for the kernel — load-time- and JIT-independent.

Loads the engine, WARMS UP (one short gen so the first-call JIT compile of every
kernel is paid and not counted), then TIMES a fixed-length greedy decode and reports
tokens/sec. Run under whatever precision env you want to compare, e.g.:

  NOMOS_WEIGHT_NVFP4=1 NOMOS_W4A4=0 ... NOMOS_NVFP4_FUSED=1  python3 tools/speed_probe.py  # fused W4A16
  NOMOS_WEIGHT_NVFP4=1 NOMOS_W4A4=0 ... NOMOS_NVFP4_FUSED=0  python3 tools/speed_probe.py  # M1 dequant->cuBLAS
  NOMOS_PRECISION_BITS=4 (q4_0)                              python3 tools/speed_probe.py  # q4 reference

Greedy (temp 0) so the token stream is deterministic; tok/s is the comparable number.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H
from transformers import AutoTokenizer

WARM = int(os.environ.get("NOMOS_PROBE_WARM", "8"))
GEN = int(os.environ.get("NOMOS_PROBE_GEN", "128"))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lib = H.load_lib()
    h = H.init_engine(lib)
    msgs = [{"role": "user", "content": "Write a short paragraph about the history of the city of Paris."}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int32)

    def gen(n: int) -> int:
        out = np.zeros(n, dtype=np.int32)
        nout = lib.nomos_generate(h, ids.ctypes.data, len(ids), n, out.ctypes.data, n, 0.0, -1.0, 1.0)
        return max(int(nout), 0)

    flags = " ".join(f"{k}={os.environ[k]}" for k in sorted(os.environ)
                     if k.startswith("NOMOS_") and k not in ("NOMOS_MAX_SEQ", "NOMOS_MODEL_PATH"))
    print(f"[probe] {flags}", flush=True)
    print(f"[probe] warmup {WARM} tok (pays first-call JIT)...", flush=True)
    gen(WARM)
    print(f"[probe] timing {GEN}-tok greedy decode...", flush=True)
    t0 = time.perf_counter()
    n = gen(GEN)
    dt = time.perf_counter() - t0
    print(f"[probe] generated {n} tok in {dt:.2f}s  =>  {n/dt:.2f} tok/s  ({1000*dt/max(n,1):.1f} ms/tok)", flush=True)


if __name__ == "__main__":
    main()
