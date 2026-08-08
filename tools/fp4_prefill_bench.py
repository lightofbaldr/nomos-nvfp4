"""Prefill-isolated speed bench: prefill+1tok across S, to characterize the naive
fp4_gemm (W4A4 prefill) vs the dequant->bf16->cuBLAS path (W4A16 prefill) as the
sequence grows. W4A16 amortizes its per-layer dequant over S, so this is where the
one-warp-per-tile kernel could lose; the crossover (if any) tells us whether to tile.

Reuses the harness FFI/init. Run with the SAME env that selects the mode:
  W4A4 :  WEIGHTS=... NOMOS_W4A4=1          python3 tools/fp4_prefill_bench.py
  W4A16:  WEIGHTS=... NOMOS_WEIGHT_NVFP4=1  python3 tools/fp4_prefill_bench.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H

SEQ_LENS = [128, 512, 2048, 4096]
REPEATS = 3


def main():
    lib = H.load_lib()
    h = H.init_engine(lib)
    mode = "W4A4" if os.environ.get("NOMOS_W4A4", "0") == "1" else \
           ("W4A16" if os.environ.get("NOMOS_WEIGHT_NVFP4", "0") == "1" else "bf16")
    print(f"=== PREFILL BENCH ({mode}) — prefill+1tok, median of {REPEATS} ===", flush=True)
    outb = np.zeros(4, dtype=np.int32)
    for S in SEQ_LENS:
        # deterministic in-vocab token ids (vocab ~256k; <100k is safe)
        ids = ((np.arange(S, dtype=np.int32)) % 100000 + 1).astype(np.int32)
        # warmup (compile + cache warm)
        lib.nomos_generate(h, ids.ctypes.data, S, 1, outb.ctypes.data, 4, 0.0, -1.0, 1.0)
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            lib.nomos_generate(h, ids.ctypes.data, S, 1, outb.ctypes.data, 4, 0.0, -1.0, 1.0)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[1]
        print(f"  S={S:5}  prefill+1 = {med*1000:9.1f} ms   = {S/med:9.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
