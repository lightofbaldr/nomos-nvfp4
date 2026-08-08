#!/usr/bin/env python3
"""#430 SWA parity — prompt-generation byte-identity between NOMOS_KV_SWA off vs on.

Reuses fp4_parity_harness's exact prompt pass (greedy continuation + last-position
top-32 logits) and its diff. Deliberately SKIPS the KV capture->inject round-trip:
that path calls reset_kv, which under kv_quant indexes the (empty) bf16 d_k_cache and
hits the PRE-EXISTING #429 bug (cache-reset not kv-quant-aware) — orthogonal to #430.

Usage (full 4-bit env):
    NOMOS_W4A4=1 NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA=0 ... swa_parity.py --out /tmp/swa_off.json
    NOMOS_W4A4=1 NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA=1 ... swa_parity.py --compare /tmp/swa_off.json
The compare asserts the 8 prompts' ids/argmax/top_ids/top_vals/greedy are byte-identical.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare")
    args = ap.parse_args()
    print(f"=== #430 SWA PARITY (prompts only) NOMOS_KV_SWA={os.environ.get('NOMOS_KV_SWA')} ===",
          flush=True)
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lib = H.load_lib()
    h = H.init_engine(lib)
    print("--- PASS 1 ---", flush=True)
    p1 = H.run_prompts(lib, h, tok)
    print("--- PASS 2 (in-process determinism) ---", flush=True)
    p2 = H.run_prompts(lib, h, tok)
    det_ok, det_msg = H.diff_passes(p1, p2)
    print(f"  DETERMINISM (2-pass): {'PASS' if det_ok else 'FAIL'} ({det_msg})", flush=True)

    if args.out:
        json.dump({"prompts": p1}, open(args.out, "w"), indent=1)
        print(f"  wrote baseline -> {args.out}", flush=True)
        print(f"\n=== SWA-OFF BASELINE: {'PASS' if det_ok else 'FAIL'} ===", flush=True)
        sys.exit(0 if det_ok else 1)

    if args.compare:
        ref = json.load(open(args.compare))
        ok, msg = H.diff_passes(ref["prompts"], p1)
        print(f"\n=== COMPARE vs {args.compare} ===", flush=True)
        print(f"  CROSS-RUN (SWA-on vs SWA-off): {'PASS' if ok else 'FAIL'} ({msg})", flush=True)
        overall = ok and det_ok
        print(f"\n=== #430 SWA PARITY: {'PASS' if overall else 'FAIL'} ===", flush=True)
        sys.exit(0 if overall else 1)

    print("  (no --out/--compare; ran determinism only)", flush=True)
    sys.exit(0 if det_ok else 1)


if __name__ == "__main__":
    main()
