#!/usr/bin/env python3
"""#431 multi-chunk parity — prove the W4A4 prefill row-chunking is BYTE-IDENTICAL to the
single-call path. A prompt longer than NOMOS_W4A4_CHUNK forces multiple chunks; compare a
small chunk (many chunks) vs a large chunk (one chunk = the original path). SWA must be OFF
(this prefills > the sliding window, which SWA guards).

Usage (full 4-bit, SWA OFF, max_seq >= prompt len):
  NOMOS_W4A4=1 NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA=0 NOMOS_MAX_SEQ=4096 \
    NOMOS_W4A4_CHUNK=4096 ... w4a4_chunk_parity.py --out /tmp/chunk_base.json    # one chunk
  NOMOS_W4A4=1 ... NOMOS_W4A4_CHUNK=256 ... w4a4_chunk_parity.py --compare /tmp/chunk_base.json  # many
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--ntok", type=int, default=3000)
    ap.add_argument("--gen", type=int, default=24)
    args = ap.parse_args()
    chunk = os.environ.get("NOMOS_W4A4_CHUNK", "?")
    print(f"=== #431 CHUNK PARITY  NOMOS_W4A4_CHUNK={chunk}  target_ntok~{args.ntok} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    base = "The quick brown fox jumps over the lazy dog near the riverbank in the morning light. "
    text = "Read the following passage carefully.\n" + base * 80 + "\nNow give a one word summary:"
    ids = np.asarray(tok(text)["input_ids"], dtype=np.int32)
    if len(ids) > args.ntok:
        ids = ids[: args.ntok]
    n = len(ids)
    print(f"  prompt tokens = {n}", flush=True)
    lib = H.load_lib()
    h = H.init_engine(lib)
    outb = np.zeros(args.gen, dtype=np.int32)
    nout = lib.nomos_generate(h, ids.ctypes.data, n, args.gen, outb.ctypes.data, args.gen,
                              0.0, -1.0, 1.0)
    greedy = outb[: max(int(nout), 0)].tolist()
    txt = tok.decode(greedy, skip_special_tokens=False).replace("\n", "\\n")
    print(f"  greedy({len(greedy)}): {txt!r}", flush=True)
    res = {"n": int(n), "greedy": greedy}
    if args.out:
        json.dump(res, open(args.out, "w"))
        print(f"  wrote baseline -> {args.out}", flush=True)
        sys.exit(0)
    if args.compare:
        ref = json.load(open(args.compare))
        ok = (ref["greedy"] == greedy) and (ref["n"] == n)
        print(f"\n=== #431 MULTI-CHUNK vs SINGLE-CHUNK: {'PASS (byte-identical)' if ok else 'FAIL'} ===",
              flush=True)
        sys.exit(0 if ok else 1)
    print("  (no --out/--compare)", flush=True)


if __name__ == "__main__":
    main()
