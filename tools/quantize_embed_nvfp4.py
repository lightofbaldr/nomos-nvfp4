#!/usr/bin/env python3
"""Quantize the fp32 embed/unembed weight to NVFP4 for the W4A4 lm-head.

The W4A4 serve path routes the vocab projection through a direct NVFP4 GEMV that reads
`embed_tokens_weight.nvfp4`; if only the fp32 `embed_tokens_weight.bin` is present the loader
returns 0 and the kernel hits a CUDA illegal-address (guarded fail-fast in a86c19c). This makes
the missing `.nvfp4` from the fp32 `.bin` that every serve dir already ships.

Gemma-4-31B: vocab 262144, hidden 5376 → embed shape [262144, 5376].

    python3 tools/quantize_embed_nvfp4.py <serve_dir> [--vocab 262144 --hidden 5376]

Idempotent: skips if `embed_tokens_weight.nvfp4` already exists (pass --force to overwrite).
"""
import argparse
import os
import sys

import numpy as np

# reuse the exact encoder the rest of the kernel dir was built with
from quantize_nvfp4 import write_nvfp4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("serve_dir", help="flat kernel serve dir containing embed_tokens_weight.bin")
    ap.add_argument("--vocab", type=int, default=262144)
    ap.add_argument("--hidden", type=int, default=5376)
    ap.add_argument("--force", action="store_true", help="overwrite an existing .nvfp4")
    a = ap.parse_args()

    src = os.path.join(a.serve_dir, "embed_tokens_weight.bin")
    dst = os.path.join(a.serve_dir, "embed_tokens_weight.nvfp4")
    if not os.path.exists(src):
        print(f"ERROR: {src} not found (need the fp32 embed to quantize)", file=sys.stderr)
        return 1
    if os.path.exists(dst) and not a.force:
        print(f"exists, skipping (pass --force to overwrite): {dst}")
        return 0

    raw = np.fromfile(src, dtype=np.float32)
    n = a.vocab * a.hidden
    if raw.size != n:
        print(
            f"ERROR: {src} has {raw.size} floats, expected vocab*hidden = "
            f"{a.vocab}*{a.hidden} = {n}",
            file=sys.stderr,
        )
        return 1

    W = raw.reshape(a.vocab, a.hidden)
    write_nvfp4(dst, W)
    print(f"wrote {dst}  ({os.path.getsize(dst):,} B)  from {a.vocab}x{a.hidden} fp32 embed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
