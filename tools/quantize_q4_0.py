#!/usr/bin/env python3
"""Offline Q4_0 weight quantizer for the Nomos kernel.

Group-32 symmetric affine INT4 (the format Google's gemma-4-31B-it-qat-q4_0 was
QAT-trained for — so it leverages the QAT fidelity, unlike a generic post-train codec).

Per 32-weight block:  scale = max(|w|)/7 ;  q = clip(round(w/scale), -7, 7)
Stored:  packed 4-bit (q+8 -> nibble [1..15], 2/byte)  +  one fp16 scale/block.
Effective ~4.5 bits/weight (4 + 16/32).  62 GB bf16 -> ~17 GB.

Layout of <out>.q4 file (little-endian):
    int64 n_orig            # original element count
    int64 n_blocks          # = ceil(n_orig/32)
    [n_blocks] float16      # per-block scales
    [n_blocks*16] uint8     # packed nibbles (32 weights -> 16 bytes/block)

CLI:
    quantize_q4_0.py <in.bin> <out_prefix>          # write <out_prefix>.q4
    quantize_q4_0.py --test <in.bin>                # round-trip error only
"""
import sys, numpy as np

GROUP = 32

def quantize(w: np.ndarray):
    w = w.astype(np.float32).ravel()
    n = w.size
    pad = (-n) % GROUP
    if pad:
        w = np.concatenate([w, np.zeros(pad, np.float32)])
    wb = w.reshape(-1, GROUP)                       # (nb, 32)
    amax = np.max(np.abs(wb), axis=1)               # (nb,)
    scale = (amax / 7.0).astype(np.float32)
    safe = np.where(scale == 0.0, 1.0, scale)
    q = np.clip(np.round(wb / safe[:, None]), -7, 7).astype(np.int8)   # [-7,7]
    qu = (q + 8).astype(np.uint8)                   # [1,15]
    packed = (qu[:, 0::2] | (qu[:, 1::2] << 4)).astype(np.uint8)       # (nb,16)
    return scale.astype(np.float16), packed, n

def dequant(scale16: np.ndarray, packed: np.ndarray, n: int):
    nb = packed.shape[0]
    lo = (packed & 0x0F).astype(np.int32) - 8
    hi = ((packed >> 4) & 0x0F).astype(np.int32) - 8
    q = np.empty((nb, GROUP), np.int32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return (q * scale16.astype(np.float32)[:, None]).ravel()[:n]

def write_q4(prefix, scale16, packed, n):
    nb = packed.shape[0]
    with open(prefix + ".q4", "wb") as f:
        f.write(np.array([n, nb], dtype=np.int64).tobytes())
        f.write(scale16.tobytes())
        f.write(packed.tobytes())

def main():
    a = sys.argv[1:]
    test = "--test" in a
    a = [x for x in a if x != "--test"]
    in_bin = a[0]
    w = np.fromfile(in_bin, dtype=np.float32)
    scale16, packed, n = quantize(w)
    if not test:
        write_q4(a[1], scale16, packed, n)
    # round-trip error report
    wd = dequant(scale16, packed, n)
    err = wd - w
    denom = np.maximum(np.abs(w), 1e-8)
    relmax = float(np.max(np.abs(err) / denom))
    relmean = float(np.mean(np.abs(err) / denom))
    rmse = float(np.sqrt(np.mean(err**2)))
    wn = float(np.sqrt(np.mean(w**2)))
    bytes_q4 = 16 + scale16.nbytes + packed.nbytes
    print(f"  n={n:,}  blocks={packed.shape[0]:,}")
    print(f"  bf16 {2*n/1e6:.1f}MB -> q4 {bytes_q4/1e6:.1f}MB  ({16*bytes_q4/(2*n):.2f} bits/wt)")
    print(f"  RMSE/||w|| = {rmse/wn:.4f}   rel-err mean={relmean:.4f} max={relmax:.3f}")

if __name__ == "__main__":
    main()
