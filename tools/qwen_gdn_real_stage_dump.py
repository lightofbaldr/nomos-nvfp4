#!/usr/bin/env python3
"""Dump real-prompt Qwen GDN layer surfaces for an external HF comparison.

This deliberately uses the production batched-prefill leg of
nomos_debug_omlp_stage_ab, not synthetic scan inputs. The supplied token IDs
are the authority; save the exact entity-probe IDs as int32 .npy and use the
same array in the HF capture.

Example:
  QWEN_SO=./libnomos_kernel-qwen3_5.so \
  QWEN_WEIGHTS=/path/to/qwen-weights/ \
  python3 tools/qwen_gdn_real_stage_dump.py \
    --ids /tmp/zephyria_ids.npy --out /tmp/kernel_gdn_l0.npz --layer 0

Output keys are named L{layer}_{surface}, each shaped [S, surface_dim].
"""
import argparse
import ctypes as C
import os
import sys

import numpy as np


STAGES = {
    18: ("input_norm", 5120),
    19: ("raw_qkv", 10240),
    20: ("raw_z", 6144),
    21: ("raw_a", 48),
    22: ("raw_b", 48),
    23: ("decay_g", 48),
    24: ("beta", 48),
    25: ("post_conv_qkv", 10240),
    13: ("recurrent_core", 6144),
    14: ("gated_norm", 6144),
    15: ("out_proj", 5120),
    16: ("post_attn_residual", 5120),
    17: ("post_mlp_residual", 5120),
}


def bind(path):
    lib = C.CDLL(path)
    lib.nomos_init.restype = C.c_int64
    lib.nomos_init.argtypes = [C.c_char_p]
    lib.nomos_shutdown.restype = None
    lib.nomos_shutdown.argtypes = [C.c_int64]
    lib.nomos_reset_kv.restype = C.c_int32
    lib.nomos_reset_kv.argtypes = [C.c_int64]
    probe = lib.nomos_debug_omlp_stage_ab
    probe.restype = C.c_int32
    probe.argtypes = [
        C.c_int64, C.c_int64, C.c_int32, C.c_int32, C.c_int32,
        C.c_int32, C.c_int32, C.c_int64, C.c_int64, C.c_int64, C.c_int64,
    ]
    return lib, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="exact int32 token IDs (.npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--stages", default="all",
        help="comma-separated stage ids; default dumps the full GDN chain",
    )
    ap.add_argument(
        "--rows", default="all",
        help="comma-separated prompt rows; default dumps every position",
    )
    args = ap.parse_args()
    so = os.environ["QWEN_SO"]
    weights = os.environ["QWEN_WEIGHTS"]
    ids = np.ascontiguousarray(np.load(args.ids), dtype=np.int32).reshape(-1)
    if not 0 < len(ids) <= 128:
        sys.exit(f"prompt length {len(ids)} outside debug contract 1..128")
    stages = list(STAGES) if args.stages == "all" else [
        int(x) for x in args.stages.split(",")
    ]
    rows = list(range(len(ids))) if args.rows == "all" else [
        int(x) for x in args.rows.split(",")
    ]
    if any(stage not in STAGES for stage in stages):
        sys.exit(f"unknown stage in {stages}; valid={list(STAGES)}")
    if any(row < 0 or row >= len(ids) for row in rows):
        sys.exit(f"row outside 0..{len(ids)-1}: {rows}")

    lib, probe = bind(so)
    handle = lib.nomos_init(weights.encode())
    if not handle:
        sys.exit("nomos_init failed")
    out = {
        "ids": ids, "layer": np.int32(args.layer),
        "row_indices": np.asarray(rows, np.int32),
    }
    meta_i32 = np.zeros(16, np.int32)
    meta_f32 = np.zeros(4, np.float32)
    scratch_dec = np.empty(max(d for _, d in STAGES.values()), np.float32)
    scratch_bat = np.empty_like(scratch_dec)
    try:
        for stage in stages:
            name, expected_dim = STAGES[stage]
            surface = np.empty((len(rows), expected_dim), np.float32)
            for out_row, row in enumerate(rows):
                if lib.nomos_reset_kv(handle) != 0:
                    sys.exit("nomos_reset_kv failed")
                meta_i32.fill(0)
                meta_f32.fill(0)
                rc = probe(
                    handle, ids.ctypes.data, len(ids), 0, row, args.layer,
                    stage, meta_i32.ctypes.data, meta_f32.ctypes.data,
                    scratch_dec.ctypes.data, scratch_bat.ctypes.data,
                )
                if rc != 0:
                    sys.exit(f"probe rc={rc} stage={stage} row={row}")
                dim = int(meta_i32[6])
                if dim != expected_dim:
                    sys.exit(
                        f"stage {stage} dimension {dim}, expected {expected_dim}"
                    )
                surface[out_row] = scratch_bat[:dim]
            key = f"L{args.layer}_{name}"
            out[key] = surface
            print(
                f"{key}: shape={surface.shape} finite={np.isfinite(surface).all()} "
                f"l2={np.linalg.norm(surface.astype(np.float64)):.9e}",
                flush=True,
            )
        np.savez_compressed(args.out, **out)
        print(f"wrote {args.out}", flush=True)
    finally:
        lib.nomos_shutdown(handle)


if __name__ == "__main__":
    main()
