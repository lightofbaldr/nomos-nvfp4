#!/usr/bin/env python3
"""Dump Qwen attention/gate surfaces at the rare-entity copy anchor.

Reads exact token IDs from qwen_lens_kernel.npz. The production batched-prefill
leg is captured with out_decode=-1, so no synthetic operands or sequential
replay contaminate the measurement.
"""
import argparse
import ctypes as C
import os
import sys

import numpy as np


ANCHOR_STAGES = {
    26: ("q_pre_rope", 6144),
    8: ("q_post_rope", 6144),
    12: ("qk_scores", None),       # [24, anchor+1]
    0: ("attn_pre_gate", 6144),
    27: ("gate_raw", 6144),
    28: ("attn_post_gate", 6144),
    1: ("o_proj", 5120),
    3: ("post_attn_residual", 5120),
    7: ("layer_out", 5120),
}


def bind(path):
    lib = C.CDLL(path)
    lib.nomos_init.restype = C.c_int64
    lib.nomos_init.argtypes = [C.c_char_p]
    lib.nomos_shutdown.restype = None
    lib.nomos_shutdown.argtypes = [C.c_int64]
    lib.nomos_reset_kv.restype = C.c_int32
    lib.nomos_reset_kv.argtypes = [C.c_int64]
    f = lib.nomos_debug_omlp_stage_ab
    f.restype = C.c_int32
    f.argtypes = [
        C.c_int64, C.c_int64, C.c_int32, C.c_int32, C.c_int32,
        C.c_int32, C.c_int32, C.c_int64, C.c_int64, C.c_int64, C.c_int64,
    ]
    return lib, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True, help="npz containing exact `ids`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="51,55,63")
    ap.add_argument("--kv-rows", default="53,54,55,299")
    args = ap.parse_args()
    ids = np.ascontiguousarray(np.load(args.lens)["ids"], dtype=np.int32)
    layers = [int(x) for x in args.layers.split(",")]
    kv_rows = [int(x) for x in args.kv_rows.split(",")]
    anchor = len(ids) - 1
    if len(ids) > 512:
        sys.exit("debug contract supports at most 512 rows")
    if any(r < 0 or r >= len(ids) for r in kv_rows):
        sys.exit(f"invalid KV row in {kv_rows}; S={len(ids)}")

    lib, probe = bind(os.environ["QWEN_SO"])
    handle = lib.nomos_init(os.environ["QWEN_WEIGHTS"].encode())
    if not handle:
        sys.exit("nomos_init failed")
    out = {
        "ids": ids,
        "anchor_row": np.int32(anchor),
        "kv_rows": np.asarray(kv_rows, np.int32),
        "layers": np.asarray(layers, np.int32),
    }
    meta_i32 = np.zeros(16, np.int32)
    meta_f32 = np.zeros(4, np.float32)
    scratch = np.empty(max(6144, 24 * len(ids)), np.float32)

    def capture(layer, row, stage, expected):
        if lib.nomos_reset_kv(handle) != 0:
            sys.exit("nomos_reset_kv failed")
        meta_i32.fill(0); meta_f32.fill(0); scratch.fill(np.nan)
        rc = probe(
            handle, ids.ctypes.data, len(ids), 0, row, layer, stage,
            meta_i32.ctypes.data, meta_f32.ctypes.data,
            C.c_int64(-1), scratch.ctypes.data,
        )
        if rc != 0:
            sys.exit(f"probe rc={rc} L{layer} row={row} stage={stage}")
        dim = int(meta_i32[6])
        if expected is not None and dim != expected:
            sys.exit(f"stage {stage}: dim={dim}, expected={expected}")
        result = scratch[:dim].copy()
        if not np.isfinite(result).all():
            sys.exit(f"non-finite output L{layer} row={row} stage={stage}")
        return result

    try:
        for layer in layers:
            for stage, (name, expected) in ANCHOR_STAGES.items():
                a = capture(layer, anchor, stage, expected)
                if stage == 12:
                    a = a.reshape(24, anchor + 1)
                out[f"L{layer}_{name}"] = a
                print(f"L{layer}_{name} {a.shape} l2={np.linalg.norm(a.astype(np.float64)):.9e}", flush=True)
            for stage, name in ((30, "k_post_rope"), (31, "v_post_rope")):
                a = np.stack([capture(layer, row, stage, 1024) for row in kv_rows])
                out[f"L{layer}_{name}"] = a
                print(f"L{layer}_{name} {a.shape} l2={np.linalg.norm(a.astype(np.float64)):.9e}", flush=True)
        np.savez_compressed(args.out, **out)
        print(f"wrote {args.out}", flush=True)
    finally:
        lib.nomos_shutdown(handle)


if __name__ == "__main__":
    main()
