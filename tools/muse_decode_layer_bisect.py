#!/usr/bin/env python3
"""Compare first-step Muse layer outputs against HF for decode and batched routes.

Uses the existing nomos_debug_omlp_stage_ab probe at stage 7 (layer output) on
prompt + first golden token. The golden step_hidden[0, layer+1] is the matching
HF decoder-layer output after consuming that token.
"""

import argparse
import ctypes as C
from pathlib import Path

import numpy as np


HIDDEN = 6656
LAYERS = 52


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    return float(a64 @ b64 / denom) if denom else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    z = np.load(args.golden)
    tokens = np.ascontiguousarray(
        np.concatenate([z["prompt_ids"], z["step_tokens"][:1]]), dtype=np.int32
    )
    row = len(tokens) - 1

    lib = C.CDLL(str(Path(args.so).resolve()))
    lib.nomos_init.restype = C.c_int64
    lib.nomos_init.argtypes = [C.c_int64]
    lib.nomos_shutdown.restype = None
    lib.nomos_shutdown.argtypes = [C.c_int64]
    probe = lib.nomos_debug_omlp_stage_ab
    probe.restype = C.c_int32
    probe.argtypes = [
        C.c_int64, C.c_int64, C.c_int32, C.c_int32, C.c_int32,
        C.c_int32, C.c_int32, C.c_int64, C.c_int64, C.c_int64, C.c_int64,
    ]

    wdir = args.weights if args.weights.endswith("/") else args.weights + "/"
    wbuf = C.create_string_buffer(wdir.encode())
    handle = lib.nomos_init(C.cast(wbuf, C.c_void_p).value)
    if not handle:
        raise SystemExit("nomos_init failed")

    info = np.zeros(16, np.int32)
    stats = np.zeros(4, np.float32)
    decode = np.zeros(HIDDEN, np.float32)
    batch = np.zeros(HIDDEN, np.float32)

    print(
        f"boundary rows={len(tokens)} row={row} token={tokens[row]} "
        f"gold_step_token={int(z['step_tokens'][0])}",
        flush=True,
    )
    print(f"{'L':>3} {'decode-HF':>11} {'batch-HF':>11} {'dec-batch':>11} {'max|D-B|':>10}")
    print("  " + "-" * 52)
    for layer in range(LAYERS):
        info.fill(0)
        stats.fill(0)
        decode.fill(0)
        batch.fill(0)
        rc = probe(
            C.c_int64(handle), C.c_int64(tokens.ctypes.data), C.c_int32(len(tokens)),
            C.c_int32(0), C.c_int32(row), C.c_int32(layer), C.c_int32(7),
            C.c_int64(info.ctypes.data), C.c_int64(stats.ctypes.data),
            C.c_int64(decode.ctypes.data), C.c_int64(batch.ctypes.data),
        )
        if rc != 0:
            raise RuntimeError(f"probe rc={rc} layer={layer}")
        hf = z["step_hidden"][0, layer + 1]
        cd = cos(decode, hf)
        cb = cos(batch, hf)
        cdb = cos(decode, batch)
        if layer < 4 or layer >= LAYERS - 4 or cd < 0.98 or cb < 0.98 or abs(cd - cb) > 1e-4:
            print(f"{layer:3d} {cd:11.6f} {cb:11.6f} {cdb:11.6f} {stats[0]:10.4f}", flush=True)

    lib.nomos_shutdown(C.c_int64(handle))


if __name__ == "__main__":
    main()
