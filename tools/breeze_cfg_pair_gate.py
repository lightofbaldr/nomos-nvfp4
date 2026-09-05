#!/usr/bin/env python3
"""Golden replay + same-session timing of sequential and paired CFG APIs.

Both prefixes are independently prefilled at their actual lengths. Only the
current codebook and earlier codes are supplied to depth; future codes are -1.
M3 acceptance: 36/36 backbone top1 per lane, zero depth misses with HF margin
>=0.1, mean depth relL2 <0.05. No sampling or codec work is in model timings.
"""
import argparse
import ctypes as C
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))
    i, p = C.c_int64, C.c_void_p
    specs = {
        "init": ([C.c_char_p], i),
        "prefill_lane": ([i, C.c_int32, p, C.c_int32, p, p, p], C.c_int32),
        "step_backbone": ([i, C.c_int32, p, p], C.c_int32),
        "depth_begin": ([i, C.c_int32, p, p], C.c_int32),
        "depth_advance": ([i, C.c_int32, C.c_int32, p, p], C.c_int32),
        "step_backbone2": ([i, p, p], C.c_int32),
        "depth_begin2": ([i, p, p], C.c_int32),
        "depth_advance2": ([i, C.c_int32, p, p], C.c_int32),
        "free": ([i], C.c_int32),
    }
    for name, (args, ret) in specs.items():
        fn = getattr(lib, "nomos_breeze_model_" + name)
        fn.argtypes, fn.restype = args, ret
    return lib


def checked(rc):
    assert rc == 0, f"FFI returned {rc}"


def replay(lib, h, prefixes, codes, paired):
    lm = np.empty((len(codes) + 1, 2, 2052), "f4")
    dp = np.empty((len(codes), 15, 2, 2051), "f4")
    for lane, prefix in enumerate(prefixes):
        assert prefix.ndim == 2 and prefix.shape[1] == 2048
        checked(lib.nomos_breeze_model_prefill_lane(
            h, lane, prefix.ctypes.data, prefix.shape[0],
            None, None, lm[0, lane].ctypes.data))
    dt, bt = [], []
    for f, frame in enumerate(codes):
        partial = np.full(16, -1, dtype="i8")
        partial[0] = frame[0]
        t = time.perf_counter()
        for cb in range(15):
            if paired:
                if cb == 0:
                    rc = lib.nomos_breeze_model_depth_begin2(
                        h, partial.ctypes.data, dp[f, cb].ctypes.data)
                else:
                    rc = lib.nomos_breeze_model_depth_advance2(
                        h, cb, partial.ctypes.data, dp[f, cb].ctypes.data)
                checked(rc)
            else:
                for lane in range(2):
                    if cb == 0:
                        rc = lib.nomos_breeze_model_depth_begin(
                            h, lane, partial.ctypes.data, dp[f, cb, lane].ctypes.data)
                    else:
                        rc = lib.nomos_breeze_model_depth_advance(
                            h, lane, cb, partial.ctypes.data, dp[f, cb, lane].ctypes.data)
                    checked(rc)
            partial[cb + 1] = frame[cb + 1]
        dt.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        if paired:
            checked(lib.nomos_breeze_model_step_backbone2(
                h, partial.ctypes.data, lm[f + 1].ctypes.data))
        else:
            for lane in range(2):
                checked(lib.nomos_breeze_model_step_backbone(
                    h, lane, partial.ctypes.data, lm[f + 1, lane].ctypes.data))
        bt.append((time.perf_counter() - t) * 1000)
    assert np.isfinite(lm).all() and np.isfinite(dp).all()
    # First frame is warmup for each replay; all frames still enter the gate.
    return lm, dp, {"backbone_ms": float(np.mean(bt[1:])),
                    "depth_ms": float(np.mean(dt[1:]))}


def metrics(lm, dp, lg, dg):
    result = []
    for lane in range(2):
        l, d = lm[:, lane], dp[:, :, lane]
        lr, dr = lg[:, lane].astype("f8"), dg[:, :, lane].astype("f8")
        top = d.argmax(-1) == dr.argmax(-1)
        sort = np.sort(dr, axis=-1)
        margin = sort[..., -1] - sort[..., -2]
        rel = np.linalg.norm(d - dr, axis=(1, 2)) / np.linalg.norm(dr, axis=(1, 2))
        row = {"lane": lane, "backbone_hits": int((l.argmax(-1) == lr.argmax(-1)).sum()),
               "backbone_total": len(l), "depth_hits": int(top.sum()),
               "depth_total": int(top.size), "depth_fat_misses": int((~top & (margin >= .1)).sum()),
               "depth_mean_relL2": float(rel.mean()), "depth_max_relL2": float(rel.max()),
               "backbone_mean_relL2": float(np.mean(np.linalg.norm(l - lr, axis=-1) / np.linalg.norm(lr, axis=-1)))}
        row["pass"] = bool(row["backbone_hits"] == len(l) and row["depth_fat_misses"] == 0 and rel.mean() < .05)
        result.append(row)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default="./libnomos_model-breeze.so")
    ap.add_argument("--blobs", default="/home/adam/nomos_data/breeze-tts-2/model-blobs/")
    ap.add_argument("--bench", default="/home/adam/kvasir/bench")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    assert a.repeats > 0
    lib = bind(a.so)
    g = np.load(Path(a.bench) / "breeze_model_goldens.npz")
    codes = np.ascontiguousarray(np.load(Path(a.bench) / "breeze_codec_codes_golden.npz")["codes_full"][0].T, dtype="i8")
    prefixes = [np.ascontiguousarray(g[key].reshape(-1, 2048), "f4")
                for key in ("merge__out_inputs_embeds", "merge__out_inputs_embeds_call1")]
    lg, dg = g["headtrace__lm_head"], g["headtrace__depth_decoder__codebooks_head"].reshape(-1, 15, 2, 2051)
    h = lib.nomos_breeze_model_init(a.blobs.encode())
    assert h
    result = {"sha256": hashlib.sha256(Path(a.so).read_bytes()).hexdigest(),
              "prefix_lengths": [len(x) for x in prefixes], "frames": len(codes), "runs": []}
    try:
        out = np.empty((2, 2052), "f4")
        assert lib.nomos_breeze_model_depth_begin2(h, None, out.ctypes.data) == -1
        assert lib.nomos_breeze_model_step_backbone2(h, codes[0].ctypes.data, out.ctypes.data) != 0
        # Re-prefill on the same handle each arm tests reset/reuse as well.
        for rep in range(a.repeats):
            for paired in ((False, True) if rep % 2 == 0 else (True, False)):
                lm, dp, times = replay(lib, h, prefixes, codes, paired)
                rows = metrics(lm, dp, lg, dg)
                arm = {"rep": rep, "paired": paired, "timing": times, "gate": rows}
                result["runs"].append(arm)
                print(json.dumps(arm), flush=True)
        # Illegal out-of-order advance must reject before it mutates either cache.
        assert lib.nomos_breeze_model_depth_advance2(h, 1, codes[0].ctypes.data, out.ctypes.data) != 0
    finally:
        checked(lib.nomos_breeze_model_free(h))
    result["pass"] = all(row["pass"] for run in result["runs"] for row in run["gate"])
    for paired in (False, True):
        runs = [r["timing"] for r in result["runs"] if r["paired"] == paired]
        med = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
        med["model_ms"] = med["backbone_ms"] + med["depth_ms"]
        result["paired" if paired else "sequential"] = med
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
