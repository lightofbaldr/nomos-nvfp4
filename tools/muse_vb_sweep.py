#!/usr/bin/env python3
"""Muse DFlash VB sweep WITH CYCLE BUDGET — draft/verify/commit ms per arm.

Codex's guard (2026-08-10): spec 5.40 < base 12.22 at E=2.67 means verify
dominates the cycle. A bare tok/s sweep would let an apparent VB optimum hide a
fixed readout cost or a verify-routing issue. So every arm reports the cycle
decomposition, and E + lossless-identity are held INVARIANT as the correctness
guard — a VB that changes identity or tanks E is disqualified, not optimised.

Times the three FFI phases per cycle (CUDA is synchronous through these calls):
  draft   = nomos_dflash_draft_block
  verify  = nomos_dflash_verify_fused
  commit  = nomos_dflash_append_verify_context (+ set_len)
Base decode rerun in the SAME session/process (Codex: rerun base same session).
"""
import argparse
import ctypes as C
import json
import os
import sys
import time
import numpy as np

VOCAB, D, BLOCK = 202048, 6656, 16


def bind(so):
    lib = C.CDLL(so)
    for n, (r, a) in {
        "nomos_init": (C.c_int64, [C.c_int64]),
        "nomos_prefill": (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
        "nomos_decode_step": (C.c_int32, [C.c_int64, C.c_int32, C.c_int64]),
        "nomos_reset_kv": (C.c_int32, [C.c_int64]),
        "nomos_kv_cache_len": (C.c_int32, [C.c_int64]),
        "nomos_kv_set_len": (C.c_int32, [C.c_int64, C.c_int32]),
        "nomos_dflash_load": (C.c_int32, [C.c_int64, C.c_int64]),
        "nomos_dflash_reset": (C.c_int32, [C.c_int64]),
        "nomos_dflash_cache_len": (C.c_int32, [C.c_int64]),
        "nomos_dflash_draft_block": (C.c_int32, [C.c_int64, C.c_int32, C.c_int32, C.c_int64]),
        "nomos_dflash_verify_fused": (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int32, C.c_int64]),
        "nomos_dflash_append_verify_context": (C.c_int32, [C.c_int64, C.c_int32, C.c_int32]),
    }.items():
        f = getattr(lib, n); f.restype, f.argtypes = r, a
    return lib


def base_gen(lib, ht, logits, ids, ntok):
    lib.nomos_reset_kv(ht); lib.nomos_dflash_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits)); out = [tok]
    t0 = time.perf_counter()
    for _ in range(ntok - 1):
        lib.nomos_decode_step(ht, C.c_int32(tok), logits.ctypes.data)
        tok = int(np.argmax(logits)); out.append(tok)
    return out, (ntok - 1) / (time.perf_counter() - t0)


def spec_gen(lib, ht, logits, ids, ntok, vb):
    rows = np.zeros((BLOCK + 1, VOCAB), np.float32)
    dout = np.zeros(BLOCK, np.int32); toks = np.zeros(BLOCK, np.int32)
    lib.nomos_reset_kv(ht); lib.nomos_dflash_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    c = int(np.argmax(logits)); spec = [c]
    steps = acc = 0
    t_draft = t_verify = t_commit = 0.0
    t0 = time.perf_counter()
    while len(spec) < ntok:
        steps += 1
        start = lib.nomos_kv_cache_len(ht)
        a = time.perf_counter()
        lib.nomos_dflash_draft_block(ht, C.c_int32(c), C.c_int32(start), dout.ctypes.data)
        b = time.perf_counter(); t_draft += b - a
        drafts = [int(x) for x in dout[:vb - 1]]
        toks[0] = c
        for i, d in enumerate(drafts):
            toks[1 + i] = d
        lib.nomos_dflash_verify_fused(ht, toks.ctypes.data, C.c_int32(vb),
                                      C.c_int32(start), rows.ctypes.data)
        cc = time.perf_counter(); t_verify += cc - b
        num_acc = vb - 1
        for j, d in enumerate(drafts):
            if d != int(np.argmax(rows[j])):
                num_acc = j; break
        nxt = int(np.argmax(rows[num_acc]))
        spec.extend(drafts[:num_acc]); spec.append(nxt); acc += num_acc
        lib.nomos_kv_set_len(ht, C.c_int32(start + 1 + num_acc))
        lib.nomos_dflash_append_verify_context(ht, C.c_int32(0), C.c_int32(num_acc + 1))
        t_commit += time.perf_counter() - cc
        c = nxt
    dt = time.perf_counter() - t0
    return (spec[:ntok], 1.0 + acc / steps, (ntok - 1) / dt, steps,
            1000 * t_draft / steps, 1000 * t_verify / steps, 1000 * t_commit / steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dflash", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--ntok", type=int, default=96)
    ap.add_argument("--vbs", default="5,7,9,11,13")
    args = ap.parse_args()

    lib = bind(args.so)
    ht = lib.nomos_init(C.cast(C.create_string_buffer(
        (args.weights.rstrip("/") + "/").encode()), C.c_void_p).value)
    assert ht, "init failed"
    assert lib.nomos_dflash_load(ht, C.cast(C.create_string_buffer(
        (args.dflash.rstrip("/") + "/").encode()), C.c_void_p).value) == 0
    logits = np.zeros(VOCAB, np.float32)
    bar = json.load(open(args.tokens))["prompts"]
    vbs = [int(v) for v in args.vbs.split(",")]

    # base once per prompt (same session), then each VB
    base_ref = {}
    base_tps = []
    for p in bar:
        ids = np.array(p["ids"], np.int32)
        b, tps = base_gen(lib, ht, logits, ids, args.ntok)
        base_ref[p["id"]] = b; base_tps.append(tps)
    print(f"BASE: {np.mean(base_tps):.2f} tok/s (mean over {len(bar)} prompts, ntok={args.ntok})\n")

    print(f"{'VB':>3} {'spec':>7} {'E':>6} {'lossless':>9} "
          f"{'cycle_ms':>9} {'draft':>7} {'verify':>8} {'commit':>7} {'verify%':>8}")
    results = []
    for vb in vbs:
        tps_l, E_l, dr_l, vf_l, cm_l = [], [], [], [], []
        ident = 0
        for p in bar:
            ids = np.array(p["ids"], np.int32)
            spec, E, tps, steps, dms, vms, cms = spec_gen(lib, ht, logits, ids, args.ntok, vb)
            if spec == base_ref[p["id"]]:
                ident += 1
            tps_l.append(tps); E_l.append(E); dr_l.append(dms); vf_l.append(vms); cm_l.append(cms)
        cyc = np.mean(dr_l) + np.mean(vf_l) + np.mean(cm_l)
        vpct = 100 * np.mean(vf_l) / cyc if cyc else 0
        print(f"{vb:>3} {np.mean(tps_l):>7.2f} {np.mean(E_l):>6.2f} {ident:>4}/{len(bar):<4} "
              f"{cyc:>9.2f} {np.mean(dr_l):>7.2f} {np.mean(vf_l):>8.2f} {np.mean(cm_l):>7.2f} "
              f"{vpct:>7.1f}%")
        results.append({"vb": vb, "spec_tps": float(np.mean(tps_l)), "E": float(np.mean(E_l)),
                        "lossless": f"{ident}/{len(bar)}", "cycle_ms": float(cyc),
                        "draft_ms": float(np.mean(dr_l)), "verify_ms": float(np.mean(vf_l)),
                        "commit_ms": float(np.mean(cm_l))})

    # correctness guard: identity must be N/N on every arm
    bad = [r for r in results if not r["lossless"].startswith(str(len(bar)))]
    print(f"\nbase {np.mean(base_tps):.2f} tok/s | "
          f"{'ALL ARMS LOSSLESS' if not bad else 'IDENTITY BROKE: ' + str([r['vb'] for r in bad])}")
    print("READ: if verify% is flat-high across VB, the cycle is verify-bound and VB only trades "
          "E against draft — the tok/s optimum is where E growth stops beating added verify cost.")


if __name__ == "__main__":
    main()
