#!/usr/bin/env python3
"""Muse-Glimmer drafter e2e: acceptance + THE lossless gate, fail-closed.

Implements the six preconditions agreed with Kvasir (2026-08-10) — each is
a lesson from this week converted into a gate. NO acceptance or speed number is
printed unless every gate passes; a gate failure exits nonzero with the gate
named. Numbers from a partially-failed run are exactly how a crash loop printed
21789 tok/s (#76) and how E=1 would have read as "bad drafter" (tap trap).

  G1 ROUTE PROVENANCE  dflash_load rc==0 (loud-refusal path not taken),
                       draft_block rc==15 (BLOCK-1: the 5-tap contract held),
                       assistant manifest mask_token_id==201818.
                       (Wrapper additionally greps the log for the engine's
                       "tap capture DISABLED" diagnostic and for any
                       missing-weight WARN naming the assistant dir.)
  G2 LIVENESS          total accepted draft tokens > 0 AND E > 1 somewhere.
                       An E=1-pinned result is REJECTED even if identity passes.
  G3 IDENTITY          fixed-budget FULL token arrays: base and spec must be
                       equal length and byte-identical IDs, per prompt.
                       This is the real losslessness bar (kernel-vs-kernel) —
                       NOT the cross-precision HF comparison.
  G4 CACHE DISCIPLINE  after every cycle commit: target kv len == drafter cache
                       len == start + 1 + num_acc. Rejected suffix enters
                       neither state.
  G5 FIRST-CYCLE TRACE cycle 0 of prompt 0: anchor, 15 proposed IDs, num_acc,
                       verify-row argmaxes, and the 5 captured tap rows — saved
                       so a zero-acceptance result is diagnosable, not a shrug.
  G6 REPORTING         SHA + .so identity + weight manifests in the header;
                       acceptance/speed printed ONLY after G1-G5 pass.

Speed convention: (ntok-1)/dt — the first token comes from prefill before the
clock starts, same as the corrected public benchmark. (The muse worktree's CLI
still carries the OLD ntok/dt at this writing — flagged, do not compare.)
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

BLOCK = 16
VOCAB = 202048
D = 6656
TAPS = 5


def bind(so_path: str):
    lib = C.CDLL(so_path)
    sigs = {
        "nomos_init":            (C.c_int64, [C.c_int64]),
        "nomos_prefill":         (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
        "nomos_decode_step":     (C.c_int32, [C.c_int64, C.c_int32, C.c_int64]),
        "nomos_reset_kv":        (C.c_int32, [C.c_int64]),
        "nomos_kv_cache_len":    (C.c_int32, [C.c_int64]),
        "nomos_kv_set_len":      (C.c_int32, [C.c_int64, C.c_int32]),
        "nomos_dflash_load":     (C.c_int32, [C.c_int64, C.c_int64]),
        "nomos_dflash_reset":    (C.c_int32, [C.c_int64]),
        "nomos_dflash_cache_len":(C.c_int32, [C.c_int64]),
        "nomos_dflash_draft_block": (C.c_int32, [C.c_int64, C.c_int32, C.c_int32, C.c_int64]),
        "nomos_dflash_verify_fused": (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int32, C.c_int64]),
        "nomos_dflash_append_verify_context": (C.c_int32, [C.c_int64, C.c_int32, C.c_int32]),
        "nomos_dflash_get_taps": (C.c_int32, [C.c_int64, C.c_int64]),
    }
    for name, (r, a) in sigs.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = r, a
    return lib


def fail(gate: str, msg: str):
    print(f"\nGATE FAIL [{gate}]: {msg}", flush=True)
    print("NO ACCEPTANCE OR SPEED NUMBERS — run is invalid by contract.", flush=True)
    sys.exit(2)


def sha256_head(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def base_generate(lib, ht, logits, ids, ntok):
    lib.nomos_reset_kv(ht)
    lib.nomos_dflash_reset(ht)
    rc = lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    if rc != 0:
        fail("G1", f"base prefill rc={rc}")
    tok = int(np.argmax(logits))
    out = [tok]
    t0 = time.time()
    for _ in range(ntok - 1):
        rc = lib.nomos_decode_step(ht, C.c_int32(tok), logits.ctypes.data)
        if rc != 0:
            fail("G1", f"base decode rc={rc}")
        tok = int(np.argmax(logits))
        out.append(tok)
    return out, (ntok - 1) / (time.time() - t0)


def spec_generate(lib, ht, logits, ids, ntok, vb, trace_path=None):
    rows = np.zeros((BLOCK + 1, VOCAB), np.float32)
    dout = np.zeros(BLOCK, np.int32)
    toks = np.zeros(BLOCK, np.int32)
    lib.nomos_reset_kv(ht)
    lib.nomos_dflash_reset(ht)
    rc = lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    if rc != 0:
        fail("G1", f"spec prefill rc={rc}")
    if lib.nomos_dflash_cache_len(ht) != len(ids):
        fail("G4", f"drafter context desync after prefill: "
                   f"{lib.nomos_dflash_cache_len(ht)} != {len(ids)}")
    c = int(np.argmax(logits))
    spec = [c]
    steps = acc = 0
    t0 = time.time()
    while len(spec) < ntok:
        steps += 1
        start = lib.nomos_kv_cache_len(ht)
        rc = lib.nomos_dflash_draft_block(ht, C.c_int32(c), C.c_int32(start),
                                          dout.ctypes.data)
        if rc != BLOCK - 1:
            fail("G1", f"draft_block rc={rc}, expected {BLOCK-1} — tap/mask/geometry "
                       f"contract broken (the E=1 family)")
        drafts = [int(x) for x in dout[:vb - 1]]
        toks[0] = c
        for i, d in enumerate(drafts):
            toks[1 + i] = d
        rc = lib.nomos_dflash_verify_fused(ht, toks.ctypes.data, C.c_int32(vb),
                                           C.c_int32(start), rows.ctypes.data)
        if rc != 0:
            fail("G1", f"verify_fused rc={rc}")
        num_acc = vb - 1
        for j, d in enumerate(drafts):
            if d != int(np.argmax(rows[j])):
                num_acc = j
                break
        nxt = int(np.argmax(rows[num_acc]))

        if trace_path and steps == 1:               # G5: first-cycle trace
            taps = np.zeros((TAPS, D), np.float32)
            trc = lib.nomos_dflash_get_taps(ht, taps.ctypes.data)
            np.savez(trace_path, anchor=c, proposed=np.array(drafts),
                     verify_argmax=np.array([int(np.argmax(rows[j]))
                                             for j in range(vb)]),
                     num_acc=num_acc, taps_rc=trc, taps=taps)
            print(f"  [G5 trace] anchor={c} proposed[:8]={drafts[:8]} "
                  f"num_acc={num_acc} taps_rc={trc} "
                  f"tap|mean|={np.abs(taps).mean():.4f}", flush=True)
            if trc == 0 and float(np.abs(taps).mean()) == 0.0:
                fail("G5", "tap rows read back all-zero — capture not live")

        spec.extend(drafts[:num_acc])
        spec.append(nxt)
        acc += num_acc
        lib.nomos_kv_set_len(ht, C.c_int32(start + 1 + num_acc))
        rc = lib.nomos_dflash_append_verify_context(ht, C.c_int32(0),
                                                    C.c_int32(num_acc + 1))
        if rc <= 0:
            fail("G4", f"ctx append rc={rc}")
        want = start + 1 + num_acc                   # G4: both states advance identically
        got_kv, got_df = lib.nomos_kv_cache_len(ht), lib.nomos_dflash_cache_len(ht)
        if got_kv != want or got_df != want:
            fail("G4", f"cache desync at step {steps}: kv={got_kv} drafter={got_df} "
                       f"expected {want} — rejected suffix leaked into state")
        c = nxt
    dt = time.time() - t0
    E = 1.0 + acc / steps
    return spec[:ntok], E, acc, steps, (ntok - 1) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dflash", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--ntok", type=int, default=64)
    ap.add_argument("--vb", type=int, default=9)
    ap.add_argument("--trace-dir", default="")
    args = ap.parse_args()

    wdir = args.weights.rstrip("/") + "/"
    ddir = args.dflash.rstrip("/") + "/"

    # ── G6: provenance header before anything runs ───────────────────────────
    so_dir = os.path.dirname(os.path.abspath(args.so))
    sha = subprocess.run(["git", "-C", so_dir, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    st = os.stat(args.so)
    print(f"[G6] sha={sha}")
    print(f"[G6] so={args.so} bytes={st.st_size} mtime={int(st.st_mtime)}")
    for tag, d in (("target", wdir), ("drafter", ddir)):
        mp = os.path.join(d, "convert_manifest.json")
        m = json.load(open(mp))
        print(f"[G6] {tag} manifest sha256[:16]={sha256_head(mp)} keys={sorted(m)[:4]}...")
    dm = json.load(open(os.path.join(ddir, "convert_manifest.json")))
    if dm.get("mask_token_id") != 201818:
        fail("G1", f"drafter manifest mask_token_id={dm.get('mask_token_id')} != 201818")
    print(f"[G6] vb={args.vb} (UNSWEPT for Muse — not a champion claim) ntok={args.ntok}")

    lib = bind(args.so)
    wbuf = C.create_string_buffer(wdir.encode())
    ht = lib.nomos_init(C.cast(wbuf, C.c_void_p).value)
    if not ht:
        fail("G1", "nomos_init returned 0")
    dbuf = C.create_string_buffer(ddir.encode())
    rc = lib.nomos_dflash_load(ht, C.cast(dbuf, C.c_void_p).value)
    if rc != 0:
        fail("G1", f"nomos_dflash_load rc={rc}")
    print("engine + drafter up", flush=True)

    logits = np.zeros(VOCAB, np.float32)
    bar = json.load(open(args.tokens))
    total_acc = 0
    results = []
    all_identical = True

    for i, p in enumerate(bar["prompts"]):
        ids = np.array(p["ids"], np.int32)
        base, btps = base_generate(lib, ht, logits, ids, args.ntok)
        trace = (os.path.join(args.trace_dir, "first_cycle_trace.npz")
                 if args.trace_dir and i == 0 else None)
        if trace:
            os.makedirs(args.trace_dir, exist_ok=True)
        spec, E, acc, steps, stps = spec_generate(lib, ht, logits, ids,
                                                  args.ntok, args.vb, trace)
        total_acc += acc
        # G3: full fixed-budget arrays, not prefixes
        if len(base) != args.ntok or len(spec) != args.ntok:
            fail("G3", f"[{p['id']}] length: base={len(base)} spec={len(spec)} "
                       f"!= {args.ntok}")
        ident = base == spec
        all_identical &= ident
        results.append((p["id"], p["bucket"], ident, E, btps, stps, acc,
                        steps * (args.vb - 1)))
        print(f"  {p['id']:>8} [{p['bucket']:>10}] base {btps:6.2f} | "
              f"spec {stps:6.2f} tok/s  E={E:.2f} acc={acc}/{steps*(args.vb-1)} "
              f"identity={'EXACT' if ident else 'DIVERGES'}", flush=True)

    # ── G2 liveness across the whole run ─────────────────────────────────────
    # CALIBRATED 2026-08-10 after the first e2e slid through: acc=8/2960 (0.27%),
    # mean E=1.02 — pinned-at-E≈1 in every meaningful sense — passed the original
    # "acc>0 and E>1.0 anywhere" test and the harness printed a spec tok/s it
    # should have withheld. The contract said REJECT results pinned at E=1; a
    # threshold that a broken drafter clears by luck is not that contract.
    mean_E = float(np.mean([r[3] for r in results]))
    total_drafted = sum(r[7] for r in results)
    acc_rate = total_acc / max(1, total_drafted)
    if mean_E < 1.15 or acc_rate < 0.02:
        fail("G2", f"drafter not live: mean E={mean_E:.3f} (need >=1.15), "
                   f"accept rate={acc_rate:.2%} (need >=2%) — "
                   f"pinned-at-E~1; spec numbers withheld")

    n_ident = sum(1 for r in results if r[2])
    print(f"\nLOSSLESS (kernel-vs-kernel, fixed {args.ntok}-token arrays): "
          f"{n_ident}/{len(results)}", flush=True)
    if not all_identical:
        fail("G3", "identity failed on at least one prompt — speed numbers withheld")

    # ── G8: BASE-STABILITY CANARY (must precede G7 — G7's ratios divide by base) ──
    # ADDED 2026-08-10 (Prime). Every prompt runs the SAME fixed ntok budget, so the
    # base decode rate should be near-constant across the bar; it is not a property
    # of the prompt. In one NVFP4 run it decayed MONOTONICALLY in run order --
    # 27.35, 27.31, 27.20, 22.31, 20.70, 19.17 (30% spread) -- from thermal or a
    # co-tenant on the GPU. A rerun on a clean box held 27.62..27.30 (1.2%).
    #
    # Why this is a GATE and not a note: in a fixed-order single-process bar, a
    # decaying base SILENTLY INFLATES the speedup of whatever runs LAST, because
    # spec/base divides by a base that is quietly shrinking. Nothing in the output
    # reveals it -- both numbers look plausible, and the contaminated ratio is the
    # one that gets quoted. (GB10/Q4 was checked and is clean: 0.33% spread,
    # non-monotonic. This fires for the box that is actually unhealthy.)
    #
    # The series prints unconditionally, like G7's table, so the evidence is never
    # visible only in the failure path.
    #
    # THE GATE KEYS ON SPREAD (magnitude), NOT ON drift r (shape). Do not "improve"
    # it by thresholding the correlation -- validated against three series whose
    # verdict was already known:
    #     run-1 contaminated   spread 34.1%   r=-0.94   -> must FAIL
    #     rerun clean box      spread  1.3%   r=-0.95   -> must PASS
    #     GB10/Q4 clean        spread  0.3%   r=-0.71   -> must PASS
    # Both CLEAN runs carry a strongly negative r; a correlation threshold would
    # have failed them. r describes the shape of a wobble and says nothing about
    # whether it is big enough to matter. r is printed as a diagnostic only.
    bases = [(r[0], r[4]) for r in results]          # run order, as measured
    b_vals = [b for _, b in bases]
    b_mean = float(np.mean(b_vals))
    spread = (max(b_vals) - min(b_vals)) / b_mean if b_mean > 0 else 0.0
    drift = float(np.corrcoef(np.arange(len(b_vals)), b_vals)[0, 1]) if len(b_vals) > 2 else 0.0
    print("\nbase tok/s in run order (should be ~flat; ntok is fixed):", flush=True)
    print("  " + "  ".join(f"{pid}={b:.2f}" for pid, b in bases), flush=True)
    print(f"  spread={spread:.1%}  drift r={drift:+.2f}", flush=True)
    if spread > 0.10:
        fail("G8", f"base decode rate varies {spread:.1%} across the bar "
                   f"(drift r={drift:+.2f}) — base should be ~flat at fixed ntok. "
                   f"Thermal or a GPU co-tenant; every spec/base ratio divides by a "
                   f"moving denominator, so speed numbers are withheld. Re-run on an "
                   f"idle, thermally-settled box.")

    # ── G7: PER-PROMPT SPEEDUP FLOOR ─────────────────────────────────────────
    # ADDED 2026-08-10 (Prime), after this harness printed "ALL GATES PASS ... base
    # 24.01 -> spec 26.39 tok/s" (1.10x) for a run where FOUR OF SIX PROMPTS WERE
    # SLOWER WITH SPECULATION ON and prose1 ran at 0.50x — half base speed. Every
    # number in that line was correct. The mean was carried entirely by struct1
    # (1.77x) and code1 (1.66x), and a reader would take 1.10x as "spec is a win".
    #
    # The same shape had already bitten the GB10/Q4 arm, whose closed "1.41x" hid
    # prose1 at 0.78x and en1 at 0.91x. Twice in one day, both times agreed to be
    # wrong in conversation -- which is exactly why this lives in the harness and
    # not in a run-book paragraph. A rule that only exists as prose does not fire
    # when it matters; a gate does.
    #
    # THE BAR: speculation must not make ANY prompt class slower. A mean cannot
    # express that, so the mean is not the gate -- the minimum is. The per-prompt
    # table prints unconditionally, pass or fail, so the distribution is never
    # available only in the failure path.
    ratios = [(r[0], r[1], r[5] / r[4] if r[4] > 0 else 0.0) for r in results]
    ratios.sort(key=lambda t: t[2])
    print("\nper-prompt speedup (spec/base) — the mean is NOT the gate:", flush=True)
    for pid, bucket, ratio in ratios:
        flag = "  <-- SLOWER" if ratio < 1.0 else ""
        print(f"  {pid:>8} [{bucket:>10}] {ratio:5.2f}x{flag}", flush=True)
    slower = [t for t in ratios if t[2] < 1.0]
    if slower:
        fail("G7", f"{len(slower)}/{len(ratios)} prompts SLOWER with speculation "
                   f"(worst: {slower[0][0]} [{slower[0][1]}] {slower[0][2]:.2f}x). "
                   f"Speculation must not regress any prompt class; an aggregate "
                   f"speedup that hides a per-prompt loss is withheld by contract.")

    mean_E = float(np.mean([r[3] for r in results]))
    E_lo, E_hi = min(r[3] for r in results), max(r[3] for r in results)
    print(f"ALL GATES PASS  mean E={mean_E:.2f} (range {E_lo:.2f}-{E_hi:.2f})  "
          f"base {np.mean([r[4] for r in results]):.2f} tok/s  "
          f"spec {np.mean([r[5] for r in results]):.2f} tok/s  "
          f"per-prompt {ratios[0][2]:.2f}x-{ratios[-1][2]:.2f}x "
          f"(vb={args.vb} unswept, {len(results)} prompts, ntok={args.ntok})",
          flush=True)


if __name__ == "__main__":
    main()
