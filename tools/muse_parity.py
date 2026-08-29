#!/usr/bin/env python3
"""Muse-Glimmer kernel-vs-HF-golden parity harness — the #79 adjudicator.

    LD_LIBRARY_PATH=<worktree>:<cuda>:<pixi-lib> python3 tools/muse_parity.py \
        --so ~/nk-codex-muse/libnomos_kernel.so \
        --weights ~/nomos_data/muse-glimmer-flat/ \
        --golden ~/nomos-ref/muse-ref/goldens/golden_en_short.npz \
        --golden ~/nomos-ref/muse-ref/goldens/golden_code_short.npz

Two gates, in order:

COARSE — greedy chain, TEACHER-FORCED along the golden tokens. We feed the
GOLDEN token at every step rather than the kernel's own argmax: a free-running
comparison diverges permanently at the first mismatch, which localizes nothing.
Teacher-forcing makes every position an independent measurement — and if all
positions agree, the free-run is identical by induction, so nothing is lost.

FINE — per-layer residuals via nomos_debug_prefill_all_layers, kernel
[S, 52, 6656] against golden prompt_hidden[1:53] (transposed). The golden's
hidden_states[0] (embedding surface) has no kernel counterpart on this API and
is deliberately not compared.

The bar: bf16 HF reference vs Q4 kernel. Per-layer cosine / max|d| in family
with the Gemma bring-up, greedy chain exact or every divergence explained.
BYTE-EXACT IS NOT CLAIMED and cannot be, cross-precision. A layer whose cosine
falls off the cliff relative to its neighbours is the bisect entry point.

ctypes discipline: every FFI symbol gets explicit argtypes/restype BEFORE use —
an undeclared pointer arg truncates on aarch64 and produces the illegal-address
class of failure, which cost us a day on 2026-08-08.
"""
import argparse
import ctypes as C
import os
import sys

import numpy as np

N_LAYERS = 52
HIDDEN = 6656
VOCAB = 202048


def bind(so_path: str):
    lib = C.CDLL(so_path)
    sigs = {
        "nomos_init":        (C.c_int64, [C.c_int64]),
        "nomos_prefill":     (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
        "nomos_decode_step": (C.c_int32, [C.c_int64, C.c_int32, C.c_int64]),
        "nomos_reset_kv":    (C.c_int32, [C.c_int64]),
        "nomos_debug_prefill_all_layers":
                             (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
    }
    for name, (r, a) in sigs.items():
        f = getattr(lib, name)
        f.restype, f.argtypes = r, a
    return lib


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--golden", action="append", required=True)
    ap.add_argument("--dump-dir", default="")
    args = ap.parse_args()

    wdir = args.weights if args.weights.endswith("/") else args.weights + "/"
    lib = bind(args.so)

    wbuf = C.create_string_buffer(wdir.encode())
    handle = lib.nomos_init(C.cast(wbuf, C.c_void_p).value)
    if not handle:
        sys.exit("FATAL: nomos_init returned 0 — weights dir wrong or geometry mismatch")
    print(f"engine up (handle={handle:#x})", flush=True)

    logits = np.zeros(VOCAB, np.float32)
    overall_ok = True

    for gpath in args.golden:
        z = np.load(gpath)
        name = os.path.basename(gpath).replace("golden_", "").replace(".npz", "")
        ids = z["prompt_ids"].astype(np.int32)
        S = len(ids)
        gold_tokens = z["step_tokens"].astype(np.int64)
        print(f"\n=== [{name}] S={S}, {len(gold_tokens)} greedy steps ===", flush=True)

        # ── FINE: per-layer taps on the prompt ───────────────────────────────
        taps = np.zeros((S, N_LAYERS, HIDDEN), np.float32)
        lib.nomos_reset_kv(handle)
        rc = lib.nomos_debug_prefill_all_layers(
            handle, ids.ctypes.data, S, taps.ctypes.data)
        if rc != 0:
            print(f"  TAP CAPTURE FAILED rc={rc} — fine gate unavailable", flush=True)
            overall_ok = False
        else:
            gold_h = z["prompt_hidden"][1:1 + N_LAYERS]          # [52, S, D]
            kern_h = np.transpose(taps, (1, 0, 2))               # [52, S, D]
            print("  layer   cosine      max|d|     mean|d|", flush=True)
            worst = (1.0, -1)
            for l in range(N_LAYERS):
                c = cos(gold_h[l], kern_h[l])
                d = np.abs(gold_h[l].astype(np.float64) - kern_h[l].astype(np.float64))
                if c < worst[0]:
                    worst = (c, l)
                if l < 3 or l >= N_LAYERS - 2 or c < 0.98:
                    print(f"   {l:>3}   {c:.6f}   {d.max():9.4f}  {d.mean():9.5f}",
                          flush=True)
            print(f"  worst layer: {worst[1]} cos={worst[0]:.6f}", flush=True)
            if args.dump_dir:
                os.makedirs(args.dump_dir, exist_ok=True)
                np.save(os.path.join(args.dump_dir, f"kernel_taps_{name}.npy"), taps)

        # ── COARSE: teacher-forced greedy chain ──────────────────────────────
        lib.nomos_reset_kv(handle)
        rc = lib.nomos_prefill(handle, ids.ctypes.data, S, logits.ctypes.data)
        if rc != 0:
            print(f"  PREFILL FAILED rc={rc}", flush=True)
            overall_ok = False
            continue
        agree, first_miss = 0, -1
        step_dump = []                      # kernel logits per teacher-forced step
        pred = int(np.argmax(logits))
        gl = z["prompt_logits_last"]
        print(f"  prompt logits: cos={cos(gl, logits):.6f} "
              f"top1 kernel={pred} golden={int(gold_tokens[0])} "
              f"{'MATCH' if pred == gold_tokens[0] else 'MISMATCH'}", flush=True)
        for k in range(len(gold_tokens)):
            if pred == gold_tokens[k]:
                agree += 1
            elif first_miss < 0:
                first_miss = k
            if k + 1 < len(gold_tokens):
                rc = lib.nomos_decode_step(handle, C.c_int32(int(gold_tokens[k])),
                                           logits.ctypes.data)
                if rc != 0:
                    print(f"  DECODE FAILED rc={rc} at step {k}", flush=True)
                    break
                step_dump.append(logits.copy())
                pred = int(np.argmax(logits))
        n = len(gold_tokens)
        verdict = "EXACT" if agree == n else f"{agree}/{n}, first miss at {first_miss}"
        print(f"  greedy chain (teacher-forced): {verdict}", flush=True)
        if args.dump_dir and step_dump:
            np.save(os.path.join(args.dump_dir, f"kernel_step_logits_{name}.npy"),
                    np.stack(step_dump))
        if agree != n:
            overall_ok = False

    print(f"\nVERDICT: {'PASS (both gates in family)' if overall_ok else 'DIVERGENCE — bisect from the worst layer / first miss'}",
          flush=True)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
