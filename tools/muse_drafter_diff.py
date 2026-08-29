#!/usr/bin/env python3
"""Diff the kernel drafter trace against the HF surfaces — layer-by-layer cliff-finder.

Runs the real production draft cycle under NOMOS_DFLASH_TRACE=1, pulls the trace
via the forward_synth sentinel (ctx_len=-1), and compares every surface to
hf_surfaces.npz. First layer whose cosine departs the HF ramp is the bug's layer.
Q8-vs-bf16: cosine + proposal identity, not bytes.
"""
import ctypes as C
import os
import sys
import numpy as np

VOCAB, D, BLOCK, TAPS = 202048, 6656, 16, 5
PROMPT_IDS = [200000, 954, 7963, 323, 11698, 373]
ANCHOR = 13796
HF = "~/nomos-ref/muse-ref/hf_surfaces.npz"

# trace layout (floats), per Codex's spec
N_NOISE = BLOCK * D
N_LAYER = TAPS * BLOCK * D
N_FINAL = BLOCK * D
N_LOG = (BLOCK - 1) * VOCAB
TOTAL = N_NOISE + N_LAYER + N_FINAL + N_LOG


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 0.0


def main():
    so = sys.argv[sys.argv.index("--so") + 1]
    wdir = sys.argv[sys.argv.index("--weights") + 1].rstrip("/") + "/"
    ddir = sys.argv[sys.argv.index("--dflash") + 1].rstrip("/") + "/"
    abs_start = int(sys.argv[sys.argv.index("--abs-start") + 1]) if "--abs-start" in sys.argv else 5

    lib = C.CDLL(so)
    for n, (r, a) in {
        "nomos_init": (C.c_int64, [C.c_int64]),
        "nomos_prefill": (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
        "nomos_reset_kv": (C.c_int32, [C.c_int64]),
        "nomos_kv_cache_len": (C.c_int32, [C.c_int64]),
        "nomos_dflash_load": (C.c_int32, [C.c_int64, C.c_int64]),
        "nomos_dflash_reset": (C.c_int32, [C.c_int64]),
        "nomos_dflash_draft_block": (C.c_int32, [C.c_int64, C.c_int32, C.c_int32, C.c_int64]),
        "nomos_dflash_forward_synth": (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64, C.c_int64]),
    }.items():
        f = getattr(lib, n); f.restype, f.argtypes = r, a

    ht = lib.nomos_init(C.cast(C.create_string_buffer(wdir.encode()), C.c_void_p).value)
    assert ht, "init failed"
    assert lib.nomos_dflash_load(ht, C.cast(C.create_string_buffer(ddir.encode()), C.c_void_p).value) == 0

    logits = np.zeros(VOCAB, np.float32)
    ids = np.array(PROMPT_IDS, np.int32)
    lib.nomos_reset_kv(ht); lib.nomos_dflash_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    # TEACHER-FORCE the anchor to HF's value. The kernel's own argmax can flip to a
    # Q4/bf16 near-tie (e.g. 262 vs HF's 13796, gap 1.125 under softcap) — a real but
    # SEPARATE result. To measure the DRAFTER we must feed both sides the same anchor
    # HF used, or the comparison is between two different draft problems. The HF
    # surface manifest records its anchor; abort before comparison on any mismatch
    # (Codex's fail-safe — a stale reference must fail loud, not silently diff).
    z_anchor = int(np.load(HF, allow_pickle=True)["anchor"])
    if z_anchor != ANCHOR:
        sys.exit(f"HF surface anchor {z_anchor} != expected {ANCHOR}; regen reference first")
    kc = int(np.argmax(logits))
    c = ANCHOR
    print(f"anchor: kernel-argmax={kc} HF={ANCHOR} -> teacher-forcing {c} "
          f"({'kernel agrees' if kc == ANCHOR else 'kernel near-tie flip, forced to HF'})", flush=True)
    dout = np.zeros(BLOCK, np.int32)
    start = lib.nomos_kv_cache_len(ht)
    lib.nomos_dflash_draft_block(ht, C.c_int32(c), C.c_int32(start), dout.ctypes.data)

    trace = np.zeros(TOTAL, np.float32)
    rc = lib.nomos_dflash_forward_synth(ht, 0, C.c_int32(-1), 0, trace.ctypes.data)
    if rc != 0:
        sys.exit(f"trace sentinel rc={rc} (-3 => NOMOS_DFLASH_TRACE not set before load)")

    o = 0
    k_noise = trace[o:o + N_NOISE].reshape(BLOCK, D); o += N_NOISE
    k_layer = trace[o:o + N_LAYER].reshape(TAPS, BLOCK, D); o += N_LAYER
    k_final = trace[o:o + N_FINAL].reshape(BLOCK, D); o += N_FINAL
    k_log = trace[o:o + N_LOG].reshape(BLOCK - 1, VOCAB)

    # allow_pickle: HF is our own file, written by hf_drafter_surfaces.py on this box
    # this session (one object field: ctx_encoded). Not untrusted input.
    z = np.load(HF, allow_pickle=True)
    h_noise = z["noise_embeds"][0]
    h_layer = z["layer_out"][:, 0]
    h_final = z["final"][0]
    h_log = z["logits"][0]

    print(f"\n  abs_start_pos used by kernel call = {start} (HF first-noise Q position = 6)")
    print(f"  {'surface':<14} {'cosine':>9}  {'k|mean|':>8} {'hf|mean|':>8}")
    print(f"  {'noise':<14} {cos(k_noise, h_noise):>9.5f}  {np.abs(k_noise).mean():>8.4f} {np.abs(h_noise).mean():>8.4f}")
    for l in range(TAPS):
        print(f"  {'layer_'+str(l):<14} {cos(k_layer[l], h_layer[l]):>9.5f}  "
              f"{np.abs(k_layer[l]).mean():>8.4f} {np.abs(h_layer[l]).mean():>8.4f}")
    # per-ROW cosine on layer_0: uniform across 16 => shared-context/attention bug;
    # row-0 (anchor) differs from rows 1..15 (mask) => mask/position geometry.
    print("  layer_0 per-row cos: " + " ".join(
        f"{cos(k_layer[0][r], h_layer[0][r]):.2f}" for r in range(BLOCK)))
    print(f"  {'final':<14} {cos(k_final, h_final):>9.5f}  {np.abs(k_final).mean():>8.4f} {np.abs(h_final).mean():>8.4f}")
    print(f"  {'logits':<14} {cos(k_log, h_log):>9.5f}")
    kp, hp = k_log.argmax(-1).tolist(), h_log.argmax(-1).tolist()
    print(f"  kernel proposed[:8]={kp[:8]}")
    print(f"  HF     proposed[:8]={hp[:8]}  match={sum(1 for a,b in zip(kp,hp) if a==b)}/{len(hp)}")


if __name__ == "__main__":
    main()
