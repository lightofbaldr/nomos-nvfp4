#!/usr/bin/env python3
"""Qwen3.8-27B (qwen3_5) kernel-vs-HF-golden parity harness — the #81 adjudicator.

    LD_LIBRARY_PATH=<repo>:<cuda>:<pixi-lib> python3 tools/qwen3_5_parity.py \
        --so ~/nomos-kernel-product/libnomos_kernel.so \
        --weights ~/nomos_data/qwen3_8_27b-nvfp4/ \
        --golden ~/nomos-ref/qwen-ref/goldens/golden_en_short.npz \
        --golden ~/nomos-ref/qwen-ref/goldens/golden_xlong_ctx.npz

STATUS: the qwen3_5 GDN-hybrid engine is implemented. This harness retains graceful optional-symbol
handling for partial/debug builds, while the deploy gate requires the complete debug surface.

THREE gates, in order (a hybrid model needs a state gate the softmax-only Muse harness did not):

FINE-HIDDEN — per-layer residual stream. golden `prompt_hidden` is captured at SAMPLED positions
(`prompt_hidden_positions`: every position for short prompts, chunk-64 boundaries + last for long
ones), so the kernel taps are compared ONLY at those positions. CORRECTED manifest semantics (Codex
audit): golden[0]=raw embed (no kernel counterpart, skipped), golden[1..63]=raw outputs of decoder
layers 0..62, golden[64]=POST-FINAL-NORM (model.norm applied) — NOT raw layer-63. So kernel layers
0..62 compare to golden[1..63]; the kernel's post-final-norm surface compares to golden[64]; raw
layer-63 has NO golden counterpart until v2's pre_final_norm tap.

FINE-STATE — the 48 GDN layers' recurrent (48,128,128) + conv (10240,4) state vs `state_*.npz`,
per layer, head axis preserved. The reference is fp16 (bf16->f32->f16, NOT byte-exact bf16) so
thresholds are fp16-reference, never byte-exact. For GDN, residual agreement alone does NOT sign off
state — a wrong state update can still produce a plausible readout; the state gate is separate and
gets prefill + per-decode-step surfaces.

COARSE — greedy chain TEACHER-FORCED along the golden tokens (feed the golden token each step so
every position is an independent measurement; a free-run diverges permanently at the first mismatch
and localizes nothing). Uses the DISTINCT untied lm_head (tie_word_embeddings=False) — not embed.

The bar: bf16 HF reference vs NVFP4 kernel. Per-surface cosine / max|d|; greedy chain exact or every
divergence explained. BYTE-EXACT IS NOT CLAIMED cross-precision. A surface whose cosine falls off the
cliff relative to its neighbours is the bisect entry point.

ctypes discipline: every FFI symbol gets explicit argtypes/restype BEFORE use — an undeclared pointer
arg truncates on aarch64 and produces the illegal-address class of failure (cost a day 2026-08-08).
"""
import argparse
import ctypes as C
import os
import sys

import numpy as np

# Geometry (config-verified — see lib/model_profiles/qwen3_5.mojo).
N_LAYERS = 64
HIDDEN = 5120
VOCAB = 248320
GDN_V_HEADS, GDN_K_HD, GDN_V_HD = 48, 128, 128     # recurrent state [48,128,128] per GDN layer
GDN_CONV_DIM, GDN_CONV_K = 10240, 4                 # conv state [10240,4] per GDN layer
# GDN layers = the non-full layers: layer_types[i] == linear_attention iff (i+1) % 4 != 0.
GDN_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % 4 != 0]   # 48 layers

# FINE-HIDDEN materialize-all guard (Codex): nomos_debug_prefill_all_layers writes [S,64,5120] — for
# xlong S=5716 that is ~7 GiB device + ~7 GiB host just to keep the sampled rows. Skip above this S;
# short prompts localize layer bugs, and long-prompt FINE-HIDDEN waits on the v2 sampled-row export.
FINE_HIDDEN_S_MAX = 512


# Debug FFI the qwen3_5 engine MUST expose (Kvasir): _opt_ symbols are looked up but tolerated
# absent so partial engines still run the gates they can.
CORE_SIGS = {
    "nomos_init":        (C.c_int64, [C.c_int64]),
    "nomos_prefill":     (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
    "nomos_decode_step": (C.c_int32, [C.c_int64, C.c_int32, C.c_int64]),
    "nomos_reset_kv":    (C.c_int32, [C.c_int64]),
}
OPT_SIGS = {
    # per-layer residual taps: out [S, N_LAYERS, HIDDEN] raw decoder-layer outputs (layers 0..63).
    "nomos_debug_prefill_all_layers": (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
    # post-final-norm surface for the last prompt position: out [HIDDEN]. Compares to golden[64].
    "nomos_debug_final_norm":         (C.c_int32, [C.c_int64, C.c_int64, C.c_int32, C.c_int64]),
    # GDN state after prefill: recurrent out [len(GDN_LAYERS), 48,128,128], conv out [.., 10240,4].
    "nomos_debug_gdn_state":          (C.c_int32, [C.c_int64, C.c_int64, C.c_int64]),
}


def bind(so_path: str):
    if not os.path.exists(so_path):
        sys.exit(f"KERNEL NOT BUILT: {so_path} absent — the qwen3_5 engine does not exist yet. "
                 f"This harness is ready; run it once Kvasir's GDN engine lands.")
    lib = C.CDLL(so_path)
    for name, (r, a) in CORE_SIGS.items():
        f = getattr(lib, name); f.restype, f.argtypes = r, a
    have = {}
    for name, (r, a) in OPT_SIGS.items():
        try:
            f = getattr(lib, name); f.restype, f.argtypes = r, a; have[name] = f
        except AttributeError:
            have[name] = None
    return lib, have


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def channel_diag(g, k, layer, topk=8):
    """g/k: [n_pos, D] for one layer. Is the error concentrated in a few channels (outlier /
    massive-activation / a specific weight column) or diffuse (a uniform per-layer bug)?"""
    g = g.astype(np.float64); k = k.astype(np.float64)
    per_ch = np.abs(g - k).sum(axis=0)                 # [D] — summed |d| over positions
    D = per_ch.shape[0]
    order = np.argsort(per_ch)[::-1]
    tot = per_ch.sum() + 1e-30
    frac_top = per_ch[order[:topk]].sum() / tot
    lp = g.shape[0] - 1                                # last position (decode-relevant)
    print(f"    L{layer} channel diag (last pos): top-{topk} chans carry {frac_top*100:.1f}% of "
          f"summed|d| over {D} chans  ->  {'CONCENTRATED (outlier)' if frac_top>0.30 else 'DIFFUSE (uniform)'}",
          flush=True)
    for ch in order[:topk]:
        print(f"       ch {int(ch):5d}: gold {g[lp,ch]:+10.4f}  kern {k[lp,ch]:+10.4f}  "
              f"|d|_sum {per_ch[ch]:9.4f}", flush=True)


def report_layer_table(gold, kern, label, worst_thresh=0.98, diag_layers=()):
    """gold/kern: [n_layers, n_pos, D] aligned. Prints a table, returns worst (cos, layer_idx)."""
    print(f"  {label}: layer   cosine      max|d|     mean|d|", flush=True)
    worst = (1.0, -1)
    nL = gold.shape[0]
    for l in range(nL):
        c = cos(gold[l], kern[l])
        d = np.abs(gold[l].astype(np.float64) - kern[l].astype(np.float64))
        if c < worst[0]:
            worst = (c, l)
        if l < 3 or l >= nL - 2 or c < worst_thresh:
            print(f"     {l:>3}   {c:.6f}   {d.max():9.4f}  {d.mean():9.5f}", flush=True)
    print(f"    worst: layer {worst[1]} cos={worst[0]:.6f}", flush=True)
    for l in sorted(set(diag_layers) | ({worst[1]} if worst[1] >= 0 else set())):
        if 0 <= l < nL:
            channel_diag(gold[l], kern[l], l)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--golden", action="append", required=True)
    ap.add_argument("--state", action="append", default=[],
                    help="state_*.npz files (optional; else derived from the golden path)")
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--state-only", action="store_true",
                    help="run ONLY FINE-STATE (GDN recurrent/conv); skip FINE-HIDDEN + the slow COARSE greedy "
                         "tail. For fast state-cliff discriminators (e.g. KV-quant A/B on long prompts).")
    args = ap.parse_args()

    wdir = args.weights if args.weights.endswith("/") else args.weights + "/"
    lib, have = bind(args.so)
    print("debug FFI present: " + ", ".join(f"{k}={'y' if v else 'ABSENT'}" for k, v in have.items()),
          flush=True)

    handle = lib.nomos_init(C.cast(C.create_string_buffer(wdir.encode()), C.c_void_p).value)
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
        hid_pos = z["prompt_hidden_positions"].astype(np.int64) if "prompt_hidden_positions" in z \
            else np.arange(S)
        gold_tokens = z["step_tokens"].astype(np.int64)
        print(f"\n=== [{name}] S={S}, hidden@{len(hid_pos)} sampled pos, {len(gold_tokens)} steps ===",
              flush=True)

        # ── FINE-HIDDEN ──────────────────────────────────────────────────────
        f = None if args.state_only else have["nomos_debug_prefill_all_layers"]
        if f is None:
            print("  FINE-HIDDEN skipped — " + ("--state-only" if args.state_only
                  else "nomos_debug_prefill_all_layers absent"), flush=True)
        elif S > FINE_HIDDEN_S_MAX:
            print(f"  FINE-HIDDEN skipped — S={S} > {FINE_HIDDEN_S_MAX}: materialize-all ABI would alloc "
                  f"{S * N_LAYERS * HIDDEN * 4 / 2**30:.1f} GiB device+host to keep {len(hid_pos)} sampled "
                  f"rows. Localize on short prompts; long-prompt FINE-HIDDEN waits on the v2 sampled-row "
                  f"export (Codex).", flush=True)
        else:
            taps = np.zeros((S, N_LAYERS, HIDDEN), np.float32)
            lib.nomos_reset_kv(handle)
            rc = f(handle, ids.ctypes.data, S, taps.ctypes.data)
            if rc != 0:
                print(f"  TAP CAPTURE FAILED rc={rc}", flush=True); overall_ok = False
            else:
                # golden[1..63] = raw decoder layers 0..62; compare to kernel layers 0..62 at hid_pos.
                gold_h = z["prompt_hidden"][1:N_LAYERS]                     # [63, len(hid_pos), D]
                kern_h = np.transpose(taps[:, :N_LAYERS - 1, :], (1, 0, 2))[:, hid_pos, :]  # [63, len(hid_pos), D]
                worst = report_layer_table(gold_h, kern_h, "residual[0..62]", diag_layers=(0,))
                if worst[0] < 0.90:
                    overall_ok = False
                # golden[64] = post-final-norm; compare to the kernel's final-norm surface (last pos).
                fn = have["nomos_debug_final_norm"]
                if fn is not None:
                    knorm = np.zeros(HIDDEN, np.float32)
                    lib.nomos_reset_kv(handle)
                    if fn(handle, ids.ctypes.data, S, knorm.ctypes.data) == 0:
                        c = cos(z["prompt_hidden"][N_LAYERS][-1], knorm)   # golden[64] last pos
                        print(f"  final-norm[64] last-pos cos={c:.6f}", flush=True)
                        if c < 0.90:
                            overall_ok = False
                else:
                    print("  final-norm[64] skipped — nomos_debug_final_norm absent (raw layer-63 "
                          "has no golden; v2 pre_final_norm tap will close it)", flush=True)

        # ── FINE-STATE (GDN recurrent + conv) ────────────────────────────────
        spath = None
        if args.state:
            spath = next((s for s in args.state if name in s), None)
        if spath is None:
            cand = gpath.replace("golden_", "state_")
            spath = cand if os.path.exists(cand) else None
        gf = have["nomos_debug_gdn_state"]
        if spath is None:
            print("  FINE-STATE skipped — no state_*.npz for this prompt", flush=True)
        elif gf is None:
            print("  FINE-STATE skipped — nomos_debug_gdn_state absent", flush=True)
        else:
            zs = np.load(spath)
            g_rec = zs["prefill_recurrent"].astype(np.float32)             # [48, 48,128,128]
            g_conv = zs["prefill_conv"].astype(np.float32)                 # [48, 10240,4]
            k_rec = np.zeros((len(GDN_LAYERS), GDN_V_HEADS, GDN_K_HD, GDN_V_HD), np.float32)
            k_conv = np.zeros((len(GDN_LAYERS), GDN_CONV_DIM, GDN_CONV_K), np.float32)
            lib.nomos_reset_kv(handle)
            # prefill first so state is populated, then read it back.
            lib.nomos_prefill(handle, ids.ctypes.data, S, logits.ctypes.data)
            rc = gf(handle, k_rec.ctypes.data, k_conv.ctypes.data)
            if rc != 0:
                print(f"  GDN STATE CAPTURE FAILED rc={rc}", flush=True); overall_ok = False
            elif not k_rec.any():
                # Sentinel (Codex): all-zero state means not-populated / wrong route / missing bf16->fp32
                # cast — NOT a passing gate. Reject loudly rather than report a spurious cos=1 vs a zero ref.
                print(f"  GDN STATE SENTINEL FAIL: recurrent state all-zero. Expected {len(GDN_LAYERS)} "
                      f"layers in order {GDN_LAYERS[:6]}...{GDN_LAYERS[-3:]}. Check the exporter populated "
                      f"+ cast bf16->fp32 + used the profile layer-type order (not layer%4).", flush=True)
                overall_ok = False
            else:
                print(f"  GDN state: {len(GDN_LAYERS)} layers, order {GDN_LAYERS[:6]}...{GDN_LAYERS[-3:]}",
                      flush=True)
                worst_r = report_layer_table(g_rec.reshape(len(GDN_LAYERS), -1)[:, None, :],
                                             k_rec.reshape(len(GDN_LAYERS), -1)[:, None, :],
                                             "gdn_recurrent(prefill)", worst_thresh=0.95)
                worst_c = report_layer_table(g_conv.reshape(len(GDN_LAYERS), -1)[:, None, :],
                                             k_conv.reshape(len(GDN_LAYERS), -1)[:, None, :],
                                             "gdn_conv(prefill)", worst_thresh=0.95)
                if worst_r[0] < 0.90 or worst_c[0] < 0.90:
                    overall_ok = False

        # ── COARSE: teacher-forced greedy chain (distinct untied lm_head) ─────
        if args.state_only:
            print("  COARSE skipped — --state-only (fast state-cliff discriminator)", flush=True)
            continue
        lib.nomos_reset_kv(handle)
        rc = lib.nomos_prefill(handle, ids.ctypes.data, S, logits.ctypes.data)
        if rc != 0:
            print(f"  PREFILL FAILED rc={rc}", flush=True); overall_ok = False; continue
        agree, first_miss = 0, -1
        pred = int(np.argmax(logits))
        print(f"  prompt logits cos={cos(z['prompt_logits_last'], logits):.6f} "
              f"top1 kernel={pred} golden={int(gold_tokens[0])} "
              f"{'MATCH' if pred == gold_tokens[0] else 'MISMATCH'}", flush=True)
        # logits-structure diagnostic: spikes (uninit/partial-write buffer) vs coherently-shifted (systematic).
        gl = z["prompt_logits_last"]; gtop = int(np.argmax(gl))
        ktop = np.argsort(logits)[-10:][::-1]
        print(f"    logits struct: kern min/mean/max={logits.min():.3f}/{logits.mean():.4f}/{logits.max():.3f} "
              f"| golden min/mean/max={gl.min():.3f}/{gl.mean():.4f}/{gl.max():.3f}", flush=True)
        print(f"    kern top10 idx={ktop.tolist()}", flush=True)
        print(f"    kern top10 val={[round(float(logits[i]),3) for i in ktop]}", flush=True)
        print(f"    kern logit @golden-top({gtop})={float(logits[gtop]):.3f} (golden={float(gl[gtop]):.3f}); "
              f"#kern>10={int((logits>10).sum())} #golden>10={int((gl>10).sum())} "
              f"#kern==0={int((logits==0).sum())}/{len(logits)}", flush=True)
        # LIVENESS/FINITENESS (Codex gold-gate (b)): the cap=0 head bug passed every hidden/state gate —
        # a live-logits gate is mandatory. all-finite + nonzero variance + zero NaN/inf, and the top1 margin.
        nfin = int(np.isfinite(logits).sum()); nnan = int(np.isnan(logits).sum()); ninf = int(np.isinf(logits).sum())
        srt = np.sort(logits[np.isfinite(logits)]) if nfin else np.array([0.0])
        margin = float(srt[-1] - srt[-2]) if srt.size >= 2 else 0.0
        live = (nfin == len(logits)) and (nnan == 0) and (ninf == 0) and (float(np.var(srt)) > 1e-6)
        print(f"    logits LIVENESS: all_finite={nfin==len(logits)} #NaN={nnan} #inf={ninf} "
              f"var={float(np.var(srt)):.4f} top1_margin={margin:.4f} -> {'LIVE' if live else 'DEAD/DEGENERATE'}",
              flush=True)
        if not live:
            overall_ok = False
        for k in range(len(gold_tokens)):
            if pred == gold_tokens[k]:
                agree += 1
            elif first_miss < 0:
                first_miss = k
            if k + 1 < len(gold_tokens):
                rc = lib.nomos_decode_step(handle, C.c_int32(int(gold_tokens[k])), logits.ctypes.data)
                if rc != 0:
                    print(f"  DECODE FAILED rc={rc} at step {k}", flush=True); break
                pred = int(np.argmax(logits))
        n = len(gold_tokens)
        print(f"  greedy chain (teacher-forced): "
              f"{'EXACT' if agree == n else f'{agree}/{n}, first miss at {first_miss}'}", flush=True)
        if agree != n:
            overall_ok = False

    print(f"\nVERDICT: {'PASS (all gates in family)' if overall_ok else 'DIVERGENCE — bisect from the worst surface / first miss'}",
          flush=True)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
