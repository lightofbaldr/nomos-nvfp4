#!/usr/bin/env python3
"""Muse assistant (DFlash drafter) bf16 -> flat .q4 dir, for the S=16 weight-reusing MMQ body.

Sibling of convert_muse_assistant_q4's Q8 form. The Q4 DRAFTER PATH differs from Q8
in ONE structural way (verified in lib/dflash_drafter.mojo): it loads UN-FUSED
projections — q_proj/k_proj/v_proj and gate_proj/up_proj as SEPARATE .q4 files,
concatenated on-device via load_to_gpu_q4_concat3/2. Q8 pre-fuses them. So this is
not a format flag on the Q8 converter; the file set is genuinely different.

WHY THIS EXISTS (Codex, 2026-08-10): the Q8 assistant body routes through
gpu_matmul_q8_rows_dev (one warp per output row, re-streams every weight 16x at
S=16, zero cross-row reuse) = 320ms draft = 65-72% of the spec cycle. DRAFTER_Q8=0
selects the existing S=16 weight-reusing MMQ Q4 body instead. This A/B measures
whether Q4-body restores spec>base and at what acceptance cost (Q4 vs Q8 is a real
precision change — gate identity + E, do not assume lossless transfers).

Same contract guards as the Q8 converter: no lm_head/embed by design, RAW norms
(NOT centered — assistant is standard RMSNorm), q/k norms present, total accounting.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dspark_q8 import write_f32          # noqa: E402 — family .bin writer
from quantize_q4_0 import quantize as q4q, write_q4  # noqa: E402

LAYERS = 5
TAPS = [1, 13, 25, 37, 49]
HIDDEN = 6656


def _q4(path_stem, w):
    s, p, n = q4q(np.ascontiguousarray(w, np.float32))
    write_q4(path_stem + ".tmp", s, p, n)
    os.replace(path_stem + ".tmp.q4", path_stem + ".q4")
    print(f"  {os.path.basename(path_stem)+'.q4':>44}  {w.shape}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    from safetensors import safe_open
    import torch

    shards = [f for f in os.listdir(in_dir) if f.endswith(".safetensors")]
    assert len(shards) == 1, shards
    st = safe_open(os.path.join(in_dir, shards[0]), framework="pt")
    have = set(st.keys())
    consumed = set()

    def t(name):
        consumed.add(name)
        return st.get_tensor(name).to(torch.float32).numpy()

    for absent in ("lm_head.weight", "embed_tokens.weight", "model.embed_tokens.weight"):
        if absent in have:
            sys.exit(f"FATAL: {absent} present — head/embed-sharing contract changed. Stop.")

    # encoder + final norms (identical to Q8: fc is Q4, norms are f32 .bin)
    fc = t("encoder.fc.weight")
    assert fc.shape == (HIDDEN, len(TAPS) * HIDDEN), fc.shape
    _q4(os.path.join(out_dir, "fc_weight"), fc)
    write_f32(os.path.join(out_dir, "hidden_norm_weight.bin"), t("encoder.output_norm_enc.weight"))
    write_f32(os.path.join(out_dir, "norm_weight.bin"), t("norm.weight"))

    for i in range(LAYERS):
        p, o = f"layers.{i}.", os.path.join(out_dir, f"layers_{i}_")
        # UN-FUSED projections — the Q4 path concats on device
        _q4(o + "self_attn_q_proj_weight", t(p + "self_attn.q_proj.weight"))
        _q4(o + "self_attn_k_proj_weight", t(p + "self_attn.k_proj.weight"))
        _q4(o + "self_attn_v_proj_weight", t(p + "self_attn.v_proj.weight"))
        _q4(o + "self_attn_o_proj_weight", t(p + "self_attn.o_proj.weight"))
        _q4(o + "mlp_gate_proj_weight", t(p + "mlp.gate_proj.weight"))
        _q4(o + "mlp_up_proj_weight", t(p + "mlp.up_proj.weight"))
        _q4(o + "mlp_down_proj_weight", t(p + "mlp.down_proj.weight"))
        for src, dst in (("input_layernorm.weight", "input_layernorm_weight"),
                         ("post_attention_layernorm.weight", "post_attention_layernorm_weight"),
                         ("self_attn.q_norm.weight", "self_attn_q_norm_weight"),
                         ("self_attn.k_norm.weight", "self_attn_k_norm_weight")):
            if p + src not in have:
                sys.exit(f"FATAL: missing {p}{src}")
            write_f32(o + dst + ".bin", t(p + src))

    leftover = have - consumed
    if leftover:
        sys.exit(f"FATAL: {len(leftover)} unconsumed tensors: {sorted(leftover)[:6]}")

    json.dump({"source": in_dir, "layers": LAYERS, "taps": TAPS, "block_size": 16,
               "hidden": HIDDEN, "kv_heads": 8, "mask_token_id": 201818, "format": "q4",
               "projections": "UNFUSED (q/k/v, gate/up separate — Q4 concat-on-device path)",
               "norms": "RAW standard RMSNorm, NOT centered",
               "head_and_embed": "NONE by design; target embed in, target head out"},
              open(os.path.join(out_dir, "convert_manifest.json"), "w"), indent=2)
    print(f"\n  done -> {out_dir}  ({len(consumed)}/{len(have)} consumed)")


if __name__ == "__main__":
    main()
