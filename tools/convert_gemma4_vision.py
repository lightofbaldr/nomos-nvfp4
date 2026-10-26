#!/usr/bin/env python3
"""Gemma-4-31B (gemma4) VISION tower -> bf16 blobs for the Nomos vision .so (#92).

KEY FINDING (verified from checkpoint + modeling_gemma4 source): Gemma-4 vision is NOT plain SigLIP — it's a
GEMMA-STYLE tower reusing text-tower machinery our gemma4 kernel already has:
 - four-norm sandwich per layer (input / post_attention / pre_feedforward / post_feedforward layernorm),
 - GATED MLP (gate_proj, up_proj, down_proj), gelu_pytorch_tanh,
 - QK-NORM (self_attn.q_norm / k_norm, RMSNorm on head_dim 72),
 - Gemma RMSNorm convention (kernel applies (1+weight) — emit RAW weight here).
Vision-specific deltas (separately gated per Codex): 2D vision RoPE theta 100 (per spatial dim), avg-pool
(pooling_kernel_size 3) -> 280 soft tokens with sqrt(hidden) scaling, standardize affine (std_scale/std_bias),
patch_embed input_proj 768->1152 (3ch*16*16, NO temporal patch), dual position_embedding_table [2,10240,1152],
embed_vision.embedding_projection 1152->5376 (=text hidden), splice at image_token_id 258880, 27 layers.
Proj modules are wrapped as `.linear.weight` (quant-aware wrapper) — extract by full key.
Usage: python3 convert_gemma4_vision.py <base_model_dir> <out_dir>
"""
import os, sys, json
import numpy as np

NL = 27
def stem_layer(i, name): return f"vision_L{i}_{name}"

LAYER_MAP = [
    ("self_attn.q_proj.linear.weight", "attn_q_w"), ("self_attn.k_proj.linear.weight", "attn_k_w"),
    ("self_attn.v_proj.linear.weight", "attn_v_w"), ("self_attn.o_proj.linear.weight", "attn_o_w"),
    ("self_attn.q_norm.weight", "attn_q_norm"), ("self_attn.k_norm.weight", "attn_k_norm"),
    ("input_layernorm.weight", "norm_input"), ("post_attention_layernorm.weight", "norm_post_attn"),
    ("pre_feedforward_layernorm.weight", "norm_pre_ff"), ("post_feedforward_layernorm.weight", "norm_post_ff"),
    ("mlp.gate_proj.linear.weight", "mlp_gate_w"), ("mlp.up_proj.linear.weight", "mlp_up_w"),
    ("mlp.down_proj.linear.weight", "mlp_down_w"),
]
TOP_MAP = [  # (blob_name, hf_key_under_model, required)
    ("patch_embed_w",      "vision_tower.patch_embedder.input_proj.weight",           True),   # [1152,768]
    ("pos_emb",            "vision_tower.patch_embedder.position_embedding_table",    True),   # [2,10240,1152]
    ("std_scale",          "vision_tower.std_scale",                                  True),   # [1152]
    ("std_bias",           "vision_tower.std_bias",                                   True),   # [1152]
    ("embed_projection_w", "embed_vision.embedding_projection.weight",                True),   # [5376,1152]
]

def main():
    base, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    from safetensors import safe_open
    import torch
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    handles = {}
    def T(key):
        sh = idx[key]
        if sh not in handles: handles[sh] = safe_open(os.path.join(base, sh), framework="pt")
        return handles[sh].get_tensor(key)
    def write_bf16(name, t):
        u16 = t.contiguous().to(torch.bfloat16).view(torch.uint16).cpu().numpy().astype(np.uint16)
        p = os.path.join(out, name + ".bf16"); u16.tofile(p + ".tmp"); os.replace(p + ".tmp", p)
        return list(t.shape)
    report = {"arch": "gemma4_vision", "layers": NL, "hidden": 1152, "text_hidden": 5376, "note": "gemma-style tower: 4-norm + gated MLP + qk-norm; RMSNorm is (1+w) in kernel", "L0": {}, "top": {}}
    for i in range(NL):
        pre = f"model.vision_tower.encoder.layers.{i}."
        for suf, name in LAYER_MAP:
            key = pre + suf
            if key not in idx: sys.exit(f"FATAL missing {key}")
            sh = write_bf16(stem_layer(i, name), T(key))
            if i == 0: report["L0"][name] = sh
        if i % 9 == 0: print(f"  layer {i}/{NL}", flush=True)
    for name, suf, req in TOP_MAP:
        key = "model." + suf
        if key not in idx:
            if req: sys.exit(f"FATAL missing required {key}")
            continue
        report["top"][name] = write_bf16(name, T(key))
    json.dump(report, open(os.path.join(out, "convert_report.json"), "w"), indent=2)
    print(f"DONE -> {out}\n  L0: {report['L0']}\n  top: {report['top']}")

if __name__ == "__main__":
    main()
