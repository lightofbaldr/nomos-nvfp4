#!/usr/bin/env python3
"""Muse-Glimmer VISION tower -> bf16 blobs (M0 of the vision-machinery Tier-0 plan).

Extracts the vision tower + patch-embedder + projector tensors from the full multimodal
safetensors and writes raw bf16 blobs (uint16, the top 16 bits of fp32 = bf16) the kernel
loads directly. Tier-0 is bf16 (no quant). perception_emb_norm and the text embed_norm are
SCALELESS RMSNorms (no learned weight) — nothing to extract; the kernel applies x*rsqrt(mean(x^2)+eps).

Naming (flat, sibling of the text converters): vision_L{i}_<name>.bf16, vision_<top>.bf16.
Usage: python3 convert_muse_vision.py <base_model_dir> <out_dir>
Verified layout (config vision_config + index): hidden 1536, FF 8960, 50 layers, 16 heads,
patch 14, patch_temporal 2, merge 2, pos_emb 32x32, ln_eps 1e-5, rope_theta 1e4, image_token_id 200092.
"""
import os, sys, json
import numpy as np

NL = 50
def stem_layer(i, name): return f"vision_L{i}_{name}"

# per-layer tensors: (hf_suffix, blob_name). LayerNorm has weight AND bias.
LAYER_MAP = [
    ("attn.q_proj.weight", "attn_q_w"), ("attn.q_proj.bias", "attn_q_b"),
    ("attn.k_proj.weight", "attn_k_w"), ("attn.k_proj.bias", "attn_k_b"),
    ("attn.v_proj.weight", "attn_v_w"), ("attn.v_proj.bias", "attn_v_b"),
    ("attn.proj.weight",   "attn_o_w"), ("attn.proj.bias",   "attn_o_b"),
    ("mlp.fc1.weight", "mlp_fc1_w"), ("mlp.fc1.bias", "mlp_fc1_b"),
    ("mlp.fc2.weight", "mlp_fc2_w"), ("mlp.fc2.bias", "mlp_fc2_b"),
    ("norm1.weight", "norm1_w"), ("norm1.bias", "norm1_b"),
    ("norm2.weight", "norm2_w"), ("norm2.bias", "norm2_b"),
]
# top-level: (hf_key_suffix_under_model, blob_name, required)
TOP_MAP = [
    ("vision_tower.ln_pre.weight",  "ln_pre_w",  True),
    ("vision_tower.ln_pre.bias",    "ln_pre_b",  True),
    ("vision_tower.ln_post.weight", "ln_post_w", True),
    ("vision_tower.ln_post.bias",   "ln_post_b", True),
    ("vision_tower.patch_embedder.patch_embedding.weight",       "patch_embed_w", True),
    ("vision_tower.patch_embedder.position_embedding_table.weight", "pos_emb",     True),
    ("vision_adapter.fc1.weight", "adapter_fc1_w", True),
    ("vision_adapter.fc1.bias",   "adapter_fc1_b", False),
    ("vision_adapter.fc2.weight", "adapter_fc2_w", True),
    ("vision_adapter.fc2.bias",   "adapter_fc2_b", False),
    ("vision_projection.weight",  "projection_w",  True),
    ("vision_projection.bias",    "projection_b",  False),
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
        # store the top-16-bits-of-fp32 = bf16 (kernel reads uint16, (v<<16).view(f32))
        u16 = t.to(torch.bfloat16).view(torch.uint16).cpu().numpy().astype(np.uint16)
        p = os.path.join(out, name + ".bf16")
        u16.tofile(p + ".tmp"); os.replace(p + ".tmp", p)
        return t.shape, float(t.to(torch.float32).mean())

    report = {"arch": "muse_glimmer_vision", "layers": NL, "hidden": 1536, "dtype": "bf16"}
    # per-layer
    for i in range(NL):
        pre = f"model.vision_tower.layers.{i}."
        for suf, name in LAYER_MAP:
            key = pre + suf
            if key not in idx: sys.exit(f"FATAL missing {key}")
            sh, mean = write_bf16(stem_layer(i, name), T(key))
            if i == 0: report[f"L0_{name}_shape"] = list(sh)
        if i % 12 == 0: print(f"  layer {i}/{NL}", flush=True)
    # top-level
    for suf, name, req in TOP_MAP:
        key = "model." + suf
        if key not in idx:
            if req: sys.exit(f"FATAL missing required {key}")
            print(f"  (absent, ok) {key}"); report[f"{name}_absent"] = True; continue
        sh, mean = write_bf16("vision_" + name, T(key))
        report[f"{name}_shape"] = list(sh)
    report["note"] = "perception_emb_norm + text embed_norm are SCALELESS (no weight) — applied param-free in kernel"
    json.dump(report, open(os.path.join(out, "vision_manifest.json"), "w"), indent=2)
    n = len([f for f in os.listdir(out) if f.endswith(".bf16")])
    print(f"  done -> {out}  ({n} bf16 blobs)")
    print(f"  patch_embed {report.get('patch_embed_w_shape')}  proj {report.get('projection_w_shape')}  adapter_fc1 {report.get('adapter_fc1_w_shape')}")

if __name__ == "__main__":
    main()
