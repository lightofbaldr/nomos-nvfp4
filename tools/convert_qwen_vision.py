#!/usr/bin/env python3
"""Qwen3.8-27B (qwen3_5) VISION tower -> bf16 blobs for the Nomos vision .so (#91).

Extracts model.visual.* from the full multimodal checkpoint and writes raw bf16 (uint16 = top-16-bits of
fp32) blobs the kernel loads directly, mirroring convert_muse_vision.py's flat naming/format so Codex's
loader pattern carries over. Tier-0 = bf16 (no quant).

Qwen ViT deltas vs Muse (verified from modeling_qwen3_5 source + checkpoint shapes, NOT inferred):
 - FUSED qkv ([3456,1152]=3x1152) WITH bias; attn.proj WITH bias.
 - Plain MLP linear_fc1[4304,1152]->gelu_tanh->linear_fc2[1152,4304] (NOT gated), both WITH bias.
 - Two LayerNorms/block (norm1,norm2) eps 1e-6, WITH bias.
 - patch_embed.proj is Conv3d[1152,3,2,16,16] with kernel==stride==full patch => equivalent to a Linear;
   proj.weight.reshape(1152,1536) matches the [in_ch,tp,ps,ps] pixel_values flatten. HAS bias.
 - pos_embed.weight [2304,1152] (48x48 learned grid, HOST-interpolated to the actual grid, by design).
 - merger: norm(1152, eps1e-6) applied per-patch, then reshape 4 patches -> 4608, fc1[4608,4608]->gelu->
   fc2[5120,4608]. This IS the projector to out_hidden 5120 (no separate adapter/projection, no ln_pre/post).
 - NO ln_pre / ln_post / adapter / projection tensors (Muse had those; Qwen doesn't).
Usage: python3 convert_qwen_vision.py <base_model_dir> <out_dir>
"""
import os, sys, json
import numpy as np

NL = 27  # depth
def stem_layer(i, name): return f"vision_L{i}_{name}"

LAYER_MAP = [
    ("attn.qkv.weight", "attn_qkv_w"), ("attn.qkv.bias", "attn_qkv_b"),
    ("attn.proj.weight", "attn_o_w"),  ("attn.proj.bias", "attn_o_b"),
    ("mlp.linear_fc1.weight", "mlp_fc1_w"), ("mlp.linear_fc1.bias", "mlp_fc1_b"),
    ("mlp.linear_fc2.weight", "mlp_fc2_w"), ("mlp.linear_fc2.bias", "mlp_fc2_b"),
    ("norm1.weight", "norm1_w"), ("norm1.bias", "norm1_b"),
    ("norm2.weight", "norm2_w"), ("norm2.bias", "norm2_b"),
]
# top-level (blob_name, hf_suffix_under_model.visual, reshape-to-2d?, required)
TOP_MAP = [
    ("patch_embed_w", "patch_embed.proj.weight", True,  True),   # Conv3d [1152,3,2,16,16] -> [1152,1536]
    ("patch_embed_b", "patch_embed.proj.bias",   False, True),
    ("pos_emb",       "pos_embed.weight",        False, True),   # [2304,1152], host-interpolated
    ("merger_norm_w", "merger.norm.weight",      False, True),
    ("merger_norm_b", "merger.norm.bias",        False, True),
    ("merger_fc1_w",  "merger.linear_fc1.weight",False, True),   # [4608,4608]
    ("merger_fc1_b",  "merger.linear_fc1.bias",  False, True),
    ("merger_fc2_w",  "merger.linear_fc2.weight",False, True),   # [5120,4608]
    ("merger_fc2_b",  "merger.linear_fc2.bias",  False, True),
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
        p = os.path.join(out, name + ".bf16")
        u16.tofile(p + ".tmp"); os.replace(p + ".tmp", p)
        return list(t.shape)

    report = {"arch": "qwen3_5_vision", "layers": NL, "hidden": 1152, "out_hidden": 5120, "dtype": "bf16", "L0": {}, "top": {}}
    for i in range(NL):
        pre = f"model.visual.blocks.{i}."
        for suf, name in LAYER_MAP:
            key = pre + suf
            if key not in idx: sys.exit(f"FATAL missing {key}")
            sh = write_bf16(stem_layer(i, name), T(key))
            if i == 0: report["L0"][name] = sh
        if i % 9 == 0: print(f"  block {i}/{NL}", flush=True)
    for name, suf, reshape2d, req in TOP_MAP:
        key = "model.visual." + suf
        if key not in idx:
            if req: sys.exit(f"FATAL missing required {key}")
            print(f"  (absent, ok) {key}"); continue
        t = T(key)
        if reshape2d: t = t.reshape(t.shape[0], -1)   # Conv3d [1152,3,2,16,16] -> [1152,1536]
        report["top"][name] = write_bf16(name, t)
    json.dump(report, open(os.path.join(out, "convert_report.json"), "w"), indent=2)
    print(f"DONE -> {out}\n  L0: {report['L0']}\n  top: {report['top']}")

if __name__ == "__main__":
    main()
