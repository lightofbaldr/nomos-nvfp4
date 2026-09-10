"""Official converter: Breeze backbone + depth decoder + heads -> raw bf16 kernel blobs.

M3 blob contract (Codex, 2026-09-04): every main-checkpoint tensor EXCEPT
  - text_encoder.* (already shipped by convert_breeze_encoder.py)
  - codec_model.*  (training-legacy mimi fallback the shipped inference path never uses —
                    excluded deliberately; the real codec ships via convert_breeze_codec.py)
Covers: backbone_model.* (309, stock Qwen3 adapter), depth_decoder.* (112),
lm_head.weight [2052,2048], embed_text_tokens.*, text_encoder_proj.weight [2048,1152].
Raw little-endian bf16 exactly as stored, source layout, dot->underscore, sha256 manifest.

Usage: python3 tools/convert_breeze_model.py [hf_dir] [out_dir] [--nvfp4]

--nvfp4 (M7): the 28-layer backbone's seven GEMM projections per layer
(q/k/v/o/gate/up/down) are emitted as .nvfp4 (tools/quantize_nvfp4.write_nvfp4 —
the exact format lib/fp4_gemm.mojo consumes) instead of raw bf16. Everything else
(embeddings, norms, lm_head, the whole depth decoder, text_encoder_proj) stays
bf16 per the precision law: quantize the measured bulk, never pre-emptively.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/nomos_data/breeze-tts-2/hf")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/nomos_data/breeze-tts-2/model-blobs")

SKIP_PREFIXES = ("text_encoder.", "codec_model.")
NVFP4 = "--nvfp4" in sys.argv
# --nvfp4-classes=q,k,v,o,gate,up,down : quantize ONLY the listed projection classes;
# all other files are SYMLINKED from --link-bf16=<dir> (the canonical bf16 blob dir)
# so per-recipe ablation dirs cost only the quantized bytes. Implies NVFP4 for the listed set.
NVFP4_DEPTH = "--nvfp4-depth" in sys.argv           # quantize ALL depth projection classes
CLASSES = None          # backbone per-class subset
DEPTH_CLASSES = None    # depth per-class subset (the DEPTH spectrum ablation)
LINK_SRC = None
for a in sys.argv:
    if a.startswith("--nvfp4-classes="):
        CLASSES = set(a.split("=", 1)[1].split(","))
    if a.startswith("--nvfp4-depth-classes="):
        DEPTH_CLASSES = set(a.split("=", 1)[1].split(","))
    if a.startswith("--link-bf16="):
        LINK_SRC = a.split("=", 1)[1]
GEMM_SUFFIXES = tuple(f"self_attn.{p}_proj.weight" for p in "qkvo") + \
                tuple(f"mlp.{p}_proj.weight" for p in ("gate", "up", "down"))


def _is_backbone_gemm(key: str) -> bool:
    return key.startswith("backbone_model.") and ".layers." in key and key.endswith(GEMM_SUFFIXES)


def _is_depth_gemm(key: str) -> bool:
    # depth-decoder layer projections only; head + norms stay bf16 (logit producer, quality-critical)
    return key.startswith("depth_decoder.model.layers.") and key.endswith(GEMM_SUFFIXES)


def _gemm_class(key: str) -> str:
    return key.rsplit(".", 2)[-2].removesuffix("_proj")


def _quant(key: str) -> bool:
    """Should this weight be emitted NVFP4? Backbone: --nvfp4 (all) or --nvfp4-classes (subset).
    Depth: --nvfp4-depth (all) or --nvfp4-depth-classes (subset)."""
    if _is_backbone_gemm(key):
        return (_gemm_class(key) in CLASSES) if CLASSES is not None else NVFP4
    if _is_depth_gemm(key):
        return (_gemm_class(key) in DEPTH_CLASSES) if DEPTH_CLASSES is not None else NVFP4_DEPTH
    return False


# A per-class recipe is active if any subset was named -> non-quantized keys get symlinked bf16.
RECIPE = (CLASSES is not None) or (DEPTH_CLASSES is not None)


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in idx if not k.startswith(SKIP_PREFIXES)]
    by_file = {}
    for k in keys:
        by_file.setdefault(idx[k], []).append(k)

    manifest = {}
    for fname, ks in sorted(by_file.items()):
        f = safe_open(f"{SRC}/{fname}", "pt")
        for k in sorted(ks):
            t = f.get_tensor(k)
            name = k.replace(".", "_")
            if _quant(k):
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from quantize_nvfp4 import write_nvfp4
                W = np.ascontiguousarray(t.to(torch.float32).numpy())
                write_nvfp4(f"{OUT}/{name}.nvfp4", W)
                blob = open(f"{OUT}/{name}.nvfp4", "rb").read()
                manifest[name] = {"shape": list(t.shape), "format": "nvfp4",
                                  "sha256": hashlib.sha256(blob).hexdigest()[:16]}
                continue
            if RECIPE and LINK_SRC:   # recipe active + this key not quantized -> symlink bf16
                os.symlink(f"{LINK_SRC}/{name}.bf16", f"{OUT}/{name}.bf16")
                manifest[name] = {"shape": list(t.shape), "link": "bf16"}
                continue
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)  # none expected; keep the emit total, note in manifest
            raw = t.contiguous().view(torch.uint16).numpy()
            raw.tofile(f"{OUT}/{name}.bf16")
            manifest[name] = {"shape": list(t.shape),
                              "sha256": hashlib.sha256(raw.tobytes()).hexdigest()[:16]}

    with open(f"{OUT}/MANIFEST.json", "w") as m:
        json.dump(manifest, m, indent=1, sort_keys=True)
    nvfp4 = sum(1 for v in manifest.values() if v.get("format") == "nvfp4")
    linked = sum(1 for v in manifest.values() if v.get("link"))
    bb = sum(1 for k, v in manifest.items() if v.get("format") == "nvfp4" and k.startswith("backbone"))
    dd = sum(1 for k, v in manifest.items() if v.get("format") == "nvfp4" and k.startswith("depth"))
    print(f"emitted {len(manifest)} entries -> {OUT}: {nvfp4} nvfp4 (backbone {bb} / depth {dd}), "
          f"{linked} symlinked bf16, {len(manifest) - nvfp4 - linked} materialized bf16")


if __name__ == "__main__":
    main()
