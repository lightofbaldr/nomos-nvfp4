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
GEMM_SUFFIXES = tuple(f"self_attn.{p}_proj.weight" for p in "qkvo") + \
                tuple(f"mlp.{p}_proj.weight" for p in ("gate", "up", "down"))


def _is_backbone_gemm(key: str) -> bool:
    return key.startswith("backbone_model.") and ".layers." in key and key.endswith(GEMM_SUFFIXES)


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
            if NVFP4 and _is_backbone_gemm(k):
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from quantize_nvfp4 import write_nvfp4
                W = np.ascontiguousarray(t.to(torch.float32).numpy())
                write_nvfp4(f"{OUT}/{name}.nvfp4", W)
                blob = open(f"{OUT}/{name}.nvfp4", "rb").read()
                manifest[name] = {"shape": list(t.shape), "format": "nvfp4",
                                  "sha256": hashlib.sha256(blob).hexdigest()[:16]}
                continue
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)  # none expected; keep the emit total, note in manifest
            raw = t.contiguous().view(torch.uint16).numpy()
            raw.tofile(f"{OUT}/{name}.bf16")
            manifest[name] = {"shape": list(t.shape),
                              "sha256": hashlib.sha256(raw.tobytes()).hexdigest()[:16]}

    with open(f"{OUT}/MANIFEST.json", "w") as m:
        json.dump(manifest, m, indent=1, sort_keys=True)
    groups = {}
    for k in manifest:
        groups[k.split("_")[0]] = groups.get(k.split("_")[0], 0) + 1
    print(f"emitted {len(manifest)} bf16 blobs -> {OUT}")
    for probe in ("lm_head_weight", "text_encoder_proj_weight"):
        if probe in manifest:
            print(f"  {probe:40s} {manifest[probe]['shape']} {manifest[probe]['sha256']}")


if __name__ == "__main__":
    main()
