"""Official converter: Breeze text encoder (T5Gemma2, 26L) -> raw bf16 kernel blobs.

M2 blob contract (Codex, 2026-09-04): embed_tokens.weight + eoi_embedding, the 13 per-layer
weights (q/k/v/o projections, q/k norms, the four-norm sandwich, gate/up/down), final norm.
Raw little-endian bf16 exactly as stored (checkpoint dtype), dot->underscore names,
sha256 manifest. text_encoder_proj rides M3.

Usage: python3 tools/convert_breeze_encoder.py [hf_dir] [out_dir]
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/adam/nomos_data/breeze-tts-2/hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/adam/nomos_data/breeze-tts-2/encoder-blobs"


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in idx if k.startswith("text_encoder.") and k != "text_encoder_proj.weight"]
    by_file = {}
    for k in keys:
        by_file.setdefault(idx[k], []).append(k)

    manifest = {}
    for fname, ks in sorted(by_file.items()):
        f = safe_open(f"{SRC}/{fname}", "pt")
        for k in sorted(ks):
            t = f.get_tensor(k)
            assert t.dtype == torch.bfloat16, f"{k}: unexpected dtype {t.dtype}"
            raw = t.contiguous().view(torch.uint16).numpy()
            name = k.replace(".", "_")
            raw.tofile(f"{OUT}/{name}.bf16")
            manifest[name] = {"shape": list(t.shape),
                              "sha256": hashlib.sha256(raw.tobytes()).hexdigest()[:16]}

    with open(f"{OUT}/MANIFEST.json", "w") as m:
        json.dump(manifest, m, indent=1, sort_keys=True)
    print(f"emitted {len(manifest)} bf16 blobs -> {OUT}")
    for probe in ("text_encoder_embed_tokens_weight", "text_encoder_embed_tokens_eoi_embedding",
                  "text_encoder_layers_0_self_attn_q_norm_weight", "text_encoder_norm_weight"):
        print(f"  {probe:52s} {manifest[probe]['shape']} {manifest[probe]['sha256']}")


if __name__ == "__main__":
    main()
