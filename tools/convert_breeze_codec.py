"""Official converter: Breeze bundled codec (Qwen3TTSTokenizerV2) decoder -> kernel blobs.

Emits the M1 blob contract (agreed with Codex 2026-09-04) from
audio_tokenizer/model.safetensors. Effective codebook rows are materialised
exactly as the reference computes them at decode time
(qwen_tts .../modeling_qwen3_tts_tokenizer_v2.py EuclideanCodebook.decode):

    embedding = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]   # epsilon = 1e-5

Codebook order: 0 = semantic (rvq_first layer 0), 1..15 = acoustic (rvq_rest layers 0..14).
output_proj weights are 1x1 convs in the checkpoint [512,256,1]; emitted squeezed [512,256].
Everything fp32 little-endian, one blob per file + manifest with sha256s.

Usage: python3 tools/convert_breeze_codec.py [src_dir] [out_dir]
"""
import hashlib
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

EPSILON = 1e-5  # EuclideanCodebook.__init__ default; config does not override it

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/adam/nomos_data/breeze-tts-2/hf/audio_tokenizer"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/adam/nomos_data/breeze-tts-2/codec-blobs"


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    f = safe_open(f"{SRC}/model.safetensors", "pt")

    def t(key):
        return f.get_tensor(key).to(torch.float32)

    manifest = {}

    def emit(name, arr):
        arr = np.ascontiguousarray(arr.numpy() if torch.is_tensor(arr) else arr, dtype=np.float32)
        path = f"{OUT}/{name}.f32"
        arr.tofile(path)
        manifest[name] = {"shape": list(arr.shape),
                          "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16]}

    # codebook 0: semantic (rvq_first); 1..15: acoustic (rvq_rest 0..14)
    def effective(prefix, layer):
        s = t(f"{prefix}.vq.layers.{layer}._codebook.embedding_sum")
        u = t(f"{prefix}.vq.layers.{layer}._codebook.cluster_usage")
        return s / u.clamp(min=EPSILON)[:, None]

    emit("codec_codebook_0", effective("decoder.quantizer.rvq_first", 0))
    for i in range(15):
        emit(f"codec_codebook_{i + 1}", effective("decoder.quantizer.rvq_rest", i))

    emit("codec_rvq_first_output_proj_weight",
         t("decoder.quantizer.rvq_first.output_proj.weight").squeeze(-1))
    emit("codec_rvq_rest_output_proj_weight",
         t("decoder.quantizer.rvq_rest.output_proj.weight").squeeze(-1))
    emit("codec_pre_conv_weight", t("decoder.pre_conv.conv.weight"))
    emit("codec_pre_conv_bias", t("decoder.pre_conv.conv.bias"))

    # Everything else under decoder.* (pre_transformer / upsample / conv stack), raw fp32 LE,
    # source layout preserved (ConvTranspose stays [Cin,Cout,K]), dots -> underscores.
    done_prefixes = ("decoder.quantizer.rvq_first.vq", "decoder.quantizer.rvq_rest.vq",
                     "decoder.quantizer.rvq_first.output_proj", "decoder.quantizer.rvq_rest.output_proj",
                     "decoder.pre_conv.")
    for key in f.keys():
        if not key.startswith("decoder."):
            continue
        if any(key.startswith(p) for p in done_prefixes):
            continue
        name = "codec_" + key[len("decoder."):].replace(".", "_")
        emit(name, t(key))

    with open(f"{OUT}/MANIFEST.json", "w") as m:
        json.dump(manifest, m, indent=1, sort_keys=True)
    print(f"emitted {len(manifest)} blobs -> {OUT}")
    for k in sorted(manifest):
        print(f"  {k:42s} {str(manifest[k]['shape']):16s} {manifest[k]['sha256']}")


if __name__ == "__main__":
    main()
