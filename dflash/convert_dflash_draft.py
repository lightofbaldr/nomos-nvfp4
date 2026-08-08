#!/usr/bin/env python3
"""Convert z-lab/gemma-4-31B-it-DFlash (HF safetensors) to the Nomos draft-engine flat format.
Spec (task #20 port, 2026-07-04) — mirrors convert_e2b_draft.py conventions:
  - Q4_0 .q4 (group-32, quantize_q4_0.py) for all dense matmuls:
    per-layer q/k/v/o + mlp gate/up/down, and the context-fusion fc [5376, 32256].
  - fp32 .bin for norms (input/post_attn layernorms, per-head q/k norms, hidden_norm, final norm).
  - NO embed/lm-head tensors in this checkpoint (tie_word_embeddings=true): the drafter gathers
    from the TARGET's embed table and reads out through the TARGET's lm-head at runtime.
  - config.json copied; manifest.json with stem -> {orig, shape, fmt, rel_err}.
Architecture truths (shapes-are-truth, 2026-07-04 dump): model_type qwen3, 5 layers,
GQA 64Q/8KV x hd128, per-head q/k norms, SiLU MLP 10752, sliding(2048) x4 + full x1,
rope_theta 1e6, fc = 6-tap concat (6x5376=32256) -> 5376.
"""
import os, sys, json, shutil
from pathlib import Path
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from quantize_q4_0 import quantize, dequant, write_q4

SRC = os.environ.get(
    "DFLASH_CKPT",
    os.path.expanduser("~/nomos_data/dflash-gemma-4-31b"),
)
OUT = os.environ.get(
    "DFLASH_OUT",
    os.path.expanduser("~/nomos_data/dflash-gemma-4-31b-flat"),
)
os.makedirs(OUT, exist_ok=True)

DENSE_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
DENSE_EXACT = ("fc.weight",)

f = safe_open(os.path.join(SRC, "model.safetensors"), "pt")  # torch reader: numpy can't parse bf16
names = list(f.keys())
print(f"tensors: {len(names)}")
manifest = {"source": os.path.join(SRC, "model.safetensors"),
            "format": "nomos-q4-g32 draft-dflash", "tied_embed_and_head": True,
            "dflash": {"block_size": 16, "mask_token_id": 4,
                       "target_layer_ids": [1, 12, 23, 35, 46, 57], "num_target_layers": 60},
            "arch": {"model_type": "qwen3", "num_layers": 5, "hidden": 5376,
                     "num_q_heads": 64, "num_kv_heads": 8, "head_dim": 128,
                     "mlp_dim": 10752, "act": "silu",
                     "layer_types": ["sliding", "sliding", "sliding", "sliding", "full"],
                     "sliding_window": 2048, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6,
                     "final_logit_softcapping": 30.0, "vocab": 262144},
            "tensors": {}}

for n in sorted(names):
    a = f.get_tensor(n).float().numpy()
    stem = n.replace(".", "_")
    dense = (n in DENSE_EXACT) or (any(n.endswith(s) for s in DENSE_SUFFIXES) and a.ndim == 2)
    if dense:
        s16, packed, cnt = quantize(a)
        write_q4(os.path.join(OUT, stem), s16, packed, cnt)
        rel = float(np.abs(dequant(s16, packed, cnt) - a.ravel()).max() / (np.abs(a).max() + 1e-9))
        manifest["tensors"][stem] = {"orig": n, "shape": list(a.shape), "fmt": "q4", "rel_err": rel}
        print(f"  q4 : {stem:<50s} {list(a.shape)} rel={rel:.3f}")
    else:
        a.astype(np.float32).tofile(os.path.join(OUT, stem + ".bin"))
        manifest["tensors"][stem] = {"orig": n, "shape": list(a.shape), "fmt": "f32bin"}
        print(f"  f32: {stem:<50s} {list(a.shape)}")

shutil.copy(os.path.join(SRC, "config.json"), os.path.join(OUT, "config.json"))
with open(os.path.join(OUT, "manifest.json"), "w") as mf:
    json.dump(manifest, mf, indent=1)
q4n = sum(1 for v in manifest["tensors"].values() if v["fmt"] == "q4")
binn = sum(1 for v in manifest["tensors"].values() if v["fmt"] == "f32bin")
total = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT)) / 1e9
print(f"DONE: {q4n} q4 + {binn} f32bin -> {OUT}  ({total:.2f} GB)")
