#!/usr/bin/env python3
"""Convert Qwen3.8-27B (qwen3_5) GDN (linear_attn) weights -> per-slot BF16 blobs for
lib/gdn_weights.mojo (GdnWeights). Run with the system python (has torch float8_e4m3fn):

    /usr/bin/python3 tools/convert_qwen3_5_gdn.py \
        ~/nomos_data/qwen3_8_27b-nvfp4 ~/nomos_data/qwen3_8_27b-nvfp4/gdn

Correctness-first (contract locked w/ Kvasir 2026-08-17): the 3 FP8 E4M3 tensors
(in_proj_qkv, in_proj_z, out_proj) are dequantised weight_fp8 * weight_scale[out,1] -> BF16;
the 6 native-BF16 tensors (in_proj_a/b, conv1d, A_log, dt_bias, norm) pass through raw.
conv1d [10240,1,4] has its singleton squeezed to [10240,4]. FP8-native GEMM + the #518 FP8
roofline are the deferred perf phase; here every GDN projection is a BF16 device blob so the
engine runs gpu_matmul_bf16_dev (BF16 weight, FP32 out).

Output: <out_dir>/<gdn_slot>.<name>.bf16  (raw row-major BF16 bytes, what load_bf16_file_to_gpu reads),
name in {in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,A_log,dt_bias,norm}. gdn_slot is the
ordered index over the linear_attention layers (same order GdnStatePools / the engine use).
"""
import json
import os
import sys

import torch
from safetensors import safe_open

FP8 = {"in_proj_qkv", "in_proj_z", "out_proj"}          # weight_fp8 * weight_scale[out,1]
BF16 = {"in_proj_a", "in_proj_b", "conv1d", "A_log", "dt_bias", "norm"}
NAMES = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
         "conv1d", "A_log", "dt_bias", "norm"]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    ckpt, out = pos[0], pos[1]
    # PHASE-B (Adam 2026-08-17: "NVFP4 all the way down"): --proj-nvfp4 emits the 3 big GDN projections
    # (in_proj_qkv/z/out_proj) as NVFP4 (4x smaller than bf16, onto the fast fp4 GEMM). The 6 small tensors
    # (in_proj_a/b, conv1d, A_log, dt_bias, norm) STAY bf16 — recurrence-critical scalars, negligible bytes,
    # NVFP4 there would corrupt the linear-attention numerics for zero perf. NEW fidelity trade — gate vs goldens.
    PROJ_NVFP4 = "--proj-nvfp4" in sys.argv
    write_nvfp4 = None
    if PROJ_NVFP4:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from quantize_nvfp4 import write_nvfp4  # noqa: F811
    os.makedirs(out, exist_ok=True)

    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    layer_types = tc["layer_types"]
    gdn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print(f"GDN layers: {len(gdn_layers)}  {gdn_layers[:6]}...{gdn_layers[-3:]}", flush=True)

    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    _handles: dict = {}

    def get(key: str) -> torch.Tensor:
        fn = idx[key]
        if fn not in _handles:
            _handles[fn] = safe_open(os.path.join(ckpt, fn), framework="pt")
        return _handles[fn].get_tensor(key)

    def full_key(layer: int, suffix: str) -> str:
        # robust to the model.* vs model.language_model.* prefix
        want = f".layers.{layer}.linear_attn.{suffix}"
        for k in idx:
            if k.endswith(want):
                return k
        raise KeyError(want)

    def write_bf16(t: torch.Tensor, path: str) -> None:
        b = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
        with open(path, "wb") as f:
            f.write(b)

    def emit_proj(w_fp32_np, slot: int, name: str) -> str:
        """A big GDN projection (qkv/z/out): NVFP4 if --proj-nvfp4, else BF16. Returns the emitted ext."""
        if PROJ_NVFP4:
            p = os.path.join(out, f"{slot}.{name}")
            write_nvfp4(p + ".tmp.nvfp4", w_fp32_np)
            os.replace(p + ".tmp.nvfp4", p + ".nvfp4")
            return "nvfp4"
        write_bf16(torch.from_numpy(w_fp32_np), os.path.join(out, f"{slot}.{name}.bf16"))
        return "bf16"

    total = 0
    for slot, layer in enumerate(gdn_layers):
        for name in NAMES:
            if name in FP8:
                # Auto-detect source: the NVFP4 ckpt stores these FP8 E4M3 + weight_scale[out,1]; the BASE
                # bf16 ckpt stores them raw bf16 (no scale). Base-bf16 source gives the cleanest parity
                # (same source as the HF goldens) — so we handle both.
                scale_suffix = f".layers.{layer}.linear_attn.{name}.weight_scale"
                if any(k.endswith(scale_suffix) for k in idx):
                    w = get(full_key(layer, name + ".weight")).to(torch.float32)        # e4m3 -> f32
                    sc = get(full_key(layer, name + ".weight_scale")).to(torch.float32)  # [out,1]
                    w_np = (w * sc).cpu().numpy()                                        # per-output-channel dequant
                else:
                    w_np = get(full_key(layer, name + ".weight")).to(torch.float32).cpu().numpy()  # base bf16 raw
                emit_proj(w_np, slot, name)                                             # NVFP4 (--proj-nvfp4) or BF16
            else:
                t = get(full_key(layer, name if name in ("A_log", "dt_bias")
                                 else (name + ".weight")))
                if name == "conv1d" and t.dim() == 3:
                    t = t.squeeze(1)                                                   # [10240,1,4]->[10240,4]
                write_bf16(t, os.path.join(out, f"{slot}.{name}.bf16"))
            total += 1
        if slot < 2 or slot == len(gdn_layers) - 1:
            print(f"  slot {slot:2} (layer {layer:2}): 9 blobs written", flush=True)
    # a tiny manifest so the loader/engine can sanity-check n_gdn + order
    json.dump({"n_gdn": len(gdn_layers), "gdn_layers": gdn_layers, "names": NAMES,
               "proj_dtype": ("nvfp4" if PROJ_NVFP4 else "bf16"),   # in_proj_qkv/z/out_proj
               "small_dtype": "bf16",                               # in_proj_a/b/conv1d/A_log/dt_bias/norm (always bf16)
               "proj_nvfp4": PROJ_NVFP4, "fp8_dequant": sorted(FP8)},
              open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"done: {total} blobs for {len(gdn_layers)} GDN slots -> {out}", flush=True)


if __name__ == "__main__":
    main()
