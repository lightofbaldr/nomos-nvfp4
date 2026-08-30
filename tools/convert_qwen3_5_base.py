#!/usr/bin/env python3
"""Qwen3.8-27B (qwen3_5) base BF16 safetensors -> flat kernel serve dir (softmax + MLP + embed + head +
norms). The GDN linear_attn weights are a SEPARATE converter (tools/convert_qwen3_5_gdn.py). Run with the
system python (torch):

    /usr/bin/python3 tools/convert_qwen3_5_base.py \
        ~/nomos_data/qwen3_8_27b-base-bf16 ~/nomos_data/qwen3_8_27b-nvfp4 --format q4     # GB10 dev
        ...                                                            --format nvfp4   # discrete deploy

SOURCE = the BASE BF16 checkpoint (muse's parity argument): the goldens are HF-bf16, so our-kernel-on-
our-quant-of-bf16 isolates kernel arithmetic + our quant; repacking the third-party FP8/NVFP4 checkpoint
would conflate a third source and make a layer-30 divergence unattributable. Emits into the SAME root the
GDN blobs (<root>/gdn/) already sit under.

QWEN CONTRACT (source-verified w/ Kvasir, all silent if wrong):
  - NORM CENTERING: Qwen3_5RMSNorm.forward = norm(x)*(1+weight); the kernel primitive multiplies the stored
    weight directly, so PRE-ADD +1 to ALL regular norms: input_layernorm, post_attention_layernorm, q_norm,
    k_norm, AND final norm. (GDN gated norm is RAW — different class, handled by the GDN converter.)
  - Only 2 layernorms/layer (input + post_attention) — NO pre/post-feedforward norms (that's Gemma/Muse).
  - SOFTMAX layers only (layer_types == full_attention) carry self_attn.{q,k,v,o}_proj + q_norm + k_norm.
    q_proj is the DOUBLE-WIDTH [NH*2*HD=12288, D] blob written RAW in checkpoint order (query|gate per head,
    interleaved); the engine GEMMs N=12288 and splits query|gate internally. Do NOT pre-split.
  - GDN layers carry NO self_attn — only the 2 norms + MLP here; their linear_attn is the GDN converter's.
  - UNTIED lm_head (tie_word_embeddings=False) — separate file, asserted distinct from embed.
  - attention_bias=False → no bias tensors anywhere.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_q4_0 import quantize as q4_quantize, write_q4  # noqa: E402

LINEARS_ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
LINEARS_MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
CENTERED_LAYER_NORMS = ("input_layernorm", "post_attention_layernorm")   # +1, every layer
CENTERED_ATTN_NORMS = ("self_attn.q_norm", "self_attn.k_norm")           # +1, softmax layers only


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) < 2:
        print(__doc__); sys.exit(1)
    in_dir, out_dir = pos[0], pos[1]
    fmt = "q4"
    for a in sys.argv[1:]:
        if a.startswith("--format"):
            fmt = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    if fmt not in ("q4", "nvfp4"):
        sys.exit(f"FATAL: --format must be q4 or nvfp4, got {fmt!r}")
    write_nvfp4 = None
    if fmt == "nvfp4":
        from quantize_nvfp4 import write_nvfp4  # noqa: F811
    os.makedirs(out_dir, exist_ok=True)

    from safetensors import safe_open
    cfg = json.load(open(os.path.join(in_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    n_layers = tc["num_hidden_layers"]
    layer_types = tc["layer_types"]
    assert not tc.get("tie_word_embeddings", True), "config says tied — contract changed, stop"
    idx = json.load(open(os.path.join(in_dir, "model.safetensors.index.json")))["weight_map"]
    _h: dict = {}

    def full_key(suffix: str) -> str:
        for k in idx:
            if k.endswith(suffix):
                return k
        raise KeyError(suffix)

    def T(suffix: str) -> np.ndarray:
        key = full_key(suffix)
        sh = idx[key]
        if sh not in _h:
            _h[sh] = safe_open(os.path.join(in_dir, sh), framework="pt")
        return _h[sh].get_tensor(key).to(torch.float32).numpy()

    def stem(name: str) -> str:
        return os.path.join(out_dir, "layers_" + name.replace(".", "_") + "_weight")

    def write_f32(path: str, a: np.ndarray) -> None:
        np.ascontiguousarray(a, dtype=np.float32).tofile(path + ".tmp"); os.replace(path + ".tmp", path)

    def quant(W: np.ndarray, s: str) -> None:
        if os.path.exists(s + "." + fmt):
            return
        if fmt == "q4":
            s16, p4, n = q4_quantize(W); write_q4(s + ".tmp", s16, p4, n); os.replace(s + ".tmp.q4", s + ".q4")
        else:
            write_nvfp4(s + ".tmp.nvfp4", W); os.replace(s + ".tmp.nvfp4", s + ".nvfp4")

    def center(suffix: str, out_name: str) -> None:
        raw = T(suffix); cen = raw + 1.0
        d = float(np.mean(cen) - np.mean(raw))
        if abs(d - 1.0) > 1e-5:
            sys.exit(f"FATAL: centering delta {d} != 1.0 for {suffix}")
        write_f32(os.path.join(out_dir, out_name + ".bin"), cen)

    report = {"source": in_dir, "format": fmt, "n_layers": n_layers,
              "centered_norms": [], "final_norm_centered": True, "untied_lm_head": None}

    for i in range(n_layers):
        pre = f".layers.{i}."
        # every layer: 2 centered norms + MLP
        for nm in CENTERED_LAYER_NORMS:
            center(pre + nm + ".weight", f"layers_{i}_{nm}_weight")
            if i == 0:
                report["centered_norms"].append(f"layers_N_{nm}")
        for lin in LINEARS_MLP:
            quant(T(pre + lin + ".weight"), stem(f"{i}.{lin}"))
        # softmax layers only: attn projections + q/k norm
        if layer_types[i] == "full_attention":
            for lin in LINEARS_ATTN:
                quant(T(pre + lin + ".weight"), stem(f"{i}.{lin}"))
            for nm in CENTERED_ATTN_NORMS:
                center(
                    pre + nm + ".weight",
                    f"layers_{i}_{nm.replace('.', '_')}_weight",
                )
        if i % 8 == 0:
            print(f"  layer {i}/{n_layers} ({layer_types[i]})", flush=True)

    # embed + UNTIED head + final norm
    emb = T("embed_tokens.weight")
    write_f32(os.path.join(out_dir, "embed_tokens_weight.bin"), emb)
    lm = T("lm_head.weight")
    nrows = min(4096, lm.shape[0])
    if np.array_equal(lm[:nrows], emb[:nrows]):
        sys.exit("FATAL: lm_head == embed on first rows but config says untied — stop")
    a, b = lm[:nrows].ravel(), emb[:nrows].ravel()
    cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
    report["untied_lm_head"] = {"cos_vs_embed_first_rows": round(cos, 6)}
    # The engine loads the head as lm_head_weight.<fmt> (q4/nvfp4) — the GEMM path, same as every other
    # linear. write_f32(.bin) alone left the head NULL → zero logits / top1=0 (Codex run1 provenance).
    # Emit the quantized head (primary) AND keep the .bin (fp32 reference, mirrors gemma's multi-format embed).
    write_f32(os.path.join(out_dir, "lm_head_weight.bin"), lm)
    quant(lm, os.path.join(out_dir, "lm_head_weight"))
    center("model.norm.weight" if any(k.endswith("model.norm.weight") for k in idx) else "norm.weight",
           "norm_weight")

    json.dump(report, open(os.path.join(out_dir, "convert_base_manifest.json"), "w"), indent=2)
    print(f"\n  done -> {out_dir}  format={fmt}")
    print(f"  centered (+1): {report['centered_norms']} + q/k_norm (softmax) + FINAL norm (centered=True)")
    print(f"  lm_head cos vs embed: {report['untied_lm_head']['cos_vs_embed_first_rows']} (near 0 => untied)")


if __name__ == "__main__":
    main()
