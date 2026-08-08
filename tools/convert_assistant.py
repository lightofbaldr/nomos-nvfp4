#!/usr/bin/env python3
"""Convert the Gemma-4-31B MTP assistant GGUF (Q8_0) -> our weight format for the MTP drafter.

Assistant = 4-layer gemma4-style MTP head (internal dim 1024), is_mem_shared:
  - input: concat[31B final hidden(5376), 31B token_embed(5376)] -> nextn.pre_projection -> 1024
  - 4 layers: attn_q + attn_output ONLY (NO own K/V — attends the 31B's SHARED int4 KV).
    head_dim 256 for layers 0-2 (SWA), 512 for layer 3 (global). gemma4 4-norm + SwiGLU(FF 8192).
  - output: output_norm -> nextn.post_projection(1024->5376) ... tied tok_embd(1024->262144) full vocab.
  - K=4 native draft (4 nextn layers). Q8 weights -> int8-MMA. NO d2t (full target vocab).

Out: ~/nomos_data/assistant/  (matmul weights -> .q8 int8+scale ; norms/embed -> fp32 .bin ; manifest.json)

.q8 format (mirror of .q4 but full int8): [int64 n][int64 nb][nb fp16 scales][n int8 values], block=32,
scale=amax/127, q=clip(round(w/scale),-127,127).  Q8_0 GGUF is already int8+per-32-scale; we dequant→requant
to our layout for a uniform loader.
"""
import os, json, numpy as np

GROUP = 32
SRC = os.path.expanduser("~/Services/inference/models/assistant/gemma4-31B-it-assistant-Q8_0.gguf")
OUT = os.path.expanduser("~/nomos_data/assistant")

def q8(w):
    w = w.astype(np.float32).ravel(); n = w.size
    pad = (-n) % GROUP
    if pad: w = np.concatenate([w, np.zeros(pad, np.float32)])
    wb = w.reshape(-1, GROUP)
    amax = np.max(np.abs(wb), axis=1)
    scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(wb / scale[:, None]), -127, 127).astype(np.int8)
    return scale.astype(np.float16), q.reshape(-1)[:n], n

def write_q8(path, w):
    scale16, qi, n = q8(w)
    with open(path, "wb") as f:
        f.write(np.array([n, scale16.shape[0]], dtype=np.int64).tobytes())
        f.write(scale16.tobytes()); f.write(qi.tobytes())
    return n

def main():
    from gguf import GGUFReader, dequantize, GGMLQuantizationType
    os.makedirs(OUT, exist_ok=True)
    r = GGUFReader(SRC)
    T = {t.name: t for t in r.tensors}
    def arr(name):
        t = T[name]
        shp = tuple(int(x) for x in reversed(t.shape))   # gguf shape is reversed vs numpy
        if t.tensor_type == GGMLQuantizationType.F32:
            return np.asarray(t.data, dtype=np.float32).reshape(shp)
        deq = dequantize(np.asarray(t.data), t.tensor_type)   # -> fp32, numpy-order
        return deq.astype(np.float32).reshape(shp)
    # name -> (out-stem, kind)  kind: q8 | bin
    MM = ["nextn.pre_projection.weight", "nextn.post_projection.weight", "token_embd.weight"]
    for i in range(4):
        MM += [f"blk.{i}.attn_q.weight", f"blk.{i}.attn_output.weight",
               f"blk.{i}.ffn_gate.weight", f"blk.{i}.ffn_up.weight", f"blk.{i}.ffn_down.weight"]
    BIN = ["output_norm.weight", "rope_freqs.weight"]
    for i in range(4):
        BIN += [f"blk.{i}.attn_norm.weight", f"blk.{i}.attn_q_norm.weight",
                f"blk.{i}.ffn_norm.weight", f"blk.{i}.post_attention_norm.weight",
                f"blk.{i}.post_ffw_norm.weight", f"blk.{i}.layer_output_scale.weight"]
    manifest = {"source": SRC, "tensors": {}}
    for name in MM:
        if name not in T: print("  !! MISSING", name); continue
        w = arr(name); stem = name.replace(".", "_")
        n = write_q8(os.path.join(OUT, stem + ".q8"), w)
        print(f"  {stem:40s} .q8  {list(w.shape)}"); manifest["tensors"][stem] = {"shape": list(w.shape), "fmt": "q8"}
    for name in BIN:
        if name not in T: print("  !! MISSING", name); continue
        w = arr(name); stem = name.replace(".", "_")
        w.astype(np.float32).ravel().tofile(os.path.join(OUT, stem + ".bin"))
        print(f"  {stem:40s} .bin {list(w.shape)}"); manifest["tensors"][stem] = {"shape": list(w.shape), "fmt": "f32"}
    # also dequant token_embd -> fp32 .bin for the INPUT embed lookup (the 31B embed is used for input concat,
    # but ship the assistant's own embed too for completeness / the tied output path).
    manifest["arch"] = {
        "internal_dim": 1024, "backbone_dim": 5376, "n_layer": 4, "ffn": 8192, "vocab": 262144,
        "head_dim": [256, 256, 256, 512], "sliding_window_pattern": [True, True, True, False],
        "window": 1024, "K_draft": 4, "shared_kv": True, "own_kv": False, "tied_output": True,
        "no_d2t": True, "input_concat": "[31B_final_hidden(5376), 31B_token_embed(5376)] -> pre_proj -> 1024",
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"\n  wrote {len(manifest['tensors'])} tensors + manifest -> {OUT}")

if __name__ == "__main__":
    main()
