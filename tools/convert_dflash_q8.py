#!/usr/bin/env python3
"""Convert the DFlash drafter (bf16 safetensors) -> .q8 for the Q8 drafter path.

Why: the bf16 acceptance-ceiling A/B (2026-07-05) measured ~0.69 tokens/cycle of
quantization-induced acceptance loss vs our Q4 drafter. Q8 drafter weights are the
established law concession (see q8_weights.mojo — the MTP/EAGLE3 drafters already
run it: "naive-Q4 on the drafter costs ~5pt, Q8 is nearly free"). The DFlash port
went straight to Q4 by reusing the target's loaders; this restores the concession.

Emits FUSED tensors so the Mojo loader needs no q8 concat variants:
  fc_weight.q8                          [5376, 32256]
  layers_{i}_self_attn_qkv_weight.q8    [10240, 5376]  (q;k;v — same order as load_to_gpu_q4_concat3)
  layers_{i}_self_attn_o_proj_weight.q8 [5376, 8192]
  layers_{i}_mlp_gate_up_weight.q8      [21504, 5376]  (gate;up — same order as load_to_gpu_q4_concat2)
  layers_{i}_mlp_down_proj_weight.q8    [5376, 10752]

.q8 format (q8_weights.mojo reader): [int64 n][int64 nb][nb fp16 scales][n int8],
group=32, symmetric scale=amax/127, q=clip(round(w/scale),-127,127). Row-major
[N,K]; every K here is divisible by 32 so rows stay group-aligned after concat.
"""
import os
import numpy as np
import torch
from safetensors import safe_open

GROUP = 32
SRC = os.path.expanduser("~/nomos_data/dflash-gemma-4-31b/model.safetensors")
OUT = os.path.expanduser("~/nomos_data/dflash-gemma-4-31b-flat")
LAYERS = 5


def write_q8(path, w):
    w = w.astype(np.float32).ravel()
    n = w.size
    assert n % GROUP == 0, f"{path}: {n} % {GROUP} != 0"
    wb = w.reshape(-1, GROUP)
    amax = np.max(np.abs(wb), axis=1)
    scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(wb / scale[:, None]), -127, 127).astype(np.int8)
    nb = scale.size
    with open(path, "wb") as f:
        f.write(np.int64(n).tobytes())
        f.write(np.int64(nb).tobytes())
        f.write(scale.astype(np.float16).tobytes())
        f.write(q.tobytes())
    print(f"{os.path.basename(path):>42}  n={n:>10}  nb={nb:>8}")


def main():
    st = safe_open(SRC, framework="pt")

    def t(name):
        return st.get_tensor(name).to(torch.float32).numpy()

    write_q8(os.path.join(OUT, "fc_weight.q8"), t("fc.weight"))
    for i in range(LAYERS):
        p = f"layers.{i}."
        o = os.path.join(OUT, f"layers_{i}_")
        qkv = np.concatenate([
            t(p + "self_attn.q_proj.weight"),
            t(p + "self_attn.k_proj.weight"),
            t(p + "self_attn.v_proj.weight"),
        ], axis=0)
        write_q8(o + "self_attn_qkv_weight.q8", qkv)
        write_q8(o + "self_attn_o_proj_weight.q8", t(p + "self_attn.o_proj.weight"))
        gup = np.concatenate([
            t(p + "mlp.gate_proj.weight"),
            t(p + "mlp.up_proj.weight"),
        ], axis=0)
        write_q8(o + "mlp_gate_up_weight.q8", gup)
        write_q8(o + "mlp_down_proj_weight.q8", t(p + "mlp.down_proj.weight"))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
