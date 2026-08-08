#!/usr/bin/env python3
"""bf16 safetensors -> our Q4_0 (.q4), per-layer Linears only. The SINGLE-hop bf16->Q4 path
(vs the NVFP4->Q4 double-hop in convert_redhat_nvfp4.py) — isolates whether double-quant was
what broke the RedHat-Q4 target. Non-quant (norms/embed/lm_head/layer_scalar) stay shared (use
the staging dir's symlinks). Atomic writes + skip-existing => kill-safe/resumable.

Usage: python3 tools/convert_bf16_to_q4.py <bf16_safetensors_dir> <out_dir>
"""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_q4_0 import quantize as q4_quantize, write_q4

LINEAR_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _our_stem(base):  # model.language_model.layers.I.A.B.weight -> layers_I_A_B_weight
    p = base.replace("model.language_model.layers.", "").split(".")
    return f"layers_{p[0]}_{p[1]}_{p[2]}_weight" if len(p) >= 3 else base.replace(".", "_")


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) < 2:
        print(__doc__); sys.exit(1)
    in_dir, out_dir = pos[0], pos[1]
    os.makedirs(out_dir, exist_ok=True)
    from safetensors import safe_open
    import torch
    shards = sorted(glob.glob(os.path.join(in_dir, "*.safetensors")))
    assert shards, f"no .safetensors in {in_dir}"
    key_to_shard = {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                key_to_shard[k] = s
    lin_keys = [k for k in key_to_shard if k.endswith(".weight") and "language_model" in k
                and "vision_tower" not in k and any(k.endswith(suf + ".weight") for suf in LINEAR_SUFFIXES)]
    print(f"{len(lin_keys)} per-layer Linears across {len(shards)} shards", flush=True)
    assert lin_keys, "no language_model Linear .weight found"

    def _get(kk):
        with safe_open(key_to_shard[kk], framework="pt") as f:
            return f.get_tensor(kk)

    done = 0
    for k in sorted(lin_keys):
        base = k[:-len(".weight")]
        stem = os.path.join(out_dir, _our_stem(base + ".weight"))
        if os.path.exists(stem + ".q4"):
            done += 1; continue
        W = _get(k).float().numpy()                          # bf16 -> fp32 [N, K]
        if done == 0 or done == len(lin_keys):
            print(f"  [first] {base}: shape={W.shape} |W|max={np.abs(W).max():.4f}", flush=True)
        s16, p4, n = q4_quantize(W.ravel())
        write_q4(stem + ".tmp", s16, p4, n)
        os.replace(stem + ".tmp.q4", stem + ".q4")            # atomic
        done += 1
        if done % 60 == 0:
            print(f"  {done}/{len(lin_keys)} ...", flush=True)
    print(f"DONE: wrote {done} .q4 to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
