#!/usr/bin/env python3
import os
"""Decisive cheap check for the full-vs-sliding attention-scale hypothesis.

The kernel applies NO runtime 1/sqrt(head_dim) attention scale (verified: not in
GEMM alpha, softmax, or qk_norm). So the scale must be folded into the q_norm (or
k_norm / q_proj) weights during flat-weight conversion. qk_norm in the kernel does
`data*ri*w` (raw w, NOT 1+w) — so the converter must also fold Gemma's +1 centering.

Gemma-4 correct attention scale = 1/sqrt(head_dim), which DIFFERS by layer type:
  sliding layers (head_dim 256): 1/16   = 0.0625
  full    layers (head_dim 512): 1/sqrt(512) = 0.04419

We compare the FOLDED scale at L20 (sliding) vs L17 (full):
  folded(L) := median( flat_qnorm(L) / (1 + hf_qnorm(L)) )   [elementwise]
Interpretation:
  folded(L20) ~ 0.0625 and folded(L17) ~ 0.0442  -> per-layer-correct  (hypothesis DEAD)
  folded(L20) ~ folded(L17) ~ 0.0625             -> UNIFORM bug         (hypothesis CONFIRMED: full layers 1.41x over-scaled)
  folded ~ 1.0 (no scale)                         -> scale folded elsewhere / missing -> check q_proj
Reads HF tensors straight from R2 via rclone --offset/--count (KBs, no full pull).
"""
import subprocess, json, struct, sys
import numpy as np
import torch

R2 = os.environ.get("NOMOS_R2_SRC", "")
FLAT = "/workspace/weights/"
SHARD = {17: "model-00002-of-00002.safetensors", 20: "model-00001-of-00002.safetensors"}
# layer -> (head_dim, attention type)
LAYERS = {20: (256, "sliding"), 17: (512, "full")}

_hdr_cache = {}

def rclone_bytes(shard, offset, count):
    r = subprocess.run(["rclone", "cat", "--offset", str(offset), "--count", str(count), R2 + shard],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"rclone failed: {r.stderr.decode()[:300]}")
    return r.stdout

def header(shard):
    if shard in _hdr_cache:
        return _hdr_cache[shard]
    n = struct.unpack("<Q", rclone_bytes(shard, 0, 8))[0]
    hj = json.loads(rclone_bytes(shard, 8, n).decode())
    _hdr_cache[shard] = (hj, 8 + n)
    return _hdr_cache[shard]

_ST_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

def hf_tensor(shard, name):
    hj, data_start = header(shard)
    if name not in hj:
        raise KeyError(f"{name} not in {shard}")
    meta = hj[name]
    s, e = meta["data_offsets"]
    raw = rclone_bytes(shard, data_start + s, e - s)
    t = torch.frombuffer(bytearray(raw), dtype=_ST_DT[meta["dtype"]]).float().numpy()
    return t.reshape(meta["shape"]), meta["shape"], meta["dtype"]

def flat_tensor(name, n_expect):
    b = open(FLAT + name, "rb").read()
    for dt, sz in (("<f4", 4),):
        if len(b) == n_expect * sz:                       # raw, no header
            return np.frombuffer(b, dtype=np.float32), "raw"
        if len(b) == n_expect * sz + 64:                  # 64-byte header
            return np.frombuffer(b[64:], dtype=np.float32), "hdr64"
    return np.frombuffer(b, dtype=np.float32), f"len={len(b)}({len(b)/4:.0f}f)"

def med_ratio(num, den):
    den = np.where(np.abs(den) < 1e-8, np.nan, den)
    r = num / den
    return float(np.nanmedian(r)), float(np.nanstd(r[np.isfinite(r)]))

print("=" * 78)
for L, (hd, kind) in LAYERS.items():
    sh = SHARD[L]
    print(f"\n### Layer {L} ({kind}, head_dim={hd})  expected 1/sqrt(hd)={1/np.sqrt(hd):.5f}")
    for proj in ("q_norm", "k_norm", "q_proj"):
        name = f"model.language_model.layers.{L}.self_attn.{proj}.weight"
        flat_name = f"layers_{L}_self_attn_{proj}_weight.bin"
        try:
            hf, shp, dt = hf_tensor(sh, name)
        except Exception as ex:
            print(f"  {proj}: HF read failed: {ex}"); continue
        n_expect = int(np.prod(shp))
        try:
            flat, how = flat_tensor(flat_name, n_expect)
        except Exception as ex:
            print(f"  {proj}: flat read failed: {ex}"); continue
        line = f"  {proj:7s} hf{tuple(shp)}{dt} flat[{flat.size}]({how})"
        if flat.size != hf.size:
            print(line + f"  SIZE MISMATCH hf={hf.size} flat={flat.size}"); continue
        hf = hf.ravel()
        if proj.endswith("norm"):
            r1, s1 = med_ratio(flat, 1.0 + hf)   # kernel uses *w, Gemma is *(1+w)
            r0, s0 = med_ratio(flat, hf)         # in case +1 already folded differently
            print(line + f"  flat/(1+hf)={r1:.5f}(sd{s1:.1e})  flat/hf={r0:.5f}")
        else:
            r0, s0 = med_ratio(flat, hf)
            print(line + f"  flat/hf={r0:.5f}(sd{s0:.1e})")
print("\n" + "=" * 78)
print("READ: compare q_norm flat/(1+hf) at L20(sliding) vs L17(full).")
print("  ~0.0625 both  -> UNIFORM scale bug (full layers 1.41x hot)  [HYPOTHESIS CONFIRMED]")
print("  L20~0.0625 L17~0.0442 -> per-layer correct                  [HYPOTHESIS DEAD]")
print("  ~1.0 -> scale not in q_norm; look at q_proj flat/hf ratio")
