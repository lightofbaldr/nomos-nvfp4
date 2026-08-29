#!/usr/bin/env python3
"""FAITHFUL direct-repack: RedHat compressed-tensors NVFP4 -> Nomos kernel .nvfp4,
preserving packed BYTES + E4M3 scale BYTES verbatim (NO dequant/requant).
Header multiplicative global = 1.0 / source weight_global_scale (CT stores a divisor;
vLLM inverts on load). Per Nano's #1 finding.

Kernel .nvfp4 = 24B header [n:i64, n16:i64, gs:f32, pad:f32] + bs[N,K/16] u8(E4M3) + packed[N,K/2] u8.
Includes a parity gate: decode(my repack) vs CT dequant -> expect ~0 rel error.
Usage: python3 direct_repack.py <redhat_dir> <out_dir> [--verify-only]
"""
import os, sys, glob, struct, json, numpy as np, torch
from safetensors import safe_open

# Nomos local NVFP4 decode LUTs (must match tools/quantize_nvfp4.py)
E2M1_MAG = np.array([0,0.5,1,1.5,2,3,4,6], dtype=np.float32)
def _e4m3_lut():
    # float8_e4m3fn positive magnitudes for all 256 byte values (sign in bit7)
    out=np.zeros(256,dtype=np.float32)
    for b in range(256):
        s=-1.0 if (b>>7)&1 else 1.0; e=(b>>3)&0xF; m=b&0x7
        if e==0: val=(m/8.0)*2**(-6)
        elif e==0xF and m==0x7: val=float("nan")
        else: val=(1+m/8.0)*2**(e-7)
        out[b]=s*val
    return out
E4M3_ALL=_e4m3_lut()

def _our_stem(name):  # HF proj name -> kernel file stem
    n=name.replace("model.language_model.","").replace(".weight","")
    n=n.replace("layers.","layers_").replace(".self_attn.","_self_attn_").replace(".mlp.","_mlp_")
    return n.replace(".","_")+"_weight"

def ct_dequant(packed, scale_e4m3, wgs, N, K):
    NB=K//16
    nib=np.empty((N,K),dtype=np.uint8)
    nib[:,0::2]=packed&0x0F; nib[:,1::2]=packed>>4
    sign=np.where((nib&8)!=0,-1.0,1.0).astype(np.float32); mag=E2M1_MAG[nib&7]
    e2=(sign*mag).reshape(N,NB,16)
    bs=E4M3_ALL[scale_e4m3].reshape(N,NB)[:,:,None]
    return (e2*bs/np.float32(wgs)).reshape(N,K).astype(np.float32)

def nomos_decode(gs, bs_bytes, packed, N, K):
    NB=K//16
    nib=np.empty((N,K),dtype=np.uint8)
    nib[:,0::2]=packed&0x0F; nib[:,1::2]=packed>>4
    sign=np.where((nib&8)!=0,-1.0,1.0).astype(np.float32); mag=E2M1_MAG[nib&7]
    fp4=(sign*mag).reshape(N,NB,16)
    bs=E4M3_ALL[bs_bytes].reshape(N,NB)[:,:,None]
    return (np.float32(gs)*bs*fp4).reshape(N,K).astype(np.float32)

def main():
    src, out = sys.argv[1], sys.argv[2]
    verify_only = "--verify-only" in sys.argv
    os.makedirs(out, exist_ok=True)
    files=sorted(glob.glob(os.path.join(src,"*.safetensors")))
    n_done=0; worst=0.0
    for f in files:
        with safe_open(f, framework="pt") as st:
            keys=st.keys()
            bases=sorted({k[:-len(".weight_packed")] for k in keys if k.endswith(".weight_packed")})
            for base in bases:
                packed=st.get_tensor(base+".weight_packed").contiguous().view(torch.uint8).numpy()   # [N,K/2] u8
                scale =st.get_tensor(base+".weight_scale").contiguous().view(torch.uint8).numpy()     # [N,K/16] E4M3 raw bytes
                _wgt  =st.get_tensor(base+".weight_global_scale")
                assert _wgt.numel()==1, f"{base}.weight_global_scale not scalar (numel={_wgt.numel()}); NVFP4 global scale must be per-tensor"
                wgs   =float(_wgt.reshape(-1)[0].item())
                N,Khalf=packed.shape; K=Khalf*2
                scale_u8=scale
                # W4A4-ready: header pad float carries input_global_multiplier = 1/input_global_scale
                # (calibrated activation scale; codex #2). 0.0 if the checkpoint has no static act global.
                igs=None
                for kk in (base+".input_global_scale", base+".input_scale"):
                    if kk in keys:
                        _ig=st.get_tensor(kk)
                        assert _ig.numel()==1, f"{kk} not scalar (numel={_ig.numel()}); calibrated input global scale must be per-tensor"
                        igs=float(_ig.reshape(-1)[0].item()); break
                pad_ig=np.float32(1.0/igs) if (igs and igs!=0) else np.float32(0.0)
                gs=np.float32(1.0/wgs)
                # parity gate (first 3 tensors): my decode vs CT dequant
                if n_done<3:
                    a=nomos_decode(gs,scale_u8,packed,N,K).ravel()
                    b=ct_dequant(packed,scale_u8,wgs,N,K).ravel()
                    rel=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-9)); worst=max(worst,rel)
                    print(f"  [gate] {_our_stem(base)}: N={N} K={K} wgs={wgs:.6g} decode-vs-CT rel={rel:.2e}", flush=True)
                if not verify_only:
                    path=os.path.join(out,_our_stem(base)+".nvfp4")
                    with open(path,"wb") as w:
                        w.write(struct.pack("<qqff", N*K, (N*K)//16, float(gs), float(pad_ig)))
                        w.write(np.ascontiguousarray(scale_u8).tobytes())
                        w.write(np.ascontiguousarray(packed).tobytes())
                n_done+=1
    print(f"[done] {n_done} projections direct-repacked -> {out} (gate worst rel={worst:.2e})", flush=True)

if __name__=="__main__":
    main()
