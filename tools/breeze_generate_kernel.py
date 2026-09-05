#!/usr/bin/env python3
"""Tier-0 Breeze sampler over the kernel model + codec ABIs.

Inputs are the already-proven host-assembled cond/uncond prefix embeddings.
Sampling follows the shipped fast runtime: CFG, reserved-token suppression,
temperature, top-k, top-p, multinomial; depth is autoregressive per codebook.
"""
import argparse, ctypes
from pathlib import Path
import numpy as np

V=2051; EOS=2051; CODEBOOK=2048; CBS=16

def bind(path, paired=False):
    x=ctypes.CDLL(str(path)); P=ctypes.c_void_p
    x.nomos_breeze_model_init.argtypes=[ctypes.c_char_p];x.nomos_breeze_model_init.restype=ctypes.c_int64
    x.nomos_breeze_model_prefill_lane.argtypes=[ctypes.c_int64,ctypes.c_int32,P,ctypes.c_int32,P,P,P]
    x.nomos_breeze_model_depth_lane.argtypes=[ctypes.c_int64,ctypes.c_int32,P,P]
    x.nomos_breeze_model_depth_begin.argtypes=[ctypes.c_int64,ctypes.c_int32,ctypes.c_int64,P]
    x.nomos_breeze_model_depth_advance.argtypes=[ctypes.c_int64,ctypes.c_int32,ctypes.c_int32,ctypes.c_int64,P]
    x.nomos_breeze_model_step_backbone.argtypes=[ctypes.c_int64,ctypes.c_int32,P,P]
    if paired:
        x.nomos_breeze_model_depth_begin2.argtypes=[ctypes.c_int64,P,P]
        x.nomos_breeze_model_depth_advance2.argtypes=[ctypes.c_int64,ctypes.c_int32,P,P]
        x.nomos_breeze_model_step_backbone2.argtypes=[ctypes.c_int64,P,P]
    return x

def pick(logits,rng,temp=.9,top_k=50,top_p=1.0):
    x=np.asarray(logits,dtype=np.float64).copy();x[CODEBOOK:V]=-np.inf
    x/=temp
    if top_k>0:
        keep=np.argpartition(x,-min(top_k,len(x)))[-min(top_k,len(x)):];mask=np.ones(len(x),bool);mask[keep]=False;x[mask]=-np.inf
    order=np.argsort(x)[::-1];p=np.exp(x[order]-np.max(x));p/=p.sum()
    if top_p<1:
        cut=np.searchsorted(np.cumsum(p),top_p)+1;order=order[:cut];p=p[:cut];p/=p.sum()
    return int(rng.choice(order,p=p))

def cfg(a,b,scale):return a if b is None else b+scale*(a-b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-so',required=True);ap.add_argument('--codec-so',required=True);ap.add_argument('--model-blobs',required=True);ap.add_argument('--codec-blobs',required=True);ap.add_argument('--cond-prefix',required=True);ap.add_argument('--uncond-prefix');ap.add_argument('--cfg-scale',type=float,default=1.0);ap.add_argument('--frames',type=int,default=100);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output-prefix',required=True);ap.add_argument('--paired-cfg',action='store_true',help='Batch cond/uncond GEMMs; requires --uncond-prefix');a=ap.parse_args()
    if a.paired_cfg and not a.uncond_prefix:ap.error('--paired-cfg requires --uncond-prefix')
    M=bind(a.model_so,a.paired_cfg);h=M.nomos_breeze_model_init(a.model_blobs.encode());assert h
    prefixes=[np.ascontiguousarray(np.load(a.cond_prefix),'f4')]
    if a.uncond_prefix:prefixes.append(np.ascontiguousarray(np.load(a.uncond_prefix),'f4'))
    lm=[]
    for lane,x in enumerate(prefixes):
        # Host assemblers may retain the leading batch dimension.
        x=x.reshape(-1,2048)
        o=np.empty(EOS+1,'f4');assert M.nomos_breeze_model_prefill_lane(h,lane,x.ctypes.data,len(x),None,None,o.ctypes.data)==0;lm.append(o)
    rng=np.random.default_rng(a.seed);frames=[];history=[]
    for _ in range(a.frames):
        mixed=cfg(lm[0],lm[1] if len(lm)>1 else None,a.cfg_scale).copy()
        if history:
            for tok in set(history):mixed[tok]=mixed[tok]/1.1 if mixed[tok]>0 else mixed[tok]*1.1
        cb0=pick(mixed,rng)
        if cb0==EOS:break
        codes=np.zeros(CBS,dtype=np.int64);codes[0]=cb0
        for cb in range(1,CBS):
            ds=[]
            if a.paired_cfg:
                d=np.empty((2,V),'f4')
                if cb==1:rc=M.nomos_breeze_model_depth_begin2(h,codes.ctypes.data,d.ctypes.data)
                else:rc=M.nomos_breeze_model_depth_advance2(h,cb-1,codes.ctypes.data,d.ctypes.data)
                assert rc==0;ds=[d[0],d[1]]
            else:
                for lane in range(len(prefixes)):
                    d=np.empty(V,'f4')
                    if cb==1:rc=M.nomos_breeze_model_depth_begin(h,lane,codes.ctypes.data,d.ctypes.data)
                    else:rc=M.nomos_breeze_model_depth_advance(h,lane,cb-1,codes.ctypes.data,d.ctypes.data)
                    assert rc==0;ds.append(d)
            codes[cb]=pick(cfg(ds[0],ds[1] if len(ds)>1 else None,a.cfg_scale),rng)
        frames.append(codes.copy());history.append(cb0)
        lm=[]
        if a.paired_cfg:
            o=np.empty((2,EOS+1),'f4');assert M.nomos_breeze_model_step_backbone2(h,codes.ctypes.data,o.ctypes.data)==0;lm=[o[0],o[1]]
        else:
            for lane in range(len(prefixes)):
                o=np.empty(EOS+1,'f4');assert M.nomos_breeze_model_step_backbone(h,lane,codes.ctypes.data,o.ctypes.data)==0;lm.append(o)
    codes=np.stack(frames,axis=1)[None] if frames else np.empty((1,CBS,0),np.int64)
    out=Path(a.output_prefix);np.save(str(out)+'.codes.npy',codes)
    if len(frames):
        C=ctypes.CDLL(a.codec_so);C.nomos_breeze_codec_init.argtypes=[ctypes.c_char_p];C.nomos_breeze_codec_init.restype=ctypes.c_int64;C.nomos_breeze_codec_decode.argtypes=[ctypes.c_int64,ctypes.c_void_p,ctypes.c_int32,ctypes.c_void_p]
        ch=C.nomos_breeze_codec_init(a.codec_blobs.encode());pcm=np.empty(1920*len(frames),'f4');assert C.nomos_breeze_codec_decode(ch,codes.ctypes.data,len(frames),pcm.ctypes.data)==0;np.save(str(out)+'.pcm.npy',pcm)
    print(f'frames={len(frames)} codes={out}.codes.npy pcm={out}.pcm.npy')
if __name__=='__main__':main()
