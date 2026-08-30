#!/usr/bin/env python3
"""GATE for 7f08ed6 (nomos_generate profile-EOS fix): K6 nomos_generate must now stop at eos and equal
K1 (prefill+decode_step host-stop) token-for-token including the trailing eos. Records rep=1.15 arm."""
import os, ctypes as C, numpy as np
from transformers import AutoTokenizer
SO=os.environ["QWEN_SO"]; WTS=os.environ["QWEN_WEIGHTS"]
tok=AutoTokenizer.from_pretrained(os.environ["TOK_DIR"])
STOP={248046,248044}
m=C.CDLL(SO)
m.nomos_init.restype=C.c_int64; m.nomos_init.argtypes=[C.c_char_p]
m.nomos_prefill.restype=C.c_int32; m.nomos_prefill.argtypes=[C.c_int64,C.c_int64,C.c_int32,C.c_int64]
m.nomos_decode_step.restype=C.c_int32; m.nomos_decode_step.argtypes=[C.c_int64,C.c_int32,C.c_int64]
m.nomos_generate.argtypes=[C.c_int64,C.c_int64,C.c_int32,C.c_int32,C.c_int64,C.c_int32,C.c_float,C.c_float,C.c_float]
m.nomos_generate.restype=C.c_int32
h=m.nomos_init(WTS.encode()); assert h
s=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],
                          add_generation_prompt=True, tokenize=False, enable_thinking=True)
ids=np.ascontiguousarray(np.asarray(tok(s,add_special_tokens=False).input_ids,dtype=np.int32))
lg=np.zeros(248320,np.float32)
# K1: host-stop greedy (reference)
rc=m.nomos_prefill(h,ids.ctypes.data,len(ids),lg.ctypes.data); assert rc==0
t=int(lg.argmax()); k1=[t]
for _ in range(223):
    if t in STOP: break
    rc=m.nomos_decode_step(h,t,lg.ctypes.data); assert rc==0
    t=int(lg.argmax()); k1.append(t)
print(f"K1 host-stop: ntok={len(k1)} last={k1[-1]}")
# K6: nomos_generate greedy rep=1.0 (must stop, must == K1)
outb=np.zeros(224,np.int32)
n=m.nomos_generate(h,ids.ctypes.data,len(ids),224,outb.ctypes.data,224,0.0,-1.0,1.0)
k6=outb[:max(n,0)].tolist()
print(f"K6 nomos_generate: nout={n} last={k6[-1] if k6 else None}")
eq = (k6 == k1)
print(f"EQUALITY K6==K1: {eq}")
if not eq:
    for i,(a,b) in enumerate(zip(k1,k6)):
        if a!=b: print(f"  first divergence @{i}: K1={a} K6={b}"); break
    print(f"  len K1={len(k1)} K6={len(k6)}")
print(f"K6 text: {tok.decode(k6, skip_special_tokens=False)!r}")
# K6b: rep=1.15 now LIVE — record (no equality requirement; must still STOP)
n2=m.nomos_generate(h,ids.ctypes.data,len(ids),224,outb.ctypes.data,224,0.0,-1.0,1.15)
k6b=outb[:max(n2,0)].tolist()
stopped = bool(k6b and k6b[-1] in STOP and n2 < 224)
print(f"K6b rep=1.15: nout={n2} stopped_on_eos={stopped} identical_to_rep1.0={k6b==k6}")
print(f"K6b text: {tok.decode(k6b, skip_special_tokens=False)!r}")
print("VERDICT:", "PASS" if (eq and k6[-1] in STOP and n<224 and stopped) else "FAIL")
