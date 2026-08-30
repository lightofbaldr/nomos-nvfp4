#!/usr/bin/env python3
"""Gemma smoke for 1c18304: nomos_generate with the bar-harness convention (0.0,-1.0,1.0) must
equal the host-stop loop token-for-token and stop on gemma eos {1,106} — no bar-path regression."""
import os, ctypes as C, numpy as np
from transformers import AutoTokenizer
SO=os.environ["GEMMA_SO"]; WTS=os.environ["GEMMA_WEIGHTS"]
tok=AutoTokenizer.from_pretrained("/home/adam/nomos_data/gemma-4-31b-it-bf16")
STOP={1,106}
m=C.CDLL(SO)
m.nomos_init.restype=C.c_int64; m.nomos_init.argtypes=[C.c_char_p]
m.nomos_prefill.restype=C.c_int32; m.nomos_prefill.argtypes=[C.c_int64,C.c_int64,C.c_int32,C.c_int64]
m.nomos_decode_step.restype=C.c_int32; m.nomos_decode_step.argtypes=[C.c_int64,C.c_int32,C.c_int64]
m.nomos_generate.argtypes=[C.c_int64,C.c_int64,C.c_int32,C.c_int32,C.c_int64,C.c_int32,C.c_float,C.c_float,C.c_float]
m.nomos_generate.restype=C.c_int32
h=m.nomos_init(WTS.encode()); assert h
s=tok.apply_chat_template([{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],
                          tokenize=False, add_generation_prompt=True)
ids=np.ascontiguousarray(np.asarray(tok(s,add_special_tokens=False).input_ids,dtype=np.int32))
lg=np.zeros(262144,np.float32)
rc=m.nomos_prefill(h,ids.ctypes.data,len(ids),lg.ctypes.data); assert rc==0
t=int(lg.argmax()); ref=[t]
for _ in range(159):
    if t in STOP: break
    rc=m.nomos_decode_step(h,t,lg.ctypes.data); assert rc==0
    t=int(lg.argmax()); ref.append(t)
outb=np.zeros(160,np.int32)
n=m.nomos_generate(h,ids.ctypes.data,len(ids),160,outb.ctypes.data,160,0.0,-1.0,1.0)
g=outb[:max(n,0)].tolist()
eq=(g==ref); stopped=bool(g and g[-1] in STOP and n<160)
print(f"host-stop: ntok={len(ref)} last={ref[-1]} | nomos_generate: nout={n} last={g[-1] if g else None}")
print(f"EQUALITY: {eq}  stopped_on_eos: {stopped}")
print(f"text: {tok.decode(g,skip_special_tokens=False)[:200]!r}")
print("VERDICT:", "PASS" if (eq and stopped) else "FAIL")
