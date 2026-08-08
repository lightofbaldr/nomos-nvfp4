#!/usr/bin/env python3
"""Base decode fine-prof: prefill one bar prompt, run N decode steps under
NOMOS_PROFILE_FINE=1 so engine_decode prints [prof]/[prof-fine] (qkv+append,
attn, o+mlp per tok). This is the bandwidth-floor the verify forward is
compared against (same 31B NVFP4 weight read, S=1 GEMV path)."""
import ctypes, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(os.path.dirname(HERE), "libnomos_kernel.so")
WTS = os.environ["WEIGHTS"]
NTOK = int(os.environ.get("PERF_NTOK", "80"))
VOCAB = 262144
lib = ctypes.CDLL(SO)
c = ctypes
for n, (r, a) in {
    "nomos_init": (c.c_void_p, [c.c_char_p]),
    "nomos_prefill": (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_decode_step": (c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_reset_kv": (c.c_int32, [c.c_void_p]),
}.items():
    f = getattr(lib, n); f.restype = r; f.argtypes = a

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
ht = lib.nomos_init(WTS.encode())
logits = np.zeros(VOCAB, np.float32)
p = bar["prompts"][0]
ids = np.array(p["ids"], dtype=np.int32)
lib.nomos_reset_kv(ht)
lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
tok = int(np.argmax(logits))
# warm a few, then timed window
for _ in range(8):
    lib.nomos_decode_step(ht, tok, logits.ctypes.data); tok = int(np.argmax(logits))
t0 = time.perf_counter()
for _ in range(NTOK):
    lib.nomos_decode_step(ht, tok, logits.ctypes.data); tok = int(np.argmax(logits))
dt = time.perf_counter() - t0
print(f"[base] {NTOK} decode steps in {dt:.3f}s -> {NTOK/dt:.2f} tok/s, {1000*dt/NTOK:.2f} ms/tok", flush=True)
