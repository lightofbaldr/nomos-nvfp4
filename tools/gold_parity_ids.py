"""Greedy-token parity dump for the R1 fp4_gemm A/B (old vs smem-staged kernel).

Runs the same 12-prompt bar as gold_nvfp4_perf.py (REPEAT=1) and writes the greedy
token ids per prompt to PARITY_OUT (json). Run once with NOMOS_FP4_OLD=1 and once
without; the two files must be identical for the e2e gate.
"""
import ctypes, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(os.path.dirname(HERE), "libnomos_kernel.so")
WTS = os.environ["WEIGHTS"]
NTOK = int(os.environ.get("PERF_NTOK", "64"))
OUT = os.environ.get("PARITY_OUT", "/tmp/parity_ids.json")
VOCAB = 262144

lib = ctypes.CDLL(SO)
lib.nomos_init.restype = ctypes.c_void_p
lib.nomos_init.argtypes = [ctypes.c_char_p]
lib.nomos_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.nomos_decode_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.nomos_reset_kv.argtypes = [ctypes.c_void_p]

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
t0 = time.time()
ht = lib.nomos_init(WTS.encode())
assert ht, "nomos_init failed"
print(f"init {time.time()-t0:.0f}s  old_kernel={os.environ.get('NOMOS_FP4_OLD','0')}", flush=True)

logits = np.zeros(VOCAB, np.float32)
out = {}
for p in bar["prompts"]:
    ids = np.array(p["ids"], dtype=np.int32)
    lib.nomos_reset_kv(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    seq = [tok]
    for _ in range(NTOK - 1):
        lib.nomos_decode_step(ht, tok, logits.ctypes.data)
        tok = int(np.argmax(logits))
        seq.append(tok)
    out[p["id"]] = seq
    print(f"{p['id']:>8}: {seq[:8]}...", flush=True)

json.dump(out, open(OUT, "w"))
print(f"wrote {OUT}", flush=True)
