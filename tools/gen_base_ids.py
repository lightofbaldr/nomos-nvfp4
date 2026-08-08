#!/usr/bin/env python3
"""Dump greedy base-decode token ids for the 12-prompt gold bar -> BASE_IDS json.
Base decode (prefill + nomos_decode_step, M=1) is the reference the spec loop's L1
losslessness gate compares against. The M=1 decode path is untouched by the M>1
verify fix, so this reference is identical whether produced by the base or new .so.
Env: WEIGHTS, EAGLE_DIR, SO_PATH, OUT (default tools/base_ids.json), PERF_NTOK (96)."""
import ctypes, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
NTOK = int(os.environ.get("PERF_NTOK", "96"))
OUT = os.environ.get("OUT", os.path.join(HERE, "base_ids.json"))
VOCAB = 262144
assert WTS.endswith("/") and EAGLE_DIR.endswith("/")

lib = ctypes.CDLL(SO)
c = ctypes
sigs = {
    "nomos_init": (c.c_void_p, [c.c_char_p]),
    "nomos_prefill": (c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_decode_step": (c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    "nomos_reset_kv": (c.c_int32, [c.c_void_p]),
    "nomos_eagle3_load": (c.c_int32, [c.c_void_p, c.c_char_p]),
    "nomos_eagle3_reset": (c.c_int32, [c.c_void_p]),
}
for n, (r, a) in sigs.items():
    f = getattr(lib, n); f.restype = r; f.argtypes = a

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
ht = lib.nomos_init(WTS.encode())
assert ht, "init failed"
assert lib.nomos_eagle3_load(ht, EAGLE_DIR.encode()) == 0, "eagle load failed"
logits = np.zeros(VOCAB, np.float32)

out = {}
for p in bar["prompts"]:
    ids = np.array(p["ids"], dtype=np.int32)
    lib.nomos_reset_kv(ht); lib.nomos_eagle3_reset(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    seq = [tok]
    for _ in range(NTOK - 1):
        lib.nomos_decode_step(ht, tok, logits.ctypes.data)
        tok = int(np.argmax(logits))
        seq.append(tok)
    out[p["id"]] = seq
    print(f"base {p['id']:>8} [{p['bucket']:>12}] {len(seq)} toks", flush=True)

json.dump(out, open(OUT, "w"))
print("wrote", OUT, flush=True)
