#!/usr/bin/env python3
"""D2 G2 input harvester — collect ~100 LIVE (aux-taps, seed_token) pairs from
the target decode on the 12-prompt bar. After prefill and after each of the
first 9 decode steps, the stash (L[1,29,56] post-block residuals at the last
processed position) is paired with the NEXT greedy token — exactly what the
live spec loop would hand the drafter. Saved to INPUTS_OUT (.npz: taps
[N,3,5376] fp32, tokens [N] int32, prompt_ids [N], step_ids [N]).

Env: WEIGHTS, SO_PATH, EAGLE_DIR (NOMOS_EAGLE=1), INPUTS_OUT, N_MAX (default 100).
"""
import ctypes, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
OUT = os.environ["INPUTS_OUT"]
N_MAX = int(os.environ.get("N_MAX", "100"))
VOCAB = 262144
D = 5376

lib = ctypes.CDLL(SO)
lib.nomos_init.restype = ctypes.c_void_p
lib.nomos_init.argtypes = [ctypes.c_char_p]
lib.nomos_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.nomos_decode_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.nomos_reset_kv.argtypes = [ctypes.c_void_p]
lib.nomos_eagle3_load.restype = ctypes.c_int32
lib.nomos_eagle3_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nomos_eagle3_get_taps.restype = ctypes.c_int32
lib.nomos_eagle3_get_taps.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
ht = lib.nomos_init(WTS.encode())
assert ht
rc = lib.nomos_eagle3_load(ht, EAGLE_DIR.encode())
assert rc == 0, f"eagle3_load rc={rc}"

logits = np.zeros(VOCAB, np.float32)
tap = np.zeros(3 * D, np.float32)
taps, tokens, pids, steps = [], [], [], []
for p in bar["prompts"]:
    ids = np.array(p["ids"], dtype=np.int32)
    lib.nomos_reset_kv(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    lib.nomos_eagle3_get_taps(ht, tap.ctypes.data)
    taps.append(tap.reshape(3, D).copy()); tokens.append(tok)
    pids.append(p["id"]); steps.append(-1)
    for s in range(9):
        lib.nomos_decode_step(ht, tok, logits.ctypes.data)
        tok = int(np.argmax(logits))
        lib.nomos_eagle3_get_taps(ht, tap.ctypes.data)
        taps.append(tap.reshape(3, D).copy()); tokens.append(tok)
        pids.append(p["id"]); steps.append(s)
    if len(taps) >= N_MAX + 20:
        break

n = min(N_MAX, len(taps))
np.savez(OUT,
         taps=np.stack(taps[:n]).astype(np.float32),
         tokens=np.array(tokens[:n], np.int32),
         prompt_ids=np.array(pids[:n]),
         step_ids=np.array(steps[:n], np.int32))
print(f"wrote {OUT}: {n} live (taps, token) pairs")
