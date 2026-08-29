#!/usr/bin/env python3
"""D2 G1 parity harness — greedy-token ids on the 12-prompt bar with the EAGLE-3
drafter OFF vs ON-but-unused (aux-tap stash armed, drafts never consumed).

Runs the same greedy loop as gold_parity_ids.py. If EAGLE_DIR is set (and
NOMOS_EAGLE=1), nomos_eagle3_load is called after init so the L[1,29,56] tap
stash fires on every decode step; the drafter itself is never invoked. The two
runs' PARITY_OUT files must be byte-identical (and equal to the clean-baseline
ids) for the G1 gate.

Optionally (TAPS_OUT set): after each decode step of the FIRST prompt, dump the
[3,5376] stash via nomos_eagle3_get_taps into an .npz for the stash-content gate.

Env: WEIGHTS (trailing slash), SO_PATH, PERF_NTOK (default 64), PARITY_OUT,
     EAGLE_DIR (trailing slash; enables the ON run), TAPS_OUT (optional .npz).
"""
import ctypes, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get("SO_PATH", os.path.join(os.path.dirname(HERE), "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
NTOK = int(os.environ.get("PERF_NTOK", "64"))
OUT = os.environ.get("PARITY_OUT", "/tmp/parity_ids.json")
EAGLE_DIR = os.environ.get("EAGLE_DIR", "")
TAPS_OUT = os.environ.get("TAPS_OUT", "")
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
t0 = time.time()
ht = lib.nomos_init(WTS.encode())
assert ht, "nomos_init failed"
eagle_on = False
if EAGLE_DIR:
    rc = lib.nomos_eagle3_load(ht, EAGLE_DIR.encode())
    assert rc == 0, f"nomos_eagle3_load rc={rc} (NOMOS_EAGLE=1 set? blobs present?)"
    eagle_on = True
print(f"init {time.time()-t0:.0f}s  eagle_armed={eagle_on} mode={os.environ.get('NOMOS_EAGLE_MODE','1(nvfp4)')}", flush=True)

logits = np.zeros(VOCAB, np.float32)
tap_buf = np.zeros(3 * D, np.float32)
out = {}
taps_dump = {}
for pi, p in enumerate(bar["prompts"]):
    ids = np.array(p["ids"], dtype=np.int32)
    lib.nomos_reset_kv(ht)
    lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
    tok = int(np.argmax(logits))
    seq = [tok]
    if eagle_on and TAPS_OUT and pi == 0:
        lib.nomos_eagle3_get_taps(ht, tap_buf.ctypes.data)
        taps_dump["step_prefill"] = tap_buf.reshape(3, D).copy()
    for s in range(NTOK - 1):
        lib.nomos_decode_step(ht, tok, logits.ctypes.data)
        tok = int(np.argmax(logits))
        seq.append(tok)
        if eagle_on and TAPS_OUT and pi == 0 and s < 12:
            lib.nomos_eagle3_get_taps(ht, tap_buf.ctypes.data)
            taps_dump[f"step_{s}"] = tap_buf.reshape(3, D).copy()
    out[p["id"]] = seq
    print(f"{p['id']:>8}: {seq[:6]}...", flush=True)

json.dump(out, open(OUT, "w"))
print("wrote", OUT, flush=True)
if taps_dump:
    np.savez(TAPS_OUT, **taps_dump)
    print("wrote", TAPS_OUT, flush=True)
