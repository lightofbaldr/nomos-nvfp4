#!/usr/bin/env python3
"""D2 G2/G3 kernel-side gate runner. For the current NOMOS_EAGLE_MODE (1=nvfp4,
2=gold/quant-bypass) this:
  1. runs the 3-step oracle-protocol draft (fixed synthetic taps, seed=1) and
     diffs ids + intermediates against eagle3/inputs/eagle3_oracle_ref.npz;
  2. runs a single-step head forward (k=1) on every harvested live input
     (INPUTS_NPZ) and saves the full [32000] draft logits per input;
  3. times the k=1 head forward (steady clock, warmup + median of N_TIME).

Env: WEIGHTS, SO_PATH, EAGLE_DIR (NOMOS_EAGLE=1 + NOMOS_EAGLE_MODE set),
     INPUTS_NPZ, LOGITS_OUT (.npz), N_TIME (default 100).
"""
import ctypes, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SO = os.environ.get("SO_PATH", os.path.join(ROOT, "libnomos_kernel.so"))
WTS = os.environ["WEIGHTS"]
EAGLE_DIR = os.environ["EAGLE_DIR"]
INPUTS = os.environ["INPUTS_NPZ"]
OUT = os.environ["LOGITS_OUT"]
N_TIME = int(os.environ.get("N_TIME", "100"))
D = 5376
DV = 32000

lib = ctypes.CDLL(SO)
lib.nomos_init.restype = ctypes.c_void_p
lib.nomos_init.argtypes = [ctypes.c_char_p]
lib.nomos_eagle3_load.restype = ctypes.c_int32
lib.nomos_eagle3_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nomos_eagle3_head_logits.restype = ctypes.c_int32
lib.nomos_eagle3_head_logits.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

ht = lib.nomos_init(WTS.encode())
assert ht
rc = lib.nomos_eagle3_load(ht, EAGLE_DIR.encode())
assert rc == 0, f"eagle3_load rc={rc} (mode={os.environ.get('NOMOS_EAGLE_MODE')})"
mode = os.environ.get("NOMOS_EAGLE_MODE", "1")
print(f"mode={mode}", flush=True)

# ── 1. oracle-protocol 3-step draft (rope/recurrence coverage) ──────────────
gold = np.load(os.path.join(ROOT, "eagle3", "inputs", "eagle3_oracle_ref.npz"))
taps_flat = np.ascontiguousarray(gold["taps_flat"], np.float32)
seed = int(gold["seed_token"])
k = int(gold["k_steps"])
did = np.zeros(k, np.int32)
tid = np.zeros(k, np.int32)
lg = np.zeros((k, DV), np.float32)
rc = lib.nomos_eagle3_head_logits(ht, taps_flat.ctypes.data, seed, k,
                                  did.ctypes.data, tid.ctypes.data, lg.ctypes.data)
assert rc == k, rc
print("oracle draft ids  kernel:", did.tolist(), " gold:", gold["draft_ids"].tolist())
print("oracle target ids kernel:", tid.tolist(), " gold:", gold["target_ids"].tolist())
for s in range(k):
    print(f"  step{s}: logits_max kernel={lg[s].max():.4f} gold={float(gold[f'step{s}_logits_max']):.4f}")

# ── 2. live-input single-step logits ────────────────────────────────────────
inp = np.load(INPUTS)
taps = inp["taps"]; toks = inp["tokens"]
n = len(toks)
out_logits = np.zeros((n, DV), np.float32)
out_ids = np.zeros(n, np.int32)
one_l = np.zeros((1, DV), np.float32)
d1 = np.zeros(1, np.int32); t1 = np.zeros(1, np.int32)
for i in range(n):
    t = np.ascontiguousarray(taps[i].reshape(-1), np.float32)
    rc = lib.nomos_eagle3_head_logits(ht, t.ctypes.data, int(toks[i]), 1,
                                      d1.ctypes.data, t1.ctypes.data, one_l.ctypes.data)
    assert rc == 1, rc
    out_logits[i] = one_l[0]
    out_ids[i] = d1[0]
np.savez(OUT, logits=out_logits, draft_ids=out_ids,
         oracle_draft_ids=did, oracle_target_ids=tid, oracle_logits=lg)
print(f"wrote {OUT}: {n} single-step head logits", flush=True)

# ── 3. timing (steady clock; k=1 full head forward incl. taps H2D + ids/logits D2H)
t0 = np.ascontiguousarray(taps[0].reshape(-1), np.float32)
for _ in range(10):   # warmup
    lib.nomos_eagle3_head_logits(ht, t0.ctypes.data, int(toks[0]), 1,
                                 d1.ctypes.data, t1.ctypes.data, one_l.ctypes.data)
lat = []
for _ in range(N_TIME):
    a = time.perf_counter_ns()
    lib.nomos_eagle3_head_logits(ht, t0.ctypes.data, int(toks[0]), 1,
                                 d1.ctypes.data, t1.ctypes.data, one_l.ctypes.data)
    lat.append(time.perf_counter_ns() - a)
lat = np.array(lat, np.float64) / 1e6
print(f"head forward k=1 (ms): median={np.median(lat):.3f} p10={np.percentile(lat,10):.3f} "
      f"p90={np.percentile(lat,90):.3f} n={N_TIME}")
# and without the logits D2H copy
for _ in range(5):
    lib.nomos_eagle3_head_logits(ht, t0.ctypes.data, int(toks[0]), 1,
                                 d1.ctypes.data, t1.ctypes.data, None)
lat2 = []
for _ in range(N_TIME):
    a = time.perf_counter_ns()
    lib.nomos_eagle3_head_logits(ht, t0.ctypes.data, int(toks[0]), 1,
                                 d1.ctypes.data, t1.ctypes.data, None)
    lat2.append(time.perf_counter_ns() - a)
lat2 = np.array(lat2, np.float64) / 1e6
print(f"head forward k=1 no-logits-copy (ms): median={np.median(lat2):.3f} "
      f"p10={np.percentile(lat2,10):.3f} p90={np.percentile(lat2,90):.3f}")
