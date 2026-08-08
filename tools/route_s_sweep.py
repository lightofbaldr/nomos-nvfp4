#!/usr/bin/env python3
"""WHERE IN S does the batched-vs-per-token disagreement begin? (#71)

The layer-0 stage probe (nomos_debug_omlp_stage_ab) is capped at MAX_VERIFY_ROWS=17 and reported
EXACT agreement (cos=1.0) at S=8 on layers 0 and 1 — but the end-to-end divergence was measured at
ctx 54+. Either the disagreement is S-dependent, or the probe misses it. This settles that with the
instrument that demonstrably works: full prefill, final-position logits, both routes.

KEY: NOMOS_BATCHED_PREFILL is read PER PREFILL CALL (engine_prefill.mojo:118 `_env_float`), not at
init — so both routes run in ONE process against ONE loaded model. Same weights, same allocator
state, same everything; only the route varies. That removes every between-process confound the
earlier two-process comparison carried.

Reads out:
  sharp onset at a specific S  -> a tiling/blocking boundary in the square kernel (a real bug)
  smooth growth from S=2       -> reduction-order amplification down the residual stream
  no divergence at any S       -> the earlier result came from something other than the route flag
"""
import ctypes as c, json, os, pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SO = pathlib.Path(os.environ.get("SO_PATH", ROOT / "libnomos_kernel.so")).resolve()
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
VOCAB = int(os.environ.get("NOMOS_VOCAB", "262144"))
BAR = os.environ.get("LONGBAR", str(ROOT / "tools/gold_bar_tokens.json"))

_so_mt = SO.stat().st_mtime
_stale = [p.name for p in (ROOT / "lib").glob("*.mojo") if p.stat().st_mtime > _so_mt]
if _stale: sys.exit(f"provenance FAIL: .so older than {_stale[:3]}")
head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
print(f"provenance sha={head[:9]}  (route toggled IN-PROCESS per prefill)", flush=True)

lib = c.CDLL(str(SO))
lib.nomos_init.restype = c.c_void_p; lib.nomos_init.argtypes=[c.c_char_p]
lib.nomos_shutdown.restype=None; lib.nomos_shutdown.argtypes=[c.c_void_p]
lib.nomos_prefill.restype=c.c_int32; lib.nomos_prefill.argtypes=[c.c_void_p,c.c_void_p,c.c_int32,c.c_void_p]
lib.nomos_reset_kv.restype=c.c_int32; lib.nomos_reset_kv.argtypes=[c.c_void_p]

bar = json.loads(pathlib.Path(BAR).read_text())
src = max(bar["prompts"], key=lambda p: len(p["ids"]))["ids"]   # longest, so we can truncate freely
h = lib.nomos_init(WTS.encode()); assert h
lg = np.zeros(VOCAB, np.float32)

def run(ids, batched):
    os.environ["NOMOS_BATCHED_PREFILL"] = "1" if batched else "0"
    a = np.asarray(ids, dtype=np.int32)
    assert lib.nomos_reset_kv(h) == 0
    assert lib.nomos_prefill(h, a.ctypes.data, len(a), lg.ctypes.data) == 0
    return lg.copy().astype(np.float64)

SS = [int(x) for x in os.environ.get("S_LIST", "2,4,8,12,16,17,18,24,32,48,54,64").split(",")]
SS = [s for s in SS if s <= len(src)]
print(f"\n{'S':>5} {'max|d|':>12} {'cos':>10} {'top1_sq':>9} {'top1_pt':>9}  {'verdict':>9}")
print("  " + "-"*62)
prev = None
for S in SS:
    ids = src[:S]
    sq, pt = run(ids, True), run(ids, False)
    d = float(np.max(np.abs(sq-pt)))
    cos = float(sq@pt/(np.linalg.norm(sq)*np.linalg.norm(pt)))
    t1, t2 = int(np.argmax(sq)), int(np.argmax(pt))
    v = "AGREE" if d == 0.0 else ("tiny" if d < 1e-3 else "DIVERGE")
    print(f"{S:>5} {d:12.6f} {cos:10.6f} {t1:>9} {t2:>9}  {v:>9}", flush=True)
# self-control at the largest S: same route twice, in-process
big = SS[-1]
a1, a2 = run(src[:big], True), run(src[:big], True)
print(f"\n  self-control (square twice, in-process, S={big}): max|d| = {float(np.max(np.abs(a1-a2))):.6f}")
lib.nomos_shutdown(c.c_void_p(h))
print("S SWEEP COMPLETE", flush=True)
