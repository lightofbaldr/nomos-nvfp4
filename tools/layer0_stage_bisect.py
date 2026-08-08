#!/usr/bin/env python3
"""#71 layer-0 stage bisect: WHERE do the batched and per-token formulations first disagree?

Uses the purpose-built probe that already exists — nomos_debug_omlp_stage_ab, a
"decode-vs-batched O/MLP stage probe for one layer and row" that runs BOTH formulations and
reports max_abs_delta / cos / first_diff / diff_count. No new kernel code needed. (Found by
reading nomos_ffi.mojo before writing anything: reference, don't re-derive.)

Runs the ladder from OPERANDS upward at layer 0, so the first diverging stage is the mechanism —
this removes the 60-layer amplification confound that made the end-to-end logit delta ambiguous:

  11 raw packed INT4 K/V source   are the stored operands identical?
  10 K/V q8 views as consumed     identical after dequant/view?
   9 q8-Q-as-f32 + scale tail     is the quantized Q identical?
   8 post-RoPE Q                  is Q identical before quant?
  12 pre-softmax QK scores        does the score computation agree on identical operands?
   0 attn_out                     does softmax+PV agree?
   1 o_proj / 7 layer_out         propagation

Reads out per Codex: QK mismatch patterned by head/row -> transpose/stride/GQA/scaling; QK close
but attn_out large -> softmax or PV; all stages close -> reduction-order amplification.
"""
import ctypes as c, os, pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SO = pathlib.Path(os.environ.get("SO_PATH", ROOT / "libnomos_kernel.so")).resolve()
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
LAYER = int(os.environ.get("BISECT_LAYER", "0"))
NROWS = int(os.environ.get("BISECT_ROWS", "8"))
ROWS_TO_PROBE = [int(x) for x in os.environ.get("BISECT_ROW_LIST", "0").split(",")]

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
_so_mt = SO.stat().st_mtime
_stale = [p.name for p in (ROOT / "lib").glob("*.mojo") if p.stat().st_mtime > _so_mt]
if _stale:
    sys.exit(f"provenance FAIL: {SO.name} older than {_stale[:3]} — rebuild first")
print(f"provenance sha={head[:9]} so={SO.name} layer={LAYER} rows={NROWS}", flush=True)

lib = c.CDLL(str(SO))
lib.nomos_init.restype = c.c_void_p; lib.nomos_init.argtypes = [c.c_char_p]
lib.nomos_shutdown.restype = None; lib.nomos_shutdown.argtypes = [c.c_void_p]
f = lib.nomos_debug_omlp_stage_ab
f.restype = c.c_int32
f.argtypes = [c.c_int64, c.c_int64, c.c_int32, c.c_int32, c.c_int32,
              c.c_int32, c.c_int32, c.c_int64, c.c_int64, c.c_int64, c.c_int64]

h = lib.nomos_init(WTS.encode()); assert h, "init failed"
toks = np.arange(1000, 1000 + NROWS, dtype=np.int32)
oi = np.zeros(16, np.int32); of = np.zeros(4, np.float32)

STAGES = [(11, "raw INT4 K/V source"), (10, "K/V q8 views (as consumed)"),
          (9, "q8 Q as f32 + scale"), (8, "post-RoPE Q"),
          (12, "PRE-SOFTMAX QK scores"), (0, "attn_out (softmax+PV)"),
          (1, "o_proj"), (7, "layer_out")]

print(f"\n{'row':>4} {'stage':>5} {'what':26s} {'max|d|':>12} {'cos':>10} {'n_diff':>8}")
print("  " + "-" * 74)
for ROW in ROWS_TO_PROBE:
    if ROW >= NROWS: continue
    for st, name in STAGES:
        oi[:] = 0; of[:] = 0
        rc = f(c.c_int64(h), c.c_int64(toks.ctypes.data), c.c_int32(NROWS), c.c_int32(0),
               c.c_int32(ROW), c.c_int32(LAYER), c.c_int32(st),
               c.c_int64(oi.ctypes.data), c.c_int64(of.ctypes.data), c.c_int64(0), c.c_int64(0))
        if rc != 0:
            continue
        flag = "  <<<" if of[0] > 0 else ""
        print(f"{ROW:>4} {st:>5} {name:26s} {of[0]:>14.3e} {of[1]:>12.9f} {oi[8]:>8}{flag}")
lib.nomos_shutdown(c.c_void_p(h))
print("\n  first stage with max|d| >> 0 is the mechanism; all-clean => amplification", flush=True)
print("BISECT COMPLETE", flush=True)
