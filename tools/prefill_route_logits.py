#!/usr/bin/env python3
"""Compare the two prefill routes at the LOGIT level — the decisive test for #71.

WHY. Square-batched prefill (materialized [NH,S,S], engine_prefill.mojo:145) and per-token prefill
produce different GREEDY TOKENS on 9 of 12 standard-bar prompts at ctx 54-216 — far below the 1024
sliding window, so this is not a window-mask defect. Two hypotheses remain, and argmax alone cannot
separate them:

  A. BENIGN NUMERICS. The routes accumulate in different orders (one big GEMM vs a per-token loop);
     with INT4 KV the logits carry real quantization noise, so a near-tie at argmax flips and the
     flip cascades through a greedy chain. Prediction: logit vectors nearly identical (cos ~1,
     max|delta| small relative to the logit scale), and divergence occurs ONLY where the top1-top2
     margin is comparable to max|delta|.

  B. ONE ROUTE IS WRONG. Prediction: logit vectors differ materially, and flips happen even where
     the top1-top2 margin is LARGE compared with the observed delta.

The discriminator is `margin vs delta`, not the token. A flip with margin >> delta cannot be
rounding — that is hypothesis B and means the shipped discrete default may be producing wrong
output on every long prompt while staying perfectly coherent.

This needs NO external reference and NO bf16 weights: it is an internal consistency measurement
that tells you WHICH QUESTION to spend the bf16 anchor on. (It still cannot say which route is
CORRECT — only a ground-truth reference can. See task #71.)

  EXPECTED_SHA=... SO_PATH=... WEIGHTS=... NOMOS_BATCHED_PREFILL=1 ROUTE_TAG=sq \\
  python3 tools/prefill_route_logits.py          # run once per route, then --compare
"""
import ctypes as c, json, os, pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("ROUTE_OUT", "~/out"))

if "--compare" in sys.argv:
    sq = np.load(OUT / "logits_sq.npy")
    pt = np.load(OUT / "logits_pt.npy")
    ids = json.loads((OUT / "logits_ids.json").read_text())
    print(f"{'prompt':14s} {'ctx':>5s} {'max|d|':>10s} {'cos':>10s} {'margin_sq':>10s} "
          f"{'margin/d':>9s} {'top1_sq':>8s} {'top1_pt':>8s}  verdict")
    for i, meta in enumerate(ids):
        a, b = sq[i].astype(np.float64), pt[i].astype(np.float64)
        d = float(np.max(np.abs(a - b)))
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        o = np.argsort(a)[::-1]
        margin = float(a[o[0]] - a[o[1]])          # top1-top2 gap on the square route
        t_sq, t_pt = int(np.argmax(a)), int(np.argmax(b))
        ratio = margin / d if d > 0 else float("inf")
        # A flip needs the perturbation to exceed roughly half the margin. margin/d >> 2 with a
        # flip is NOT explainable by this much numerical difference.
        if t_sq == t_pt:
            v = "same-token"
        elif ratio <= 2.0:
            v = "FLIP (tie, explainable by delta)"
        else:
            v = "*** FLIP WITH LARGE MARGIN — NOT ROUNDING ***"
        print(f"{meta['id']:14s} {meta['ctx']:5d} {d:10.4f} {cos:10.6f} {margin:10.4f} "
              f"{ratio:9.2f} {t_sq:8d} {t_pt:8d}  {v}")
    sys.exit(0)

SO = pathlib.Path(os.environ.get("SO_PATH", ROOT / "libnomos_kernel.so")).resolve()
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
VOCAB = int(os.environ.get("NOMOS_VOCAB", "262144"))
TAG = os.environ.get("ROUTE_TAG", "sq")
BAR = os.environ.get("LONGBAR", str(ROOT / "tools/gold_bar_tokens.json"))

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
exp = os.environ.get("EXPECTED_SHA")
if exp and exp != head:
    sys.exit(f"provenance FAIL: {head} != {exp}")
# EXPECTED_SHA=$(git rev-parse HEAD) compares HEAD to itself and proves NOTHING. The real question
# is whether the .so contains the source in the tree: assert it is newer than every Mojo file.
# (My own standard after the 2026-08-01 "byte-exact 18.84 tok/s" run that compared unchanged code
# to itself; I failed to apply it here until it was pointed out by re-reading my own rule.)
_so_mt = SO.stat().st_mtime
_stale = [p.name for p in (ROOT / "lib").glob("*.mojo") if p.stat().st_mtime > _so_mt]
if _stale:
    sys.exit(f"provenance FAIL: {SO.name} is OLDER than {len(_stale)} source file(s): "
             f"{', '.join(sorted(_stale)[:4])} — rebuild before measuring")
print(f"provenance PASS sha={head} route={TAG} "
      f"BATCHED_PREFILL={os.environ.get('NOMOS_BATCHED_PREFILL','?')}", flush=True)

lib = c.CDLL(str(SO))
for name, res, args in [
    ("nomos_init", c.c_void_p, [c.c_char_p]),
    ("nomos_shutdown", None, [c.c_void_p]),
    ("nomos_prefill", c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    ("nomos_reset_kv", c.c_int32, [c.c_void_p]),
]:
    fn = getattr(lib, name); fn.restype = res; fn.argtypes = args

bar = json.loads(pathlib.Path(BAR).read_text())
prompts = sorted(bar["prompts"], key=lambda p: len(p["ids"]))
handle = lib.nomos_init(WTS.encode())
assert handle
logits = np.zeros(VOCAB, np.float32)
rows, meta = [], []
for p in prompts:
    ids = np.asarray(p["ids"], dtype=np.int32)
    assert lib.nomos_reset_kv(handle) == 0
    if lib.nomos_prefill(handle, ids.ctypes.data, len(ids), logits.ctypes.data) != 0:
        print(f"  prefill failed for {p.get('id')}"); continue
    rows.append(logits.copy())
    meta.append({"id": p.get("id"), "ctx": len(ids)})
    print(f"  {p.get('id'):14s} ctx={len(ids):5d} top1={int(np.argmax(logits)):7d}", flush=True)
lib.nomos_shutdown(handle)
np.save(OUT / f"logits_{TAG}.npy", np.stack(rows))
(OUT / "logits_ids.json").write_text(json.dumps(meta))
print(f"ROUTE {TAG} COMPLETE  {len(rows)} prompts -> {OUT}/logits_{TAG}.npy", flush=True)
