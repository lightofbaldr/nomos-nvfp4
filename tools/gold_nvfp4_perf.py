#!/usr/bin/env python3
"""Base-decode perf (tokens/s) on the 12-prompt bar — self-contained, no server,
no HuggingFace. Built for the discrete-Blackwell (RTX PRO 4000) NVFP4 fit/perf test,
works on any box the .so was built for.

What this measures: PURE GREEDY DECODE tokens/s (prefill timed separately and
excluded from the headline rate, matching the GB10 baseline convention where
the GB10 Q4_0 base path). It does NOT run speculative decoding — the
DFlash spec stack (21.10 on GB10) needs its NVFP4 drafter port before it
travels to discrete cards.

Prompts ship pre-tokenized in gold_bar_tokens.json (bar-identical chat
template) — deps are python3 + numpy only.

Usage (from the repo root, after building libnomos_kernel.so ON THIS BOX):
  CUDA_VISIBLE_DEVICES=<free-gpu-index> \
  WEIGHTS=$HOME/nomos_data/gemma-4-31b/ \
  python3 tools/gold_nvfp4_perf.py

Env: WEIGHTS (trailing slash REQUIRED), SO_PATH (default ./libnomos_kernel.so),
PERF_NTOK (default 96), PERF_REPEAT (default 2; first pass warms, last is scored).
NVFP4 selection happens via the NOMOS_* env you export before running — see
docs/GOLD_NVFP4_PERF.md for the exact block.
"""
import ctypes, json, os, time
import numpy as np

SO = os.environ.get("SO_PATH", os.path.join(os.getcwd(), "libnomos_kernel.so"))
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
NTOK = int(os.environ.get("PERF_NTOK", "96"))
REPEAT = int(os.environ.get("PERF_REPEAT", "2"))
VOCAB = 262144
HERE = os.path.dirname(os.path.abspath(__file__))

assert WTS.endswith("/"), "WEIGHTS must end with '/' (engine concats WEIGHTS+filename)"

lib = ctypes.CDLL(SO)
for n, r, a in [
    ("nomos_init", ctypes.c_int64, [ctypes.c_int64]),
    ("nomos_prefill", ctypes.c_int32, [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]),
    ("nomos_decode_step", ctypes.c_int32, [ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]),
    ("nomos_reset_kv", ctypes.c_int32, [ctypes.c_int64]),
]:
    f = getattr(lib, n); f.restype = r; f.argtypes = a


def cstr(s):
    b = ctypes.create_string_buffer(s.encode() + b"\x00")
    return b, ctypes.cast(b, ctypes.c_void_p).value


bar = json.load(open(os.path.join(HERE, "gold_bar_tokens.json")))
_, wptr = cstr(WTS)
t0 = time.time()
ht = lib.nomos_init(wptr)
assert ht, "nomos_init failed — check WEIGHTS path/format (.nvfp4 present?)"
print(f"engine init {time.time()-t0:.0f}s | SO={os.path.basename(SO)} | WEIGHTS={WTS}", flush=True)

logits = np.zeros(VOCAB, np.float32)
results = {}
for rep in range(REPEAT):
    tag = "warm" if rep < REPEAT - 1 else "SCORED"
    for p in bar["prompts"]:
        ids = np.array(p["ids"], dtype=np.int32)
        lib.nomos_reset_kv(ht)
        tp0 = time.time()
        lib.nomos_prefill(ht, ids.ctypes.data, len(ids), logits.ctypes.data)
        tp = time.time() - tp0
        tok = int(np.argmax(logits))
        td0 = time.time()
        for _ in range(NTOK - 1):
            lib.nomos_decode_step(ht, tok, logits.ctypes.data)
            tok = int(np.argmax(logits))
        td = time.time() - td0
        dec_tps = (NTOK - 1) / td
        e2e_tps = NTOK / (tp + td)
        if rep == REPEAT - 1:
            results[p["id"]] = {"bucket": p["bucket"], "dec": dec_tps, "e2e": e2e_tps}
        print(f"[{tag}] {p['id']:>8} [{p['bucket']:>12}] prefix={len(ids):>4} "
              f"prefill={tp*1000:6.0f}ms decode={dec_tps:6.2f} tok/s (e2e {e2e_tps:5.2f})",
              flush=True)

print("\n== BASE DECODE (greedy, no spec) — per-bucket median, weighted ==")
wsum = 0.0
for b, wgt in bar["weights"].items():
    v = sorted(r["dec"] for r in results.values() if r["bucket"] == b)
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    wsum += wgt * med
    print(f"{b:>12}: median {med:6.2f} tok/s  (weight {wgt})")
print(f"WEIGHTED: {wsum:.2f} tok/s   | GB10 Q4_0 reference: 10.17 | spec-decode (GB10 only, for context): 21.10")
