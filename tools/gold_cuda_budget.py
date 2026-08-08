#!/usr/bin/env python3
"""CUDA-event decode budget for the exact greedy serve fast-token path.

Runs one uninstrumented warm/overhead-control pass, then enables the in-library
event recorder for one scored pass. The C shim prints the category totals at
process exit. Requires EXPECTED_SHA and enforces source/artifact/ABI provenance.
"""
import ctypes as c
import json
import os
import pathlib
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SO = pathlib.Path(os.environ.get("SO_PATH", ROOT / "libnomos_kernel.so")).resolve()
WTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/gemma-4-31b/"))
EXPECTED = os.environ["EXPECTED_SHA"]
NTOK = int(os.environ.get("PERF_NTOK", "64"))
VOCAB = 262144


def output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


head = output("git", "rev-parse", "HEAD")
assert head == EXPECTED, f"SHA mismatch: expected {EXPECTED}, got {head}"
assert "cuda_budget_token_begin" in (ROOT / "lib/engine_decode.mojo").read_text()
newest_source = max(
    p.stat().st_mtime for p in ROOT.rglob("*")
    if p.is_file() and p.suffix in {".mojo", ".c"}
)
assert SO.stat().st_mtime > newest_source, ".so is older than source under test"
symbols = output("nm", "-D", str(SO))
assert sum(" T nomos_" in line for line in symbols.splitlines()) == 85
assert WTS.endswith("/"), "WEIGHTS must have a trailing slash"
print(f"provenance PASS sha={head} so={SO} symbols=85", flush=True)

lib = c.CDLL(str(SO))
for name, result, args in [
    ("nomos_init", c.c_void_p, [c.c_char_p]),
    ("nomos_shutdown", None, [c.c_void_p]),
    ("nomos_prefill", c.c_int32, [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p]),
    ("nomos_decode_step_token", c.c_int32, [c.c_void_p, c.c_int32, c.c_void_p]),
    ("nomos_reset_kv", c.c_int32, [c.c_void_p]),
]:
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args

bar = json.loads((ROOT / "tools/gold_bar_tokens.json").read_text())
logits = np.zeros(VOCAB, np.float32)
out_token = c.c_int32()
handle = lib.nomos_init(WTS.encode())
assert handle


def run_pass():
    sequences = {}
    elapsed = 0.0
    count = 0
    for prompt in bar["prompts"]:
        ids = np.asarray(prompt["ids"], dtype=np.int32)
        assert lib.nomos_reset_kv(handle) == 0
        assert lib.nomos_prefill(handle, ids.ctypes.data, len(ids), logits.ctypes.data) == 0
        token = int(np.argmax(logits))
        seq = [token]
        start = time.perf_counter()
        for _ in range(NTOK - 1):
            assert lib.nomos_decode_step_token(handle, token, c.addressof(out_token)) == 0
            token = int(out_token.value)
            seq.append(token)
        elapsed += time.perf_counter() - start
        count += NTOK - 1
        sequences[prompt["id"]] = seq
    return sequences, count, elapsed


warm_ids, warm_n, warm_s = run_pass()
print(f"uninstrumented token path: {warm_n / warm_s:.3f} tok/s "
      f"({1000 * warm_s / warm_n:.3f} ms/token)", flush=True)

# The C recorder notices this on the next decode call, after all warm-up work.
os.environ["NOMOS_CUDA_BUDGET"] = "1"
scored_ids, scored_n, scored_s = run_pass()
assert warm_ids == scored_ids, "instrumented token IDs differ from control"
print(f"instrumented token path:   {scored_n / scored_s:.3f} tok/s "
      f"({1000 * scored_s / scored_n:.3f} ms/token)", flush=True)
print(f"instrumentation movement:  {(scored_s / scored_n) / (warm_s / warm_n) - 1:+.2%}", flush=True)
print("TOKEN-IDENTITY PASS", flush=True)
lib.nomos_shutdown(handle)
