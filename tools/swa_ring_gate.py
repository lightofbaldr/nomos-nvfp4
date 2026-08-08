#!/usr/bin/env python3
"""Pin/compare greedy outputs for the SWA KV-ring memory optimization.

The pre-ring oracle is today's linear-cache NOMOS_KV_SWA=1 path. The prompt is
deterministically truncated to each requested token length, covering no wrap,
one wrap, and multiple wraps without checking a large input vector into git.
"""

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import subprocess

import numpy as np
from transformers import AutoTokenizer


REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LENGTHS = (512, 1100, 3000)
SEED_TEXT = "The quick brown fox jumps over the lazy dog. "


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def load_library():
    lib = ctypes.CDLL(os.environ.get("SO_PATH", str(REPO / "libnomos_kernel.so")))
    lib.nomos_init.restype = ctypes.c_int64
    lib.nomos_init.argtypes = [ctypes.c_int64]
    lib.nomos_generate.restype = ctypes.c_int32
    lib.nomos_generate.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    return lib


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path)
    mode.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--expect-sha")
    args = parser.parse_args()

    current_sha = git_sha()
    if args.expect_sha and current_sha != args.expect_sha:
        raise SystemExit(f"expected git SHA {args.expect_sha}, got {current_sha}")
    so_path = pathlib.Path(os.environ.get("SO_PATH", str(REPO / "libnomos_kernel.so")))
    newest_source = max(
        (REPO / "lib/engine_prefill.mojo").stat().st_mtime,
        (REPO / "lib/gemma4_engine.mojo").stat().st_mtime,
    )
    if not so_path.exists() or so_path.stat().st_mtime <= newest_source:
        raise SystemExit("libnomos_kernel.so is missing or older than the SWA sources")
    if os.environ.get("NOMOS_KV_SWA") != "1":
        raise SystemExit("gate requires NOMOS_KV_SWA=1")
    weights = os.environ["WEIGHTS"]
    if not weights.endswith("/"):
        weights += "/"
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TOK_DIR"])
    enough_text = SEED_TEXT * (max(args.lengths) // 8 + 32)
    all_ids = tokenizer.encode(enough_text, add_special_tokens=True)

    lib = load_library()
    weight_buf = ctypes.create_string_buffer(weights.encode())
    handle = lib.nomos_init(ctypes.addressof(weight_buf))
    if not handle:
        raise SystemExit("nomos_init failed")

    cases = []
    for length in args.lengths:
        ids = np.asarray(all_ids[:length], dtype=np.int32)
        if len(ids) != length:
            raise SystemExit(f"requested {length} tokens, generated only {len(ids)}")
        output = np.zeros(args.max_new, dtype=np.int32)
        nout = lib.nomos_generate(
            handle,
            ids.ctypes.data,
            len(ids),
            args.max_new,
            output.ctypes.data,
            len(output),
            0.0,
            -1.0,
            1.0,
        )
        case = {
            "length": length,
            "input_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "nout": int(nout),
            "tokens": output[: max(0, nout)].tolist(),
        }
        cases.append(case)
        print(f"S={length}: {case['tokens']}", flush=True)

    result = {
        "meta": {
            "oracle_git_sha": current_sha,
            "kv_swa": os.environ["NOMOS_KV_SWA"],
            "max_seq": os.environ.get("NOMOS_MAX_SEQ"),
            "max_new": args.max_new,
            "seed_text": SEED_TEXT,
        },
        "cases": cases,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")
        return

    expected = json.loads(args.compare.read_text())
    expected_cases = [
        case for case in expected["cases"] if case["length"] in args.lengths
    ]
    if expected_cases != result["cases"]:
        raise SystemExit("SWA ring parity FAIL: generated cases differ from oracle")
    print("SWA ring parity PASS: all cases byte-exact")


if __name__ == "__main__":
    main()
