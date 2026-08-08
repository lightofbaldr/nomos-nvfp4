#!/usr/bin/env python3
"""FP4 parity harness — the gate for the FP4-all-the-way conversion (#421).

KERNEL-ONLY (no HF model load): captures a precision baseline straight from the
Nomos kernel .so so it can be diffed against itself after each FP4 step. This is
the §F.2 / §I.2 parity instrument, adapted to be
self-referential (kernel-vs-kernel across steps) rather than kernel-vs-HF.

Two acceptance signals:
  A. DETERMINISM (§I.2.a): two passes in one process + (via --compare) two
     process invocations must be byte-identical — greedy continuations exact,
     last-position top-32 logit ids exact, logit values exact.
  B. KV-CACHE PREFIX ROUND-TRIP (§I.2.b, the ACTIVE test of the kvb stride fix at
     bf16): greedy continuation must be deterministic.
     At NOMOS_PRECISION_BITS=16 the cache is bf16 (kvb=2); the old `*4` hardcode
     mis-strode it and corrupted the prefix. Identical round-trip == fix correct.

Usage:
    # write the bf16 baseline
    NOMOS_PRECISION_BITS=16 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        python3 fp4_parity_harness.py --out baseline_bf16.json

    # later: after an FP4 step, diff against the saved baseline
    NOMOS_PRECISION_BITS=... python3 fp4_parity_harness.py --compare baseline_bf16.json

Env:
    SO_PATH   $REPO/libnomos_kernel.so
    WEIGHTS   /path/to/gemma-4-31b-weights/

    TOK_DIR   /path/to/gemma-4-31b-it   (HF tokenizer only, no model)
    MAXNEW    40
    NOMOS_PRECISION_BITS  16  (bf16 cache -> kvb=2, the active stride test)
"""
import os, sys, json, ctypes, argparse, hashlib, time
import numpy as np

SO_PATH = os.environ.get("SO_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libnomos_kernel.so"))
WEIGHTS = os.environ.get("WEIGHTS", "")

# The kernel loader concatenates weights_dir + filename with NO separator, so the
# dir MUST end with a slash (else every "<dir>layers_N_..." file is "missing" ->
# empty layer list -> out-of-bounds on the first forward pass). Normalize both.
if not WEIGHTS.endswith("/"):
    WEIGHTS += "/"


TOK_DIR = os.environ.get("TOK_DIR", "")
MAXNEW  = int(os.environ.get("MAXNEW", "40"))
PREC    = int(os.environ.get("NOMOS_PRECISION_BITS", "16"))
D          = 5376
PROBE_TOPK = 32
TOTAL_LAYERS = 60
NKV_MAX    = 16
FULL_HD    = 512
KVB        = 2 if PREC <= 16 else 4   # KV bytes/elem

PROMPTS = [
    "Write a Python function to compute the nth Fibonacci number.",
    "List three commands to check disk usage on Linux, one per line.",
    "What is the capital of France? Answer in one word.",
    "Explain how a hash map works, step by step.",
    "Write a bash one-liner that finds the 5 largest files under /var/log.",
    "Count from 1 to 10, then say done.",
    "Give me a JSON object with keys name, age, and city for a fictional person.",
    "Summarize why the sky is blue in exactly two sentences.",
]

# Short, distinctive prompt for the KV round-trip (a would-be memory atom).
KV_PROMPT = "The Reykjavik launch code is alpha-seven-tango-niner. Remember it exactly."


def addr(a):
    return a.ctypes.data


def load_lib():
    lib = ctypes.CDLL(SO_PATH)
    lib.nomos_init.restype = ctypes.c_int64
    lib.nomos_init.argtypes = [ctypes.c_int64]
    lib.nomos_generate.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
                                   ctypes.c_int32, ctypes.c_int64, ctypes.c_int32,
                                   ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.nomos_generate.restype = ctypes.c_int32
    lib.nomos_reset_kv.argtypes = [ctypes.c_int64]
    lib.nomos_kv_cache_len.argtypes = [ctypes.c_int64]
    lib.nomos_kv_cache_len.restype = ctypes.c_int32
    # Optional symbols — parity-harness probes + KV-memory ops. Clean-room contract
    # kernels (competition seats) strip these; guard so load_lib doubles as a serving
    # loader for ANY kernel (the serving path never calls the probe surface). hasattr on
    # a ctypes CDLL returns False for an undefined symbol. occ4 still binds them all.
    if hasattr(lib, "nomos_set_probe_logits"):
        lib.nomos_set_probe_logits.argtypes = [ctypes.c_int64, ctypes.c_int32]
        lib.nomos_probe_l20.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
                                        ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
        lib.nomos_get_probe_logits.argtypes = [ctypes.c_int64, ctypes.c_int64,
                                               ctypes.c_int64, ctypes.c_int32]




    return lib


def init_engine(lib):
    wd = ctypes.create_string_buffer(WEIGHTS.encode())

    print(f"  nomos_init(prec={PREC}, kvb={KVB}) ...", flush=True)
    h = lib.nomos_init(ctypes.addressof(wd))
    print(f"  handle={h}", flush=True)
    if h == 0:
        sys.exit("nomos_init failed")
    return h


def run_prompts(lib, h, tok):
    """One pass over the fixed prompts: prefill top-32 logits @ last position +
    greedy continuation. Mirrors the proven parity_kernel_vs_hf order (probe then
    generate, no reset between — generate re-prefills fresh; kv_continue stays
    False the whole loop)."""
    out = []
    gen_tok = 0
    gen_sec = 0.0
    for p in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True,
                                      return_tensors="np", return_dict=True)
        ids = np.asarray(enc["input_ids"][0], dtype=np.int32)
        n = len(ids)
        # TEST 1: prefill, capture last-position top-32 logits
        lib.nomos_set_probe_logits(h, 1)
        cloud = np.zeros(n * D, dtype=np.float32)
        n_tok = np.zeros(1, dtype=np.int32)
        rc = lib.nomos_probe_l20(h, addr(ids), n, addr(cloud), n * D, addr(n_tok))
        cap = n * PROBE_TOPK
        kids = np.zeros(cap, dtype=np.int32)
        kvals = np.zeros(cap, dtype=np.float32)
        n_pos = lib.nomos_get_probe_logits(h, addr(kids), addr(kvals), cap)
        if n_pos > 0:
            last = (n_pos - 1) * PROBE_TOPK
            top_ids = kids[last:last + PROBE_TOPK].tolist()
            top_vals = [round(float(v), 6) for v in kvals[last:last + PROBE_TOPK]]
            argmax = int(top_ids[0])
        else:
            top_ids, top_vals, argmax = [], [], -1
        # TEST 2: greedy continuation (timed — pure decode tok/s, prefill incl.)
        outb = np.zeros(MAXNEW, dtype=np.int32)
        _t0 = time.perf_counter()
        nout = lib.nomos_generate(h, addr(ids), n, MAXNEW, addr(outb), MAXNEW,
                                  0.0, -1.0, 1.0)
        gen_sec += time.perf_counter() - _t0
        gen_tok += max(int(nout), 0)
        greedy = outb[:max(int(nout), 0)].tolist()
        out.append({
            "prompt": p,
            "ids": ids.tolist(),
            "probe_rc": int(rc),
            "n_pos": int(n_pos),
            "argmax": argmax,
            "top_ids": top_ids,
            "top_vals": top_vals,
            "greedy": greedy,
            "greedy_txt": tok.decode(greedy, skip_special_tokens=False).replace("\n", "\\n"),
        })
        print(f"  [{p[:30]:30}] gen={len(greedy):3} :: "
              f"{tok.decode(greedy, skip_special_tokens=False).replace(chr(10), ' ')[:62]!r}",
              flush=True)
    if gen_sec > 0:
        print(f"  [TIMING] {gen_tok} gen tok over {len(PROMPTS)} prompts in "
              f"{gen_sec:.2f}s = {gen_tok / gen_sec:.2f} tok/s (prefill+decode)",
              flush=True)
    return out


def diff_passes(a, b):
    """Byte-identity check between two prompt-pass result lists."""
    if len(a) != len(b):
        return False, f"len {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        for key in ("ids", "argmax", "top_ids", "top_vals", "greedy"):
            if x.get(key) != y.get(key):
                return False, f"prompt {i} field {key} differs"
    return True, "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write baseline json here")
    ap.add_argument("--compare", help="diff against a saved baseline json")
    args = ap.parse_args()

    print("=== FP4 PARITY HARNESS (kernel-only) ===", flush=True)
    print(f"  SO={SO_PATH}\n  WEIGHTS={WEIGHTS}\n  prec={PREC} kvb={KVB} MAXNEW={MAXNEW}",
          flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)

    lib = load_lib()
    h = init_engine(lib)

    print("\n--- PASS 1 (prompts) ---", flush=True)
    pass1 = run_prompts(lib, h, tok)
    print("\n--- PASS 2 (determinism, in-process) ---", flush=True)
    pass2 = run_prompts(lib, h, tok)
    det_ok, det_msg = diff_passes(pass1, pass2)
    print(f"\n  DETERMINISM (in-process 2-pass): {'PASS' if det_ok else 'FAIL'} ({det_msg})",
          flush=True)

    print(f"  {json.dumps(kv)}", flush=True)
    kv_ok = kv.get("ok") and kv.get("identical")
    print(f"  KV-CACHE INT4 parity: {'PASS' if kv_ok else 'FAIL'} "
          f"(byte-identical at kvb={KVB})", flush=True)

    result = {
        "meta": {"prec": PREC, "kvb": KVB, "maxnew": MAXNEW, "so": SO_PATH,
                 "weights": WEIGHTS},
        "prompts": pass1,
        "determinism_inproc": {"ok": det_ok, "msg": det_msg},
    }

    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        cmp_ok, cmp_msg = diff_passes(ref["prompts"], pass1)
        print(f"\n=== COMPARE vs {args.compare} ===", flush=True)
        print(f"  cross-run prompt parity: {'PASS' if cmp_ok else 'FAIL'} ({cmp_msg})",
              flush=True)
        overall = cmp_ok and det_ok and kv_ok
        print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}", flush=True)
        sys.exit(0 if overall else 1)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\n  wrote baseline -> {args.out}", flush=True)

    overall = det_ok and kv_ok
    print(f"\n=== STEP-0 BASELINE: {'PASS' if overall else 'FAIL'} ===", flush=True)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
