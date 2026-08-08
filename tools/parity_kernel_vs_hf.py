#!/usr/bin/env python3
"""DECISIVE forward-pass parity: Nomos kernel vs stock HF Gemma-4, identical token IDs.

Two tests, same prompts, greedy:
  TEST 1 (forward pass): kernel prefill top-32 logits at the LAST position vs HF logits[-1] top-32.
     - softcap is monotonic so top-token IDENTITIES are comparable even if values differ.
     - MATCH  -> prefill forward pass is correct end-to-end (norm/rope/mlp/attn/head all OK).
     - DIVERGE-> forward-pass bug; the audit workflow's ranked suspects say where.
  TEST 2 (full pipeline): kernel greedy continuation vs HF greedy continuation.
     - MATCH (T1 also matches) -> kernel is correct; the degradation was elsewhere/already-gone.
     - DIVERGE while T1 matches -> bug is in DECODE path or sampling bandaids (not prefill forward).

Run order: HF first (capture reference, free GPU), then ctypes-load the kernel .so.
"""
import os, sys, json, ctypes
import numpy as np

HF_DIR   = os.environ.get("HF_DIR",   "/workspace/hf_ref")
SO_PATH  = os.environ.get("SO_PATH",  "/workspace/nomos_kernel_src/libnomos_kernel.so")
WEIGHTS  = os.environ.get("WEIGHTS",  "/workspace/weights")

MAXNEW   = int(os.environ.get("MAXNEW", "40"))
D        = 5376
PROBE_TOPK = 32

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


def loop_flag(ids):
    """Crude degeneracy detector: any token repeated >=6x in a row, or any
    3-gram repeated >=4x. Returns a short string or '' if clean."""
    for i in range(len(ids) - 5):
        if len(set(ids[i:i + 6])) == 1:
            return f"SAME-TOKEN x6 @{i}"
    seen = {}
    for i in range(len(ids) - 2):
        g = tuple(ids[i:i + 3])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= 4:
            return f"3-GRAM x4 {g}"
    return ""

# ───────────────────────── PART 1: HF reference ─────────────────────────
print("=== PART 1: HF reference (tokenize + logits[-1] top32 + greedy) ===", flush=True)
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
tok = AutoTokenizer.from_pretrained(HF_DIR)
hf = Gemma4ForConditionalGeneration.from_pretrained(
    HF_DIR, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()

ref = []
for p in PROMPTS:
    enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"]
    with torch.no_grad():
        logits = hf(input_ids=ids.to("cuda")).logits[0, -1].float().cpu()
        gen = hf.generate(input_ids=ids.to("cuda"),
                          attention_mask=enc.get("attention_mask", torch.ones_like(ids)).to("cuda"),
                          max_new_tokens=MAXNEW, do_sample=False, repetition_penalty=1.0,
                          pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else 0))
    top = torch.topk(logits, PROBE_TOPK)
    ref.append({
        "ids": ids[0].tolist(),
        "hf_top_ids": top.indices.tolist(),
        "hf_top_vals": top.values.tolist(),
        "hf_argmax": int(logits.argmax()),
        "hf_greedy": gen[0][ids.shape[1]:].tolist(),
        "hf_greedy_txt": tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=False).replace("\n", "\\n"),
    })
    print(f"  [{p[:42]}] n_ids={len(ref[-1]['ids'])} hf_argmax={ref[-1]['hf_argmax']} "
          f"({tok.decode([ref[-1]['hf_argmax']])!r})", flush=True)

del hf
torch.cuda.empty_cache()
print("  HF freed.\n", flush=True)

# ───────────────────────── PART 2: kernel via ctypes ─────────────────────────
print("=== PART 2: kernel (ctypes) ===", flush=True)
lib = ctypes.CDLL(SO_PATH)
lib.nomos_init.restype = ctypes.c_int64
lib.nomos_init.argtypes = [ctypes.c_int64]
lib.nomos_set_probe_logits.argtypes = [ctypes.c_int64, ctypes.c_int32]
lib.nomos_probe_l20.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
                                ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
lib.nomos_get_probe_logits.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]
lib.nomos_generate.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
                               ctypes.c_int64, ctypes.c_int32,
                               ctypes.c_float, ctypes.c_float, ctypes.c_float]

def addr(a): return a.ctypes.data

wd = ctypes.create_string_buffer(WEIGHTS.encode())

print("  nomos_init ...", flush=True)
h = lib.nomos_init(ctypes.addressof(wd))
print(f"  handle={h}", flush=True)
if h == 0:
    sys.exit("nomos_init failed")

for r in ref:
    ids = np.array(r["ids"], dtype=np.int32)
    n = len(ids)
    # --- TEST 1: prefill top-32 logits at last position ---
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
        k_top_ids = kids[last:last + PROBE_TOPK].tolist()
        k_argmax = int(k_top_ids[0])
        overlap = len(set(k_top_ids) & set(r["hf_top_ids"]))
        argmax_match = (k_argmax == r["hf_argmax"])
    else:
        k_top_ids, k_argmax, overlap, argmax_match = [], -1, -1, False
    # --- TEST 2: greedy continuation (rep_penalty=1.0; greedy floor may clamp to 1.15) ---
    outb = np.zeros(MAXNEW, dtype=np.int32)
    nout = lib.nomos_generate(h, addr(ids), n, MAXNEW, addr(outb), MAXNEW, 0.0, -1.0, 1.0)
    k_greedy = outb[:max(nout, 0)].tolist()
    # match length of common prefix with HF greedy
    common = 0
    for a, b in zip(k_greedy, r["hf_greedy"]):
        if a == b: common += 1
        else: break

    print("\n" + "-" * 76, flush=True)
    print(f"PROMPT: {tok.decode(r['ids'][-12:])!r}...", flush=True)
    print(f"  T1 forward: argmax_match={argmax_match}  top32_overlap={overlap}/32  "
          f"probe_rc={rc} n_pos={n_pos}", flush=True)
    print(f"       kernel argmax={k_argmax}({tok.decode([k_argmax]) if k_argmax>=0 else '?'!r})  "
          f"hf argmax={r['hf_argmax']}({tok.decode([r['hf_argmax']])!r})", flush=True)
    klp = loop_flag(k_greedy)
    exact = (k_greedy == r["hf_greedy"])
    print(f"  T2 greedy : common_prefix={common}/{min(len(k_greedy), len(r['hf_greedy']))} tokens  "
          f"exact={exact}  kernel_loop=[{klp}]", flush=True)
    print(f"       kernel: {tok.decode(k_greedy, skip_special_tokens=False).replace(chr(10),'  ')[:220]!r}", flush=True)
    print(f"       hf    : {r['hf_greedy_txt'][:220]!r}", flush=True)

print("\n" + "=" * 76, flush=True)
print("READ: T1 argmax+overlap high => prefill forward CORRECT. T1 diverges => forward bug.", flush=True)
print("      T1 ok but T2 common_prefix small => decode-path or sampling-bandaid bug.", flush=True)
