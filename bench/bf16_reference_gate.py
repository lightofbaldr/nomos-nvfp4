#!/usr/bin/env python3
"""Compare Nomos Q4 residuals with the bf16 prompt-forward reference.

This is a deterministic quality gate, not a benchmark. Run once with the shipped
KV layout and once with block-32:

  NOMOS_KV_I4_BLOCK=0  python bench/bf16_reference_gate.py --label shipped
  NOMOS_KV_I4_BLOCK=32 python bench/bf16_reference_gate.py --label block32

The current bare-instruct greedy reference is informational only; residual cosine
is the meaningful gate until a chat-templated greedy reference exists.
"""
import argparse
import ctypes as c
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


D = 5376
LAYERS = 60
VOCAB = 262144
ROOT = Path(os.environ.get("NOMOS_ROOT", str(Path(__file__).resolve().parents[1])))
REF = Path(os.environ.get("BF16_REF", ""))
MODEL = os.environ.get("NOMOS_WEIGHTS", "")
TOKENIZER = os.environ.get("BF16_TOKENIZER", "")


def bind(lib, name, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def cosine_rows(a, b):
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    return np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    meta = json.loads((REF / "meta.json").read_text())
    greedy_ref = json.loads((REF / "greedy.json").read_text())
    greedy_ids = greedy_ref.get("greedy_token_ids", meta.get("greedy"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt_ids = []
    for i, prompt in enumerate(meta["prompts"]):
        ids = np.asarray(tok(prompt, add_special_tokens=True).input_ids, np.int32)
        ref_n = int(np.load(REF / f"embed_p{i}.npy", mmap_mode="r").shape[0])
        print(f"TOKEN_ALIGN p{i}: kernel={len(ids)} ref={ref_n}", flush=True)
        if len(ids) != ref_n:
            raise SystemExit(f"tokenization mismatch p{i}: {len(ids)} != {ref_n}")
        prompt_ids.append(ids)

    lib = c.CDLL(str(ROOT / "libnomos_kernel.so"))
    init = bind(lib, "nomos_init", c.c_void_p, [c.c_char_p])
    capture = bind(
        lib, "nomos_debug_prefill_all_layers", c.c_int32,
        [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p],
    )
    reset = bind(lib, "nomos_reset_kv", c.c_int32, [c.c_void_p])
    prefill = bind(
        lib, "nomos_prefill", c.c_int32,
        [c.c_void_p, c.c_void_p, c.c_int32, c.c_void_p],
    )
    decode = bind(
        lib, "nomos_decode_step", c.c_int32,
        [c.c_void_p, c.c_int32, c.c_void_p],
    )
    h = init(MODEL.encode())
    if not h:
        raise SystemExit("nomos_init failed")

    layer_cos = [[] for _ in range(LAYERS)]
    prompt_layer_cos = []
    argmax_rows = []
    greedy_rows = []
    logits = np.empty(VOCAB, np.float32)
    for i, ids in enumerate(prompt_ids):
        states = np.empty((len(ids), LAYERS, D), np.float32)
        rc = capture(h, ids.ctypes.data, len(ids), states.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"capture p{i} rc={rc}")
        pc = []
        for layer in range(LAYERS):
            ref = np.load(REF / f"layer{layer}_p{i}.npy")
            cs = cosine_rows(states[:, layer], ref)
            layer_cos[layer].extend(float(x) for x in cs)
            pc.append(float(np.mean(cs)))
        prompt_layer_cos.append(pc)

        reset(h)
        if prefill(h, ids.ctypes.data, len(ids), logits.ctypes.data) != 0:
            raise RuntimeError(f"prefill p{i} failed")
        first = int(np.argmax(logits))
        ref_first = int(meta["stages"][f"p{i}"]["argmax_last"])
        argmax_rows.append({"prompt": i, "kernel": first, "ref": ref_first,
                            "match": first == ref_first})
        generated = [first]
        cur = first
        for _ in range(7):
            if decode(h, cur, logits.ctypes.data) != 0:
                raise RuntimeError(f"decode p{i} failed")
            cur = int(np.argmax(logits))
            generated.append(cur)
        ref_gen = [int(x) for x in greedy_ids[i]]
        greedy_rows.append({"prompt": i, "match": sum(a == b for a, b in zip(generated, ref_gen)),
                            "kernel": generated, "ref": ref_gen})

    means = [float(np.mean(x)) for x in layer_cos]
    result = {
        "label": args.label,
        "kv_i4_block": int(os.environ.get("NOMOS_KV_I4_BLOCK", "0") or 0),
        "layer_mean_cosine": means,
        "prompt_layer_cosine": prompt_layer_cos,
        "argmax_last": argmax_rows,
        "greedy_informational": greedy_rows,
        "greedy_caveat": "bare instruct prompts; informational, not a quality gate",
    }
    out = args.output or (Path("bench") / f"bf16_gate_{args.label}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("LAYER_MEAN", " ".join(f"{x:.6f}" for x in means), flush=True)
    print("TAIL_48_59", " ".join(f"L{i}={means[i]:.6f}" for i in range(48, 60)), flush=True)
    print("ARGMAX", json.dumps(argmax_rows), flush=True)
    print("GREEDY_INFORMATIONAL", json.dumps(greedy_rows), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
