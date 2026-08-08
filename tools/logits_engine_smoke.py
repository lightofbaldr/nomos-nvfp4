#!/usr/bin/env python3
"""Spike: prove the Python-host logits-engine loop.

Kernel returns logits (nomos_prefill / nomos_decode_step); the HOST does the
sampling (argmax here) and feeds tokens back. If the text is coherent ("Paris"),
the architecture holds — sampling has left the kernel and lives in Python, which
is where the grammar library plugs in next.
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H
from transformers import AutoTokenizer

VOCAB = 262144


def main() -> None:
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lib = H.load_lib()
    lib.nomos_prefill.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
    lib.nomos_prefill.restype = ctypes.c_int32
    lib.nomos_decode_step.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
    lib.nomos_decode_step.restype = ctypes.c_int32
    h = H.init_engine(lib)

    msgs = [{"role": "user", "content": "What is the capital of France? Think briefly, then answer."}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int32)
    logits = np.zeros(VOCAB, dtype=np.float32)

    rc = lib.nomos_prefill(h, ids.ctypes.data, len(ids), logits.ctypes.data)
    print(f"[prefill] rc={rc}  prompt_tok={len(ids)}  argmax0={int(np.argmax(logits))}", flush=True)
    if rc != 0:
        return

    out = [int(np.argmax(logits))]                       # first generated token (host argmax)
    for _ in range(70):
        rc = lib.nomos_decode_step(h, out[-1], logits.ctypes.data)
        if rc != 0:
            print(f"[step] rc={rc} — stopping", flush=True)
            break
        out.append(int(np.argmax(logits)))

    print("\n=== HOST-sampled (argmax) text — kernel = pure logits engine ===", flush=True)
    print(repr(tok.decode(out, skip_special_tokens=False)[:500]), flush=True)


if __name__ == "__main__":
    main()
