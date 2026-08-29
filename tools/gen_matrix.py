#!/usr/bin/env python3
"""Isolate WHICH prompts the NVFP4 W4A16 kernel garbles. One load, several prompt
types, greedy, kernel-direct. Factual reasoning is known-clean (Paris); find the
boundary: creative gen? code gen? the phrasing? Run under the serve's precision env."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import fp4_parity_harness as H
from transformers import AutoTokenizer

PROMPTS = [
    ("factual (control)", "What is the capital of France? Think briefly, then answer."),
    ("creative", "Write a short haiku about the ocean."),
    ("code+thinking", "Write a Python function is_prime(n). Think step by step, then give the code."),
    ("code+just", "Write a Python function is_prime(n). Just the code."),
    ("list-no-think", "List three prime numbers."),
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lib = H.load_lib()
    h = H.init_engine(lib)
    for label, content in PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int32)
        out = np.zeros(100, dtype=np.int32)
        n = lib.nomos_generate(h, ids.ctypes.data, len(ids), 100, out.ctypes.data, 100, 0.0, -1.0, 1.0)
        txt = tok.decode(out[: max(int(n), 0)].tolist(), skip_special_tokens=False)
        print(f"\n[{label}]  gen={max(int(n),0)}\n  {repr(txt[:260])}", flush=True)


if __name__ == "__main__":
    main()
