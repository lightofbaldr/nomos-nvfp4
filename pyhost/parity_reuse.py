"""Parity gate for KV reuse (#2): a full fresh prefill must produce the SAME last-position
logits + greedy continuation as prefill(prefix) + prefill_cont(suffix). Argmax-level match is
the bar (suffix attention is per-token-offset vs batched in the full path → tiny FP drift OK).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import KernelLM, H            # noqa: E402
from transformers import AutoTokenizer   # noqa: E402

GEN = 24
tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
lm = KernelLM()

msgs = [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
s += "<|channel>thought\n"
ids = list(np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int64))
print("prompt tokens:", len(ids), flush=True)


def greedy(logits, n):
    out = [int(np.argmax(logits))]
    for _ in range(n - 1):
        out.append(int(np.argmax(lm.step(out[-1]))))
    return out


# A) full fresh prefill
lf = lm.prefill(ids).copy()
seq_full = greedy(lf, GEN)

# B) reuse: seed cache with a prefix, continue-prefill the suffix at the offset
P = len(ids) - 8
lm.prefill(ids[:P])                          # fresh prefill of prefix → cache = [0,P)
lr = lm.prefill_cont(ids[P:]).copy()         # batched-offset append of suffix at P
seq_reuse = greedy(lr, GEN)

print(f"P(reused)={P} suffix={len(ids) - P}", flush=True)
print("last-pos argmax  full=", int(np.argmax(lf)), " reuse=", int(np.argmax(lr)),
      "  MATCH" if int(np.argmax(lf)) == int(np.argmax(lr)) else "  DIFFER", flush=True)
print("last-pos logit max|diff|=", float(np.max(np.abs(lf - lr))), flush=True)
print("greedy continuation identical:", seq_full == seq_reuse, flush=True)
print("  full :", repr(tok.decode(seq_full, skip_special_tokens=False)[:140]), flush=True)
print("  reuse:", repr(tok.decode(seq_reuse, skip_special_tokens=False)[:140]), flush=True)

# C) the prefill_reuse helper (prefix-match path the serve will use)
lm.prefill(ids[:P])
lr2, reused = lm.prefill_reuse(ids, ids[:P])
print(f"prefill_reuse reused_len={reused} argmax={int(np.argmax(lr2))} "
      f"== full:{int(np.argmax(lr2)) == int(np.argmax(lf))}", flush=True)
