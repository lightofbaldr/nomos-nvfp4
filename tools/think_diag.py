#!/usr/bin/env python3
"""Does the full-4-bit kernel generate COHERENT THINKING, or only direct answers?

Drives the kernel directly (via the fp4 harness FFI — no serve, no tokenizer-svc) with the
two chat-template shapes:
  (a) enable_thinking=True  -> <|think|> system signal; model must GENERATE its own
      <|channel>thought ... <channel|> reasoning, then the answer. (Gemma-4 reasoning mode.)
  (b) enable_thinking=False -> empty closed thought channel; model answers directly.

If (a) garbles and (b) is clean, the FP4 build can't generate reasoning (a real quality
finding). If both are clean, the garbage was in the serve plumbing, not the kernel.
Run with the full-4-bit env (NOMOS_W4A4=1 NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA=1).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H
from transformers import AutoTokenizer


def main() -> None:
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lib = H.load_lib()
    h = H.init_engine(lib)
    msgs = [{"role": "user", "content": "What is the capital of France? Think briefly, then answer."}]

    def run(label: str, enable_thinking: bool, n: int = 220) -> None:
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        ids = np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int32)
        out = np.zeros(n, dtype=np.int32)
        nout = lib.nomos_generate(h, ids.ctypes.data, len(ids), n, out.ctypes.data, n, 0.0, -1.0, 1.0)
        txt = tok.decode(out[: max(int(nout), 0)].tolist(), skip_special_tokens=False)
        print(f"\n===== {label}  (prompt {len(ids)} tok, generated {max(int(nout),0)}) =====", flush=True)
        print(repr(txt[:600]), flush=True)

    run("(a) enable_thinking=True  [REASONING mode — the serve's format]", True)
    run("(b) enable_thinking=False [direct-answer mode — what the harness used]", False)


if __name__ == "__main__":
    main()
