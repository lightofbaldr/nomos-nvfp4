#!/usr/bin/env python3
"""Does the kernel generate a COHERENT tool-call when given the Gemma-4 tool prompt,
or garbage? Drives the kernel DIRECTLY (fp4 harness FFI — no serve, no Go parser) with
the exact chat-template-rendered tool prompt, greedy. Plain reasoning already works
(capital-of-France -> Paris), so if THIS garbles the bug is in the kernel/model's
handling of the tool-format prompt, not the serve plumbing or sampling.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_parity_harness as H
from transformers import AutoTokenizer

TOOLS = [{
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in the sandbox",
        "parameters": {"type": "object", "properties": {}},
    },
}]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lib = H.load_lib()
    h = H.init_engine(lib)
    msgs = [{"role": "user", "content": "List the files in the current directory."}]

    def run(label: str, tools, n: int = 80) -> None:
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            **({"tools": tools} if tools else {}),
        )
        ids = np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int32)
        out = np.zeros(n, dtype=np.int32)
        nout = lib.nomos_generate(h, ids.ctypes.data, len(ids), n, out.ctypes.data, n, 0.0, -1.0, 1.0)
        txt = tok.decode(out[: max(int(nout), 0)].tolist(), skip_special_tokens=False)
        print(f"\n===== {label}  (prompt {len(ids)} tok, gen {max(int(nout),0)}) =====", flush=True)
        print(repr(txt[:500]), flush=True)

    run("(a) NO tools  [plain reasoning — known good]", None)
    run("(b) WITH tools [the failing path]", TOOLS)


if __name__ == "__main__":
    main()
