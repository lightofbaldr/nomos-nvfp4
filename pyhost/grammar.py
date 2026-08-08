"""Tool-call grammar FSM — a logits processor for KernelLM.generate (Stage A).

Holds the model to valid Gemma-4 tool-call structure ONCE it commits to a call,
so 4-bit precision noise can't produce `call://...` garbage or invalid tool names.
Free text / thinking is UNCONSTRAINED — we only shape the call envelope.

Token IDs verified 2026-06-17 against gemma4_qat_unq:
  <|tool_call>=48  <tool_call|>=49  <|"|>=52  call:=[6639,236787]  {=236782  }=236783  {}=16454
"""
from __future__ import annotations

import numpy as np

TOOL_OPEN = 48
TOOL_CLOSE = 49
QUOTE = 52
CHANNEL_END = 101   # <channel|> — closes the thinking channel; after it the answer/tool begins
CALL_SEQ = [6639, 236787]   # 'call' ':'
LBRACE = 236782
RBRACE = 236783
EMPTY_BRACES = 16454        # '{}'
NEG = np.float32(-1e30)


class ToolGrammar:
    """Callable: (step_idx, prev_tokens, logits) -> logits. Reset() per generation."""

    def __init__(self, tokenizer, tool_names: list[str], force: bool = False,
                 thinking: bool = True) -> None:
        # trie of tool-name token sequences; node["__end__"]=True marks a complete name
        self.root: dict = {}
        for name in tool_names:
            seq = tokenizer(name, add_special_tokens=False).input_ids
            cur = self.root
            for tid in seq:
                cur = cur.setdefault(tid, {})
            cur["__end__"] = True
        # force=True (OpenAI tool_choice "required" / a named function) makes the model emit a
        # tool call instead of free text: once thinking closes (<channel|>), the next token is
        # FORCED to <|tool_call>. Needed for pydantic-ai output_type (structured outputs) — without
        # it a 4-bit model may answer in prose and output-validation fails. thinking=False means no
        # thought channel is pre-filled, so the answer starts immediately (force from step 0).
        self.force = force
        self.thinking = thinking
        self.reset()

    def reset(self) -> None:
        self.state = "FREE"
        self._seen = 0
        self._call_i = 0          # progress through CALL_SEQ in OPENED
        self.node = self.root     # current trie node in NAME
        self.depth = 0            # brace depth in ARGS
        self.in_quote = False
        self._think_done = not self.thinking   # forced tool opens only after thinking closes
        self._did_call = False                 # force ONE tool call, not repeated

    def _advance(self, t: int) -> None:
        if t == CHANNEL_END:
            self._think_done = True   # thinking channel closed → answer/tool region begins
        s = self.state
        if s == "FREE":
            if t == TOOL_OPEN:
                self.state, self._call_i = "OPENED", 0
        elif s == "OPENED":
            self._call_i += 1
            if self._call_i >= len(CALL_SEQ):
                self.state, self.node = "NAME", self.root
        elif s == "NAME":
            if t == LBRACE:
                self.state, self.depth, self.in_quote = "ARGS", 1, False
            elif t == EMPTY_BRACES:
                self.state = "ARGS_DONE"
            elif t in self.node:
                self.node = self.node[t]
        elif s == "ARGS":
            if t == QUOTE:
                self.in_quote = not self.in_quote
            elif not self.in_quote:
                if t == LBRACE:
                    self.depth += 1
                elif t == RBRACE:
                    self.depth -= 1
                    if self.depth <= 0:
                        self.state = "ARGS_DONE"
        elif s == "ARGS_DONE":
            if t == TOOL_CLOSE:
                self.state = "FREE"
                self._did_call = True   # a forced call is satisfied; don't force another

    def _allowed(self) -> list[int] | None:
        """Return the allow-list for the CURRENT state, or None for 'unconstrained'."""
        s = self.state
        if s == "FREE":
            # force=True: once thinking has closed, the model MUST open a tool call (no free text).
            if self.force and self._think_done and not self._did_call:
                return [TOOL_OPEN]
            return None
        if s == "ARGS":
            return None
        if s == "OPENED":
            return [CALL_SEQ[self._call_i]]
        if s == "NAME":
            allow = [k for k in self.node.keys() if k != "__end__"]
            if self.node.get("__end__"):
                allow += [LBRACE, EMPTY_BRACES]
            return allow
        if s == "ARGS_DONE":
            return [TOOL_CLOSE]
        return None

    def __call__(self, i: int, prev: list[int], logits: np.ndarray) -> np.ndarray:
        # advance state for any tokens chosen since the last call
        for t in prev[self._seen:]:
            self._advance(t)
        self._seen = len(prev)
        allow = self._allowed()
        if allow is None:
            return logits
        masked = np.full_like(logits, NEG)
        idx = np.asarray(allow, dtype=np.int64)
        masked[idx] = logits[idx]
        return masked


# ── self-test: does the grammar force a clean tool call? ─────────────────────
if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from host import KernelLM, H
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lm = KernelLM()
    tools = ["write_file", "read_file", "list_files", "run_python", "run_pytest"]
    gram = ToolGrammar(tok, tools)

    tool_defs = [{"type": "function", "function": {"name": n, "description": n,
                  "parameters": {"type": "object", "properties": {}}}} for n in tools]
    msgs = [{"role": "user", "content": "List the files in the sandbox using the tool."}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True, tools=tool_defs)
    ids = list(np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int64))

    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
    gram.reset()
    out = lm.generate(ids, max_new_tokens=120, stop_ids=stop, processor=gram)
    txt = tok.decode(out, skip_special_tokens=False)
    print("=== GRAMMAR-CONSTRAINED output ===", flush=True)
    print(repr(txt[:400]), flush=True)
    print("contains well-formed call:", "<|tool_call>call:" in txt and "<tool_call|>" in txt, flush=True)
