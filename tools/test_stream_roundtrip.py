#!/usr/bin/env python3
"""Plan Task 10b: a streaming client that sends Muse's replies back verbatim re-renders the next prompt to exactly the
token ids the KV holds (the premise of plan note 12 for Adam's client). Plain generation, no store: each turn is a fresh
prefill of the serve's own prompt, then decode steps; the stop token is sampled, never stepped. GPU; run alone."""
import sys
import unittest

import numpy as np

from kvsession_testlib import ROOT, Kernel, _tokenizer

sys.path.insert(0, str(ROOT / "pyhost"))
from serve_protocol import AtemServeProtocol, atem_exact_fields, normalize_atem_messages  # noqa: E402

MAX_NEW = 1500
QUESTIONS = [
    "In two sentences, why is the sky blue?",
    "Now give me three emoji that capture that, and the Japanese word for sky.",
    "Write a Python one-liner that reverses a string, in a fenced code block.",
]


class StreamRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()
        cls.proto = AtemServeProtocol(cls.tok, "muse")
        cls.k = Kernel()

    @classmethod
    def tearDownClass(cls):
        cls.k.shutdown()

    def render(self, msgs):
        s = self.tok.apply_chat_template(normalize_atem_messages(msgs), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        return [int(x) for x in self.tok(self.proto.finish_prompt(s, True), add_special_tokens=False).input_ids]

    def generate(self, ids, temp, rng):
        lg, out = self.k.prefill(ids), []
        for _ in range(MAX_NEW):
            if temp <= 0:
                t = int(np.argmax(lg))
            else:
                z = lg.astype(np.float64) / temp
                p = np.exp(z - z.max())
                t = int(rng.choice(len(p), p=p / p.sum()))
            out.append(t)
            if t in self.proto.stop_ids:
                break
            lg = self.k.step(t)
        return out

    def conversation(self, temp, seed):
        rng, msgs, split_turns = np.random.default_rng(seed), [], 0
        for q in QUESTIONS:
            msgs.append({"role": "user", "content": q})
            ids = self.render(msgs)
            gen = self.generate(ids, temp, rng)
            if any("�" in self.tok.decode([t]) for t in gen):
                split_turns += 1
            held = ids + (gen[:-1] if gen and gen[-1] in self.proto.stop_ids else gen)
            self.assertEqual(self.k.cache_len(), len(held))
            reasoning, content, _ = atem_exact_fields(self.tok, self.proto, gen)
            msgs.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
            nxt = self.render(msgs + [{"role": "user", "content": "(next)"}])
            m = next((i for i, (x, y) in enumerate(zip(held, nxt)) if x != y), min(len(held), len(nxt)))
            self.assertEqual(m, len(held), f"{q!r}: history diverges at {m}/{len(held)}: "
                             f"held {self.tok.decode(held[m:m + 4])!r} vs sent {self.tok.decode(nxt[m:m + 4])!r}")
        return split_turns

    def test_greedy(self):
        split_turns = self.conversation(0.0, 0)
        self.assertGreaterEqual(split_turns, 1,
                                "the greedy conversation produced no split character: this run proves nothing about the hold-back")

    def test_sampled(self):
        self.conversation(0.8, 7)


if __name__ == "__main__":
    unittest.main()
