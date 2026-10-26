#!/usr/bin/env python3
"""The DFlash drafter's context grows past the old 8,192-row limit (Adam 2026-10-13: 32K testing on the laptop,
128K on an RTX Pro 6000). GPU; run alone: pixi run --no-install python3 tools/test_dflash_long_context.py -v

Speculative decoding is lossless whatever the drafter proposes, because the target verifies every token. The first test
therefore passes with a drafter that holds every row but ignores those past 8,192, which would show only as long
conversations decoding at base speed. The second test reads the drafter's proposals instead."""
import unittest

import numpy as np

from kvsession_testlib import Kernel, doc_tokens

N = 8500          # past the old compile-time limit of 8,192 context rows
OLD_LIMIT = 8192


def proposals(k, seeds):
    """The drafter's 15 proposed tokens for a block that opens with each seed, at the end of the context it holds."""
    out, res = np.zeros(15, dtype=np.int32), []
    for s in seeds:
        rc = k.lib.nomos_dflash_draft_block(k.h, int(s), k.cache_len(), out.ctypes.data)
        assert rc == 15, f"nomos_dflash_draft_block rc={rc}"
        res.append(out.tolist())
    return res


class DrafterPastEightK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Chunked prefill keeps prefill memory bounded at this length (RUNBOOK §3c); NOMOS_MAX_SEQ sizes the drafter.
        cls.k = Kernel({"NOMOS_MAX_SEQ": "16384", "NOMOS_PREFILL_CHUNK": "2048"})
        cls.addClassCleanup(cls.k.shutdown)     # the engine and drafter hold ~16 GB: a next class in this process needs them
        cls.k.dflash_load()

    def test_spec_decode_past_8k_is_lossless(self):
        ids = doc_tokens(N)
        base = self.k.greedy(self.k.prefill(ids), 64)
        lg = self.k.prefill(ids)                          # the same prefill again: an identical state
        self.assertEqual(self.k.cache_len(), N)
        self.assertEqual(self.k.dflash_len(), N)          # every row, not the first 8,192
        toks, acc = self.k.spec_trace(lg, 64)
        self.assertEqual(toks, base)                      # lossless past the old limit
        # Every cycle gave the drafter its seed row and the rows of the drafts it accepted. The KV is one row ahead of
        # that when the loop ended on its exact final step (spec_trace, unlike the host since ecc2140, does not drop
        # that row), and whether it does depends on the acceptance. So the drafter is held to its own bookkeeping.
        self.assertEqual(self.k.dflash_len(), N + len(acc) + sum(acc))
        self.assertIn(self.k.cache_len() - self.k.dflash_len(), (0, 1))
        self.assertGreater(sum(acc), 0, "a dead drafter (misplaced context) accepts nothing")
        e = 1 + float(np.mean(acc))
        print(f"DFlash at {N} tokens: E = {e:.2f} over {len(acc)} cycles", flush=True)
        # A drafter that holds the rows but reads the wrong ones still accepts a little. E is 2.42 on this prompt, and
        # 1.17 when the drafter's attention stops at row 8,192.
        self.assertGreater(e, 1.5, f"E = {e:.2f}: speculation does not pay at {N} tokens")

    def test_the_drafter_attends_the_rows_past_8k(self):
        """Two prompts share their first 8,192 tokens and their last token, and differ in between. A block with the
        same seed at the same position then reaches the drafter differently only through the context rows from 8,192
        on, so a drafter that ignored those rows would propose the same 15 tokens for both. The seeds are tokens of the
        shared text: nothing about the seed differs. All five drafter layers attend only the last 2,048 keys, and the
        rows that differ here, 8,192 to the end, sit well inside that window."""
        long = doc_tokens(N + 400)
        a = long[:N]
        b = a[:OLD_LIMIT] + long[N:N + (N - OLD_LIMIT - 1)] + a[-1:]       # the text after a's end fills the middle
        self.assertEqual((len(b), b[:OLD_LIMIT], b[-1]), (N, a[:OLD_LIMIT], a[-1]))
        self.assertNotEqual(b[OLD_LIMIT:-1], a[OLD_LIMIT:-1])
        seeds = [a[100 + 500 * i] for i in range(16)]
        self.k.prefill(a)
        self.assertEqual(self.k.dflash_len(), N)
        first = proposals(self.k, seeds)
        self.k.prefill(a)
        self.assertEqual(proposals(self.k, seeds), first, "one context, two answers: the comparison below means nothing")
        self.k.prefill(b)
        self.assertEqual(self.k.dflash_len(), N)
        other = proposals(self.k, seeds)
        seeds_differ = sum(x != y for x, y in zip(first, other))
        tokens_differ = sum(p != q for x, y in zip(first, other) for p, q in zip(x, y))
        print(f"DFlash proposals, prompts that differ past row {OLD_LIMIT}: {seeds_differ} of {len(seeds)} seeds "
              f"differ, {tokens_differ} of {15 * len(seeds)} proposed tokens", flush=True)
        self.assertGreaterEqual(seeds_differ, 12, "the drafter's proposals do not follow its context rows past 8,192")


if __name__ == "__main__":
    unittest.main()
