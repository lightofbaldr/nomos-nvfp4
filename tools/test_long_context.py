#!/usr/bin/env python3
"""Chunked prefill on the INT8 serving path (RUNBOOK §3c), the route every prompt longer than a chunk now takes.
GPU; run alone: pixi run --no-install python3 tools/test_long_context.py -v"""
import os
import unittest

import numpy as np

from kvsession_testlib import STOP, Kernel, doc_tokens


class ChunkedPrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = Kernel({"NOMOS_MAX_SEQ": "16384"})

    def _run(self, chunk, ids, n_gen):
        os.environ["NOMOS_PREFILL_CHUNK"] = str(chunk)    # the engine reads it at every prefill (_env_float)
        try:
            lg = self.k.prefill(ids)
            return lg, self.k.greedy(lg, n_gen)
        finally:
            os.environ["NOMOS_PREFILL_CHUNK"] = "0"

    def test_chunked_matches_unchunked_where_both_fit(self):
        ids = doc_tokens(3500)                            # unchunked still fits on 24 GB at this length
        lg_u, gen_u = self._run(0, ids, 32)
        lg_c, gen_c = self._run(1024, ids, 32)
        self.assertFalse(np.isnan(lg_c).any())
        cos = float(lg_c @ lg_u / (np.linalg.norm(lg_c) * np.linalg.norm(lg_u)))
        # Batched prefill is not batch-shape invariant (ledger R18): the routes legitimately round differently. These
        # bars catch the failure that matters (garbage, the 2026-08-05 <pad> loop) without demanding bit-equality.
        self.assertGreater(cos, 0.99)
        self.assertIn(int(np.argmax(lg_c)), np.argsort(lg_u)[-5:].tolist())
        # The two prefill routes round differently (RUNBOOK §3b, task #71), so later greedy tokens may legitimately
        # part at a near-tie: reported, not required.
        agree = sum(a == b for a, b in zip(gen_c, gen_u))
        print(f"chunked vs unchunked at 3500 tokens: cos {cos:.5f}, greedy agree {agree}/{len(gen_u)}", flush=True)

    def test_a_prompt_past_the_quadratic_ceiling_works_chunked(self):
        ids = doc_tokens(9000)                            # unchunked would need ~15 GiB of attention scores
        lg, gen = self._run(256, ids, 32)                 # the laptop's chunk at 32K (ledger R67)
        self.assertFalse(np.isnan(lg).any())
        stepped = len(gen) - (1 if gen[-1] in STOP else 0)
        self.assertEqual(self.k.cache_len(), 9000 + stepped)
        self.assertGreater(len(set(gen)), 4, "a degenerate loop (the 2026-08-05 <pad> failure) repeats one token")


if __name__ == "__main__":
    unittest.main()
