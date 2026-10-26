#!/usr/bin/env python3
"""A continuation's suffix feeds the drafter wherever the target accepts it (ledger R50, Task 15).

nomos_prefill_cont captures a suffix into the drafter only when it fits: kv_drafter_len + n <= the loaded drafter's
max_ctx, checked before the target runs, so a drafter capacity limit never fails a turn. Task 12 built that gate when the
drafter held a compile-time 8,192 rows while the target's KV reached 16,384 here. A suffix past 8,192 was then left out,
the call still succeeded and the drafter trailed the KV.

The drafter's context is now sized by NOMOS_MAX_SEQ, the target's own bound, so a suffix the drafter cannot hold no
longer exists (the target's capacity guard, engine_decode.mojo, fires first). The same three suffixes (8000 + 192 = 8192,
+ 193, + 300) are all captured, and the drafter's length stays equal to cache_len. A gate that still compared with
8,192 would leave the last two behind, which is what this pins.

One Glimmer + drafter at NOMOS_MAX_SEQ=16384 with its own store directory: about 35 s (under KVSESSION_KV=int4 it passes
but takes about 6 minutes; not profiled). GPU; run alone:
pixi run --no-install python3 tools/test_kv_session_dflash_capacity.py -v"""
import shutil
import unittest

import numpy as np

from kvsession_testlib import ROOT, STOP, Kernel, doc_tokens

DIR = ROOT / "build/kvstore_dflash_capacity_test"
OLD_LIMIT = 8192               # the drafter's context before it followed NOMOS_MAX_SEQ
START = 8000                   # where the store, the target and the drafter are in step
SENTINEL = np.float32(-1.0e30)


def cont(k, suffix):
    """nomos_prefill_cont: (rc, whether it wrote the logits). The testlib's prefill_cont asserts rc == 0, which is what
    these cases must observe instead. The logits' first and last values are poisoned first, as nomos_prefill does."""
    a = np.ascontiguousarray(suffix, dtype=np.int32)
    k.logits[0] = k.logits[-1] = SENTINEL
    rc = k.lib.nomos_prefill_cont(k.h, a.ctypes.data, len(a), k.logits.ctypes.data)
    return rc, bool(k.logits[0] != SENTINEL and k.logits[-1] != SENTINEL)


class DrafterCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)    # the store's files are not kept (final review M11)

    def test_a_suffix_across_the_old_limit_feeds_the_drafter(self):
        shutil.rmtree(DIR, ignore_errors=True)
        # NOMOS_PREFILL_CHUNK bounds the prefill's memory at 8000 tokens (RUNBOOK 3c); NOMOS_MAX_SEQ is what lets the
        # target, and with it the drafter, go past 8192 rows.
        k = Kernel({"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR), "NOMOS_MAX_SEQ": "16384",
                    "NOMOS_PREFILL_CHUNK": "2048"})
        self.addCleanup(k.shutdown)
        k.dflash_load()
        ids = doc_tokens(START + 400)
        self.assertEqual(k.prepare(ids[:START], spec=True), 0)          # creates the store
        k.prefill(ids[:START])
        self.assertEqual((k.cache_len(), k.dflash_len()), (START, START))   # in step: the drafter holds every row
        cases = (("ends exactly at the old limit (8000 + 192 = 8192)", OLD_LIMIT - START),
                 ("one row past it (8000 + 193)", OLD_LIMIT - START + 1),
                 ("a batch past it (8000 + 300)", 300))
        for name, n in cases:
            with self.subTest(name):
                self.assertEqual(k.truncate(START), 0)                  # back to the in-step state, whatever ran before
                self.assertEqual(k.dflash_len(), START)
                rc, wrote = cont(k, ids[START:START + n])
                self.assertEqual(rc, 0, "a drafter capacity limit must not fail the turn")
                self.assertTrue(wrote, "the logits must be written")
                self.assertEqual(k.cache_len(), START + n)
                self.assertEqual(k.live_tokens(), ids[:START + n])      # the engine still tracks what it holds
                self.assertEqual(k.dflash_len(), k.cache_len())         # captured: the drafter keeps step with the KV
                # and the turn goes on: decode steps do not feed the drafter, so it stays where the capture left it
                out = k.greedy(k.logits.copy(), 4)
                stepped = len(out) - (1 if out[-1] in STOP else 0)      # a stop token ends the loop unstepped
                self.assertGreater(stepped, 0)
                self.assertEqual(k.cache_len(), START + n + stepped)
                self.assertEqual(k.dflash_len(), START + n)


if __name__ == "__main__":
    unittest.main()
