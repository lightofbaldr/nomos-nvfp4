#!/usr/bin/env python3
"""The store through the FFI on Glimmer (Task 9). GPU; run alone."""
import ctypes
import fcntl
import json
import os
import shutil
import unittest

import numpy as np

from kvsession_testlib import ROOT, TOK_DIR, Kernel, doc_tokens, store_dir_name

DIR = ROOT / "build/kvstore_gpu_test"


class StoreFfiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Class cleanups run after tearDownClass, which checks the lock files first. The sessions are not kept: they took
        # disk space under build/ until the next run (final review M11).
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)
        shutil.rmtree(DIR, ignore_errors=True)
        cls.k = Kernel({"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR),
                        "NOMOS_KVSTORE_RAM_GB": "0.002", "NOMOS_KVSTORE_CHUNK_TOKENS": "64"})
        # The RAM budget is hard and must hold one chunk (ledger R27: SessionStore refuses less). 0.002 GiB holds one
        # 64-token chunk but not the 300-token conversation that parks A, so A's restore must come from NVMe.
        bpt, budget = cls.k.bytes_per_token(), int(0.002 * 2**30)
        assert 64 * bpt <= budget < 300 * bpt, f"{bpt} bytes per token"

    @classmethod
    def tearDownClass(cls):
        """nomos_shutdown closes the store first, which releases the directory lock. The process outlives the call,
        so only this check can tell a shutdown that closes the store from one that leaves it to the exit."""
        used = cls.k.stats()["prepares"] > 0          # some test made the store
        locks = list(DIR.glob("muse-*/.lock"))
        cls.k.shutdown()
        assert locks or not used, "the store left no .lock file, so the check below would pass for nothing"
        for lock in locks:
            fd = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # BlockingIOError while the store still holds it
            finally:
                os.close(fd)

    def test_status_is_active(self):
        self.assertEqual(self.k.status(), 1)

    def test_open_names_the_store_directory(self):
        """nomos_session_open (ruling R79) returns 1 and the store's directory as the kernel spells it, whether it creates
        the store or another test already did, and again on a second call."""
        want = (1, str(DIR / store_dir_name()))
        self.assertEqual(self.k.session_open(), want)
        self.assertEqual(self.k.session_open(), want)
        self.assertTrue((DIR / store_dir_name() / ".lock").exists(), "the store exists once open has returned")

    def test_second_turn_reuses_the_first(self):
        ids1 = doc_tokens(300)
        c1, gen = self.k.turn(ids1, 24)
        self.assertEqual(c1, 0)
        kv1 = self.k.cache_len()   # ids1 + every generated token except a final stop token (greedy steps the rest)
        ids2 = ids1 + gen + doc_tokens(340)[300:]
        c2, _ = self.k.turn(ids2, 8)
        self.assertEqual(c2, kv1)
        self.assertGreater(c2, 300)

    def test_ram_eviction_resumes_bit_identically(self):
        """Spec tests 2 and 5: a RAM tier of one chunk forces the restore through NVMe."""
        a = doc_tokens(600)[300:]                          # a different conversation from the other tests
        _, gen_a = self.k.turn(a, 32)
        n = len(a) + len(gen_a)
        # The never-parked reference runs the op sequence the resume will run: KV for (a + gen_a)[:n-1], then a
        # 1-token continuation. truncate() is a no-op when a stop token ended the turn without being stepped.
        self.assertEqual(self.k.truncate(n - 1), 0)
        ref = self.k.greedy(self.k.prefill_cont([gen_a[-1]]), 32)
        self.k.turn(doc_tokens(900)[600:], 8)             # another conversation parks A and pushes it out of RAM
        before = self.k.stats()
        c = self.k.prepare(a + gen_a)                     # A again: restored, cut to n - 1 (a fork)
        self.assertEqual(c, n - 1)
        self.assertEqual(self.k.live_tokens(), (a + gen_a)[:c])
        after = self.k.stats()
        self.assertGreaterEqual(after["restores_nvme"], before["restores_nvme"] + 1)
        self.assertEqual(self.k.greedy(self.k.prefill_cont([gen_a[-1]]), 32), ref)

    def test_foreign_generate_does_not_corrupt(self):       # Review Focus 2
        a = doc_tokens(1200)[900:]
        self.k.turn(a, 16)
        p, n = np.asarray(doc_tokens(50), dtype=np.int32), 50
        out = np.zeros(64, dtype=np.int32)
        rc = self.k.lib.nomos_generate(self.k.h, p.ctypes.data, n, 8, out.ctypes.data, 64, 0.0, 1.0, 1.0)
        self.assertGreater(rc, 0)                                                     # the foreign generate ran ...
        self.assertNotEqual(len(self.k.live_tokens()), self.k.cache_len())            # ... and left KV no tokens describe
        before = self.k.stats()
        ids = a + doc_tokens(1240)[1200:]
        c = self.k.prepare(ids)
        self.assertEqual(self.k.cache_len(), c)
        self.assertEqual(self.k.live_tokens(), ids[:c])   # never the foreign KV under the wrong tokens
        self.assertGreaterEqual(c, len(a))                # the committed turn came back ...
        after = self.k.stats()
        restores = lambda s: s["restores_ram"] + s["restores_nvme"]   # a RAM tier of one chunk makes most of them NVMe
        self.assertEqual(restores(after), restores(before) + 1)       # ... restored from the store, not trusted in the engine


    def test_store_off_leaves_the_engine_alone(self):      # NOMOS_KVSTORE unset is the default: no store, no side effects
        ids = doc_tokens(2100)[1800:]
        self.k.prefill(ids)
        before = self.k.stats()
        os.environ.pop("NOMOS_KVSTORE")
        try:
            self.assertEqual(self.k.status(), 0)
            self.assertEqual(self.k.prepare(ids), 0)
            self.assertEqual(self.k.cache_len(), len(ids))        # prepare did not touch the KV ...
            self.assertEqual(self.k.live_tokens(), ids)           # ... nor the tokens that describe it
            self.assertEqual(self.k.commit(), 0)
            self.assertEqual(self.k.stats(), before)              # and the store counted nothing
        finally:
            os.environ["NOMOS_KVSTORE"] = "1"
            self.k.reset()                                        # the next test starts from a clean engine



if __name__ == "__main__":
    unittest.main()
