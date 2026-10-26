#!/usr/bin/env python3
"""DFlash state in the KV session store (Tasks 11-13). GPU; run alone."""
import contextlib
import io
import itertools
import os
import shutil
import unittest

import numpy as np

from kvsession_testlib import BOS, EXPECTED_BPT, ROOT, STOP, Kernel, doc_tokens

DIR = ROOT / "build/kvstore_dflash_test"
_K = None


def kernel():
    """One Glimmer + drafter per process, shared by every class here (unittest runs classes in name order)."""
    global _K
    if _K is None:
        shutil.rmtree(DIR, ignore_errors=True)
        _K = Kernel({"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR)})
        _K.dflash_load()
    return _K


def tearDownModule():
    """After the last class: free the shared engine (it closes the store), then remove the store's files, about 0.6 GB
    (final review M11)."""
    if _K is not None:
        _K.shutdown()
    shutil.rmtree(DIR, ignore_errors=True)


class DrafterContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = kernel()

    def test_bytes_per_token_include_the_drafter(self):
        self.assertEqual(self.k.bytes_per_token(), EXPECTED_BPT + 40960)   # + 5 layers x K,V x 1024 fp32

    def test_drafter_context_round_trips(self):
        ids = doc_tokens(300)
        self.k.prefill(ids)
        self.assertEqual(self.k.dflash_len(), 300)
        full = self.k.export(0, 300)
        self.k.reset()
        self.k.lib.nomos_dflash_reset(self.k.h)
        self.assertEqual(self.k.import_(0, 300, full, 300, ids), 0)
        self.assertEqual(self.k.dflash_len(), 300)
        self.assertTrue(np.array_equal(self.k.export(0, 300), full))

    def test_truncate_cuts_the_drafter_too(self):
        self.k.prefill(doc_tokens(200))
        self.assertEqual(self.k.truncate(150), 0)
        self.assertEqual(self.k.dflash_len(), 150)

    # -- beyond the brief's three: the paths its round trip cannot tell apart from a no-op ------------------------------
    # A chunk is the target's components followed by the drafter's ten, each packed [token][row] (one head). The
    # drafter's ten take 40960 B per token, so the drafter part of an n-token export starts at n * EXPECTED_BPT.
    ROW = 4096   # one drafter row: 1024 fp32

    def drafter_part(self, buf, n):
        return buf[n * EXPECTED_BPT:]

    def test_restore_overwrites_the_drafter_rows_of_another_session(self):
        """reset() and nomos_dflash_reset zero the lengths, not the rows (the target's trap, see the engine suite's
        test_restore_after_another_session_ran): an import that copies no drafter rows still passes the round trip
        above, because the old rows are still there. Park a session as two consecutive chunks (the second starts at
        a > 0), run a different session over the same buffers, then restore: the rows are the parked session's again."""
        a = doc_tokens(300)
        other = [BOS] + doc_tokens(700)[401:]          # 300 different tokens
        self.k.prefill(a)
        full = self.k.export(0, 300)
        first, second = self.k.export(0, 150), self.k.export(150, 300)
        self.assertTrue(self.drafter_part(full, 300).any(), "the drafter rows must not be all zero")
        # K0 V0 K1 V1 ... K4 V4 must all differ: a K or V mapped to the wrong cache (or a layer to the wrong layer) would
        # export and import consistently with itself, and every equality below would still hold
        comps = self.drafter_part(full, 300).reshape(10, 300 * self.ROW)
        self.assertEqual(len({c.tobytes() for c in comps}), 10)
        self.k.prefill(other)                          # the drafter's rows now belong to a different session
        self.assertFalse(np.array_equal(self.drafter_part(self.k.export(0, 300), 300), self.drafter_part(full, 300)))
        self.k.reset()
        self.k.lib.nomos_dflash_reset(self.k.h)
        self.assertEqual(self.k.import_(0, 150, first, 150, a[:150]), 0)
        self.assertEqual(self.k.dflash_len(), 150)
        self.assertEqual(self.k.import_(150, 300, second, 150, a[150:]), 0)
        self.assertEqual(self.k.dflash_len(), 300)
        self.assertTrue(np.array_equal(self.k.export(0, 300), full))

    def test_a_partial_chunk_import_takes_the_drafter_rows_of_its_prefix(self):
        """The store imports a prefix of a chunk when a request leaves the session inside it: here [0, 120) of a
        300-token chunk. Every component, the drafter's included, is laid out for the chunk's 300 tokens, so the
        drafter components are found by the chunk's length and not by the prefix's."""
        ids = doc_tokens(300)
        other = [BOS] + doc_tokens(700)[401:]          # 300 different tokens
        self.k.prefill(ids)
        ref, full = self.k.export(0, 120), self.k.export(0, 300)
        self.k.prefill(other)                          # the drafter's rows now belong to a different session
        self.k.reset()
        self.k.lib.nomos_dflash_reset(self.k.h)
        self.assertEqual(self.k.import_(0, 120, full, 300, ids[:120]), 0)
        self.assertEqual(self.k.dflash_len(), 120)
        self.assertTrue(np.array_equal(self.k.export(0, 120), ref))

    def test_a_drafter_with_a_gap_imports_nothing(self):
        """Drafter rows are copied only onto a drafter that holds exactly [0, a): rows after a gap would be read as
        contiguous context. The target's KV is restored either way."""
        ids = doc_tokens(300)
        self.k.prefill(ids)
        first, second = self.k.export(0, 150), self.k.export(150, 300)
        self.k.reset()
        self.k.lib.nomos_dflash_reset(self.k.h)
        self.assertEqual(self.k.import_(0, 150, first, 150, ids[:150]), 0)
        self.assertEqual(self.k.dflash_len(), 150)
        self.k.lib.nomos_dflash_reset(self.k.h)        # the drafter loses its context, the target keeps its KV
        self.assertEqual(self.k.import_(150, 300, second, 150, ids[150:]), 0)
        self.assertEqual(self.k.cache_len(), 300)
        self.assertEqual(self.k.live_tokens(), ids)
        self.assertEqual(self.k.dflash_len(), 0)

    def test_export_holds_drafter_rows_only_up_to_the_drafter_length(self):
        """A base-path turn decodes through decode_step, which does not feed the drafter, so the target gets ahead of
        it. A chunk then carries drafter rows for what the drafter has, and leaves the rest of each drafter component
        as the caller had it (zero here)."""
        self.k.prefill(doc_tokens(340))                # leaves real rows 300..339 in the drafter's cache, beyond what follows
        lg = self.k.prefill(doc_tokens(300))
        ref = self.k.export(0, 300)
        self.k.greedy(lg, 6)
        n = self.k.cache_len()
        self.assertGreater(n, 300)
        self.assertEqual(self.k.dflash_len(), 300)
        part = self.k.export(0, n)
        self.assertEqual(len(ref), 300 * (EXPECTED_BPT + 40960))     # else the slices below would be empty
        self.assertEqual(len(part), n * (EXPECTED_BPT + 40960))
        for j in range(10):
            lo, ref_lo = n * EXPECTED_BPT + j * n * self.ROW, 300 * EXPECTED_BPT + j * 300 * self.ROW
            comp, want = part[lo: lo + n * self.ROW], ref[ref_lo: ref_lo + 300 * self.ROW]
            self.assertTrue(want.any(), f"drafter component {j}: its rows must not be all zero")
            self.assertTrue(np.array_equal(comp[: 300 * self.ROW], want), f"drafter component {j}: its 300 rows")
            self.assertFalse(comp[300 * self.ROW:].any(), f"drafter component {j}: nothing past row 300")

    def test_truncate_never_lengthens_the_drafter(self):
        self.k.greedy(self.k.prefill(doc_tokens(200)), 6)    # the target is now ahead of the drafter
        n = self.k.cache_len()
        self.assertGreater(n, 201)
        self.assertEqual(self.k.dflash_len(), 200)
        self.assertEqual(self.k.truncate(n - 1), 0)          # a cut above the drafter's length leaves it alone
        self.assertEqual(self.k.cache_len(), n - 1)
        self.assertEqual(self.k.dflash_len(), 200)


class ContinuationFeedsDrafterTest(unittest.TestCase):
    """Spec test 7. A prefix + continuation prefill is not bit-identical to one batched prefill (ledger R18), so
    acceptance is compared loosely (plan notes 2 and 14); a dead drafter would sit near E = 1 and fail. Both runs
    must be lossless: base greedy from the same state gives the same tokens."""

    @classmethod
    def setUpClass(cls):
        cls.k = kernel()

    # -- beyond the brief's one: the gates and shapes its single scenario cannot tell apart from a bare "always feed" ------
    def test_a_continuation_before_the_store_exists_leaves_the_drafter_alone(self):
        """The store gate. The drafter is in step here, but the store does not exist yet (the process's first
        nomos_session_prepare creates it), so the continuation behaves as it did before this task. It needs a process
        that has not prepared yet: unittest runs this class, and in it this test, before anything else prepares."""
        if self.k.stats()["prepares"] > 0:
            self.skipTest("the store already exists in this process: run this file whole, and this test first")
        ids = doc_tokens(700)
        self.k.prefill(ids[:300])
        self.assertEqual(self.k.dflash_len(), 300)
        self.k.prefill_cont(ids[300:600])
        self.assertEqual(self.k.cache_len(), 600)
        self.assertEqual(self.k.dflash_len(), 300)

    def test_suffix_reaches_the_drafter(self):
        ids = doc_tokens(1200)[600:]
        toks_fresh, acc_fresh = self.k.spec_trace(self.k.prefill(ids), 96)
        base_fresh = self.k.greedy(self.k.prefill(ids), 96)
        self.k.prepare(ids[:300], spec=True)            # creates the store, which turns on continuation capture
        self.k.prefill(ids[:300])
        lg = self.k.prefill_cont(ids[300:])
        self.assertEqual(self.k.dflash_len(), self.k.cache_len())
        toks_split, acc_split = self.k.spec_trace(lg, 96)
        self.assertEqual(self.k.truncate(300), 0)       # the same state again, for the base-greedy comparison
        base_split = self.k.greedy(self.k.prefill_cont(ids[300:]), 96)
        self.assertEqual(toks_fresh, base_fresh)
        self.assertEqual(toks_split, base_split)
        e_fresh, e_split = 1 + np.mean(acc_fresh), 1 + np.mean(acc_split)
        print(f"test 7: E fresh {e_fresh:.2f}, E split {e_split:.2f}", flush=True)
        self.assertGreater(e_split, 1.5, "the drafter must see the suffix")
        # Batched prefill is not batch-shape invariant on this kernel (ledger R18), so the split run's taps differ a
        # little from the fresh run's. Compare loosely. The bar catches a drafter that missed the suffix or holds no
        # context (E near 1). It does not catch rows of the right length with the wrong content (another text's rows
        # pass it): test_the_last_token_of_a_suffix_feeds_its_own_row pins the content.
        self.assertGreater(e_split, 0.75 * e_fresh)

    def test_a_drafter_behind_the_kv_is_left_alone(self):
        """The in-step gate. A base-path turn decodes through decode_step, which does not feed the drafter, so the target
        gets ahead of it. The drafter appends at its own cursor, so rows added now would land at the wrong positions:
        the continuation extends the target and leaves the drafter exactly where it was."""
        ids = doc_tokens(700)
        self.k.prepare(ids[:300])
        self.assertGreater(self.k.stats()["prepares"], 0, "the store must exist, or the missing store is the gate")
        self.k.greedy(self.k.prefill(ids[:300]), 6)
        ahead = self.k.cache_len()
        self.assertGreater(ahead, 300)
        self.assertEqual(self.k.dflash_len(), 300)
        self.k.prefill_cont(ids[400:440])
        self.assertEqual(self.k.cache_len(), ahead + 40)
        self.assertEqual(self.k.dflash_len(), 300)

    def test_the_lengths_stay_equal_for_every_suffix_shape(self):
        """The host sends one token when a prompt equals a cached session (plan Review Focus 5); a suffix that starts with
        BOS loses it in the KV (engine_decode's strip), so the batched part is shorter than the suffix; a long suffix
        may be prefilled in chunks (NOMOS_PREFILL_CHUNK, Task 16). The drafter gains exactly as many rows as the KV."""
        ids = doc_tokens(900)
        self.k.prepare(ids[:300], spec=True)
        self.k.prefill(ids[:300])
        shapes = (  # (name, suffix, tokens the KV gains, NOMOS_PREFILL_CHUNK)
            ("one token", ids[300:301], 1, None), ("two tokens", ids[301:303], 2, None),
            ("BOS and 40", [BOS] + ids[303:343], 40, None), ("BOS and one", [BOS, ids[343]], 1, None),
            ("a lone BOS", [BOS], 1, None), ("in chunks of 64", ids[344:644], 300, "64"))
        saved = os.environ.get("NOMOS_PREFILL_CHUNK")
        try:
            for name, suffix, grew, chunk in shapes:
                if chunk is None:
                    os.environ.pop("NOMOS_PREFILL_CHUNK", None)
                else:
                    os.environ["NOMOS_PREFILL_CHUNK"] = chunk
                before = self.k.cache_len()
                self.k.prefill_cont(suffix)
                self.assertEqual(self.k.cache_len(), before + grew, name)
                self.assertEqual(self.k.dflash_len(), self.k.cache_len(), name)
        finally:
            if saved is None:
                os.environ.pop("NOMOS_PREFILL_CHUNK", None)
            else:
                os.environ["NOMOS_PREFILL_CHUNK"] = saved

    def test_the_last_token_of_a_suffix_feeds_its_own_row(self):
        """The content of the rows a continuation feeds. A suffix's first n-1 rows come from the batched prefill and its last
        from the drafter's one-row sink. A token run again alone on the same KV goes through the decode path: for the last
        token that is the sink again, and for a token inside the batched part it gives the row the batch wrote, bit for bit
        on this kernel (INT8 and INT4 block-32; an offset prefill attends token by token. If a kernel change makes the
        batched rows depend on the batch shape, compare by cosine instead). So each row rebuilt by a one-token continuation
        must equal the row it replaces: 599 (the sink), 450 (inside the suffix) and 300 (its first). Every export is taken
        before any row is rebuilt, and the rows before the rebuilt one must stay as they were. The E bar of test 7 cannot
        see this: rows of the right length holding another text's taps pass it."""
        ids = doc_tokens(700)
        self.k.prepare(ids[:300], spec=True)
        self.k.prefill(ids[:300])
        self.k.prefill_cont(ids[300:600])
        self.assertEqual(self.k.dflash_len(), 600)
        probes = (599, 450, 300)                        # descending: each cut is within what the last rebuild left
        before = {p: self.k.export(0, p + 1) for p in probes}
        for p in probes:
            self.assertEqual(self.k.truncate(p), 0)
            self.assertEqual(self.k.dflash_len(), p)
            self.k.prefill_cont(ids[p:p + 1])
            self.assertEqual(self.k.dflash_len(), p + 1)
            after = self.k.export(0, p + 1)
            if p == 599:                                # the same decode path: the target's KV is rebuilt exactly too
                self.assertTrue(np.array_equal(after, before[p]), f"row {p}: the whole export")
            lo = (p + 1) * EXPECTED_BPT                 # the drafter's ten components follow the target's
            self.assertTrue(np.array_equal(after[lo:], before[p][lo:]), f"drafter rows 0..{p}")

    def test_a_suffix_the_batched_route_did_not_feed_leaves_the_drafter_trailing(self):
        """The sink row belongs at cache_len - 1, so it is appended only when the batched part left the drafter exactly one
        row behind. Without the batched route (NOMOS_BATCHED_PREFILL=0 here; a tap-count mismatch does the same, with a
        warning) the suffix runs token by token and feeds no batched rows: the drafter must trail, its length unchanged,
        and not hold the last token's taps at the cursor, where they would be read as an earlier token's."""
        ids = doc_tokens(700)
        self.k.prepare(ids[:300], spec=True)
        self.k.prefill(ids[:300])
        self.assertEqual((self.k.cache_len(), self.k.dflash_len()), (300, 300))
        saved = os.environ.get("NOMOS_BATCHED_PREFILL")
        os.environ["NOMOS_BATCHED_PREFILL"] = "0"
        try:
            self.k.prefill_cont(ids[300:340])
        finally:
            if saved is None:
                os.environ.pop("NOMOS_BATCHED_PREFILL", None)
            else:
                os.environ["NOMOS_BATCHED_PREFILL"] = saved
        self.assertEqual(self.k.cache_len(), 340)
        self.assertEqual(self.k.dflash_len(), 300)


class SpecLoopOnStoreTest(unittest.TestCase):
    """Spec tests 8 and 9, through pyhost's own loop (host.KernelLM.spec_generate_stream)."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT / "pyhost"))
        from host import KernelLM
        k = kernel()
        cls.lm = KernelLM.__new__(KernelLM)            # reuse the loaded engine: one Glimmer per process
        cls.lm.lib, cls.lm.h, cls.lm.vocab_size = k.lib, k.h, 202048
        cls.lm._logits = np.zeros(202048, dtype=np.float32)
        cls.lm._dflash_loaded = True
        cls.lm._session_ok = True                      # __new__ skips __init__, which sets this from the kernel's exports
        # ... and which types the exports the host calls: here the testlib's SIGNATURES do. An export it lacks is called
        # untyped, which truncates the 64-bit handle to an int and crashes the process instead of failing the test.
        for name in ("nomos_dflash_cache_len", "nomos_dflash_set_len", "nomos_kv_set_len"):
            assert getattr(k.lib, name).argtypes is not None, f"kvsession_testlib.SIGNATURES does not type {name}"
        cls.k = k

    def _spec(self, ids, n=48):
        trace = []
        toks = list(self.lm.spec_generate_stream(ids, max_new_tokens=n, stop_ids=STOP, vb=8, trace=trace))
        return toks, trace

    def test_park_resume_with_dflash_matches_never_parked(self):    # spec test 8
        p1 = doc_tokens(2000)[1200:1700]
        reply1, _ = self._spec(p1)
        t3 = p1 + reply1 + doc_tokens(2040)[2000:]
        self._spec(t3)                                               # the conversation runs on past t3
        # Never parked: the same request again on the live engine. prepare cuts it to len(t3) - 1, then a 1-token
        # continuation runs: the exact op sequence the resume below runs on restored bytes.
        ref, ref_trace = self._spec(t3)
        self._spec(doc_tokens(2600)[2100:])                          # another conversation parks it
        before = self.k.stats()
        got, got_trace = self._spec(t3)                              # restored from the store
        after = self.k.stats()
        self.assertGreater(after["restores_ram"] + after["restores_nvme"],
                           before["restores_ram"] + before["restores_nvme"])
        self.assertEqual(got, ref)
        self.assertEqual(got_trace, ref_trace)
        # Lossless: base greedy from the same restored state gives the same tokens.
        self.assertEqual(self.k.truncate(len(t3) - 1), 0)
        self.assertEqual(self.k.greedy(self.k.prefill_cont(t3[-1:]), len(got)), got)
        # Beyond the brief. Two empty traces are equal, and a turn that fell back to base decode (ledger R53) is lossless
        # too, so nothing above shows that these turns speculated. The trace must be non-empty and the loop's own: the
        # testlib's loop, run from the same state, accepts the same number of drafts in every cycle.
        self.assertTrue(got_trace, "an empty trace: the turn decoded on the base path instead of speculating")
        self.assertEqual(self.k.truncate(len(t3) - 1), 0)
        self.assertEqual(self.k.spec_trace(self.k.prefill_cont(t3[-1:]), len(got)), (got, got_trace))

    def test_base_turn_in_the_middle(self):                          # spec test 9
        p1 = doc_tokens(3200)[2700:3100]
        reply1, _ = self._spec(p1)
        t2 = p1 + reply1 + doc_tokens(3230)[3200:]
        reply2 = self.lm.generate(t2, max_new_tokens=24, stop_ids=STOP)   # base path: its decode skips the drafter
        drafter_after_2 = self.k.dflash_len()
        self.assertLessEqual(drafter_after_2, len(t2))
        self.assertGreater(self.k.cache_len(), drafter_after_2)
        t3 = t2 + reply2 + doc_tokens(3260)[3230:]
        before = self.k.stats()
        spec_out, trace = self._spec(t3, 32)
        reused = self.k.stats()["reused_tokens"] - before["reused_tokens"]
        self.assertEqual(reused, drafter_after_2)    # resumed where the drafter stopped; only the base turn's
                                                     # decoded tokens are prefilled again
        self.assertTrue(trace, "an empty trace: the continuation left the drafter behind and the turn did not speculate")
        # Lossless: base greedy from the same state (same prefill split) gives the same tokens.
        self.assertEqual(self.k.truncate(drafter_after_2), 0)
        self.assertEqual(self.k.greedy(self.k.prefill_cont(t3[drafter_after_2:]), len(spec_out)), spec_out)

    def test_a_drafter_behind_the_kv_decodes_the_turn_on_the_base_path(self):    # ledger R53
        """The turn keeps the KV the store restored and decodes greedily, without speculating, instead of prefilling the
        whole context again. A continuation of more than one token that runs without the batched part
        (NOMOS_BATCHED_PREFILL=0) feeds the drafter nothing (R52), so the drafter trails the KV by the whole suffix."""
        p1 = doc_tokens(4400)[3900:4300]
        reply1, _ = self._spec(p1)
        t2 = p1 + reply1 + doc_tokens(4440)[4400:]                   # 40 tokens beyond the cached conversation
        calls, saved = [], os.environ.get("NOMOS_BATCHED_PREFILL")
        for name in ("prefill", "prefill_cont"):                     # record which one the turn calls, and with how much
            real = getattr(self.lm, name)
            setattr(self.lm, name, lambda ids, real=real, name=name: (calls.append((name, len(ids))), real(ids))[1])
        os.environ["NOMOS_BATCHED_PREFILL"] = "0"
        try:
            before, trace = self.k.stats(), []
            with contextlib.redirect_stdout(io.StringIO()) as log:
                g = self.lm.spec_generate_stream(t2, max_new_tokens=24, stop_ids=STOP, vb=8, trace=trace)
                out = [next(g)]                                      # prepare and the continuation have run
                disk = self.k.stats()["nvme_bytes"]                  # (prepare saves a parked session's tail: count from here)
                out += list(g)
            after = self.k.stats()
            c = after["reused_tokens"] - before["reused_tokens"]     # what prepare gave the turn: the cached prefix
            self.assertGreater(len(t2) - c, 1)                       # a suffix of one token is fed by the sink row alone
            self.assertEqual(self.k.dflash_len(), c)                 # the drafter was left where the store put it
            self.assertEqual(self.k.cache_len(), len(t2) + len(out) - 1)    # the KV decoded on: every token but the last
            # Base greedy from the same state: restore the same prefix, run the same continuation, decode.
            self.assertEqual(self.k.truncate(c), 0)
            ref = self.k.greedy(self.k.prefill_cont(t2[c:]), len(out))
        finally:
            for name in ("prefill", "prefill_cont"):
                delattr(self.lm, name)
            if saved is None:
                os.environ.pop("NOMOS_BATCHED_PREFILL", None)
            else:
                os.environ["NOMOS_BATCHED_PREFILL"] = saved
        self.assertEqual(len(out), 24)
        self.assertEqual(out, ref)                                   # exactly what greedy base decode gives
        self.assertEqual(trace, [])                                  # no speculative cycle ran
        self.assertIn("[kvstore] drafter behind the KV; this turn decodes without speculation", log.getvalue())
        self.assertEqual(calls, [("prefill_cont", len(t2) - c)])     # the restored KV was kept: one continuation, no prefill
        self.assertGreater(after["nvme_bytes"], disk, "the turn was not committed")

    def test_a_turn_that_ends_in_the_exact_step_does_not_make_the_next_turn_fork(self):
        """The loop's last token, when one is left in the budget, comes from an exact M=1 step: the decode path, which
        captures no taps for the drafter. If that step's KV row stayed, the turn would end with the drafter one row short
        of the KV. prepare then caps the next speculative request at the drafter's length, finds a gap in the session's
        drafter rows and forks it, and the fork's commit rewrites the whole conversation (the serve at temperature 0
        showed it: turn 5 of the six-turn test forked). A budget of 2 takes the step every time: the seed, then one
        token left."""
        p1 = doc_tokens(6400)[6100:6400]
        trace = []
        g = self.lm.spec_generate_stream(p1, max_new_tokens=2, stop_ids=STOP, vb=8, trace=trace)
        reply1 = list(itertools.islice(g, 2))                        # the seed and the exact step; suspended on the latter
        self.assertEqual((len(reply1), trace), (2, []))              # no speculative cycle
        with self.subTest("the step's row is gone before its token is handed out"):    # (the settle only acts when it ends)
            self.assertEqual(self.k.dflash_len(), self.k.cache_len())
        g.close()
        with self.subTest("the turn ends with the drafter in step"):
            self.assertEqual(self.k.dflash_len(), self.k.cache_len())
        forks = self.k.stats()["forks"]
        t2 = p1 + reply1 + doc_tokens(6440)[6400:]
        _, trace2 = self._spec(t2, 24)
        self.assertTrue(trace2, "the next turn must speculate")
        with self.subTest("the next turn does not fork"):
            self.assertEqual(self.k.stats()["forks"], forks, "the next turn forked: the conversation was rewritten")

    # -- ruling R55: a turn ends as the base path's does -------------------------------------------------------------
    # A burst's accepted drafts are in the KV and the drafter before the first of them is yielded. A client that goes away after
    # the first token of a burst has seen one draft and sends exactly that back. If the turn left the whole burst behind, its
    # history diverges from the KV inside the burst: prepare truncates, finds the session no longer continued, and forks, and
    # the fork's commit rewrites the whole conversation (about 9 GB at 128K with the drafter's rows). The base path ends a turn
    # with every token handed out in the KV but the last; so must this one.

    def _past_a_burst(self, n0, first_occurrence=False):
        """A conversation whose next turn is traced once. The turn is a request that repeats: prepare cuts it to len(t) - 1 each
        time, so every repeat starts from the same bytes and hands out the same tokens. Returns (t, the tokens the traced turn
        handed out, how many are handed out through the first token of a burst of at least two accepted drafts: strictly
        inside it, with accepted drafts after that token which nobody has seen). first_occurrence: that token appears nowhere
        earlier in the turn."""
        p1 = doc_tokens(n0 + 300)[n0:n0 + 300]
        reply1, _ = self._spec(p1)
        t = p1 + reply1 + doc_tokens(n0 + 340)[n0 + 300:]
        self._spec(t, 64)                                            # the conversation runs on past t
        toks, trace = self._spec(t, 64)
        start = 1                                        # toks[0] is the seed; a cycle's burst is toks[start:start + acc + 1]
        for acc in trace:
            if acc >= 2 and start + acc < len(toks) and (not first_occurrence or toks.index(toks[start]) == start):
                return t, toks, start + 1
            start += acc + 1
        self.fail(f"no burst of two accepted drafts to cut in {len(toks)} tokens (trace {trace})")

    def _the_next_turn_continues(self, n0, t, got):
        """The client sends back t, what it saw, and a new message. Nothing may be rewritten; the loop must go on speculating."""
        kept = len(t) + len(got) - 1                     # the base path's state: every token handed out but the last
        with self.subTest("the KV and the drafter hold what the client saw, but the last token"):
            self.assertEqual((self.k.cache_len(), self.k.dflash_len()), (kept, kept))
        before = self.k.stats()
        _, trace = self._spec(t + got + doc_tokens(n0 + 380)[n0 + 340:], 24)
        after = self.k.stats()
        with self.subTest("the next turn continues the session"):
            self.assertEqual(after["forks"] - before["forks"], 0, "it forked: the conversation was rewritten")
            self.assertEqual(after["reused_tokens"] - before["reused_tokens"], kept)
        with self.subTest("the next turn speculates"):
            self.assertTrue(trace, "an empty trace: the drafter was left out of step and the turn decoded on the base path")

    def test_a_generator_closed_inside_a_burst_leaves_nothing_the_client_did_not_see(self):
        t, ref, cut = self._past_a_burst(7000)
        g = self.lm.spec_generate_stream(t, max_new_tokens=64, stop_ids=STOP, vb=8)
        got = list(itertools.islice(g, cut))
        self.assertEqual(got, ref[:cut])                             # the same turn as the traced one
        g.close()
        self._the_next_turn_continues(7000, t, got)

    def test_a_stop_token_accepted_as_a_draft_leaves_nothing_after_it(self):
        t, ref, cut = self._past_a_burst(7500, first_occurrence=True)
        got = list(self.lm.spec_generate_stream(t, max_new_tokens=64, stop_ids=STOP + (ref[cut - 1],), vb=8))
        self.assertEqual(got, ref[:cut])                             # it ends on the burst's first token, a draft
        self._the_next_turn_continues(7500, t, got)

    def test_a_turn_is_committed_the_moment_its_generator_ends(self):    # Review Focus 1, on the speculative path
        """However the generator ends, the turn is on disk by then. Counted from after the first token, because prepare
        (which has run by then) saves the previous session's tail when it parks it: bytes written during a whole turn
        can come from the turn before, whatever became of this turn's commit. The client that goes away mid-stream is
        the case the serve cares about."""
        for name, ids, pulls in (("runs to its end", doc_tokens(5000)[4700:5000], None),
                                 ("closed after its first token", doc_tokens(5400)[5100:5400], 1),
                                 ("closed in the middle of a burst", doc_tokens(5800)[5500:5800], 14)):
            with self.subTest(name):
                g = self.lm.spec_generate_stream(ids, max_new_tokens=48, stop_ids=STOP, vb=8)
                got = [next(g)]                                      # prepare has run
                before = self.k.stats()["nvme_bytes"]
                got += list(itertools.islice(g, None if pulls is None else pulls - 1))
                if pulls is not None:
                    self.assertEqual(len(got), pulls)                # the generator is suspended, not finished
                    g.close()
                self.assertGreater(self.k.stats()["nvme_bytes"], before, "the turn was not committed")

if __name__ == "__main__":
    unittest.main()
