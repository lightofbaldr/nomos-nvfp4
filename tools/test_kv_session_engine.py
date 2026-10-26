#!/usr/bin/env python3
"""Engine primitives for the KV session store (Tasks 4-5). GPU; run alone:
pixi run --no-install python3 tools/test_kv_session_engine.py -v"""
import ctypes
import unittest

import numpy as np

from kvsession_testlib import BOS, EXPECTED_BPT, VOCAB, Kernel, doc_tokens

_K = None


def kernel():
    """One Glimmer per process, shared by every class here: the GPU holds one."""
    global _K
    if _K is None:
        _K = Kernel()
    return _K


_DFLASH = False


def kernel_with_dflash():
    """The same Glimmer with the DFlash drafter loaded, once, and left loaded (a second load leaks ~3.3 GB)."""
    global _DFLASH
    k = kernel()
    if not _DFLASH:
        k.dflash_load()
        _DFLASH = True
    return k


def untracked_write(k, ids, n_new=6):
    """The Go-era nomos_generate: resets, then writes KV through a path that does not track tokens.
    Returns the generated tokens."""
    p, out = np.asarray(ids, dtype=np.int32), np.zeros(16, dtype=np.int32)
    n = k.lib.nomos_generate(k.h, p.ctypes.data, len(p), n_new, out.ctypes.data, len(out), 0.0, 1.0, 1.0)
    assert n > 0, f"nomos_generate rc={n}"
    return out[:n].tolist()


class LiveTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = kernel()

    def test_prefill_records_the_prompt(self):
        ids = doc_tokens(40)
        self.k.prefill(ids)
        self.assertEqual(self.k.live_tokens(), ids)
        self.assertEqual(self.k.cache_len(), 40)

    def test_step_appends(self):
        ids = doc_tokens(20)
        t = int(np.argmax(self.k.prefill(ids)))
        self.k.step(t)
        self.assertEqual(self.k.live_tokens(), ids + [t])

    def test_fast_step_appends(self):
        ids = doc_tokens(20)
        t = int(np.argmax(self.k.prefill(ids)))
        self.k.step_token(t)
        self.assertEqual(self.k.live_tokens(), ids + [t])
        self.assertEqual(self.k.cache_len(), 21)

    def test_prefill_cont_appends(self):
        ids = doc_tokens(60)
        self.k.prefill(ids[:30])
        self.k.prefill_cont(ids[30:])
        self.assertEqual(self.k.live_tokens(), ids)
        self.assertEqual(self.k.cache_len(), 60)

    def test_prefill_cont_mirrors_the_bos_strip(self):
        ids = doc_tokens(30)
        self.k.prefill(ids[:20])
        self.k.prefill_cont([BOS] + ids[20:])
        self.assertEqual(self.k.live_tokens(), ids)
        self.assertEqual(self.k.cache_len(), 30)

    def test_set_len_truncates_and_refuses_growth(self):
        ids = doc_tokens(50)
        self.k.prefill(ids)
        self.assertEqual(self.k.set_len(20), 0)
        self.assertEqual(self.k.live_tokens(), ids[:20])
        self.assertEqual(self.k.set_len(21), -2)
        self.assertEqual(self.k.cache_len(), 20)

    def test_reset_clears(self):
        self.k.prefill(doc_tokens(10))
        self.k.reset()
        self.assertEqual(self.k.live_tokens(), [])
        self.assertEqual(self.k.cache_len(), 0)

    def test_prefill_cont_single_bos_is_not_stripped(self):
        # engine_decode strips a leading BOS only from a MULTI-token continuation (num_new > 1). A lone [BOS] is an
        # ordinary appended token: it has KV, so live_tokens must hold it.
        ids = doc_tokens(30)
        self.k.prefill(ids[:20])
        self.k.prefill_cont([BOS])
        self.assertEqual(self.k.live_tokens(), ids[:20] + [BOS])
        self.assertEqual(self.k.cache_len(), 21)

    def test_untracked_write_is_never_trusted(self):
        # Review Focus 2: KV written by a path that does not track tokens (the Go-era nomos_generate here) must
        # never be described by live_tokens, not even after later tracked steps and a set_len back down. Equal
        # lengths must mean the tokens describe the KV.
        ids = doc_tokens(40)
        gen = untracked_write(self.k, ids)
        self.assertNotEqual(len(self.k.live_tokens()), self.k.cache_len())
        self.k.step(gen[0])
        self.assertEqual(self.k.set_len(1), 0)
        # A tracker that appended gen[0] to the (empty) list would have length 1 == cache_len 1 here, over a KV row
        # that was written from ids[0].
        self.assertNotEqual(len(self.k.live_tokens()), self.k.cache_len())
        self.k.reset()
        self.k.prefill(ids)
        self.assertEqual(self.k.live_tokens(), ids)

    def test_untracked_verify_is_never_trusted(self):
        # The same guarantee for a writer that reaches prefill_batch_impl (MTP / Eagle verify): it adds rows at
        # start_pos == cache_len and the host rolls back later. Tracked steps in between must not re-fill the list.
        ids = doc_tokens(34)
        lib = self.k.lib
        lib.nomos_mtp_verify.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64]
        lib.nomos_mtp_verify.restype = ctypes.c_int32
        rows, logits = np.asarray(ids[30:34], dtype=np.int32), np.zeros((8, VOCAB), dtype=np.float32)
        self.k.prefill(ids[:30])
        self.assertEqual(lib.nomos_mtp_verify(self.k.h, rows.ctypes.data, 4, 30, logits.ctypes.data), 0)
        self.assertNotEqual(len(self.k.live_tokens()), self.k.cache_len())
        self.k.step(ids[1])
        self.assertEqual(self.k.set_len(31), 0)
        self.assertNotEqual(len(self.k.live_tokens()), self.k.cache_len())
        # A rewind: the verify rewrites rows below the tracked length and ends exactly on it (22 + 8 == 30). The
        # list must be EMPTY, not a stale list that happens to have the right length.
        rows = np.asarray(doc_tokens(60)[40:48], dtype=np.int32)
        self.k.prefill(ids[:30])
        self.assertEqual(lib.nomos_mtp_verify(self.k.h, rows.ctypes.data, 8, 22, logits.ctypes.data), 0)
        self.assertEqual(self.k.cache_len(), 30)
        self.assertEqual(self.k.live_tokens(), [])

    def test_reset_and_empty_set_len_restore_tracking(self):
        # After an untracked write only a reset, a fresh prefill or set_len(0) trust the KV again.
        ids = doc_tokens(40)
        for restore in (self.k.reset, lambda: self.k.set_len(0)):
            gen = untracked_write(self.k, ids)
            restore()
            self.k.step(gen[0])
            self.assertEqual(self.k.live_tokens(), [gen[0]])
            self.assertEqual(self.k.cache_len(), 1)

    def test_tracking_survives_a_mixed_turn(self):
        # A turn the way the serve runs it: prefill, decode through both step paths, roll back, continue. Every
        # tracked call must hand the state to the next one still tracked.
        ids = doc_tokens(60)
        t1 = int(np.argmax(self.k.prefill(ids[:30])))
        t2 = int(np.argmax(self.k.step(t1)))
        t3 = self.k.step_token(t2)
        self.k.step(t3)
        self.assertEqual(self.k.live_tokens(), ids[:30] + [t1, t2, t3])
        self.assertEqual(self.k.set_len(32), 0)
        self.k.prefill_cont(ids[32:40])
        self.assertEqual(self.k.live_tokens(), ids[:30] + [t1, t2] + ids[32:40])
        self.assertEqual(self.k.cache_len(), 40)

    def test_multimodal_prefill_is_never_matched(self):
        # KV spliced from image features is not described by token IDs, so nomos_prefill_mm leaves the list empty
        # (and the state untracked) even though the prompt IS a list of token IDs.
        dim, img, n_vis = 6656, 200092, 4        # Muse: model dim and <|image|> id (lib/model_config.mojo)
        ids = doc_tokens(30)
        prompt = np.asarray(ids[:10] + [img] * n_vis + ids[10:16], dtype=np.int32)
        feats = (np.random.default_rng(0).standard_normal((n_vis, dim)) * 0.02).astype(np.float32)
        lib = self.k.lib
        lib.nomos_prefill_mm.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64,
                                         ctypes.c_int32, ctypes.c_int64]
        lib.nomos_prefill_mm.restype = ctypes.c_int32
        self.k.prefill(ids)
        rc = lib.nomos_prefill_mm(self.k.h, prompt.ctypes.data, len(prompt), feats.ctypes.data, n_vis,
                                  self.k.logits.ctypes.data)
        self.assertEqual(rc, 0)
        self.assertEqual(self.k.cache_len(), len(prompt))
        self.assertEqual(self.k.live_tokens(), [])
        self.k.step(ids[1])
        self.assertEqual(self.k.live_tokens(), [])


class SpecLiveTokensTest(unittest.TestCase):
    """DFlash speculative decode: verify writes k+1 rows at once and the host rolls back to the accepted prefix."""

    @classmethod
    def setUpClass(cls):
        cls.k = kernel_with_dflash()

    def test_spec_decode_keeps_live_tokens_in_step(self):
        ids = doc_tokens(48)
        toks, acc = self.k.spec_trace(self.k.prefill(ids), 32)
        live = self.k.live_tokens()
        self.assertTrue(acc, "no verify cycle ran")
        self.assertEqual(len(live), self.k.cache_len())
        self.assertEqual(live[: len(ids)], ids)
        # The KV holds every emitted token except the last (a bonus token is only fed by the next cycle); after an
        # early return it also holds accepted drafts beyond the emitted list. Either way the emitted tokens that
        # have KV are, in order, what live_tokens says.
        k = min(len(toks), len(live) - len(ids))
        self.assertGreaterEqual(k, len(toks) - 1)
        self.assertEqual(live[len(ids): len(ids) + k], toks[:k])

    def test_dflash_verify_on_untracked_kv_stays_untracked(self):
        # Verify rows written on top of KV the tracker never saw (nomos_generate here), or past a gap in it, are
        # described by nothing: the list stays empty.
        ids = doc_tokens(48)
        lib, rows = self.k.lib, np.asarray(ids[40:44], dtype=np.int32)
        logits = np.zeros((4, VOCAB), dtype=np.float32)
        untracked_write(self.k, ids)
        c = self.k.cache_len()
        self.assertEqual(lib.nomos_dflash_verify_fused(self.k.h, rows.ctypes.data, 4, c, logits.ctypes.data), 0)
        self.assertEqual(self.k.live_tokens(), [])
        self.assertEqual(self.k.cache_len(), c + 4)
        self.k.prefill(ids)                        # tracked again; now leave a 3-row gap
        c = self.k.cache_len()
        self.assertEqual(lib.nomos_dflash_verify_fused(self.k.h, rows.ctypes.data, 4, c + 3, logits.ctypes.data), 0)
        self.assertEqual(self.k.live_tokens(), [])
        self.assertEqual(self.k.cache_len(), c + 3 + 4)


class SessionPrimitivesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = kernel()

    def test_bytes_per_token_match_the_kv_mode(self):
        self.assertEqual(self.k.bytes_per_token(), EXPECTED_BPT)

    def test_round_trip_is_byte_identical(self):
        ids = doc_tokens(300)
        self.k.prefill(ids)
        full = self.k.export(0, 300)
        self.k.reset()
        self.assertEqual(self.k.import_(0, 300, full, 300, ids), 0)
        self.assertEqual(self.k.cache_len(), 300)
        self.assertEqual(self.k.live_tokens(), ids)
        self.assertTrue(np.array_equal(self.k.export(0, 300), full))

    def test_resume_continues_bit_identically(self):
        """Spec test 2. The never-parked reference runs the SAME op sequence as the resume (a 399-token prefill,
        then a 1-token continuation), so the only difference is the round trip through host memory and any
        mismatch is the store's. (One 400-token batched prefill is a different computation; see test 3.)"""
        ids = doc_tokens(400)
        self.k.prefill(ids[:399])
        ref_logits = self.k.prefill_cont(ids[399:])
        ref = self.k.greedy(ref_logits, 64)
        self.k.prefill(ids[:399])
        chunk = self.k.export(0, 399)
        self.k.reset()
        self.assertEqual(self.k.import_(0, 399, chunk, 399, ids[:399]), 0)
        lg = self.k.prefill_cont(ids[399:])
        self.assertTrue(np.array_equal(lg, ref_logits), "restored KV must give bit-identical logits")
        self.assertEqual(self.k.greedy(lg, 64), ref)

    def test_truncate_then_continue_matches_fresh_prefill(self):
        """Spec test 3: truncate_to(p) then continue vs a fresh prefill(ids[:p]) then continue. The live session got
        its [0, p) from that same prefill(ids[:p]) and then ran on past p, so the prefix bytes are the same and the
        match is exact. It cannot be built from a longer batched prefill: the W4A4 GEMMs round by batch shape and
        the difference compounds with depth. A 500-token prefill's KV[0, 200) differs from a 200-token prefill's in
        45% (INT4) to 69% (INT8) of its bytes, the continuation's logits reach only cosine ~0.996, and greedy
        decoding has diverged within 13 tokens."""
        ids = doc_tokens(500)
        self.k.prefill(ids[:200])
        fresh = self.k.prefill_cont(ids[200:])
        ref = self.k.greedy(fresh, 32)
        self.k.prefill(ids[:200])
        self.k.prefill_cont(ids[200:350])   # the live session runs on past p
        self.assertEqual(self.k.truncate(200), 0)
        self.assertEqual(self.k.live_tokens(), ids[:200])
        lg = self.k.prefill_cont(ids[200:])
        self.assertTrue(np.array_equal(lg, fresh), "truncate must leave exactly the prefix: bit-identical logits")
        self.assertEqual(self.k.greedy(lg, 32), ref)

    def test_partial_chunk_import(self):
        ids = doc_tokens(300)
        self.k.prefill(ids)
        ref = self.k.export(0, 120)
        full = self.k.export(0, 300)
        self.k.reset()
        self.assertEqual(self.k.import_(0, 120, full, 300, ids[:120]), 0)
        self.assertTrue(np.array_equal(self.k.export(0, 120), ref))

    def test_restore_after_another_session_ran(self):
        """reset() zeroes the lengths, not the quantized KV, so a restore into untouched buffers cannot tell an
        import that copies from one that does nothing: the old bytes are still there. Park a session as two
        consecutive chunks (the second starts at a > 0), run a different session over the same buffers, then
        restore: the KV is byte-identical and the continuation bit-identical to never parking."""
        a = doc_tokens(300)
        other = [BOS] + doc_tokens(700)[401:]       # 300 different tokens
        self.k.prefill(a[:299])
        ref = self.k.prefill_cont(a[299:])
        self.k.prefill(a[:299])
        full = self.k.export(0, 299)
        first, second = self.k.export(0, 150), self.k.export(150, 299)
        self.k.reset()
        self.k.prefill(other)                       # the buffers now hold a different session's KV
        self.assertFalse(np.array_equal(self.k.export(0, 299), full))
        self.k.reset()
        self.assertEqual(self.k.import_(0, 150, first, 150, a[:150]), 0)
        self.assertEqual(self.k.import_(150, 299, second, 149, a[150:299]), 0)
        self.assertEqual(self.k.live_tokens(), a[:299])
        self.assertTrue(np.array_equal(self.k.export(0, 299), full))
        self.assertTrue(np.array_equal(self.k.prefill_cont(a[299:]), ref), "restored KV must give bit-identical logits")

    def test_import_refuses_a_gap(self):
        ids = doc_tokens(100)
        self.k.prefill(ids)
        chunk = self.k.export(0, 100)
        self.k.reset()
        self.assertLess(self.k.import_(10, 100, chunk, 100, ids[10:]), 0)


if __name__ == "__main__":
    unittest.main()
