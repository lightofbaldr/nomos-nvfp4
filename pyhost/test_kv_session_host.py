#!/usr/bin/env python3
"""_GenMixin, and KernelLM's speculative loop, drive the KV session store correctly (CPU only, fake backends)."""
import contextlib
import ctypes
import io
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import KernelLM, _GenMixin  # noqa: E402


class FakeBackend(_GenMixin):
    def __init__(self, cached):
        self.cached, self.calls = cached, []
        self.seen = []        # the token ids each prepare / prefill call was given (`calls` keeps only their number)

    def _logits(self, tok):
        lg = np.zeros(16, dtype=np.float32)
        lg[tok % 16] = 1.0
        return lg

    def session_prepare(self, ids, spec=False):
        self.calls.append(("prepare", len(ids)))
        self.seen.append(("prepare", tuple(ids), spec))
        return min(self.cached, len(ids) - 1)

    def prefill(self, ids):
        self.calls.append(("prefill", len(ids)))
        self.seen.append(("prefill", tuple(ids)))
        return self._logits(3)

    def prefill_cont(self, ids):
        self.calls.append(("prefill_cont", len(ids)))
        self.seen.append(("prefill_cont", tuple(ids)))
        return self._logits(3)

    def prefill_mm(self, ids, feats, nvis):
        self.calls.append(("prefill_mm", len(ids)))
        self.seen.append(("prefill_mm", tuple(ids)))
        return self._logits(3)

    def step(self, tok):
        self.calls.append(("step", tok))
        return self._logits(tok + 1)

    def session_commit(self):
        self.calls.append(("commit",))
        return 0


class GenMixinStoreTest(unittest.TestCase):
    def test_uncached_prompt_prefills_fully_then_commits(self):
        b = FakeBackend(cached=0)
        b.generate(list(range(10)), max_new_tokens=3)
        self.assertEqual(b.calls[:2], [("prepare", 10), ("prefill", 10)])
        self.assertEqual(b.calls[-1], ("commit",))

    def test_cached_prefix_prefills_only_the_rest(self):
        b = FakeBackend(cached=7)
        b.generate(list(range(10)), max_new_tokens=2)
        self.assertEqual(b.calls[1], ("prefill_cont", 3))

    def test_early_close_still_commits(self):          # Review Focus 1
        b = FakeBackend(cached=0)
        g = b.generate_stream(list(range(10)), max_new_tokens=50)
        next(g)
        g.close()                                       # the client went away mid-stream
        self.assertEqual(b.calls[-1], ("commit",))

    def test_multimodal_bypasses_the_store(self):
        b = FakeBackend(cached=5)
        b.generate(list(range(10)), max_new_tokens=1, mm_features=np.zeros(4), mm_nvis=1)
        self.assertNotIn(("commit",), b.calls)
        self.assertEqual(b.calls[0], ("prefill_mm", 10))

    def test_commit_failure_does_not_fail_the_turn(self):
        b = FakeBackend(cached=0)

        def boom():
            raise RuntimeError("engine daemon closed the connection")

        b.session_commit = boom
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(len(b.generate(list(range(10)), max_new_tokens=3)), 3)
        self.assertIn("[kvstore] commit failed: engine daemon closed the connection", out.getvalue())

    # The tests below are not in the plan's list. The first three pin WHICH tokens reach prefill, which the lengths in
    # `calls` cannot (a slice from the wrong end of the prompt has the same length); the last two pin the other half of
    # each pair above.
    def test_a_cached_prefix_is_skipped_and_the_rest_is_prefilled_whole(self):
        b = FakeBackend(cached=7)
        b.generate(list(range(10)), max_new_tokens=2)
        self.assertEqual(b.seen, [("prepare", tuple(range(10)), False), ("prefill_cont", (7, 8, 9))])

    def test_an_uncached_prompt_is_prefilled_token_for_token(self):
        b = FakeBackend(cached=0)
        b.generate(list(range(10)), max_new_tokens=2)
        self.assertEqual(b.seen, [("prepare", tuple(range(10)), False), ("prefill", tuple(range(10)))])

    def test_a_prompt_cached_but_for_its_last_token_prefills_that_one(self):     # Review Focus 5: c = n - 1
        b = FakeBackend(cached=99)
        b.generate(list(range(10)), max_new_tokens=2)
        self.assertEqual(b.seen[1], ("prefill_cont", (9,)))

    def test_generate_commits_even_when_a_step_fails(self):
        b = FakeBackend(cached=0)

        def boom(tok):
            raise RuntimeError("nomos_decode_step rc=-99")

        b.step = boom
        with self.assertRaises(RuntimeError):
            b.generate(list(range(10)), max_new_tokens=5)
        self.assertEqual(b.calls[-1], ("commit",))      # the turn's prefill is persisted; the error still propagates

    def test_multimodal_stream_bypasses_the_store(self):
        b = FakeBackend(cached=5)
        list(b.generate_stream(list(range(10)), max_new_tokens=2, mm_features=np.zeros(4), mm_nvis=1))
        self.assertNotIn(("commit",), b.calls)
        self.assertEqual(b.calls[0], ("prefill_mm", 10))


# -- the speculative loop (KernelLM.spec_generate_stream) on a toy engine ---------------------------------------------------
# The GPU tests (tools/test_kv_session_dflash.py) run the loop on the real engine. This pins its control flow, which the real
# engine cannot be made to take on demand: how a turn ends (a closed generator, an error), a drafter that trails the KV, a stop
# token, and the acceptance trace's values.

V = 40                      # the toy vocabulary


def nxt(t):
    return (5 * t + 3) % V  # the toy target's greedy next token after t


def chain(t, n):
    """The n tokens greedy decode emits after token t."""
    out = []
    for _ in range(n):
        t = nxt(t)
        out.append(t)
    return out


def ints(ptr, n):
    return list((ctypes.c_int32 * n).from_address(ptr))


def one_hot(ptr, tok):
    ctypes.memset(ptr, 0, V * 4)
    (ctypes.c_float * V).from_address(ptr)[tok] = 1.0


class ToyKernel:
    """The exports KernelLM.spec_generate_stream reaches, over a model whose greedy next token is nxt(last token), as a plain
    object (the way test_kv_session_old_kernel.Kernel stands in for a .so). `log` records every call. cached: what the store
    gives prepare to return (capped at n - 1, as the real one is). feeds: whether a continuation extends the drafter, as the
    kernel's does when the store keeps it aligned; False leaves it behind the KV (ledger R50, R52). script: the number of
    drafts the drafter gets right in each cycle, in order; the rest of the block is wrong. toks: the tokens whose KV the
    engine holds, in order (kv is their count); dr: the drafter's length. strip: a continuation of two tokens or more that starts
    with this token loses it in the KV, as one that starts with BOS does in the engine (the KV then holds fewer rows than the
    prompt has tokens). fail_at: {export: n} makes the n-th call of that export fail (a verify fails AFTER it appended its
    rows, as an engine can)."""

    def __init__(self, cached=0, feeds=True, script=(), strip=None):
        self.toks, self.dr = [], 0
        self.cached, self.feeds, self.script, self.strip = cached, feeds, list(script), strip
        self.log, self.fail_at, self.count = [], {}, {}

    @property
    def kv(self):
        return len(self.toks)

    def hit(self, name):
        """Count one call of the export `name`; True when it is the call that fails."""
        self.count[name] = self.count.get(name, 0) + 1
        return self.fail_at.get(name) == self.count[name]

    def nomos_session_prepare(self, h, ids_ptr, n, spec):
        ids = ints(ids_ptr, n)
        self.log.append(("prepare", tuple(ids), spec))
        c = min(self.cached, n - 1)
        self.toks, self.dr = ids[:c], c                 # the engine now holds ids[:c], the drafter in step with it
        return c

    def nomos_session_commit(self, h):
        self.log.append(("commit",))
        return 0

    def nomos_dflash_reset(self, h):
        self.log.append(("dflash_reset",))
        self.dr = 0
        return 0

    def nomos_prefill(self, h, ids_ptr, n, out):
        ids = ints(ids_ptr, n)
        self.log.append(("prefill", tuple(ids)))
        self.toks, self.dr = ids, n
        one_hot(out, nxt(ids[-1]))
        return 0

    def nomos_prefill_cont(self, h, ids_ptr, n, out):
        ids = ints(ids_ptr, n)
        self.log.append(("prefill_cont", tuple(ids)))
        rows = ids[1:] if (self.strip is not None and n > 1 and ids[0] == self.strip) else ids
        self.toks += rows
        self.dr += len(rows) if self.feeds else 0
        one_hot(out, nxt(ids[-1]))
        return 0

    def nomos_decode_step(self, h, tok, out):
        self.log.append(("step", tok))
        if self.hit("step"):
            return -99
        self.toks.append(tok)
        one_hot(out, nxt(tok))
        return 0

    def nomos_kv_cache_len(self, h):
        return self.kv

    def nomos_dflash_cache_len(self, h):
        return self.dr

    def nomos_kv_set_len(self, h, n):
        self.log.append(("set_len", n))
        if self.hit("set_len"):
            return -1
        if n > self.kv:
            return -2                                    # the real export refuses to grow the KV
        del self.toks[n:]
        return 0

    def nomos_dflash_set_len(self, h, n):
        self.log.append(("dflash_set_len", n))
        if self.hit("dflash_set_len"):
            return -1
        self.dr = n
        return 0

    def nomos_dflash_draft_block(self, h, seed, L, out_ptr):
        self.log.append(("draft", seed))
        good = self.script.pop(0) if self.script else 0
        drafts, t = [], seed
        for i in range(15):
            d = nxt(t) if i < good else (nxt(t) + 1) % V
            drafts.append(d)
            t = d
        (ctypes.c_int32 * 15).from_address(out_ptr)[:] = drafts
        return 15

    def nomos_dflash_verify_fused(self, h, rows_ptr, m, L, out_ptr):
        rows = ints(rows_ptr, m)
        self.log.append(("verify", tuple(rows)))
        self.toks = self.toks[:L] + rows                 # every row is appended; the host cuts back to what it accepts
        for i, t in enumerate(rows):
            one_hot(out_ptr + i * V * 4, nxt(t))
        return -99 if self.hit("verify") else 0

    def nomos_dflash_append_verify_context(self, h, start, m):
        self.dr += m
        return self.dr

    def calls(self, name):
        return [c for c in self.log if c[0] == name]


def toy_lm(kernel, store=True):
    lm = KernelLM.__new__(KernelLM)                      # KernelLM's own methods over the toy kernel; __init__ would load a .so
    lm.lib, lm.h, lm.vocab_size = kernel, 1, V
    lm._logits = np.zeros(V, dtype=np.float32)
    lm._dflash_loaded, lm._session_ok = True, store
    return lm


IDS = list(range(1, 11))     # a ten-token prompt


def run(kernel, n, vb=4, stop=(), ids=IDS, trace=None, store=True):
    """One turn to its end. Returns (tokens, what the turn printed)."""
    with contextlib.redirect_stdout(io.StringIO()) as out:
        toks = list(toy_lm(kernel, store).spec_generate_stream(ids, max_new_tokens=n, stop_ids=stop, vb=vb, trace=trace))
    return toks, out.getvalue()


BEHIND = "[kvstore] drafter behind the KV; this turn decodes without speculation"


class SpecLoopTest(unittest.TestCase):
    def test_a_fresh_prompt_resets_the_drafter_and_prefills_it_whole(self):
        k = ToyKernel(cached=0, script=[3, 3, 3])
        toks, _ = run(k, 8)
        self.assertEqual(k.log[:3], [("prepare", tuple(IDS), 1), ("dflash_reset",), ("prefill", tuple(IDS))])
        self.assertEqual(toks, chain(IDS[-1], 8))
        self.assertEqual(k.calls("commit"), [("commit",)])

    def test_a_cached_prefix_is_continued_not_reprefilled(self):
        k = ToyKernel(cached=7, script=[3, 3, 3])
        toks, _ = run(k, 8)
        self.assertEqual(k.log[:2], [("prepare", tuple(IDS), 1), ("prefill_cont", (8, 9, 10))])
        self.assertEqual(k.calls("prefill") + k.calls("dflash_reset"), [], "the store kept the drafter aligned")
        self.assertEqual(toks, chain(IDS[-1], 8))
        self.assertEqual(k.calls("commit"), [("commit",)])

    def test_the_trace_holds_the_drafts_accepted_in_each_cycle(self):
        # vb 4: three drafts a cycle. 12 tokens: the seed, then cycles of 4, 2, 1 and 4 tokens.
        trace = []
        toks, _ = run(ToyKernel(script=[3, 1, 0, 3]), 12, trace=trace)
        self.assertEqual(trace, [3, 1, 0, 3])
        self.assertEqual(toks, chain(IDS[-1], 12))
        self.assertEqual(run(ToyKernel(script=[3, 1, 0, 3]), 12)[0], toks, "trace=None must change nothing")

    def test_the_budget_ends_the_turn_in_a_burst_or_in_an_exact_step(self):
        for n, script, trace in ((10, [3, 3], [3, 3]),       # 1 + 4 + 4 = 9 tokens, then one left: the exact M=1 step
                                 (9, [3, 3], [3, 3])):       # the second burst takes the last token
            with self.subTest(n=n):
                k, got = ToyKernel(script=list(script)), []
                toks, _ = run(k, n, trace=got)
                self.assertEqual(got, trace)
                self.assertEqual(toks, chain(IDS[-1], n))
                self.assertEqual(len(k.calls("step")), 1 if n == 10 else 0)
                # The exact step writes a KV row the drafter never gets (the decode path captures no taps for it). A turn
                # that ended with the drafter one row short of the KV would make the store's next speculative request fork.
                self.assertEqual(k.kv, k.dr, "the turn must end with the drafter in step with the KV")
                self.assertEqual(k.calls("commit"), [("commit",)])

    def test_the_exact_steps_row_is_dropped_before_its_token_is_handed_out(self):    # ledger R54
        # The settle (R55) would cut that row too, but only when the generator ends. The drop itself comes first, so the
        # state is right while the last token is being handed out, whether or not the consumer ever resumes the generator.
        k = ToyKernel(script=[3, 3])
        with contextlib.redirect_stdout(io.StringIO()):
            g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=10, stop_ids=(), vb=4)
            got = [next(g) for _ in range(10)]               # the seed, two bursts of four, and the exact step's token
            self.assertEqual(got, chain(IDS[-1], 10))
            self.assertEqual((k.kv, k.dr), (len(IDS) + 8, len(IDS) + 8), "the step's row is already gone")
            self.assertEqual(k.calls("commit"), [], "the turn is not over yet")
            g.close()
        self.assertEqual(k.calls("commit"), [("commit",)])

    def test_a_failed_drop_of_the_exact_steps_row_costs_the_reply_nothing(self):    # final review, ledger L354
        # The two cycles' cuts are the first two set_len calls; the exact step's drop is the third. It fails: the reply still
        # gets its last token, the failure is logged, and the settle makes the same cut when the generator ends.
        k = ToyKernel(script=[3, 3])
        k.fail_at["set_len"] = 3
        toks, out = run(k, 10)
        self.assertEqual(toks, chain(IDS[-1], 10))
        self.assertIn("[kvstore] could not drop the exact step's row: ", out)
        self.assertEqual(k.calls("set_len")[2:], [("set_len", len(IDS) + 8), ("set_len", len(IDS) + 8)])
        self.assertEqual((k.kv, k.dr), (len(IDS) + 8, len(IDS) + 8), "the settle made the cut the drop could not")
        self.assertEqual(k.calls("commit"), [("commit",)])

    def test_a_drafter_behind_the_kv_decodes_on_the_base_path(self):             # ledger R53
        k, trace = ToyKernel(cached=7, feeds=False, script=[3, 3, 3]), []
        toks, printed = run(k, 6, trace=trace)
        self.assertEqual(toks, chain(IDS[-1], 6))
        self.assertEqual(trace, [], "no speculative cycle ran")
        self.assertEqual(printed, BEHIND + "\n")
        self.assertEqual(k.log[:2], [("prepare", tuple(IDS), 1), ("prefill_cont", (8, 9, 10))])
        self.assertEqual([c[1] for c in k.calls("step")], toks[:-1], "every token but the last is stepped")
        for name in ("prefill", "dflash_reset", "draft", "verify"):
            self.assertEqual(k.calls(name), [], f"the fallback must not call {name}: the restored KV is kept")
        self.assertEqual(k.log[-1], ("commit",))
        self.assertEqual(k.calls("commit"), [("commit",)])

    def test_a_fresh_prefill_never_trails_so_it_never_takes_the_fallback(self):
        k, trace = ToyKernel(cached=0, feeds=False, script=[3, 3]), []
        toks, printed = run(k, 9, trace=trace)
        self.assertEqual((trace, printed), ([3, 3], ""))
        self.assertEqual(toks, chain(IDS[-1], 9))

    def test_the_base_path_turn_stops_at_a_stop_token_without_stepping_it(self):
        want = chain(IDS[-1], 6)
        stop = want[2]
        self.assertNotIn(stop, want[:2])
        k = ToyKernel(cached=7, feeds=False)
        toks, _ = run(k, 6, stop=(stop,))
        self.assertEqual(toks, want[:3])                  # the stop token is yielded, then the turn ends
        self.assertEqual([c[1] for c in k.calls("step")], want[:2])
        self.assertEqual(k.calls("commit"), [("commit",)])

    def test_the_base_path_turn_honours_the_token_budget(self):
        for n in (1, 2, 5):
            with self.subTest(n=n):
                k = ToyKernel(cached=7, feeds=False)
                toks, _ = run(k, n)
                self.assertEqual(toks, chain(IDS[-1], n))
                self.assertEqual(len(k.calls("step")), n - 1)
                self.assertEqual(k.calls("commit"), [("commit",)])

    def test_the_turn_is_committed_once_however_its_generator_ends(self):       # Review Focus 1, on the speculative path
        # The speculative loop, and the base-path fallback with its own try block. The failure to inject: the loop's verify
        # call, or the fallback's step.
        for what, make, call in (("speculating", lambda: ToyKernel(cached=7, script=[3, 3, 3]), "verify"),
                                 ("on the base path", lambda: ToyKernel(cached=7, feeds=False), "step")):
            for how, pulls in (("runs to its end", None), ("is closed after its first token", 1),
                               ("is closed after three tokens", 3)):      # inside the first burst, when speculating
                with self.subTest(f"{what}, the generator {how}"), contextlib.redirect_stdout(io.StringIO()):
                    k = make()
                    g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=8, stop_ids=(), vb=4)
                    got = list(itertools.islice(g, pulls))
                    if pulls is not None:
                        self.assertEqual(len(got), pulls)
                        self.assertEqual(k.calls("commit"), [], "the turn is not over yet")
                        g.close()
                    self.assertEqual(k.calls("commit"), [("commit",)])
            with self.subTest(f"{what}, the engine fails"), contextlib.redirect_stdout(io.StringIO()):
                k = make()
                k.fail_at[call] = 1
                with self.assertRaises(RuntimeError):
                    list(toy_lm(k).spec_generate_stream(IDS, max_new_tokens=8, stop_ids=(), vb=4))
                self.assertEqual(k.calls("commit"), [("commit",)])

    def test_without_the_store_the_loop_runs_as_it_always_did(self):
        k, trace = ToyKernel(cached=7, script=[3, 3]), []
        toks, printed = run(k, 9, trace=trace, store=False)    # a kernel built before the store: prepare 0, commit 0
        self.assertEqual(k.log[:2], [("dflash_reset",), ("prefill", tuple(IDS))])
        self.assertEqual(k.calls("prepare") + k.calls("commit"), [])
        self.assertEqual((toks, trace, printed), (chain(IDS[-1], 9), [3, 3], ""))


    # -- ruling R55: a turn ends as the base path's does ---------------------------------------------------------------
    # The base path leaves every token it handed out in the KV but the last (a stop token, or the token whose yield close()
    # interrupted, is never stepped). A burst is different: its accepted drafts are in the KV and the drafter before the first
    # of them is yielded. A turn that ends inside the burst, on a stop token or a closed generator, must cut them back, or the
    # client's history diverges from the KV inside the burst and the store forks, which rewrites the whole conversation.
    # vb 4, script [3, ...]: the first burst is three accepted drafts and the correction token, d1 d2 d3 c, after the seed.

    def cuts_before_the_commit(self, k, burst_end, t):
        """After the cycle's own set_len(burst_end): a cut of the drafter to t and a cut of the KV to t, in either order, then
        the commit."""
        self.assertEqual(k.log[-4], ("set_len", burst_end))
        self.assertEqual(sorted(k.log[-3:-1]), sorted([("dflash_set_len", t), ("set_len", t)]))
        self.assertEqual(k.log[-1], ("commit",))

    def test_a_generator_closed_inside_a_burst_ends_as_the_base_path_does(self):
        k = ToyKernel(script=[3, 3])
        with contextlib.redirect_stdout(io.StringIO()):
            g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=20, stop_ids=(), vb=4)
            got = list(itertools.islice(g, 3))               # the seed, then d1 and d2: the client has seen these
            self.assertEqual(k.kv, len(IDS) + 4, "all of the burst's rows are in the KV until the close")
            g.close()
        self.assertEqual(got, chain(IDS[-1], 3))
        want = IDS + got[:-1]                                # every token handed out but the last
        self.assertEqual(k.toks, want)
        self.assertEqual(k.dr, len(want))
        self.cuts_before_the_commit(k, len(IDS) + 4, len(want))

    def test_a_stop_token_accepted_as_a_draft_takes_its_row_and_the_drafts_after_it_with_it(self):
        want = chain(IDS[-1], 5)                             # the seed, d1, d2, d3, the correction token
        stop = want[2]                                       # d2 is a stop token; d3 was accepted after it
        self.assertNotIn(stop, want[:2])
        k = ToyKernel(script=[3, 3])
        toks, _ = run(k, 20, stop=(stop,))
        self.assertEqual(toks, want[:3])                     # the stop token is handed out, then the turn ends
        self.assertEqual(k.toks, IDS + toks[:-1])            # neither its row nor d3's is in the KV
        self.assertEqual(k.dr, len(IDS) + 2)
        self.cuts_before_the_commit(k, len(IDS) + 4, len(IDS) + 2)

    def test_a_turn_that_ends_with_nothing_beyond_its_last_token_cuts_nothing(self):
        want = chain(IDS[-1], 5)
        for what, n, stop, cycles in (("at its budget, in a burst", 9, (), 2),
                                      ("at its budget, in the exact step", 10, (), 2 + 1),    # + the step's own row, dropped
                                      ("on a stop token that is the correction token", 20, (want[4],), 1)):
            with self.subTest(what):
                k = ToyKernel(script=[3, 3])
                run(k, n, stop=stop)
                self.assertEqual(k.calls("dflash_set_len"), [])
                self.assertEqual(len(k.calls("set_len")), cycles, "only the loop's own cuts")
                self.assertEqual(k.kv, k.dr)

    def test_an_engine_error_after_a_verify_cuts_back_to_the_last_good_length(self):
        k = ToyKernel(script=[3, 3])
        k.fail_at["verify"] = 2                              # the second verify appends its rows, then fails
        with contextlib.redirect_stdout(io.StringIO()):
            g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=20, stop_ids=(), vb=4)
            got = [next(g) for _ in range(5)]                # the seed and the first burst: the engine is at L = P + 4
            self.assertEqual(k.kv, len(IDS) + 4)
            with self.assertRaisesRegex(RuntimeError, "nomos_dflash_verify_fused rc=-99"):
                next(g)
        self.assertEqual(k.toks, IDS + got[:-1])
        self.assertEqual(k.dr, len(IDS) + 4)
        self.assertEqual(k.log[-3][0], "verify")
        self.assertEqual(k.log[-2:], [("set_len", len(IDS) + 4), ("commit",)])    # the unverified rows go before the commit

    def test_a_failing_cut_is_reported_and_neither_hides_the_error_nor_stops_the_commit(self):
        # The cycle's own set_len is the first call of that export, so the settle's KV cut is the second. After a failed verify
        # the drafter was never extended (that happens only after a verify), so it has nothing to cut there.
        for what, fail_at, behind_an_error in (("the drafter's", {"dflash_set_len": 1}, False),
                                               ("the KV's", {"set_len": 2}, True)):
            with self.subTest(f"{what} cut fails when a generator is closed"), contextlib.redirect_stdout(io.StringIO()) as out:
                k = ToyKernel(script=[3, 3])
                k.fail_at.update(fail_at)
                g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=20, stop_ids=(), vb=4)
                list(itertools.islice(g, 3))
                g.close()                                    # must not raise
                self.assertIn("[kvstore] settle failed: ", out.getvalue())
                self.assertEqual(k.calls("commit"), [("commit",)])
                self.assertLessEqual(k.dr, k.kv, "a failed cut must not leave the drafter ahead of the KV")
            if behind_an_error:
                with self.subTest(f"{what} cut fails behind an engine error"), contextlib.redirect_stdout(io.StringIO()) as out:
                    k = ToyKernel(script=[3, 3])
                    k.fail_at.update(fail_at)
                    k.fail_at["verify"] = 2
                    g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=20, stop_ids=(), vb=4)
                    list(itertools.islice(g, 5))
                    with self.assertRaisesRegex(RuntimeError, "nomos_dflash_verify_fused"):    # the engine's, not the cut's
                        next(g)
                    self.assertIn("[kvstore] settle failed: ", out.getvalue())
                    self.assertEqual(k.calls("commit"), [("commit",)])
                    self.assertLessEqual(k.dr, k.kv)

    def test_keep_counts_from_the_kv_not_from_the_prompt(self):                  # ledger R56
        # An engine can hold fewer rows than the prompt has tokens: a continuation that starts with BOS loses it in the KV. The
        # turn's rows are counted from what the KV holds once the prompt is in, so the settle still cuts exactly.
        k = ToyKernel(cached=7, script=[3, 3], strip=8)       # the suffix (8, 9, 10) loses its first token, as a BOS would
        with contextlib.redirect_stdout(io.StringIO()):
            g = toy_lm(k).spec_generate_stream(IDS, max_new_tokens=20, stop_ids=(), vb=4)
            got = list(itertools.islice(g, 3))                 # the seed, then d1 and d2 of the first burst
            self.assertEqual(k.kv, len(IDS) - 1 + 4, "one row fewer than the prompt has tokens, then the burst's four")
            g.close()
        rows = IDS[:7] + [9, 10]                               # the prompt's rows: the token that was dropped has none
        self.assertEqual(k.toks, rows + got[:-1])
        self.assertEqual(k.dr, len(rows) + 2)

    def test_the_settle_cuts_to_the_shortest_of_the_kv_the_drafter_and_what_was_handed_out(self):
        # The drafter is in step with the KV whenever the loop ends, so a turn cannot reach every case here. The settle still
        # must not leave the drafter ahead of the KV, nor a KV longer than the drafter that feeds it.
        for (kv, dr), keep, want in (((30, 30), 22, (22, 22)),      # rows the client never saw
                                     ((30, 30), 30, (30, 30)),      # a turn that ended at its budget
                                     ((30, 30), 35, (30, 30)),      # keep never grows anything
                                     ((30, 26), 40, (26, 26)),      # the drafter trails: the KV is cut to it
                                     ((26, 30), 40, (26, 26)),      # the drafter is ahead: it is cut to the KV
                                     ((30, 26), 22, (22, 22))):
            with self.subTest(kv=kv, drafter=dr, keep=keep):
                k = ToyKernel()
                k.toks, k.dr = list(range(kv)), dr
                toy_lm(k)._settle_spec_turn(keep)
                self.assertEqual((k.kv, k.dr), want)


if __name__ == "__main__":
    unittest.main()
