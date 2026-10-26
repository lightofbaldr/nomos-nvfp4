#!/usr/bin/env python3
"""The serve's engine lock (CPU only, fake engine).

1. The serve commits a turn to the KV session store while it still holds the lock. serve._stream holds eng.lock for
   the whole generation, and generate_stream commits in its `finally`. A token source left suspended (the token loop
   broke out at the stop token, or the client disconnected) is finalised only when _stream's frame is torn down, which
   is after `with eng.lock` has exited. The commit would then run unlocked, racing the next request's prepare on the
   same engine, or on the same daemon socket (ledger R39).
2. GET /v1/kvstore/stats waits for that lock without starving the stream that holds it (ledger R43). A sync endpoint
   runs on anyio's 40-thread pool and keeps its thread through the whole wait, while the stream needs a pool thread
   for every chunk: about 40 overlapping polls freeze a reply that holds the lock.
Run: pixi run --no-install python3 pyhost/test_kv_session_serve_lock.py -v"""
import contextlib
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import anyio
import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402  (importing it loads no model: the engine is built at startup)
from host import _GenMixin  # noqa: E402

STOP = 7
COUNTERS = dict(zip(serve._KVSTORE_STATS, range(1, 11)))   # what the fake engine's session_stats() reports
CAP = 2.0                                                   # see CappedLock


class CappedLock:
    """A threading.Lock whose TIMED acquire waits at most CAP seconds. The stats endpoint's real wait is 60 s, which a
    test cannot sit through: with the cap, an endpoint that blocks for its whole timeout shows up in seconds. Plain
    acquire (`with lock`), non-blocking acquire, release and locked() are the real lock's."""

    def __init__(self, cap=CAP):
        self._lock, self._cap = threading.Lock(), cap

    def acquire(self, blocking=True, timeout=-1):
        if blocking and timeout > 0:
            timeout = min(timeout, self._cap)
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


class Backend(_GenMixin):
    """A fake logits engine whose argmax is token 3 until `stop_after` tokens were stepped, then STOP; each step takes
    `delay` seconds. Records whether the engine lock was held each time the turn was committed."""

    def __init__(self, lock, stop_after, delay=0.0):
        self.lock, self.stop_after, self.delay, self.stepped, self.commits = lock, stop_after, delay, 0, []

    def _logits(self):
        lg = np.zeros(16, dtype=np.float32)
        lg[STOP if self.stepped >= self.stop_after else 3] = 1.0
        return lg

    def session_prepare(self, ids, spec=False):
        return 0

    def prefill(self, ids):
        self.stepped = 0
        return self._logits()

    def step(self, tok):
        if self.delay:
            time.sleep(self.delay)
        self.stepped += 1
        return self._logits()

    def session_commit(self):
        self.commits.append(self.lock.locked())
        return 0

    def session_stats(self):
        return list(COUNTERS.values())


class Parser:
    """The ATEM stream parser's interface as _stream uses it: one content delta per token, saw_stop at STOP."""

    def __init__(self):
        self.saw_stop, self.raw_ids = False, []

    def feed(self, tok):
        self.raw_ids.append(tok)
        self.saw_stop = tok == STOP
        return [("content", "x")]

    def finish(self):
        return SimpleNamespace(tool_call=None), []


def engine(kind, stop_after=3, lock=None, delay=0.0):
    lock = lock if lock is not None else threading.Lock()
    return SimpleNamespace(
        lock=lock, lm=Backend(lock, stop_after, delay), stop={STOP}, model_name="fake", spec_loaded=False,
        rep_penalty_default=1.0, tok=SimpleNamespace(decode=lambda ids, **kw: "x" * len(ids)),
        protocol=SimpleNamespace(kind=kind, stream_parser=lambda tok: Parser(),
                                 channel_end_id=-1, tool_open_id=-2, tool_close_id=-3))


def stream(eng):
    req = serve.ChatReq(messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    return serve._stream(req, eng, [1, 2, 3, 4], None, thinking=False)


class StreamCommitUnderLockTest(unittest.TestCase):
    def check_runs_to_the_end(self, kind):
        eng = engine(kind)
        chunks = list(stream(eng))
        self.assertTrue(any('"content"' in c for c in chunks), "the fake engine must have streamed some text")
        self.assertEqual(eng.lm.commits, [True], "one commit, made while the engine lock was still held")
        self.assertFalse(eng.lock.locked())

    def test_muse_stream_ending_at_the_stop_token(self):
        self.check_runs_to_the_end("atem")

    def test_generic_stream_ending_at_the_stop_token(self):
        self.check_runs_to_the_end("gemma")

    def test_client_disconnect_mid_stream(self):          # Review Focus 1, through the serve
        for kind in ("atem", "gemma"):
            with self.subTest(kind=kind):
                eng = engine(kind, stop_after=50)
                g = stream(eng)
                next(g)                                   # the role chunk
                next(g)                                   # the first text: the turn is under way
                self.assertEqual(eng.lm.commits, [])
                g.close()                                 # what _astream does when the client goes away
                self.assertEqual(eng.lm.commits, [True], "one commit, made while the engine lock was still held")
                self.assertFalse(eng.lock.locked())


@contextlib.contextmanager
def serving(eng):
    """eng installed as the app's engine, as the lifespan does at startup, and put back afterwards."""
    previous = getattr(serve.app.state, "engine", None)
    serve.app.state.engine = eng
    try:
        yield eng
    finally:
        if previous is None:
            del serve.app.state.engine
        else:
            serve.app.state.engine = previous


def http():
    """An HTTP client straight into the real FastAPI app (routing, the dispatch of sync endpoints to the thread pool,
    JSON), with no socket."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=serve.app), base_url="http://test")


def run(main, timeout=60):
    """main() to completion on a fresh event loop, and so a fresh 40-thread pool, on a helper thread. A hang fails the
    test: anyio's fail_after ends a stuck await, and the join ends a BLOCKED event loop, which nothing running inside
    that loop could interrupt (the bug this file guards against, in its worst form)."""
    box = {}

    async def guarded():
        with anyio.fail_after(timeout):
            return await main()

    def target():
        try:
            box["result"] = anyio.run(guarded)
        except BaseException as e:  # noqa: BLE001 (re-raised in the caller's thread)
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout + 5)
    if t.is_alive():
        raise AssertionError(f"the event loop was still blocked {timeout + 5} s in: it never yielded")
    if "error" in box:
        raise box["error"]
    return box["result"]


async def get_stats():
    async with http() as client:
        return await client.get("/v1/kvstore/stats")


async def never():
    return False


class StatsEndpointTest(unittest.TestCase):
    """GET /v1/kvstore/stats (ledger R43)."""

    def test_overlapping_polls_do_not_starve_a_running_stream(self):
        # 45 polls, more than the pool's 40 threads, while the serve's own _astream drives a reply that holds the
        # engine lock. Each decode step of the fake engine takes 20 ms, so a healthy gap between chunks is about that.
        eng = engine("atem", stop_after=60, lock=CappedLock(), delay=0.02)
        request = SimpleNamespace(is_disconnected=never)
        gaps, replies = [], []

        async def stream():
            req = serve.ChatReq(messages=[{"role": "user", "content": "hi"}], max_tokens=200)
            last = time.monotonic()
            async for _ in serve._astream(req, eng, [1, 2, 3, 4], None, False, request):
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        async def poll(client):
            replies.append(await client.get("/v1/kvstore/stats"))

        async def main():
            async with http() as client, anyio.create_task_group() as tg:
                tg.start_soon(stream)
                await anyio.sleep(0.15)                       # the reply is under way, and holds the lock
                for _ in range(45):
                    tg.start_soon(poll, client)

        with serving(eng):
            run(main, timeout=20)
        self.assertLess(max(gaps), 0.6, f"the stream froze for {max(gaps):.2f} s while stats calls waited for its lock")
        self.assertEqual(len(replies), 45)
        self.assertTrue(all(r.status_code == 200 and r.json() == COUNTERS for r in replies),
                        [(r.status_code, r.text) for r in replies if r.status_code != 200][:3])
        self.assertEqual(eng.lm.commits, [True], "the reply ran to its end and committed under the lock")
        self.assertFalse(eng.lock.locked())

    def test_it_reports_the_counters_and_leaves_the_lock_free(self):
        eng = engine("atem")
        with serving(eng):
            r = run(get_stats)
        self.assertEqual((r.status_code, r.json()), (200, COUNTERS))
        self.assertFalse(eng.lock.locked())

    def test_busy_after_the_wait_without_touching_a_lock_it_does_not_hold(self):
        eng = engine("atem")
        eng.lock.acquire()                                    # a generation is running
        try:
            with serving(eng), mock.patch.object(serve, "_STATS_WAIT_S", 0.3):
                t0 = time.monotonic()
                r = run(get_stats, timeout=10)                # a deadline that never fires fails here, it does not hang
                waited = time.monotonic() - t0
            self.assertEqual((r.status_code, r.json()), (503, {"status": "busy"}))
            self.assertGreaterEqual(waited, 0.3)
            self.assertTrue(eng.lock.locked(), "the endpoint released a lock it never took")
        finally:
            eng.lock.release()

    def test_starting_while_the_engine_is_still_loading(self):
        self.assertIsNone(getattr(serve.app.state, "engine", None))
        r = run(get_stats)
        self.assertEqual((r.status_code, r.json()), (503, {"status": "starting"}))

    def test_the_lock_is_released_when_session_stats_raises(self):
        eng = engine("atem")

        def boom():
            raise RuntimeError("engine daemon closed the connection")

        eng.lm.session_stats = boom
        with serving(eng), self.assertRaises(RuntimeError):
            run(get_stats)
        self.assertFalse(eng.lock.locked())

    def test_cancelled_while_waiting_it_leaves_the_lock_alone(self):
        eng = engine("atem")
        eng.lock.acquire()                                    # a generation is running
        try:
            async def main():
                with anyio.move_on_after(0.2) as scope:
                    await serve.kvstore_stats()
                return scope.cancelled_caught

            with serving(eng):
                self.assertTrue(run(main), "the call should have been cancelled while it waited")
            self.assertTrue(eng.lock.locked(), "a cancelled waiter released a lock it never took")
        finally:
            eng.lock.release()
        with serving(eng), mock.patch.object(serve, "_STATS_WAIT_S", 0.5):
            self.assertEqual(run(get_stats).status_code, 200, "the lock was left held by the cancelled waiter")

    def test_cancelled_during_the_stats_call_it_keeps_the_lock_until_the_call_is_done(self):
        # The thread is using the engine socket. Releasing the lock while it still runs would let the next request
        # share that socket: the cancelled call has to wait for the thread, so anyio's default shielding must stay.
        eng = engine("atem")
        finished = threading.Event()

        def slow():
            time.sleep(0.4)
            finished.set()
            return list(COUNTERS.values())

        eng.lm.session_stats = slow

        async def main():
            with anyio.move_on_after(0.1) as scope:
                await serve.kvstore_stats()
            return scope.cancel_called, finished.is_set(), eng.lock.locked()

        with serving(eng):
            cancelled, done, locked = run(main)
        self.assertTrue(cancelled)
        self.assertTrue(done, "the cancelled call went on without waiting for its thread")
        self.assertFalse(locked)


if __name__ == "__main__":
    unittest.main()
