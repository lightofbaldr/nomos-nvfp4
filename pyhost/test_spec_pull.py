#!/usr/bin/env python3
"""The engine daemon's speculative stream is PULL-driven (ledger R56): one token per frame, and the daemon advances its loop only
when the client asks for the next token. CPU only. The daemon's real connection handler (engine_daemon._serve_conn) and the real
KernelClient talk over a socketpair; a fake lm supplies a recording generator in place of the engine.

Why it matters: a daemon that runs ahead of the socket reader hands out tokens the client never took, and the KV session store
(which settles each turn to the tokens handed out, R55) then holds rows the client never saw. The next turn diverges inside them
and forks, which rewrites the whole conversation.
Run: pixi run --no-install python3 pyhost/test_spec_pull.py -v"""
import contextlib
import io
import os
import socket
import struct
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_daemon  # noqa: E402
import wire  # noqa: E402
from kernel_client import KernelClient  # noqa: E402

PING = struct.pack("<B", wire.OP_PING)


class FakeLM:
    """Stands in for host.KernelLM. spec_generate_stream is a recording generator over `tokens`. `events` gets ("yield", i) as
    token i (1-based) is about to be handed out, ("closed", i) when close() lands on the yield of token i, and ("finally", i) when
    the loop ends however it ends, with i the tokens handed out. Every generator is kept in `gens`: with a reference held, only an
    explicit close() or the loop's own end can run its finally (a dropped reference would close it too, and hide a missing close).
    fail_after=k makes the loop raise fail_with (a RuntimeError unless given) after k tokens."""

    def __init__(self, events, tokens, fail_after=None, fail_with=None):
        self.events, self.tokens, self.fail_after = events, list(tokens), fail_after
        self.fail_with = fail_with or RuntimeError("engine fell over")
        self.requests, self.gens = [], []

    def spec_generate_stream(self, ids, max_new_tokens=1024, stop_ids=(), vb=8):
        self.requests.append((list(ids), max_new_tokens, list(stop_ids), vb))
        g = self._loop()
        self.gens.append(g)
        return g

    def _loop(self):
        y = 0
        try:
            for tok in self.tokens:
                if self.fail_after is not None and y == self.fail_after:
                    raise self.fail_with
                y += 1
                self.events.append(("yield", y))
                try:
                    yield tok
                except GeneratorExit:
                    self.events.append(("closed", y))
                    raise
        finally:
            self.events.append(("finally", y))


class Tap:
    """The daemon's end of the socket. It logs what the daemon sends (a token frame, the end frame, an error code) and an EOF it
    reads into the same event list as the fake generator. Both happen on the daemon's thread, so their order is exact."""

    def __init__(self, sock, events):
        self.sock, self.events = sock, events

    def sendall(self, data):
        body = data[4:]
        (n,) = struct.unpack_from("<i", body, 0)
        kind = "token" if (n == 1 and len(body) == 8) else "end" if (n == 0 and len(body) == 4) else \
            f"rc{n}" if (n < 0 and len(body) == 4) else f"n={n},{len(body)}B"
        self.events.append(("send", kind))
        self.sock.sendall(data)

    def recv(self, k):
        data = self.sock.recv(k)
        if not data:
            self.events.append(("eof",))
        return data


def bare_client(sock):
    """A real KernelClient on a socket that is already connected (its __init__ connects to a path)."""
    c = KernelClient.__new__(KernelClient)
    c.sock_path, c._dflash_dir, c._dirty, c.sock = "", "", False, sock
    return c


@contextlib.contextmanager
def serving(lm, events):
    """engine_daemon._serve_conn(lm) on one end of a socketpair, a real KernelClient on the other. On exit the client closes its
    end, the daemon's handler is joined, and the errors it raised (there must be none) are returned in `errors`."""
    d, c = socket.socketpair()
    d.settimeout(10)
    c.settimeout(10)                                  # a daemon that deadlocks fails the test instead of hanging it
    errors = []

    def run():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                engine_daemon._serve_conn(Tap(d, events), lm)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            d.close()                                 # main() closes the connection when the handler returns

    t = threading.Thread(target=run, daemon=True)
    t.start()
    client = bare_client(c)
    client.daemon_thread = t                          # so a test can see the handler finish on its own
    try:
        yield client
    finally:
        c.close()
        t.join(10)
        if t.is_alive():
            raise AssertionError("the daemon's handler did not finish")
        d.close()
        if errors:
            raise errors[0]


def sends(events):
    return [e[1] for e in events if e[0] == "send"]


class PullStreamTest(unittest.TestCase):
    def test_a_consumer_that_takes_three_tokens_and_closes_leaves_the_loop_at_the_third(self):
        events = []
        lm = FakeLM(events, range(100, 140))
        with serving(lm, events) as client:
            g = client.spec_generate_stream([1, 2, 3], max_new_tokens=40, stop_ids=(7,), vb=8)
            self.assertEqual([next(g) for _ in range(3)], [100, 101, 102])
            g.close()                                 # the serve closes the token source when the client goes away
            # Closing the generator is all it takes: the client drops its socket, the daemon reads EOF and closes its loop where
            # it is parked, and the handler returns, with nobody else closing anything.
            self.assertTrue(client._dirty)
            self.assertEqual(client.sock.fileno(), -1, "the client must close its socket the moment its generator is closed")
            client.daemon_thread.join(10)
            self.assertFalse(client.daemon_thread.is_alive(), "the daemon is still waiting for the next-token message")
        self.assertEqual([e for e in events if e[0] == "yield"], [("yield", 1), ("yield", 2), ("yield", 3)],
                         "the loop ran ahead of the client")
        self.assertEqual(sends(events), ["token"] * 3, "exactly the three tokens, one per frame, and no end frame")
        # the loop was closed at the yield of the token the client took last, and its finally ran: the settle and the commit
        self.assertEqual(events[-3:], [("eof",), ("closed", 3), ("finally", 3)])
        self.assertEqual(lm.requests, [([1, 2, 3], 40, [7], 8)], "the request body is unchanged")

    def test_a_consumer_that_drains_the_stream_gets_the_end_frame_after_the_loop_finished(self):
        events = []
        lm = FakeLM(events, range(100, 105))
        with serving(lm, events) as client:
            self.assertEqual(list(client.spec_generate_stream([1, 2], max_new_tokens=5, stop_ids=(), vb=8)),
                             [100, 101, 102, 103, 104])
            self.assertFalse(client._dirty, "a stream that ended cleanly leaves the connection usable")
            first_end = events.index(("send", "end"))
            self.assertEqual(client._rc_call(PING), 0)    # ... and it is
        self.assertEqual([e for e in events if e[0] == "yield"], [("yield", i) for i in range(1, 6)])
        self.assertLess(events.index(("finally", 5)), first_end, "the end frame must follow the loop's finally")
        self.assertNotIn(("closed", 5), events)            # it ended by itself
        self.assertEqual(sends(events)[:6], ["token"] * 5 + ["end"])

    def test_the_daemon_gives_the_loop_one_token_at_a_time(self):
        events = []
        with serving(FakeLM(events, range(100, 110)), events) as client:
            g = client.spec_generate_stream([1], max_new_tokens=10, stop_ids=(), vb=8)
            self.assertEqual(next(g), 100)
            self.assertEqual([e for e in events if e[0] == "yield"], [("yield", 1)], "the loop must wait for the client")
            self.assertEqual(next(g), 101)
            self.assertEqual([e for e in events if e[0] == "yield"], [("yield", 1), ("yield", 2)])
            g.close()

    def test_a_message_that_is_not_next_ends_the_stream_and_the_connection(self):
        events = []
        lm = FakeLM(events, range(100, 140))
        with serving(lm, events) as client:
            g = client.spec_generate_stream([1], max_new_tokens=40, stop_ids=(), vb=8)
            self.assertEqual(next(g), 100)
            wire.send_msg(client.sock, PING)          # a request in the middle of a stream: not "next"
            self.assertIsNone(wire.recv_msg(client.sock), "the daemon must drop the connection, not guess")
        self.assertEqual(events[-2:], [("closed", 1), ("finally", 1)])
        self.assertEqual([e for e in events if e[0] == "yield"], [("yield", 1)])

    def test_an_engine_error_mid_stream_reaches_the_client_as_an_error_frame(self):
        events = []
        lm = FakeLM(events, range(100, 110), fail_after=2)
        with serving(lm, events) as client:
            g = client.spec_generate_stream([1], max_new_tokens=10, stop_ids=(), vb=8)
            self.assertEqual([next(g), next(g)], [100, 101])
            with self.assertRaisesRegex(RuntimeError, "spec stream rc=-99"):
                next(g)
            self.assertTrue(client._dirty)
        self.assertLess(events.index(("finally", 2)), events.index(("send", "rc-99")),
                        "the loop's finally (settle and commit) must run before the client hears of the error")

    def test_an_oserror_from_the_loop_is_an_engine_error_not_a_client_that_left(self):
        # Only socket errors mean the client is gone. An OSError raised by the loop itself (a disk, a device) is the engine's.
        events = []
        lm = FakeLM(events, range(100, 110), fail_after=1, fail_with=OSError("disk fell over"))
        with serving(lm, events) as client:
            g = client.spec_generate_stream([1], max_new_tokens=10, stop_ids=(), vb=8)
            self.assertEqual(next(g), 100)
            with self.assertRaisesRegex(RuntimeError, "spec stream rc=-99"):
                next(g)

    def test_the_retired_push_opcode_is_refused(self):
        self.assertFalse(hasattr(wire, "OP_SPEC_STREAM"), "the push stream's opcode is gone")
        self.assertNotEqual(wire.OP_SPEC_PULL, 8, "and its number is not reused: a mismatched pair must fail, not misread")
        events = []
        with serving(FakeLM(events, [100]), events) as client:
            old_request = struct.pack("<Bi", 8, 1) + np.int32(7).tobytes() + struct.pack("<iii", 4, 8, 0)
            wire.send_msg(client.sock, old_request)
            self.assertEqual(struct.unpack("<i", wire.recv_msg(client.sock))[0], -1)
            self.assertEqual(client._rc_call(PING), 0, "the daemon refused it and went on serving")
        self.assertEqual([e for e in events if e[0] in ("yield", "finally")], [], "the loop never started")

    def test_a_client_against_a_daemon_that_does_not_know_the_pull_opcode_fails_loudly(self):
        # A daemon from before R56 answers any opcode it does not know with -1.
        d, c = socket.socketpair()
        c.settimeout(10)
        d.settimeout(10)

        def old_daemon():
            wire.recv_msg(d)
            wire.send_msg(d, struct.pack("<i", -1))
            d.close()

        t = threading.Thread(target=old_daemon, daemon=True)
        t.start()
        client = bare_client(c)
        with self.assertRaisesRegex(RuntimeError, "spec stream rc=-1"):
            next(client.spec_generate_stream([1, 2], max_new_tokens=4, stop_ids=(), vb=8))
        self.assertTrue(client._dirty)
        t.join(10)
        c.close()

    def test_a_batched_frame_from_a_push_era_daemon_is_malformed_to_the_client(self):
        d, c = socket.socketpair()
        c.settimeout(10)
        d.settimeout(10)

        def push_daemon():
            wire.recv_msg(d)
            wire.send_msg(d, struct.pack("<i", 3) + np.asarray([5, 6, 7], dtype=np.int32).tobytes())    # a burst in one frame
            d.close()

        t = threading.Thread(target=push_daemon, daemon=True)
        t.start()
        client = bare_client(c)
        with self.assertRaisesRegex(RuntimeError, "malformed spec frame"):
            next(client.spec_generate_stream([1], max_new_tokens=4, stop_ids=(), vb=8))
        t.join(10)
        c.close()


if __name__ == "__main__":
    unittest.main()
