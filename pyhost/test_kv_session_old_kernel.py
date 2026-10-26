#!/usr/bin/env python3
"""A kernel built before the KV session store (ledger R40). CPU only: a fake kernel stands in for the .so.

A kernel .so from before the store exports no nomos_session_*. The host then behaves as follows, in process, through
the engine daemon and through KernelClient alike:
  NOMOS_KVSTORE unset or 0   the store is off, as it always was: status 0, prepare 0 (so the whole prompt is
                             prefilled), commit 0, stats ten zeros.
  NOMOS_KVSTORE=1            KernelLM refuses at init, before it loads the engine: the operator asked for a store
                             this kernel does not have.
A kernel that does export them behaves as before (the last class pins that).
Run: pixi run --no-install python3 pyhost/test_kv_session_old_kernel.py -v"""
import contextlib
import ctypes
import io
import os
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host  # noqa: E402
import engine_daemon  # noqa: E402
from kernel_client import KernelClient  # noqa: E402


class Fn:
    """A ctypes foreign function as KernelLM uses it: callable, with settable argtypes and restype."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class Kernel:
    """The part of a kernel .so that KernelLM touches, as a plain object, so an export the .so lacks is simply absent
    (hasattr and getattr behave as on a ctypes CDLL). `session`, a dict of callables keyed status / prepare / commit /
    stats, adds the nomos_session_* exports. Every prefill, step and shutdown is recorded in `calls`; for a prefill
    the token ids at the address the host passed are recorded too, in `prompts`, so a test sees WHICH tokens reached
    the kernel and not only how many."""

    def __init__(self, session=None):
        self.calls, self.prompts = [], []
        exports = {
            "nomos_model_id": lambda: 2,                                              # Muse: vocab 202048
            "nomos_prefill": lambda h, ids, n, out: self._prefill("prefill", ids, n),
            "nomos_prefill_cont": lambda h, ids, n, out: self._prefill("prefill_cont", ids, n),
            "nomos_decode_step": lambda h, tok, out: self._rec("step", tok),
            "nomos_kv_cache_len": lambda h: 0,
            "nomos_kv_set_len": lambda h, n: 0,
            "nomos_shutdown": lambda h: self._rec("shutdown", h),
        }
        exports.update({"nomos_session_" + k: v for k, v in (session or {}).items()})
        for name, fn in exports.items():
            setattr(self, name, Fn(fn))

    def _rec(self, *call):
        self.calls.append(call)
        return 0

    def _prefill(self, name, ptr, n):
        self.prompts.append((name, tuple((ctypes.c_int32 * n).from_address(ptr))))
        return self._rec(name, n)


class Stop(Exception):
    """Raised by the stub socket's accept(): engine_daemon.main() would otherwise serve for ever, so a test whose
    daemon wrongly gets that far fails instead of hanging."""


def stub_socket():
    srv = mock.MagicMock()
    srv.accept.side_effect = Stop
    return srv


@contextlib.contextmanager
def patched(kernel, store=None, init=None):
    """host.H (the harness that loads the .so and inits the engine) and NOMOS_KVSTORE (None: unset), patched. Yields the
    engine-init stub so a test can see whether the engine was ever loaded."""
    init = init if init is not None else mock.Mock(return_value=1)
    with mock.patch.dict(os.environ):
        os.environ.pop("NOMOS_KVSTORE", None)
        if store is not None:
            os.environ["NOMOS_KVSTORE"] = store
        with mock.patch.object(host.H, "load_lib", return_value=kernel), mock.patch.object(host.H, "init_engine", init):
            yield init


def load(kernel, store=None, init=None):
    with patched(kernel, store, init):
        return host.KernelLM()


@contextlib.contextmanager
def connected(lm):
    """A KernelClient on a real Unix socket to engine_daemon._serve_conn(lm). The server's accept and the client's socket
    both have timeouts, so a daemon that died fails the test instead of hanging it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "engine.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        srv.settimeout(10)

        def serve():
            conn, _ = srv.accept()
            with conn:
                engine_daemon._serve_conn(conn, lm)

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        client = KernelClient(path)
        client.sock.settimeout(10)
        try:
            yield client
        finally:
            client.shutdown()
            t.join(10)
            srv.close()


class KernelWithoutTheStoreTest(unittest.TestCase):
    def test_store_unset_or_zero_runs_with_the_store_off(self):
        for store in (None, "", "0", "0.0"):
            with self.subTest(NOMOS_KVSTORE=store):
                lm = load(Kernel(), store)
                self.assertEqual(lm.session_status(), 0)
                self.assertEqual(lm.session_prepare([1, 2, 3]), 0)
                self.assertEqual(lm.session_commit(), 0)
                self.assertEqual(lm.session_stats(), [0] * 10)

    def test_a_turn_prefills_the_whole_prompt_as_it_always_did(self):
        for name, turn in (("generate", lambda lm: lm.generate(list(range(10)), max_new_tokens=3)),
                           ("generate_stream", lambda lm: list(lm.generate_stream(list(range(10)), max_new_tokens=3)))):
            with self.subTest(name):
                k = Kernel()
                self.assertEqual(len(turn(load(k))), 3)
                self.assertEqual([c for c in k.calls if c[0] != "step"], [("prefill", 10)])
                self.assertEqual(k.prompts, [("prefill", tuple(range(10)))])

    def test_store_on_is_refused_at_init_before_the_engine_loads(self):
        for store in ("1", "1.0", "2"):
            with self.subTest(NOMOS_KVSTORE=store):
                init, k = mock.Mock(return_value=1), Kernel()
                with self.assertRaises(RuntimeError) as cm:
                    load(k, store, init)
                for text in ("NOMOS_KVSTORE", "no KV session store", "refresh_build.sh", "NOMOS_KVSTORE=0"):
                    self.assertIn(text, str(cm.exception))
                init.assert_not_called()
                self.assertEqual(k.calls, [], "nothing was loaded, so there is nothing to shut down")

    def test_a_partial_set_of_exports_counts_as_none(self):
        # Half an API cannot run a turn: only the complete set of four enables the store.
        lm = load(Kernel({"status": lambda h: 1}))
        self.assertEqual((lm.session_status(), lm.session_prepare([1, 2, 3]), lm.session_commit()), (0, 0, 0))
        with self.assertRaises(RuntimeError):
            load(Kernel({"status": lambda h: 1}), "1")


class DaemonAndClientTest(unittest.TestCase):
    def test_a_client_of_a_daemon_on_such_a_kernel_behaves_the_same(self):
        k = Kernel()
        with connected(load(k)) as client:
            self.assertEqual(client.model_id(), 2)
            self.assertEqual(client.session_prepare([1, 2, 3]), 0)
            self.assertEqual(client.session_commit(), 0)
            self.assertEqual(client.session_stats(), [0] * 10)
            self.assertEqual(len(client.generate(list(range(10)), max_new_tokens=3)), 3)
        self.assertEqual([c for c in k.calls if c[0] != "step"], [("prefill", 10)])
        self.assertEqual(k.prompts, [("prefill", tuple(range(10)))])

    def test_the_daemon_announces_the_store_status_when_it_is_ready(self):
        for kernel, store, want in ((Kernel(), None, "kvstore status=0"),
                                    (Kernel({"status": lambda h: 1, "prepare": lambda *a: 0, "commit": lambda h: 0,
                                             "stats": lambda h, out: 0}), "1", "kvstore status=1")):
            with self.subTest(want):
                srv = stub_socket()
                with tempfile.TemporaryDirectory() as d, \
                        mock.patch.dict(os.environ, {"NOMOS_ENGINE_SOCKET": os.path.join(d, "engine.sock")}), \
                        patched(kernel, store), mock.patch.object(engine_daemon.socket, "socket", return_value=srv), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    with self.assertRaises(Stop):               # main() got as far as accept(), after READY
                        engine_daemon.main()
                self.assertIn("READY", out.getvalue())
                self.assertIn(want, out.getvalue())

    def test_the_daemon_refuses_to_start_when_the_store_is_requested(self):
        srv = stub_socket()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"NOMOS_ENGINE_SOCKET": os.path.join(d, "engine.sock")}), \
                patched(Kernel(), "1") as init, mock.patch.object(engine_daemon.socket, "socket", return_value=srv):
            with self.assertRaisesRegex(RuntimeError, "no KV session store"):
                engine_daemon.main()
            init.assert_not_called()
        srv.bind.assert_not_called()                            # a refused daemon never opens its socket


class StoreRequestedTest(unittest.TestCase):
    def test_the_variable_is_read_the_way_the_kernel_reads_it(self):
        # gemma4_engine._env_float: the leading decimal ("-", digits, ".", digits), on above 0.5; anything else is off.
        table = (("", False), ("0", False), ("0.0", False), ("0.5", False), ("0.6", True), (".6", True), ("1", True),
                 ("1.0", True), ("1.", True), ("2", True), ("1abc", True), ("-1", False), ("true", False),
                 (" 1", False), ("+1", False), ("\u0661", False))      # U+0661, Arabic-Indic one: not a digit to the kernel
        for value, want in table:
            with self.subTest(NOMOS_KVSTORE=value), mock.patch.dict(os.environ, {"NOMOS_KVSTORE": value}):
                self.assertEqual(host._store_requested(), want)
        with mock.patch.dict(os.environ):
            os.environ.pop("NOMOS_KVSTORE", None)
            self.assertFalse(host._store_requested())


class KernelWithTheStoreTest(unittest.TestCase):
    """The behaviour of a kernel that exports the session API, unchanged by R40."""

    STATS = list(range(1, 11))

    def setUp(self):
        self.prepares, self.commits = [], []   # what the fake kernel was given: (ids, n, spec) and the commit's handle

    def session(self, status=1, prepare=0, commit=0):
        def fill(h, out):                       # what nomos_session_stats does: ten int64 at the address
            ctypes.memmove(out, (ctypes.c_int64 * 10)(*self.STATS), 80)
            return 0

        def record_prepare(h, ptr, n, spec):
            self.prepares.append((tuple((ctypes.c_int32 * n).from_address(ptr)), n, spec))
            return prepare

        def record_commit(h):
            self.commits.append(h)
            return commit

        return {"status": lambda h: status, "prepare": record_prepare, "commit": record_commit, "stats": fill}

    def test_the_exports_are_used(self):
        lm = load(Kernel(self.session(prepare=7, commit=5)), "1")
        self.assertEqual((lm.session_status(), lm.session_prepare([1] * 9), lm.session_commit()), (1, 7, 5))
        self.assertEqual(lm.session_stats(), self.STATS)

    def test_a_refused_status_raises_at_init_and_frees_the_engine(self):
        for status, text in ((-10, "not supported"), (-11, "NOMOS_KV_SWA"), (-12, "unquantized")):
            with self.subTest(status=status):
                k = Kernel(self.session(status=status))
                with self.assertRaises(RuntimeError) as cm:
                    load(k, "1")
                self.assertIn(f"({status})", str(cm.exception))
                self.assertIn(text, str(cm.exception))
                self.assertEqual([c for c in k.calls if c[0] == "shutdown"], [("shutdown", 1)],
                                 "the engine was already loaded: it must be shut down before the refusal leaves __init__")

    def test_a_negative_prepare_raises_and_a_negative_commit_is_only_returned(self):
        lm = load(Kernel(self.session(prepare=-1, commit=-3)), "1")
        with self.assertRaises(RuntimeError):
            lm.session_prepare([1, 2, 3])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(lm.session_commit(), -3)
        self.assertIn("commit rc=-3", out.getvalue())

    def test_the_wire_carries_ids_n_spec_and_the_counters_unchanged(self):
        # KernelClient -> engine_daemon -> KernelLM -> the kernel, with distinct values at every field: a swapped pair or a
        # shifted offset anywhere on the way shows up here.
        k = Kernel(self.session(prepare=1, commit=5))
        with connected(load(k, "1")) as client:
            self.assertEqual(client.session_prepare([5, 6, 7, 8]), 1)
            self.assertEqual(client.session_prepare([9, 10], spec=True), 1)
            self.assertEqual(client.session_commit(), 5)
            self.assertEqual(client.session_stats(), self.STATS)
        self.assertEqual(self.prepares, [((5, 6, 7, 8), 4, 0), ((9, 10), 2, 1)])
        self.assertEqual(self.commits, [1])

    def test_a_turn_through_the_daemon_prefills_exactly_the_uncached_suffix(self):
        k = Kernel(self.session(prepare=7))                    # the store has the first 7 tokens
        with connected(load(k, "1")) as client:
            self.assertEqual(len(client.generate(list(range(10)), max_new_tokens=2)), 2)
        self.assertEqual(self.prepares, [(tuple(range(10)), 10, 0)])
        self.assertEqual(k.prompts, [("prefill_cont", (7, 8, 9))], "the kernel must be given tokens 7..9, and no full prefill")
        self.assertEqual(self.commits, [1], "the turn is committed once")


if __name__ == "__main__":
    unittest.main()
