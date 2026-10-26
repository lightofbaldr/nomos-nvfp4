#!/usr/bin/env python3
"""The KV session store opens at launch, not at the first chat request (ruling R79; final review I3). CPU only: fake
kernels stand in for the .so (test_kv_session_old_kernel's), and the real engine daemon handler and the real KernelClient
talk over a socket, as in test_spec_pull.

- KernelLM.session_open returns the kernel's status and the store's directory, resolved to its canonical path;
- a kernel without nomos_session_open (any built before R79, the production OLMo .so among them) reports
  "off (unsupported)", never an error;
- OP_SESSION_OPEN carries both through the daemon unchanged, and the connection stays usable;
- a daemon from before this change answers the new request with -1, and the serve fails saying what to do;
- the serve opens the store after the drafter decision, never asks while NOMOS_KVSTORE is off, and /health reports it.
Run: pixi run --no-install python3 pyhost/test_kv_session_open.py -v"""
import contextlib
import ctypes
import io
import os
import pathlib
import shutil
import socket
import struct
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host  # noqa: E402
import serve  # noqa: E402  (importing it loads no model: the engine is built at startup)
import wire  # noqa: E402
from kernel_client import KernelClient  # noqa: E402
from test_kv_session_old_kernel import Kernel, connected, load  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OLD_DAEMON = "the engine daemon predates this serve's protocol: restart the daemon and the serve together (RUNBOOK §7)"


def store_exports(status=1, store_dir=""):
    """The five nomos_session_* exports of a kernel built with the store; open writes store_dir and returns status."""
    def open_(h, ptr, cap):
        raw = store_dir.encode() + b"\0"
        assert len(raw) <= cap, "the host's buffer is too small for the directory"
        ctypes.memmove(ptr, raw, len(raw))
        return status
    return {"status": lambda h: 1, "prepare": lambda *a: 0, "commit": lambda h: 0, "stats": lambda h, out: 0,
            "open": open_}


class KernelLMTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="kvstore_open_", dir=ROOT / "build"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_open_returns_the_status_and_the_canonical_directory(self):
        # NOMOS_KVSTORE_DIR as the operator spelled it: a symlinked root, with a trailing slash. The kernel names the store's
        # directory root + "/muse-<id>" as given, and the host resolves it to one canonical path.
        (self.tmp / "real").mkdir()
        (self.tmp / "link").symlink_to(self.tmp / "real")
        spelled = f"{self.tmp}/link//muse-0123456789abcdef"
        lm = load(Kernel(store_exports(1, spelled)), "1")
        self.assertEqual(lm.session_open(), (1, str(self.tmp / "real" / "muse-0123456789abcdef")))

    def test_a_status_other_than_open_names_no_directory(self):
        for status in (0, -2, -10, -11, -12):
            with self.subTest(status=status):
                lm = load(Kernel(store_exports(status, "/left/over/in/the/buffer")))
                self.assertEqual(lm.session_open(), (status, ""))

    def test_a_kernel_without_the_export_is_unsupported_not_an_error(self):
        exports = store_exports()
        del exports["open"]
        for name, kernel in (("built before the store (the production OLMo .so)", Kernel()),
                             ("built with the store, before R79", Kernel(exports))):
            with self.subTest(name):
                self.assertEqual(load(kernel).session_open(), (host.SESSION_OPEN_UNSUPPORTED, ""))


class DaemonTest(unittest.TestCase):
    def test_the_daemon_carries_status_and_directory_unchanged(self):
        for status, store_dir in ((1, "/home/someone/kv store/é/muse-00000000000000ff"), (-2, ""), (0, "")):
            with self.subTest(status=status):
                lm = load(Kernel(store_exports(status, store_dir)), "1")
                with connected(lm) as client:
                    self.assertEqual(client.session_open(), (status, store_dir))
                    self.assertFalse(client._dirty)
                    self.assertEqual(client.model_id(), 2, "the connection is still in step")

    def test_an_unsupported_kernel_is_reported_through_the_daemon_too(self):
        with connected(load(Kernel())) as client:
            self.assertEqual(client.session_open(), (host.SESSION_OPEN_UNSUPPORTED, ""))

    def test_an_old_daemon_fails_with_a_message_that_says_what_to_do(self):
        """A daemon from before this change does not know OP_SESSION_OPEN (nor, from before the store, prepare or stats; from
        before R56, the pull stream): it answers -1, the reply of a daemon to an opcode it does not know."""
        calls = (("session_open", lambda c: c.session_open()),
                 ("session_prepare", lambda c: c.session_prepare([1, 2, 3])),
                 ("session_stats", lambda c: c.session_stats()),
                 ("engine daemon spec stream", lambda c: next(c.spec_generate_stream([1, 2], max_new_tokens=4))))
        for what, call in calls:
            with self.subTest(what):
                d, c = socket.socketpair()
                d.settimeout(10)
                c.settimeout(10)

                def old_daemon():
                    wire.recv_msg(d)
                    wire.send_msg(d, struct.pack("<i", -1))

                t = threading.Thread(target=old_daemon, daemon=True)
                t.start()
                client = KernelClient.__new__(KernelClient)
                client.sock_path, client._dflash_dir, client._dirty, client.sock = "", "", False, c
                with self.assertRaises(RuntimeError) as cm:
                    call(client)
                self.assertEqual(str(cm.exception), f"{what} rc=-1: {OLD_DAEMON}")
                t.join(10)
                c.close()
                d.close()


class FakeOpener:
    def __init__(self, status=1, store_dir="/kv/muse-1"):
        self.status, self.store_dir, self.calls = status, store_dir, 0

    def session_open(self):
        self.calls += 1
        return self.status, self.store_dir


class ServeTest(unittest.TestCase):
    def test_the_serve_asks_only_when_the_store_is_requested(self):
        for value in (None, "", "0"):
            with self.subTest(NOMOS_KVSTORE=value), mock.patch.dict(os.environ):
                os.environ.pop("NOMOS_KVSTORE", None)
                if value is not None:
                    os.environ["NOMOS_KVSTORE"] = value
                lm = FakeOpener()
                self.assertEqual(serve._open_kvstore(lm), {"status": "off", "dir": None})
                self.assertEqual(lm.calls, 0, "the launcher forced it off: an engine of another model is never asked")

    def test_the_serve_reports_what_the_engine_said(self):
        table = ((1, "/kv/muse-1", {"status": "open", "dir": "/kv/muse-1"}),
                 (0, "", {"status": "off", "dir": None}),
                 (-2, "", {"status": "disabled", "dir": None}),
                 (host.SESSION_OPEN_UNSUPPORTED, "", {"status": "off (unsupported)", "dir": None}),
                 (-10, "", {"status": "refused (-10: this model profile is not Muse)", "dir": None}),
                 (-77, "", {"status": "error (-77)", "dir": None}))
        for status, store_dir, want in table:
            with self.subTest(status=status), mock.patch.dict(os.environ, {"NOMOS_KVSTORE": "1"}):
                lm = FakeOpener(status, store_dir)
                self.assertEqual(serve._open_kvstore(lm), want)
                self.assertEqual(lm.calls, 1)

    def test_the_serve_opens_the_store_after_the_drafter_decision(self):
        """The cache key includes the drafter, so the store opens only once the drafter is loaded (or failed to load):
        opened before, it would be re-keyed, and scanned again, by the first speculative request."""
        for drafter_loads in (True, False):
            with self.subTest(drafter_loads=drafter_loads):
                order = []

                class Client:
                    def __init__(self, sock):
                        order.append("connect")

                    def model_id(self):
                        order.append("model_id")
                        return 2

                    def dflash_load(self, d):
                        order.append("dflash_load")
                        if not drafter_loads:
                            raise RuntimeError("no drafter here")

                    def session_open(self):
                        order.append("session_open")
                        return 1, "/kv/muse-1"

                protocol = SimpleNamespace(kind="atem", default_model_name="muse", stop_ids=[7])
                env = {"NOMOS_ENGINE_SOCKET": "/nonexistent.sock", "NOMOS_SERVE_SPEC": "1", "DFLASH_DIR": "/drafter",
                       "NOMOS_KVSTORE": "1"}
                with mock.patch.dict(os.environ, env), mock.patch.object(serve, "AutoTokenizer"), \
                        mock.patch.object(serve, "load_serve_protocol", return_value=protocol), \
                        mock.patch("kernel_client.KernelClient", Client), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    eng = serve.Engine()
                self.assertEqual(order, ["connect", "model_id", "dflash_load", "session_open"])
                self.assertEqual(eng.kvstore, {"status": "open", "dir": "/kv/muse-1"})
                self.assertIn("[serve] kvstore open dir=/kv/muse-1", out.getvalue())
                self.assertIn("[serve] NOMOS_MAX_SEQ=", out.getvalue())

    def test_health_reports_the_store(self):
        kv = {"status": "open", "dir": "/kv/muse-1"}
        eng = SimpleNamespace(lock=threading.Lock(), model_name="muse", spec_loaded=True, kvstore=kv,
                              lm=SimpleNamespace(cache_len=lambda: 0))
        previous = getattr(serve.app.state, "engine", None)
        serve.app.state.engine = eng
        try:
            self.assertEqual(serve.health()["kvstore"], kv)
            with eng.lock:                                   # a generation is running: answered without the engine
                self.assertEqual(serve.health()["kvstore"], kv)
        finally:
            if previous is None:
                del serve.app.state.engine
            else:
                serve.app.state.engine = previous


if __name__ == "__main__":
    unittest.main()
