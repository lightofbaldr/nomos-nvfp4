#!/usr/bin/env python3
"""tools/kvstore_dirs.sh, the KV session store's clean-up tool (docs/RUNBOOK.md §7; final review C1). CPU only, seconds.

The live directory must never be removed, however NOMOS_KVSTORE_DIR was spelled when the daemon named it (a trailing slash,
a symlinked root, a path relative to the daemon's working directory) and however the operator names it to the tool.
Listing changes nothing; removing a named directory that is not live works; the live one, anything outside the root, a
directory whose store is running, and anything at all when no authoritative source names the live directory, are refused.

Everything lives in a temporary directory under build/. The tool is always given --port and --log, so it never asks a real
serve or reads the real daemon.log, and it runs with a temporary HOME.
Run: python3 tools/test_kvstore_dirs.py -v"""
import http.server
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools/kvstore_dirs.sh"
AA, BB, CC = "muse-00000000000000aa", "muse-00000000000000bb", "muse-00000000000000cc"
# Another process with the store open: holds the directory's .lock until its stdin closes.
HOLDER = ("import fcntl, os, sys\n"
          "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
          "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
          "print('locked', flush=True)\n"
          "sys.stdin.read()\n")


def free_port():
    """A port nothing listens on: the serve is down."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeServe:
    """GET /health answering `body` as JSON, on an ephemeral port."""

    def __init__(self, body):
        payload = json.dumps(body).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class KvstoreDirsTest(unittest.TestCase):
    def setUp(self):
        (REPO / "build").mkdir(exist_ok=True)
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="kvstore_dirs_", dir=REPO / "build"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "kv"
        for name, size in ((AA, 3000), (BB, 5000), (CC, 7000)):
            d = self.root / name
            d.mkdir(parents=True)
            (d / ".lock").touch()
            (d / "00000000000000a1.nkvs").write_bytes(b"\0" * size)
        self.live = self.root / AA
        (self.tmp / "link").symlink_to(self.root)                      # a symlinked NOMOS_KVSTORE_DIR
        (self.tmp / "home").mkdir()
        self.log = self.tmp / "daemon.log"
        self.port = free_port()

    def run_tool(self, *args, port=None, log=None, cwd=None):
        env = {"PATH": os.environ["PATH"], "HOME": str(self.tmp / "home")}
        return subprocess.run(["bash", str(TOOL), *args, "--port", str(port or self.port), "--log", str(log or self.log)],
                              cwd=cwd or self.tmp, env=env, capture_output=True, text=True, timeout=60)

    def daemon_named(self, spelled):
        """The daemon's log, its newest store line naming the live directory spelled this way (an older line names BB)."""
        self.log.write_text(f"[kvstore] dir={self.root}/{BB}\n[daemon] READY on /tmp/x.sock  (kvstore status=1)\n"
                            f"[kvstore] dir={spelled}\n[daemon] client connected\n")

    def snapshot(self):
        """Every path under the temporary directory, with its type, size and modification time (links not followed)."""
        out = {}
        for dirpath, dirnames, filenames in os.walk(self.tmp):
            for name in dirnames + filenames:
                p = os.path.join(dirpath, name)
                st = os.lstat(p)
                out[p] = (st.st_mode, st.st_size, st.st_mtime_ns)
        return out

    def assert_live_intact(self):
        self.assertTrue((self.live / ".lock").exists() and (self.live / "00000000000000a1.nkvs").exists(),
                        "the live directory was touched")

    # -- the live directory, however it is spelled ----------------------------------------------------------------------
    def spellings(self):
        """How the daemon may have named the live directory: the kernel prints NOMOS_KVSTORE_DIR + "/muse-<id>" as given."""
        return {"canonical": str(self.live),
                "trailing slash on the root": f"{self.root}//{AA}",
                "symlinked root": f"{self.tmp}/link/{AA}",
                "relative to the daemon's working directory (the repo)": os.path.relpath(self.live, REPO),
                "relative, with ./ and ..": "./" + os.path.relpath(self.root, REPO) + f"/../kv/./{AA}"}

    def test_the_live_directory_is_refused_however_either_side_spells_it(self):
        names = [AA, str(self.live), f"{self.root}//{AA}/", f"{self.tmp}/link/{AA}", f"{self.tmp}/link//{AA}/",
                 f"kv/{AA}", f"./kv/../kv/{AA}/"]                    # the last two are relative to the tool's cwd (self.tmp)
        for how, spelled in self.spellings().items():
            self.daemon_named(spelled)
            for name in names:
                with self.subTest(daemon=how, operator=name):
                    before = self.snapshot()
                    r = self.run_tool("remove", name)
                    self.assertNotEqual(r.returncode, 0, r.stdout)
                    self.assertIn("LIVE", r.stderr)
                    self.assertIn("nothing was removed", r.stderr)
                    self.assertEqual(self.snapshot(), before)
            with self.subTest(daemon=how, listing=True):
                r = self.run_tool("list")
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(f"live: {self.live}   (from the newest [kvstore] dir= line", r.stdout)
                self.assertRegex(r.stdout, rf"{AA}   LIVE\n")
        self.assert_live_intact()

    def test_a_relative_spelling_resolves_against_the_repo_not_the_callers_directory(self):
        # The same relative line, read from a directory where it would name something else: still the repo's.
        elsewhere = self.tmp / "elsewhere"
        (elsewhere / os.path.relpath(self.live, REPO)).mkdir(parents=True)   # a decoy at that path under the caller's cwd
        self.daemon_named(os.path.relpath(self.live, REPO))
        r = self.run_tool("remove", AA, cwd=elsewhere)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("LIVE", r.stderr)
        self.assert_live_intact()

    # -- listing --------------------------------------------------------------------------------------------------------
    def test_listing_changes_nothing(self):
        self.daemon_named(f"{self.root}//{AA}")
        before = self.snapshot()
        for args in ((), ("list",)):
            with self.subTest(args=args):
                r = self.run_tool(*args)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(self.snapshot(), before)
                self.assertIn(f"root: {self.root}\n", r.stdout)
                for name, size in ((AA, "3.0KB"), (BB, "4.9KB"), (CC, "6.9KB")):     # numfmt rounds up
                    self.assertRegex(r.stdout, rf"\s{size}\s+\d{{4}}-\d\d-\d\d \d\d:\d\d  {name}")
                self.assertIn("Nothing was changed", r.stdout)

    # -- removing -------------------------------------------------------------------------------------------------------
    def test_a_named_directory_that_is_not_live_is_removed(self):
        self.daemon_named(f"{self.tmp}/link/{AA}")
        for name in (BB, f"{self.tmp}/link//{CC}/"):                      # by name, and by a path through the link
            with self.subTest(name=name):
                r = self.run_tool("remove", name)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("removed: ", r.stdout)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [AA])
        self.assert_live_intact()

    def test_anything_outside_the_root_is_refused(self):
        self.daemon_named(str(self.live))
        outside = self.tmp / "elsewhere" / "muse-00000000000000ee"
        outside.mkdir(parents=True)
        (self.root / "muse-00000000000000ff").symlink_to(outside)        # inside by name, outside in fact
        (self.root / "notes").mkdir()
        (self.root / "muse-0000000000000011").write_text("a file")
        (self.root / "muse-00000000000000a2").symlink_to(self.live)      # another name for the live directory
        for name, why in ((str(outside), "not directly under the store root"),
                          ("muse-00000000000000ff", "not directly under the store root"),
                          (str(self.root), "not directly under the store root"),
                          (str(self.tmp), "not directly under the store root"),
                          (f"{self.root}/notes", "not a muse-* store directory"),
                          ("muse-0000000000000011", "not a directory"),
                          ("muse-00000000000000a2", "LIVE"),
                          ("muse-0000000000000099", "no such directory")):
            with self.subTest(name=name):
                before = self.snapshot()
                r = self.run_tool("remove", name)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(why, r.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_one_refusal_removes_nothing(self):
        self.daemon_named(str(self.live))
        r = self.run_tool("remove", BB, AA)
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue((self.root / BB).is_dir(), "an all-or-nothing request removed part of what it named")

    def test_a_directory_whose_store_is_running_is_refused(self):
        # CC is live to nobody's log, but a process has its store open: the store's own lock says so.
        self.daemon_named(str(self.live))
        with subprocess.Popen([sys.executable, "-c", HOLDER, str(self.root / CC / ".lock")], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as holder:
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            r = self.run_tool("remove", CC)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("a running store holds it", r.stderr)
        self.assertTrue((self.root / CC / "00000000000000a1.nkvs").exists())

    def test_a_directory_that_contains_the_live_one_is_refused(self):
        # A store root that is itself named muse-*, seen from its parent: removing it would take the live directory with
        # it, even while that store runs (the lock it holds is the live directory's, not the root's). Final re-review m3.
        outer = self.tmp / "muse-outer"
        live = outer / "muse-00000000000000dd"
        live.mkdir(parents=True)
        (live / ".lock").touch()
        (live / "00000000000000d1.nkvs").write_bytes(b"\0" * 1000)
        self.daemon_named(str(live))
        with subprocess.Popen([sys.executable, "-c", HOLDER, str(live / ".lock")], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as holder:
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            r = self.run_tool("remove", "muse-outer", "--root", str(self.tmp))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("contains the LIVE store directory", r.stderr)
        self.assertTrue((live / "00000000000000d1.nkvs").exists(), "the live store went with the directory around it")

    # -- the authoritative source ---------------------------------------------------------------------------------------
    def test_with_no_authoritative_source_removal_refuses(self):
        for how in ("no daemon log", "a daemon log that names no store directory"):
            with self.subTest(how):
                if how == "no daemon log":
                    self.log.unlink(missing_ok=True)
                else:
                    self.log.write_text("[daemon] READY on /tmp/x.sock  (kvstore status=0)\n")
                before = self.snapshot()
                r = self.run_tool("remove", BB)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("refusing to remove anything: no authoritative live directory", r.stderr)
                self.assertEqual(self.snapshot(), before)
                r = self.run_tool("list", "--root", str(self.root))     # listing still works, and says so
                self.assertEqual(r.returncode, 0)
                self.assertIn("live: UNKNOWN", r.stdout)

    def test_the_serves_health_is_the_source_when_the_serve_answers(self):
        self.daemon_named(f"{self.root}/{BB}")                          # a stale log: the serve knows better
        serve = FakeServe({"status": "ok", "ready": True, "kvstore": {"status": "open", "dir": f"{self.tmp}/link//{AA}"}})
        self.addCleanup(serve.close)
        r = self.run_tool("remove", AA, port=serve.port)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the LIVE store directory, from the serve's /health", r.stderr)
        r = self.run_tool("remove", BB, port=serve.port)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assert_live_intact()

    def test_a_serve_without_the_kvstore_field_falls_back_to_the_log(self):
        self.daemon_named(f"{self.root}//{AA}")
        serve = FakeServe({"status": "ok", "ready": True})              # a serve from before /health reported the store
        self.addCleanup(serve.close)
        r = self.run_tool("remove", AA, port=serve.port)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("from the newest [kvstore] dir= line", r.stderr)
        self.assert_live_intact()

    def test_remove_never_picks_the_directories(self):
        self.daemon_named(str(self.live))
        before = self.snapshot()
        r = self.run_tool("remove")
        self.assertEqual(r.returncode, 2)
        self.assertIn("never picks them", r.stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
