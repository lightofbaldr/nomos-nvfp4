#!/usr/bin/env python3
"""A store that cannot run leaves the engine running, and says why. Each case runs in its own child process: the GPU
holds one Glimmer at a time and the parent never loads the model.

1. The session primitives refuse a KV they cannot copy (ruling R20): unquantized KV (NOMOS_KV_QUANT=0, no INT8/INT4
   components to address) and the sliding-window ring (NOMOS_KV_SWA=1, slot = pos % cache_cap, not pos). With
   NOMOS_KVSTORE=1 the session FFI reports the same refusal as -12 and -11, from status, open and prepare, and touches
   nothing.
2. Another process holding the store's directory turns the store off for the engine's lifetime (spec §3.1, plan note
   11): the launch-time open returns -2 (ruling R79), one "[kvstore] DISABLED for this process" line, prepare returns 0
   from then on, and every turn is a plain fresh prefill.

GPU; run alone: pixi run --no-install python3 tools/test_kv_session_unsupported.py -v"""
import os
import shutil
import subprocess
import sys
import unittest

from kvsession_testlib import ROOT, store_dir_name

CASES = {"swa": {"NOMOS_KV_SWA": "1"}, "bf16": {"NOMOS_KV_QUANT": "0"}, "locked": {}}
REFUSED = {"swa": -11, "bf16": -12}
# The other process: takes the directory's lock and holds it until its stdin closes (so it dies with the test).
HOLDER = ("import fcntl, os, sys\n"
          "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
          "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
          "print('locked', flush=True)\n"
          "sys.stdin.read()\n")


def case_dir(case):
    return ROOT / f"build/kvstore_unsupported_{case}"


def child(case):
    import numpy as np
    from kvsession_testlib import Kernel, doc_tokens
    d = case_dir(case)
    k = Kernel({**CASES[case], "NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(d)})
    problems = []
    if case == "locked":
        ids = doc_tokens(300)
        opened = k.session_open()       # the launch-time open finds the directory held: the store is off for this process
        if opened != (-2, ""):
            problems.append(f"nomos_session_open returned {opened}, expected (-2, '')")
        c = k.prepare(ids)              # and stays off: prepare starts afresh, and says nothing more
        if c != 0:
            problems.append(f"the first prepare returned {c}, expected 0")
        c, gen = k.turn(ids, 8)         # from here on a plain fresh prefill (turn checks that the KV is exactly ids[:c])
        if c != 0:
            problems.append(f"a turn reused {c} tokens with the store off")
        if k.prepare(ids + gen) != 0:
            problems.append("prepare returned a reuse with the store off")
        if k.commit() != 0:
            problems.append("commit did not return 0 with the store off")
        if any(k.stats().values()):
            problems.append(f"the counters moved with the store off: {k.stats()}")
        if k.status() != 1:             # the configuration is fine: status says enabled and supported, not "working"
            problems.append(f"status returned {k.status()}, expected 1")
        if list((d / store_dir_name()).glob("*.nkvs")):
            problems.append("a session file was written with the store off")
        k.shutdown()
        assert not problems, "; ".join(problems)
        return
    ids = doc_tokens(100)
    bpt = k.bytes_per_token()
    if bpt != -1:
        problems.append(f"nomos_kv_bytes_per_token returned {bpt}, expected -1")
    k.prefill(ids)
    # 1 MiB: more than 10 tokens of any layout (27,456 B/token at most), so a copy that wrongly proceeds cannot
    # overrun it.
    buf = np.zeros(1 << 20, dtype=np.uint8)
    rc = k.lib.nomos_kv_export(k.h, 0, 10, buf.ctypes.data)
    if rc == 0:
        problems.append("nomos_kv_export(0, 10) returned 0, expected a refusal")
    rc = k.import_(100, 110, buf, 10, ids[:10])
    if rc == 0:
        problems.append("nomos_kv_import(100, 110) returned 0, expected a refusal")
    want = REFUSED[case]
    if k.status() != want:
        problems.append(f"nomos_session_status returned {k.status()}, expected {want}")
    if k.session_open() != (want, ""):
        problems.append(f"nomos_session_open returned {k.session_open()}, expected ({want}, '')")
    c = k.prepare(ids)
    if c != want:
        problems.append(f"nomos_session_prepare returned {c}, expected {want}")
    if k.commit() != 0:
        problems.append("nomos_session_commit did not return 0 for a refused store")
    if k.cache_len() != 100:
        problems.append(f"cache_len is {k.cache_len()} after the refused calls, expected 100")
    if k.live_tokens() != ids:
        problems.append("live_tokens changed across the refused calls")
    if d.exists() and any(d.iterdir()):
        problems.append("a refused store still created files")
    k.shutdown()
    assert not problems, "; ".join(problems)


class UnsupportedKvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for case in CASES:                                               # nothing is kept (final review M11)
            cls.addClassCleanup(shutil.rmtree, case_dir(case), ignore_errors=True)

    def _run(self, case, fresh=True):
        if fresh:
            shutil.rmtree(case_dir(case), ignore_errors=True)
        r = subprocess.run([sys.executable, __file__, case], cwd=ROOT / "tools", env={**os.environ},
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return r

    def test_swa_ring_is_refused(self):
        self._run("swa")

    def test_unquantized_kv_is_refused(self):
        self._run("bf16")

    def test_a_directory_held_by_another_process_turns_the_store_off(self):
        d = case_dir("locked")
        shutil.rmtree(d, ignore_errors=True)
        store = d / store_dir_name()       # where the engine's store will look: the predicted name is part of the test
        store.mkdir(parents=True)
        with subprocess.Popen([sys.executable, "-c", HOLDER, str(store / ".lock")], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as holder:     # leaving the block closes its stdin: it lets go
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            r = self._run("locked", fresh=False)     # its directory exists already, and is held
        self.assertEqual(r.stdout.count("[kvstore] DISABLED for this process"), 1, r.stdout[-3000:])   # logged once


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in CASES:
        child(sys.argv[1])
    else:
        unittest.main()
