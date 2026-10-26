#!/usr/bin/env python3
"""Spec test 4 (and test 10 in Task 13): a session survives a process restart. The parent never loads the model:
each phase runs in its own child process (the GPU holds one Glimmer at a time). A restart under other numerics restores
nothing (ledger R35). GPU; run alone."""
import json
import os
import shutil
import subprocess
import sys
import unittest

from kvsession_testlib import ROOT, store_dir_name

DIR = ROOT / "build/kvstore_restart_test"
OUT = ROOT / "build/kvstore_restart_test.json"
OTHER = {"NOMOS_W4A4": "0"}   # NVFP4 weights with bf16 activations (W4A16): the same build and weights, other K/V values


def host_lm(k):
    """pyhost's KernelLM over the engine this process loaded (one Glimmer per process): the speculative phases run
    through host.spec_generate_stream, the loop the serve runs, so they exercise the host's use of the store and not only
    the kernel's. The testlib's own spec_trace, which test_kv_session_dflash uses beside it, never calls it."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "pyhost"))
    from host import KernelLM
    lm = KernelLM.__new__(KernelLM)
    lm.lib, lm.h, lm.vocab_size = k.lib, k.h, 202048
    lm._logits = np.zeros(202048, dtype=np.float32)
    lm._dflash_loaded = lm._session_ok = True      # __new__ skips __init__, which sets these
    return lm


def child(phase, spec):
    from kvsession_testlib import STOP, Kernel, doc_tokens
    env = {"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR)}
    if phase == "other":
        env.update(OTHER)
    k = Kernel(env)
    if spec:
        k.dflash_load()
    a = doc_tokens(500)
    if phase == "write":
        if spec:
            gen = list(host_lm(k).spec_generate_stream(a, max_new_tokens=32, stop_ids=STOP, vb=8))   # prepare ... commit
        else:
            _, gen = k.turn(a, 32)
        ids = a + gen
        c = len(ids) - 1
        if spec:
            # The session holds len(ids) - 2 tokens when the loop's last token came from its exact M=1 step: R54 drops
            # that step's KV row (host.spec_generate_stream), and KV and drafter end level there.
            c = min(c, k.dflash_len())
        assert k.truncate(c) == 0                        # the never-parked reference runs the resume's op sequence
        lg = k.prefill_cont(ids[c:])
        ref = k.spec_trace(lg, 32)[0] if spec else k.greedy(lg, 32)
        OUT.write_text(json.dumps({"gen": gen, "c": c, "ref": ref}))
    elif phase == "read":
        saved = json.loads(OUT.read_text())
        ids = a + saved["gen"]
        if spec:
            trace = []
            got = list(host_lm(k).spec_generate_stream(ids, max_new_tokens=32, stop_ids=STOP, vb=8, trace=trace))
            c = k.stats()["reused_tokens"]               # the one prepare of this process
            assert trace, "the resumed turn decoded on the base path instead of speculating"
        else:
            c = k.prepare(ids)
        assert c == saved["c"], f"restored {c}, expected {saved['c']}"
        assert k.stats()["restores_nvme"] == 1
        if not spec:
            got = k.greedy(k.prefill_cont(ids[c:]), 32)
        assert got == saved["ref"], "the resumed continuation must equal the never-parked one"
    else:
        # The conversation the "write" phase saved, asked for under other numerics: nothing may come back.
        assert phase == "other", phase
        saved = json.loads(OUT.read_text())
        c = k.prepare(a + saved["gen"], spec=spec)
        st = k.stats()
        assert c == 0, f"restored {c} tokens whose KV was computed under other numerics"
        assert st["restores_nvme"] == 0 and st["restores_ram"] == 0 and st["reused_tokens"] == 0, st
    k.shutdown()


class RestartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(OUT.unlink, missing_ok=True)
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)    # the sessions are not kept (final review M11)

    def _phase(self, phase, spec):
        r = subprocess.run([sys.executable, __file__, phase, "1" if spec else "0"], cwd=ROOT / "tools",
                           env={**os.environ}, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

    def _run(self, spec):
        shutil.rmtree(DIR, ignore_errors=True)
        for phase in ("write", "read"):
            self._phase(phase, spec)

    def test_restart_restores_from_nvme(self):
        self._run(spec=False)

    def test_restart_restores_from_nvme_with_dflash(self):       # spec test 10
        self._run(spec=True)

    def test_other_numerics_restore_nothing(self):
        """Ledger R35, design §3.6: KV computed under NOMOS_W4A4=1 is not restored by a restart of the same build on the
        same weights under NOMOS_W4A4=0. The two numerics keep separate store directories (so they share no files, locks
        or budgets), and the first still restores its own session afterwards."""
        shutil.rmtree(DIR, ignore_errors=True)
        self._phase("write", False)
        self._phase("other", False)
        self._phase("read", False)
        self.assertEqual(sorted(p.name for p in DIR.glob("muse-*")), sorted([store_dir_name(), store_dir_name(OTHER)]))
        self.assertEqual(list((DIR / store_dir_name(OTHER)).glob("*.nkvs")), [])     # nothing was saved under the new numerics
        self.assertEqual(len(list((DIR / store_dir_name()).glob("*.nkvs"))), 1)      # the original session is untouched


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("write", "read", "other"):
        child(sys.argv[1], sys.argv[2] == "1")
    else:
        unittest.main()
