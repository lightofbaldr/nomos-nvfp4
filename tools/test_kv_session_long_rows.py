#!/usr/bin/env python3
"""Drafter rows past the old 8,192-row limit survive the store's park and restore, from RAM and from NVMe, bit for
bit. Task 15 made such rows possible; no earlier test exported or imported one.
GPU; run alone: pixi run --no-install python3 tools/test_kv_session_long_rows.py -v"""
import shutil
import unittest

from kvsession_testlib import ROOT, Kernel, doc_tokens

DIR = ROOT / "build/kvstore_long_rows_test"
ENV = {"NOMOS_MAX_SEQ": "16384", "NOMOS_PREFILL_CHUNK": "256", "NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR)}
N = 9000          # past 8,192; doc_tokens supplies about 9K
A, B = 8100, N    # the compared range: 92 rows before the old limit, 808 after it


class DrafterRowsPastEightKThroughTheStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)    # about 1.2 GB of sessions (final review M11)

    def test_rows_past_8192_round_trip_from_ram_and_nvme(self):
        shutil.rmtree(DIR, ignore_errors=True)
        ids = doc_tokens(N)
        # As long as ids and sharing no prefix with it: switching to it parks ids AND overwrites every drafter row a
        # restore must bring back. A shorter one would leave rows past 8,192 untouched in GPU memory, and a RAM
        # restore that skipped them would still pass.
        other = ids[::-1]
        self.assertNotEqual(other[0], ids[0])
        k = Kernel(ENV)
        k.dflash_load()
        k.turn(ids, 0, spec=True)                         # 9,000 tokens of KV and drafter rows, committed
        self.assertEqual(k.dflash_len(), N)
        before = k.export(A, B)                           # target KV + drafter rows, as the store writes them
        k.turn(other, 0, spec=True)                       # park ids in RAM
        ram = k.stats()["restores_ram"]
        self.assertEqual(k.prepare(ids + [ids[-1]], spec=True), N)    # RAM restore
        self.assertEqual(k.stats()["restores_ram"] - ram, 1, "this restore must be served from the RAM tier")
        self.assertEqual(k.export(A, B).tobytes(), before.tobytes())
        k.turn(other + [other[-1]], 0, spec=True)         # park it again, then a new engine in this process: an NVMe restore
        k.shutdown()
        k2 = Kernel(ENV)
        k2.dflash_load()
        nvme = k2.stats()["restores_nvme"]                # zeros: a new engine has no store, and so no RAM tier, yet
        self.assertEqual(k2.prepare(ids + [ids[-1]], spec=True), N)
        self.assertEqual(k2.stats()["restores_nvme"] - nvme, 1, "this restore must be served from the NVMe tier")
        self.assertEqual(k2.export(A, B).tobytes(), before.tobytes())
        self.assertEqual(k2.dflash_len(), N)
        k2.shutdown()


if __name__ == "__main__":
    unittest.main()
