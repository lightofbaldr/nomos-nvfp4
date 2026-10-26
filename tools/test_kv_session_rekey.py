#!/usr/bin/env python3
"""Review Focus 3: loading DFlash after the store exists re-keys it; old-layout sessions are never imported.
Also the store with a drafter loaded: its files record the drafter's layout and id, a parked session brings its drafter
rows back, another drafter never shares its sessions, and a failed drafter reload restores the no-drafter key.
GPU; run alone."""
import ctypes
import json
import os
import pathlib
import shutil
import struct
import unittest

import numpy as np

from kvsession_testlib import BOS, DFLASH_DIR, EXPECTED_BPT, ROOT, Kernel, doc_tokens, store_dir_name, weights_id_of

DIR = ROOT / "build/kvstore_rekey_test"
OTHER_DRAFTER = ROOT / "build/kvstore_rekey_other_drafter"
ENV = {"NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR)}


def header(path):
    """(drafter_id, components) from a session file's header (lib/kvstore/disk_tier.mojo): u32 magic, version, model_id,
    kv_mode | u64 build_id, layout_hash, variant_hash, drafter_id | u32 n | n x (u32 kind, layer, heads, row_bytes)."""
    with open(path, "rb") as f:
        raw = f.read(16384)
    drafter_id, n = struct.unpack_from("<QI", raw, 40)
    return drafter_id, [struct.unpack_from("<4I", raw, 52 + 16 * i) for i in range(n)]


def other_drafter():
    """A drafter directory that loads exactly like DFLASH_DIR (every weight file linked) but whose convert_manifest.json
    differs, so its id differs: the weights' identity is the manifest plus the files' names and sizes."""
    shutil.rmtree(OTHER_DRAFTER, ignore_errors=True)
    OTHER_DRAFTER.mkdir(parents=True)
    for name in os.listdir(DFLASH_DIR):
        if name != "convert_manifest.json":
            os.symlink(os.path.join(DFLASH_DIR, name), OTHER_DRAFTER / name)
    manifest = json.loads((pathlib.Path(DFLASH_DIR) / "convert_manifest.json").read_text())
    manifest["kvstore_test_variant"] = "other"
    (OTHER_DRAFTER / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(OTHER_DRAFTER)


class RekeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Each test shuts its engine down in its own cleanup, which runs before these. OTHER_DRAFTER holds symlinks to
        # DFLASH_DIR's files and a manifest: rmtree removes the links, never the files they point to.
        cls.addClassCleanup(shutil.rmtree, OTHER_DRAFTER, ignore_errors=True)
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)       # the sessions are not kept (final review M11)

    def test_drafter_load_rekeys_the_store(self):
        shutil.rmtree(DIR, ignore_errors=True)
        k = Kernel(ENV)
        self.addCleanup(k.shutdown)
        a = doc_tokens(600)
        k.turn(a, 8)                                  # a session under the no-drafter key
        k.dflash_load()
        c = k.prepare(a + doc_tokens(620)[600:], spec=True)
        self.assertEqual(c, 0)                        # the old-layout session is not visible under the new key
        self.assertEqual(k.stats()["restores_ram"] + k.stats()["restores_nvme"], 0)
        self.assertEqual(k.cache_len(), 0)
        # The three checks above hold without any re-key too: a speculative request is capped at the old session's
        # drafter length, which is 0. The next two tell a re-key from that cap.
        self.assertEqual(k.stats()["prepares"], 1)    # the counters belong to the store, so a new store starts at zero
        c, _ = k.turn(a, 8)                           # the prompt the old session holds, as a base request: no cap
        self.assertEqual(c, 0)
        files = sorted((DIR / store_dir_name()).glob("*.nkvs"))
        self.assertEqual(len(files), 2)               # the old key's file is kept, and the new key has its own
        drafter_id = weights_id_of(DFLASH_DIR)
        by_id = {header(p)[0]: header(p)[1] for p in files}
        self.assertEqual(sorted(by_id), [0, drafter_id])
        self.assertTrue(all(kind < 6 for kind, *_ in by_id[0]), "the old file has no drafter components")
        drafter_comps = [(6 + i % 2, i // 2, 1, 4096) for i in range(10)]   # K, V of each of the 5 layers: 1024 fp32 a row
        self.assertEqual(by_id[drafter_id][-10:], drafter_comps)

    def test_a_parked_session_restores_its_drafter_rows(self):
        shutil.rmtree(DIR, ignore_errors=True)
        k = Kernel(ENV)
        self.addCleanup(k.shutdown)
        k.dflash_load()                               # before the store exists: it is created with the drafter's layout
        a = doc_tokens(400)
        other = [BOS] + doc_tokens(900)[501:]         # 400 different tokens
        k.turn(a, 8)                                  # decode_step does not feed the drafter, which ends up to 8 rows behind
        held = k.live_tokens()
        self.assertEqual(k.dflash_len(), 400)
        ref = k.export(0, 400)
        self.assertEqual(len(ref), 400 * (EXPECTED_BPT + 40960))
        k.turn(other, 4)                              # parks a's session, and the drafter's rows now belong to `other`
        self.assertEqual(k.dflash_len(), 400)
        c = k.prepare(held + doc_tokens(1000)[900:910], spec=False)
        self.assertEqual(c, len(held))
        self.assertEqual(k.stats()["restores_ram"], 1)
        self.assertEqual(k.live_tokens(), held)
        self.assertEqual(k.dflash_len(), 400)         # context for what the drafter had, not for the rows after it
        self.assertTrue(np.array_equal(k.export(0, 400), ref))   # target KV and drafter rows, byte for byte

    def test_a_different_drafter_never_shares_sessions(self):
        """Design 3.4, 'the drafter's weights change: drafter_id is in the key'. Loading another drafter re-keys the
        store, so the first drafter's sessions are not imported; loading the first one again finds them where they were."""
        shutil.rmtree(DIR, ignore_errors=True)
        other = other_drafter()
        self.assertNotEqual(weights_id_of(other), weights_id_of(DFLASH_DIR))
        k = Kernel(ENV)
        self.addCleanup(k.shutdown)
        k.dflash_load()
        k.turn(doc_tokens(400), 8)                    # a session under the first drafter's key
        held = k.live_tokens()
        ids = held + doc_tokens(1000)[900:910]
        k.dflash_load(other)                          # releases the first drafter, loads the other one
        self.assertEqual(k.prepare(ids, spec=False), 0)
        self.assertEqual(k.stats()["prepares"], 1)    # a new store
        k.dflash_load()                               # back to the first drafter: its key, and its session on disk
        self.assertEqual(k.prepare(ids, spec=False), len(held))
        self.assertEqual(k.stats()["restores_nvme"], 1)
        self.assertEqual(k.dflash_len(), 400)         # its drafter rows came back with it
        self.assertEqual(len(list((DIR / store_dir_name()).glob("*.nkvs"))), 1)   # the other drafter saved nothing

    def test_a_failed_reload_leaves_no_drafter_and_no_drafter_id(self):
        """nomos_dflash_load releases the loaded drafter before it looks for the new weights. When it finds none the engine
        has no drafter, so its key must be the no-drafter key again: the session stored under it is still found."""
        shutil.rmtree(DIR, ignore_errors=True)
        k = Kernel(ENV)
        self.addCleanup(k.shutdown)
        k.turn(doc_tokens(400), 8)                    # under the no-drafter key
        held = k.live_tokens()
        k.dflash_load()
        self.assertEqual(k.bytes_per_token(), EXPECTED_BPT + 40960)
        nowhere = ctypes.create_string_buffer(b"/nonexistent/drafter/")
        self.assertEqual(k.lib.nomos_dflash_load(k.h, ctypes.addressof(nowhere)), -2)
        self.assertEqual(k.bytes_per_token(), EXPECTED_BPT)
        self.assertEqual(k.dflash_len(), -2)          # no drafter
        self.assertEqual(k.prepare(held + doc_tokens(1000)[900:910], spec=False), len(held))
        self.assertEqual(k.stats()["prepares"], 2)    # the same store throughout: the key never changed


if __name__ == "__main__":
    unittest.main()
