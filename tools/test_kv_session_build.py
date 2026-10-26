#!/usr/bin/env python3
"""BUILD_ID baked into libnomos_kernel-muse.so matches its sources (the KV session store's cache key)."""
import ctypes
import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def expected_build_id() -> int:
    skip = {"build_stamp.mojo", "model_config.mojo"}
    rest = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("lib/**/*.mojo") if p.name not in skip)
    h = hashlib.sha256()
    for rel in ["lib/model_config.mojo", "nomos_ffi.mojo", "pixi.lock", *rest]:     # pixi.lock: the toolchain
        h.update((ROOT / rel).read_bytes())
    return int(h.hexdigest()[:16], 16)


class BuildIdTest(unittest.TestCase):
    def setUp(self):
        self.lib = ctypes.CDLL(str(ROOT / "libnomos_kernel-muse.so"))

    def test_build_id_matches_sources(self):
        self.lib.nomos_build_id.restype = ctypes.c_int64
        self.assertEqual(self.lib.nomos_build_id() & 0xFFFFFFFFFFFFFFFF, expected_build_id())

    def test_muse_supports_the_store(self):
        self.lib.nomos_session_supported.restype = ctypes.c_int32
        self.assertEqual(self.lib.nomos_session_supported(), 1)


if __name__ == "__main__":
    unittest.main()
