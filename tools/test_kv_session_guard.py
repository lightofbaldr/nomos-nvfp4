#!/usr/bin/env python3
"""Spec test 6: a non-Muse build refuses the KV session store. No model is loaded.
Needs SO_PATH: a non-Muse kernel built from this branch, for example SO_PATH=$PWD/libnomos_kernel-olmo3.so after
./refresh_build.sh --model olmo3 --tokenizer-dir <olmo tokenizer dir>. A kernel built before the store (the committed
production OLMo .so) has no nomos_session_supported and cannot run it. That build also rewrites lib/model_config.mojo and
publishes libnomos_kernel.so: back up the production OLMo .so first and rebuild Muse afterwards (plan Task 14, Step 4)."""
import ctypes
import os
import unittest


class GuardTest(unittest.TestCase):
    def test_non_muse_build_refuses_the_store(self):
        so = os.environ.get("SO_PATH")
        if not so:
            self.fail("SO_PATH is not set: point it at a non-Muse kernel built from this branch (see this file's docstring)")
        lib = ctypes.CDLL(so)
        if not hasattr(lib, "nomos_session_supported"):
            self.fail(f"{so} has no nomos_session_supported: it was built before the KV session store. "
                      "Build a non-Muse kernel from this branch and point SO_PATH at that")
        lib.nomos_model_id.restype = ctypes.c_int32
        lib.nomos_session_supported.restype = ctypes.c_int32
        self.assertNotEqual(lib.nomos_model_id(), 2)
        self.assertEqual(lib.nomos_session_supported(), -10)


if __name__ == "__main__":
    unittest.main()
