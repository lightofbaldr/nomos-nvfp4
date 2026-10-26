"""Unit test for the KV session store profile guard.

Run: bash tools/run_mojo_test.sh tools/kvstore_guard_test.mojo
"""
from std.testing import assert_true, assert_false
from lib.kvstore.support import store_supported


def main() raises:
    assert_true(store_supported(2))      # Muse-Glimmer
    assert_false(store_supported(1))     # Gemma
    assert_false(store_supported(3))     # Qwen (GDN state is not saved in v1)
    assert_false(store_supported(4))     # OLMo
    print("kvstore_guard_test: all passed")
