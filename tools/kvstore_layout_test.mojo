"""Unit tests for the KV session chunk layout.

Run: bash tools/run_mojo_test.sh tools/kvstore_layout_test.mojo
"""
from std.testing import assert_equal, assert_true, assert_false
from lib.kvstore.layout import KvLayout, ComponentDesc, target_layout, KIND_DRAFT_K, KIND_K_DATA
from lib.kvstore.support import KV_MODE_INT8, KV_MODE_INT4_B32


def _glimmer(mode: Int) -> KvLayout:
    var nkv = List[Int]()
    var hd = List[Int]()
    for _ in range(52):
        nkv.append(2)
        hd.append(128)
    return target_layout(nkv, hd, mode)


def test_glimmer_bytes_per_token() raises:
    assert_equal(_glimmer(KV_MODE_INT8).bytes_per_token(), 27456)       # spec §2
    assert_equal(_glimmer(KV_MODE_INT4_B32).bytes_per_token(), 15808)


def test_offsets_and_chunk_bytes() raises:
    var lay = KvLayout()
    lay.comps.append(ComponentDesc(KIND_K_DATA, 0, 2, 3))   # 6 bytes per token
    lay.comps.append(ComponentDesc(KIND_K_DATA, 1, 1, 4))   # 4 bytes per token
    assert_equal(lay.chunk_bytes(10), 100)
    assert_equal(lay.component_offset(0, 10), 0)
    assert_equal(lay.component_offset(1, 10), 60)
    assert_false(lay.has_drafter())
    lay.comps.append(ComponentDesc(KIND_DRAFT_K, 0, 1, 8))
    assert_true(lay.has_drafter())


def test_hash_tracks_every_field() raises:
    var a = _glimmer(KV_MODE_INT8)
    var b = _glimmer(KV_MODE_INT8)
    assert_equal(a.hash(), b.hash())
    b.comps[7].row_bytes += 1
    assert_true(a.hash() != b.hash())
    assert_true(_glimmer(KV_MODE_INT8).hash() != _glimmer(KV_MODE_INT4_B32).hash())


def main() raises:
    test_glimmer_bytes_per_token()
    test_offsets_and_chunk_bytes()
    test_hash_tracks_every_field()
    print("kvstore_layout_test: all passed")
