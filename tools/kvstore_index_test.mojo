"""Unit tests for the KV session cache key and index.

Run: bash tools/run_mojo_test.sh tools/kvstore_index_test.mojo
"""
from std.os import listdir, makedirs, remove, rmdir
from std.os.path import isdir
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from lib.kvstore.fileio import O_RDWR, O_CREAT, open_file, close_file, pwrite_all, random_u64
from lib.kvstore.hashing import fnv1a64_tokens, hex16
from lib.kvstore.key import CacheKey, weights_id_of, numerics_id_of
from lib.kvstore.index import BLOCK, block_hashes, lcp, sub, SessionIndex, SessionRecord, ChunkInfo


def _toks(n: Int, salt: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append((i * 31 + salt * 1000003) % 200000)
    return out^


def _rec(id: UInt64, tokens: List[Int], drafter_len: Int) -> SessionRecord:
    var chunks = List[ChunkInfo]()
    chunks.append(ChunkInfo(0, len(tokens), drafter_len, -1, UInt64(0)))
    return SessionRecord(id, tokens.copy(), chunks^, drafter_len, 0, False)


def test_key_seed_tracks_every_field() raises:
    var k = CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(0), UInt64(0))
    var seeds = List[UInt64]()
    seeds.append(k.seed())
    seeds.append(CacheKey(3, UInt64(7), 1, UInt64(9), UInt64(0), UInt64(0)).seed())
    seeds.append(CacheKey(2, UInt64(8), 1, UInt64(9), UInt64(0), UInt64(0)).seed())
    seeds.append(CacheKey(2, UInt64(7), 2, UInt64(9), UInt64(0), UInt64(0)).seed())
    seeds.append(CacheKey(2, UInt64(7), 1, UInt64(10), UInt64(0), UInt64(0)).seed())
    seeds.append(CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(1), UInt64(0)).seed())
    seeds.append(CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(0), UInt64(1)).seed())
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert_true(seeds[i] != seeds[j])
    assert_true(k == CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(0), UInt64(0)))


def test_block_hashes_are_prefix_stable() raises:
    var t = _toks(800, 1)
    var h512 = block_hashes(UInt64(5), t, 512)
    var h800 = block_hashes(UInt64(5), t, 800)
    assert_equal(len(h512), 2)
    assert_equal(len(h800), 3)
    assert_equal(h512[0], h800[0])
    assert_equal(h512[1], h800[1])
    assert_true(block_hashes(UInt64(6), t, 256)[0] != h800[0])   # the seed (cache key) matters


def test_longest_match_full_blocks_then_tokens() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(600, 1)
    idx.add(_rec(UInt64(1), a, 0))
    var req = List[Int]()
    for i in range(550):
        req.append(a[i])
    for i in range(20):
        req.append(199999 - i)
    var m = idx.longest_match(req, len(req) - 1, UInt64(0), False)
    assert_equal(m[0], 0)
    assert_equal(m[1], 550)


def test_short_prompt_matches_without_a_full_block() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(100, 2)
    idx.add(_rec(UInt64(1), a, 0))
    var req = List[Int]()
    for i in range(80):
        req.append(a[i])
    req.append(7)
    assert_equal(idx.longest_match(req, len(req) - 1, UInt64(0), False)[1], 80)


def test_excluded_session_is_skipped() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(300, 3)
    idx.add(_rec(UInt64(1), a, 0))
    var m = idx.longest_match(a, len(a) - 1, UInt64(1), False)
    assert_equal(m[0], -1)
    assert_equal(m[1], 0)


def test_spec_caps_at_drafter_len() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(600, 4)
    idx.add(_rec(UInt64(1), a, 300))
    assert_equal(idx.longest_match(a, len(a) - 1, UInt64(0), True)[1], 300)
    assert_equal(idx.longest_match(a, len(a) - 1, UInt64(0), False)[1], 599)


def test_hash_hits_are_verified_by_tokens() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(512, 5)
    var b = _toks(512, 6)
    idx.add(_rec(UInt64(1), a, 0))
    idx.add(_rec(UInt64(2), b, 0))
    # Simulate a collision: point a's second-block hash at session b.
    idx.owner[block_hashes(UInt64(5), a, 512)[1]] = UInt64(2)
    var req = a.copy()
    req.append(1)
    var m = idx.longest_match(req, len(req) - 1, UInt64(0), False)
    assert_equal(m[1], lcp(req, b, len(req) - 1))   # the verified length, not 512


def test_append_chunk_and_drop() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(300, 7)
    idx.add(_rec(UInt64(1), a, 300))
    var more = _toks(256, 8)
    idx.append_chunk(0, ChunkInfo(300, 556, 556, -1, UInt64(0)), more)
    assert_equal(idx.sessions[0].kv_len(), 556)
    assert_equal(idx.sessions[0].drafter_len, 556)
    assert_equal(len(idx.owner), 2)            # 556 tokens = 2 full blocks
    idx.drop(UInt64(1))
    assert_equal(len(idx.sessions), 0)
    assert_equal(len(idx.owner), 0)


def _flip(tokens: List[Int], i: Int) -> List[Int]:
    var out = tokens.copy()
    out[i] = out[i] + 1
    return out^


def _prefix_then_other(base: List[Int], keep: Int, total: Int) -> List[Int]:
    """base[0:keep], then tokens that no _toks() call produces, up to total tokens."""
    var out = sub(base, 0, keep)
    for i in range(total - keep):
        out.append(199999 - i)
    return out^


# The tests below pin what the first eight leave open: equality of every key field; the block-hash chain and its block
# edges; lcp and sub; which session find, drop and append_chunk act on and what they change in it (tokens, chunks, the
# drafter prefix, the owner map); and the cases where longest_match must look past the one session a block hash names.


def test_key_equality_is_per_field() raises:
    # A store is rebuilt when its key stops comparing equal (a drafter or layout change): a field that == ignores would
    # let KV from the old layout be imported.
    var k = CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(4), UInt64(5))
    var same = CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(4), UInt64(5))
    assert_true(k == same)
    assert_false(k != same)
    var others = List[CacheKey]()
    others.append(CacheKey(3, UInt64(7), 1, UInt64(9), UInt64(4), UInt64(5)))    # model_id
    others.append(CacheKey(2, UInt64(8), 1, UInt64(9), UInt64(4), UInt64(5)))    # build_id
    others.append(CacheKey(2, UInt64(7), 2, UInt64(9), UInt64(4), UInt64(5)))    # kv_mode
    others.append(CacheKey(2, UInt64(7), 1, UInt64(10), UInt64(4), UInt64(5)))   # layout_hash
    others.append(CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(0), UInt64(5)))    # variant_hash
    others.append(CacheKey(2, UInt64(7), 1, UInt64(9), UInt64(4), UInt64(0)))    # drafter_id
    for i in range(len(others)):
        assert_true(k != others[i])
        assert_false(k == others[i])
        assert_true(others[i] != k)


def test_block_hash_is_the_chain_of_its_256_token_blocks() raises:
    var t = _toks(768, 3)
    var base = block_hashes(UInt64(5), t, 768)
    assert_equal(len(base), 3)
    # The documented definition, spelled out with the hashing primitive: h[-1] = seed, h[i] = H(h[i-1], block i).
    var h0 = fnv1a64_tokens(UInt64(5), t, 0, 256)
    var h1 = fnv1a64_tokens(h0, t, 256, 512)
    assert_equal(base[0], h0)
    assert_equal(base[1], h1)
    assert_equal(base[2], fnv1a64_tokens(h1, t, 512, 768))
    # One changed token changes its own block's hash and every later one (the chain), never an earlier one. The tokens
    # on either side of each block edge (255/256, 511/512) say which block each belongs to.
    var edges: List[Int] = [0, 255, 256, 511, 512, 767]
    for e in range(len(edges)):
        var h = block_hashes(UInt64(5), _flip(t, edges[e]), 768)
        for b in range(3):
            if b < edges[e] // BLOCK:
                assert_equal(h[b], base[b])
            else:
                assert_true(h[b] != base[b])
    # Only full blocks are hashed.
    assert_equal(len(block_hashes(UInt64(5), t, 0)), 0)
    assert_equal(len(block_hashes(UInt64(5), t, 255)), 0)
    assert_equal(len(block_hashes(UInt64(5), t, 256)), 1)
    assert_equal(len(block_hashes(UInt64(5), t, 767)), 2)


def test_lcp_and_sub() raises:
    var a: List[Int] = [1, 2, 3, 4, 5]
    var parts: List[Int] = [1, 2, 3, 9, 5, 6]
    var longer: List[Int] = [1, 2, 3, 4, 5, 6, 7]
    assert_equal(lcp(a, parts, 100), 3)       # they part at index 3
    assert_equal(lcp(a, parts, 2), 2)         # the limit wins over the match
    assert_equal(lcp(a, parts, 0), 0)
    assert_equal(lcp(a, a, 100), 5)
    assert_equal(lcp(a, longer, 100), 5)      # one is a prefix of the other: the shorter one's length, either way round
    assert_equal(lcp(longer, a, 100), 5)
    var s = sub(longer, 2, 5)
    assert_equal(len(s), 3)
    assert_equal(s[0], 3)
    assert_equal(s[1], 4)
    assert_equal(s[2], 5)
    assert_equal(len(sub(longer, 0, 7)), 7)
    assert_equal(len(sub(longer, 4, 4)), 0)
    assert_equal(len(sub(longer, 5, 2)), 0)   # an inverted range is empty


def test_find_and_drop_pick_the_right_session() raises:
    var idx = SessionIndex(UInt64(5))
    var s1 = _toks(300, 1)
    var s3 = _toks(300, 3)
    idx.add(_rec(UInt64(10), s1, 0))
    idx.add(_rec(UInt64(20), _toks(300, 2), 0))
    idx.add(_rec(UInt64(30), s3, 0))
    assert_equal(idx.find(UInt64(10)), 0)
    assert_equal(idx.find(UInt64(20)), 1)
    assert_equal(idx.find(UInt64(30)), 2)
    assert_equal(idx.find(UInt64(99)), -1)
    assert_equal(len(idx.owner), 3)
    idx.drop(UInt64(99))                      # an unknown id changes nothing
    assert_equal(len(idx.sessions), 3)
    assert_equal(len(idx.owner), 3)
    idx.drop(UInt64(20))                      # the middle one: neither the first nor the last goes
    assert_equal(len(idx.sessions), 2)
    assert_equal(idx.sessions[0].id, UInt64(10))
    assert_equal(idx.sessions[1].id, UInt64(30))
    assert_equal(idx.find(UInt64(30)), 1)
    assert_equal(idx.find(UInt64(20)), -1)
    assert_equal(len(idx.owner), 2)
    assert_equal(idx.owner[block_hashes(UInt64(5), s1, 300)[0]], UInt64(10))
    assert_equal(idx.owner[block_hashes(UInt64(5), s3, 300)[0]], UInt64(30))


def test_drop_hands_shared_blocks_to_the_remaining_holder() raises:
    var idx = SessionIndex(UInt64(5))
    var older = _toks(600, 1)                          # blocks 0 and 1
    idx.add(_rec(UInt64(1), older, 0))
    idx.add(_rec(UInt64(2), _prefix_then_other(older, 512, 612), 0))   # the same two blocks, then its own tail
    var hs = block_hashes(UInt64(5), older, 600)
    assert_equal(len(idx.owner), 2)
    assert_equal(idx.owner[hs[0]], UInt64(2))          # the newer holder owns the shared blocks
    assert_equal(idx.owner[hs[1]], UInt64(2))
    idx.drop(UInt64(2))
    assert_equal(len(idx.owner), 2)                    # handed back to session 1, not lost
    assert_equal(idx.owner[hs[0]], UInt64(1))
    assert_equal(idx.owner[hs[1]], UInt64(1))


def test_append_chunk_keeps_the_drafter_prefix_contiguous() raises:
    var idx = SessionIndex(UInt64(5))
    idx.add(_rec(UInt64(1), _toks(300, 1), 300))       # drafter rows for all 300 tokens
    idx.add(_rec(UInt64(2), _toks(300, 2), 100))       # a gap already: rows only for the first 100
    idx.add(_rec(UInt64(3), _toks(300, 3), 300))
    # Contiguous, and the chunk's own drafter rows stop short of its end: drafter_len follows drafter_end.
    idx.append_chunk(0, ChunkInfo(300, 400, 350, -1, UInt64(0)), _toks(100, 8))
    assert_equal(idx.sessions[0].drafter_len, 350)
    # Not contiguous with the stored rows: the prefix stays where it was.
    idx.append_chunk(1, ChunkInfo(300, 400, 400, -1, UInt64(0)), _toks(100, 8))
    assert_equal(idx.sessions[1].drafter_len, 100)
    # A base-path turn stores no drafter rows (drafter_end == start); the speculative chunk after it does not reconnect.
    idx.append_chunk(2, ChunkInfo(300, 400, 300, -1, UInt64(0)), _toks(100, 8))
    idx.append_chunk(2, ChunkInfo(400, 500, 500, -1, UInt64(0)), _toks(100, 8))
    assert_equal(idx.sessions[2].drafter_len, 300)
    assert_equal(idx.sessions[2].kv_len(), 500)
    # Every chunk is recorded, in order.
    assert_equal(len(idx.sessions[2].chunks), 3)
    assert_equal(idx.sessions[2].chunks[1].start, 300)
    assert_equal(idx.sessions[2].chunks[2].start, 400)
    assert_equal(idx.sessions[2].chunks[2].end, 500)


def test_append_chunk_extends_the_session_it_is_given() raises:
    # Three sessions that differ in length, block count and drafter_len, and the chunk goes to the middle one, so
    # that position 0, the last position and the right one are three different sessions.
    var idx = SessionIndex(UInt64(5))
    var base = _toks(300, 2)                           # the target: 1 full block, drafter rows for all 300 tokens
    var tail = _toks(256, 8)
    idx.add(_rec(UInt64(1), _toks(600, 1), 0))         # position 0: 2 blocks, no drafter rows
    idx.add(_rec(UInt64(2), base, 300))                # position 1: the target
    idx.add(_rec(UInt64(3), _toks(900, 3), 100))       # position 2: 3 blocks, drafter rows for the first 100 only
    assert_equal(len(idx.owner), 6)
    idx.append_chunk(1, ChunkInfo(300, 556, 556, -1, UInt64(0)), tail)
    var full = base.copy()                             # what the session must hold now: base, then tail
    for i in range(len(tail)):
        full.append(tail[i])
    # Its tokens, in order, and nobody else's.
    assert_equal(idx.sessions[1].kv_len(), 556)
    assert_equal(lcp(idx.sessions[1].tokens, full, 556), 556)
    assert_equal(idx.sessions[0].kv_len(), 600)
    assert_equal(idx.sessions[2].kv_len(), 900)
    # Its drafter prefix followed the chunk (the rows were contiguous up to chunk.start); the others' did not move.
    assert_equal(idx.sessions[1].drafter_len, 556)
    assert_equal(idx.sessions[0].drafter_len, 0)
    assert_equal(idx.sessions[2].drafter_len, 100)
    # Its chunk list, and nobody else's.
    assert_equal(len(idx.sessions[1].chunks), 2)
    assert_equal(idx.sessions[1].chunks[1].start, 300)
    assert_equal(idx.sessions[1].chunks[1].end, 556)
    assert_equal(len(idx.sessions[0].chunks), 1)
    assert_equal(len(idx.sessions[2].chunks), 1)
    # The block the chunk completed is owned by this session, under the chain hash of base + tail.
    assert_equal(len(idx.owner), 7)
    assert_equal(idx.owner[block_hashes(UInt64(5), full, 556)[1]], UInt64(2))
    # So a request for what it holds now finds it, whole up to the cap, speculative or not.
    var m = idx.longest_match(full, len(full) - 1, UInt64(0), False)
    assert_equal(m[0], 1)
    assert_equal(m[1], 555)
    m = idx.longest_match(full, len(full) - 1, UInt64(0), True)
    assert_equal(m[0], 1)
    assert_equal(m[1], 555)


def test_the_deepest_matching_block_beats_a_newer_owner_of_a_shallower_one() raises:
    var idx = SessionIndex(UInt64(5))
    var t = _toks(600, 1)
    idx.add(_rec(UInt64(1), t, 0))                                      # blocks 0 and 1
    idx.add(_rec(UInt64(2), _prefix_then_other(t, 256, 600), 0))        # only block 0 in common, and newer: owns it
    var req = sub(t, 0, 550)
    req.append(199999)
    var m = idx.longest_match(req, len(req) - 1, UInt64(0), False)
    assert_equal(m[0], 0)                              # the walk starts at the deepest block, held only by session 1
    assert_equal(m[1], 550)


def test_the_walk_steps_down_past_blocks_nobody_holds() raises:
    # A request that runs past everything stored (the usual next turn) has deeper blocks with no owner; the walk must
    # step down one block at a time to the deepest block somebody holds, not hop to a shallower holder.
    var idx = SessionIndex(UInt64(5))
    var t = _toks(600, 1)
    idx.add(_rec(UInt64(1), t, 0))                                       # X: blocks 0 and 1, and a partial block
    idx.add(_rec(UInt64(2), _prefix_then_other(t, 256, 300), 0))         # Y: block 0 only, and newer: owns it
    var req = t.copy()
    for i in range(200):
        req.append(199999 - i)                                           # 800 tokens: block 2 is held by nobody
    var m = idx.longest_match(req, len(req) - 1, UInt64(0), False)
    assert_equal(m[0], 0)
    assert_equal(m[1], 600)


def test_a_cap_beyond_the_request_hashes_exactly_its_full_blocks() raises:
    # With no effective cap the walk covers every full block of ids and reads nothing past its end.
    var idx = SessionIndex(UInt64(5))
    var t = _toks(600, 1)
    idx.add(_rec(UInt64(1), t, 0))                                       # X: blocks 0 and 1
    idx.add(_rec(UInt64(2), _prefix_then_other(t, 256, 300), 0))         # Y: block 0 only, and newer: owns it
    var m = idx.longest_match(sub(t, 0, 512), 100000, UInt64(0), False)  # exactly two blocks: block 1 leads to X
    assert_equal(m[0], 0)
    assert_equal(m[1], 512)
    # One token short of that there is no block 1 to hash, and nothing may be read past the request.
    m = idx.longest_match(sub(t, 0, 511), 100000, UInt64(0), False)
    assert_true(m[1] >= 256)


def test_longest_match_compares_every_session_when_no_block_hash_leads() raises:
    # A request shorter than one block has no hash to find a candidate by, so every session is compared.
    var idx = SessionIndex(UInt64(5))
    var r = _toks(100, 4)
    idx.add(_rec(UInt64(1), _toks(30, 5), 0))                   # position 0: shares nothing, and is the shortest
    idx.add(_rec(UInt64(2), _prefix_then_other(r, 70, 100), 0))  # position 1: shares 70
    idx.add(_rec(UInt64(3), _prefix_then_other(r, 40, 100), 0))  # position 2: shares 40
    var m = idx.longest_match(r, 99, UInt64(0), False)
    assert_equal(m[0], 1)
    assert_equal(m[1], 70)


def test_longest_match_reports_no_match_as_position_minus_one() raises:
    var idx = SessionIndex(UInt64(5))
    var r = _toks(100, 4)
    assert_equal(idx.longest_match(r, 99, UInt64(0), False)[0], -1)    # an empty index
    idx.add(_rec(UInt64(1), _toks(100, 5), 0))                          # shares nothing with r
    var m = idx.longest_match(r, 99, UInt64(0), False)
    assert_equal(m[0], -1)
    assert_equal(m[1], 0)
    # Speculative: a session with no drafter context matches nothing, however well its tokens match.
    idx.add(_rec(UInt64(2), r, 0))
    m = idx.longest_match(r, 99, UInt64(0), True)
    assert_equal(m[0], -1)
    assert_equal(m[1], 0)
    m = idx.longest_match(r, 99, UInt64(0), False)
    assert_equal(m[0], 1)
    assert_equal(m[1], 99)


def test_longest_match_stops_at_the_session_end_and_at_the_cap() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(300, 1)
    idx.add(_rec(UInt64(1), a, 300))
    var req = a.copy()
    for i in range(50):
        req.append(199999 - i)                         # the next turn: the whole session, then new tokens
    assert_equal(idx.longest_match(req, len(req) - 1, UInt64(0), False)[1], 300)
    assert_equal(idx.longest_match(req, len(req) - 1, UInt64(0), True)[1], 300)
    assert_equal(idx.longest_match(req, 100, UInt64(0), False)[1], 100)    # the cap, inside the first block
    assert_equal(idx.longest_match(req, 255, UInt64(0), False)[1], 255)    # one short of a full block
    assert_equal(idx.longest_match(req, 256, UInt64(0), False)[1], 256)    # exactly one block
    assert_equal(idx.longest_match(req, 299, UInt64(0), False)[1], 299)
    assert_equal(idx.longest_match(req, 1000, UInt64(0), False)[1], 300)   # a cap past the session changes nothing


def test_excluding_the_newest_holder_still_finds_the_other_session() raises:
    # The store excludes the live session. It registered the shared blocks last, so the block hashes lead only to it,
    # and the parked session with the same prefix must still be found.
    var idx = SessionIndex(UInt64(5))
    var a = _toks(600, 1)
    idx.add(_rec(UInt64(1), a, 0))                                     # parked, position 0
    idx.add(_rec(UInt64(2), _prefix_then_other(a, 520, 700), 0))        # live: owns the shared blocks now
    var m = idx.longest_match(a, len(a) - 1, UInt64(2), False)
    assert_equal(m[0], 0)
    assert_equal(m[1], 599)
    var other = idx.longest_match(a, len(a) - 1, UInt64(1), False)     # and the other way round
    assert_equal(other[0], 1)
    assert_equal(other[1], 520)


def test_an_excluded_holder_of_deeper_blocks_does_not_hide_the_best_session() raises:
    # A holds blocks 0 and 1. The excluded live session L holds the same two blocks and was added later, so the hashes
    # name L for both. B, added last, holds only block 0 and is named for it: settling for B would return 256, while A
    # matches 599.
    var idx = SessionIndex(UInt64(5))
    var a = _toks(600, 1)
    idx.add(_rec(UInt64(1), a, 0))                                      # A, position 0
    idx.add(_rec(UInt64(2), _prefix_then_other(a, 520, 700), 0))        # L: blocks 0 and 1 shared with A
    idx.add(_rec(UInt64(3), _prefix_then_other(a, 256, 300), 0))        # B: block 0 only, added last
    var m = idx.longest_match(a, len(a) - 1, UInt64(2), False)
    assert_equal(m[0], 0)
    assert_equal(m[1], 599)
    # The excluded session is not returned even when it is the best match: a request for L's own tokens gets A's 520.
    var live = idx.sessions[1].tokens.copy()
    m = idx.longest_match(live, len(live) - 1, UInt64(2), False)
    assert_equal(m[0], 0)
    assert_equal(m[1], 520)


def test_spec_finds_the_holder_whose_drafter_context_reaches_furthest() raises:
    # Two sessions hold the same tokens and the block hashes name only the one added last. A speculative request must
    # still find the one with drafter context, whichever order they were added in (after a restart it is the
    # directory's order).
    var t = _toks(600, 2)
    for order in range(2):
        var idx = SessionIndex(UInt64(5))
        if order == 0:
            idx.add(_rec(UInt64(10), t, 0))                # A: tokens only
            idx.add(_rec(UInt64(11), t, 600))              # F: tokens and drafter context
        else:
            idx.add(_rec(UInt64(11), t, 600))
            idx.add(_rec(UInt64(10), t, 0))
        var m = idx.longest_match(t, len(t) - 1, UInt64(0), True)
        assert_equal(m[1], 599)
        assert_equal(m[0], idx.find(UInt64(11)))
        assert_equal(idx.longest_match(t, len(t) - 1, UInt64(0), False)[1], 599)   # without spec either will do


def test_an_owner_entry_for_an_unknown_session_is_ignored() raises:
    var idx = SessionIndex(UInt64(5))
    var a = _toks(600, 1)
    idx.add(_rec(UInt64(1), a, 0))
    var req = a.copy()
    req.append(1)
    idx.owner[block_hashes(UInt64(5), req, 600)[1]] = UInt64(77)       # stale: names no session in the index
    var m = idx.longest_match(req, len(req) - 1, UInt64(0), False)
    assert_equal(m[0], 0)                              # the next block down still leads to session 1
    assert_equal(m[1], 600)


def _rmtree(path: String):
    """Remove a directory tree with std.os only. Best effort: a directory that cannot be listed or emptied stays."""
    try:
        var names = listdir(path)
        for i in range(len(names)):
            var child = path + "/" + names[i]
            if isdir(child):
                _rmtree(child)
            else:
                remove(child)
        rmdir(path)
    except:
        pass


def _put(dir: String, name: String, text: String, size: Int) raises:
    """dir/name: the bytes of `text`, then zeros up to `size` bytes (never fewer than the text)."""
    var tb = text.as_bytes()
    var body = List[UInt8](length=max(size, len(tb)), fill=0)
    for i in range(len(tb)):
        body[i] = tb[i]
    var fd = open_file(dir + "/" + name, O_RDWR | O_CREAT, 420)   # 0o644
    pwrite_all(fd, UInt64(Int(body.unsafe_ptr())), len(body), 0)
    _ = body^   # an untracked address is not a use: keep the bytes alive until the write has read them
    close_file(fd)


def _weights_dir(mut made: List[String], manifest: String, w0: Int, w1: Int) raises -> String:
    """A scratch weights directory under build/ (recorded in `made` for the cleanup): convert_manifest.json with this
    text, and two weight files w0.bin and w1.bin of these sizes."""
    var dir = "build/kvstore_index_test/" + hex16(random_u64())
    makedirs(dir, exist_ok=True)
    made.append(dir)
    _put(dir, "convert_manifest.json", manifest, 0)
    _put(dir, "w0.bin", "", w0)
    _put(dir, "w1.bin", "", w1)
    return dir


def _cleanup(made: List[String]):
    for i in range(len(made)):
        _rmtree(made[i])


def test_weights_id_tracks_the_manifest_and_every_files_name_and_size() raises:
    """Ledger R35: weights that differ in their manifest, in a file's size or name, or in the set of files, never share
    an id; the same contents in another directory do."""
    var made = List[String]()
    try:
        var base = _weights_dir(made, '{"source": "base"}', 100, 200)
        var same = _weights_dir(made, '{"source": "base"}', 100, 200)          # identical contents, another path
        assert_equal(weights_id_of(base), weights_id_of(same))
        var ids = List[UInt64]()
        ids.append(weights_id_of(base))
        ids.append(weights_id_of(_weights_dir(made, '{"source": "finetuned"}', 100, 200)))   # only the manifest differs
        ids.append(weights_id_of(_weights_dir(made, '{"source": "base"}', 100, 201)))        # one file one byte bigger
        ids.append(weights_id_of(_weights_dir(made, '{"source": "base"}', 101, 200)))        # the other file instead
        ids.append(weights_id_of(_weights_dir(made, '{"source": "base"}', 200, 100)))        # the sizes swapped between files
        var renamed = _weights_dir(made, '{"source": "base"}', 100, 200)
        remove(renamed + "/w1.bin")
        _put(renamed, "w9.bin", "", 200)                                                       # same size, another name
        ids.append(weights_id_of(renamed))
        var extra = _weights_dir(made, '{"source": "base"}', 100, 200)
        _put(extra, "w2.bin", "", 0)                                                           # one more (empty) file
        ids.append(weights_id_of(extra))
        var bare = _weights_dir(made, '{"source": "base"}', 100, 200)
        remove(bare + "/convert_manifest.json")                                                # no manifest at all
        ids.append(weights_id_of(bare))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                assert_true(ids[i] != ids[j])
    finally:
        _cleanup(made)


def test_weights_id_known_answer() raises:
    """The id of a fixed directory, computed independently by tools/kvsession_testlib.py (weights_id_of): the recipe
    is part of the store's on-disk naming, so it must not drift."""
    var made = List[String]()
    try:
        var dir = "build/kvstore_index_test/" + hex16(random_u64())
        makedirs(dir, exist_ok=True)
        made.append(dir)
        _put(dir, "convert_manifest.json", '{"layers": 52}', 0)
        _put(dir, "a.bin", "", 10)
        _put(dir, "b.bin", "", 20)
        assert_equal(weights_id_of(dir), UInt64(0xC42549D10CDF0DEE))
    finally:
        _cleanup(made)


def test_weights_id_of_a_missing_directory_raises() raises:
    with assert_raises():
        _ = weights_id_of("build/kvstore_index_test/no_such_directory")


def test_numerics_id_tracks_the_weights_and_every_precision_flag() raises:
    """Ledger R35: NOMOS_PRECISION_BITS, NOMOS_WEIGHT_NVFP4, NOMOS_W4A4, NOMOS_GDN_NVFP4 and the weights each change the
    id; the same inputs give the same id."""
    var ids = List[UInt64]()
    ids.append(numerics_id_of(UInt64(5), 16, True, True, False))
    ids.append(numerics_id_of(UInt64(6), 16, True, True, False))     # other weights
    ids.append(numerics_id_of(UInt64(5), 32, True, True, False))     # NOMOS_PRECISION_BITS
    ids.append(numerics_id_of(UInt64(5), 16, False, True, False))    # NOMOS_WEIGHT_NVFP4
    ids.append(numerics_id_of(UInt64(5), 16, True, False, False))    # NOMOS_W4A4: the W4A16 path
    ids.append(numerics_id_of(UInt64(5), 16, True, True, True))      # NOMOS_GDN_NVFP4
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert_true(ids[i] != ids[j])
    assert_equal(ids[0], numerics_id_of(UInt64(5), 16, True, True, False))


def main() raises:
    test_key_seed_tracks_every_field()
    test_block_hashes_are_prefix_stable()
    test_longest_match_full_blocks_then_tokens()
    test_short_prompt_matches_without_a_full_block()
    test_excluded_session_is_skipped()
    test_spec_caps_at_drafter_len()
    test_hash_hits_are_verified_by_tokens()
    test_append_chunk_and_drop()
    test_key_equality_is_per_field()
    test_block_hash_is_the_chain_of_its_256_token_blocks()
    test_lcp_and_sub()
    test_find_and_drop_pick_the_right_session()
    test_drop_hands_shared_blocks_to_the_remaining_holder()
    test_append_chunk_keeps_the_drafter_prefix_contiguous()
    test_append_chunk_extends_the_session_it_is_given()
    test_the_deepest_matching_block_beats_a_newer_owner_of_a_shallower_one()
    test_the_walk_steps_down_past_blocks_nobody_holds()
    test_a_cap_beyond_the_request_hashes_exactly_its_full_blocks()
    test_longest_match_compares_every_session_when_no_block_hash_leads()
    test_longest_match_reports_no_match_as_position_minus_one()
    test_longest_match_stops_at_the_session_end_and_at_the_cap()
    test_excluding_the_newest_holder_still_finds_the_other_session()
    test_an_excluded_holder_of_deeper_blocks_does_not_hide_the_best_session()
    test_spec_finds_the_holder_whose_drafter_context_reaches_furthest()
    test_an_owner_entry_for_an_unknown_session_is_ignored()
    test_weights_id_tracks_the_manifest_and_every_files_name_and_size()
    test_weights_id_known_answer()
    test_weights_id_of_a_missing_directory_raises()
    test_numerics_id_tracks_the_weights_and_every_precision_flag()
    print("kvstore_index_test: all passed")
