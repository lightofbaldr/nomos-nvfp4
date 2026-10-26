"""Unit tests for the KV session store hashes (FNV-1a 64, CRC-32C, hex).

Run: bash tools/run_mojo_test.sh tools/kvstore_hash_test.mojo
"""
from std.memory import UnsafePointer
from std.sys.info import CompilationTarget
from std.testing import assert_equal, assert_true, assert_false
from lib.kvstore.hashing import (
    FNV_OFFSET, fnv1a64_bytes, fnv1a64_u64, fnv1a64_tokens, Crc32c, crc32c_hw, hex16, parse_hex16,
)


def _p(b: List[UInt8]) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    return UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(b.unsafe_ptr()))


def _at(b: List[UInt8], off: Int) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    """The address off bytes into b: the CRC is read from every start alignment, not only from the allocator's."""
    return UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(b.unsafe_ptr()) + off)


def _lcg_bytes(n: Int, seed: UInt64) -> List[UInt8]:
    """n pseudo-random bytes from a fixed 64-bit LCG (Knuth's MMIX constants, top byte of the state): the same on every run."""
    var out = List[UInt8](capacity=n)
    var s = seed
    for _ in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        out.append(UInt8(s >> 56))
    return out^


def _ascii(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(s.byte_length()):
        out.append(s.as_bytes()[i])
    return out^


def test_fnv_known_vectors() raises:
    # Each List stays alive (`_ = x^`) until the call reading through its untracked pointer has returned:
    # Mojo destroys a value right after its last use, and an untracked address does not count as one.
    var a = _ascii("a")
    assert_equal(fnv1a64_bytes(FNV_OFFSET, _p(a), 0), UInt64(0xCBF29CE484222325))   # the empty input
    assert_equal(fnv1a64_bytes(FNV_OFFSET, _p(a), 1), UInt64(0xAF63DC4C8601EC8C))
    _ = a^
    var foobar = _ascii("foobar")
    assert_equal(fnv1a64_bytes(FNV_OFFSET, _p(foobar), 6), UInt64(0x85944171F73967E8))
    _ = foobar^


def test_fnv_tokens_and_u64_are_little_endian_bytes() raises:
    var toks: List[Int] = [1, 2, 300]
    var raw: List[UInt8] = [1, 0, 0, 0, 2, 0, 0, 0, 44, 1, 0, 0]
    assert_equal(fnv1a64_tokens(FNV_OFFSET, toks, 0, 3), fnv1a64_bytes(FNV_OFFSET, _p(raw), 12))
    _ = raw^
    var raw8: List[UInt8] = [8, 7, 6, 5, 4, 3, 2, 1]
    assert_equal(fnv1a64_u64(FNV_OFFSET, UInt64(0x0102030405060708)), fnv1a64_bytes(FNV_OFFSET, _p(raw8), 8))
    _ = raw8^


def test_crc32c_known_vector_and_chaining() raises:
    var crc = Crc32c()
    var check = _ascii("123456789")
    assert_equal(crc.update(0, _p(check), 9), UInt32(0xE3069283))
    _ = check^
    var data = List[UInt8]()
    for i in range(1000):
        data.append(UInt8((i * 7 + 3) % 256))
    var whole = crc.update(0, _p(data), 1000)
    for cut in [1, 7, 8, 9, 500, 999]:
        var a = crc.update(0, _p(data), cut)
        var tail = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(data.unsafe_ptr()) + cut)
        assert_equal(crc.update(a, tail, 1000 - cut), whole)
    _ = data^


def test_hex_round_trip() raises:
    assert_equal(hex16(UInt64(0x0123456789ABCDEF)), "0123456789abcdef")
    assert_equal(parse_hex16("0123456789abcdef").value(), UInt64(0x0123456789ABCDEF))
    assert_false(parse_hex16("xyz").__bool__())
    assert_false(parse_hex16("0123456789ABCDEF").__bool__())   # uppercase is not what hex16 writes


def test_fnv_pinned_encodings() raises:
    # Expected values come from an independent reference (textbook FNV-1a 64 over struct.pack('<I' / '<Q'), anchored to
    # the published FNV vectors), not from this code. They pin the on-disk encodings the small cases above leave free:
    # token IDs above 2^16 (vocabularies reach 262,143) and at the 2^31-1 bound, bytes >= 0x80 on every FNV path, and
    # sub-ranges with a != 0 and b < len (the block hashes use (i * 256, (i + 1) * 256)).
    var toks: List[Int] = [262143, 65536, 2147483647, 255, 128]
    assert_equal(fnv1a64_tokens(FNV_OFFSET, toks, 0, 5), UInt64(0x86F9309AB9B0444A))
    assert_equal(fnv1a64_tokens(FNV_OFFSET, toks, 1, 4), UInt64(0xD8369A137C96F787))   # a != 0 and b < len
    assert_equal(fnv1a64_tokens(FNV_OFFSET, toks, 3, 5), UInt64(0x8D6CB5C1FE27EB9A))   # a != 0 and b == len
    assert_equal(fnv1a64_u64(FNV_OFFSET, UInt64(0xFFFFFFFFFFFFFFFF)), UInt64(0x8CF51A8BFCA3883D))
    assert_equal(fnv1a64_u64(FNV_OFFSET, UInt64(0x8000000000000001)), UInt64(0x89CCB1291D2A1624))
    var high: List[UInt8] = [0x80, 0xFF, 0xAB]
    assert_equal(fnv1a64_bytes(FNV_OFFSET, _p(high), 3), UInt64(0x2EE6E31AE0C93571))
    _ = high^


def test_crc32c_rfc3720_vectors() raises:
    # RFC 3720 appendix B.4: the published CRC-32C of four 32-byte patterns, each four 8-byte steps (by instruction or by table).
    var crc = Crc32c()
    var zeros = List[UInt8]()
    var ones = List[UInt8]()
    var ascending = List[UInt8]()
    var descending = List[UInt8]()
    for i in range(32):
        zeros.append(0)
        ones.append(0xFF)
        ascending.append(UInt8(i))
        descending.append(UInt8(31 - i))
    assert_equal(crc.update(0, _p(zeros), 32), UInt32(0x8A9136AA))
    assert_equal(crc.update(0, _p(ones), 32), UInt32(0x62A8AB43))
    assert_equal(crc.update(0, _p(ascending), 32), UInt32(0x46DD794E))
    assert_equal(crc.update(0, _p(descending), 32), UInt32(0x113FDB5C))
    # An empty update returns the running CRC unchanged, so a chain may end on a zero-length piece.
    assert_equal(crc.update(UInt32(0x46DD794E), _p(ascending), 0), UInt32(0x46DD794E))
    _ = zeros^
    _ = ones^
    _ = ascending^
    _ = descending^


def test_hex_boundaries() raises:
    assert_equal(hex16(UInt64(0xFEDCBA9876543210)), "fedcba9876543210")
    assert_equal(parse_hex16("fedcba9876543210").value(), UInt64(0xFEDCBA9876543210))
    # The ASCII neighbours of the accepted ranges, each as the last of 16 characters: none is a lowercase hex digit.
    assert_false(parse_hex16("0123456789abcdeg").__bool__())   # one past 'f'
    assert_false(parse_hex16("0123456789abcde:").__bool__())   # one past '9'
    assert_false(parse_hex16("0123456789abcde`").__bool__())   # one before 'a'
    assert_false(parse_hex16("0123456789abcde/").__bool__())   # one before '0'


def _disagree(what: String, off: Int, c0: UInt32, n: Int, got: UInt32, want: UInt32) -> Error:
    return Error(what + ": start offset " + String(off) + ", crc0 " + hex(c0) + ", length " + String(n)
                 + ": got " + hex(got) + ", want " + hex(want))


def test_crc32c_hw_matches_table() raises:
    # update() runs the CPU's CRC32 instruction where it is compiled in; update_table() is the slicing-by-8 loop that
    # every file written so far was checksummed with. The values are on disk, so the two must agree on every input.
    var crc = Crc32c()
    var crc0s: List[UInt32] = [UInt32(0), UInt32(0xDEADBEEF)]

    # Every length 0..2049 (every 8-byte-step count and every tail length, several times over) from every start
    # alignment 0..7, with both starting values.
    var data = _lcg_bytes(2049 + 8, 0x0123456789ABCDEF)
    for off in range(8):
        for k in range(2):
            for n in range(2050):
                var got = crc.update(crc0s[k], _at(data, off), n)
                var want = crc.update_table(crc0s[k], _at(data, off), n)
                if got != want:
                    raise _disagree("update != update_table", off, crc0s[k], n, got, want)
    _ = data^

    # One 1 MiB buffer: the real calls are chunks of tens of MB, so the loop must also hold up over a long run.
    var big = _lcg_bytes((1 << 20) + 8, 0xFEEDFACECAFEBEEF)
    for off in range(8):
        for k in range(2):
            var got = crc.update(crc0s[k], _at(big, off), 1 << 20)
            var want = crc.update_table(crc0s[k], _at(big, off), 1 << 20)
            if got != want:
                raise _disagree("update != update_table (1 MiB)", off, crc0s[k], 1 << 20, got, want)
    _ = big^

    # Chaining: update(update(c, a), b) == update(c, a ++ b) for a split at every offset 0..16, from every start
    # alignment. Also with the paths mixed at the split: the CRC after any prefix is the same value from both, so
    # finishing a chain with the other path must give the same result (the startup scan chains update() over 16 MiB
    # slices of a record that an earlier, table-only build wrote).
    var chain = _lcg_bytes(2049 + 8, 0x9E3779B97F4A7C15)
    var lens = List[Int]()
    for n in range(41):
        lens.append(n)
    lens.append(255)
    lens.append(256)
    lens.append(257)
    lens.append(2049)
    for li in range(len(lens)):
        var total = lens[li]
        for off in range(8):
            for k in range(2):
                var whole = crc.update_table(crc0s[k], _at(chain, off), total)
                for s in range(17):
                    if s > total:
                        break
                    var rest = _at(chain, off + s)
                    var hw_hw = crc.update(crc.update(crc0s[k], _at(chain, off), s), rest, total - s)
                    var hw_tb = crc.update_table(crc.update(crc0s[k], _at(chain, off), s), rest, total - s)
                    var tb_hw = crc.update(crc.update_table(crc0s[k], _at(chain, off), s), rest, total - s)
                    var tb_tb = crc.update_table(crc.update_table(crc0s[k], _at(chain, off), s), rest, total - s)
                    if hw_hw != whole:
                        raise _disagree("chained update(update(c, a), b) != whole, split " + String(s), off, crc0s[k], total, hw_hw, whole)
                    if hw_tb != whole:
                        raise _disagree("update then update_table != whole, split " + String(s), off, crc0s[k], total, hw_tb, whole)
                    if tb_hw != whole:
                        raise _disagree("update_table then update != whole, split " + String(s), off, crc0s[k], total, tb_hw, whole)
                    if tb_tb != whole:
                        raise _disagree("chained update_table != whole, split " + String(s), off, crc0s[k], total, tb_tb, whole)
    _ = chain^


def test_crc32c_hw_compiled_in_on_this_host() raises:
    # A silent fallback to the table passes every value test above, so say it out loud: an x86-64 build with SSE4.2
    # (this laptop) must have the hardware path compiled in. Other targets keep the table (aarch64: a follow-up).
    comptime if CompilationTarget.is_x86():
        assert_true(crc32c_hw(), "x86-64 build fell back to the CRC-32C table: the hardware path is not compiled in")


def _crc32c_bitwise(crc: UInt32, p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> UInt32:
    """CRC-32C straight from the definition, one bit at a time (reflected, polynomial 0x82F63B78): no table, no instruction."""
    var c = ~crc
    for i in range(n):
        c ^= UInt32(p[unsafe_offset=i])
        for _ in range(8):
            c = (c >> 1) ^ (UInt32(0x82F63B78) if (c & 1) != 0 else UInt32(0))
    return ~c


def test_crc32c_matches_bitwise_definition() raises:
    # The table is the oracle above; this pins the table and the instruction both to the definition, so they cannot
    # share a mistake (the published vectors cover 9 and 32 bytes only).
    var crc = Crc32c()
    var crc0s: List[UInt32] = [UInt32(0), UInt32(0xDEADBEEF)]
    var data = _lcg_bytes(130 + 8, 0x0F1E2D3C4B5A6978)
    for off in range(8):
        for k in range(2):
            for n in range(131):
                var want = _crc32c_bitwise(crc0s[k], _at(data, off), n)
                var got = crc.update(crc0s[k], _at(data, off), n)
                var tbl = crc.update_table(crc0s[k], _at(data, off), n)
                if got != want:
                    raise _disagree("update != bitwise definition", off, crc0s[k], n, got, want)
                if tbl != want:
                    raise _disagree("update_table != bitwise definition", off, crc0s[k], n, tbl, want)
    _ = data^
    var big = _lcg_bytes((1 << 20) + 8, 0xA5A5A5A55A5A5A5A)
    var want_big = _crc32c_bitwise(UInt32(0xDEADBEEF), _at(big, 3), 1 << 20)
    assert_equal(crc.update(UInt32(0xDEADBEEF), _at(big, 3), 1 << 20), want_big)
    assert_equal(crc.update_table(UInt32(0xDEADBEEF), _at(big, 3), 1 << 20), want_big)
    _ = big^


def main() raises:
    test_fnv_known_vectors()
    test_fnv_tokens_and_u64_are_little_endian_bytes()
    test_crc32c_known_vector_and_chaining()
    test_hex_round_trip()
    test_fnv_pinned_encodings()
    test_crc32c_rfc3720_vectors()
    test_hex_boundaries()
    test_crc32c_hw_matches_table()
    test_crc32c_hw_compiled_in_on_this_host()
    test_crc32c_matches_bitwise_definition()
    print("kvstore_hash_test: all passed")
