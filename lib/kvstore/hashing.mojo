"""Stable hashes for the KV session store. Block hashes and checksums are saved on disk, so they must never change
between builds or Mojo versions: FNV-1a 64 and CRC-32C (Castagnoli), written out here, never std hash()."""
from std.memory import UnsafePointer
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic

comptime FNV_OFFSET: UInt64 = 0xCBF29CE484222325
comptime FNV_PRIME: UInt64 = 0x100000001B3
comptime _CRC32C_POLY: UInt32 = 0x82F63B78  # reflected Castagnoli polynomial

# The x86 `crc32` instruction (SSE4.2) is CRC-32C itself: the same reflected Castagnoli polynomial, the same bit order,
# so it returns exactly the table's values. It is chosen at compile time from the build's own target features (the
# kernel is built for the host CPU), so a target without it never sees the intrinsic and runs the table. Both features
# are asked for because the instruction selector wants `crc32`, which every x86 CPU model with `sse4.2` carries. The
# stdlib has no public SSE4.2 query (has_sse4() is SSE4.1), hence the underscore method. aarch64 has its own CRC
# extension and keeps the table until that is wired in.
comptime _CRC32C_HW: Bool = (
    CompilationTarget.is_x86()
    and CompilationTarget._has_feature["sse4.2"]()
    and CompilationTarget._has_feature["crc32"]()
)
comptime _CRC32C_STEP_64: StaticString = "llvm.x86.sse42.crc32.64.64"   # (crc: i64, 8 bytes: i64) -> i64
comptime _CRC32C_STEP_8: StaticString = "llvm.x86.sse42.crc32.32.8"     # (crc: i32, 1 byte: i8) -> i32


def crc32c_hw() -> Bool:
    """True when Crc32c.update runs on the CPU's CRC-32C instruction in this build, False when it runs the table."""
    return _CRC32C_HW


def fnv1a64_bytes(h: UInt64, p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> UInt64:
    var x = h
    for i in range(n):
        x = (x ^ UInt64(p[i])) * FNV_PRIME
    return x


def fnv1a64_u64(h: UInt64, v: UInt64) -> UInt64:
    """Absorb v as 8 little-endian bytes."""
    var x = h
    for i in range(8):
        x = (x ^ ((v >> UInt64(8 * i)) & 0xFF)) * FNV_PRIME
    return x


def fnv1a64_tokens(h: UInt64, tokens: List[Int], a: Int, b: Int) -> UInt64:
    """Absorb tokens[a:b], each as 4 little-endian bytes (token IDs are below 2^31)."""
    var x = h
    for i in range(a, b):
        var t = UInt64(tokens[i])
        for k in range(4):
            x = (x ^ ((t >> UInt64(8 * k)) & 0xFF)) * FNV_PRIME
    return x


struct Crc32c(Movable):
    """CRC-32C. update() uses the CPU's CRC-32C instruction where the build has it (x86-64 with SSE4.2, see
    crc32c_hw()) and update_table() elsewhere: slicing-by-8, eight 256-entry tables, eight input bytes per step.
    The two return the same value for every input, so a checksum computed by one verifies against the other."""

    var table: List[UInt32]

    def __init__(out self):
        self.table = List[UInt32](capacity=8 * 256)
        for i in range(256):
            var c = UInt32(i)
            for _ in range(8):
                c = (c >> 1) ^ (_CRC32C_POLY if (c & 1) != 0 else UInt32(0))
            self.table.append(c)
        for k in range(1, 8):
            for i in range(256):
                var prev = self.table[(k - 1) * 256 + i]
                self.table.append((prev >> 8) ^ self.table[Int(prev & 0xFF)])

    def update(self, crc: UInt32, p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> UInt32:
        """Continue a CRC-32C over n bytes. Start from 0; update(update(0, a), b) == update(0, a ++ b).
        Any alignment of p; any n (n <= 0 returns crc). One dependent chain of 8-byte steps, then single bytes."""
        comptime if _CRC32C_HW:
            var c = UInt64(~crc)                  # as the table loop: run on ~crc, return ~c
            var i = 0
            # Four 8-byte steps per trip. It is still ONE dependent chain (each step feeds the next), so the value is
            # the same as with one step per trip; the extra steps only share the loop's own add/compare/branch. Measured
            # on this laptop (56 MB buffer): P-cores about 7 GB/s with one step per trip, 7.7-7.8 with four (9.1 when the
            # data is cache-resident); E-cores, bound by the instruction's 3-cycle latency, 12.1 GB/s either way.
            while i + 32 <= n:
                var q = p.unsafe_offset(i).unsafe_bitcast[UInt64]()    # little-endian words at any alignment
                c = llvm_intrinsic[_CRC32C_STEP_64, UInt64, has_side_effect=False](c, q.unsafe_load[alignment=1](0))
                c = llvm_intrinsic[_CRC32C_STEP_64, UInt64, has_side_effect=False](c, q.unsafe_load[alignment=1](1))
                c = llvm_intrinsic[_CRC32C_STEP_64, UInt64, has_side_effect=False](c, q.unsafe_load[alignment=1](2))
                c = llvm_intrinsic[_CRC32C_STEP_64, UInt64, has_side_effect=False](c, q.unsafe_load[alignment=1](3))
                i += 32
            while i + 8 <= n:
                var w = p.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load[alignment=1](0)
                c = llvm_intrinsic[_CRC32C_STEP_64, UInt64, has_side_effect=False](c, w)
                i += 8
            var c32 = UInt32(c)
            while i < n:
                c32 = llvm_intrinsic[_CRC32C_STEP_8, UInt32, has_side_effect=False](c32, p[unsafe_offset=i])
                i += 1
            return ~c32
        else:
            return self.update_table(crc, p, n)

    def update_table(self, crc: UInt32, p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> UInt32:
        """update() by table, slicing-by-8: the fallback where the instruction is missing, and the oracle in the tests."""
        var c = ~crc
        var i = 0
        while i + 8 <= n:
            var lo = c ^ (UInt32(p[i]) | (UInt32(p[i + 1]) << 8) | (UInt32(p[i + 2]) << 16) | (UInt32(p[i + 3]) << 24))
            var hi = UInt32(p[i + 4]) | (UInt32(p[i + 5]) << 8) | (UInt32(p[i + 6]) << 16) | (UInt32(p[i + 7]) << 24)
            c = (
                self.table[7 * 256 + Int(lo & 0xFF)] ^ self.table[6 * 256 + Int((lo >> 8) & 0xFF)]
                ^ self.table[5 * 256 + Int((lo >> 16) & 0xFF)] ^ self.table[4 * 256 + Int(lo >> 24)]
                ^ self.table[3 * 256 + Int(hi & 0xFF)] ^ self.table[2 * 256 + Int((hi >> 8) & 0xFF)]
                ^ self.table[256 + Int((hi >> 16) & 0xFF)] ^ self.table[Int(hi >> 24)]
            )
            i += 8
        while i < n:
            c = self.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
            i += 1
        return ~c


def hex16(v: UInt64) -> String:
    """v as exactly 16 lowercase hex digits."""
    comptime DIGITS: StaticString = "0123456789abcdef"
    var s = String()
    for i in range(15, -1, -1):
        s += String(DIGITS[byte=Int((v >> UInt64(4 * i)) & 0xF)])
    return s


def parse_hex16(s: String) -> Optional[UInt64]:
    """Inverse of hex16; None for anything hex16 could not have written."""
    if s.byte_length() != 16:
        return None
    var v: UInt64 = 0
    var b = s.as_bytes()
    for i in range(16):
        var c = Int(b[i])
        var d: Int
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        else:
            return None
        v = (v << 4) | UInt64(d)
    return v
