"""Cache key of the KV session store (docs/KV_SESSION_STORE_DESIGN.md §3.1): KV is reused only under an equal key."""
from std.memory import UnsafePointer
from std.os import listdir
from lib.kvstore.fileio import O_RDONLY, open_file, close_file, file_size, pread_all
from lib.kvstore.hashing import FNV_OFFSET, FNV_PRIME, fnv1a64_u64, fnv1a64_bytes


@fieldwise_init
struct CacheKey(Copyable, Movable, ImplicitlyCopyable, Equatable):
    var model_id: Int
    var build_id: UInt64        # the kernel build and the numerics its K/V values come from (session_store: BUILD_ID mixed with numerics_id)
    var kv_mode: Int
    var layout_hash: UInt64
    var variant_hash: UInt64    # anything besides the tokens that changes the K/V (0 = none)
    var drafter_id: UInt64      # 0 when no DFlash drafter is loaded

    def seed(self) -> UInt64:
        """h[-1] of every block-hash chain: equal prefixes under different keys never collide."""
        var h = FNV_OFFSET
        h = fnv1a64_u64(h, UInt64(self.model_id))
        h = fnv1a64_u64(h, self.build_id)
        h = fnv1a64_u64(h, UInt64(self.kv_mode))
        h = fnv1a64_u64(h, self.layout_hash)
        h = fnv1a64_u64(h, self.variant_hash)
        h = fnv1a64_u64(h, self.drafter_id)
        return h

    def __eq__(self, other: Self) -> Bool:
        return (self.model_id == other.model_id and self.build_id == other.build_id
                and self.kv_mode == other.kv_mode and self.layout_hash == other.layout_hash
                and self.variant_hash == other.variant_hash and self.drafter_id == other.drafter_id)

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


def weights_id_of(dir: String) raises -> UInt64:
    """The identity of a weights directory (docs/KV_SESSION_STORE_DESIGN.md §3.6-3.7, ledger R35): a hash of its
    convert_manifest.json plus every file's name and size, order-independent over the files. KV computed from other
    weights must never be reused. This is not a content hash (that would read tens of GB at every start): weights that
    differ only in their values, with the same manifest and file sizes, are not told apart."""
    var h = FNV_OFFSET
    var fd = open_file(dir + "/convert_manifest.json", O_RDONLY, 0)
    if fd >= 0:
        var n = file_size(fd)
        var buf = List[UInt8](length=max(n, 1), fill=0)
        if pread_all(fd, UInt64(Int(buf.unsafe_ptr())), n, 0):
            h = fnv1a64_bytes(h, UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(buf.unsafe_ptr())), n)
        _ = buf^   # an untracked address is not a use: keep the bytes alive until the hash has read them
        close_file(fd)
    var names = listdir(dir)
    var files: UInt64 = 0
    for i in range(len(names)):
        var f = open_file(dir + "/" + names[i], O_RDONLY, 0)
        if f < 0:
            continue
        var name_bytes = names[i].as_bytes()
        var e = FNV_OFFSET
        for j in range(len(name_bytes)):
            e = (e ^ UInt64(name_bytes[j])) * FNV_PRIME
        files = files + fnv1a64_u64(e, UInt64(file_size(f)))   # wrapping sum: order does not matter
        close_file(f)
    return fnv1a64_u64(h, files)


def numerics_id_of(weights_id: UInt64, precision_bits: Int, weight_nvfp4: Bool, w4a4: Bool, gdn_nvfp4: Bool) -> UInt64:
    """What the target's K/V values depend on besides the kernel's code (ledger R35): its weights (weights_id_of) and
    the runtime precision flags NOMOS_PRECISION_BITS, NOMOS_WEIGHT_NVFP4, NOMOS_W4A4 and NOMOS_GDN_NVFP4, as the engine
    resolved them. A restart under any other value computes other K/V, so it must not restore these sessions."""
    var h = fnv1a64_u64(FNV_OFFSET, weights_id)
    h = fnv1a64_u64(h, UInt64(precision_bits))
    h = fnv1a64_u64(h, UInt64(1) if weight_nvfp4 else UInt64(0))
    h = fnv1a64_u64(h, UInt64(1) if w4a4 else UInt64(0))
    h = fnv1a64_u64(h, UInt64(1) if gdn_nvfp4 else UInt64(0))
    return h
