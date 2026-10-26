"""Byte layout of a session chunk (docs/KV_SESSION_STORE_DESIGN.md §3.1). A chunk of ntok tokens is, component by
component in order, ntok * bytes_per_token bytes, each laid out [head][token][row]."""
from lib.kvstore.hashing import FNV_OFFSET, fnv1a64_u64
from lib.kvstore.support import KV_MODE_INT8, KV_MODE_INT4_B32

comptime KIND_K_DATA = 0
comptime KIND_V_DATA = 1
comptime KIND_K_SCALE = 2
comptime KIND_V_SCALE = 3
comptime KIND_K_TAIL = 4
comptime KIND_V_TAIL = 5
comptime KIND_DRAFT_K = 6
comptime KIND_DRAFT_V = 7


@fieldwise_init
struct ComponentDesc(Copyable, Movable, ImplicitlyCopyable):
    var kind: Int
    var layer: Int
    var heads: Int        # head rows; 1 for position-major components (the drafter's context)
    var row_bytes: Int    # bytes per token per head

    def bytes_per_token(self) -> Int:
        return self.heads * self.row_bytes

    def is_drafter(self) -> Bool:
        return self.kind == KIND_DRAFT_K or self.kind == KIND_DRAFT_V


struct KvLayout(Copyable, Movable):
    var comps: List[ComponentDesc]

    def __init__(out self):
        self.comps = List[ComponentDesc]()

    def bytes_per_token(self) -> Int:
        var t = 0
        for i in range(len(self.comps)):
            t += self.comps[i].bytes_per_token()
        return t

    def chunk_bytes(self, ntok: Int) -> Int:
        return ntok * self.bytes_per_token()

    def component_offset(self, idx: Int, ntok: Int) -> Int:
        var off = 0
        for i in range(idx):
            off += ntok * self.comps[i].bytes_per_token()
        return off

    def has_drafter(self) -> Bool:
        for i in range(len(self.comps)):
            if self.comps[i].is_drafter():
                return True
        return False

    def hash(self) -> UInt64:
        var h = FNV_OFFSET
        for i in range(len(self.comps)):
            var c = self.comps[i]
            h = fnv1a64_u64(h, UInt64(c.kind))
            h = fnv1a64_u64(h, UInt64(c.layer))
            h = fnv1a64_u64(h, UInt64(c.heads))
            h = fnv1a64_u64(h, UInt64(c.row_bytes))
        return h


def target_layout(nkv: List[Int], hd: List[Int], mode: Int) -> KvLayout:
    """The target model's KV components, in lib/kv_cache_quant.mojo's layouts (rows are h * cache_cap + pos):
    INT8 data hd bytes/row + one fp32 scale/row; INT4 block-1 data hd/2 bytes/row + one fp32 scale/row;
    INT4 block-32 data hd/2 bytes/row + ceil(hd/32) fp16 scales/row + an fp32 tail/row."""
    var lay = KvLayout()
    for l in range(len(nkv)):
        var data_row = hd[l] if mode == KV_MODE_INT8 else hd[l] // 2
        lay.comps.append(ComponentDesc(KIND_K_DATA, l, nkv[l], data_row))
        lay.comps.append(ComponentDesc(KIND_V_DATA, l, nkv[l], data_row))
        if mode == KV_MODE_INT4_B32:
            var sb = (hd[l] + 31) // 32
            lay.comps.append(ComponentDesc(KIND_K_SCALE, l, nkv[l], sb * 2))
            lay.comps.append(ComponentDesc(KIND_V_SCALE, l, nkv[l], sb * 2))
            lay.comps.append(ComponentDesc(KIND_K_TAIL, l, nkv[l], 4))
            lay.comps.append(ComponentDesc(KIND_V_TAIL, l, nkv[l], 4))
        else:
            lay.comps.append(ComponentDesc(KIND_K_SCALE, l, nkv[l], 4))
            lay.comps.append(ComponentDesc(KIND_V_SCALE, l, nkv[l], 4))
    return lay^
