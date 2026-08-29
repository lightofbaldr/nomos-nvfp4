"""EAGLE-3 draft-token selection: argmax over the 32K draft vocab → d2t → 262K target id.

The drafter's lm_head is [draft_vocab=32000, 5376] (NOT tied — tie_word_embeddings false).
A draft step picks argmax over the 32000 draft logits, then maps that draft id into the 31B's
262144-token target vocab. The d2t LUT (int64 [32000]) stores OFFSETS, so the map is ADDITIVE:
  target_id = draft_id + d2t[draft_id]   (NOT d2t[draft_id]) — verified vs speculators core.py:250
  (`input_ids = input_ids + self.d2t[input_ids]`); a direct LUT is the documented silent-failure.

Returns BOTH (draft_id, target_id) host Ints: target_id is the proposed token (verify batch) + the
index for the next step's 262K target-embed fetch; draft_id is the gold-diff capture (kills M3's -1
sentinel). M=1 per draft step, so the single host sync per step is unavoidable (the forward needs the
host token to fetch the next embed) — argmax + d2t + D2H folded into one launch.

Tie-break: first (lowest) index on equal logits, matching torch.argmax / np.argmax — so the forward
oracle matches exactly (ties are measure-zero in real fp32 logits, but it's free).

Reduction mirrors softmax_gpu_mojo's warp-shuffle + smem cross-warp pattern, extended to carry the
argmax index alongside the max value.
"""

from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from max.gpu import barrier
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from max.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from layout import row_major
from layout import stack_allocation
from lib.cuda import cuda_malloc, cuda_free, cuda_sync, cuda_download_i32


comptime MAX_WARPS = 32


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_i32_ptr(addr: UInt64) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(addr))


def eagle3_argmax_d2t_kernel(
    logits: UnsafePointer[Float32, MutAnyOrigin],     # [DV] draft logits (32000)
    d2t: UnsafePointer[Float32, MutAnyOrigin],        # [DV] draft->target OFFSET LUT (fp32; offsets exact <2^24)
    out_pair: UnsafePointer[Int32, MutAnyOrigin],     # [2] = (draft_id, target_id)
    dv_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var dv = Int(dv_arg)
    var smem_v = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())
    var smem_i = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var tid = thread_idx.x
    var n_threads = block_dim.x

    # ── local argmax over a strided slice ──────────────────────────────────
    var best_v = Scalar[DType.float32](-3.4e38)
    var best_i = Int32(0)
    var i = tid
    while i < dv:
        var v = Scalar[DType.float32](logits[i])
        if v > best_v:
            best_v = v
            best_i = Int32(i)
        i += n_threads

    # ── warp reduce (value + index together; tie -> lower index) ───────────
    var off = WARP_SIZE // 2
    while off > 0:
        var ov = shuffle_xor(best_v, UInt32(off))
        var oi = shuffle_xor(best_i, UInt32(off))
        if ov > best_v or (ov == best_v and oi < best_i):
            best_v = ov
            best_i = oi
        off //= 2

    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem_v[wid] = best_v
        smem_i[wid] = best_i
    barrier()

    # ── cross-warp reduce in warp 0 ────────────────────────────────────────
    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var pv = Scalar[DType.float32](-3.4e38)
        var pi = Int32(0)
        if lane < n_warps:
            pv = smem_v[lane]
            pi = smem_i[lane]
        var o2 = WARP_SIZE // 2
        while o2 > 0:
            var ov = shuffle_xor(pv, UInt32(o2))
            var oi = shuffle_xor(pi, UInt32(o2))
            if ov > pv or (ov == pv and oi < pi):
                pv = ov
                pi = oi
            o2 //= 2
        if lane == 0:
            out_pair[0] = pi                                  # draft_id (argmax)
            out_pair[1] = pi + Int32(d2t[Int(pi)])            # target = draft + d2t[draft] (ADDITIVE)


# Returns (draft_id, target_id) as host Ints. draft_id ∈ [0,dv); target_id ∈ [0,262144).
def eagle3_argmax_d2t(
    ctx: DeviceContext,
    d_logits: UInt64,    # [DV] draft logits
    d_d2t: UInt64,       # [DV] int64 OFFSET LUT
    dv: Int,             # draft vocab (32000)
) raises -> Tuple[Int, Int]:
    if dv <= 0:
        return (0, 0)
    var threads = 1024
    if dv < 1024:
        threads = ((dv + 31) // 32) * 32
        if threads < 32:
            threads = 32
    var d_pair = cuda_malloc(8)  # [2] int32
    var k = ctx.compile_function[eagle3_argmax_d2t_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_logits), _as_f32_ptr(d_d2t), _as_i32_ptr(d_pair), Int32(dv),
        grid_dim=1, block_dim=threads,
    )
    cuda_sync()
    var host = List[Int32]()
    host.append(Int32(0))
    host.append(Int32(0))
    cuda_download_i32(host, d_pair, 2)
    cuda_free(d_pair)
    return (Int(host[0]), Int(host[1]))
