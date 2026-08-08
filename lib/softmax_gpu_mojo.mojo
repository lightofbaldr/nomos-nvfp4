"""Phase 3: pure-Mojo softmax-over-heads kernel.

Replaces softmax_kernel.cu (libsoftmax.so). The scores tensor is [nh, seq_len]
laid out tightly. We softmax along seq_len for each head, in-place.

Three phases:
  1. block_max — find row max for numerical stability
  2. block_sum — sum of exp(x - row_max), with x updated in-place
  3. divide   — each thread normalizes its slice by 1/row_sum

Layout:
  - One thread block per head (grid_dim = nh).
  - Block sum reduction uses std.gpu.primitives.block.sum.
  - Block max reduction is hand-rolled inline (no stock primitive yet).

Drop-in for lib/softmax_gpu.mojo — same Python-level signature.
"""

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives import block
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from std.math import exp
from layout import row_major
from layout import stack_allocation


comptime MAX_WARPS = 32                  # max threads/block / WARP_SIZE


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def softmax_over_heads_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],
    nh_arg: Int32,
    seq_len_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var nh = Int(nh_arg)
    var seq_len = Int(seq_len_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var head = block_idx.x
    var tid = thread_idx.x
    var n_threads = block_dim.x
    if head >= nh:
        return

    var row_off = head * seq_len

    # ── Phase 1: block max (for numerical stability) ───────────────────────
    var local_max = Scalar[DType.float32](-3.4e38)
    var i = tid
    while i < seq_len:
        var v = Scalar[DType.float32](scores[row_off + i])
        if v > local_max:
            local_max = v
        i += n_threads

    # Warp-level max
    var warp_off = WARP_SIZE // 2
    while warp_off > 0:
        var other = shuffle_xor(local_max, UInt32(warp_off))
        if other > local_max:
            local_max = other
        warp_off //= 2

    # Cross-warp via smem
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = local_max
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](-3.4e38)
        if lane < n_warps:
            partial = smem[lane]
        var off2 = WARP_SIZE // 2
        while off2 > 0:
            var other = shuffle_xor(partial, UInt32(off2))
            if other > partial:
                partial = other
            off2 //= 2
        if lane == 0:
            smem[0] = partial
    barrier()
    var row_max = smem[0]

    # ── Phase 2: exp(x - max) and sum ──────────────────────────────────────
    var local_sum = Scalar[DType.float32](0.0)
    var j = tid
    while j < seq_len:
        var v = exp(Scalar[DType.float32](scores[row_off + j]) - row_max)
        scores[row_off + j] = Float32(v)
        local_sum += v
        j += n_threads

    # Cross-warp sum via shuffle+smem (matches the max-reduction pattern above).
    var warp_total = local_sum
    var so = WARP_SIZE // 2
    while so > 0:
        warp_total += shuffle_xor(warp_total, UInt32(so))
        so //= 2
    if lane == 0:
        smem[wid] = warp_total
    barrier()
    var row_sum = Scalar[DType.float32](0.0)
    if wid == 0:
        var partial2 = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial2 = smem[lane]
        var so2 = WARP_SIZE // 2
        while so2 > 0:
            partial2 += shuffle_xor(partial2, UInt32(so2))
            so2 //= 2
        if lane == 0:
            smem[0] = partial2
    barrier()
    row_sum = smem[0]
    var inv = Scalar[DType.float32](1.0) / row_sum

    # ── Phase 3: divide ────────────────────────────────────────────────────
    var k = tid
    while k < seq_len:
        scores[row_off + k] = Float32(Scalar[DType.float32](scores[row_off + k]) * inv)
        k += n_threads


def gpu_softmax_over_heads_mojo(
    ctx: DeviceContext, d_scores: UInt64, nh: Int, seq_len: Int
) raises:
    if nh <= 0 or seq_len <= 0:
        return
    var threads = 256
    if seq_len < 256:
        threads = ((seq_len + 31) // 32) * 32
        if threads < 32:
            threads = 32
    if threads > 1024:
        threads = 1024
    var k = ctx.compile_function[softmax_over_heads_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_scores), Int32(nh), Int32(seq_len),
        grid_dim=nh, block_dim=threads,
    )
