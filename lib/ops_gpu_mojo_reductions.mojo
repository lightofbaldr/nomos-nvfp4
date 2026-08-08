"""Phase 2B: reduction kernels — rmsnorm family + qk_norm in pure Mojo.

These kernels need shared memory for cross-warp reduction. Layout:
  - One thread block per row (rmsnorm) or per head (rmsnorm_no_weight,
    qk_norm).
  - Threads cooperate on a sum-of-squares reduction:
    1. Each thread accumulates a partial sum_sq over its strided slice.
    2. Warp-level reduction via `warp_sum` (Mojo stdlib helper).
    3. Cross-warp reduction by writing one value per warp to smem,
       barrier, then a single warp does another `warp_sum` over those.
    4. Final value is broadcast to all threads via smem[0].
  - Then each thread normalizes its strided slice and writes back.

Smem requirement: 32 floats (one slot per warp, max 32 warps per block).

These complement lib/ops_gpu_mojo.mojo (the trivial kernels). Together
they cover all 11 kernels in ops_kernel.cu.
"""

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives.warp import sum as warp_sum, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from std.math import sqrt
from layout import row_major
from layout import stack_allocation


comptime MAX_WARPS = 32                # max threads/block / WARP_SIZE = 1024/32
comptime EPS_RMSNORM = Scalar[DType.float32](1e-6)


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


# ═════════════════════════════════════════════════════════════════════════════
# RMSNorm (out-of-place): y = x * rsqrt(mean(x^2)+eps) * w
# ═════════════════════════════════════════════════════════════════════════════

def rmsnorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    d_arg: Int32,
):
    var d = Int(d_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var tid = thread_idx.x
    var n_threads = block_dim.x

    # Phase 1: per-thread partial sum of squares
    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < d:
        var v = x[i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads

    # Phase 2: warp reduction
    var warp_total = warp_sum(local_sq)

    # Phase 3: cross-warp reduction via smem
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()

    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](d) + EPS_RMSNORM)

    # Phase 4: apply
    var j = tid
    while j < d:
        y[j] = Float32(x[j] * Float32(rms_inv) * w[j])
        j += n_threads


def gpu_rmsnorm_mojo(
    ctx: DeviceContext, d_x: UInt64, d_y: UInt64, d_w: UInt64, d: Int
) raises:
    var threads = 256 if d < 1024 else 1024
    var k = ctx.compile_function[rmsnorm_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_y), _as_f32_ptr(d_w), Int32(d),
        grid_dim=1, block_dim=threads,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RMSNorm in-place: x = x * rsqrt(mean(x^2)+eps) * w
# ═════════════════════════════════════════════════════════════════════════════

def rmsnorm_inplace_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    d_arg: Int32,
):
    var d = Int(d_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var tid = thread_idx.x
    var n_threads = block_dim.x

    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < d:
        var v = x[i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads

    var warp_total = warp_sum(local_sq)
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()

    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](d) + EPS_RMSNORM)

    var j = tid
    while j < d:
        x[j] = Float32(x[j] * Float32(rms_inv) * w[j])
        j += n_threads


def gpu_rmsnorm_inplace_mojo(
    ctx: DeviceContext, d_x: UInt64, d_w: UInt64, d: Int
) raises:
    var threads = 256 if d < 1024 else 1024
    var k = ctx.compile_function[rmsnorm_inplace_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_w), Int32(d),
        grid_dim=1, block_dim=threads,
    )


def rmsnorm_residual_add_scaled_kernel(
    residual: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    d_arg: Int32,
    scale: Float32,
):
    var d = Int(d_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var tid = thread_idx.x
    var n_threads = block_dim.x

    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < d:
        var v = x[i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads

    var warp_total = warp_sum(local_sq)
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()

    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](d) + EPS_RMSNORM)

    var j = tid
    while j < d:
        var normed = Float32(x[j] * Float32(rms_inv) * w[j])
        x[j] = normed
        var added = Float32(residual[j] + normed)
        residual[j] = added * scale
        j += n_threads


def gpu_rmsnorm_residual_add_scaled_mojo(
    ctx: DeviceContext,
    d_residual: UInt64,
    d_x: UInt64,
    d_w: UInt64,
    d: Int,
    scale: Float32,
) raises:
    var threads = 256 if d < 1024 else 1024
    var k = ctx.compile_function[rmsnorm_residual_add_scaled_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_residual), _as_f32_ptr(d_x), _as_f32_ptr(d_w), Int32(d),
        scale,
        grid_dim=1, block_dim=threads,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Batched RMSNorm (out-of-place + in-place): one block per row, grid_dim = S.
# row_off = block_idx.x * d. Norm weight w[d] is shared across rows (per-feature).
# Used by batched prefill. The no-weight (V-norm) and qk_norm kernels need NO
# batched variant — they already key off block_idx.x as a per-head row, so a
# [S, n_heads, hd] layout works by launching grid_dim = S*n_heads unchanged.
# ═════════════════════════════════════════════════════════════════════════════

def rmsnorm_kernel_batched(
    x: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    d_arg: Int32,
):
    var d = Int(d_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())
    var row_off = block_idx.x * d
    var tid = thread_idx.x
    var n_threads = block_dim.x

    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < d:
        var v = x[row_off + i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads
    var warp_total = warp_sum(local_sq)
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()
    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()
    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](d) + EPS_RMSNORM)
    var j = tid
    while j < d:
        y[row_off + j] = Float32(x[row_off + j] * Float32(rms_inv) * w[j])
        j += n_threads


def gpu_rmsnorm_batched_mojo(
    ctx: DeviceContext, d_x: UInt64, d_y: UInt64, d_w: UInt64, d: Int, S: Int
) raises:
    var threads = 256 if d < 1024 else 1024
    var k = ctx.compile_function[rmsnorm_kernel_batched]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_y), _as_f32_ptr(d_w), Int32(d),
        grid_dim=S, block_dim=threads,
    )


def rmsnorm_inplace_kernel_batched(
    x: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    d_arg: Int32,
):
    var d = Int(d_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())
    var row_off = block_idx.x * d
    var tid = thread_idx.x
    var n_threads = block_dim.x

    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < d:
        var v = x[row_off + i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads
    var warp_total = warp_sum(local_sq)
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()
    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()
    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](d) + EPS_RMSNORM)
    var j = tid
    while j < d:
        x[row_off + j] = Float32(x[row_off + j] * Float32(rms_inv) * w[j])
        j += n_threads


def gpu_rmsnorm_inplace_batched_mojo(
    ctx: DeviceContext, d_x: UInt64, d_w: UInt64, d: Int, S: Int
) raises:
    var threads = 256 if d < 1024 else 1024
    var k = ctx.compile_function[rmsnorm_inplace_kernel_batched]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_w), Int32(d),
        grid_dim=S, block_dim=threads,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RMSNorm per-head, no weight (V-norm)
# ═════════════════════════════════════════════════════════════════════════════

def rmsnorm_no_weight_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    n_heads_arg: Int32,
    hd_arg: Int32,
):
    var n_heads = Int(n_heads_arg)
    var hd = Int(hd_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var head = block_idx.x
    var tid = thread_idx.x
    var n_threads = block_dim.x
    if head >= n_heads:
        return

    var row_off = head * hd

    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < hd:
        var v = x[row_off + i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads

    var warp_total = warp_sum(local_sq)
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()

    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](hd) + EPS_RMSNORM)

    var j = tid
    while j < hd:
        x[row_off + j] = Float32(x[row_off + j] * Float32(rms_inv))
        j += n_threads


def gpu_rmsnorm_no_weight_mojo(
    ctx: DeviceContext, d_x: UInt64, n_heads: Int, hd: Int
) raises:
    var threads = 64 if hd < 256 else 256
    if threads < 32:
        threads = 32
    var k = ctx.compile_function[rmsnorm_no_weight_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), Int32(n_heads), Int32(hd),
        grid_dim=n_heads, block_dim=threads,
    )


# ═════════════════════════════════════════════════════════════════════════════
# QK norm: per-head RMSNorm with shared weight vector
# ═════════════════════════════════════════════════════════════════════════════

def qk_norm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    n_heads_arg: Int32,
    hd_arg: Int32,
):
    var n_heads = Int(n_heads_arg)
    var hd = Int(hd_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var head = block_idx.x
    var tid = thread_idx.x
    var n_threads = block_dim.x
    if head >= n_heads:
        return

    var row_off = head * hd

    var local_sq = Scalar[DType.float32](0.0)
    var i = tid
    while i < hd:
        var v = x[row_off + i]
        local_sq += Scalar[DType.float32](v * v)
        i += n_threads

    var warp_total = warp_sum(local_sq)
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = warp_total
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial = smem[lane]
        var total = warp_sum(partial)
        if lane == 0:
            smem[0] = total
    barrier()

    var sum_sq = smem[0]
    var rms_inv = Scalar[DType.float32](1.0) / sqrt(sum_sq / Scalar[DType.float32](hd) + EPS_RMSNORM)

    var j = tid
    while j < hd:
        x[row_off + j] = Float32(x[row_off + j] * Float32(rms_inv) * w[j])
        j += n_threads


def gpu_qk_norm_mojo(
    ctx: DeviceContext, d_x: UInt64, d_w: UInt64, n_heads: Int, hd: Int
) raises:
    var threads = 64 if hd < 256 else 256
    if threads < 32:
        threads = 32
    var k = ctx.compile_function[qk_norm_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_w), Int32(n_heads), Int32(hd),
        grid_dim=n_heads, block_dim=threads,
    )
