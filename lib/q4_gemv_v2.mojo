"""q4_gemv_v2.mojo — Unit A (GLM), decode-round 2: GB10 occupancy + MLP lift for the
Q4_0 fused GEMV. Additive to lib/q4_weights.mojo (which stays UNTOUCHED).

PROBLEM (decode round 2): the original q4_gemv_kernel is grid_dim=N,
block_dim=WARP_SIZE = ONE warp/block, ONE row/block. On GB10 (48 SMs, 24 blocks/SM
cap, 48 warps/SM thread cap) that HARD-CAPS at 24 warps/SM = 50% occupancy, and more
importantly leaves too few outstanding loads in flight per SM to saturate the ~150
GB/s HBM bus (95% util / 67W = busy-but-starved: SMs stalled on single outstanding
loads, bus idle). Bottleneck = MEMORY-LEVEL PARALLELISM (in-flight loads), not raw
occupancy alone.

This file ports the Q4 GEMV onto the v3 template (WPB warps/block) in three
incremental kernels so each win is attributable:

  q4_gemv_v2_occ_kernel[WPB]   — WPB warps/block, 1 warp/row (n = blk*WPB+warp_id),
                                 SAME lane->K mapping + SAME sequential accumulation
                                 as q4_gemv_kernel. ONLY change is occupancy (block_dim
                                 32 -> WPB*32) => BIT-IDENTICAL to the original. Isolates
                                 the 50%->100% occupancy lift. WPB=4 -> block_dim=128.

  q4_gemv_v2_sx_kernel[WPB]    — + shared-staged input: all WPB warps in a block share
                                 ONE K-vector (it is a GEMV), so co-load it into shared
                                 once per block and read from shared in the K-loop
                                 (original re-reads from global/L2 per row). Still
                                 BIT-IDENTICAL (shared holds identical values; FP order
                                 unchanged). Isolates the input-read win.

  q4_gemv_v2_vec_kernel[WPB,U] — + vectorized 16B (4xfp32) COALESCED input loads + K-loop
                                 unroll U. Coalesced loads need CONSECUTIVE elements per
                                 lane => the lane->K mapping CHANGES vs the strided-32
                                 original => FP accumulation order differs => NOT
                                 bit-identical to q4_gemv_kernel. Gated under the
                                 CORRECTED policy (2026-06-24): deterministic +
                                 within Q4 tolerance of an fp32 reference, accumulator
                                 fp32, near-tie argmax flips semantically free. This is
                                 the MLP/load-unroll saturator (multiple outstanding
                                 vector loads per lane fills the bus).

Parity gates (tools/):
  q4_gemv_v2_parity.mojo    — Phase #1/#2 bit-identical-to-original gate (max|orig-new|=0).
  q4_gemv_v2_vec_validate.mojo — Phase #2 deterministic + Q4-tolerance-vs-fp64 gate.

All additive. lib/q4_weights.mojo and the _mm_dev seam UNTOUCHED.
"""
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.memory import AddressSpace
from std.gpu import barrier
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer, bitcast
from layout import row_major
from layout import stack_allocation
from lib.q4_weights import GROUP, _as_u8_ptr, _as_f32_ptr

comptime VEC: Int = 4   # fp32 lanes per 16B vector load (width= must be comptime)
comptime V8: Int = 8   # vdec chunk width (8 fp32 = 4 packed bytes, group-aligned)


# ─────────────────────────────────────────────────────────────────────────
# Phase #1 — OCCUPANCY lift only (BIT-IDENTICAL to q4_gemv_kernel).
# Same per-row math; WPB independent rows packed per block -> block_dim = WPB*32.
# ─────────────────────────────────────────────────────────────────────────
def q4_gemv_v2_occ_kernel[
    WPB: Int,    # warps per block (1->keeps original; 4->100% GB10 occupancy)
](
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [nb fp16 scales][nb*16 packed nibbles]
    inp: UnsafePointer[Float32, MutAnyOrigin],    # input activation, K elems, FP32
    outp: UnsafePointer[Float32, MutAnyOrigin],   # output, N elems, FP32
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,                                       # (N*K + 31)//32
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id        # this warp's output row (identical mapping)
    if n >= N:
        return
    var row_base = n * K

    var local = Scalar[DType.float32](0.0)
    var kk = lane
    while kk < K:
        var i = row_base + kk
        var block = i // GROUP
        var j = i % GROUP
        var sbits = blob.bitcast[UInt16]()[block]
        var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
        var byte = blob[nb * 2 + block * 16 + (j // 2)]
        var nib = (byte & 0x0F) if (j & 1) == 0 else ((byte >> 4) & 0x0F)
        var q = Int(nib) - 8
        var w = Float32(q) * scale[0]
        local += w * inp[kk]
        kk += WARP_SIZE

    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


# ─────────────────────────────────────────────────────────────────────────
# Phase #1b — + SHARED-STAGED input (BIT-IDENTICAL). All WPB warps in a block
# share the same K-vector (GEMV); co-load it to shared once, read from shared.
# ─────────────────────────────────────────────────────────────────────────
def q4_gemv_v2_sx_kernel[
    WPB: Int,
    KILE: Int,    # shared-input tile width (K must be tiled; KILE divides or pads)
](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    if n >= N:
        return
    var row_base = n * K
    var nthreads = WPB * WARP_SIZE

    var xs = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[KILE]())

    var local = Scalar[DType.float32](0.0)
    var kt0 = 0
    while kt0 < K:
        var kt1 = kt0 + KILE
        if kt1 > K:
            kt1 = K
        var klen = kt1 - kt0
        # Co-load input tile [kt0, kt1) into shared, once per block, all threads help.
        var t = tid
        while t < klen:
            xs[t] = inp[kt0 + t]
            t += nthreads
        barrier()
        # Each warp's K-loop reads its strided elements from shared (same mapping as occ).
        var kk = lane
        while kk < klen:
            var i = row_base + kt0 + kk
            var block = i // GROUP
            var j = i % GROUP
            var sbits = blob.bitcast[UInt16]()[block]
            var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
            var byte = blob[nb * 2 + block * 16 + (j // 2)]
            var nib = (byte & 0x0F) if (j & 1) == 0 else ((byte >> 4) & 0x0F)
            var q = Int(nib) - 8
            var w = Float32(q) * scale[0]
            local += w * xs[kk]
            kk += WARP_SIZE
        barrier()    # before next tile overwrites xs
        kt0 = kt1

    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


# ─────────────────────────────────────────────────────────────────────────
# Phase #2 — + VECTORIZED 16B coalesced loads + K-loop UNROLL (deterministic +
# Q4-tolerance gate). Coalesced => consecutive elements per lane => lane->K mapping
# changes vs strided-32 original => NOT bit-identical, but deterministic (fixed order)
# and within Q4 tol of fp32. U = outstanding vector loads per lane (MLP multiplier).
# Q4 dequant is affine (w=(nib-8)*scale) => trivially vectorizable (no E2M1 bit-decode).
# ─────────────────────────────────────────────────────────────────────────
def q4_gemv_v2_vec_kernel[
    WPB: Int,
    U: Int,       # unroll = outstanding 4-elem vector loads per lane per K-iteration
](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    if n >= N:
        return
    var row_base = n * K

    # NEW lane->K mapping: lane owns a RUN of 4 consecutive elements, strided by
    # (WARP_SIZE*4*U) per outer iteration -> U outstanding 16B loads per lane.
    # E.g. lane L owns k = L*4, L*4 + WARP_SIZE*4, ... (U of them per iter).
    var local = Scalar[DType.float32](0.0)
    var lane_stride = WARP_SIZE * VEC * U   # K advance per outer iter (all lanes)
    var k0 = lane * VEC
    while k0 < K:
        # Issue U independent 16B loads (the compiler keeps them in flight -> MLP).
        @parameter
        for u in range(U):
            var k = k0 + u * WARP_SIZE * VEC
            if k + VEC <= K:
                var act = (inp + k).load[width=VEC]()              # 16B coalesced
                # Dequant the 4 consecutive weights at row_base+k..k+3 (affine Q4).
                var i0 = row_base + k
                var blk = i0 // GROUP
                var j0 = i0 % GROUP
                var sbits = blob.bitcast[UInt16]()[blk]
                var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
                # The 4 elements may straddle a group boundary (GROUP=32); recompute per elem.
                @parameter
                for vv in range(VEC):
                    var ii = i0 + vv
                    var b2 = ii // GROUP
                    var jj = ii % GROUP
                    var sc: Float32
                    if b2 == blk:
                        sc = scale[0]
                    else:
                        var sb2 = blob.bitcast[UInt16]()[b2]
                        sc = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sb2)).cast[DType.float32]()[0]
                    var byte = blob[nb * 2 + b2 * 16 + (jj // 2)]
                    var nib = (byte & 0x0F) if (jj & 1) == 0 else ((byte >> 4) & 0x0F)
                    var w = Float32(Int(nib) - 8) * sc
                    local += w * act[vv]
        k0 += lane_stride

    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


# ─────────────────────────────────────────────────────────────────────────
# Dispatchers
# ─────────────────────────────────────────────────────────────────────────
def gpu_matmul_q4_v2_occ_dev[
    WPB: Int,
](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    K: Int,
    N: Int,
) raises:
    """Phase #1 occupancy lift (bit-identical to q4_gemv_kernel). WPB warps/block,
    1 warp/row. WPB=4 -> 100% GB10 occupancy."""
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q4_gemv_v2_occ_kernel[WPB]]()
    var blocks = (N + WPB - 1) // WPB
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb),
        grid_dim=blocks, block_dim=WPB * WARP_SIZE,
    )


def gpu_matmul_q4_v2_sx_dev[
    WPB: Int,
    KILE: Int,
](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    K: Int,
    N: Int,
) raises:
    """Phase #1b occupancy + shared-staged input (bit-identical)."""
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q4_gemv_v2_sx_kernel[WPB, KILE]]()
    var blocks = (N + WPB - 1) // WPB
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb),
        grid_dim=blocks, block_dim=WPB * WARP_SIZE,
    )


def gpu_matmul_q4_v2_vec_dev[
    WPB: Int,
    U: Int,
](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    K: Int,
    N: Int,
) raises:
    """Phase #2 occupancy + vectorized unrolled loads (deterministic + Q4-tol gate).
    U = outstanding 16B loads per lane."""
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q4_gemv_v2_vec_kernel[WPB, U]]()
    var blocks = (N + WPB - 1) // WPB
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb),
        grid_dim=blocks, block_dim=WPB * WARP_SIZE,
    )


# ─────────────────────────────────────────────────────────────────────────
# Phase #2b — FULLY VECTORIZED dequant (vdec). Unlike `vec` (vectorized load +
# scalar nibble unpack + scalar FMA), this vectorizes the NIBBLE UNPACK and the
# affine dequant + FMA as SIMD. Tests hypothesis: the scalar byte-op nibble
# extract (not FLOPs) is the instruction-issue limiter. Deterministic + Q4-tol gate.
# VEC=8, group-aligned chunks (j = lane*8 % 32 in {0,8,16,24} => 8 nibbles = 4
# contiguous packed bytes, all within one Q4 group of 32).
# ─────────────────────────────────────────────────────────────────────────
def q4_gemv_v2_vdec_kernel[
    WPB: Int,
](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    if n >= N:
        return
    var row_base = n * K

    var local = Scalar[DType.float32](0.0)
    var lane_stride = WARP_SIZE * V8         # K advance per outer iter
    var k0 = lane * V8
    while k0 < K:
        if k0 + V8 <= K:
            var i0 = row_base + k0
            var g = i0 // GROUP                 # the one Q4 group these 8 elems share
            var j = i0 % GROUP                 # in {0,8,16,24} (group-aligned)
            # scale (fp16) for this group -> broadcast fp32.
            var sbits = blob.bitcast[UInt16]()[g]
            var scale_f16 = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits))
            var scale = scale_f16.cast[DType.float32]()[0]
            # 8 nibbles = 4 contiguous packed bytes (j even, j//2 in {0,4,8,12}) -> UInt32.
            var poff = nb * 2 + g * 16 + (j // 2)
            var pk = (blob + poff).bitcast[UInt32]()[0]
            # SIMD nibble unpack: little-endian bytes -> [b0lo,b0hi,b1lo,b1hi,b2lo,b2hi,b3lo,b3hi].
            var nib = SIMD[DType.uint32, V8]()
            nib[0] = pk & UInt32(0xF)
            nib[1] = (pk >> UInt32(4)) & UInt32(0xF)
            nib[2] = (pk >> UInt32(8)) & UInt32(0xF)
            nib[3] = (pk >> UInt32(12)) & UInt32(0xF)
            nib[4] = (pk >> UInt32(16)) & UInt32(0xF)
            nib[5] = (pk >> UInt32(20)) & UInt32(0xF)
            nib[6] = (pk >> UInt32(24)) & UInt32(0xF)
            nib[7] = (pk >> UInt32(28)) & UInt32(0xF)
            # Affine dequant (SIMD): w = (nib - 8) * scale.
            var wf = nib.cast[DType.float32]()
            var w = (wf - SIMD[DType.float32, V8](8.0)) * SIMD[DType.float32, V8](scale)
            # Coalesced 8x fp32 activation load (32B).
            var act = (inp + k0).load[width=V8]()
            var contrib = w * act
            # Sequential accumulate (parity-deterministic: fixed t order).
            @parameter
            for vv in range(V8):
                local += contrib[vv]
        k0 += lane_stride

    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


def gpu_matmul_q4_v2_vdec_dev[
    WPB: Int,
](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    K: Int,
    N: Int,
) raises:
    """Phase #2b fully-vectorized dequant (det + Q4-tol gate)."""
    # GUARD: each lane consumes V8(=8)-wide chunks (k0 += WARP_SIZE*V8) and only
    # accumulates when k0+V8<=K, so a K not divisible by V8 SILENTLY DROPS the
    # straddling tail -> a structural (not fp-reorder) error vs occ4. Live decode K
    # in {5376,21504} are %8==0; fail loudly if a future shape isn't, instead of
    # corrupting silently. (Found by the vdec e2e-parity adversarial review, 2026-06-25.)
    if K % 8 != 0:
        raise Error("gpu_matmul_q4_v2_vdec_dev: K=" + String(K) +
                    " not divisible by V8=8 -> vdec would drop the K-tail; use occ4 for this shape")
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q4_gemv_v2_vdec_kernel[WPB]]()
    var blocks = (N + WPB - 1) // WPB
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb),
        grid_dim=blocks, block_dim=WPB * WARP_SIZE,
    )
