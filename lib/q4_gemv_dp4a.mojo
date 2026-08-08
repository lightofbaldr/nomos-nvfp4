"""dp4a Q4_0 decode GEMV — int8 `dp4a` on q8_1-quantized activations (the path to
10 tok/s on GB10). Validated standalone with the Q4 DP4A microbenchmark:
EXACT vs host int-ref (max_rel 1.1e-7), 211 GB/s on the real gate/up shape (>llama.cpp
191, 2.85x our fp32-act occ4 kernel which is ALU-bound on the dequant).

Math (per 32-block, our Q4_0 format = [nb fp16 d4][nb*16 nibble-pairs]):
  w_i = d4*(nib_i - 8);  a_i = d8*q8_i  =>  sum_i w_i a_i = d4*d8*(sumi - 8*sumq)
  sumi = Σ nib_i·q8_i   (8× dp4a, raw nibbles 0-15)
  sumq = Σ q8_i         (dp4a(0x01010101, q8) ones-trick)
The -8 is the Q4_0 zero-point (q8_1 stores the per-block scale; sumq carries the offset).

This is the q8-acts PRODUCTION precision mode. The dispatch (_mm_dev) selects activation
precision via `act_precision()`: 8 = q8 dp4a (this, production), 32 = fp32 acts (occ4,
research/high-precision). 16 (fp16) and 4 (q4 acts, end-state) are documented bolt-in
seams — add a branch here + an act_precision() value, no structural change.
"""
from std.gpu.host import DeviceContext
from std.gpu import barrier, thread_idx, block_idx, block_dim
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
)
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer, bitcast, stack_allocation as raw_stack_allocation
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.math import round
from std.collections import InlineArray
from layout import row_major
from layout import stack_allocation
from lib.q4_weights import GROUP, _as_u8_ptr, _as_f32_ptr
from lib.engine_init import _env_flag_cached, ENV_Q4_DP4A


def _as_i8_ptr(addr: UInt64) -> UnsafePointer[Int8, MutAnyOrigin]:
    return UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(addr))


def act_precision() -> Int:
    """Activation-precision selector for the Q4 decode GEMV. 8 = q8 dp4a (production),
    32 = fp32 acts (occ4). Seam for 16 (fp16) / 4 (q4 acts) — add a value + a _mm_dev
    branch. NOMOS_Q4_DP4A=1 -> 8; default (unset/0) -> 32 (occ4, safe current behavior)."""
    return 8 if _env_flag_cached(ENV_Q4_DP4A) == 1 else 32


@always_inline("nodebug")
def _dp4a(a: Int32, b: Int32, c: Int32) -> Int32:
    return inlined_assembly[
        "dp4a.s32.s32 $0, $1, $2, $3;",
        Int32,
        Int32,
        Int32,
        Int32,
        constraints="=r,r,r,r",
        has_side_effect=False,
    ](a, b, c)


@always_inline("nodebug")
def _q4_u16_to_dp4a_word(w: UInt16) -> Int32:
    var byte0 = UInt8(w & 0xFF)
    var byte1 = UInt8((w >> 8) & 0xFF)
    return Int32(Int(byte0 & 0x0F)) | (
        Int32(Int((byte0 >> 4) & 0x0F)) << 8
    ) | (Int32(Int(byte1 & 0x0F)) << 16) | (
        Int32(Int((byte1 >> 4) & 0x0F)) << 24
    )


@always_inline("nodebug")
def _q8_dot8_words(
    q8w: UnsafePointer[Int32, MutAnyOrigin],
    ones: Int32,
    a0: Int32,
    a1: Int32,
    a2: Int32,
    a3: Int32,
    a4: Int32,
    a5: Int32,
    a6: Int32,
    a7: Int32,
) -> Int32:
    var sumi = Int32(0)
    var sumq = Int32(0)
    var bp = q8w[0]
    sumi = _dp4a(a0, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[1]
    sumi = _dp4a(a1, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[2]
    sumi = _dp4a(a2, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[3]
    sumi = _dp4a(a3, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[4]
    sumi = _dp4a(a4, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[5]
    sumi = _dp4a(a5, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[6]
    sumi = _dp4a(a6, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    bp = q8w[7]
    sumi = _dp4a(a7, bp, sumi)
    sumq = _dp4a(ones, bp, sumq)
    return sumi - 8 * sumq


# ── q8_1 activation quant: input[K] fp32 -> scratch [nbk fp32 d8][nbk*32 int8 q8] ──
def _q8_dot8_vals(
    b0: Int32,
    b1: Int32,
    b2: Int32,
    b3: Int32,
    b4: Int32,
    b5: Int32,
    b6: Int32,
    b7: Int32,
    ones: Int32,
    a0: Int32,
    a1: Int32,
    a2: Int32,
    a3: Int32,
    a4: Int32,
    a5: Int32,
    a6: Int32,
    a7: Int32,
) -> Int32:
    # Value-input twin of _q8_dot8_words (activation words pre-loaded from
    # shared memory). Same dp4a sequence, same sumi/sumq interleave: integer
    # ops, bit-identical to the pointer variant.
    var sumi = Int32(0)
    var sumq = Int32(0)
    sumi = _dp4a(a0, b0, sumi)
    sumq = _dp4a(ones, b0, sumq)
    sumi = _dp4a(a1, b1, sumi)
    sumq = _dp4a(ones, b1, sumq)
    sumi = _dp4a(a2, b2, sumi)
    sumq = _dp4a(ones, b2, sumq)
    sumi = _dp4a(a3, b3, sumi)
    sumq = _dp4a(ones, b3, sumq)
    sumi = _dp4a(a4, b4, sumi)
    sumq = _dp4a(ones, b4, sumq)
    sumi = _dp4a(a5, b5, sumi)
    sumq = _dp4a(ones, b5, sumq)
    sumi = _dp4a(a6, b6, sumi)
    sumq = _dp4a(ones, b6, sumq)
    sumi = _dp4a(a7, b7, sumi)
    sumq = _dp4a(ones, b7, sumq)
    return sumi - 8 * sumq


comptime Q4_FIXED_ACC_SCALE = Float32(16384.0)  # 2^14
comptime Q4_FIXED_ACC_INV_SCALE = Float32(0.00006103515625)


@always_inline("nodebug")
def _q4_fixed_term(d4: Float32, d8v: Float32, dot: Int32) -> Int32:
    # Shared fixed-point epilogue term for decode and verify. The int dot is
    # exact; Q14 is enough below Q4 noise while keeping the hot reductions in
    # Int32. Decode and verify use the same term, so block order is irrelevant.
    return Int32(round(d4 * d8v * Float32(dot) * Q4_FIXED_ACC_SCALE))


@always_inline("nodebug")
def _q4_fixed_to_f32(acc: Int32) -> Float32:
    return Float32(acc) * Q4_FIXED_ACC_INV_SCALE


def quantize_q8_1_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    K_arg: Int32,
    nbk_arg: Int32,
):
    var K = Int(K_arg)
    var nbk = Int(nbk_arg)
    var bk = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bk >= nbk:
        return
    var amax = Float32(0.0)
    var p = 0
    while p < 32:
        var k = bk * 32 + p
        var x = inp[k] if k < K else Float32(0.0)
        var ax = x if x >= 0 else -x
        if ax > amax:
            amax = ax
        p += 1
    var dscale = amax / 127.0 if amax > 0 else Float32(1.0)
    d8[bk] = dscale
    p = 0
    while p < 32:
        var k = bk * 32 + p
        var x = inp[k] if k < K else Float32(0.0)
        var qi = Int(round(x / dscale))
        if qi > 127:
            qi = 127
        if qi < -128:
            qi = -128
        q8[bk * 32 + p] = Int8(qi)
        p += 1


def quantize_q8_1_rows_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    K_arg: Int32,
    nbk_arg: Int32,
    rows_arg: Int32,
):
    var K = Int(K_arg)
    var nbk = Int(nbk_arg)
    var rows = Int(rows_arg)
    # Batched rows variant: one launch quantizes all S rows (was one launch per
    # row — ~471 launches/row across the 62-layer verify pass). Per-(row, block)
    # math is verbatim quantize_q8_1_kernel: values bit-identical.
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= rows * nbk:
        return
    var r = idx // nbk
    var bk = idx % nbk
    var rin = inp + r * K
    var amax = Float32(0.0)
    var p = 0
    while p < 32:
        var k = bk * 32 + p
        var x = rin[k] if k < K else Float32(0.0)
        var ax = x if x >= 0 else -x
        if ax > amax:
            amax = ax
        p += 1
    var dscale = amax / 127.0 if amax > 0 else Float32(1.0)
    d8[r * nbk + bk] = dscale
    p = 0
    while p < 32:
        var k = bk * 32 + p
        var x = rin[k] if k < K else Float32(0.0)
        var qi = Int(round(x / dscale))
        if qi > 127:
            qi = 127
        if qi < -128:
            qi = -128
        q8[(r * nbk + bk) * 32 + p] = Int8(qi)
        p += 1


# ── q8_1-quantize an activation [K] into d_scratch ([nbk f32 d8][nbk*32 i8 q8]).
#    Split out so GEMVs that SHARE an input (q/k/v read d_normed; gate/up read d_pn)
#    quantize ONCE then run gemv-only N times — saves the redundant quant + launches. ──
def gpu_q8_quantize_dev(
    ctx: DeviceContext, d_in_fp32: UInt64, d_scratch: UInt64, K: Int
) raises:
    var nbk = (K + GROUP - 1) // GROUP
    var qk = ctx.compile_function[quantize_q8_1_kernel]()
    var qblocks = (nbk + 255) // 256
    ctx.enqueue_function(
        qk, _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_scratch),
        _as_i8_ptr(d_scratch + UInt64(nbk * 4)), Int32(K), Int32(nbk),
        grid_dim=qblocks, block_dim=256,
    )


# ── dp4a GEMV reading a PRE-quantized q8_1 scratch (no quant). ──
def gpu_matmul_q4_dp4a_gemv_dev[WPB: Int](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_scratch: UInt64,  # pre-quantized q8_1 of the input (gpu_q8_quantize_dev)
    d_w_q4: UInt64,
    K: Int,
    N: Int,
) raises:
    var nbtot = (K * N + GROUP - 1) // GROUP
    var nbk = (K + GROUP - 1) // GROUP
    # Production decode uses the wide-scalar-load kernel: byte-identical dp4a
    # math, fewer load instructions per Q4 block.
    var gk = ctx.compile_function[q4_gemv_dp4a_v4_kernel[WPB]]()
    var blocks = (N + WPB - 1) // WPB
    ctx.enqueue_function(
        gk, _as_u8_ptr(d_w_q4), _as_f32_ptr(d_scratch),
        _as_i8_ptr(d_scratch + UInt64(nbk * 4)), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nbtot),
        grid_dim=blocks, block_dim=WPB * WARP_SIZE,
    )


# ── (a) wide-scalar-load variant — BYTE-IDENTICAL production decode kernel. ──
#    The base kernel reads each 32-block as 16 scalar UInt8 nibble + 32 scalar Int8 q8 loads
#    (48 byte-loads/block) — instruction-heavy vs the 214 float4 read ceiling. This cuts that to
#    8 uint16 + 8 int32 loads/block (16 total) WITHOUT SIMD: bp is just the little-endian int32
#    `(q8+q_base).bitcast[Int32]()[g]` (identical bits → identical dp4a); the nibble pair is one
#    uint16, unpacked in registers. BYTE-IDENTICAL (bit-compared, all shapes). Bench (2026-06-29):
#    big HBM-bound shapes D_FF/FF_D (62MB) 177→184 GB/s (~4%, 84→87% of ceiling); small shapes
#    show 2.6× but that EXCEEDS the 210 HBM ceiling = L2-resident artifact (bench reuses weights),
#    not real BW — needs a cache-busting bench or the e2e A/B to trust.
#    NOTE: the FULL-vectorized 16B SIMD load (one load/block) REGRESSED 20× — per-lane getitem
#    `nibs[2*g]` spilled the SIMD to local memory. Getting beyond this needs spill-free vector
#    COMPUTE (vector bitops on the loaded SIMD, not scalar extraction). ──
def q4_gemv_dp4a_v4_kernel[WPB: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nbtot_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nbtot = Int(nbtot_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    if n >= N:
        return
    var nbk = K // 32
    var ones = Int32(0x01010101)
    var acc = Int32(0)
    var b = lane
    while b < nbk:
        var gb = n * nbk + b
        var sbits = blob.bitcast[UInt16]()[gb]
        var d4 = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[
            DType.float32
        ]()[0]
        var nib_base = nbtot * 2 + gb * 16
        var q_base = b * 32
        var nib16 = (blob + nib_base).bitcast[UInt16]()   # 8 uint16 loads/block (was 16 scalar UInt8)
        var q8w = (q8 + q_base).bitcast[Int32]()           # 8 int32 loads/block (was 32 scalar Int8);
        var sumi = Int32(0)                                #   bp IS the little-endian int32 → identical bits
        var sumq = Int32(0)
        var g = 0
        while g < 8:
            var w = nib16[g]
            var byte0 = UInt8(w & 0xFF)
            var byte1 = UInt8((w >> 8) & 0xFF)
            var a = Int32(Int(byte0 & 0x0F)) | (
                Int32(Int((byte0 >> 4) & 0x0F)) << 8
            ) | (Int32(Int(byte1 & 0x0F)) << 16) | (
                Int32(Int((byte1 >> 4) & 0x0F)) << 24
            )
            var bp = q8w[g]
            sumi = _dp4a(a, bp, sumi)
            sumq = _dp4a(ones, bp, sumq)
            g += 1
        acc += _q4_fixed_term(d4, d8[b], sumi - 8 * sumq)
        b += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        acc += shuffle_xor(acc, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = _q4_fixed_to_f32(acc)


# ── combined quant+GEMV (single-input GEMVs: o-proj, down-proj, lm_head). ──
def gpu_matmul_q4_dp4a_dev[WPB: Int](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    d_scratch: UInt64,  # reuses the (unused-on-Q4) bf16 scratch: [nbk f32 d8][nbk*32 i8 q8]
    K: Int,
    N: Int,
) raises:
    gpu_q8_quantize_dev(ctx, d_in_fp32, d_scratch, K)
    gpu_matmul_q4_dp4a_gemv_dev[WPB](ctx, d_out_fp32, d_scratch, d_w_q4, K, N)


def q4_gemv_dp4a_s8_v4_kernel[WPB: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nbtot_arg: Int32,
    Sdraft_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nbtot = Int(nbtot_arg)
    var Sdraft = Int(Sdraft_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    if n >= N:
        return
    var nbk = K // 32
    var ones = Int32(0x01010101)
    var acc0 = Int32(0)
    var acc1 = Int32(0)
    var acc2 = Int32(0)
    var acc3 = Int32(0)
    var acc4 = Int32(0)
    var acc5 = Int32(0)
    var acc6 = Int32(0)
    var acc7 = Int32(0)
    var b = lane
    while b < nbk:
        var gb = n * nbk + b
        var sbits = blob.bitcast[UInt16]()[gb]
        var d4 = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[
            DType.float32
        ]()[0]
        var nib_base = nbtot * 2 + gb * 16
        var q_base = b * 32
        var nib16 = (blob + nib_base).bitcast[UInt16]()
        var a0 = _q4_u16_to_dp4a_word(nib16[0])
        var a1 = _q4_u16_to_dp4a_word(nib16[1])
        var a2 = _q4_u16_to_dp4a_word(nib16[2])
        var a3 = _q4_u16_to_dp4a_word(nib16[3])
        var a4 = _q4_u16_to_dp4a_word(nib16[4])
        var a5 = _q4_u16_to_dp4a_word(nib16[5])
        var a6 = _q4_u16_to_dp4a_word(nib16[6])
        var a7 = _q4_u16_to_dp4a_word(nib16[7])
        if Sdraft > 0:
            acc0 += _q4_fixed_term(d4, d8[b], _q8_dot8_words(
                (q8 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 1:
            acc1 += _q4_fixed_term(d4, d8[nbk + b], _q8_dot8_words(
                (q8 + nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 2:
            acc2 += _q4_fixed_term(d4, d8[2 * nbk + b], _q8_dot8_words(
                (q8 + 2 * nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 3:
            acc3 += _q4_fixed_term(d4, d8[3 * nbk + b], _q8_dot8_words(
                (q8 + 3 * nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 4:
            acc4 += _q4_fixed_term(d4, d8[4 * nbk + b], _q8_dot8_words(
                (q8 + 4 * nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 5:
            acc5 += _q4_fixed_term(d4, d8[5 * nbk + b], _q8_dot8_words(
                (q8 + 5 * nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 6:
            acc6 += _q4_fixed_term(d4, d8[6 * nbk + b], _q8_dot8_words(
                (q8 + 6 * nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        if Sdraft > 7:
            acc7 += _q4_fixed_term(d4, d8[7 * nbk + b], _q8_dot8_words(
                (q8 + 7 * nbk * 32 + q_base).bitcast[Int32](),
                ones, a0, a1, a2, a3, a4, a5, a6, a7,
            ))
        b += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        acc0 += shuffle_xor(acc0, UInt32(off))
        acc1 += shuffle_xor(acc1, UInt32(off))
        acc2 += shuffle_xor(acc2, UInt32(off))
        acc3 += shuffle_xor(acc3, UInt32(off))
        acc4 += shuffle_xor(acc4, UInt32(off))
        acc5 += shuffle_xor(acc5, UInt32(off))
        acc6 += shuffle_xor(acc6, UInt32(off))
        acc7 += shuffle_xor(acc7, UInt32(off))
        off //= 2
    if lane == 0:
        if Sdraft > 0:
            outp[n] = _q4_fixed_to_f32(acc0)
        if Sdraft > 1:
            outp[N + n] = _q4_fixed_to_f32(acc1)
        if Sdraft > 2:
            outp[2 * N + n] = _q4_fixed_to_f32(acc2)
        if Sdraft > 3:
            outp[3 * N + n] = _q4_fixed_to_f32(acc3)
        if Sdraft > 4:
            outp[4 * N + n] = _q4_fixed_to_f32(acc4)
        if Sdraft > 5:
            outp[5 * N + n] = _q4_fixed_to_f32(acc5)
        if Sdraft > 6:
            outp[6 * N + n] = _q4_fixed_to_f32(acc6)
        if Sdraft > 7:
            outp[7 * N + n] = _q4_fixed_to_f32(acc7)


def q4_gemv_dp4a_s7_v4_kernel[WPB: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nbtot_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nbtot = Int(nbtot_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    if n >= N:
        return
    var nbk = K // 32
    var ones = Int32(0x01010101)
    var acc0 = Int32(0)
    var acc1 = Int32(0)
    var acc2 = Int32(0)
    var acc3 = Int32(0)
    var acc4 = Int32(0)
    var acc5 = Int32(0)
    var acc6 = Int32(0)
    var b = lane
    while b < nbk:
        var gb = n * nbk + b
        var sbits = blob.bitcast[UInt16]()[gb]
        var d4 = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[
            DType.float32
        ]()[0]
        var nib_base = nbtot * 2 + gb * 16
        var q_base = b * 32
        var nib16 = (blob + nib_base).bitcast[UInt16]()
        var a0 = _q4_u16_to_dp4a_word(nib16[0])
        var a1 = _q4_u16_to_dp4a_word(nib16[1])
        var a2 = _q4_u16_to_dp4a_word(nib16[2])
        var a3 = _q4_u16_to_dp4a_word(nib16[3])
        var a4 = _q4_u16_to_dp4a_word(nib16[4])
        var a5 = _q4_u16_to_dp4a_word(nib16[5])
        var a6 = _q4_u16_to_dp4a_word(nib16[6])
        var a7 = _q4_u16_to_dp4a_word(nib16[7])
        acc0 += _q4_fixed_term(d4, d8[b], _q8_dot8_words(
            (q8 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        acc1 += _q4_fixed_term(d4, d8[nbk + b], _q8_dot8_words(
            (q8 + nbk * 32 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        acc2 += _q4_fixed_term(d4, d8[2 * nbk + b], _q8_dot8_words(
            (q8 + 2 * nbk * 32 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        acc3 += _q4_fixed_term(d4, d8[3 * nbk + b], _q8_dot8_words(
            (q8 + 3 * nbk * 32 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        acc4 += _q4_fixed_term(d4, d8[4 * nbk + b], _q8_dot8_words(
            (q8 + 4 * nbk * 32 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        acc5 += _q4_fixed_term(d4, d8[5 * nbk + b], _q8_dot8_words(
            (q8 + 5 * nbk * 32 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        acc6 += _q4_fixed_term(d4, d8[6 * nbk + b], _q8_dot8_words(
            (q8 + 6 * nbk * 32 + q_base).bitcast[Int32](),
            ones, a0, a1, a2, a3, a4, a5, a6, a7,
        ))
        b += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        acc0 += shuffle_xor(acc0, UInt32(off))
        acc1 += shuffle_xor(acc1, UInt32(off))
        acc2 += shuffle_xor(acc2, UInt32(off))
        acc3 += shuffle_xor(acc3, UInt32(off))
        acc4 += shuffle_xor(acc4, UInt32(off))
        acc5 += shuffle_xor(acc5, UInt32(off))
        acc6 += shuffle_xor(acc6, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = _q4_fixed_to_f32(acc0)
        outp[N + n] = _q4_fixed_to_f32(acc1)
        outp[2 * N + n] = _q4_fixed_to_f32(acc2)
        outp[3 * N + n] = _q4_fixed_to_f32(acc3)
        outp[4 * N + n] = _q4_fixed_to_f32(acc4)
        outp[5 * N + n] = _q4_fixed_to_f32(acc5)
        outp[6 * N + n] = _q4_fixed_to_f32(acc6)


def q4_gemv_dp4a_s16_actshare_v4_kernel[WPB: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nbtot_arg: Int32,
    Sdraft_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nbtot = Int(nbtot_arg)
    var Sdraft = Int(Sdraft_arg)
    # s16_v4 with the ACTIVATION traffic fixed. The per-row cost of the fused
    # verify was never weights (s8/s16 already load them once per column): it
    # is every column-warp re-reading each row's Q8 activations from L2 —
    # 32B of acts per 18B weight block PER ROW (32x the weight bytes at S=16).
    # Here the WPB warps of a block cooperatively stage one K-tile
    # (WARP_SIZE blocks) of all S activation rows + scales in shared memory,
    # cutting activation reads by WPB. Grid/occupancy match s16_v4. Per-lane
    # block sequence (tile*32 + lane == lane, lane+32, ...), dp4a interleave,
    # and shuffle_xor tree match v4. The epilogue uses the shared fixed-point
    # term, so reduction order is irrelevant as long as decode uses the same
    # term.
    # Padded smem stride (9 words/block) keeps compute reads bank-conflict-free.
    var sq = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[16 * WARP_SIZE * 9]())
    var sd = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[16 * WARP_SIZE]())

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x) * WPB + warp_id
    var active = n < N
    var nbk = K // 32
    var ones = Int32(0x01010101)
    var acc = InlineArray[Int32, 16](fill=Int32(0))
    var q8w = q8.bitcast[Int32]()
    var nthreads = WPB * WARP_SIZE
    var tile = 0
    while tile * WARP_SIZE < nbk:
        var tb0 = tile * WARP_SIZE
        # cooperative stage: activation words (coalesced: consecutive threads →
        # consecutive global words) and per-block scales for rows 0..Sdraft-1
        var widx = tid
        while widx < Sdraft * WARP_SIZE * 8:
            var r = widx // (WARP_SIZE * 8)
            var wr = widx % (WARP_SIZE * 8)
            var blk = wr // 8
            var w = wr % 8
            if tb0 + blk < nbk:
                sq[r * (WARP_SIZE * 9) + blk * 9 + w] = q8w[
                    r * nbk * 8 + (tb0 + blk) * 8 + w
                ]
            widx += nthreads
        var sidx = tid
        while sidx < Sdraft * WARP_SIZE:
            var r = sidx // WARP_SIZE
            var blk = sidx % WARP_SIZE
            if tb0 + blk < nbk:
                sd[r * WARP_SIZE + blk] = d8[r * nbk + tb0 + blk]
            sidx += nthreads
        barrier()

        var b = tb0 + lane
        if active and b < nbk:
            var gb = n * nbk + b
            var sbits = blob.bitcast[UInt16]()[gb]
            var d4 = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[
                DType.float32
            ]()[0]
            var nib_base = nbtot * 2 + gb * 16
            var nib16 = (blob + nib_base).bitcast[UInt16]()
            var a0 = _q4_u16_to_dp4a_word(nib16[0])
            var a1 = _q4_u16_to_dp4a_word(nib16[1])
            var a2 = _q4_u16_to_dp4a_word(nib16[2])
            var a3 = _q4_u16_to_dp4a_word(nib16[3])
            var a4 = _q4_u16_to_dp4a_word(nib16[4])
            var a5 = _q4_u16_to_dp4a_word(nib16[5])
            var a6 = _q4_u16_to_dp4a_word(nib16[6])
            var a7 = _q4_u16_to_dp4a_word(nib16[7])
            var sb = lane * 9
            var r = 0
            while r < Sdraft:
                var rb = r * (WARP_SIZE * 9) + sb
                acc[r] += _q4_fixed_term(d4, sd[r * WARP_SIZE + lane], _q8_dot8_vals(
                    sq[rb], sq[rb + 1], sq[rb + 2], sq[rb + 3],
                    sq[rb + 4], sq[rb + 5], sq[rb + 6], sq[rb + 7],
                    ones, a0, a1, a2, a3, a4, a5, a6, a7,
                ))
                r += 1
        barrier()
        tile += 1
    var off = WARP_SIZE // 2
    while off > 0:
        var r = 0
        while r < Sdraft:
            acc[r] += shuffle_xor(acc[r], UInt32(off))
            r += 1
        off //= 2
    if active and lane == 0:
        var r = 0
        while r < Sdraft:
            outp[r * N + n] = _q4_fixed_to_f32(acc[r])
            r += 1


def gpu_matmul_q4_s8_v4_gemv_dev[WPB: Int](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    d_scratch: UInt64,
    Kdim: Int,
    N: Int,
    S: Int,
) raises:
    var nbk = (Kdim + GROUP - 1) // GROUP
    var nbtot = (Kdim * N + GROUP - 1) // GROUP
    var d8_base = d_scratch
    var q8_base = d_scratch + UInt64(16 * nbk * 4)
    var qk = ctx.compile_function[quantize_q8_1_rows_kernel]()
    var blocks = (N + WPB - 1) // WPB
    var g0 = 0
    while g0 < S:
        var rows = 16 if (S - g0) >= 16 else (S - g0)
        ctx.enqueue_function(
            qk,
            _as_f32_ptr(d_in_fp32 + UInt64(g0 * Kdim * 4)),
            _as_f32_ptr(d8_base),
            _as_i8_ptr(q8_base),
            Int32(Kdim), Int32(nbk), Int32(rows),
            grid_dim=(rows * nbk + 255) // 256, block_dim=256,
        )
        if rows > MMQ_DRAFTS:
            var gk16 = ctx.compile_function[q4_gemv_dp4a_s16_actshare_v4_kernel[WPB]]()
            ctx.enqueue_function(
                gk16, _as_u8_ptr(d_w_q4), _as_f32_ptr(d8_base), _as_i8_ptr(q8_base),
                _as_f32_ptr(d_out_fp32 + UInt64(g0 * N * 4)), Int32(Kdim), Int32(N), Int32(nbtot), Int32(rows),
                grid_dim=blocks, block_dim=WPB * WARP_SIZE,
            )
        elif rows == 7:
            var gk7 = ctx.compile_function[q4_gemv_dp4a_s7_v4_kernel[WPB]]()
            ctx.enqueue_function(
                gk7, _as_u8_ptr(d_w_q4), _as_f32_ptr(d8_base), _as_i8_ptr(q8_base),
                _as_f32_ptr(d_out_fp32 + UInt64(g0 * N * 4)), Int32(Kdim), Int32(N), Int32(nbtot),
                grid_dim=blocks, block_dim=WPB * WARP_SIZE,
            )
        else:
            var gk = ctx.compile_function[q4_gemv_dp4a_s8_v4_kernel[WPB]]()
            ctx.enqueue_function(
                gk, _as_u8_ptr(d_w_q4), _as_f32_ptr(d8_base), _as_i8_ptr(q8_base),
                _as_f32_ptr(d_out_fp32 + UInt64(g0 * N * 4)), Int32(Kdim), Int32(N), Int32(nbtot), Int32(rows),
                grid_dim=blocks, block_dim=WPB * WARP_SIZE,
            )
        g0 += rows


# ═══════════════════════════════════════════════════════════════════════════════════
# MMQ: int8 TENSOR-CORE verify GEMM (M=8 drafts). The spec-decode verify path — same
# Q4×q8_1 math as the dp4a GEMV but the M=K batch runs on int8 tensor cores (mma.sync
# m16n8k32 s8), NOT the dp4a ALU. Validated lossless (argmax-exact vs dp4a M=1, 1e-7 fp32
# noise) in the validated Q4 MMQ int8 microbenchmark. Drafts fixed at 8
# (the mma n-dim); verify with K<8 quantizes K real rows, ignores out[K:].
# ═══════════════════════════════════════════════════════════════════════════════════
alias MMQ_DRAFTS = 8


# int8 m16n8k32 mma: D[16,8] s32 = A[16,32] s8 · B[8,32] s8 (C=0). stdlib _mma_nvidia has
# no int8 branch — hand-rolled off its fp8 e4m3 template (same shape, 4×A/2×B/4×C-D regs).
def _mma_s8(
    a: SIMD[DType.int32, 4], b: SIMD[DType.int32, 2], c: SIMD[DType.int32, 4]
) -> SIMD[DType.int32, 4]:
    var r = inlined_assembly[
        (
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0, $1, $2, $3},"
            " {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
        ),
        _RegisterPackType[Int32, Int32, Int32, Int32],
        constraints="=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        has_side_effect=False,
    ](a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


@always_inline("nodebug")
def _q4_s8_mma_word_shared[origin: MutOrigin, //](
    smem: UnsafePointer[
        Scalar[DType.uint8], origin, address_space=AddressSpace.SHARED
    ],
    base: Int,
    kbase: Int,
) -> Int32:
    var w = (smem + base + (kbase >> 1)).bitcast[UInt16]()[0]
    var bb0 = UInt8(w & 0xFF)
    var bb1 = UInt8((w >> 8) & 0xFF)
    var s0 = Int32(Int(bb0 & 0xF)) - 8
    var s1 = Int32(Int((bb0 >> 4) & 0xF)) - 8
    var s2 = Int32(Int(bb1 & 0xF)) - 8
    var s3 = Int32(Int((bb1 >> 4) & 0xF)) - 8
    return (s0 & 0xFF) | ((s1 & 0xFF) << 8) | ((s2 & 0xFF) << 16) | (
        (s3 & 0xFF) << 24
    )


# 1 warp → 16 weight-rows × 8 drafts; 1 mma per Q4 block. OFFICIAL PTX m16n8k32 fragment
# layout (gid=lane//4, tig=lane%4): A row=gid+8*(i&1) kbase=tig*4+16*(i>>1); B draft=gid
# kbase=tig*4+16*j; D row=gid+8*(i>>1) draft=tig*2+(i&1). Weight loaded as (nib-8) signed
# s8 → MMA int dot == dp4a's (sumi-8*sumq); per-block fp32 scale d4*d8.
def q4_mmq_s8_kernel[WARPS: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nbtot_arg: Int32,
    Sdraft_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nbtot = Int(nbtot_arg)
    var Sdraft = Int(Sdraft_arg)
    # d8 = [8, nbk] (draft scale @ draft*nbk+b); q8 = [8, nbk*32]; out = [Sdraft, N] (draft*N+n).
    # The mma always computes 8 draft rows; only the first Sdraft (<=8) are written (out is
    # sized [Sdraft, N] — rows >= Sdraft would be OOB; their inputs are stale scratch anyway).
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n_base = (Int(block_idx.x) * WARPS + warp_id) * 16
    if n_base >= N:
        return
    var nbk = K // 32
    var gid = lane // 4
    var tig = lane % 4
    var acc = InlineArray[Float32, 4](fill=Float32(0.0))
    var b = 0
    while b < nbk:
        var A = SIMD[DType.int32, 4](0)
        var i = 0
        while i < 4:
            var wr = gid + 8 * (i & 1)
            var kbase = tig * 4 + 16 * (i >> 1)
            var nib_base = nbtot * 2 + ((n_base + wr) * nbk + b) * 16 + (kbase >> 1)
            var bb0 = blob[nib_base]
            var bb1 = blob[nib_base + 1]
            var s0 = Int32(Int(bb0 & 0xF)) - 8
            var s1 = Int32(Int((bb0 >> 4) & 0xF)) - 8
            var s2 = Int32(Int(bb1 & 0xF)) - 8
            var s3 = Int32(Int((bb1 >> 4) & 0xF)) - 8
            A[i] = (s0 & 0xFF) | ((s1 & 0xFF) << 8) | ((s2 & 0xFF) << 16) | (
                (s3 & 0xFF) << 24
            )
            i += 1
        var B = SIMD[DType.int32, 2](0)
        var j = 0
        while j < 2:
            var kbase = tig * 4 + 16 * j
            var qb = gid * nbk * 32 + b * 32 + kbase
            B[j] = (Int32(Int(q8[qb])) & 0xFF) | ((Int32(Int(q8[qb + 1])) & 0xFF) << 8) | (
                (Int32(Int(q8[qb + 2])) & 0xFF) << 16
            ) | ((Int32(Int(q8[qb + 3])) & 0xFF) << 24)
            j += 1
        var D = _mma_s8(A, B, SIMD[DType.int32, 4](0))
        var ld = 0
        while ld < 4:
            var wr = gid + 8 * (ld >> 1)
            var draft = tig * 2 + (ld & 1)
            var gb = (n_base + wr) * nbk + b
            var d4 = bitcast[DType.float16, 1](
                SIMD[DType.uint16, 1](blob.bitcast[UInt16]()[gb])
            ).cast[DType.float32]()[0]
            acc[ld] += d4 * d8[draft * nbk + b] * Float32(D[ld])
            ld += 1
        b += 1
    var lw = 0
    while lw < 4:
        var wr = gid + 8 * (lw >> 1)
        var draft = tig * 2 + (lw & 1)
        if draft < Sdraft:
            outp[draft * N + (n_base + wr)] = acc[lw]
        lw += 1


def q4_mmq_s16_q8stage_verify_kernel[WARPS: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K_arg: Int32,
    N_arg: Int32,
    nbtot_arg: Int32,
    Sdraft_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nbtot = Int(nbtot_arg)
    var Sdraft = Int(Sdraft_arg)
    # Two-slot cp.async pipeline: tile 0 is made visible before the loop, then
    # tile n+1 is copied while tile n is consumed. The MMA inputs and Q14
    # epilogue are otherwise identical to the single-stage kernel.
    comptime STAGE_B = 8
    comptime PIPE_STAGES = 2
    var sw = raw_stack_allocation[
        PIPE_STAGES * WARPS * 16 * STAGE_B * 16,
        Scalar[DType.uint8],
        address_space=AddressSpace.SHARED,
    ]()
    var ss = raw_stack_allocation[
        PIPE_STAGES * WARPS * 16 * STAGE_B,
        Scalar[DType.uint16],
        address_space=AddressSpace.SHARED,
    ]()
    var sq = raw_stack_allocation[
        PIPE_STAGES * 16 * STAGE_B * 32,
        Scalar[DType.int8],
        address_space=AddressSpace.SHARED,
    ]()
    var sd = raw_stack_allocation[
        PIPE_STAGES * 16 * STAGE_B,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n_base = (Int(block_idx.x) * WARPS + warp_id) * 16
    if n_base >= N:
        return
    var nbk = K // 32
    var gid = lane // 4
    var tig = lane % 4
    var acc00 = Int32(0)
    var acc01 = Int32(0)
    var acc02 = Int32(0)
    var acc03 = Int32(0)
    var acc10 = Int32(0)
    var acc11 = Int32(0)
    var acc12 = Int32(0)
    var acc13 = Int32(0)
    var draft0 = tig * 2
    var draft1 = draft0 + 1
    var draft_hi0 = draft0 + MMQ_DRAFTS
    var draft_hi1 = draft1 + MMQ_DRAFTS
    var n0 = n_base + gid
    var n1 = n0 + 8
    var kb0 = tig * 4
    var kb1 = kb0 + 16
    var low_active = gid < Sdraft
    var high_active = gid + MMQ_DRAFTS < Sdraft
    var nthreads = WARPS * WARP_SIZE

    var w_stage_stride = WARPS * 16 * STAGE_B * 16
    var scale_stage_stride = WARPS * 16 * STAGE_B
    var q_stage_stride = 16 * STAGE_B * 32
    var d_stage_stride = 16 * STAGE_B

    var first_w_base = 0
    var first_scale_base = 0
    var first_q_base = 0
    var first_d_base = 0
    var first_wchunks = WARPS * 16 * STAGE_B
    var first_wi = tid
    while first_wi < first_wchunks:
        var first_wp = first_wi // (16 * STAGE_B)
        var first_remw = first_wi - first_wp * 16 * STAGE_B
        var first_col = first_remw // STAGE_B
        var first_lbw = first_remw - first_col * STAGE_B
        var first_n_w = (Int(block_idx.x) * WARPS + first_wp) * 16 + first_col
        if first_n_w < N and first_lbw < nbk:
            ss[first_scale_base + (first_wp * 16 + first_col) * STAGE_B + first_lbw] = blob.bitcast[UInt16]()[first_n_w * nbk + first_lbw]
            var first_src_w = nbtot * 2 + (first_n_w * nbk + first_lbw) * 16
            var first_dst_w = first_w_base + ((first_wp * 16 + first_col) * STAGE_B + first_lbw) * 16
            var first_gsw = UnsafePointer[
                UInt8, MutAnyOrigin, address_space=AddressSpace.GLOBAL
            ](unsafe_from_address=Int(blob + first_src_w))
            async_copy[16](first_gsw, sw + first_dst_w)
        first_wi += nthreads
    var first_chunks = Sdraft * STAGE_B * 2
    var first_ci = tid
    while first_ci < first_chunks:
        var first_row = first_ci // (STAGE_B * 2)
        var first_rem = first_ci - first_row * STAGE_B * 2
        var first_lb = first_rem // 2
        var first_half = first_rem - first_lb * 2
        if first_lb < nbk:
            var first_src_off = first_row * nbk * 32 + first_lb * 32 + first_half * 16
            var first_dst_off = first_q_base + (first_row * STAGE_B + first_lb) * 32 + first_half * 16
            var first_gsrc = UnsafePointer[
                Int8, MutAnyOrigin, address_space=AddressSpace.GLOBAL
            ](unsafe_from_address=Int(q8 + first_src_off))
            async_copy[16](first_gsrc, sq + first_dst_off)
        first_ci += nthreads
    var first_si = tid
    while first_si < Sdraft * STAGE_B:
        var first_row_s = first_si // STAGE_B
        var first_lb_s = first_si - first_row_s * STAGE_B
        if first_lb_s < nbk:
            sd[first_d_base + first_row_s * STAGE_B + first_lb_s] = d8[first_row_s * nbk + first_lb_s]
        first_si += nthreads
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    var tb = 0
    var slot = 0
    while tb < nbk:
        var next_tb = tb + STAGE_B
        var next_slot = 1 - slot
        var next_exists = next_tb < nbk
        if next_exists:
            var next_w_base = next_slot * w_stage_stride
            var next_scale_base = next_slot * scale_stage_stride
            var next_q_base = next_slot * q_stage_stride
            var next_d_base = next_slot * d_stage_stride
            var next_wchunks = WARPS * 16 * STAGE_B
            var next_wi = tid
            while next_wi < next_wchunks:
                var next_wp = next_wi // (16 * STAGE_B)
                var next_remw = next_wi - next_wp * 16 * STAGE_B
                var next_col = next_remw // STAGE_B
                var next_lbw = next_remw - next_col * STAGE_B
                var next_n_w = (Int(block_idx.x) * WARPS + next_wp) * 16 + next_col
                var next_b_w = next_tb + next_lbw
                if next_n_w < N and next_b_w < nbk:
                    ss[next_scale_base + (next_wp * 16 + next_col) * STAGE_B + next_lbw] = blob.bitcast[UInt16]()[next_n_w * nbk + next_b_w]
                    var next_src_w = nbtot * 2 + (next_n_w * nbk + next_b_w) * 16
                    var next_dst_w = next_w_base + ((next_wp * 16 + next_col) * STAGE_B + next_lbw) * 16
                    var next_gsw = UnsafePointer[
                        UInt8, MutAnyOrigin, address_space=AddressSpace.GLOBAL
                    ](unsafe_from_address=Int(blob + next_src_w))
                    async_copy[16](next_gsw, sw + next_dst_w)
                next_wi += nthreads
            var next_chunks = Sdraft * STAGE_B * 2
            var next_ci = tid
            while next_ci < next_chunks:
                var next_row = next_ci // (STAGE_B * 2)
                var next_rem = next_ci - next_row * STAGE_B * 2
                var next_lb = next_rem // 2
                var next_half = next_rem - next_lb * 2
                var next_b_abs = next_tb + next_lb
                if next_b_abs < nbk:
                    var next_src_off = next_row * nbk * 32 + next_b_abs * 32 + next_half * 16
                    var next_dst_off = next_q_base + (next_row * STAGE_B + next_lb) * 32 + next_half * 16
                    var next_gsrc = UnsafePointer[
                        Int8, MutAnyOrigin, address_space=AddressSpace.GLOBAL
                    ](unsafe_from_address=Int(q8 + next_src_off))
                    async_copy[16](next_gsrc, sq + next_dst_off)
                next_ci += nthreads
            var next_si = tid
            while next_si < Sdraft * STAGE_B:
                var next_row_s = next_si // STAGE_B
                var next_lb_s = next_si - next_row_s * STAGE_B
                var next_b_s = next_tb + next_lb_s
                if next_b_s < nbk:
                    sd[next_d_base + next_row_s * STAGE_B + next_lb_s] = d8[next_row_s * nbk + next_b_s]
                next_si += nthreads
            async_copy_commit_group()
        var cur_w_base = slot * w_stage_stride
        var cur_scale_base = slot * scale_stage_stride
        var cur_q_base = slot * q_stage_stride
        var cur_d_base = slot * d_stage_stride

        var lb2 = 0
        while lb2 < STAGE_B and tb + lb2 < nbk:
            var wbase0 = cur_w_base + ((warp_id * 16 + gid) * STAGE_B + lb2) * 16
            var wbase1 = cur_w_base + ((warp_id * 16 + gid + 8) * STAGE_B + lb2) * 16
            var A = SIMD[DType.int32, 4](
                _q4_s8_mma_word_shared(sw, wbase0, kb0),
                _q4_s8_mma_word_shared(sw, wbase1, kb0),
                _q4_s8_mma_word_shared(sw, wbase0, kb1),
                _q4_s8_mma_word_shared(sw, wbase1, kb1),
            )
            var B0 = SIMD[DType.int32, 2](0)
            if low_active:
                var qb0 = cur_q_base + (gid * STAGE_B + lb2) * 32
                B0 = SIMD[DType.int32, 2](
                    (sq + qb0 + kb0).bitcast[Int32]()[0],
                    (sq + qb0 + kb1).bitcast[Int32]()[0],
                )
            var B1 = SIMD[DType.int32, 2](0)
            if high_active:
                var qb1 = cur_q_base + ((gid + MMQ_DRAFTS) * STAGE_B + lb2) * 32
                B1 = SIMD[DType.int32, 2](
                    (sq + qb1 + kb0).bitcast[Int32]()[0],
                    (sq + qb1 + kb1).bitcast[Int32]()[0],
                )
            var D0 = _mma_s8(A, B0, SIMD[DType.int32, 4](0))
            var D1 = _mma_s8(A, B1, SIMD[DType.int32, 4](0))

            if n0 < N:
                var d40 = bitcast[DType.float16, 1](
                    SIMD[DType.uint16, 1](
                        ss[cur_scale_base + (warp_id * 16 + gid) * STAGE_B + lb2]
                    )
                ).cast[DType.float32]()[0]
                if draft0 < Sdraft:
                    acc00 += _q4_fixed_term(d40, sd[cur_d_base + draft0 * STAGE_B + lb2], D0[0])
                if draft1 < Sdraft:
                    acc01 += _q4_fixed_term(d40, sd[cur_d_base + draft1 * STAGE_B + lb2], D0[1])
                if draft_hi0 < Sdraft:
                    acc10 += _q4_fixed_term(d40, sd[cur_d_base + draft_hi0 * STAGE_B + lb2], D1[0])
                if draft_hi1 < Sdraft:
                    acc11 += _q4_fixed_term(d40, sd[cur_d_base + draft_hi1 * STAGE_B + lb2], D1[1])
            if n1 < N:
                var d41 = bitcast[DType.float16, 1](
                    SIMD[DType.uint16, 1](
                        ss[cur_scale_base + (warp_id * 16 + gid + 8) * STAGE_B + lb2]
                    )
                ).cast[DType.float32]()[0]
                if draft0 < Sdraft:
                    acc02 += _q4_fixed_term(d41, sd[cur_d_base + draft0 * STAGE_B + lb2], D0[2])
                if draft1 < Sdraft:
                    acc03 += _q4_fixed_term(d41, sd[cur_d_base + draft1 * STAGE_B + lb2], D0[3])
                if draft_hi0 < Sdraft:
                    acc12 += _q4_fixed_term(d41, sd[cur_d_base + draft_hi0 * STAGE_B + lb2], D1[2])
                if draft_hi1 < Sdraft:
                    acc13 += _q4_fixed_term(d41, sd[cur_d_base + draft_hi1 * STAGE_B + lb2], D1[3])
            lb2 += 1
        if next_exists:
            async_copy_wait_all()
            barrier()
        tb = next_tb
        slot = next_slot

    if n0 < N:
        if draft0 < Sdraft:
            outp[draft0 * N + n0] = _q4_fixed_to_f32(acc00)
        if draft1 < Sdraft:
            outp[draft1 * N + n0] = _q4_fixed_to_f32(acc01)
        if draft_hi0 < Sdraft:
            outp[draft_hi0 * N + n0] = _q4_fixed_to_f32(acc10)
        if draft_hi1 < Sdraft:
            outp[draft_hi1 * N + n0] = _q4_fixed_to_f32(acc11)
    if n1 < N:
        if draft0 < Sdraft:
            outp[draft0 * N + n1] = _q4_fixed_to_f32(acc02)
        if draft1 < Sdraft:
            outp[draft1 * N + n1] = _q4_fixed_to_f32(acc03)
        if draft_hi0 < Sdraft:
            outp[draft_hi0 * N + n1] = _q4_fixed_to_f32(acc12)
        if draft_hi1 < Sdraft:
            outp[draft_hi1 * N + n1] = _q4_fixed_to_f32(acc13)


# ── MMQ batched GEMM dev: out[S,N] = Q4 weight[N,Kdim] · Q8(in[S,Kdim]), int8 tensor-core,
#    law-compliant (Q4 wt × Q8 acts → int32 → fp32). Tiles S into groups of MMQ_DRAFTS(=8)
#    rows — the mma n-dim is 8, so a group of 8 rows is one mma sweep; ceil(S/8) groups cover
#    any S (verify S<=8 = 1 group; prefill S>8 = multiple). Groups serialize on the ctx stream
#    (group g's mma finishes before g+1's quantize overwrites the shared q8 scratch). d_scratch
#    holds [8*nbk f32 d8][8*nbk*32 i8 q8] (one group). NOTE: re-reads the Q4 weight per group —
#    correct + law-compliant; the weight-once M-tiled int8 GEMM is the later MMQ-pipeline opt. ──
def gpu_matmul_q4_mmq_dev[WARPS: Int](
    ctx: DeviceContext,
    d_out_fp32: UInt64,  # [S, N]
    d_in_fp32: UInt64,  # [S, Kdim] — S rows, contiguous
    d_w_q4: UInt64,
    d_scratch: UInt64,  # [8*nbk f32 d8][8*nbk*32 i8 q8] (one group)
    Kdim: Int,
    N: Int,
    S: Int,  # rows (drafts for verify, sequence for prefill)
) raises:
    var nbk = (Kdim + GROUP - 1) // GROUP
    var nbtot = (Kdim * N + GROUP - 1) // GROUP
    var d8_base = d_scratch
    var q8_base = d_scratch + UInt64(MMQ_DRAFTS * nbk * 4)
    var qk = ctx.compile_function[quantize_q8_1_kernel]()
    var mk = ctx.compile_function[q4_mmq_s8_kernel[WARPS]]()
    var qblocks = (nbk + 255) // 256
    var blocks = (N + 16 * WARPS - 1) // (16 * WARPS)
    var g0 = 0
    while g0 < S:
        var gs = MMQ_DRAFTS if (S - g0) >= MMQ_DRAFTS else (S - g0)  # rows in this group (<=8)
        for m in range(gs):
            ctx.enqueue_function(
                qk,
                _as_f32_ptr(d_in_fp32 + UInt64((g0 + m) * Kdim * 4)),
                _as_f32_ptr(d8_base + UInt64(m * nbk * 4)),
                _as_i8_ptr(q8_base + UInt64(m * nbk * 32)),
                Int32(Kdim), Int32(nbk),
                grid_dim=qblocks, block_dim=256,
            )
        ctx.enqueue_function(
            mk, _as_u8_ptr(d_w_q4), _as_f32_ptr(d8_base), _as_i8_ptr(q8_base),
            _as_f32_ptr(d_out_fp32 + UInt64(g0 * N * 4)), Int32(Kdim), Int32(N), Int32(nbtot), Int32(gs),
            grid_dim=blocks, block_dim=WARPS * WARP_SIZE,
        )
        g0 += MMQ_DRAFTS


def gpu_matmul_q4_mmq_relaxed_verify_dev[WARPS: Int](
    ctx: DeviceContext,
    d_out_fp32: UInt64,  # [S, N]
    d_in_fp32: UInt64,  # [S, Kdim]
    d_w_q4: UInt64,
    d_scratch: UInt64,  # [16*nbk f32 d8][16*nbk*32 i8 q8], same as v4 verify
    Kdim: Int,
    N: Int,
    S: Int,
) raises:
    if S > 16:
        # The current MMQ body is 16-row native. S=17 is faster on the v4
        # actshare route than MMQ-16 plus a second one-row pass.
        gpu_matmul_q4_s8_v4_gemv_dev[8](
            ctx, d_out_fp32, d_in_fp32, d_w_q4, d_scratch, Kdim, N, S
        )
        return
    var nbk = (Kdim + GROUP - 1) // GROUP
    var nbtot = (Kdim * N + GROUP - 1) // GROUP
    var d8_base = d_scratch
    var q8_base = d_scratch + UInt64(16 * nbk * 4)
    var qk = ctx.compile_function[quantize_q8_1_rows_kernel]()
    var mk = ctx.compile_function[q4_mmq_s16_q8stage_verify_kernel[WARPS]]()
    var rows = 16 if S >= 16 else S
    ctx.enqueue_function(
        qk,
        _as_f32_ptr(d_in_fp32),
        _as_f32_ptr(d8_base),
        _as_i8_ptr(q8_base),
        Int32(Kdim), Int32(nbk), Int32(rows),
        grid_dim=(rows * nbk + 255) // 256, block_dim=256,
    )
    ctx.enqueue_function(
        mk, _as_u8_ptr(d_w_q4), _as_f32_ptr(d8_base), _as_i8_ptr(q8_base),
        _as_f32_ptr(d_out_fp32), Int32(Kdim), Int32(N), Int32(nbtot), Int32(rows),
        grid_dim=(N + 16 * WARPS - 1) // (16 * WARPS), block_dim=WARPS * WARP_SIZE,
    )
