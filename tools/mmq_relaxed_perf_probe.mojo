"""Synthetic timing probe for relaxed verify MMQ.

No model weights. Builds a gate-shaped Q4 blob and S=16 activation block, then
times the verify projection routes that matter after the ship gate was relaxed.

Build:
  pixi run mojo build tools/mmq_relaxed_perf_probe.mojo -I . \
      --target-accelerator sm_121a -o build/mmq_relaxed_perf_probe \
      -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib -Xlinker -lcudart -Xlinker -lm

Run:
  gpurun ./build/mmq_relaxed_perf_probe
"""

from std.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from std.memory import UnsafePointer
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.time import perf_counter_ns
from std.collections import InlineArray

from lib.q4_gemv_dp4a import (
    MMQ_DRAFTS,
    gpu_matmul_q4_mmq_dev,
    gpu_matmul_q4_mmq_relaxed_verify_dev,
    gpu_matmul_q4_s8_v4_gemv_dev,
    quantize_q8_1_rows_kernel,
)
from lib.q4_weights import GROUP


comptime PROBE_S = 16
comptime PROBE_MAX_S = 17
comptime PROBE_K = 5376
comptime PROBE_N = 21504
comptime PROBE_NBK = PROBE_K // GROUP
comptime PROBE_NBTOT = PROBE_N * PROBE_NBK
comptime PROBE_BLOB_BYTES = PROBE_NBTOT * 2 + PROBE_NBTOT * 16
comptime PROBE_SCRATCH_BYTES = 16 * PROBE_NBK * 4 + 16 * PROBE_K
comptime PROBE_ITERS = 20
comptime PROBE_WARM = 5


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_i8_ptr(addr: UInt64) -> UnsafePointer[Int8, MutAnyOrigin]:
    return UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_u8_ptr(addr: UInt64) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(addr))


def _mma_s8_probe(
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


def mmq_const_s16_kernel[WARPS: Int](
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K: Int,
    N: Int,
    Sdraft: Int,
):
    var tid = Int(thread_idx.x)
    var warp_id = tid // 32
    var lane = tid % 32
    var gid = lane // 4
    var tig = lane % 4
    var n_base = (Int(block_idx.x) * WARPS + warp_id) * 16
    if n_base >= N:
        return
    var nbk = K // 32
    var A = SIMD[DType.int32, 4](0x01010101)
    var B = SIMD[DType.int32, 2](0x01010101)
    var acc0 = InlineArray[Float32, 4](fill=Float32(0.0))
    var acc1 = InlineArray[Float32, 4](fill=Float32(0.0))
    var b = 0
    while b < nbk:
        var D0 = _mma_s8_probe(A, B, SIMD[DType.int32, 4](0))
        var D1 = _mma_s8_probe(A, B, SIMD[DType.int32, 4](0))
        var ld = 0
        while ld < 4:
            acc0[ld] += Float32(D0[ld])
            acc1[ld] += Float32(D1[ld])
            ld += 1
        b += 1
    var lw = 0
    while lw < 4:
        var wr = gid + 8 * (lw >> 1)
        var draft = tig * 2 + (lw & 1)
        var n = n_base + wr
        if n < N:
            outp[draft * N + n] = acc0[lw]
            if draft + MMQ_DRAFTS < Sdraft:
                outp[(draft + MMQ_DRAFTS) * N + n] = acc1[lw]
        lw += 1


def mmq_i8w_s16_kernel[WARPS: Int](
    w8: UnsafePointer[Int8, MutAnyOrigin],
    d8: UnsafePointer[Float32, MutAnyOrigin],
    q8: UnsafePointer[Int8, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    K: Int,
    N: Int,
    Sdraft: Int,
):
    var tid = Int(thread_idx.x)
    var warp_id = tid // 32
    var lane = tid % 32
    var n_base = (Int(block_idx.x) * WARPS + warp_id) * 16
    if n_base >= N:
        return
    var nbk = K // 32
    var gid = lane // 4
    var tig = lane % 4
    var acc0 = InlineArray[Float32, 4](fill=Float32(0.0))
    var acc1 = InlineArray[Float32, 4](fill=Float32(0.0))

    var b = 0
    while b < nbk:
        var A = SIMD[DType.int32, 4](0)
        var i = 0
        while i < 4:
            var wr = gid + 8 * (i & 1)
            var n = n_base + wr
            var kbase = tig * 4 + 16 * (i >> 1)
            var wb = n * K + b * 32 + kbase
            A[i] = (Int32(Int(w8[wb])) & 0xFF) | (
                (Int32(Int(w8[wb + 1])) & 0xFF) << 8
            ) | ((Int32(Int(w8[wb + 2])) & 0xFF) << 16) | (
                (Int32(Int(w8[wb + 3])) & 0xFF) << 24
            )
            i += 1

        var B0 = SIMD[DType.int32, 2](0)
        var B1 = SIMD[DType.int32, 2](0)
        var j = 0
        while j < 2:
            var kbase2 = tig * 4 + 16 * j
            var qb0 = gid * nbk * 32 + b * 32 + kbase2
            B0[j] = (Int32(Int(q8[qb0])) & 0xFF) | (
                (Int32(Int(q8[qb0 + 1])) & 0xFF) << 8
            ) | ((Int32(Int(q8[qb0 + 2])) & 0xFF) << 16) | (
                (Int32(Int(q8[qb0 + 3])) & 0xFF) << 24
            )
            var draft_hi = gid + MMQ_DRAFTS
            if draft_hi < Sdraft:
                var qb1 = draft_hi * nbk * 32 + b * 32 + kbase2
                B1[j] = (Int32(Int(q8[qb1])) & 0xFF) | (
                    (Int32(Int(q8[qb1 + 1])) & 0xFF) << 8
                ) | ((Int32(Int(q8[qb1 + 2])) & 0xFF) << 16) | (
                    (Int32(Int(q8[qb1 + 3])) & 0xFF) << 24
                )
            j += 1

        var D0 = _mma_s8_probe(A, B0, SIMD[DType.int32, 4](0))
        var D1 = _mma_s8_probe(A, B1, SIMD[DType.int32, 4](0))
        var ld = 0
        while ld < 4:
            var draft0 = tig * 2 + (ld & 1)
            acc0[ld] += d8[draft0 * nbk + b] * Float32(D0[ld])
            if draft0 + MMQ_DRAFTS < Sdraft:
                acc1[ld] += d8[(draft0 + MMQ_DRAFTS) * nbk + b] * Float32(D1[ld])
            ld += 1
        b += 1

    var lw = 0
    while lw < 4:
        var wr2 = gid + 8 * (lw >> 1)
        var draft1 = tig * 2 + (lw & 1)
        var n2 = n_base + wr2
        outp[draft1 * N + n2] = acc0[lw]
        if draft1 + MMQ_DRAFTS < Sdraft:
            outp[(draft1 + MMQ_DRAFTS) * N + n2] = acc1[lw]
        lw += 1


def time_v4(ctx: DeviceContext, out_addr: UInt64, act: UInt64, blob: UInt64, scratch: UInt64) raises -> Float64:
    for _ in range(PROBE_WARM):
        gpu_matmul_q4_s8_v4_gemv_dev[8](ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, PROBE_S)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        gpu_matmul_q4_s8_v4_gemv_dev[8](ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, PROBE_S)
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def time_v4_rows(
    ctx: DeviceContext, out_addr: UInt64, act: UInt64, blob: UInt64, scratch: UInt64, rows: Int
) raises -> Float64:
    for _ in range(PROBE_WARM):
        gpu_matmul_q4_s8_v4_gemv_dev[8](ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, rows)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        gpu_matmul_q4_s8_v4_gemv_dev[8](ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, rows)
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def time_relaxed[WARPS: Int](
    ctx: DeviceContext, out_addr: UInt64, act: UInt64, blob: UInt64, scratch: UInt64
) raises -> Float64:
    for _ in range(PROBE_WARM):
        gpu_matmul_q4_mmq_relaxed_verify_dev[WARPS](
            ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, PROBE_S
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        gpu_matmul_q4_mmq_relaxed_verify_dev[WARPS](
            ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, PROBE_S
        )
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def time_relaxed_rows[WARPS: Int](
    ctx: DeviceContext, out_addr: UInt64, act: UInt64, blob: UInt64, scratch: UInt64, rows: Int
) raises -> Float64:
    for _ in range(PROBE_WARM):
        gpu_matmul_q4_mmq_relaxed_verify_dev[WARPS](
            ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, rows
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        gpu_matmul_q4_mmq_relaxed_verify_dev[WARPS](
            ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, rows
        )
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def print_crossover_row(
    ctx: DeviceContext, out_addr: UInt64, act: UInt64, blob: UInt64, scratch: UInt64, rows: Int
) raises:
    print(
        "rows=", rows,
        " v4=", time_v4_rows(ctx, out_addr, act, blob, scratch, rows),
        " w1=", time_relaxed_rows[1](ctx, out_addr, act, blob, scratch, rows),
        " w2=", time_relaxed_rows[2](ctx, out_addr, act, blob, scratch, rows),
        " w4=", time_relaxed_rows[4](ctx, out_addr, act, blob, scratch, rows),
        " w8=", time_relaxed_rows[8](ctx, out_addr, act, blob, scratch, rows),
        " ms",
    )


def time_old_mmq(ctx: DeviceContext, out_addr: UInt64, act: UInt64, blob: UInt64, scratch: UInt64) raises -> Float64:
    for _ in range(PROBE_WARM):
        gpu_matmul_q4_mmq_dev[4](ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, PROBE_S)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        gpu_matmul_q4_mmq_dev[4](ctx, out_addr, act, blob, scratch, PROBE_K, PROBE_N, PROBE_S)
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def quantize_rows(ctx: DeviceContext, act: UInt64, scratch: UInt64) raises:
    var qk = ctx.compile_function[quantize_q8_1_rows_kernel]()
    ctx.enqueue_function(
        qk,
        _as_f32_ptr(act),
        _as_f32_ptr(scratch),
        _as_i8_ptr(scratch + UInt64(16 * PROBE_NBK * 4)),
        PROBE_K,
        PROBE_NBK,
        PROBE_S,
        grid_dim=(PROBE_S * PROBE_NBK + 255) // 256,
        block_dim=256,
    )


def time_const[WARPS: Int](ctx: DeviceContext, out_addr: UInt64) raises -> Float64:
    var mk = ctx.compile_function[mmq_const_s16_kernel[WARPS]]()
    var blocks = (PROBE_N + 16 * WARPS - 1) // (16 * WARPS)
    for _ in range(PROBE_WARM):
        ctx.enqueue_function(
            mk, _as_f32_ptr(out_addr), PROBE_K, PROBE_N, PROBE_S,
            grid_dim=blocks, block_dim=WARPS * 32,
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        ctx.enqueue_function(
            mk, _as_f32_ptr(out_addr), PROBE_K, PROBE_N, PROBE_S,
            grid_dim=blocks, block_dim=WARPS * 32,
        )
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def time_i8w[WARPS: Int](ctx: DeviceContext, out_addr: UInt64, act: UInt64, w8: UInt64, scratch: UInt64) raises -> Float64:
    var mk = ctx.compile_function[mmq_i8w_s16_kernel[WARPS]]()
    var blocks = (PROBE_N + 16 * WARPS - 1) // (16 * WARPS)
    for _ in range(PROBE_WARM):
        quantize_rows(ctx, act, scratch)
        ctx.enqueue_function(
            mk,
            _as_i8_ptr(w8),
            _as_f32_ptr(scratch),
            _as_i8_ptr(scratch + UInt64(16 * PROBE_NBK * 4)),
            _as_f32_ptr(out_addr),
            PROBE_K,
            PROBE_N,
            PROBE_S,
            grid_dim=blocks,
            block_dim=WARPS * 32,
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(PROBE_ITERS):
        quantize_rows(ctx, act, scratch)
        ctx.enqueue_function(
            mk,
            _as_i8_ptr(w8),
            _as_f32_ptr(scratch),
            _as_i8_ptr(scratch + UInt64(16 * PROBE_NBK * 4)),
            _as_f32_ptr(out_addr),
            PROBE_K,
            PROBE_N,
            PROBE_S,
            grid_dim=blocks,
            block_dim=WARPS * 32,
        )
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(PROBE_ITERS) / 1.0e6


def main() raises:
    var ctx = DeviceContext()
    print("== relaxed MMQ synthetic perf probe ==")
    print("shape S=", PROBE_S, " K=", PROBE_K, " N=", PROBE_N)
    print("iters=", PROBE_ITERS, " warm=", PROBE_WARM)

    var act_h = ctx.enqueue_create_host_buffer[DType.float32](PROBE_MAX_S * PROBE_K)
    var blob_h = ctx.enqueue_create_host_buffer[DType.uint8](PROBE_BLOB_BYTES)
    var r = 0
    while r < PROBE_MAX_S:
        var k = 0
        while k < PROBE_K:
            var v = Float32(((r + 1) * (k % 29) + k * 3) % 61) - 30.0
            act_h[r * PROBE_K + k] = v * 0.03125
            k += 1
        r += 1
    var gb = 0
    while gb < PROBE_NBTOT:
        # fp16 1.0 scale, little-endian.
        blob_h[gb * 2] = UInt8(0x00)
        blob_h[gb * 2 + 1] = UInt8(0x3C)
        gb += 1
    var off = PROBE_NBTOT * 2
    var i = 0
    while i < PROBE_NBTOT * 16:
        var lo = UInt8((i * 3 + i // 17) & 0xF)
        var hi = UInt8((i * 5 + 7) & 0xF)
        blob_h[off + i] = UInt8(lo | (hi << 4))
        i += 1

    var act_d = ctx.enqueue_create_buffer[DType.float32](PROBE_MAX_S * PROBE_K)
    var blob_d = ctx.enqueue_create_buffer[DType.uint8](PROBE_BLOB_BYTES)
    var scratch_d = ctx.enqueue_create_buffer[DType.uint8](PROBE_SCRATCH_BYTES)
    var out_d = ctx.enqueue_create_buffer[DType.float32](PROBE_MAX_S * PROBE_N)
    var w8_d = ctx.enqueue_create_buffer[DType.int8](PROBE_N * PROBE_K)
    ctx.enqueue_copy(dst_buf=act_d, src_buf=act_h)
    ctx.enqueue_copy(dst_buf=blob_d, src_buf=blob_h)
    w8_d.enqueue_fill(Int8(1))
    out_d.enqueue_fill(Float32(0.0))
    scratch_d.enqueue_fill(UInt8(0))
    ctx.synchronize()

    var act = UInt64(Int(act_d.unsafe_ptr()))
    var blob = UInt64(Int(blob_d.unsafe_ptr()))
    var scratch = UInt64(Int(scratch_d.unsafe_ptr()))
    var out_addr = UInt64(Int(out_d.unsafe_ptr()))
    var w8 = UInt64(Int(w8_d.unsafe_ptr()))

    print("v4_actshare[8]        ", time_v4(ctx, out_addr, act, blob, scratch), " ms")
    print("old_mmq_2x_s8[4]      ", time_old_mmq(ctx, out_addr, act, blob, scratch), " ms")
    print("relaxed_q4 WARPS=1    ", time_relaxed[1](ctx, out_addr, act, blob, scratch), " ms")
    print("relaxed_q4 WARPS=2    ", time_relaxed[2](ctx, out_addr, act, blob, scratch), " ms")
    print("relaxed_q4 WARPS=4    ", time_relaxed[4](ctx, out_addr, act, blob, scratch), " ms")
    print("relaxed_q4 WARPS=8    ", time_relaxed[8](ctx, out_addr, act, blob, scratch), " ms")
    print("relaxed_wrapper S=17  ", time_relaxed_rows[2](ctx, out_addr, act, blob, scratch, 17), " ms")
    print("const_mma WARPS=4     ", time_const[4](ctx, out_addr), " ms")
    print("i8_weight WARPS=4     ", time_i8w[4](ctx, out_addr, act, w8, scratch), " ms")
    print("== row crossover ==")
    print_crossover_row(ctx, out_addr, act, blob, scratch, 9)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 10)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 11)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 12)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 13)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 14)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 15)
    print_crossover_row(ctx, out_addr, act, blob, scratch, 16)
