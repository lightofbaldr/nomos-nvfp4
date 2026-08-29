"""W4A4 GEMM composition test (FP4 Step 3b) — the speed path end-to-end.
Host-quant a weight to NVFP4 (per-tensor global, like the real model), then run
gpu_matmul_nvfp4_w4a4_dev (GPU-quant activation -> native FP4 MMA -> post-scale) and
compare to the fp32 reference act@weight^T. Both operands FP4, so RMSE lands in the FP4
band (~10-20%). Run: pixi run mojo run -I . --target-accelerator sm_121a tools/fp4_w4a4_test.mojo
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.random import random_ui64
from std.math import sqrt
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_dev, _f32_to_e4m3_byte, _e2m1_nibble
from lib.fp4_weights import e4m3_decode


def _u64(p: UnsafePointer[Float32, MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def _u64u8(p: UnsafePointer[UInt8, MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def main() raises:
    with DeviceContext() as ctx:
        var M = 1                       # DECODE case (M=1, Mpad=16) — the live thread-1 path
        var Mpad = 16
        var K = 256
        var N = 262144
        var KB = K // 2
        var NB = K // 16
        var nb_w = (N * K) // 16
        var n_w = N * K

        # random fp32 activation [M, K] and weight [N, K]
        var act_h = ctx.enqueue_create_host_buffer[DType.float32](M * K)
        for i in range(M * K):
            act_h[i] = Float32(Int(random_ui64(0, 2000))) / 200.0 - 5.0   # ~U[-5,5)
        var wf = ctx.enqueue_create_host_buffer[DType.float32](N * K)
        for i in range(N * K):
            wf[i] = Float32(Int(random_ui64(0, 2000))) / 1000.0 - 1.0     # ~U[-1,1)

        # weight -> NVFP4 (per-tensor global), blob = [e4m3 scales : nb_w][E2M1 packed]
        var wamax = Float32(0.0)
        for i in range(N * K):
            var a = abs(wf[i])
            if a > wamax:
                wamax = a
        var wglobal = wamax / Float32(6.0 * 448.0)
        if wglobal == Float32(0.0):
            wglobal = Float32(1.0)
        var wblob = ctx.enqueue_create_host_buffer[DType.uint8](nb_w + n_w // 2)
        for b in range(nb_w):
            var bamax = Float32(0.0)
            for e in range(16):
                var a = abs(wf[b * 16 + e])
                if a > bamax:
                    bamax = a
            var bsb = _f32_to_e4m3_byte(bamax / (Float32(6.0) * wglobal))
            wblob[b] = UInt8(bsb)
            var denom = wglobal * e4m3_decode(bsb)
            if denom == Float32(0.0):
                denom = Float32(1.0)
            for j in range(8):
                var n0 = _e2m1_nibble(wf[b * 16 + 2 * j] / denom)
                var n1 = _e2m1_nibble(wf[b * 16 + 2 * j + 1] / denom)
                wblob[nb_w + b * 8 + j] = UInt8((n1 << 4) | n0)

        var act_d = ctx.enqueue_create_buffer[DType.float32](M * K)
        var wblob_d = ctx.enqueue_create_buffer[DType.uint8](nb_w + n_w // 2)
        var apack_d = ctx.enqueue_create_buffer[DType.uint8](Mpad * KB)
        var abs_d = ctx.enqueue_create_buffer[DType.uint8](Mpad * NB)
        var aglob_d = ctx.enqueue_create_buffer[DType.float32](Mpad)
        var cpad_d = ctx.enqueue_create_buffer[DType.float32](Mpad * N)
        var out_d = ctx.enqueue_create_buffer[DType.float32](M * N)
        var out_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        ctx.enqueue_copy(act_d, act_h)
        ctx.enqueue_copy(wblob_d, wblob)

        gpu_matmul_nvfp4_w4a4_dev(
            ctx, UInt64(0),
            _u64(out_d.unsafe_ptr()), _u64(act_d.unsafe_ptr()), _u64u8(wblob_d.unsafe_ptr()),
            wglobal, Float32(0.0),
            _u64u8(apack_d.unsafe_ptr()), _u64u8(abs_d.unsafe_ptr()), _u64(aglob_d.unsafe_ptr()),
            UInt64(0), UInt64(0),        # d_act_bs_sf, d_w_bs_sf = 0 on GB10 (sm_120 path, no SF-atom scatter)
            _u64(cpad_d.unsafe_ptr()),
            M, Mpad, K, N,               # Mpad serves as cap (chunk capacity)
        )
        ctx.synchronize()
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()

        # fp32 reference + RMSE
        var sq = Float64(0.0)
        var sn = Float64(0.0)
        for m in range(M):
            for n in range(N):
                var acc = Float32(0.0)
                for k in range(K):
                    acc += act_h[m * K + k] * wf[n * K + k]
                var d = Float64(out_h[m * N + n] - acc)
                sq += d * d
                sn += Float64(acc) * Float64(acc)
        var rel = sqrt(sq / sn)
        print("W4A4 GEMM vs fp32: RMSE/||C|| =", rel, " out[0..3]=",
              out_h[0], out_h[1], out_h[2])
        if rel < 0.30:
            print("W4A4 GEMM OK — native FP4 MMA path correct (FP4 band)")
        else:
            print("FAIL: W4A4 error too large")
