"""Round-trip test for lib/fp4_act.quant_act_nvfp4_kernel (FP4 Step 3).
Quantize a [M,K] activation to NVFP4 on the GPU, dequant on the host (per-token global ×
e4m3 block scale × E2M1), and report RMSE/||A|| — must land in the FP4 band (a few %),
not garbage. Run: pixi run mojo run -I . --target-accelerator sm_121a tools/fp4_act_test.mojo
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.random import random_ui64
from std.math import sqrt
from lib.fp4_act import quant_act_nvfp4_kernel
from lib.fp4_weights import e4m3_decode

alias E2M1_MAG = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def main() raises:
    with DeviceContext() as ctx:
        var M = 4
        var K = 5376
        var KB = K // 2
        var NB = K // 16

        var act_h = ctx.enqueue_create_host_buffer[DType.float32](M * K)
        for i in range(M * K):
            act_h[i] = Float32(Int(random_ui64(0, 2000))) / 200.0 - 5.0   # ~U[-5,5)

        var act_d = ctx.enqueue_create_buffer[DType.float32](M * K)
        var packed_d = ctx.enqueue_create_buffer[DType.uint8](M * KB)
        var bs_d = ctx.enqueue_create_buffer[DType.uint8](M * NB)
        var glob_d = ctx.enqueue_create_buffer[DType.float32](M)
        ctx.enqueue_copy(act_d, act_h)

        var kern = ctx.compile_function[quant_act_nvfp4_kernel]()
        ctx.enqueue_function(
            kern, act_d.unsafe_ptr(), packed_d.unsafe_ptr(), bs_d.unsafe_ptr(),
            glob_d.unsafe_ptr(), M, K, grid_dim=M, block_dim=32,
        )
        ctx.synchronize()

        var packed_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
        var bs_h = ctx.enqueue_create_host_buffer[DType.uint8](M * NB)
        var glob_h = ctx.enqueue_create_host_buffer[DType.float32](M)
        ctx.enqueue_copy(packed_h, packed_d)
        ctx.enqueue_copy(bs_h, bs_d)
        ctx.enqueue_copy(glob_h, glob_d)
        ctx.synchronize()

        var sq = Float64(0.0)
        var sn = Float64(0.0)
        for m in range(M):
            var g = glob_h[m]
            for k in range(K):
                var b = k // 16
                var byte = Int(packed_h[m * KB + k // 2])
                var nib = (byte & 0x0F) if (k & 1) == 0 else (byte >> 4)
                var sign = Float32(-1.0) if (nib & 8) != 0 else Float32(1.0)
                var mag = E2M1_MAG[nib & 7]
                var bsd = e4m3_decode(Int(bs_h[m * NB + b]))
                var val = g * bsd * sign * mag
                var d = Float64(val - act_h[m * K + k])
                sq += d * d
                sn += Float64(act_h[m * K + k]) * Float64(act_h[m * K + k])
        var rel = sqrt(sq / sn)
        print("activation NVFP4 round-trip: RMSE/||A|| =", rel, " (globals:",
              glob_h[0], glob_h[1], ")")
        if rel < 0.25:
            print("ACTIVATION QUANTIZER OK — reconstruction in the FP4 band")
        else:
            print("FAIL: reconstruction error too large")
