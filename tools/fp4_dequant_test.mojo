"""Correctness test for lib/fp4_weights.dequant_nvfp4_kernel (FP4 Step 2b).
GPU NVFP4→bf16 dequant vs a host dequant with identical formulas → bit-exact (same
bf16 bits). Proves the GPU weight-dequant kernel matches the NVFP4 decode the encoder
(tools/quantize_nvfp4.py) and the MMA use. Run from repo root:
  pixi run mojo run -I . --target-accelerator sm_121a tools/fp4_dequant_test.mojo
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.random import random_ui64
from lib.fp4_weights import dequant_nvfp4_kernel, e4m3_decode, E2M1_MAG, FP4_GROUP


def main() raises:
    with DeviceContext() as ctx:
        var n = 4096                       # elements (multiple of 16)
        var nb = n // 16
        var gs = Float32(7.5e-5)           # a realistic per-tensor global scale
        var blob_bytes = nb + n // 2       # [e4m3 scales : nb][E2M1 packed : n/2]

        var blob_h = ctx.enqueue_create_host_buffer[DType.uint8](blob_bytes)
        for b in range(nb):                # SWEEP ALL e4m3 scale bytes 0x00-0x7F (full exp/mant, not 8 fixed)
            blob_h[b] = UInt8(b % 0x80)
        for b in range(nb, blob_bytes):    # random packed E2M1 nibbles
            blob_h[b] = UInt8(Int(random_ui64(0, 255)))

        var blob_d = ctx.enqueue_create_buffer[DType.uint8](blob_bytes)
        var out_d = ctx.enqueue_create_buffer[DType.bfloat16](n)
        var out_h = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
        ctx.enqueue_copy(blob_d, blob_h)

        var k = ctx.compile_function[dequant_nvfp4_kernel]()
        var threads = 256
        var blocks = (n + threads - 1) // threads
        ctx.enqueue_function(
            k, blob_d.unsafe_ptr(), out_d.unsafe_ptr(), gs, n, nb,
            grid_dim=blocks, block_dim=threads,
        )
        ctx.synchronize()
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()

        var bad = 0
        for i in range(n):
            var block = i // FP4_GROUP
            var bs_byte = Int(blob_h[block])
            var byte = Int(blob_h[nb + (i // 2)])
            var nib = (byte & 0x0F) if (i & 1) == 0 else ((byte >> 4) & 0x0F)
            var sign = Float32(-1.0) if (nib & 8) != 0 else Float32(1.0)
            var mag = E2M1_MAG[nib & 7]
            var w = SIMD[DType.float32, 1](gs * e4m3_decode(bs_byte) * sign * mag)
            var want = w.cast[DType.bfloat16]()[0]
            if out_h[i] != want:
                bad += 1
                if bad <= 10:
                    print("  MISMATCH i=", i, " scale_byte=", bs_byte, " exp=", (bs_byte >> 3) & 0xF,
                          " mant=", bs_byte & 0x7, " host=", Float32(want), " gpu=", Float32(out_h[i]))
        print("NVFP4 dequant GPU-vs-host: bad=", bad, "/", n)
        if bad == 0:
            print("DEQUANT NVFP4 BIT-EXACT — lib/fp4_weights OK")
        else:
            print("FAIL")
        _ = blob_d^
        _ = out_d^
