"""fp4_gemv_v3_parity.mojo — verify v3 (vectorized arithmetic decode) is BIT-IDENTICAL
to v1 (nvfp4_gemv_kernel) on synthetic NVFP4 blobs. Greedy-lossless gate. Run on GPU
via gpurun. Exits 0 if max|v3-v1| == 0 across all shapes; nonzero otherwise.

Compile+run:
  pixi run mojo build tools/fp4_gemv_v3_parity.mojo -I . --target-accelerator sm_121a \
      -o build/v3_parity -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib \
      -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
  gpurun ./build/v3_parity
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List

from lib.cuda import cuda_malloc, cuda_upload, cuda_upload_u8, cuda_sync, cuda_free, cuda_memcpy
from lib.fp4_weights import gpu_matmul_nvfp4_fused_dev
from lib.fp4_gemv_v2 import gpu_matmul_nvfp4_fused_v3_dev


# Varied deterministic activation + weight pattern (NOT all-ones, to exercise all E2M1 levels).
def build_act(K: Int) -> List[Float32]:
    var a = List[Float32]()
    var i = 0
    while i < K:
        # mix of small + larger values to hit every E2M1 magnitude bucket
        a.append(Float32(0.7) * Float32((i % 13) - 6))
        i += 1
    return a^


def build_blob(K: Int, N: Int) -> List[UInt8]:
    var n = K * N
    var nb = n // 16
    var blob = List[UInt8]()
    # e4m3 scales: varied, nonzero (exp 8..15 -> 1.0..448)
    var i = 0
    while i < nb:
        blob.append(UInt8(0x40 + ((i * 5) & 0x3F)))
        i += 1
    # E2M1 packed: walk so every nibble value 0..15 appears many times
    var j = 0
    while j < n // 2:
        var lo = UInt8((j * 3) & 0xF)
        var hi = UInt8((j * 5 + 1) & 0xF)
        blob.append(UInt8(lo | (hi << 4)))
        j += 1
    return blob^


def main() raises:
    var shapes = List[Tuple[Int, Int]]()
    shapes.append(Tuple(5376, 4096))     # D_Dkv (small/fast; full coverage suffices)
    shapes.append(Tuple(5376, 8192))     # D_Dq
    shapes.append(Tuple(21504, 5376))    # FF_D (exercises K=21504)

    var any_bad = False
    with DeviceContext() as ctx:
        for idx in range(len(shapes)):
            var K = shapes[idx][0]
            var N = shapes[idx][1]
            var blob = build_blob(K, N)
            var act = build_act(K)
            var d_w = cuda_malloc(len(blob))
            cuda_upload_u8(d_w, blob)
            var d_in = cuda_malloc(K * 4)
            cuda_upload(d_in, act)
            var d_v1 = cuda_malloc(N * 4)
            var d_v3 = cuda_malloc(N * 4)
            var gs = Float32(0.0078125)   # arbitrary nonzero global scale

            for n in range(N):
                # zero outputs
                pass
            # zero d_v1, d_v3 via a host zero buffer
            var z = List[Float32]()
            for n in range(N):
                z.append(Float32(0.0))
            cuda_memcpy(d_v1, UInt64(Int(z.unsafe_ptr())), N * 4, 1)
            cuda_memcpy(d_v3, UInt64(Int(z.unsafe_ptr())), N * 4, 1)

            gpu_matmul_nvfp4_fused_dev(ctx, d_v1, d_in, d_w, gs, K, N)
            gpu_matmul_nvfp4_fused_v3_dev(ctx, d_v3, d_in, d_w, gs, K, N)
            cuda_sync()

            var h_v1 = List[Float32](length=N, fill=Float32(0.0))
            var h_v3 = List[Float32](length=N, fill=Float32(0.0))
            cuda_memcpy(UInt64(Int(h_v1.unsafe_ptr())), d_v1, N * 4, 2)
            cuda_memcpy(UInt64(Int(h_v3.unsafe_ptr())), d_v3, N * 4, 2)
            cuda_sync()

            var max_abs = Float32(0.0)
            var max_i = -1
            for n in range(N):
                var d = h_v1[n] - h_v3[n]
                if d < Float32(0.0):
                    d = -d
                if d > max_abs:
                    max_abs = d
                    max_i = n
            # also report a sample of values to sanity-check the decode
            var sample_v1 = Float32(0.0)
            var sample_v3 = Float32(0.0)
            for n in range(min(N, 64)):
                if n == 7:
                    sample_v1 = h_v1[n]
                    sample_v3 = h_v3[n]
            print("shape K=", K, " N=", N, " max|v1-v3|=", max_abs, " at idx=", max_i,
                  " sample[7] v1=", sample_v1, " v3=", sample_v3, flush=True)
            if max_abs != Float32(0.0):
                any_bad = True

            cuda_free(d_w)
            cuda_free(d_in)
            cuda_free(d_v1)
            cuda_free(d_v3)

    if any_bad:
        print("\nPARITY FAIL: v3 != v1 (nonzero diff). DO NOT ship.", flush=True)
        raise Error("v3 parity failure vs v1")
    print("\nPARITY OK: v3 bit-identical to v1 on all tested shapes.", flush=True)
