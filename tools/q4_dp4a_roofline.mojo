"""q4_dp4a_roofline.mojo — per-shape dp4a GEMV roofline + quant overhead (Lever 2 localization).

For each real decode shape: time gemv-ONLY (pre-quantized scratch, the inner-loop roofline =
the occupancy question) AND quant+gemv (the e2e per-shape). The DIFFERENCE is the per-GEMV q8
quant-launch overhead flagged. No nsys (CUDA trace is dead on GB10's iGPU) — self-timed.

  GB/s = weight_bytes(=len(blob)) * ITERS / dt.   ~273 GB/s is the GB10 HBM ceiling.
  If gate/up(D_FF) ~211 gemv-only but FF_D/D_Dkv lag -> occupancy (grid-stride/split-K).
  If all near-peak gemv-only but quant+gemv drags -> overhead (quant+gemv fusion).

  pixi run mojo build tools/q4_dp4a_roofline.mojo -I . --target-accelerator sm_121a \
      -o build/q4_roof -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib \
      -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
  gpurun ./build/q4_roof
"""
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer, bitcast
from std.collections import List
from std.time import perf_counter_ns

from lib.cuda import cuda_malloc, cuda_upload, cuda_upload_u8, cuda_sync, cuda_free, cuda_memcpy
from lib.q4_weights import GROUP, _as_f32_ptr
from lib.q4_gemv_dp4a import (
    gpu_q8_quantize_dev, gpu_matmul_q4_dp4a_gemv_dev, gpu_matmul_q4_dp4a_v4_gemv_dev,
)


# Pure read-only streaming kernel: grid-stride sum of a big array, write 1 partial/thread.
# acc is stored (can't be elided) so every read is issued; 256MB >> L2 => all reads hit HBM.
def read_stream_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    n_elem: Int,
    n_threads: Int,
):
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = gid
    while i < n_elem:
        acc += src[i]
        i += n_threads
    outp[gid] = acc


# float4-vectorized read: 16B/load (matches the gemv's 16B/lane granularity) — the true
# compute-kernel read ceiling (scalar loads can be instruction-bound below the BW ceiling).
def read_stream_v4_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    n_elem: Int,
    n_threads: Int,
):
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = SIMD[DType.float32, 4](0.0)
    var stride = n_threads * 4
    var i = gid * 4
    while i + 3 < n_elem:
        acc += src.load[width=4](i)
        i += stride
    outp[gid] = acc.reduce_add()


def build_blob(K: Int, N: Int) -> List[UInt8]:
    var n = K * N
    var nb = (n + GROUP - 1) // GROUP
    var blob = List[UInt8]()
    var i = 0
    while i < nb * 2:
        blob.append(UInt8(0x40 + ((i * 5) & 0x3F)))
        i += 1
    var j = 0
    while j < nb * 16:
        var lo = UInt8((j * 3) & 0xF)
        var hi = UInt8((j * 5 + 1) & 0xF)
        blob.append(UInt8(lo | (hi << 4)))
        j += 1
    return blob^


def build_act(K: Int) -> List[Float32]:
    var a = List[Float32]()
    var i = 0
    while i < K:
        a.append(Float32(0.7) * Float32((i % 13) - 6))
        i += 1
    return a^


def bench_shape(ctx: DeviceContext, label: String, K: Int, N: Int) raises:
    var ITERS = 200
    var WARM = 20
    var blob = build_blob(K, N)
    var wbytes = len(blob)
    var d_w = cuda_malloc(wbytes)
    cuda_upload_u8(d_w, blob)
    var act = build_act(K)
    var d_in = cuda_malloc(K * 4)
    cuda_upload(d_in, act)
    var nbk = (K + GROUP - 1) // GROUP
    var d_scratch = cuda_malloc(nbk * 36 + 64)   # [nbk f32 d8][nbk*32 i8 q8]
    var d_out = cuda_malloc(N * 4)

    # pre-quantize once for the gemv-only roofline
    gpu_q8_quantize_dev(ctx, d_in, d_scratch, K)
    cuda_sync()

    # warmup
    for _ in range(WARM):
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_out, d_scratch, d_w, K, N)
    cuda_sync()

    # (1) gemv-only roofline
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_out, d_scratch, d_w, K, N)
    cuda_sync()
    var t1 = perf_counter_ns()
    var dt_g = Float64(t1 - t0) / 1.0e9
    var gbs_g = Float64(wbytes) * Float64(ITERS) / dt_g / 1.0e9
    var us_g = dt_g / Float64(ITERS) * 1.0e6

    # capture the ORIGINAL kernel output for byte-parity, then run v4
    var h_orig = List[Float32](length=N, fill=Float32(0.0))
    cuda_memcpy(UInt64(Int(h_orig.unsafe_ptr())), d_out, N * 4, 2)
    cuda_sync()
    for _ in range(WARM):
        gpu_matmul_q4_dp4a_v4_gemv_dev[4](ctx, d_out, d_scratch, d_w, K, N)
    cuda_sync()
    var tv0 = perf_counter_ns()
    for _ in range(ITERS):
        gpu_matmul_q4_dp4a_v4_gemv_dev[4](ctx, d_out, d_scratch, d_w, K, N)
    cuda_sync()
    var tv1 = perf_counter_ns()
    var dt_v = Float64(tv1 - tv0) / 1.0e9
    var gbs_v = Float64(wbytes) * Float64(ITERS) / dt_v / 1.0e9
    var us_v = dt_v / Float64(ITERS) * 1.0e6
    var h_v4 = List[Float32](length=N, fill=Float32(0.0))
    cuda_memcpy(UInt64(Int(h_v4.unsafe_ptr())), d_out, N * 4, 2)
    cuda_sync()
    var nmis = 0   # true bit-compare (Inf==Inf matches; only genuine bit-diffs count)
    for i in range(N):
        var ob = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](h_orig[i]))[0]
        var vb = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](h_v4[i]))[0]
        if ob != vb:
            nmis += 1

    # (2) quant + gemv (e2e per-shape)
    cuda_sync()
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        gpu_q8_quantize_dev(ctx, d_in, d_scratch, K)
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_out, d_scratch, d_w, K, N)
    cuda_sync()
    var t3 = perf_counter_ns()
    var dt_qg = Float64(t3 - t2) / 1.0e9
    var gbs_qg = Float64(wbytes) * Float64(ITERS) / dt_qg / 1.0e9
    var us_qg = dt_qg / Float64(ITERS) * 1.0e6
    var quant_us = us_qg - us_g
    var quant_pct = (us_qg - us_g) / us_qg * 100.0

    var parity = String("BYTE-IDENTICAL") if nmis == 0 else String("MISMATCH n=", nmis)
    var speedup = us_g / us_v
    print(label, " K=", K, " N=", N, " wMB=", wbytes // (1024 * 1024), flush=True)
    print("   gemv base:  ", us_g, " us  ", gbs_g, " GB/s   (roofline)", flush=True)
    print("   gemv v4:    ", us_v, " us  ", gbs_v, " GB/s   ", speedup, "x  parity=", parity, flush=True)
    print("   quant+gemv: ", us_qg, " us  ", gbs_qg, " GB/s   quant=", quant_us, " us (", quant_pct, "% of e2e)", flush=True)

    cuda_free(d_w)
    cuda_free(d_in)
    cuda_free(d_scratch)
    cuda_free(d_out)


def read_bw_probe(ctx: DeviceContext) raises:
    # PURE read-only streaming BW — the correct ceiling for a ~99.9%-read GEMV (the gate).
    var n_elem = 64 * 1024 * 1024          # 64M floats = 256 MB >> L2 (cache-busted)
    var nbytes = n_elem * 4
    var BLOCKS = 1024
    var TPB = 256
    var n_threads = BLOCKS * TPB
    var d_src = cuda_malloc(nbytes)
    var d_out = cuda_malloc(n_threads * 4)
    var kern = ctx.compile_function[read_stream_kernel]()
    for _ in range(5):
        ctx.enqueue_function(kern, _as_f32_ptr(d_src), _as_f32_ptr(d_out), n_elem, n_threads,
                             grid_dim=BLOCKS, block_dim=TPB)
    cuda_sync()
    var IT = 50
    var t0 = perf_counter_ns()
    for _ in range(IT):
        ctx.enqueue_function(kern, _as_f32_ptr(d_src), _as_f32_ptr(d_out), n_elem, n_threads,
                             grid_dim=BLOCKS, block_dim=TPB)
    cuda_sync()
    var t1 = perf_counter_ns()
    var dt = Float64(t1 - t0) / 1.0e9
    var bw = Float64(nbytes) * Float64(IT) / dt / 1.0e9   # read-only: bytes READ per iter
    print("  read-only streaming BW (scalar): ", bw, " GB/s (pure reads, 256MB)", flush=True)

    # float4-vectorized read ceiling (16B/load, matches gemv granularity)
    var kern4 = ctx.compile_function[read_stream_v4_kernel]()
    for _ in range(5):
        ctx.enqueue_function(kern4, _as_f32_ptr(d_src), _as_f32_ptr(d_out), n_elem, n_threads,
                             grid_dim=BLOCKS, block_dim=TPB)
    cuda_sync()
    var t2 = perf_counter_ns()
    for _ in range(IT):
        ctx.enqueue_function(kern4, _as_f32_ptr(d_src), _as_f32_ptr(d_out), n_elem, n_threads,
                             grid_dim=BLOCKS, block_dim=TPB)
    cuda_sync()
    var t3 = perf_counter_ns()
    var dt4 = Float64(t3 - t2) / 1.0e9
    var bw4 = Float64(nbytes) * Float64(IT) / dt4 / 1.0e9
    print("  read-only streaming BW (float4): ", bw4, " GB/s <- correct GEMV ceiling", flush=True)
    cuda_free(d_src)
    cuda_free(d_out)


def bw_probe(ctx: DeviceContext) raises:
    # D2D memcpy = read+write; achievable BW ceiling reference (vs 273 theoretical).
    var nbytes = 256 * 1024 * 1024
    var d_a = cuda_malloc(nbytes)
    var d_b = cuda_malloc(nbytes)
    for _ in range(5):
        cuda_memcpy(d_b, d_a, nbytes, 3)
    cuda_sync()
    var IT = 50
    var t0 = perf_counter_ns()
    for _ in range(IT):
        cuda_memcpy(d_b, d_a, nbytes, 3)
    cuda_sync()
    var t1 = perf_counter_ns()
    var dt = Float64(t1 - t0) / 1.0e9
    var bw = Float64(nbytes) * 2.0 * Float64(IT) / dt / 1.0e9   # 2x: read + write
    print("  D2D memcpy achievable BW: ", bw, " GB/s (read+write traffic)", flush=True)
    cuda_free(d_a)
    cuda_free(d_b)


def main() raises:
    print("=== dp4a GEMV per-shape roofline (GB10 HBM theoretical ~273 GB/s) ===", flush=True)
    with DeviceContext() as ctx:
        read_bw_probe(ctx)
        bw_probe(ctx)
        bench_shape(ctx, String("D_Dq  "), 5376, 8192)    # q-proj
        bench_shape(ctx, String("D_Dkv "), 5376, 4096)    # k/v-proj (GQA, low blocks)
        bench_shape(ctx, String("O_D   "), 8192, 5376)    # o-proj
        bench_shape(ctx, String("D_FF  "), 5376, 21504)   # gate/up (the 211 shape)
        bench_shape(ctx, String("FF_D  "), 21504, 5376)   # down (long-K, low blocks)
