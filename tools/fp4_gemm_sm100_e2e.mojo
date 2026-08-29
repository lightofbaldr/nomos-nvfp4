"""Real-Gemma-4-shape W4A4 validation + throughput for lib/fp4_gemm_sm100 (B200).

  pixi run mojo run -I . tools/fp4_gemm_sm100_e2e.mojo

The 128^3 unit tests prove correctness; this proves the kernel + SF-atom scatter hold at the
ACTUAL projection shapes the engine drives (large N tiling over N//128, large K over K//64
slices), and measures per-GEMM latency -> an estimated decode tok/s for Gemma-4-31B.

Correctness pattern (per shape): A row m = FP4 nibble (m%8), B row n = (n%8), scales=1.0
=> C[m,n] = K * fp4(m%8) * fp4(n%8)  (fp4 lut {0,.5,1,1.5,2,3,4,6}); checked on a sampled
grid for the huge (lm-head) shape. M is padded to 128 (the sm_100 MMA m-dim, == decode M=1).
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from lib.fp4_gemm_sm100 import gpu_fp4_gemm_sm100, gpu_nvfp4_sf_scatter

comptime M = 128                         # decode M=1 padded to the MMA m-dim
comptime E4M3_ONE = UInt8(0x38)


def _addr(p: UnsafePointer) -> UInt64:
    return UInt64(Int(p))


def fp4(nib: Int) -> Float32:
    var lut = [Float32(0), 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return lut[nib & 7]


def run_shape(ctx: DeviceContext, label: String, K: Int, N: Int, check: Bool) raises -> Float64:
    var KB = K // 2
    var NBLK = K // 16
    var sf_atom_a = ((M + 127) // 128) * (((NBLK) + 3) // 4) * (32 * 4 * 4)
    var sf_atom_b = ((N + 127) // 128) * (((NBLK) + 3) // 4) * (32 * 4 * 4)

    var a_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
    for m in range(M):
        var nib = UInt8(m % 8); var byte = (nib << 4) | nib
        for c in range(KB): a_h[m * KB + c] = byte
    for n in range(N):
        var nib = UInt8(n % 8); var byte = (nib << 4) | nib
        for c in range(KB): b_h[n * KB + c] = byte
    var a = ctx.enqueue_create_buffer[DType.uint8](M * KB)
    var b = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    ctx.enqueue_copy(dst_buf=a, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b, src_buf=b_h)

    var sfa_flat = ctx.enqueue_create_buffer[DType.uint8](M * NBLK)
    var sfb_flat = ctx.enqueue_create_buffer[DType.uint8](N * NBLK)
    sfa_flat.enqueue_fill(E4M3_ONE); sfb_flat.enqueue_fill(E4M3_ONE)
    var sfa = ctx.enqueue_create_buffer[DType.uint8](sf_atom_a)
    var sfb = ctx.enqueue_create_buffer[DType.uint8](sf_atom_b)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    c.enqueue_fill(Float32(-1.0))

    gpu_nvfp4_sf_scatter(ctx, _addr(sfa_flat.unsafe_ptr()), _addr(sfa.unsafe_ptr()), M, NBLK)
    gpu_nvfp4_sf_scatter(ctx, _addr(sfb_flat.unsafe_ptr()), _addr(sfb.unsafe_ptr()), N, NBLK)
    gpu_fp4_gemm_sm100(ctx, _addr(c.unsafe_ptr()), _addr(a.unsafe_ptr()), _addr(b.unsafe_ptr()),
                       _addr(sfa.unsafe_ptr()), _addr(sfb.unsafe_ptr()), M, N, K)
    ctx.synchronize()

    if check:
        var bad = 0; var maxerr = Float32(0)
        var step = 1 if N <= 32768 else 97   # sample columns for the huge lm-head shape
        with c.map_to_host() as host:
            for m in range(M):
                var n = 0
                while n < N:
                    var expv = Float32(K) * fp4(m % 8) * fp4(n % 8)
                    var e = abs(host[m * N + n] - expv)
                    if e > maxerr: maxerr = e
                    if e > 0.05 * (expv + 1.0): bad += 1
                    n += step
        print("  ", label, " K=", K, " N=", N, "  bad=", bad, " maxerr=", maxerr,
              "  C[1,1]=", "?" )
        _ = bad

    # ---- timing ----
    var iters = 30
    for _ in range(3):   # warmup
        gpu_fp4_gemm_sm100(ctx, _addr(c.unsafe_ptr()), _addr(a.unsafe_ptr()), _addr(b.unsafe_ptr()),
                           _addr(sfa.unsafe_ptr()), _addr(sfb.unsafe_ptr()), M, N, K)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        gpu_fp4_gemm_sm100(ctx, _addr(c.unsafe_ptr()), _addr(a.unsafe_ptr()), _addr(b.unsafe_ptr()),
                           _addr(sfa.unsafe_ptr()), _addr(sfb.unsafe_ptr()), M, N, K)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / Float64(iters) / 1.0e6
    var tflops = (2.0 * Float64(M) * Float64(N) * Float64(K)) / (ms * 1.0e-3) / 1.0e12
    print("  ", label, " K=", K, " N=", N, "  ", ms, "ms/call  ", tflops, "TFLOP/s")
    return ms


def main() raises:
    var ctx = DeviceContext()
    print("compute =", ctx.default_device_info.compute)
    print("== correctness @ real Gemma-4-31B shapes (M=128) ==")
    _ = run_shape(ctx, "qkv ", 5376, 8192, True)
    _ = run_shape(ctx, "o   ", 8192, 5376, True)
    _ = run_shape(ctx, "gate", 5376, 21504, True)
    _ = run_shape(ctx, "down", 21504, 5376, True)
    _ = run_shape(ctx, "lmh ", 5376, 262144, True)

    print("== per-GEMM latency (M=128 = decode M=1 padded) ==")
    var t_q = run_shape(ctx, "q   ", 5376, 8192, False)
    var t_o = run_shape(ctx, "o   ", 8192, 5376, False)
    var t_g = run_shape(ctx, "gate", 5376, 21504, False)
    var t_u = run_shape(ctx, "up  ", 5376, 21504, False)
    var t_d = run_shape(ctx, "down", 21504, 5376, False)
    var t_lm = run_shape(ctx, "lmh ", 5376, 262144, False)

    # crude decode-time model: per layer ~ q + k + v + o + gate + up + down.
    # k,v are small (approx with q); 62 layers; + 1 lm-head/token.
    var per_layer = t_q + 0.5 * t_q + 0.5 * t_q + t_o + t_g + t_u + t_d
    var per_tok_ms = per_layer * 62.0 + t_lm
    var tok_s = 1000.0 / per_tok_ms
    print("== estimated decode (GEMM-only, M->128 over-pad) ==")
    print("  per-layer GEMM ms =", per_layer, "  per-token ms (62L + lmhead) =", per_tok_ms)
    print("  => GEMM-bound est. tok/s =", tok_s, " (decode pads M=1->128; prefill amortizes)")
