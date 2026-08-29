"""Test 2 for lib/fp4_gemm_sm100 — VARYING A/B values (uniform scales). Runs on B200.

  pixi run mojo run -I . tools/fp4_gemm_sm100_test2.mojo

Row m of A is filled with FP4 nibble (m%8); row n of B with (n%8); scales = 1.0. So
C[m,n] = sum_{k<K} fp4(m%8)*fp4(n%8) = K * fp4(m%8) * fp4(n%8) — a value that DIFFERS
per (m,n), unlike the all-ones case. This catches a wrong store-map (which a uniform
tile hides) and verifies per-value E2M1 dequant. Scale-layout (SF-atom) is still
unit here; test 3 varies scales. E2M1 nibbles 0..7 = {0,.5,1,1.5,2,3,4,6}.
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from lib.fp4_gemm_sm100 import gpu_fp4_gemm_sm100, gpu_nvfp4_sf_scatter

comptime M = 128
comptime N = 128
comptime K = 128
comptime KB = K // 2
comptime SF_FLAT = M * (K // 16)
comptime SF_ATOM = 1 * 2 * 32 * 4 * 4
comptime E4M3_ONE = UInt8(0x38)


def _addr(p: UnsafePointer) -> UInt64:
    return UInt64(Int(p))


def fp4(nib: Int) -> Float32:
    var lut = [Float32(0), 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return lut[nib & 7]


def main() raises:
    var ctx = DeviceContext()
    var a_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
    for m in range(M):
        var nib = UInt8(m % 8)
        var byte = (nib << 4) | nib
        for col in range(KB):
            a_h[m * KB + col] = byte
    for n in range(N):
        var nib = UInt8(n % 8)
        var byte = (nib << 4) | nib
        for col in range(KB):
            b_h[n * KB + col] = byte

    var a = ctx.enqueue_create_buffer[DType.uint8](M * KB)
    var b = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    ctx.enqueue_copy(dst_buf=a, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b, src_buf=b_h)
    var sfa_flat = ctx.enqueue_create_buffer[DType.uint8](SF_FLAT)
    var sfb_flat = ctx.enqueue_create_buffer[DType.uint8](SF_FLAT)
    var sfa = ctx.enqueue_create_buffer[DType.uint8](SF_ATOM)
    var sfb = ctx.enqueue_create_buffer[DType.uint8](SF_ATOM)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    sfa_flat.enqueue_fill(E4M3_ONE)
    sfb_flat.enqueue_fill(E4M3_ONE)
    sfa.enqueue_fill(UInt8(0))
    sfb.enqueue_fill(UInt8(0))
    c.enqueue_fill(Float32(-1.0))

    gpu_nvfp4_sf_scatter(ctx, _addr(sfa_flat.unsafe_ptr()), _addr(sfa.unsafe_ptr()), M, K // 16)
    gpu_nvfp4_sf_scatter(ctx, _addr(sfb_flat.unsafe_ptr()), _addr(sfb.unsafe_ptr()), N, K // 16)
    gpu_fp4_gemm_sm100(
        ctx, _addr(c.unsafe_ptr()),
        _addr(a.unsafe_ptr()), _addr(b.unsafe_ptr()),
        _addr(sfa.unsafe_ptr()), _addr(sfb.unsafe_ptr()),
        M, N, K,
    )
    ctx.synchronize()

    var bad = 0
    var maxerr = Float32(0)
    with c.map_to_host() as host:
        for m in range(M):
            for n in range(N):
                var expv = Float32(K) * fp4(m % 8) * fp4(n % 8)
                var e = abs(host[m * N + n] - expv)
                if e > maxerr:
                    maxerr = e
                if e > 0.05:
                    bad += 1
        print("C[1,1]=", host[N + 1], " ref=", Float32(K) * fp4(1) * fp4(1),
              "  C[3,5]=", host[3 * N + 5], " ref=", Float32(K) * fp4(3) * fp4(5),
              "  C[7,2]=", host[7 * N + 2], " ref=", Float32(K) * fp4(7) * fp4(2))
    print("bad=", bad, "/", M * N, " maxerr=", maxerr)
    print("PASS — store-map + dequant" if bad == 0 else "FAIL — store-map or dequant wrong")
