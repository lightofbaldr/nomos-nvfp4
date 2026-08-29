"""Known-answer test for the tcgen05 W4A4 GEMM (lib/fp4_gemm_sm100). Runs ONLY on
sm_100 (B200/B300) — the tcgen05 block-scale MMA does not exist on sm_120/121, so
this JIT-fails on GB10. Build locally to compile-check; run on a B200.

  pixi run mojo run -I . --target-accelerator sm_100a tools/fp4_gemm_sm100_test.mojo

Test 1 (this file): A,B all FP4=1.0 (E2M1 nibble 0x2), all scales e4m3=1.0 (0x38).
Then C[m,n] = sum_{k<K} 1*1 = K. Every C must equal K (=128). A clean first signal
that the tcgen05 pipeline + SF-atom staging + fragment store are wired right; the
random + CPU-dequant parity pass comes once this passes.
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from lib.fp4_gemm_sm100 import gpu_fp4_gemm_sm100, gpu_nvfp4_sf_scatter

comptime M = 128
comptime N = 128
comptime K = 128
comptime AB_BYTES = M * (K // 2)        # packed FP4: 2 vals/byte
comptime SF_FLAT = M * (K // 16)        # flat e4m3 block scales [MN, K/16]
comptime SF_ATOM = 1 * 2 * 32 * 4 * 4   # ceil(M/128)*ceil((K/16)/4)*32*4*4
comptime FP4_ONE = UInt8(0x22)          # two E2M1 1.0 nibbles
comptime E4M3_ONE = UInt8(0x38)         # e4m3 1.0


def _addr(p: UnsafePointer) -> UInt64:
    return UInt64(Int(p))


def main() raises:
    var ctx = DeviceContext()

    var a = ctx.enqueue_create_buffer[DType.uint8](AB_BYTES)
    var b = ctx.enqueue_create_buffer[DType.uint8](AB_BYTES)
    var sfa_flat = ctx.enqueue_create_buffer[DType.uint8](SF_FLAT)
    var sfb_flat = ctx.enqueue_create_buffer[DType.uint8](SF_FLAT)
    var sfa = ctx.enqueue_create_buffer[DType.uint8](SF_ATOM)
    var sfb = ctx.enqueue_create_buffer[DType.uint8](SF_ATOM)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)

    a.enqueue_fill(FP4_ONE)
    b.enqueue_fill(FP4_ONE)
    sfa_flat.enqueue_fill(E4M3_ONE)
    sfb_flat.enqueue_fill(E4M3_ONE)
    sfa.enqueue_fill(UInt8(0))
    sfb.enqueue_fill(UInt8(0))
    c.enqueue_fill(Float32(-1.0))

    # scales flat -> SF-atom layout
    gpu_nvfp4_sf_scatter(ctx, _addr(sfa_flat.unsafe_ptr()), _addr(sfa.unsafe_ptr()), M, K // 16)
    gpu_nvfp4_sf_scatter(ctx, _addr(sfb_flat.unsafe_ptr()), _addr(sfb.unsafe_ptr()), N, K // 16)

    gpu_fp4_gemm_sm100(
        ctx,
        _addr(c.unsafe_ptr()),
        _addr(a.unsafe_ptr()), _addr(b.unsafe_ptr()),
        _addr(sfa.unsafe_ptr()), _addr(sfb.unsafe_ptr()),
        M, N, K,
    )
    ctx.synchronize()

    var expected = Float32(K)
    var bad = 0
    var maxerr = Float32(0)
    with c.map_to_host() as host:
        for i in range(M * N):
            var v = host[i]
            var e = abs(v - expected)
            if e > maxerr:
                maxerr = e
            if e > 0.01:
                bad += 1
        print("C[0,0]=", host[0], " C[1,1]=", host[N + 1], " C[127,127]=", host[(M - 1) * N + (N - 1)])
    print("expected=", expected, " bad=", bad, "/", M * N, " maxerr=", maxerr)
    if bad == 0:
        print("PASS — tcgen05 W4A4 all-ones known-answer")
    else:
        print("FAIL — see C samples above")
