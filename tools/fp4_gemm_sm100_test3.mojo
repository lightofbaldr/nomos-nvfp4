"""Test 3 for lib/fp4_gemm_sm100 — VARYING per-block SCALES (uniform A/B). Runs on B200.

  pixi run mojo run -I . tools/fp4_gemm_sm100_test3.mojo

test1/test2 both held all e4m3 block scales = 1.0, so they CANNOT catch a bug in the
SF-atom scale staging (flat[MN,K/16] -> gpu_nvfp4_sf_scatter -> SMEM -> tcgen05_cp -> TMEM
-> block-scale MMA). This test isolates the scale path: A=B=FP4 1.0 everywhere, but the
A block scales vary per (row, k-block). E2M1 1.0 nibble = 0x2 (byte 0x22). e4m3 powers of
two are exact: 2^e has byte 0x38 + (e<<3)  (1.0=0x38, 2.0=0x40, 4.0=0x48).

A scale for (row m, kblock kb in 0..7) = 2^((m + kb) % 3).  B scale = 1.0.
Each NVFP4 block = 16 K-elements sharing one scale. So with fp4_a=fp4_b=1:
  C[m,n] = sum_{kb=0..7} 16 * 1 * sa(m,kb) * 1 = 16 * sum_{kb} 2^((m+kb)%3).
This depends on m (not n), and exercises BOTH the row (m) and k-block (kb) axes of the
SF-atom layout — a wrong SF placement gives a wrong per-row sum.
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from lib.fp4_gemm_sm100 import gpu_fp4_gemm_sm100, gpu_nvfp4_sf_scatter

comptime M = 128
comptime N = 128
comptime K = 128
comptime KB = K // 2
comptime NBLK = K // 16                 # e4m3 block scales per row (=8)
comptime SF_FLAT = M * NBLK
comptime SF_ATOM = 1 * 2 * 32 * 4 * 4
comptime FP4_ONE = UInt8(0x22)          # two E2M1 1.0 nibbles


def _addr(p: UnsafePointer) -> UInt64:
    return UInt64(Int(p))


def main() raises:
    var ctx = DeviceContext()

    var a = ctx.enqueue_create_buffer[DType.uint8](M * KB)
    var b = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    a.enqueue_fill(FP4_ONE)
    b.enqueue_fill(FP4_ONE)

    # A scales vary per (m, kb) = 2^((m+kb)%3); B scales all 1.0.
    var sfa_flat = ctx.enqueue_create_host_buffer[DType.uint8](SF_FLAT)
    for m in range(M):
        for kb in range(NBLK):
            var e = (m + kb) % 3
            sfa_flat[m * NBLK + kb] = UInt8(0x38 + (e << 3))
    var sfa_flat_d = ctx.enqueue_create_buffer[DType.uint8](SF_FLAT)
    ctx.enqueue_copy(dst_buf=sfa_flat_d, src_buf=sfa_flat)
    var sfb_flat = ctx.enqueue_create_buffer[DType.uint8](SF_FLAT)
    sfb_flat.enqueue_fill(UInt8(0x38))   # 1.0

    var sfa = ctx.enqueue_create_buffer[DType.uint8](SF_ATOM)
    var sfb = ctx.enqueue_create_buffer[DType.uint8](SF_ATOM)
    var c = ctx.enqueue_create_buffer[DType.float32](M * N)
    sfa.enqueue_fill(UInt8(0))
    sfb.enqueue_fill(UInt8(0))
    c.enqueue_fill(Float32(-1.0))

    gpu_nvfp4_sf_scatter(ctx, _addr(sfa_flat_d.unsafe_ptr()), _addr(sfa.unsafe_ptr()), M, NBLK)
    gpu_nvfp4_sf_scatter(ctx, _addr(sfb_flat.unsafe_ptr()), _addr(sfb.unsafe_ptr()), N, NBLK)
    gpu_fp4_gemm_sm100(
        ctx, _addr(c.unsafe_ptr()),
        _addr(a.unsafe_ptr()), _addr(b.unsafe_ptr()),
        _addr(sfa.unsafe_ptr()), _addr(sfb.unsafe_ptr()),
        M, N, K,
    )
    ctx.synchronize()

    # expected per-row sum
    var exp_row = [Float32(0)] * M
    for m in range(M):
        var s = Float32(0)
        for kb in range(NBLK):
            var e = (m + kb) % 3
            var sa = Float32(1 << e)      # 2^e
            s += 16.0 * 1.0 * sa * 1.0
        exp_row[m] = s

    var bad = 0
    var maxerr = Float32(0)
    with c.map_to_host() as host:
        for m in range(M):
            for n in range(N):
                var e = abs(host[m * N + n] - exp_row[m])
                if e > maxerr:
                    maxerr = e
                if e > 0.05:
                    bad += 1
        print("C[0,0]=", host[0], " exp=", exp_row[0],
              "  C[1,0]=", host[N], " exp=", exp_row[1],
              "  C[2,0]=", host[2 * N], " exp=", exp_row[2])
    print("bad=", bad, "/", M * N, " maxerr=", maxerr)
    print("PASS — SF-atom scale staging" if bad == 0 else "FAIL — scale staging wrong")
