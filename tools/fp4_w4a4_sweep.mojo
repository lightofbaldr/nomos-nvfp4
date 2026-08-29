"""W4A4 GEMM 2-D boundary sweep (2026-06-29) — find WHERE the M=1 W4A4 lm-head GEMM
breaks. Synthetic controlled weights, ARGMAX-MARGIN host-ref (not RMSE — argmax is margin-
sensitive; garbage hides under a band-looking RMSE). N-sweep at K=5376 + K-sweep at N=262144.
A clean threshold (e.g. N>=131072 where weight byte-offset crosses ~2^29) => offset/index-width
bug; "only both large" with no threshold => codegen/occupancy. Run:
  pixi run mojo run -I . --target-accelerator sm_121a tools/fp4_w4a4_sweep.mojo
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_dev, _f32_to_e4m3_byte, _e2m1_nibble
from lib.fp4_weights import e4m3_decode


def _u64(p: UnsafePointer[Float32, MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def _u64u8(p: UnsafePointer[UInt8, MutAnyOrigin]) -> UInt64:
    return UInt64(Int(p))


def main() raises:
    with DeviceContext() as ctx:
        var Ks = [5376, 5376, 5376, 5376, 5376, 256, 1024, 2688]
        var Ns = [64, 32768, 65536, 131072, 262144, 262144, 262144, 262144]
        var M = 1
        var Mpad = 16
        for t in range(len(Ks)):
            var K = Ks[t]
            var N = Ns[t]
            var KB = K // 2
            var NB = K // 16
            var nb_w = (N * K) // 16
            var n_w = N * K

            var act_h = ctx.enqueue_create_host_buffer[DType.float32](M * K)
            for i in range(M * K):
                act_h[i] = Float32(Int((i * 1103515245 + 12345) % 2000)) / 200.0 - 5.0
            var wf = ctx.enqueue_create_host_buffer[DType.float32](N * K)
            for i in range(N * K):
                wf[i] = Float32(Int((i * 2654435761 + 7) % 2000)) / 1000.0 - 1.0

            var wamax = Float32(0.0)
            for i in range(N * K):
                var a = abs(wf[i])
                if a > wamax:
                    wamax = a
            var wglobal = wamax / (Float32(6.0) * Float32(448.0))
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
                UInt64(0), UInt64(0),
                _u64(cpad_d.unsafe_ptr()),
                M, Mpad, K, N,
            )
            ctx.synchronize()
            ctx.enqueue_copy(out_h, out_d)
            ctx.synchronize()

            # ARGMAX-MARGIN host-ref: a correct fp4 GEMM has GPU argmax == host argmax (or a near-tie);
            # a BROKEN GEMM has the host argmax position nowhere near the GPU max, or NaN/degenerate.
            var gpu_am = 0
            var gpu_max = out_h[0]
            var gpu_min = out_h[0]
            var host_am = 0
            var host_max = Float32(-1.0e30)
            var nan_ct = 0
            for n in range(N):
                var v = out_h[n]
                if not (v == v):
                    nan_ct += 1
                if v > gpu_max:
                    gpu_max = v; gpu_am = n
                if v < gpu_min:
                    gpu_min = v
                var acc = Float32(0.0)
                for k in range(K):
                    acc += act_h[k] * wf[n * K + k]
                if acc > host_max:
                    host_max = acc; host_am = n
            var gpu_at_host_am = out_h[host_am]
            var rng = gpu_max - gpu_min
            # FAIL if NaN, degenerate (flat), or host-argmax lands in the bottom half of GPU's range
            var degenerate = rng <= Float32(1.0e-6) * (abs(gpu_max) + Float32(1.0))
            var disagree = gpu_at_host_am < (gpu_min + Float32(0.5) * rng)
            var fail = (nan_ct > 0) or degenerate or disagree
            print("K=", K, " N=", N, " | gpu_argmax=", gpu_am, " host_argmax=", host_am,
                  " | gpu_max=", gpu_max, " gpu@host_am=", gpu_at_host_am, " gpu_min=", gpu_min,
                  " nan=", nan_ct, " => ", "**FAIL**" if fail else "PASS")
            _ = act_d^
            _ = wblob_d^
            _ = apack_d^
            _ = abs_d^
            _ = aglob_d^
            _ = cpad_d^
            _ = out_d^
