"""fp4 M=4 spec-verify BIT-EXACT seam gate (GEMM-isolated, no engine/attention).

Proves the NEW M>1 fused verify path is BYTE-IDENTICAL to the OLD M-row chain it
replaces, on the real gemma-4-31b verify shapes at M=4 (Mpad=16), REAL random inputs:

  OLD (pre-fix, the chain gpu_matmul_nvfp4_w4a4_dev used for M>1):
    gpu_quant_act_nvfp4(Mpad=16, Mreal=4)  [one-warp/row serial]
    -> gpu_fp4_gemm (bare smem GEMM into c_pad[16,N])
    -> postscale_w4a4_kernel (out[m,n] = c_pad[m,n] * act_global[m] * weight_global)

  NEW (this fix):
    gpu_quant_act_nvfp4_fast_m(16, 4)      [amax micro-kernel + grid-parallel encode]
    -> gpu_fp4_gemm_ps_m (fused-postscale M-row GEMM, no c_pad, no postscale launch)

Gate (ALL must be byte-identical, else the fix is WRONG):
  * quant globals[16]  (fast-m gscale == one-warp gscale, per row + pad rows)
  * quant packed[16,K/2] + scales[16,K/16]  (encode bytes, incl pad rows)
  * output[4,N] fp32   (final postscaled result, bitwise)

Build (discrete Blackwell, sm_120a):
  pixi run mojo build tools/fp4_m4_parity.mojo -I . \
      --target-accelerator sm_120a -o build/fp4_m4_parity \
      -Xlinker -L/usr/local/cuda/lib64 -Xlinker -lcudart -Xlinker -lm
Run:  CUDA_VISIBLE_DEVICES=0 build/fp4_m4_parity
"""

from max.gpu.host import DeviceContext
from std.memory import UnsafePointer

from lib.fp4_gemm import gpu_fp4_gemm, gpu_fp4_gemm_ps_m
from lib.fp4_act import (
    gpu_quant_act_nvfp4,
    gpu_quant_act_nvfp4_fast_m,
    postscale_w4a4_kernel,
)

comptime MPAD = 16          # M=4 verify padded to the MMA m-dim
comptime MREAL = 4          # spec-verify k+1 = 4 rows


@fieldwise_init
struct VShape(Copyable, Movable):
    var name: String
    var k: Int
    var n: Int


def _xs(state: UInt64) -> UInt64:
    var x = state
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


def check_shape(ctx: DeviceContext, sh: VShape) raises -> Int:
    """Returns 0 on full byte-parity, else #mismatches (and prints where)."""
    var K = sh.k
    var N = sh.n
    var KB = K // 2
    var NBK = K // 16
    var nb_w = N * NBK
    var wpk = N * KB
    var blob_bytes = nb_w + wpk

    # weight blob: [e4m3 scales : nb_w][E2M1 packed], random (engine layout)
    var blob_h = ctx.enqueue_create_host_buffer[DType.uint8](blob_bytes)
    ctx.synchronize()
    var wp = blob_h.unsafe_ptr().bitcast[UInt64]()
    var s = UInt64(0x9E3779B97F4A7C15) ^ (UInt64(K) << 32) ^ UInt64(N)
    var blob_words = blob_bytes // 8
    var scale_words = nb_w // 8
    for i in range(blob_words):
        s = _xs(s)
        if i < scale_words:
            wp[i] = (s & 0x1F1F1F1F1F1F1F1F) + 0x2828282828282828
        else:
            wp[i] = s

    # activation: MREAL random fp32 rows, pad rows zeroed (quant kernel overwrites)
    var act_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD * K)
    for i in range(MPAD * K):
        act_h[i] = Float32(0)
    for i in range(MREAL * K):
        s = _xs(s)
        act_h[i] = Float32(Int((s >> 40) & 0xFFF)) / Float32(1024.0) - Float32(2.0)

    var blob_d = ctx.enqueue_create_buffer[DType.uint8](blob_bytes)
    var act_d = ctx.enqueue_create_buffer[DType.float32](MPAD * K)
    # OLD-chain scratch
    var apk_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * KB)
    var abs_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * NBK)
    var agl_d = ctx.enqueue_create_buffer[DType.float32](MPAD)
    var c_d = ctx.enqueue_create_buffer[DType.float32](MPAD * N)
    var outo_d = ctx.enqueue_create_buffer[DType.float32](MREAL * N)
    # NEW-chain scratch (separate, so we can diff bytes)
    var apk2_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * KB)
    var abs2_d = ctx.enqueue_create_buffer[DType.uint8](MPAD * NBK)
    var agl2_d = ctx.enqueue_create_buffer[DType.float32](MPAD)
    var outn_d = ctx.enqueue_create_buffer[DType.float32](MREAL * N)

    ctx.enqueue_copy(blob_d, blob_h)
    ctx.enqueue_copy(act_d, act_h)
    c_d.enqueue_fill(Float32(0.0))
    outo_d.enqueue_fill(Float32(0.0))
    outn_d.enqueue_fill(Float32(0.0))
    # seed the NEW quant scratch with junk so a partial encode is caught
    apk2_d.enqueue_fill(UInt8(0xAA))
    abs2_d.enqueue_fill(UInt8(0xAA))
    agl2_d.enqueue_fill(Float32(-1.0))
    ctx.synchronize()

    var blob_a = UInt64(Int(blob_d.unsafe_ptr()))
    var act_a = UInt64(Int(act_d.unsafe_ptr()))
    var apk_a = UInt64(Int(apk_d.unsafe_ptr()))
    var abs_a = UInt64(Int(abs_d.unsafe_ptr()))
    var agl_a = UInt64(Int(agl_d.unsafe_ptr()))
    var c_a = UInt64(Int(c_d.unsafe_ptr()))
    var outo_a = UInt64(Int(outo_d.unsafe_ptr()))
    var apk2_a = UInt64(Int(apk2_d.unsafe_ptr()))
    var abs2_a = UInt64(Int(abs2_d.unsafe_ptr()))
    var agl2_a = UInt64(Int(agl2_d.unsafe_ptr()))
    var outn_a = UInt64(Int(outn_d.unsafe_ptr()))

    var wg = Float32(0.01)

    # ── OLD chain ──
    gpu_quant_act_nvfp4(ctx, act_a, apk_a, abs_a, agl_a, MPAD, MREAL, K)
    gpu_fp4_gemm(ctx, c_a, apk_a, blob_a + UInt64(nb_w), abs_a, blob_a, MPAD, N, K)
    var pk = ctx.compile_function[postscale_w4a4_kernel]()
    ctx.enqueue_function(
        pk,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(outo_a)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c_a)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(agl_a)),
        wg, MREAL, N,
        grid_dim=(MREAL * N + 255) // 256, block_dim=256,
    )

    # ── NEW chain ──
    gpu_quant_act_nvfp4_fast_m(ctx, act_a, apk2_a, abs2_a, agl2_a, MPAD, MREAL, K)
    gpu_fp4_gemm_ps_m(ctx, outn_a, apk2_a, blob_a + UInt64(nb_w), abs2_a, blob_a,
                      agl2_a, wg, MREAL, N, K)
    ctx.synchronize()

    # ── download + diff ──
    var agl_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD)
    var agl2_h = ctx.enqueue_create_host_buffer[DType.float32](MPAD)
    var apk_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * KB)
    var apk2_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * KB)
    var abs_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * NBK)
    var abs2_h = ctx.enqueue_create_host_buffer[DType.uint8](MPAD * NBK)
    var outo_h = ctx.enqueue_create_host_buffer[DType.float32](MREAL * N)
    var outn_h = ctx.enqueue_create_host_buffer[DType.float32](MREAL * N)
    ctx.enqueue_copy(agl_h, agl_d)
    ctx.enqueue_copy(agl2_h, agl2_d)
    ctx.enqueue_copy(apk_h, apk_d)
    ctx.enqueue_copy(apk2_h, apk2_d)
    ctx.enqueue_copy(abs_h, abs_d)
    ctx.enqueue_copy(abs2_h, abs2_d)
    ctx.enqueue_copy(outo_h, outo_d)
    ctx.enqueue_copy(outn_h, outn_d)
    ctx.synchronize()

    var gmm = 0
    for i in range(MPAD):
        if agl_h.unsafe_ptr().bitcast[UInt32]()[i] != agl2_h.unsafe_ptr().bitcast[UInt32]()[i]:
            gmm += 1
    var pmm = 0
    for i in range(MPAD * KB):
        if apk_h[i] != apk2_h[i]:
            pmm += 1
    var smm = 0
    for i in range(MPAD * NBK):
        if abs_h[i] != abs2_h[i]:
            smm += 1
    var omm = 0
    var first = -1
    for i in range(MREAL * N):
        if outo_h.unsafe_ptr().bitcast[UInt32]()[i] != outn_h.unsafe_ptr().bitcast[UInt32]()[i]:
            omm += 1
            if first < 0:
                first = i
    var total = gmm + pmm + smm + omm
    var nz = 0
    for i in range(N):
        if outo_h[i] != Float32(0.0):
            nz += 1
    print(sh.name, ": K=", K, " N=", N,
          " | globals_mm=", gmm, " packed_mm=", pmm, " scales_mm=", smm,
          " out_mm=", omm, "/", MREAL * N,
          (" firstdiff@" + String(first)) if omm > 0 else "",
          " | row0 nonzero=", nz, "/", N,
          ("  PASS" if total == 0 and nz > N // 2 else "  *** FAIL ***"))
    _ = blob_d^
    _ = act_d^
    _ = apk_d^
    _ = abs_d^
    _ = agl_d^
    _ = c_d^
    _ = outo_d^
    _ = apk2_d^
    _ = abs2_d^
    _ = agl2_d^
    _ = outn_d^
    return total


def main() raises:
    var ctx = DeviceContext()
    print("== fp4 M=4 spec-verify bit-exact seam gate (Mpad=16, Mreal=4, sm_120) ==")
    print("device compute=", Float64(ctx.default_device_info.compute))
    var shapes = List[VShape]()
    shapes.append(VShape("kfull", 5376, 2048))     # k-proj full (N=2048)
    shapes.append(VShape("kvslide", 5376, 4096))   # k/v-proj sliding
    shapes.append(VShape("qslide", 5376, 8192))    # q-proj sliding
    shapes.append(VShape("qfull", 5376, 16384))    # q-proj full
    shapes.append(VShape("oslide", 8192, 5376))    # o-proj sliding (K=8192)
    shapes.append(VShape("ofull", 16384, 5376))    # o-proj full (K=16384)
    shapes.append(VShape("gateup", 5376, 21504))   # gate/up (N=21504)
    shapes.append(VShape("down", 21504, 5376))     # down (K=21504)
    shapes.append(VShape("lmhead", 5376, 262144))  # lm-head readout (N=262144)
    var fails = 0
    for i in range(len(shapes)):
        fails += check_shape(ctx, shapes[i])
    print("")
    if fails == 0:
        print("SEAM M=4 PASS: all shapes byte-identical (quant globals+packed+scales, out[4,N])")
    else:
        print("SEAM M=4 FAIL:", fails, "total byte mismatches")
