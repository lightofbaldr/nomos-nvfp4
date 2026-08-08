"""q4_gemv_v2_parity.mojo — gate the Q4 v2 port.
  Phase #1 (occ, sx): BIT-IDENTICAL to the original q4_gemv_kernel. Assert max|orig-new|=0.
  Phase #2 (vec): NOT bit-identical (coalesced loads change lane->K mapping -> FP order).
    Gate under the CORRECTED policy: deterministic + within Q4 tolerance of an fp64
    reference, accumulator fp32. (Q4 affine dequant -> tiny per-element error.)

  pixi run mojo build tools/q4_gemv_v2_parity.mojo -I . --target-accelerator sm_121a \
      -o build/q4_parity -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib \
      -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
  gpurun ./build/q4_parity
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List

from lib.cuda import cuda_malloc, cuda_upload, cuda_upload_u8, cuda_sync, cuda_free, cuda_memcpy
from lib.q4_weights import GROUP, gpu_matmul_q4_fused_dev
from lib.q4_gemv_v2 import (
    gpu_matmul_q4_v2_occ_dev, gpu_matmul_q4_v2_sx_dev, gpu_matmul_q4_v2_vec_dev,
    gpu_matmul_q4_v2_vdec_dev,
)


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


def run_orig(ctx: DeviceContext, d_out: UInt64, d_in: UInt64, d_w: UInt64, K: Int, N: Int) raises -> List[Float32]:
    gpu_matmul_q4_fused_dev(ctx, d_out, d_in, d_w, K, N)
    cuda_sync()
    var h = List[Float32](length=N, fill=Float32(0.0))
    cuda_memcpy(UInt64(Int(h.unsafe_ptr())), d_out, N * 4, 2)
    cuda_sync()
    return h^


def run_variant(ctx: DeviceContext, tag: String, d_out: UInt64, d_in: UInt64, d_w: UInt64,
                K: Int, N: Int) raises -> List[Float32]:
    if tag == "occ4":
        gpu_matmul_q4_v2_occ_dev[4](ctx, d_out, d_in, d_w, K, N)
    elif tag == "occ8":
        gpu_matmul_q4_v2_occ_dev[8](ctx, d_out, d_in, d_w, K, N)
    elif tag == "occ16":
        gpu_matmul_q4_v2_occ_dev[16](ctx, d_out, d_in, d_w, K, N)
    elif tag == "sx4":
        gpu_matmul_q4_v2_sx_dev[4, 1024](ctx, d_out, d_in, d_w, K, N)
    elif tag == "vec4_2":
        gpu_matmul_q4_v2_vec_dev[4, 2](ctx, d_out, d_in, d_w, K, N)
    elif tag == "vec4_4":
        gpu_matmul_q4_v2_vec_dev[4, 4](ctx, d_out, d_in, d_w, K, N)
    elif tag == "vec8_4":
        gpu_matmul_q4_v2_vec_dev[8, 4](ctx, d_out, d_in, d_w, K, N)
    elif tag == "vdec4":
        gpu_matmul_q4_v2_vdec_dev[4](ctx, d_out, d_in, d_w, K, N)
    elif tag == "vdec8":
        gpu_matmul_q4_v2_vdec_dev[8](ctx, d_out, d_in, d_w, K, N)
    cuda_sync()
    var h = List[Float32](length=N, fill=Float32(0.0))
    cuda_memcpy(UInt64(Int(h.unsafe_ptr())), d_out, N * 4, 2)
    cuda_sync()
    return h^


def fp64_ref(blob: List[UInt8], act: List[Float32], K: Int, N: Int) -> List[Float64]:
    """Host fp64 reference Q4 GEMV. out[n] = sum_k (nib-8)*scale * act[k], fp64 accumulate."""
    var nb = (K * N + GROUP - 1) // GROUP
    var out = List[Float64](length=N, fill=Float64(0.0))
    for n in range(N):
        var s = Float64(0.0)
        for k in range(K):
            var i = n * K + k
            var block = i // GROUP
            var j = i % GROUP
            var sbits = Int(blob[block * 2]) | (Int(blob[block * 2 + 1]) << 8)
            var scale = Float64(_fp16_to_f64(sbits))
            var byte = Int(blob[nb * 2 + block * 16 + (j // 2)])
            var nib = (byte & 0x0F) if (j & 1) == 0 else ((byte >> 4) & 0x0F)
            var w = (Float64(Int(nib)) - 8.0) * scale
            s += w * Float64(act[k])
        out[n] = s
    return out^


def _fp16_to_f64(bits: Int) -> Float64:
    # minimal fp16 -> fp64 (assumes normal numbers; the synthetic scales are normal).
    var sign = (bits >> 15) & 1
    var exp = (bits >> 10) & 0x1F
    var mant = bits & 0x3FF
    var val = Float64(0.0)
    if exp == 0:
        val = Float64(mant) * Float64(2.0) ** Float64(-24)   # subnormal
    else:
        val = (Float64(1.0) + Float64(mant) / 1024.0) * Float64(2.0) ** Float64(exp - 15)
    if sign == 1:
        val = -val
    return val


def check_bitidentical(label: String, tag: String, K: Int, N: Int, orig: List[Float32], new: List[Float32]) raises -> Bool:
    var max_diff = Float64(0.0)
    for n in range(N):
        var d = Float64(orig[n]) - Float64(new[n])
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
    var ok = max_diff == Float64(0.0)
    print(String("  ["), label, " ", tag, "] bit-identical-to-orig: ", ok,
          " max|o-n|=", max_diff, flush=True)
    return ok


def check_fp4tol(label: String, tag: String, K: Int, N: Int,
                 new: List[Float32], reference: List[Float64]) raises -> Bool:
    var max_abs = Float64(0.0)
    var max_rel = Float64(0.0)
    for n in range(N):
        var r = reference[n]
        var t = Float64(new[n])
        var d = t - r
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var ar = r
        if ar < 0:
            ar = -ar
        if ar > Float64(1.0):
            var rel = d / ar
            if rel > max_rel:
                max_rel = rel
    print(String("  ["), label, " ", tag, "] vs fp64: max_abs=", max_abs,
          " max_rel=", max_rel, flush=True)
    return max_rel < Float64(1.0e-3)   # Q4 tol: <0.1% rel


def check_determinism(label: String, tag: String, N: Int,
                      run1: List[Float32], run2: List[Float32]) raises -> Bool:
    var det_max = Float64(0.0)
    for n in range(N):
        var d = Float64(run1[n]) - Float64(run2[n])
        if d < 0:
            d = -d
        if d > det_max:
            det_max = d
    print(String("  ["), label, " ", tag, "] determinism max|run1-run2|=", det_max, flush=True)
    return det_max == Float64(0.0)


def main() raises:
    # Small shapes for the host fp64 ref to be tractable; D_Dq covers K=5376 (the real D).
    var shapes = List[List[Int]]()
    var s1 = List[Int](length=2, fill=0); s1[0] = 5376; s1[1] = 1024; shapes.append(s1^)
    var s2 = List[Int](length=2, fill=0); s2[0] = 5376; s2[1] = 8192; shapes.append(s2^)

    var any_fail = False
    with DeviceContext() as ctx:
        for si in range(len(shapes)):
            var K = shapes[si][0]
            var N = shapes[si][1]
            var label = String("K=", K, "/N=", N)
            print(String("=== "), label, " ===", flush=True)
            var blob = build_blob(K, N)
            var act = build_act(K)
            var d_w = cuda_malloc(len(blob))
            cuda_upload_u8(d_w, blob)
            var d_in = cuda_malloc(K * 4)
            cuda_upload(d_in, act)
            var d_out = cuda_malloc(N * 4)

            var orig = run_orig(ctx, d_out, d_in, d_w, K, N)

            # Phase #1: bit-identical gates.
            for tag in ["occ4", "occ8", "occ16", "sx4"]:
                var new = run_variant(ctx, tag, d_out, d_in, d_w, K, N)
                if not check_bitidentical(label, tag, K, N, orig, new):
                    any_fail = True

            # Phase #2: deterministic + Q4-tolerance gate.
            var reference = fp64_ref(blob, act, K, N)
            for tag in ["vec4_2", "vec4_4", "vec8_4", "vdec4", "vdec8"]:
                var r1 = run_variant(ctx, tag, d_out, d_in, d_w, K, N)
                var r2 = run_variant(ctx, tag, d_out, d_in, d_w, K, N)
                var det = check_determinism(label, tag, N, r1, r2)
                var tol = check_fp4tol(label, tag, K, N, r1, reference)
                if not (det and tol):
                    any_fail = True

            cuda_free(d_w)
            cuda_free(d_in)
            cuda_free(d_out)

    print(String(""), flush=True)
    if any_fail:
        print("GATE FAIL", flush=True)
        raise Error("q4 v2 parity gate failed")
    print("GATE OK: Phase #1 (occ/sx) bit-identical to orig; Phase #2 (vec) det+Q4-tol.", flush=True)
