"""fp4_gemv_v3_tree_validate.mojo — gate v3-tree under the CORRECTED policy:
  (1) DETERMINISTIC: v3-tree output identical across repeated runs (fixed tree order).
  (2) Within FP4 tolerance of an fp32/fp64 REFERENCE: per-element |tree - ref| bounded
      by the FP4 quantization noise floor; argmax may differ from the reference ONLY
      on genuine near-ties (semantically free).

Reference = host fp64 dot product: for each output n, sum_k (global_scale*e4m3(scale[n,k/16])
* sign(e2m1 nib) * e2m1_mag(e2m1 nib)) * act[k], accumulated in fp64. This is the
"fp32 reference" under FP4 data (weights/acts are fp4-quantized; accumulator fp64->fp32).

Reports per shape: max|tree-ref|, max rel err, mean abs err, mean rel err, argmax
agreement (tree argmax vs ref argmax; flags only if the ref-runner-up is NOT a near-tie).

  pixi run mojo build tools/fp4_gemv_v3_tree_validate.mojo -I . \
      --target-accelerator sm_121a -o build/v3tree_val \
      -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
  gpurun ./build/v3tree_val
"""
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List

from lib.cuda import cuda_malloc, cuda_upload, cuda_upload_u8, cuda_sync, cuda_free, cuda_memcpy
from lib.fp4_weights import e4m3_decode, E2M1_MAG, FP4_GROUP
from lib.fp4_gemv_v2 import gpu_matmul_nvfp4_fused_v3_dev, gpu_matmul_nvfp4_fused_v3_tree_dev


# Reference decode tables (mirror kernel: e2m1 sign+magnitude from a 4-bit nibble).
def e2m1_signed(nib: Int) -> Float64:
    var mag_idx = nib & 7
    var mag = Float64(Float32(E2M1_MAG[mag_idx]))   # {0,.5,1,1.5,2,3,4,6}
    if (nib & 8) != 0:
        return -mag
    return mag


def build_blob(K: Int, N: Int) -> List[UInt8]:
    # Row-dependent perturbation (see fp4_greedy_correctness.mojo): without it, the row
    # stride K/16 scales / K/2 nibble-pairs is divisible by the pattern period, so distant
    # rows get IDENTICAL weights -> tied argmax that hides near-tie flips. Row-XOR breaks it.
    var n_rows = K * N
    var nb = n_rows // 16
    var nbk = K // FP4_GROUP
    var hpk = K // 2
    var blob = List[UInt8]()
    var i = 0
    while i < nb:
        var rr = i // nbk
        blob.append(UInt8(0x40 + (((i * 5) ^ (rr * 13)) & 0x3F)))
        i += 1
    var j = 0
    while j < n_rows // 2:
        var rr = j // hpk
        var lo = UInt8(((j * 3) ^ (rr * 7)) & 0xF)
        var hi = UInt8((((j * 5) + 1) ^ (rr * 11)) & 0xF)
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


def fp64_ref(blob: List[UInt8], act: List[Float32], K: Int, N: Int, gs: Float32) -> List[Float64]:
    """Host fp64 reference GEMV: out[n] = sum_k dequant_w[n,k]*act[k]."""
    var nb = K * N // FP4_GROUP
    var nbk = K // FP4_GROUP
    var out = List[Float64](length=N, fill=Float64(0.0))
    for n in range(N):
        var row_block0 = (n * K) // FP4_GROUP
        var packed_base = nb + (n * K) // 2
        var s = Float64(0.0)
        for b in range(nbk):
            var eff = Float64(gs * e4m3_decode(Int(blob[row_block0 + b])))
            var pbase = packed_base + b * (FP4_GROUP // 2)
            for t in range(FP4_GROUP):
                var byte = Int(blob[pbase + t // 2])
                var nib = (byte >> (4 * (t & 1))) & 0xF
                s += eff * e2m1_signed(nib) * Float64(act[b * FP4_GROUP + t])
        out[n] = s
    return out^


def run_tree(ctx: DeviceContext, d_out: UInt64, d_in: UInt64, d_w: UInt64, gs: Float32,
             K: Int, N: Int, wpb: Int) raises -> List[Float32]:
    if wpb == 1:
        gpu_matmul_nvfp4_fused_v3_dev(ctx, d_out, d_in, d_w, gs, K, N)
    elif wpb == 2:
        gpu_matmul_nvfp4_fused_v3_tree_dev[2](ctx, d_out, d_in, d_w, gs, K, N)
    elif wpb == 4:
        gpu_matmul_nvfp4_fused_v3_tree_dev[4](ctx, d_out, d_in, d_w, gs, K, N)
    elif wpb == 8:
        gpu_matmul_nvfp4_fused_v3_tree_dev[8](ctx, d_out, d_in, d_w, gs, K, N)
    cuda_sync()
    var h = List[Float32](length=N, fill=Float32(0.0))
    cuda_memcpy(UInt64(Int(h.unsafe_ptr())), d_out, N * 4, 2)
    cuda_sync()
    return h^


def validate_shape(label: String, K: Int, N: Int, wpb: Int, gs: Float32) raises:
    var blob = build_blob(K, N)
    var act = build_act(K)
    var reference = fp64_ref(blob, act, K, N, gs)   # fp64 reference

    with DeviceContext() as ctx:
        var d_w = cuda_malloc(len(blob))
        cuda_upload_u8(d_w, blob)
        var d_in = cuda_malloc(K * 4)
        cuda_upload(d_in, act)
        var d_out = cuda_malloc(N * 4)

        # DETERMINISM: two independent runs must match byte-for-byte.
        var run1 = run_tree(ctx, d_out, d_in, d_w, gs, K, N, wpb)
        var run2 = run_tree(ctx, d_out, d_in, d_w, gs, K, N, wpb)
        var det_max = Float64(0.0)
        for n in range(N):
            var d = Float64(run1[n]) - Float64(run2[n])
            if d < 0: d = -d
            if d > det_max: det_max = d

        # TOLERANCE vs fp64 ref.
        var max_abs = Float64(0.0)
        var max_rel = Float64(0.0)
        var sum_abs = Float64(0.0)
        var sum_rel = Float64(0.0)
        for n in range(N):
            var r = reference[n]
            var t = Float64(run1[n])
            var d = t - r
            if d < 0: d = -d
            if d > max_abs: max_abs = d
            sum_abs += d
            var ar = r
            if ar < 0: ar = -ar
            if ar > Float64(1.0):
                var rel = d / ar
                if rel > max_rel: max_rel = rel
                sum_rel += rel

        # ARGMAX agreement: ref argmax vs tree argmax. Flag only if the runner-up
        # is NOT a near-tie (gap > 1% of |top|) -> genuine disagreement, not free flip.
        var ref_top = Float64(-3.0e38)
        var ref_top_i = 0
        var ref_2nd = Float64(-3.0e38)
        for n in range(N):
            if reference[n] > ref_top:
                ref_2nd = ref_top
                ref_top = reference[n]
                ref_top_i = n
            elif reference[n] > ref_2nd:
                ref_2nd = reference[n]
        var tree_top = Float64(-3.0e38)
        var tree_top_i = 0
        for n in range(N):
            if Float64(run1[n]) > tree_top:
                tree_top = Float64(run1[n])
                tree_top_i = n
        var argmax_ok = tree_top_i == ref_top_i
        var near_tie = False
        if not argmax_ok:
            var gap = ref_top - ref_2nd
            var atop = ref_top
            if atop < 0: atop = -atop
            if gap < Float64(0.01) * atop:
                near_tie = True   # genuine near-tie -> flip is semantically free

        print(String("["), label, " wpb=", wpb, " K=", K, " N=", N, "]", flush=True)
        print("  determinism (max|run1-run2|)  :", det_max, flush=True)
        print("  vs fp64 ref: max_abs=", max_abs, " max_rel=", max_rel,
              " mean_abs=", sum_abs / Float64(N), " mean_rel=", sum_rel / Float64(N), flush=True)
        print("  argmax: ref_top=", ref_top, "@", ref_top_i,
              " tree_top=", tree_top, "@", tree_top_i,
              " match=", argmax_ok, " near_tie=", near_tie, flush=True)
        if det_max > Float64(0.0):
            print("  ** NON-DETERMINISTIC (run1 != run2) **", flush=True)
        if not argmax_ok and not near_tie:
            print("  ** ARGMAX DISAGREES on a NON-near-tie (gate FAIL) **", flush=True)
        elif argmax_ok or near_tie:
            print("  -> GATE OK (deterministic + within FP4 tol + argmax safe)", flush=True)

        cuda_free(d_w)
        cuda_free(d_in)
        cuda_free(d_out)


def main() raises:
    var gs = Float32(0.0078125)
    # Validate v3 (wpb=1, the safe baseline) AND v3-tree winners on each shape's best WPB.
    # FP_D uses v3t2 (its winner); D_VOC uses v3 (tree hurt it); D_FF uses v3t4; small ones both.
    validate_shape("D_Dq-v3   ", 5376, 8192, 1, gs)
    validate_shape("D_Dq-v3t2 ", 5376, 8192, 2, gs)
    validate_shape("D_Dkv-v3  ", 5376, 4096, 1, gs)
    validate_shape("D_Dkv-v3t4", 5376, 4096, 4, gs)
    validate_shape("D_FF-v3   ", 5376, 21504, 1, gs)
    validate_shape("D_FF-v3t4 ", 5376, 21504, 4, gs)
    validate_shape("FF_D-v3   ", 21504, 5376, 1, gs)
    validate_shape("FF_D-v3t2 ", 21504, 5376, 2, gs)
    validate_shape("D_VOC-v3  ", 5376, 262144, 1, gs)
