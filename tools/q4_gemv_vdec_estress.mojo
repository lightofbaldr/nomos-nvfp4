"""q4_gemv_vdec_estress.mojo — STRESS proxy gate for vdec (the e2e-token-parity gate
is the BINDING one; this is a stronger MICROBENCH proxy to de-risk that run).

Why this exists: tools/q4_gemv_v2_parity.mojo uses a 'nice' periodic input where the FP
reorder is masked -> max_rel=0.0 (vacuous pass; both orig and vdec exactly equal fp64).
That does NOT exercise the vdec accumulation reorder. This tool feeds a NON-PERIODIC input
where fp32 accumulation rounding is non-trivial, and surfaces:
  (1) the e2e-relevant LOGIT DELTA from the swap: max|vdec - orig| (what the e2e
      stream will actually see when _mm_dev swaps orig->vdec),
  (2) vdec's tolerance vs fp64: max|vdec - fp64ref|,
  (3) orig's tolerance vs fp64 (context — orig is also not fp64-exact),
  (4) ARGMAX stability: orig-argmax vs vdec-argmax vs fp64-argmax; flag any disagreement
      as a near-tie (semantically free per corrected policy) or a genuine flip (gate fail).

This is a PROXY, not validation. The binding gate is live greedy-token parity on the GB10 box.
ADDITIVE only — does not touch the 4 staged files pulled into canonical.

  pixi run mojo build tools/q4_gemv_vdec_estress.mojo -I . --target-accelerator sm_121a \
      -o build/q4_vdec_estress -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib \
      -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
  gpurun ./build/q4_vdec_estress
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List

from lib.cuda import cuda_malloc, cuda_upload, cuda_upload_u8, cuda_sync, cuda_free, cuda_memcpy
from lib.q4_weights import GROUP, gpu_matmul_q4_fused_dev
from lib.q4_gemv_v2 import gpu_matmul_q4_v2_vdec_dev


def build_blob(K: Int, N: Int) -> List[UInt8]:
    # Row-XOR varied blob. Scale bytes bounded to NORMAL fp16 range (high byte 0x20..0x3C
    # -> exp field 8..15 -> scales ~0.0078..1.0). Avoids 0x7F high byte = fp16 inf/nan,
    # which otherwise makes every output NaN and the gate vacuous.
    var n = K * N
    var nb = (n + GROUP - 1) // GROUP
    var blob = List[UInt8]()
    var i = 0
    while i < nb * 2:
        var hi = (i & 1) == 1   # odd index = high byte of the fp16 scale
        if hi:
            blob.append(UInt8(0x20 + ((i * 5) & 0x1C)))   # 0x20..0x3C, normal exp
        else:
            blob.append(UInt8((i * 7) & 0xFF))             # low byte: varied mantissa
        i += 1
    var j = 0
    while j < nb * 16:
        var lo = UInt8((j * 3) & 0xF)
        var hi = UInt8((j * 5 + 1) & 0xF)
        blob.append(UInt8(lo | (hi << 4)))
        j += 1
    return blob^


def build_act_nonperiodic(K: Int) -> List[Float32]:
    # NON-PERIODIC activation via an LCG -> fp32 accumulation rounding is non-trivial
    # (mix of large/small contributions, varied signs). Small magnitude (~+-8) so the
    # K=5376 dot products land in a meaningful fp32 range (not gigantic, not negligible).
    var a = List[Float32]()
    var state = UInt64(0x9E3779B97F4A7C15)   # nonzero seed
    var i = 0
    while i < K:
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var raw = Int(state >> UInt64(48)) & 0x3FFF          # 14-bit
        a.append((Float32(raw) - Float32(8192.0)) * Float32(0.001))  # ~[-8, 8], varied
        i += 1
    return a^


def run_kernel(ctx: DeviceContext, which: String, d_out: UInt64, d_in: UInt64, d_w: UInt64,
               K: Int, N: Int) raises -> List[Float32]:
    if which == "orig":
        gpu_matmul_q4_fused_dev(ctx, d_out, d_in, d_w, K, N)
    elif which == "vdec4":
        gpu_matmul_q4_v2_vdec_dev[4](ctx, d_out, d_in, d_w, K, N)
    cuda_sync()
    var h = List[Float32](length=N, fill=Float32(0.0))
    cuda_memcpy(UInt64(Int(h.unsafe_ptr())), d_out, N * 4, 2)
    cuda_sync()
    return h^


def _fp16_to_f64(bits: Int) -> Float64:
    var sign = (bits >> 15) & 1
    var exp = (bits >> 10) & 0x1F
    var mant = bits & 0x3FF
    var val = Float64(0.0)
    if exp == 0:
        val = Float64(mant) * Float64(2.0) ** Float64(-24)
    else:
        val = (Float64(1.0) + Float64(mant) / 1024.0) * Float64(2.0) ** Float64(exp - 15)
    if sign == 1:
        val = -val
    return val


def fp64_ref(blob: List[UInt8], act: List[Float32], K: Int, N: Int) -> List[Float64]:
    var nb = (K * N + GROUP - 1) // GROUP
    var out = List[Float64](length=N, fill=Float64(0.0))
    for n in range(N):
        var s = Float64(0.0)
        for k in range(K):
            var i = n * K + k
            var block = i // GROUP
            var j = i % GROUP
            var sbits = Int(blob[block * 2]) | (Int(blob[block * 2 + 1]) << 8)
            var scale = _fp16_to_f64(sbits)
            var byte = Int(blob[nb * 2 + block * 16 + (j // 2)])
            var nib = (byte & 0x0F) if (j & 1) == 0 else ((byte >> 4) & 0x0F)
            s += (Float64(Int(nib)) - 8.0) * scale * Float64(act[k])
        out[n] = s
    return out^


def stats_pair(a: List[Float32], b: List[Float32], N: Int) -> List[Float64]:
    # returns [max_abs, max_rel, mean_abs]
    var max_abs = Float64(0.0)
    var max_rel = Float64(0.0)
    var sum_abs = Float64(0.0)
    for n in range(N):
        var d = Float64(a[n]) - Float64(b[n])
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        sum_abs += d
        var denom = Float64(b[n])
        if denom < 0:
            denom = -denom
        if denom > Float64(1.0):
            var rel = d / denom
            if rel > max_rel:
                max_rel = rel
    var out = List[Float64](length=3, fill=Float64(0.0))
    out[0] = max_abs; out[1] = max_rel; out[2] = sum_abs / Float64(N)
    return out^


def stats_vs_fp64(a: List[Float32], reference: List[Float64], N: Int) -> List[Float64]:
    var max_abs = Float64(0.0)
    var max_rel = Float64(0.0)
    for n in range(N):
        var d = Float64(a[n]) - reference[n]
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        var denom = reference[n]
        if denom < 0:
            denom = -denom
        if denom > Float64(1.0):
            var rel = d / denom
            if rel > max_rel:
                max_rel = rel
    var out = List[Float64](length=2, fill=Float64(0.0))
    out[0] = max_abs; out[1] = max_rel
    return out^


def argmax(scores: List[Float64], N: Int) -> Int:
    var best = Float64(-3.0e38)
    var bi = -1
    for n in range(N):
        if scores[n] > best:
            best = scores[n]
            bi = n
    return bi


def argmax32(scores: List[Float32], N: Int) -> Int:
    var best = Float32(-3.0e38)
    var bi = -1
    for n in range(N):
        if scores[n] > best:
            best = scores[n]
            bi = n
    return bi


def main() raises:
    # Real K=5376; N sized so host fp64 (O(N*K)) is tractable but argmax is meaningful.
    var shapes = List[List[Int]]()
    var s1 = List[Int](length=2, fill=0); s1[0] = 5376; s1[1] = 4096; shapes.append(s1^)
    var s2 = List[Int](length=2, fill=0); s2[0] = 5376; s2[1] = 8192; shapes.append(s2^)

    with DeviceContext() as ctx:
        for si in range(len(shapes)):
            var K = shapes[si][0]
            var N = shapes[si][1]
            print(String("=== K="), K, " N=", N, " (non-periodic stress input) ===", flush=True)
            var blob = build_blob(K, N)
            var act = build_act_nonperiodic(K)
            var d_w = cuda_malloc(len(blob))
            cuda_upload_u8(d_w, blob)
            var d_in = cuda_malloc(K * 4)
            cuda_upload(d_in, act)
            var d_out = cuda_malloc(N * 4)

            var h_orig = run_kernel(ctx, "orig", d_out, d_in, d_w, K, N)
            var h_vdec = run_kernel(ctx, "vdec4", d_out, d_in, d_w, K, N)
            var reference = fp64_ref(blob, act, K, N)

            # (1) e2e-relevant logit delta from the swap (the number the stream sees).
            var dv = stats_pair(h_vdec, h_orig, N)
            print(String("  vdec vs orig  : max_abs="), dv[0], " max_rel=", dv[1],
                  " mean_abs=", dv[2], flush=True)

            # (2) vdec tolerance vs fp64; (3) orig tolerance vs fp64 (context).
            var vf = stats_vs_fp64(h_vdec, reference, N)
            var of = stats_vs_fp64(h_orig, reference, N)
            print(String("  vdec vs fp64  : max_abs="), vf[0], " max_rel=", vf[1], flush=True)
            print(String("  orig vs fp64  : max_abs="), of[0], " max_rel=", of[1],
                  " (context: orig is also not fp64-exact)", flush=True)

            # (4) ARGMAX stability. ref argmax + runner-up gap (near-tie test).
            var ref_scores = List[Float64](length=N, fill=Float64(0.0))
            for n in range(N):
                ref_scores[n] = reference[n]
            var ai_ref = argmax(ref_scores, N)
            var ai_orig = argmax32(h_orig, N)
            var ai_vdec = argmax32(h_vdec, N)
            # runner-up gap on fp64 ref
            var top = reference[ai_ref]
            var second = Float64(-3.0e38)
            for n in range(N):
                if n != ai_ref and reference[n] > second:
                    second = reference[n]
            var gap = top - second
            var atop = top
            if atop < 0:
                atop = -atop
            var near_tie = gap < Float64(0.01) * atop

            print(String("  argmax: fp64ref="), ai_ref, " (top=", top, " 2nd=", second,
                  " gap=", gap, " near_tie=", near_tie, ")", flush=True)
            print(String("         orig  ="), ai_orig,
                  " match_ref=", ai_orig == ai_ref, flush=True)
            print(String("         vdec  ="), ai_vdec,
                  " match_ref=", ai_vdec == ai_ref, " match_orig=", ai_vdec == ai_orig, flush=True)

            if ai_vdec != ai_orig:
                if near_tie:
                    print(String("  -> vdec argmax differs from orig BUT it's a near-tie (semantically free per corrected policy)"), flush=True)
                else:
                    print(String("  -> ** vdec argmax differs from orig on a NON-near-tie ** (proxy gate FAIL — would likely flip a greedy token e2e)"), flush=True)
            else:
                print(String("  -> vdec argmax == orig argmax (greedy-token-safe on this input)"), flush=True)

            cuda_free(d_w)
            cuda_free(d_in)
            cuda_free(d_out)

    print(String(""), flush=True)
    print("NOTE: this is a MICROBENCH proxy. The binding gate is live greedy-token", flush=True)
    print("parity on the GB10 box vs the reference stream. max|vdec-orig| above is the", flush=True)
    print("logit-delta magnitude the e2e stream will see from the swap.", flush=True)
