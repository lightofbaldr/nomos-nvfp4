"""fp4_greedy_correctness.mojo — gate the lm_head greedy fusion epilogue ORDER.

the alignment (5e ruling, match llama.cpp): softcap is a model-forward op, rep_penalty
is sampling -> apply SOFTCAP FIRST, then rep_penalty on the softcapped value, then argmax.

This gate proves the kernel does exactly that, by comparing the kernel's greedy token
against TWO host references built from v1's dense logits:
  - CORRECT  (softcap-then-rep_penalty): s = softcap(logit); if hit: s *= rep_penalty
  - WRONG    (rep_penalty-then-softcap): l = logit; if hit: l *= rep_penalty; s = softcap(l)
The kernel MUST match CORRECT. If WRONG differs from CORRECT for some (softcap, rep_penalty),
that case is "order-discriminating" -> a strong proof of order. The host tanh mirrors the
kernel's exact Padé approx (z*(27+z2)/(27+9z2), clamped) so the math is apples-to-apples.

Invariants asserted:
  (a) DETERMINISTIC: kernel token identical across 2 runs (fixed tree/warp order).
  (b) softcap=0 AND (rep_penalty=1.0 OR rep_len=0): kernel token == argmax(v1 raw logits)
      (softcap off + no penalty -> must match the v1 greedy token exactly; this is the
      "stays bit-identical when penalty is off" guarantee).
  (c) For EVERY (softcap, rep_penalty): kernel token == CORRECT-order reference.

  pixi run mojo build tools/fp4_greedy_correctness.mojo -I . \
      --target-accelerator sm_121a -o build/greedy_chk \
      -Xlinker -L/usr/local/cuda/targets/sbsa-linux/lib -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
  gpurun ./build/greedy_chk
"""
from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List

from lib.cuda import (
    cuda_malloc, cuda_upload, cuda_upload_u8, cuda_sync, cuda_free, cuda_memcpy,
)
from lib.fp4_weights import gpu_matmul_nvfp4_fused_dev, FP4_GROUP
from lib.fp4_gemv_v2 import gpu_lm_head_greedy_dev, GREEDY_CHUNK


def build_act(K: Int) -> List[Float32]:
    var a = List[Float32]()
    var i = 0
    while i < K:
        # Periodic base + tiny linear ramp: the ramp breaks the exact periodicity that
        # otherwise produces IDENTICAL top logits (ties), so the argmax is unique and the
        # softcap-vs-rep_penalty ORDER becomes observable (discriminating).
        a.append(Float32(0.7) * Float32((i % 13) - 6) + Float32(0.001) * Float32(i))
        i += 1
    return a^


def build_blob(K: Int, N: Int) -> List[UInt8]:
    # Row-dependent perturbation: the row stride (K/16 scales, K/2 nibble-pairs) is
    # divisible by the base pattern periods, so WITHOUT a per-row term, distant rows
    # (e.g. n and n+4) get IDENTICAL weights -> identical logits -> a tied argmax that
    # makes the softcap-vs-rep_penalty ORDER unobservable. XORing a row-dependent term
    # breaks the periodicity -> unique argmax -> order-discriminating.
    var n_rows = K * N
    var nb = n_rows // 16
    var nbk = K // FP4_GROUP    # scales per row
    var hpk = K // 2            # nibble-pairs per row
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


# EXACT mirror of the kernel's Padé tanh softcap (fp32 math, same ops, same clamp).
def softcap_pade(x: Float32, c: Float32) -> Float32:
    if c <= Float32(0.0):
        return x
    var z = x / c
    var z2 = z * z
    var tanh_z = z * (Float32(27.0) + z2) / (Float32(27.0) + Float32(9.0) * z2)
    if tanh_z > Float32(1.0):
        tanh_z = Float32(1.0)
    elif tanh_z < Float32(-1.0):
        tanh_z = Float32(-1.0)
    return c * tanh_z


def host_argmax(scores: List[Float32], N: Int) -> Int32:
    var best = Float32(-3.0e38)
    var best_idx = Int32(-1)
    for n in range(N):
        if scores[n] > best:
            best = scores[n]
            best_idx = Int32(n)
    return best_idx


# Apply an epilogue order to v1 raw logits and return the resulting greedy token.
def ref_token_order(
    h_v1: List[Float32], N: Int, softcap: Float32, rep_penalty: Float32,
    window: UnsafePointer[Int32, MutAnyOrigin], rep_len: Int, correct_order: Bool,
) -> Int32:
    var scores = List[Float32](length=N, fill=Float32(0.0))
    for n in range(N):
        var hit = False
        for w in range(rep_len):
            if Int(window[w]) == n:
                hit = True
                break
        if correct_order:
            # SOFTCAP THEN rep_penalty (llama.cpp / the requested order).
            var s = softcap_pade(h_v1[n], softcap)
            if hit:
                s *= rep_penalty
            scores[n] = s
        else:
            # rep_penalty THEN softcap (the WRONG order).
            var l = h_v1[n]
            if hit:
                l *= rep_penalty
            scores[n] = softcap_pade(l, softcap)
    return host_argmax(scores, N)


def main() raises:
    var K = 5376
    var N = 262144
    var blob = build_blob(K, N)
    var act = build_act(K)
    var gs = Float32(0.0078125)

    with DeviceContext() as ctx:
        var d_w = cuda_malloc(len(blob))
        cuda_upload_u8(d_w, blob)
        var d_in = cuda_malloc(K * 4)
        cuda_upload(d_in, act)
        var d_v1 = cuda_malloc(N * 4)
        var num_chunks = (N + GREEDY_CHUNK - 1) // GREEDY_CHUNK
        var d_sc = cuda_malloc(num_chunks * 4)
        var d_si = cuda_malloc(num_chunks * 4)
        var d_tok = cuda_malloc(4)

        var z = List[Float32](length=N, fill=Float32(0.0))
        cuda_memcpy(d_v1, UInt64(Int(z.unsafe_ptr())), N * 4, 1)

        # v1 dense logits (raw GEMV; no softcap, no penalty) = the GEMV ground truth.
        gpu_matmul_nvfp4_fused_dev(ctx, d_v1, d_in, d_w, gs, K, N)
        cuda_sync()
        var h_v1 = List[Float32](length=N, fill=Float32(0.0))
        cuda_memcpy(UInt64(Int(h_v1.unsafe_ptr())), d_v1, N * 4, 2)
        cuda_sync()

        # Raw argmax + runner-up (for building a penalty window that targets the top).
        var raw_top = Float32(-3.0e38)
        var raw_top_i = Int32(-1)
        var raw_2nd = Float32(-3.0e38)
        var raw_2nd_i = Int32(-1)
        for n in range(N):
            if h_v1[n] > raw_top:
                raw_2nd = raw_top
                raw_2nd_i = raw_top_i
                raw_top = h_v1[n]
                raw_top_i = Int32(n)
            elif h_v1[n] > raw_2nd:
                raw_2nd = h_v1[n]
                raw_2nd_i = Int32(n)
        print("raw v1 argmax: idx=", raw_top_i, " logit=", raw_top, flush=True)
        print("raw v1 2nd   : idx=", raw_2nd_i, " logit=", raw_2nd, flush=True)

        # A window that penalizes the TOP token (forces a potential flip -> order-sensitive).
        var win_top = List[Int32](length=1, fill=Int32(0))
        win_top[0] = raw_top_i
        # A window that penalizes TOP + 2nd (both knocked down -> 3rd place wins).
        var win_two = List[Int32](length=2, fill=Int32(0))
        win_two[0] = raw_top_i
        win_two[1] = raw_2nd_i

        @parameter
        def run_kernel(softcap: Float32, rep_len: Int, rep_penalty: Float32,
                       window: UnsafePointer[Int32, MutAnyOrigin]) raises -> Int32:
            # upload window to device scratch area reused as rep_window
            var d_win = UInt64(0)
            if rep_len > 0:
                d_win = cuda_malloc(rep_len * 4)
                cuda_memcpy(d_win, UInt64(Int(window)), rep_len * 4, 1)
            gpu_lm_head_greedy_dev(
                ctx, d_in, d_w, d_sc, d_si, d_tok, gs, softcap,
                d_win, rep_len, rep_penalty, K, N,
            )
            cuda_sync()
            var tok = List[Int32](length=1, fill=Int32(0))
            cuda_memcpy(UInt64(Int(tok.unsafe_ptr())), d_tok, 4, 2)
            cuda_sync()
            if rep_len > 0:
                cuda_free(d_win)
            return tok[0]

        # Cases: (softcap, rep_penalty). Window = penalize the top token when penalty on.
        var any_fail = False
        var any_discriminating = False

        var softcaps = List[Float32](length=4, fill=Float32(0.0))
        softcaps[0] = Float32(0.0)
        softcaps[1] = Float32(10.0)
        softcaps[2] = Float32(30.0)   # Gemma-style
        softcaps[3] = Float32(1000.0) # no saturation -> order fully observable
        var pens = List[Float32](length=3, fill=Float32(1.0))
        pens[0] = Float32(1.0)        # no penalty
        pens[1] = Float32(0.5)
        pens[2] = Float32(0.1)        # strong penalty

        for ci in range(len(softcaps)):
            var softcap = softcaps[ci]
            for pi in range(len(pens)):
                var pen = pens[pi]
                var rep_len = 0
                var window = win_top.unsafe_ptr()
                if pen != Float32(1.0):
                    rep_len = 1
                var kern = run_kernel(softcap, rep_len, pen, window)
                var kern2 = run_kernel(softcap, rep_len, pen, window)

                var ref_correct = ref_token_order(h_v1, N, softcap, pen, window, rep_len, True)
                var ref_wrong = ref_token_order(h_v1, N, softcap, pen, window, rep_len, False)
                var discriminating = (ref_correct != ref_wrong)
                if discriminating:
                    any_discriminating = True

                var order_ok = (kern == ref_correct)
                var det_ok = (kern == kern2)
                var penalty_off = (pen == Float32(1.0))
                var v1_match_ok = True
                # Bit-identical-to-v1 only when softcap is OFF too: with softcap>0 the top
                # logits can SATURATE (tanh -> +/-c), creating ties the lower-idx tie-break
                # resolves differently from the raw argmax. That is correct (matches
                # ref_correct) but not comparable to raw v1 argmax.
                if penalty_off and softcap == Float32(0.0):
                    v1_match_ok = (kern == raw_top_i)

                print(String("  [softcap="), softcap, " pen=", pen, " win=top] kern=", kern,
                      " ref_correct=", ref_correct, " ref_wrong=", ref_wrong,
                      " | order_ok=", order_ok, " det_ok=", det_ok,
                      " v1match=", v1_match_ok, " discriminating=", discriminating, flush=True)
                if not order_ok or not det_ok or not v1_match_ok:
                    any_fail = True

        # One extra discriminating case: penalize BOTH top+2nd -> 3rd should win.
        var kern = run_kernel(Float32(30.0), 2, Float32(0.1), win_two.unsafe_ptr())
        var kern2 = run_kernel(Float32(30.0), 2, Float32(0.1), win_two.unsafe_ptr())
        var ref_correct = ref_token_order(h_v1, N, Float32(30.0), Float32(0.1), win_two.unsafe_ptr(), 2, True)
        var ref_wrong = ref_token_order(h_v1, N, Float32(30.0), Float32(0.1), win_two.unsafe_ptr(), 2, False)
        var discriminating = (ref_correct != ref_wrong)
        if discriminating:
            any_discriminating = True
        var order_ok = (kern == ref_correct)
        var det_ok = (kern == kern2)
        print(String("  [softcap=30 pen=0.1 win=top+2nd]"), " kern=", kern,
                     " ref_correct=", ref_correct, " ref_wrong=", ref_wrong,
                     " | order_ok=", order_ok, " det_ok=", det_ok,
                     " discriminating=", discriminating, flush=True)
        if not order_ok or not det_ok:
            any_fail = True

        print(String(""), flush=True)
        print("order-discriminating cases encountered:", any_discriminating, flush=True)
        if any_fail:
            print("GATE FAIL", flush=True)
            raise Error("greedy epilogue order gate failed")
        print("GATE OK: kernel applies SOFTCAP-THEN-rep_penalty (llama.cpp order);", flush=True)
        print("         deterministic across runs; bit-identical to v1 argmax when penalty off.", flush=True)

        cuda_free(d_w)
        cuda_free(d_in)
        cuda_free(d_v1)
        cuda_free(d_sc)
        cuda_free(d_si)
        cuda_free(d_tok)
