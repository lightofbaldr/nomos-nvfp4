"""decode_argmax.mojo — Perf Lever A: on-device softcap → argmax (greedy fast path).

Returns the chosen token on-device so the decode FFI returns 4 bytes instead of
copying VOCAB=262144 fp32 logits (~1 MB) to host every token (the per-token
overhead profiled at engine_decode.mojo:483 `cuda_download(logits,
self.d_lmhead_logits, VOCAB)`).

A1 = device argmax ONLY. The serve's greedy sampler (`pyhost/host.py _pick`) is pure
`np.argmax` over the softcapped logits — NO rep-penalty, NO masking ("removed by
design", host.py docstring). So we match it exactly with:
  1. softcap:  logits = 30*tanh(logits/30)   (== lib/sampling.final_logit_softcap;
               d_lmhead_logits is PRE-softcap, host softcaps after the copy).
  2. argmax:   strict `>` keeps the FIRST/lowest index on ties == lib/sampling.argmax
               == np.argmax. Softcap fused-before-argmax so saturation ties match.

Greedy-only (temp<=0). Sampling (temp>0) / grammar steps keep the full-logits host
path (the serve falls back to nomos_decode_step for those).
"""
from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim, grid_dim
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from std.math import exp, tanh
from layout import row_major
from layout import stack_allocation
from lib.cuda import (
    cuda_malloc,
    cuda_free,
    cuda_download_i32,
    cuda_memcpy_async,
    cuda_stream_sync,
)
from lib.q4_weights import _as_f32_ptr   # UInt64 addr -> UnsafePointer[Float32]


def _as_i32_ptr(addr: UInt64) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(addr))


# Argmax (single block, shared-mem tree reduce; lowest-index on ties).
alias ARGMAX_TPB = 1024


def argmax_logits_kernel(
    out_token: UnsafePointer[Int32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
    cap: Float32,
    multiplier: Float32,
):
    # Mojo dev2026080106: bare Int/UInt no longer conform to DevicePassable — GPU
    # kernel args must be fixed-width. Take Int32 and widen once here so the body
    # (and every index expression below) is unchanged.
    var n = Int(n_arg)
    # Parallel: fused softcap + argmax. One block of ARGMAX_TPB threads strided over
    # VOCAB, then a shared-mem tree reduce. Strict `>` everywhere keeps the FIRST
    # (lowest) index on ties == lib/sampling.argmax == np.argmax. (Single-thread
    # version parity-verified; this is the fast equivalent.)
    var score_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[ARGMAX_TPB]())
    var idx_s = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[ARGMAX_TPB]())
    var tid = Int(thread_idx.x)
    var best = Float32(-3.4e38)
    var best_i = Int32(0)
    var i = tid
    while i < n:
        var v = logits[i] * multiplier
        if cap > Float32(0.0):
            v = cap * tanh(v / cap)
        if v > best:
            best = v
            best_i = Int32(i)
        i += ARGMAX_TPB
    score_s[tid] = best
    idx_s[tid] = best_i
    barrier()
    var stride = ARGMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            # `>` so the lower thread-slot (holding the lower index for an equal
            # max from the strided scan) wins ties -> lowest index, like np.argmax.
            if score_s[tid + stride] > score_s[tid]:
                score_s[tid] = score_s[tid + stride]
                idx_s[tid] = idx_s[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_token[0] = idx_s[0]


def argmax_conf_logits_kernel(
    out_token: UnsafePointer[Int32, MutAnyOrigin],
    out_stats: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
    cap: Float32,
):
    # See argmax_logits_kernel: Int32 arg widened once (DevicePassable, dev2026080106).
    var n = Int(n_arg)
    # out_stats[0] = top-1 probability after softmax(logits)
    # out_stats[1] = top-1 probability minus top-2 probability
    var score_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[ARGMAX_TPB]())
    var second_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[ARGMAX_TPB]())
    var sum_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[ARGMAX_TPB]())
    var idx_s = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[ARGMAX_TPB]())

    var tid = Int(thread_idx.x)
    var best = Float32(-3.4e38)
    var second = Float32(-3.4e38)
    var best_i = Int32(0)
    var i = tid
    while i < n:
        var v = logits[i]
        if cap > Float32(0.0):
            v = cap * tanh(v / cap)
        if v > best:
            second = best
            best = v
            best_i = Int32(i)
        elif v > second:
            second = v
        i += ARGMAX_TPB

    score_s[tid] = best
    second_s[tid] = second
    idx_s[tid] = best_i
    barrier()

    var stride = ARGMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            var other_best = score_s[tid + stride]
            var other_second = second_s[tid + stride]
            if other_best > score_s[tid]:
                var new_second = score_s[tid]
                if other_second > new_second:
                    new_second = other_second
                score_s[tid] = other_best
                second_s[tid] = new_second
                idx_s[tid] = idx_s[tid + stride]
            else:
                if other_best > second_s[tid]:
                    second_s[tid] = other_best
                if other_second > second_s[tid]:
                    second_s[tid] = other_second
        barrier()
        stride //= 2

    var max_v = score_s[0]
    var local_sum = Float32(0.0)
    i = tid
    while i < n:
        var v = logits[i]
        if cap > Float32(0.0):
            v = cap * tanh(v / cap)
        local_sum += exp(v - max_v)
        i += ARGMAX_TPB
    sum_s[tid] = local_sum
    barrier()

    stride = ARGMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            sum_s[tid] = sum_s[tid] + sum_s[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        var denom = sum_s[0]
        var p1 = Float32(0.0)
        var p2 = Float32(0.0)
        if denom > Float32(0.0):
            p1 = Float32(1.0) / denom
            p2 = exp(second_s[0] - score_s[0]) / denom
        out_token[0] = idx_s[0]
        out_stats[0] = p1
        out_stats[1] = p1 - p2


# ── Host dispatch: softcap -> argmax on d_logits, return the token ───────────
def device_decode_token(
    ctx: DeviceContext,
    d_logits: UInt64,        # [VOCAB] fp32 GPU logits (self.d_lmhead_logits), PRE-softcap
    vocab: Int,
    pen_ids: List[Int32],    # unused in A1 (greedy has no rep-penalty); kept for ABI compat
    rep_penalty: Float32,    # unused in A1
    softcap: Float32 = 30.0,
    multiplier: Float32 = 1.0,
) raises -> Int:
    _ = pen_ids
    _ = rep_penalty
    var lp = _as_f32_ptr(d_logits)

    # Flush + wait the Modular ctx queue so the in-flight lm_head GEMV has finished
    # writing d_lmhead_logits before the argmax reads it. (Raw cuda_sync does NOT
    # coordinate with the Modular DeviceContext queue -> argmax raced on a stale/zero
    # buffer and returned index 0. ctx.synchronize() is the right barrier.)
    ctx.synchronize()

    # fused softcap+argmax (one block, parallel shared-mem reduce)
    var d_tok = cuda_malloc(4)
    var k_am = ctx.compile_function[argmax_logits_kernel]()
    ctx.enqueue_function(k_am, _as_i32_ptr(d_tok), lp, Int32(vocab), softcap, multiplier,
                         grid_dim=1, block_dim=ARGMAX_TPB)
    ctx.synchronize()

    var host_tok = List[Int32]()
    host_tok.append(0)
    cuda_download_i32(host_tok, d_tok, 1)
    cuda_free(d_tok)
    return Int(host_tok[0])


def device_decode_token_into_stream(
    ctx: DeviceContext,
    d_logits: UInt64,
    vocab: Int,
    d_tok: UInt64,
    softcap: Float32 = 30.0,
    multiplier: Float32 = 1.0,
) raises -> Int:
    var lp = _as_f32_ptr(d_logits)

    var k_am = ctx.compile_function[argmax_logits_kernel]()
    ctx.enqueue_function(
        k_am,
        _as_i32_ptr(d_tok),
        lp,
        Int32(vocab),
        softcap,
        multiplier,
        grid_dim=1,
        block_dim=ARGMAX_TPB,
    )

    var host_tok = List[Int32]()
    host_tok.append(0)
    cuda_memcpy_async(UInt64(Int(host_tok.unsafe_ptr())), d_tok, 4, 2, ctx)
    cuda_stream_sync(ctx)
    return Int(host_tok[0])


def device_decode_token_conf_into(
    ctx: DeviceContext,
    d_logits: UInt64,
    vocab: Int,
    d_tok: UInt64,
    d_stats: UInt64,
    mut confs: List[Float32],
    mut gaps: List[Float32],
    softcap: Float32 = 30.0,
) raises -> Int:
    var lp = _as_f32_ptr(d_logits)

    var k_am = ctx.compile_function[argmax_conf_logits_kernel]()
    ctx.enqueue_function(
        k_am,
        _as_i32_ptr(d_tok),
        _as_f32_ptr(d_stats),
        lp,
        Int32(vocab),
        softcap,
        grid_dim=1,
        block_dim=ARGMAX_TPB,
    )

    var host_tok = List[Int32]()
    host_tok.append(0)
    var host_stats = List[Float32]()
    host_stats.append(Float32(0.0))
    host_stats.append(Float32(0.0))
    cuda_memcpy_async(UInt64(Int(host_tok.unsafe_ptr())), d_tok, 4, 2, ctx)
    cuda_memcpy_async(UInt64(Int(host_stats.unsafe_ptr())), d_stats, 8, 2, ctx)
    cuda_stream_sync(ctx)
    confs.append(host_stats[0])
    gaps.append(host_stats[1])
    return Int(host_tok[0])
