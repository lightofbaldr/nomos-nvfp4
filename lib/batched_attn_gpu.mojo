"""Batched prefill attention — full-materialized, causal, one shot per layer.

Replaces the O(S²) per-token attention loop in engine_prefill (S calls ×
~3 cuBLAS launches each = launch-bound at large context). Here the WHOLE
prompt's attention for a layer is 5 launches: transpose-Q, batched Q·Kᵀ,
causal softmax, batched scores·V, transpose-out.

The GQA layout problem + fix (see internal notes):
  d_q_b is [S, nkv, kvg, l_hd] (token-major). A kv-head's kvg*S query rows
  are interleaved across tokens, so a strided GEMM can't read them. We first
  block-transpose Q to [nkv, S, kvg*l_hd] → then kv-head h is the contiguous
  block [S*kvg, l_hd], a clean strided-GEMM batch. Scores are [nkv, S*kvg, S];
  within a kv-head, row r = s*kvg + g maps to query position s = r // kvg, so
  the causal mask is "row r attends to keys 0..s". Output transposes back.

Parity: identical math to the per-token fp32 path for keys 0..s (same Q[s],
K[0..s], plain softmax); masked keys are zeroed so they drop out of scores·V.
Not byte-identical (cuBLAS reduces N=S vs N=s+1 differently) — gate on greedy
tokens / residual cosine, not bytes.
"""

from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from max.gpu import barrier
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from max.gpu.memory import AddressSpace
from std.memory import UnsafePointer, bitcast
from std.math import exp
from layout import row_major
from layout import stack_allocation

from lib.cublas import cublas_sgemm_strided_batched_nt, cublas_sgemm_strided_batched_nn
from lib.cublas import cublas_gemm_ex_sb_bf16_nt, cublas_gemm_ex_sb_bf16_nn
from lib.cast_gpu_mojo import gpu_fp32_to_bf16_mojo


comptime MAX_WARPS = 32


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_u16_ptr(addr: UInt64) -> UnsafePointer[UInt16, MutAnyOrigin]:
    return UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=Int(addr))


# ─────────────────────────────────────────────────────────────────────────
# Batched KV-append: cast all S tokens' FP32 K/V -> BF16 and scatter into the
# BF16 cache at positions 0..S-1, in ONE launch per layer (was S per-token
# append_kv_gpu_bf16_dev calls = the prefill launch-overhead wall, ~78% of
# prefill time). Source d_src is token-major [S, nkv*l_hd]; cache is
# [nkv, max_seq, cache_hd]. Writes l_hd dims per (token, head) — identical
# bytes to the per-token path. 2026-06-15 prefill perf.
# ─────────────────────────────────────────────────────────────────────────
def append_kv_batch_bf16_kernel(
    src_k: UnsafePointer[Float32, MutAnyOrigin],
    src_v: UnsafePointer[Float32, MutAnyOrigin],
    dst_k: UnsafePointer[UInt16, MutAnyOrigin],
    dst_v: UnsafePointer[UInt16, MutAnyOrigin],
    n_s_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, l_kvd_arg: Int32, max_seq_arg: Int32, cache_hd_arg: Int32,
    base_pos_arg: Int32,
):
    var n_s = Int(n_s_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var l_kvd = Int(l_kvd_arg)
    var max_seq = Int(max_seq_arg)
    var cache_hd = Int(cache_hd_arg)
    var base_pos = Int(base_pos_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = n_s * nkv * l_hd
    if idx < total:
        var d = idx % l_hd
        var rem = idx // l_hd
        var h = rem % nkv
        var s = rem // nkv
        var src_off = s * l_kvd + h * l_hd + d
        var dst_off = h * max_seq * cache_hd + (base_pos + s) * cache_hd + d
        var kf = SIMD[DType.float32, 1](src_k[src_off]).cast[DType.bfloat16]()
        var vf = SIMD[DType.float32, 1](src_v[src_off]).cast[DType.bfloat16]()
        dst_k[dst_off] = bitcast[DType.uint16, 1](kf)[0]
        dst_v[dst_off] = bitcast[DType.uint16, 1](vf)[0]


# base_pos = cache position of the first token (0 for prefill; `pos` for the
# single decode token — fuses the 4-launch per-token append to 1).
def gpu_append_kv_batched_bf16(
    ctx: DeviceContext, d_src_k: UInt64, d_src_v: UInt64,
    d_dst_k_cache: UInt64, d_dst_v_cache: UInt64,
    n_s: Int, nkv: Int, l_hd: Int, l_kvd: Int, max_seq: Int, cache_hd: Int,
    base_pos: Int = 0,
) raises:
    var total = n_s * nkv * l_hd
    var threads = 256
    var blocks = (total + threads - 1) // threads
    var k = ctx.compile_function[append_kv_batch_bf16_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_src_k), _as_f32_ptr(d_src_v),
        _as_u16_ptr(d_dst_k_cache), _as_u16_ptr(d_dst_v_cache),
        Int32(n_s), Int32(nkv), Int32(l_hd), Int32(l_kvd), Int32(max_seq), Int32(cache_hd), Int32(base_pos),
        grid_dim=blocks, block_dim=threads,
    )


# ─────────────────────────────────────────────────────────────────────────
# Block transpose: [dim0, dim1, blk] -> [dim1, dim0, blk]
# Swaps the outer two dims, keeping each blk-sized inner row contiguous.
# Used for Q  (dim0=S,   dim1=nkv) and the inverse for attn-out (dim0=nkv, dim1=S).
# ─────────────────────────────────────────────────────────────────────────
def block_transpose_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    dim0_arg: Int32,
    dim1_arg: Int32,
    blk_arg: Int32,
):
    var dim0 = Int(dim0_arg)
    var dim1 = Int(dim1_arg)
    var blk = Int(blk_arg)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = dim0 * dim1 * blk
    if t >= total:
        return
    var stripe = dim1 * blk
    var i = t // stripe
    var rem = t % stripe
    var j = rem // blk
    var b = rem % blk
    dst[(j * dim0 + i) * blk + b] = src[t]


def gpu_block_transpose(
    ctx: DeviceContext, d_src: UInt64, d_dst: UInt64, dim0: Int, dim1: Int, blk: Int
) raises:
    var total = dim0 * dim1 * blk
    if total <= 0:
        return
    var threads = 256
    var blocks = (total + threads - 1) // threads
    var k = ctx.compile_function[block_transpose_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_src), _as_f32_ptr(d_dst), Int32(dim0), Int32(dim1), Int32(blk),
        grid_dim=blocks, block_dim=threads,
    )


# ─────────────────────────────────────────────────────────────────────────
# Causal (+ optional sliding-window) softmax over scores [nkv, S*kvg, S].
# One block per row. Global row index `row` in [0, nkv*S*kvg).
#   within-head row r = row % (S*kvg);  query pos s = r // kvg.
# Causal valid keys = [lo, s] where lo=0 for full-attention layers (window<=0)
# and lo=max(0, s-window+1) for sliding layers (Gemma-4 window=1024). Cols
# outside [lo, s] are zeroed so they vanish in the subsequent scores·V GEMM.
# ─────────────────────────────────────────────────────────────────────────
def softmax_causal_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],
    total_rows_arg: Int32,
    seq_arg: Int32,
    kvg_arg: Int32,
    window_arg: Int32,
):
    var total_rows = Int(total_rows_arg)
    var seq = Int(seq_arg)
    var kvg = Int(kvg_arg)
    var window = Int(window_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var row = Int(block_idx.x)
    if row >= total_rows:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)

    var rows_per_head = seq * kvg
    var r = row % rows_per_head
    var s = r // kvg
    var lo = 0                      # causal lower bound (sliding window for local layers)
    if window > 0:
        var cand = s - window + 1
        if cand > 0:
            lo = cand
    var row_off = row * seq

    # ── Phase 1: block max over the valid window [lo, s] ───────────────────
    var local_max = Scalar[DType.float32](-3.4e38)
    var i = lo + tid
    while i <= s:
        var v = Scalar[DType.float32](scores[row_off + i])
        if v > local_max:
            local_max = v
        i += n_threads

    var warp_off = WARP_SIZE // 2
    while warp_off > 0:
        var other = shuffle_xor(local_max, UInt32(warp_off))
        if other > local_max:
            local_max = other
        warp_off //= 2

    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = local_max
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](-3.4e38)
        if lane < n_warps:
            partial = smem[lane]
        var off2 = WARP_SIZE // 2
        while off2 > 0:
            var other = shuffle_xor(partial, UInt32(off2))
            if other > partial:
                partial = other
            off2 //= 2
        if lane == 0:
            smem[0] = partial
    barrier()
    var row_max = smem[0]

    # ── Phase 2: exp(x - max) and sum over the valid window [lo, s] ────────
    var local_sum = Scalar[DType.float32](0.0)
    var j = lo + tid
    while j <= s:
        var v = exp(Scalar[DType.float32](scores[row_off + j]) - row_max)
        scores[row_off + j] = Float32(v)
        local_sum += v
        j += n_threads

    var warp_total = local_sum
    var so = WARP_SIZE // 2
    while so > 0:
        warp_total += shuffle_xor(warp_total, UInt32(so))
        so //= 2
    if lane == 0:
        smem[wid] = warp_total
    barrier()
    if wid == 0:
        var partial2 = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial2 = smem[lane]
        var so2 = WARP_SIZE // 2
        while so2 > 0:
            partial2 += shuffle_xor(partial2, UInt32(so2))
            so2 //= 2
        if lane == 0:
            smem[0] = partial2
    barrier()
    var row_sum = smem[0]
    var inv = Scalar[DType.float32](1.0) / row_sum

    # ── Phase 3: normalize the window [lo, s], zero everything else ────────
    var k = tid
    while k < seq:
        if k >= lo and k <= s:
            scores[row_off + k] = Float32(Scalar[DType.float32](scores[row_off + k]) * inv)
        else:
            scores[row_off + k] = Float32(0.0)
        k += n_threads


def gpu_softmax_causal(
    ctx: DeviceContext, d_scores: UInt64, total_rows: Int, seq: Int, kvg: Int,
    window: Int,
) raises:
    if total_rows <= 0 or seq <= 0:
        return
    var threads = 256
    if seq < 256:
        threads = ((seq + 31) // 32) * 32
        if threads < 32:
            threads = 32
    if threads > 1024:
        threads = 1024
    var k = ctx.compile_function[softmax_causal_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_scores), Int32(total_rows), Int32(seq), Int32(kvg), Int32(window),
        grid_dim=total_rows, block_dim=threads,
    )


def softmax_causal_prefix_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],
    total_rows_arg: Int32,
    S_arg: Int32,
    key_len_arg: Int32,
    kvg_arg: Int32,
    window_arg: Int32,
    query_base_pos_arg: Int32,
    key_base_pos_arg: Int32,
):
    var total_rows = Int(total_rows_arg)
    var S = Int(S_arg)
    var key_len = Int(key_len_arg)
    var kvg = Int(kvg_arg)
    var window = Int(window_arg)
    var query_base_pos = Int(query_base_pos_arg)
    var key_base_pos = Int(key_base_pos_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var row = Int(block_idx.x)
    if row >= total_rows:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)

    var rows_per_head = S * kvg
    var r = row % rows_per_head
    var s = r // kvg
    var q_abs = query_base_pos + s
    var hi = q_abs - key_base_pos
    if hi >= key_len:
        hi = key_len - 1
    var lo_abs = 0
    if window > 0:
        var cand = q_abs - window + 1
        if cand > 0:
            lo_abs = cand
    var lo = lo_abs - key_base_pos
    if lo < 0:
        lo = 0
    var row_off = row * key_len

    if hi < lo:
        var z = tid
        while z < key_len:
            scores[row_off + z] = Float32(0.0)
            z += n_threads
        return

    var local_max = Scalar[DType.float32](-3.4e38)
    var i = lo + tid
    while i <= hi:
        var v = Scalar[DType.float32](scores[row_off + i])
        if v > local_max:
            local_max = v
        i += n_threads

    var warp_off = WARP_SIZE // 2
    while warp_off > 0:
        var other = shuffle_xor(local_max, UInt32(warp_off))
        if other > local_max:
            local_max = other
        warp_off //= 2

    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    if lane == 0:
        smem[wid] = local_max
    barrier()

    var n_warps = (n_threads + WARP_SIZE - 1) // WARP_SIZE
    if wid == 0:
        var partial = Scalar[DType.float32](-3.4e38)
        if lane < n_warps:
            partial = smem[lane]
        var off2 = WARP_SIZE // 2
        while off2 > 0:
            var other = shuffle_xor(partial, UInt32(off2))
            if other > partial:
                partial = other
            off2 //= 2
        if lane == 0:
            smem[0] = partial
    barrier()
    var row_max = smem[0]

    var local_sum = Scalar[DType.float32](0.0)
    var j = lo + tid
    while j <= hi:
        var v = exp(Scalar[DType.float32](scores[row_off + j]) - row_max)
        scores[row_off + j] = Float32(v)
        local_sum += v
        j += n_threads

    var warp_total = local_sum
    var so = WARP_SIZE // 2
    while so > 0:
        warp_total += shuffle_xor(warp_total, UInt32(so))
        so //= 2
    if lane == 0:
        smem[wid] = warp_total
    barrier()
    if wid == 0:
        var partial2 = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial2 = smem[lane]
        var so2 = WARP_SIZE // 2
        while so2 > 0:
            partial2 += shuffle_xor(partial2, UInt32(so2))
            so2 //= 2
        if lane == 0:
            smem[0] = partial2
    barrier()
    var row_sum = smem[0]
    var inv = Scalar[DType.float32](1.0) / row_sum

    var k = tid
    while k < key_len:
        if k >= lo and k <= hi:
            scores[row_off + k] = Float32(Scalar[DType.float32](scores[row_off + k]) * inv)
        else:
            scores[row_off + k] = Float32(0.0)
        k += n_threads


def gpu_softmax_causal_prefix(
    ctx: DeviceContext,
    d_scores: UInt64,
    total_rows: Int,
    S: Int,
    key_len: Int,
    kvg: Int,
    window: Int,
    query_base_pos: Int,
    key_base_pos: Int,
) raises:
    if total_rows <= 0 or S <= 0 or key_len <= 0:
        return
    var threads = 256
    if key_len < 256:
        threads = ((key_len + 31) // 32) * 32
        if threads < 32:
            threads = 32
    if threads > 1024:
        threads = 1024
    var k = ctx.compile_function[softmax_causal_prefix_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_scores), Int32(total_rows), Int32(S), Int32(key_len), Int32(kvg), Int32(window),
        Int32(query_base_pos), Int32(key_base_pos),
        grid_dim=total_rows, block_dim=threads,
    )


# ─────────────────────────────────────────────────────────────────────────
# Full-materialized causal prefill attention for one layer (fp32).
# d_q_b   : [S, nkv, kvg, l_hd] (token-major)            — input
# d_q_t   : [nkv, S, kvg*l_hd] scratch (>= S*l_qd elems)
# d_scores: [nkv, S*kvg, S]    scratch (>= nh*S*S elems)
# d_attn_t: [nkv, S, kvg*l_hd] scratch (>= S*l_qd elems)
# d_attn_b: [S, nkv, kvg, l_hd] output (>= S*l_qd elems) — written for the layer
# ─────────────────────────────────────────────────────────────────────────
def attention_prefill_batched_fp32(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_b: UInt64,
    d_q_t: UInt64,
    d_scores: UInt64,
    d_attn_t: UInt64,
    d_attn_b: UInt64,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    kvg: Int,
    window: Int,
) raises:
    var blk = kvg * l_hd

    # 1. Transpose Q [S, nkv, blk] -> [nkv, S, blk] (per-kv-head contiguous).
    gpu_block_transpose(ctx, d_q_b, d_q_t, S, nkv, blk)

    # 2. Q·Kᵀ : per kv-head A[S*kvg, l_hd] @ K[S, l_hd]ᵀ -> scores[S*kvg, S].
    cublas_sgemm_strided_batched_nt(
        handle,
        d_q_t,           l_hd,     S * kvg * l_hd,        # A
        d_k_cache_layer, cache_hd, max_seq * cache_hd,    # B (all S keys)
        d_scores,        S,        S * kvg * S,           # C
        S * kvg, l_hd, S,
        nkv,
    )

    # 3. Causal (+ sliding-window for local layers) softmax.
    gpu_softmax_causal(ctx, d_scores, nh * S, S, kvg, window)

    # 4. scores·V : per kv-head A[S*kvg, S] @ V[S, l_hd] -> out[S*kvg, l_hd].
    cublas_sgemm_strided_batched_nn(
        handle,
        d_scores,        S,        S * kvg * S,           # A
        d_v_cache_layer, cache_hd, max_seq * cache_hd,    # B
        d_attn_t,        l_hd,     S * kvg * l_hd,        # C
        S * kvg, S, l_hd,
        nkv,
    )

    # 5. Transpose out [nkv, S, blk] -> [S, nkv, blk] = d_attn_b.
    gpu_block_transpose(ctx, d_attn_t, d_attn_b, nkv, S, blk)


def attention_prefill_prefix_batched_fp32(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_b: UInt64,
    d_q_t: UInt64,
    d_scores: UInt64,
    d_attn_t: UInt64,
    d_attn_b: UInt64,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    key_len: Int,
    kvg: Int,
    window: Int,
    query_base_pos: Int,
    key_base_pos: Int,
) raises:
    var blk = kvg * l_hd

    gpu_block_transpose(ctx, d_q_b, d_q_t, S, nkv, blk)

    cublas_sgemm_strided_batched_nt(
        handle,
        d_q_t,           l_hd,     S * kvg * l_hd,
        d_k_cache_layer, cache_hd, max_seq * cache_hd,
        d_scores,        key_len,  S * kvg * key_len,
        S * kvg, l_hd, key_len,
        nkv,
    )

    gpu_softmax_causal_prefix(
        ctx, d_scores, nh * S, S, key_len, kvg, window,
        query_base_pos, key_base_pos,
    )

    cublas_sgemm_strided_batched_nn(
        handle,
        d_scores,        key_len,  S * kvg * key_len,
        d_v_cache_layer, cache_hd, max_seq * cache_hd,
        d_attn_t,        l_hd,     S * kvg * l_hd,
        S * kvg, key_len, l_hd,
        nkv,
    )

    gpu_block_transpose(ctx, d_attn_t, d_attn_b, nkv, S, blk)


# ─────────────────────────────────────────────────────────────────────────
# BF16 twin: Q·Kᵀ and scores·V run on bf16 tensor cores over the bf16 KV cache;
# transpose + causal softmax stay fp32 (softmax is numerically sensitive). Extra
# scratch: d_q_t_bf16 (>= S*l_qd bf16 elems), d_scores_bf16 (>= NH*S*S bf16 elems).
# cuBLAS strides are ELEMENT counts, so they match the fp32 path even though the
# bf16 cache is 2 bytes/elem.
# ─────────────────────────────────────────────────────────────────────────
def attention_prefill_batched_bf16(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_b: UInt64,
    d_q_t: UInt64,
    d_q_t_bf16: UInt64,
    d_scores: UInt64,
    d_scores_bf16: UInt64,
    d_attn_t: UInt64,
    d_attn_b: UInt64,
    d_k_cache_layer: UInt64,    # bf16 cache
    d_v_cache_layer: UInt64,    # bf16 cache
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    kvg: Int,
    window: Int,
) raises:
    var blk = kvg * l_hd

    # 1. Transpose Q [S, nkv, blk] -> [nkv, S, blk], then cast to bf16.
    gpu_block_transpose(ctx, d_q_b, d_q_t, S, nkv, blk)
    gpu_fp32_to_bf16_mojo(ctx, d_q_t, d_q_t_bf16, S * nkv * blk)

    # 2. Q·Kᵀ on bf16 tensor cores -> fp32 scores.
    cublas_gemm_ex_sb_bf16_nt(
        handle,
        d_q_t_bf16,      l_hd,     S * kvg * l_hd,
        d_k_cache_layer, cache_hd, max_seq * cache_hd,
        d_scores,        S,        S * kvg * S,
        S * kvg, l_hd, S,
        nkv,
    )

    # 3. Causal (+ sliding-window) softmax (fp32), then cast to bf16 for scores·V.
    gpu_softmax_causal(ctx, d_scores, nh * S, S, kvg, window)
    gpu_fp32_to_bf16_mojo(ctx, d_scores, d_scores_bf16, nh * S * S)

    # 4. scores·V on bf16 tensor cores -> fp32 attn.
    cublas_gemm_ex_sb_bf16_nn(
        handle,
        d_scores_bf16,   S,        S * kvg * S,
        d_v_cache_layer, cache_hd, max_seq * cache_hd,
        d_attn_t,        l_hd,     S * kvg * l_hd,
        S * kvg, S, l_hd,
        nkv,
    )

    # 5. Transpose out [nkv, S, blk] -> [S, nkv, blk] = d_attn_b.
    gpu_block_transpose(ctx, d_attn_t, d_attn_b, nkv, S, blk)


def attention_prefill_prefix_batched_bf16(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_b: UInt64,
    d_q_t: UInt64,
    d_q_t_bf16: UInt64,
    d_scores: UInt64,
    d_scores_bf16: UInt64,
    d_attn_t: UInt64,
    d_attn_b: UInt64,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    key_len: Int,
    kvg: Int,
    window: Int,
    query_base_pos: Int,
    key_base_pos: Int,
) raises:
    var blk = kvg * l_hd

    gpu_block_transpose(ctx, d_q_b, d_q_t, S, nkv, blk)
    gpu_fp32_to_bf16_mojo(ctx, d_q_t, d_q_t_bf16, S * nkv * blk)

    cublas_gemm_ex_sb_bf16_nt(
        handle,
        d_q_t_bf16,      l_hd,     S * kvg * l_hd,
        d_k_cache_layer, cache_hd, max_seq * cache_hd,
        d_scores,        key_len,  S * kvg * key_len,
        S * kvg, l_hd, key_len,
        nkv,
    )

    gpu_softmax_causal_prefix(
        ctx, d_scores, nh * S, S, key_len, kvg, window,
        query_base_pos, key_base_pos,
    )
    gpu_fp32_to_bf16_mojo(ctx, d_scores, d_scores_bf16, nh * S * key_len)

    cublas_gemm_ex_sb_bf16_nn(
        handle,
        d_scores_bf16,   key_len,  S * kvg * key_len,
        d_v_cache_layer, cache_hd, max_seq * cache_hd,
        d_attn_t,        l_hd,     S * kvg * l_hd,
        S * kvg, key_len, l_hd,
        nkv,
    )

    gpu_block_transpose(ctx, d_attn_t, d_attn_b, nkv, S, blk)
