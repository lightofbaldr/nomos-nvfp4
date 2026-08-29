"""EAGLE-3 drafter own-KV attention (persistent prefix, M=1, ≤KMAX-deep).

The drafter is a single Llama-style layer that runs autoregressively over ITS OWN
draft sequence — distinct from the MTP path, which shared the 31B's KV. Each draft
step p projects q/k/v from the recurred hidden, appends k/v at position p into a
tiny own-KV cache, attends causally over [0,p], and feeds o_proj → the next hidden.

Seam-facts (VERIFIED against eagle3-redhat/config.json transformer_layer_config):
  - model_type "llama", head_dim 256, num_attention_heads 32, num_key_value_heads 16
    → NH=32, NKV=16, kvg=2 (each KV head serves 2 Q heads, contiguous: kvh = h // kvg,
    matching the engine's int8 decode convention).
  - NO q_norm / k_norm tensors and NO query_pre_attn_scalar → standard Llama attention.
    *** This is THE difference from the 31B. *** The 31B folds its scale into q_norm
    (kq_scale=1.0 in-kernel); this drafter has nothing to fold into, so the attend
    MUST apply an explicit 1/sqrt(head_dim) = 1/sqrt(256) = 0.0625. <-- review this.
  - attention_bias false; no attn_logit_softcapping (Llama, not Gemma) → no softcap.
  - rope_theta 1e4, rope_type "default" (full rotary) — applied in the forward BEFORE
    this attend; q/k arrive here already roped.

Own-KV cache layout: kc, vc = [NKV, KMAX, HD] fp32. KMAX is chosen by the caller
and may be the full persistent drafter prefix depth, not just the local draft K.
Score scratch must be sized for nh * (p+1) because scores are stored as tight
[nh, seq] rows for seq=p+1.

fp32 is law-acceptable here: this is a transient, recomputed-every-round M=1 attend over
the drafter's own cache, not a persistent weight or the 31B's INT4 KV.
"""

from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.math import sqrt


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def _threads_for_cols(cols: Int) -> Int:
    # Round the column count up to a sane block width, capped at 1024 (mirrors the
    # int8 decode dispatch). seq is tiny (<=16) so this mostly returns 32.
    var t = 32
    while t < cols:
        t *= 2
    if t > 1024:
        t = 1024
    return t


from lib.softmax_gpu_mojo import gpu_softmax_over_heads_mojo


def eagle3_kv_append_kernel(
    k_new: UnsafePointer[Float32, MutAnyOrigin],   # [NKV, HD]  roped k for step p
    v_new: UnsafePointer[Float32, MutAnyOrigin],   # [NKV, HD]  v for step p
    kc: UnsafePointer[Float32, MutAnyOrigin],      # [NKV, KMAX, HD]
    vc: UnsafePointer[Float32, MutAnyOrigin],      # [NKV, KMAX, HD]
    nkv_arg: Int32, hd_arg: Int32, kmax_arg: Int32, p_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var nkv = Int(nkv_arg)
    var hd = Int(hd_arg)
    var kmax = Int(kmax_arg)
    var p = Int(p_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= nkv * hd:
        return
    var kvh = idx // hd
    var d = idx % hd
    var dst = (kvh * kmax + p) * hd + d
    kc[dst] = k_new[idx]
    vc[dst] = v_new[idx]


def eagle3_qk_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],       # [NH, HD]  roped query (step p)
    kc: UnsafePointer[Float32, MutAnyOrigin],      # [NKV, KMAX, HD]
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [NH, seq]  raw scores out
    nh_arg: Int32, nkv_arg: Int32, hd_arg: Int32, kmax_arg: Int32, seq_arg: Int32, kvg_arg: Int32, scale: Float32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var nh = Int(nh_arg)
    var nkv = Int(nkv_arg)
    var hd = Int(hd_arg)
    var kmax = Int(kmax_arg)
    var seq = Int(seq_arg)
    var kvg = Int(kvg_arg)
    var h = Int(block_idx.x)
    if h >= nh:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var kvh = h // kvg                              # GQA: contiguous group of kvg Q heads
    var key = tid
    while key < seq:                               # causal [0,p]: seq = p+1, keys 0..p
        var acc = Float32(0.0)
        for d in range(hd):
            acc += q[h * hd + d] * kc[(kvh * kmax + key) * hd + d]
        scores[h * seq + key] = acc * scale        # explicit 1/sqrt(hd) — no folded q_norm
        key += n_threads


def eagle3_pv_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [NH, seq]  post-softmax weights
    vc: UnsafePointer[Float32, MutAnyOrigin],      # [NKV, KMAX, HD]
    outp: UnsafePointer[Float32, MutAnyOrigin],    # [NH, HD]   attn out -> o_proj
    nh_arg: Int32, nkv_arg: Int32, hd_arg: Int32, kmax_arg: Int32, seq_arg: Int32, kvg_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var nh = Int(nh_arg)
    var nkv = Int(nkv_arg)
    var hd = Int(hd_arg)
    var kmax = Int(kmax_arg)
    var seq = Int(seq_arg)
    var kvg = Int(kvg_arg)
    var h = Int(block_idx.x)
    if h >= nh:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var kvh = h // kvg
    var d = tid
    while d < hd:
        var acc = Float32(0.0)
        for key in range(seq):
            acc += scores[h * seq + key] * vc[(kvh * kmax + key) * hd + d]
        outp[h * hd + d] = acc
        d += n_threads


def eagle3_own_kv_attend(
    ctx: DeviceContext,
    d_q: UInt64,        # [NH, HD]        roped query for step p
    d_k_new: UInt64,    # [NKV, HD]       roped key   for step p
    d_v_new: UInt64,    # [NKV, HD]       value       for step p
    d_kc: UInt64,       # [NKV, KMAX, HD] own-K cache (appended in place)
    d_vc: UInt64,       # [NKV, KMAX, HD] own-V cache
    d_scores: UInt64,   # [NH, KMAX]      scratch (uses [NH, p+1] this step)
    d_out: UInt64,      # [NH, HD]        attention output
    p: Int,             # draft position 0..KMAX-1; attends causally over [0,p]
    nh: Int = 32,
    nkv: Int = 16,
    hd: Int = 256,
    kmax: Int = 16,
    kvg: Int = 2,
) raises:
    if p < 0 or p >= kmax:
        return
    var seq = p + 1
    var scale = Float32(1.0) / sqrt(Float32(hd))   # 1/sqrt(256) = 0.0625

    # 1. append step-p k/v into the own-KV cache at position p
    var ap = ctx.compile_function[eagle3_kv_append_kernel]()
    var total = nkv * hd
    ctx.enqueue_function(
        ap, _as_f32_ptr(d_k_new), _as_f32_ptr(d_v_new),
        _as_f32_ptr(d_kc), _as_f32_ptr(d_vc),
        Int32(nkv), Int32(hd), Int32(kmax), Int32(p),
        grid_dim=(total + 255) // 256, block_dim=256,
    )

    # 2. raw scores = scale * (q . K[0..p])
    var qk = ctx.compile_function[eagle3_qk_kernel]()
    ctx.enqueue_function(
        qk, _as_f32_ptr(d_q), _as_f32_ptr(d_kc), _as_f32_ptr(d_scores),
        Int32(nh), Int32(nkv), Int32(hd), Int32(kmax), Int32(seq), Int32(kvg), scale,
        grid_dim=nh, block_dim=_threads_for_cols(seq),
    )

    # 3. softmax over the causal window (reuses the engine's fp32 softmax)
    gpu_softmax_over_heads_mojo(ctx, d_scores, nh, seq)

    # 4. out = scores . V[0..p]
    var pv = ctx.compile_function[eagle3_pv_kernel]()
    ctx.enqueue_function(
        pv, _as_f32_ptr(d_scores), _as_f32_ptr(d_vc), _as_f32_ptr(d_out),
        Int32(nh), Int32(nkv), Int32(hd), Int32(kmax), Int32(seq), Int32(kvg),
        grid_dim=nh, block_dim=_threads_for_cols(hd),
    )
