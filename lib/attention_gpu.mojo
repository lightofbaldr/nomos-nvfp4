"""GPU attention for the Gemma-4 single-position decode.

Two batched cuBLAS SGEMMs with a CPU softmax sandwiched in the middle.
The previous CPU-Mojo attention loop was iterating nh * seq_len * hd
scalars per layer inside the Mojo interpreter — measured at ~200-300 ms
of the ~670 ms-per-token forward pass. This module moves the two big
matmuls (Q·K^T and scores·V) to cuBLAS using cublasSgemmStridedBatched
and keeps softmax on host for now. A fused GPU softmax is a follow-up.

KV cache layout (one persistent GPU buffer per layer):
    [max_nkv, max_seq, cache_hd]
  all three dims row-major contiguous. max_nkv and cache_hd are the
  per-layer MAX values across the whole model, so sliding-attention
  layers (hd=256, nkv=16) and full-attention layers (hd=512, nkv=4)
  can share the same pre-allocated buffer shape. Sliding layers just
  use the first 256 elements of each cache_hd=512 row, and only the
  first 4 kv-heads for layers that use full attention. The batched
  GEMMs respect that via the `cache_hd`/`max_seq` stride parameters.

Function contract for attention_gpu_fp32:
  - d_q_gpu is a device pointer holding the tight [nh, l_hd] Q tensor
    for the current position. The caller uploads it.
  - d_k_cache_layer / d_v_cache_layer are the layer's persistent GPU
    KV cache pointers, both laid out [max_nkv, max_seq, cache_hd].
  - d_scores_scratch / d_attn_out_scratch are re-usable device scratch
    buffers of size nh*max_seq and nh*cache_hd floats respectively.
    Pass the same pointers every call to avoid cuda_malloc churn.
  - attn_out (host List[Float32]) receives the final [nh, l_hd] output.
  - scores_host is a host scratch List used for the softmax step; it
    gets resized to nh*seq_len.
"""

from std.math import exp
from std.memory import UnsafePointer
from std.collections import List
from max.gpu.host import DeviceContext

from lib.cuda import cuda_sync, cuda_upload, cuda_download, cuda_memcpy_2d, cuda_memcpy
from lib.cuda import cuda_memcpy_2d_async
from lib.cublas import cublas_sgemm_strided_batched_nt, cublas_sgemm_strided_batched_nn
from lib.cublas import cublas_gemm_ex_sb_bf16_nt, cublas_gemm_ex_sb_bf16_nn
from lib.cast_gpu_mojo import gpu_fp32_to_bf16_mojo
from lib.softmax_gpu_mojo import gpu_softmax_over_heads_mojo


def append_kv_gpu(
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    mut k_new: List[Float32],
    mut v_new: List[Float32],
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    cl: Int,
):
    """Write one new position's K and V into the layer's persistent GPU cache.

    Layout: the cache is [max_nkv, max_seq, cache_hd] row-major. For the
    NEW position `cl`, each of the `nkv` KV heads needs l_hd floats
    written at offset (kvh * max_seq + cl) * cache_hd.

    We do this with two cudaMemcpy2D calls — one for K, one for V — each
    of which scatters all nkv rows in a single H2D transfer. The
    destination pitch (dpitch) is max_seq * cache_hd * 4 bytes (the
    distance between consecutive KV head slots), and the source pitch
    (spitch) is l_hd * 4 bytes (the source k_new/v_new are tight
    [nkv, l_hd]). Width is l_hd * 4 and height is nkv.

    This replaces the naive per-head loop (nkv small uploads per layer
    per step = up to 16*30 = 480 launches per token), which was a
    measurable share of the total forward-pass latency.
    """
    # Destination starts at position `cl` within KV head 0.
    var base_offset_bytes = UInt64(cl * cache_hd * 4)
    var dst_k = d_k_cache_layer + base_offset_bytes
    var dst_v = d_v_cache_layer + base_offset_bytes
    var dpitch = max_seq * cache_hd * 4    # bytes between KV heads in cache
    var spitch = l_hd * 4                   # bytes between KV heads in source
    var width_bytes = l_hd * 4              # active row width
    var src_k = UInt64(Int(k_new.unsafe_ptr()))
    var src_v = UInt64(Int(v_new.unsafe_ptr()))
    cuda_memcpy_2d(dst_k, dpitch, src_k, spitch, width_bytes, nkv, 1)  # kind=1 H2D
    cuda_memcpy_2d(dst_v, dpitch, src_v, spitch, width_bytes, nkv, 1)


def append_kv_gpu_dev(
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    d_k_new: UInt64,                # device pointer to K [nkv, l_hd]
    d_v_new: UInt64,                # device pointer to V [nkv, l_hd]
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    cl: Int,
):
    """append_kv_gpu when K and V are already on device (D2D memcpy)."""
    var base_offset_bytes = UInt64(cl * cache_hd * 4)
    var dst_k = d_k_cache_layer + base_offset_bytes
    var dst_v = d_v_cache_layer + base_offset_bytes
    var dpitch = max_seq * cache_hd * 4
    var spitch = l_hd * 4
    var width_bytes = l_hd * 4
    cuda_memcpy_2d(dst_k, dpitch, d_k_new, spitch, width_bytes, nkv, 3)  # kind=3 D2D
    cuda_memcpy_2d(dst_v, dpitch, d_v_new, spitch, width_bytes, nkv, 3)


def append_kv_gpu_dev_async(
    ctx: DeviceContext,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    d_k_new: UInt64,                # device pointer to K [nkv, l_hd]
    d_v_new: UInt64,                # device pointer to V [nkv, l_hd]
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    cl: Int,
) raises:
    """append_kv_gpu_dev on ctx's stream, avoiding legacy default-stream fences."""
    var base_offset_bytes = UInt64(cl * cache_hd * 4)
    var dst_k = d_k_cache_layer + base_offset_bytes
    var dst_v = d_v_cache_layer + base_offset_bytes
    var dpitch = max_seq * cache_hd * 4
    var spitch = l_hd * 4
    var width_bytes = l_hd * 4
    cuda_memcpy_2d_async(
        dst_k, dpitch, d_k_new, spitch, width_bytes, nkv, 3, ctx
    )
    cuda_memcpy_2d_async(
        dst_v, dpitch, d_v_new, spitch, width_bytes, nkv, 3, ctx
    )


def append_kv_gpu_bf16_dev(
    ctx: DeviceContext,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    d_k_new: UInt64,                # FP32 K [nkv, l_hd]
    d_v_new: UInt64,                # FP32 V [nkv, l_hd]
    d_k_bf16: UInt64,               # BF16 scratch >= nkv*l_hd elems
    d_v_bf16: UInt64,               # BF16 scratch >= nkv*l_hd elems
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    cl: Int,
) raises:
    """BF16 twin of append_kv_gpu_dev: cast FP32 K/V -> BF16, then scatter into the
    BF16 cache (2 bytes/elem) at position cl. cache_hd is in ELEMENTS (same as fp32);
    only the byte multiplier changes 4 -> 2."""
    gpu_fp32_to_bf16_mojo(ctx, d_k_new, d_k_bf16, nkv * l_hd)
    gpu_fp32_to_bf16_mojo(ctx, d_v_new, d_v_bf16, nkv * l_hd)
    var base_offset_bytes = UInt64(cl * cache_hd * 2)
    var dst_k = d_k_cache_layer + base_offset_bytes
    var dst_v = d_v_cache_layer + base_offset_bytes
    var dpitch = max_seq * cache_hd * 2
    var spitch = l_hd * 2
    var width_bytes = l_hd * 2
    cuda_memcpy_2d(dst_k, dpitch, d_k_bf16, spitch, width_bytes, nkv, 3)
    cuda_memcpy_2d(dst_v, dpitch, d_v_bf16, spitch, width_bytes, nkv, 3)


def attention_gpu_fp32_dev(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_gpu: UInt64,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    d_scores_scratch: UInt64,
    d_attn_out_scratch: UInt64,     # output stays on device — caller reads it
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    seq_len: Int,
    kvg: Int,
) raises:
    """Same as attention_gpu_fp32 but skips the final download — output stays
    on device in d_attn_out_scratch for the device-resident layer path."""
    cublas_sgemm_strided_batched_nt(
        handle,
        d_q_gpu,          l_hd,     kvg * l_hd,
        d_k_cache_layer,  cache_hd, max_seq * cache_hd,
        d_scores_scratch, seq_len,  kvg * seq_len,
        kvg, l_hd, seq_len,
        nkv,
    )
    # PREFILL-PERF EXPERIMENT (2026-06-14): removed 3 cudaDeviceSynchronize that
    # bridged cuBLAS(default stream) <-> Mojo softmax(ctx stream). ~800K of these in
    # prefill = the 190x-off-peak gap. If output stays byte-identical to the parity
    # reference, DeviceContext() shares the default stream and these were redundant;
    # if it races (wrong output), we need cublasSetStream(handle, ctx stream) instead.
    gpu_softmax_over_heads_mojo(ctx, d_scores_scratch, nh, seq_len)
    cublas_sgemm_strided_batched_nn(
        handle,
        d_scores_scratch,   seq_len,  kvg * seq_len,
        d_v_cache_layer,    cache_hd, max_seq * cache_hd,
        d_attn_out_scratch, l_hd,     kvg * l_hd,
        kvg, seq_len, l_hd,
        nkv,
    )


def attention_gpu_bf16_dev(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_gpu: UInt64,            # FP32 Q for this token, [nh, l_hd]
    d_k_cache_layer: UInt64,    # BF16 K cache
    d_v_cache_layer: UInt64,    # BF16 V cache
    d_scores_scratch: UInt64,   # FP32 scores [nh, seq_len]
    d_q_bf16: UInt64,           # BF16 scratch, >= nh*l_hd elems
    d_scores_bf16: UInt64,      # BF16 scratch, >= nh*seq_len elems
    d_attn_out_scratch: UInt64, # FP32 output [nh, l_hd]
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    seq_len: Int,
    kvg: Int,
) raises:
    """BF16 twin of attention_gpu_fp32_dev. Q·K^T and scores·V run on bf16 tensor
    cores over the bf16 KV cache; softmax stays FP32 (numerically sensitive). The
    KV cache is bf16 (2 bytes/elem) but cuBLAS strides are ELEMENT counts, so
    cache_hd / max_seq*cache_hd are IDENTICAL to the fp32 path. Two casts: Q
    fp32->bf16 (nh*l_hd elems), scores fp32->bf16 (nh*seq_len elems)."""
    gpu_fp32_to_bf16_mojo(ctx, d_q_gpu, d_q_bf16, nh * l_hd)
    cublas_gemm_ex_sb_bf16_nt(
        handle,
        d_q_bf16,         l_hd,     kvg * l_hd,
        d_k_cache_layer,  cache_hd, max_seq * cache_hd,
        d_scores_scratch, seq_len,  kvg * seq_len,
        kvg, l_hd, seq_len,
        nkv,
    )
    gpu_softmax_over_heads_mojo(ctx, d_scores_scratch, nh, seq_len)
    gpu_fp32_to_bf16_mojo(ctx, d_scores_scratch, d_scores_bf16, nh * seq_len)
    cublas_gemm_ex_sb_bf16_nn(
        handle,
        d_scores_bf16,      seq_len,  kvg * seq_len,
        d_v_cache_layer,    cache_hd, max_seq * cache_hd,
        d_attn_out_scratch, l_hd,     kvg * l_hd,
        kvg, seq_len, l_hd,
        nkv,
    )


def attention_gpu_fp32(
    ctx: DeviceContext,
    handle: UInt64,
    d_q_gpu: UInt64,
    d_k_cache_layer: UInt64,
    d_v_cache_layer: UInt64,
    d_scores_scratch: UInt64,
    d_attn_out_scratch: UInt64,
    mut attn_out: List[Float32],
    mut scores_host: List[Float32],
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    seq_len: Int,
    kvg: Int,
) raises:
    """Run single-position GQA attention for one layer on the GPU.

    Steps:
      1. Batched Q·K^T   → scores on device, [nkv, kvg, seq_len]
      2. Download scores → softmax per head on CPU → upload back
      3. Batched scores·V → attn_out on device, [nkv, kvg, l_hd]
      4. Download attn_out into the host output list

    On exit, `attn_out` is filled with the [nh, l_hd] tight layout the
    rest of the layer expects for the O projection.
    """

    # ── Step 1: Q · K^T ────────────────────────────────────────────────
    # Per batch (one KV head):
    #   A = Q_group [kvg, l_hd]      row stride = l_hd     (tight)
    #   B = K_slice [seq_len, l_hd]  row stride = cache_hd (padded)
    #   C = scores  [kvg, seq_len]   row stride = seq_len  (tight)
    cublas_sgemm_strided_batched_nt(
        handle,
        d_q_gpu,          l_hd,     kvg * l_hd,           # A
        d_k_cache_layer,  cache_hd, max_seq * cache_hd,   # B
        d_scores_scratch, seq_len,  kvg * seq_len,        # C
        kvg, l_hd, seq_len,
        nkv,
    )
    cuda_sync()

    # ── Step 2: softmax on GPU (fused kernel — no PCIe round-trips) ──
    # scores_host layout: [nkv, kvg, seq_len] = [nh, seq_len] row-major.
    # In-place on device; the old download/CPU/upload sandwich is gone.
    gpu_softmax_over_heads_mojo(ctx, d_scores_scratch, nh, seq_len)
    cuda_sync()

    # ── Step 3: scores · V ─────────────────────────────────────────────
    # Per batch (one KV head):
    #   A = scores [kvg, seq_len]     row stride = seq_len  (tight)
    #   B = V      [seq_len, l_hd]    row stride = cache_hd (padded)
    #   C = out    [kvg, l_hd]        row stride = l_hd     (tight)
    cublas_sgemm_strided_batched_nn(
        handle,
        d_scores_scratch,   seq_len,  kvg * seq_len,        # A
        d_v_cache_layer,    cache_hd, max_seq * cache_hd,   # B
        d_attn_out_scratch, l_hd,     kvg * l_hd,           # C
        kvg, seq_len, l_hd,
        nkv,
    )
    cuda_sync()

    # ── Step 4: download to host ──────────────────────────────────────
    cuda_download(attn_out, d_attn_out_scratch, nh * l_hd)
