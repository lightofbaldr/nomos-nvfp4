"""Gemma-4 per-layer forward-pass building blocks shared by both Sparks.

Each transformer layer in Gemma-4 runs roughly:

  1. input_layernorm
  2. Q, K, V projections (V = K for full-attention layers)
  3. V-norm (per-head RMSNorm without scale)
  4. qk_norm on Q and K
  5. RoPE on Q and K (partial rotary for full attention, full for sliding)
  6. KV cache append                              ← differs by KV-cache precision config
  7. Attention (softmax over positions, weighted V)← differs by KV-cache precision config
  8. O projection
  9. post_attention_layernorm + residual
  10. pre_feedforward_layernorm
  11. MLP: gate_proj, up_proj, GeGLU, down_proj
  12. post_feedforward_layernorm + residual
  13. * layer_scalar (multiplied over the entire hidden state)

Steps 1-5 are identical between the FP32-KV reference config and the
INT8-KV production config. Same for steps 8-13. This module extracts
those into `prepare_qkv` and `apply_output_and_mlp` so both sparks only
need to write the per-Spark KV-cache plumbing (steps 6-7) themselves.

Design notes:
- Every buffer a caller wants mutated is a `mut List[Float32]` — no
  global state, no hidden side effects. The caller allocates the
  temporaries once per layer.
- `gpu_matmul` calls allocate their own cuda temporaries; that's a
  follow-up optimization (see cublas.mojo's note on the leak) but
  preserves behavior-identical output to the pre-refactor code.
- The V-projection branch matches Gemma-4's `attention_k_eq_v=True`
  config: for full-attention layers (`is_full=True`, `d_vw=0`) we copy
  k_new into v_new instead of running a separate v_proj SGEMM.
"""

from std.collections import List
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from lib.cublas import gpu_matmul_bf16, gpu_matmul_bf16_dev, gpu_matmul_pooled, gpu_matmul_bf16_dev_batched
from lib.q4_weights import gpu_matmul_q4_dev, gpu_matmul_q4_fused_dev, gpu_matmul_q4_auto_dev, gpu_matmul_q4_dev_batched
from lib.fp4_weights import gpu_matmul_nvfp4_dev, gpu_matmul_nvfp4_dev_batched
from lib.fp4_weights import gpu_matmul_nvfp4_fused_dev, _nvfp4_use_fused
from lib.fp4_gemv_v2 import gpu_matmul_nvfp4_fused_v3_dev, _nvfp4_use_v3
from lib.fp4_gemv_v2 import gpu_matmul_nvfp4_fused_v3_tree_dev, _nvfp4_use_v3_tree
from lib.q4_gemv_v2 import gpu_matmul_q4_v2_occ_dev
from lib.q4_gemv_dp4a import gpu_matmul_q4_dp4a_dev, act_precision
from lib.q4_gemv_dp4a import (
    gpu_matmul_q4_mmq_dev,
    gpu_matmul_q4_mmq_relaxed_verify_dev,
    gpu_matmul_q4_s8_v4_gemv_dev,
    MMQ_DRAFTS,
)
from lib.engine_init import _strict_q4, _flag_violation, _read_env_bytes
from lib.engine_init import VIOL_BF16_PROJ, VIOL_FP32_PROJ, VIOL_BF16_PROJ_BATCHED
from lib.q4_gemv_dp4a import gpu_q8_quantize_dev, gpu_matmul_q4_dp4a_gemv_dev
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_dev
from lib.fp4_act import (
    gpu_quant_act_nvfp4_dec,
    gpu_matmul_nvfp4_w4a4_prequant_dev,
    gpu_matmul_nvfp4_w4a4_prequant_grouped4_dev,
    w4a4_prequant_grouped4_route,
    _w4a4_prequant_off,
)


def _verify_mmq_small() -> Bool:
    var v = _read_env_bytes("NOMOS_VERIFY_MMQ_SMALL")
    return len(v) > 0 and v[0] == UInt8(49)


# W4A4 decode scratch: 4 device buffers (act NVFP4 packed/bs/global + raw-MMA C),
# shared across all GEMMs in a layer (sequential). Bundled so the W4A4 dispatch
# threads one struct, not five UInt64s, through the layer functions.
@fieldwise_init
struct W4A4Scratch(Copyable, Movable):
    var packed: UInt64      # [cap, K/2] u8
    var bs: UInt64          # [cap, K/16] u8
    var glob: UInt64        # [cap] fp32
    var cpad: UInt64        # [cap, N] fp32
    var cap: Int            # #431: scratch row capacity = the prefill chunk size (mult of 16)
    var on: Bool            # W4A4 active (env NOMOS_W4A4 + NVFP4 weights)
    var bs_sf: UInt64       # sm_100: SF-atom-scattered ACTIVATION scales (0 on sm_120/121)
    var wbs_sf: UInt64      # sm_100: SF-atom-scattered WEIGHT scales scratch (0 on sm_120/121)


alias W4A4_MPAD = 16        # decode M=1 padded to the MMA m-dim (m16)


def _is_sm100_dev(ctx: DeviceContext) raises -> Bool:
    """Datacenter Blackwell (sm_100/B200) = compute 10.x. Same check `_mm_dev` makes
    inline; shared so the R2 Phase-A dedup branches mirror its W4A4 dispatch exactly."""
    return (Float64(ctx.default_device_info.compute) >= 10.0
            and Float64(ctx.default_device_info.compute) < 12.0)


def _mm_dev(
    ctx: DeviceContext, handle: UInt64, d_out: UInt64, d_in: UInt64,
    d_w: UInt64, d_scratch_in_bf16: UInt64, K: Int, N: Int, q4_scratch: UInt64,
    global_scale: Float32 = 0.0,
    input_global: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    layer_idx: Int = -1,
    projection: String = "unknown",
) raises:
    """Weight-GEMM dispatch (decode, M=1). w4a4.on & global_scale!=0 -> native FP4 W4A4
    (quantize the activation -> FP4 MMA vs the NVFP4 weight -> post-scale; the speed path).
    global_scale!=0 alone -> NVFP4 weight dequant->bf16->cuBLAS (W4A16). q4_scratch!=0 ->
    Q4_0. Else bf16 GEMM unchanged. Weight has K*N elements."""
    if d_w == 0 and global_scale != 0.0:
        raise Error(
            "refusing W4A4 GEMM with null weight and live scale: layer="
            + String(layer_idx) + " projection=" + projection
            + " global_scale=" + String(global_scale)
        )
    # Per-phase precision (sm_100/B200): DECODE (M=1) is HBM-bandwidth-bound on the 4-bit
    # weight read — FP4 *activations* buy nothing at M=1, and the tcgen05 W4A4 MMA only tiles
    # in 128-row blocks (M=1->128 = ~30x wasted compute). So on sm_100 decode falls through to
    # the NVFP4 fused dequant-GEMV (bandwidth-optimal + higher quality fp32 act). W4A4's win is
    # PREFILL (compute-bound), which still routes to the tcgen05 kernel via _mm_dev_batched.
    # sm_120/121 keep the warp-MMA W4A4 decode (16-pad) unchanged.
    var is_sm100 = (Float64(ctx.default_device_info.compute) >= 10.0
                    and Float64(ctx.default_device_info.compute) < 12.0)
    if global_scale != 0.0 and w4a4.on and not is_sm100:
        gpu_matmul_nvfp4_w4a4_dev(ctx, handle, d_out, d_in, d_w, global_scale, input_global,
                                  w4a4.packed, w4a4.bs, w4a4.glob, w4a4.bs_sf, w4a4.wbs_sf,
                                  w4a4.cpad, 1, W4A4_MPAD, K, N)
    elif global_scale != 0.0:
        # NVFP4 decode: M2 FUSED dequant-GEMV by default (reads the NVFP4 blob directly,
        # fp32 act -> reasoning preserved, no per-token whole-weight dequant). Also the sm_100
        # W4A4 decode path per the note above. NOMOS_NVFP4_FUSED=0 falls back to the M1 path.
        if _nvfp4_use_fused():
            if _nvfp4_use_v3():
                # Unit A (swarm build round): arithmetic-E2M1 branchless fused GEMV v3
                # (~1.8x vs v2, parity-clean, FP4-tol gated). NOMOS_NVFP4_V3=0 -> v2.
                if _nvfp4_use_v3_tree():
                    # Per-shape v3-tree (GLM's validated dispatch table; det + within-FP4-tol).
                    # Tree-reduce shortens the per-lane K-chain -> high-K shapes to ~57-59%
                    # roofline (FF_D laggard 50->58%). _mm_dev only sees per-layer GEMVs
                    # (lm_head is engine_decode's own path); untabled o-proj falls to plain v3.
                    if K == 5376 and N == 8192:        # D_Dq
                        gpu_matmul_nvfp4_fused_v3_tree_dev[2](ctx, d_out, d_in, d_w, global_scale, K, N)
                    elif K == 5376 and N == 4096:      # D_Dkv
                        gpu_matmul_nvfp4_fused_v3_tree_dev[4](ctx, d_out, d_in, d_w, global_scale, K, N)
                    elif K == 5376 and N == 21504:     # D_FF (gate/up)
                        gpu_matmul_nvfp4_fused_v3_tree_dev[4](ctx, d_out, d_in, d_w, global_scale, K, N)
                    elif K == 21504 and N == 5376:     # FF_D (down) — the laggard lift
                        gpu_matmul_nvfp4_fused_v3_tree_dev[2](ctx, d_out, d_in, d_w, global_scale, K, N)
                    else:                               # o-proj + any untabled shape
                        gpu_matmul_nvfp4_fused_v3_dev(ctx, d_out, d_in, d_w, global_scale, K, N)
                else:
                    gpu_matmul_nvfp4_fused_v3_dev(ctx, d_out, d_in, d_w, global_scale, K, N)
            else:
                gpu_matmul_nvfp4_fused_dev(ctx, d_out, d_in, d_w, global_scale, K, N)
        else:
            gpu_matmul_nvfp4_dev(ctx, handle, d_out, d_in, d_w, global_scale, K * N, q4_scratch, d_scratch_in_bf16, K, N)
    elif q4_scratch != 0:
        # Q4 decode (M=1): occ4 fused GEMV — WPB=4 (4 warps/block) widens occupancy
        # 50%->100% on GB10 (+40-48% on the HBM-bound D_FF/FF_D shapes), BIT-IDENTICAL to the
        # old 1-warp/block fused path (same per-row math, packed 4 rows/block). Reads the Q4
        # blob directly, no bf16 materialization. (GLM/swarm 2026-06-25; internal notes)
        # Activation-precision selector (2026-06-27): 8 = q8 dp4a (production,
        # ~2.85x BW, int8 dp4a on q8_1 acts; q4_dp4a.mojo), 32 = fp32 acts (occ4). The
        # q8_1-quantized input lands in the (Q4-unused) bf16 scratch. q16/q4 bolt in here.
        if act_precision() == 8:
            gpu_matmul_q4_dp4a_dev[4](ctx, d_out, d_in, d_w, d_scratch_in_bf16, K, N)
        else:
            if _strict_q4(): _flag_violation(VIOL_FP32_PROJ)  # occ4 = fp32 acts
            gpu_matmul_q4_v2_occ_dev[4](ctx, d_out, d_in, d_w, K, N)
    else:
        if _strict_q4(): _flag_violation(VIOL_BF16_PROJ)
        gpu_matmul_bf16_dev(ctx, handle, d_out, d_in, d_w, d_scratch_in_bf16, K, N)


def _mm_dev_batched(
    ctx: DeviceContext, handle: UInt64, d_out: UInt64, d_in: UInt64,
    d_w: UInt64, d_scratch_in_bf16: UInt64, S: Int, K: Int, N: Int, q4_scratch: UInt64,
    global_scale: Float32 = 0.0,
    input_global: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    layer_idx: Int = -1,
    projection: String = "unknown",
) raises:
    """Batched (prefill, M=S) weight-GEMM dispatch. w4a4.on & global_scale!=0 -> native FP4
    W4A4 over the whole prompt at once (M=S, Mpad=ceil(S/16)*16; quant act -> FP4 MMA vs the
    NVFP4 weight -> post-scale). global_scale!=0 alone -> NVFP4 weight dequant once ->
    q4_scratch (bf16) then batched cuBLAS reused across S (W4A16). q4_scratch!=0 -> Q4_0.
    Else bf16 unchanged."""
    if d_w == 0 and global_scale != 0.0:
        raise Error(
            "refusing W4A4 GEMM with null weight and live scale: layer="
            + String(layer_idx) + " projection=" + projection
            + " global_scale=" + String(global_scale)
        )
    if global_scale != 0.0 and w4a4.on:
        # #431: cap = the per-chunk scratch row capacity; the GEMM chunks the S rows
        # internally (S<=cap => one chunk = the original single-call path).
        gpu_matmul_nvfp4_w4a4_dev(ctx, handle, d_out, d_in, d_w, global_scale, input_global,
                                  w4a4.packed, w4a4.bs, w4a4.glob, w4a4.bs_sf, w4a4.wbs_sf,
                                  w4a4.cpad, S, w4a4.cap, K, N)
    elif global_scale != 0.0:
        gpu_matmul_nvfp4_dev_batched(ctx, handle, d_out, d_in, d_w, global_scale, K * N, q4_scratch, d_scratch_in_bf16, S, K, N)
    elif q4_scratch != 0:
        # Spec-decode verify (S<=8 drafts) + dp4a mode: route the batched projection through
        # the int8 TENSOR-CORE MMQ (int32-exact == per-token dp4a decode → kills the bf16
        # batched-vs-decode divergence; M=S amortizes the Q4 weight). Prefill (S>8) keeps the
        # bf16-materialized path (one variable at a time — the seam gate). q4_scratch is
        # reused as the MMQ q8 scratch (>> the ~193KB it needs).
        if act_precision() == 8:
            if S <= 8 and (S <= 4 or not _verify_mmq_small()):
                # Small fused-verify batches stay on the proven decode-order v4
                # GEMV route; MMQ's fixed overhead only amortizes at larger S.
                gpu_matmul_q4_s8_v4_gemv_dev[8](ctx, d_out, d_in, d_w, q4_scratch, K, N, S)
            elif S <= 16:
                # S=9..16 uses tensor-core MMQ. Decode and verify share the
                # fixed-point Q4 epilogue, so MMQ can reduce blocks in its fast
                # order without fp32-association stream flips. The staged body
                # has a shape-dependent best WARPS value: gate/up are
                # output-wide, down is K-wide, and O is square-ish.
                if K <= 8192 and N > 8192:
                    gpu_matmul_q4_mmq_relaxed_verify_dev[8](
                        ctx, d_out, d_in, d_w, q4_scratch, K, N, S
                    )
                elif K > 8192 and N <= 8192:
                    gpu_matmul_q4_mmq_relaxed_verify_dev[2](
                        ctx, d_out, d_in, d_w, q4_scratch, K, N, S
                    )
                else:
                    gpu_matmul_q4_mmq_relaxed_verify_dev[4](
                        ctx, d_out, d_in, d_w, q4_scratch, K, N, S
                    )
            elif S <= 17:
                # S=17 pays a second-pass tax on the current 16-row MMQ body;
                # v4 is faster for this rare DFlash block-16+bonus shape.
                gpu_matmul_q4_s8_v4_gemv_dev[8](ctx, d_out, d_in, d_w, q4_scratch, K, N, S)
            else:
                # Large prefill keeps the tensor-core MMQ path.
                gpu_matmul_q4_mmq_dev[4](ctx, d_out, d_in, d_w, q4_scratch, K, N, S)
        else:
            if _strict_q4(): _flag_violation(VIOL_BF16_PROJ_BATCHED)  # Q4->bf16 + bf16 cuBLAS
            gpu_matmul_q4_dev_batched(ctx, handle, d_out, d_in, d_w, q4_scratch, d_scratch_in_bf16, S, K, N)
    else:
        gpu_matmul_bf16_dev_batched(ctx, handle, d_out, d_in, d_w, d_scratch_in_bf16, S, K, N)
from lib.gemma4_ops import rmsnorm, rmsnorm_inplace, rmsnorm_no_weight, qk_norm, rope, gelu
# Pure-Mojo GPU op wrappers (replaces lib/gpu_ops.mojo's libops.so FFI).
from lib.ops_gpu_mojo import (
    gpu_residual_add_mojo, gpu_elementwise_mul_mojo, gpu_scalar_mul_mojo,
    gpu_residual_add_scaled_mojo, gpu_gelu_mojo, gpu_gelu_mul_mojo,
    gpu_silu_mul_mojo, gpu_sigmoid_mul_inplace_mojo,
    gpu_embed_load_mojo, gpu_embed_copy_fp32_mojo, gpu_rope_mojo,
    gpu_rope_batched_mojo, gpu_yarn_rope_mojo, gpu_yarn_rope_batched_mojo,
    gpu_split_q_gate_interleaved_mojo,
)
from lib.model_config import (
    HAS_ATTN_GATE, FULL_LAYERS_NOPE, ROPE_THETA_FULL, ROPE_THETA_SLIDING,
    ROPE_FULL_PARTIAL_DIM, FULL_LAYERS_V_EQ_K,
    RMS_EPS_INPUT, RMS_EPS_QK, RMS_EPS_POST_ATTN, RMS_EPS_PRE_FF,
    RMS_EPS_POST_FF,
    HAS_LEARNED_QK_NORM, QK_NORM_Q_SCALE, HAS_V_NORM,
    ATTN_GATE_IN_QPROJ, HAS_LINEAR_ATTENTION,
    MLP_ACT_SILU, NORM_STYLE_POST, QK_NORM_FULL_VECTOR,
    TARGET_ROPE_YARN, TARGET_YARN_FACTOR, TARGET_YARN_ORIGINAL_MAX,
    TARGET_YARN_BETA_FAST, TARGET_YARN_BETA_SLOW,
    TARGET_YARN_ATTN_FACTOR,
)
from lib.ops_gpu_mojo_reductions import (
    gpu_rmsnorm_mojo, gpu_rmsnorm_inplace_mojo,
    gpu_rmsnorm_no_weight_mojo, gpu_qk_norm_mojo,
    gpu_rmsnorm_batched_mojo, gpu_rmsnorm_inplace_batched_mojo,
)
from lib.cuda import cuda_memcpy, cuda_download, cuda_budget_mark
from std.ffi import external_call, c_int, c_size_t


# ── Device-resident variants — activations stay on GPU ─────────────────────
# Replaces the CPU/GPU ping-pong above. d_x and all per-layer intermediate
# buffers are device pointers. Only the per-layer weights and norms read from
# device memory; no PCIe traffic in the steady state.


def prepare_qkv_dev(
    ctx: DeviceContext,
    d_x: UInt64,                    # residual stream on device [d]
    d_normed: UInt64,               # output norm scratch [d]
    d_q: UInt64,                    # output Q [l_qd]
    d_k_new: UInt64,                # output K [l_kvd]
    d_v_new: UInt64,                # output V [l_kvd]
    d_attn_gate: UInt64,            # optional early attention-gate output [l_qd]
    d_q_gate_raw: UInt64,           # Qwen raw interleaved [2*l_qd]
    handle: UInt64,
    d_in_norm_w: UInt64,            # per-layer input_layernorm weight
    d_q_norm_w: UInt64,             # per-layer q_norm weight
    d_k_norm_w: UInt64,             # per-layer k_norm weight
    d_qw: UInt64,                   # BF16 Q weight
    d_kw: UInt64,                   # BF16 K weight
    d_vw: UInt64,                   # BF16 V weight (or 0 for full attention)
    d_attn_gw: UInt64,              # attention-gate weight (grouped W4A4 path)
    d_scratch_in_bf16: UInt64,
    is_full: Bool,
    d: Int,
    l_nh: Int,
    l_nkv: Int,
    l_hd: Int,
    l_qd: Int,
    l_kvd: Int,
    pos: Int,
    layer_idx: Int,
    d_q4_scratch: UInt64 = 0,
    g_q: Float32 = 0.0,             # NVFP4 per-tensor global scales (0 = bf16/Q4)
    g_k: Float32 = 0.0,
    g_v: Float32 = 0.0,
    g_attn_g: Float32 = 0.0,
    ag_q: Float32 = 0.0,
    ag_k: Float32 = 0.0,
    ag_v: Float32 = 0.0,
    ag_attn_g: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    grouped_qkv: Bool = False,
    debug_stage: Int = -1,
    debug_out: Int64 = 0,
) raises -> Bool:
    """Steps 1-5 of a Gemma-4 layer forward pass, fully on device."""
    # 1. Pre-norm models normalize the residual before QKV. OLMo2/3 is
    # post-norm: attention consumes the raw residual and normalizes O-proj output.
    @parameter
    if NORM_STYLE_POST:
        cuda_memcpy(d_normed, d_x, d * 4, 3)
    else:
        gpu_rmsnorm_mojo(ctx, d_x, d_normed, d_in_norm_w, d, RMS_EPS_INPUT)

    # 2. Q, K, V projections (input is d_normed on device). Q/K/V SHARE d_normed, so
    # the dp4a Q4 path (act_precision==8, Q4 weights g==0) q8-quantizes it ONCE then runs
    # gemv-only ×3 — drops 2 redundant quants + 2 launches/layer. Other modes keep _mm_dev.
    var grouped_attn_gate = False
    var v_eq_k = is_full and FULL_LAYERS_V_EQ_K
    if ATTN_GATE_IN_QPROJ:
        _mm_dev(ctx, handle, d_q_gate_raw, d_normed, d_qw, d_scratch_in_bf16,
                d, 2 * l_qd, d_q4_scratch, g_q, ag_q, w4a4, layer_idx, "q_gate_proj")
        gpu_split_q_gate_interleaved_mojo(
            ctx, d_q, d_attn_gate, d_q_gate_raw, 1, l_qd, l_hd
        )
        grouped_attn_gate = True
        _mm_dev(ctx, handle, d_k_new, d_normed, d_kw, d_scratch_in_bf16,
                d, l_kvd, d_q4_scratch, g_k, ag_k, w4a4, layer_idx, "k_proj")
        _mm_dev(ctx, handle, d_v_new, d_normed, d_vw, d_scratch_in_bf16,
                d, l_kvd, d_q4_scratch, g_v, ag_v, w4a4, layer_idx, "v_proj")
    elif act_precision() == 8 and d_q4_scratch != 0 and g_q == 0.0:
        gpu_q8_quantize_dev(ctx, d_normed, d_scratch_in_bf16, d)
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_q, d_scratch_in_bf16, d_qw, d, l_qd)
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_k_new, d_scratch_in_bf16, d_kw, d, l_kvd)
        if v_eq_k:
            cuda_memcpy(d_v_new, d_k_new, l_kvd * 4, 3)
        else:
            gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_v_new, d_scratch_in_bf16, d_vw, d, l_kvd)
    elif (
        w4a4.on and not _is_sm100_dev(ctx)
        and g_q != 0.0 and g_k != 0.0 and (v_eq_k or g_v != 0.0)
        and ag_q == ag_k and (v_eq_k or ag_q == ag_v)
    ):
        # W4A4 prequant dedup (mirrors the dp4a branch above). Q/K/V all consume
        # d_normed, so quantize it ONCE into the shared w4a4 scratch and run
        # prequant-only GEMMs — drops 2 act-quant launches/layer (the profile showed
        # act-quant launches 1:1 with GEMMs: 7/layer where only 4 inputs are distinct).
        # GUARD: only when the per-tensor calibrated activation globals AGREE. They are
        # 0 in the shipped weights (uncalibrated => byte-identical quantization), but a
        # future calibrated repack could differ per projection, which would make the
        # shared quantization numerically WRONG — fall through to per-GEMM quant then.
        gpu_quant_act_nvfp4_dec(ctx, d_normed, w4a4.packed, w4a4.bs, w4a4.glob, d, ag_q)
        # Experimental grouped arm: one block-index range covers Q/K/V/attention-gate,
        # while each segment keeps its own B pointers, global multiplier, and output.
        # The fourth activation-global equality is load-bearing: otherwise gate needs
        # different activation bytes and must stay on the ordinary later dispatch.
        if (
            grouped_qkv
            and g_attn_g != 0.0
            and ag_q == ag_attn_g
            and w4a4_prequant_grouped4_route(d, l_qd, l_kvd, l_kvd, l_qd)
        ):
            gpu_matmul_nvfp4_w4a4_prequant_grouped4_dev(
                ctx,
                d_q, d_qw, g_q, l_qd,
                d_k_new, d_kw, g_k, l_kvd,
                d_v_new, d_vw, g_v, l_kvd,
                d_attn_gate, d_attn_gw, g_attn_g, l_qd,
                w4a4.packed, w4a4.bs, w4a4.glob, d,
            )
            grouped_attn_gate = True
        else:
            gpu_matmul_nvfp4_w4a4_prequant_dev(
                ctx, d_q, d_qw, g_q, w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad, 1, d, l_qd)
            gpu_matmul_nvfp4_w4a4_prequant_dev(
                ctx, d_k_new, d_kw, g_k, w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad, 1, d, l_kvd)
            if v_eq_k:
                cuda_memcpy(d_v_new, d_k_new, l_kvd * 4, 3)
            else:
                gpu_matmul_nvfp4_w4a4_prequant_dev(
                    ctx, d_v_new, d_vw, g_v, w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad, 1, d, l_kvd)
    else:
        _mm_dev(ctx, handle, d_q, d_normed, d_qw, d_scratch_in_bf16, d, l_qd, d_q4_scratch, g_q, ag_q, w4a4, layer_idx, "q_proj")
        _mm_dev(ctx, handle, d_k_new, d_normed, d_kw, d_scratch_in_bf16, d, l_kvd, d_q4_scratch, g_k, ag_k, w4a4, layer_idx, "k_proj")
        if v_eq_k:
            cuda_memcpy(d_v_new, d_k_new, l_kvd * 4, 3)
        else:
            _mm_dev(ctx, handle, d_v_new, d_normed, d_vw, d_scratch_in_bf16, d, l_kvd, d_q4_scratch, g_v, ag_v, w4a4, layer_idx, "v_proj")

    @parameter
    if HAS_LEARNED_QK_NORM:
        # Gemma normalizes V without a learned weight, then applies learned
        # per-layer Q/K norm weights.
        @parameter
        if HAS_V_NORM:
            gpu_rmsnorm_no_weight_mojo(ctx, d_v_new, l_nkv, l_hd, RMS_EPS_QK, 1.0)
        @parameter
        if QK_NORM_FULL_VECTOR:
            gpu_rmsnorm_inplace_mojo(ctx, d_q, d_q_norm_w, l_qd, RMS_EPS_QK)
            gpu_rmsnorm_inplace_mojo(ctx, d_k_new, d_k_norm_w, l_kvd, RMS_EPS_QK)
        else:
            gpu_qk_norm_mojo(ctx, d_q, d_q_norm_w, l_nh, l_hd, RMS_EPS_QK)
            gpu_qk_norm_mojo(ctx, d_k_new, d_k_norm_w, l_nkv, l_hd, RMS_EPS_QK)
    else:
        # Muse Q/K norm is parameterless. Q alone gets the profile multiplier.
        gpu_rmsnorm_no_weight_mojo(ctx, d_q, l_nh, l_hd, RMS_EPS_QK, QK_NORM_Q_SCALE)
        gpu_rmsnorm_no_weight_mojo(ctx, d_k_new, l_nkv, l_hd, RMS_EPS_QK, 1.0)

    # Stage 26: Q after the architecture-specific Q norm, before any RoPE.
    if debug_stage == 26 and debug_out != 0:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_q, l_qd * 4, 2)

    # 5. RoPE — thetas and the NoPE decision come from the model profile.
    # Muse ropes ONLY the sliding layers (5e5, whole head) and leaves the full layers
    # NoPE — that absence is what makes them its long-range path, so skipping the call
    # IS the architecture. Gemma-4 ropes both: full at 1e6 over a PARTIAL head dim
    # (l_hd//4), sliding at 1e4 over the whole head.
    @parameter
    if TARGET_ROPE_YARN:
        # OLMo3 training/main semantics: YaRN extends only full-attention
        # layers. Sliding layers retain default RoPE at the same base theta.
        if is_full:
            gpu_yarn_rope_mojo(
                ctx, d_q, pos, l_nh, l_hd, ROPE_THETA_FULL,
                TARGET_YARN_FACTOR, TARGET_YARN_ORIGINAL_MAX,
                TARGET_YARN_BETA_FAST, TARGET_YARN_BETA_SLOW,
                TARGET_YARN_ATTN_FACTOR,
            )
            gpu_yarn_rope_mojo(
                ctx, d_k_new, pos, l_nkv, l_hd, ROPE_THETA_FULL,
                TARGET_YARN_FACTOR, TARGET_YARN_ORIGINAL_MAX,
                TARGET_YARN_BETA_FAST, TARGET_YARN_BETA_SLOW,
                TARGET_YARN_ATTN_FACTOR,
            )
        else:
            gpu_rope_mojo(
                ctx, d_q, pos, l_nh, l_hd, ROPE_THETA_SLIDING, l_hd
            )
            gpu_rope_mojo(
                ctx, d_k_new, pos, l_nkv, l_hd, ROPE_THETA_SLIDING, l_hd
            )
    else:
        if is_full:
            @parameter
            if not FULL_LAYERS_NOPE:
                var rdim = l_hd // 4 if ROPE_FULL_PARTIAL_DIM else l_hd
                gpu_rope_mojo(ctx, d_q, pos, l_nh, l_hd, ROPE_THETA_FULL, rdim)
                gpu_rope_mojo(ctx, d_k_new, pos, l_nkv, l_hd, ROPE_THETA_FULL, rdim)
        else:
            gpu_rope_mojo(ctx, d_q, pos, l_nh, l_hd, ROPE_THETA_SLIDING, l_hd)
            gpu_rope_mojo(ctx, d_k_new, pos, l_nkv, l_hd, ROPE_THETA_SLIDING, l_hd)
    if debug_out != 0 and debug_stage == 30:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_k_new, l_kvd * 4, 2)
    elif debug_out != 0 and debug_stage == 31:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_v_new, l_kvd * 4, 2)
    return grouped_attn_gate


def prepare_qkv_batched(
    ctx: DeviceContext,
    d_x: UInt64,                    # [S, d] residual (row-major, on device)
    d_normed: UInt64,               # [S, d] norm scratch
    d_q: UInt64,                    # [S, l_qd] output Q
    d_k_new: UInt64,                # [S, l_kvd] output K
    d_v_new: UInt64,                # [S, l_kvd] output V
    d_attn_gate: UInt64,            # [S, l_qd] embedded attention gate
    d_q_gate_raw: UInt64,           # [S, 2*l_qd] raw interleaved projection
    handle: UInt64,
    d_in_norm_w: UInt64,
    d_q_norm_w: UInt64,
    d_k_norm_w: UInt64,
    d_qw: UInt64,
    d_kw: UInt64,
    d_vw: UInt64,
    d_scratch_in_bf16: UInt64,      # >= S*d*2 bytes
    is_full: Bool,
    d: Int,
    l_nh: Int,
    l_nkv: Int,
    l_hd: Int,
    l_qd: Int,
    l_kvd: Int,
    S: Int,                         # sequence length (batch)
    layer_idx: Int,
    base_pos: Int = 0,              # absolute cache position of row 0
    d_q4_scratch: UInt64 = 0,       # weight-bf16 scratch; !=0 => d_qw/d_kw/d_vw are Q4 blobs
    g_q: Float32 = 0.0,             # NVFP4 per-tensor global scales (0 = bf16/Q4)
    g_k: Float32 = 0.0,
    g_v: Float32 = 0.0,
    ag_q: Float32 = 0.0,
    ag_k: Float32 = 0.0,
    ag_v: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    debug_row: Int = -1,
    debug_stage: Int = -1,
    debug_out: Int64 = 0,
) raises -> Bool:
    """Batched steps 1-5 of a Gemma-4 layer over S tokens [S,d]. Positions are
    base_pos..base_pos+S-1. Mirrors prepare_qkv_dev but every op is batched:
    rmsnorm per row, QKV as GEMM (M=S), V-norm/qk-norm over S*heads
    (per-head kernels unchanged), RoPE with per-token absolute position."""
    # 1. Input norm (per row), or identity for post-norm OLMo.
    @parameter
    if NORM_STYLE_POST:
        cuda_memcpy(d_normed, d_x, S * d * 4, 3)
    else:
        gpu_rmsnorm_batched_mojo(ctx, d_x, d_normed, d_in_norm_w, d, S, RMS_EPS_INPUT)

    # 2. Q,K,V projections (GEMM, M=S) — Q4/W4A4-aware via _mm_dev_batched
    var attn_gate_ready = False
    if ATTN_GATE_IN_QPROJ:
        _mm_dev_batched(ctx, handle, d_q_gate_raw, d_normed, d_qw,
                        d_scratch_in_bf16, S, d, 2 * l_qd, d_q4_scratch,
                        g_q, ag_q, w4a4, layer_idx, "q_gate_proj")
        gpu_split_q_gate_interleaved_mojo(
            ctx, d_q, d_attn_gate, d_q_gate_raw, S, l_qd, l_hd
        )
        attn_gate_ready = True
    else:
        _mm_dev_batched(ctx, handle, d_q, d_normed, d_qw, d_scratch_in_bf16, S, d, l_qd, d_q4_scratch, g_q, ag_q, w4a4, layer_idx, "q_proj")
    _mm_dev_batched(ctx, handle, d_k_new, d_normed, d_kw, d_scratch_in_bf16, S, d, l_kvd, d_q4_scratch, g_k, ag_k, w4a4, layer_idx, "k_proj")
    if is_full and FULL_LAYERS_V_EQ_K:
        cuda_memcpy(d_v_new, d_k_new, S * l_kvd * 4, 3)
    else:
        _mm_dev_batched(ctx, handle, d_v_new, d_normed, d_vw, d_scratch_in_bf16, S, d, l_kvd, d_q4_scratch, g_v, ag_v, w4a4, layer_idx, "v_proj")

    @parameter
    if HAS_LEARNED_QK_NORM:
        @parameter
        if HAS_V_NORM:
            gpu_rmsnorm_no_weight_mojo(ctx, d_v_new, S * l_nkv, l_hd, RMS_EPS_QK, 1.0)
        @parameter
        if QK_NORM_FULL_VECTOR:
            gpu_rmsnorm_inplace_batched_mojo(ctx, d_q, d_q_norm_w, l_qd, S, RMS_EPS_QK)
            gpu_rmsnorm_inplace_batched_mojo(ctx, d_k_new, d_k_norm_w, l_kvd, S, RMS_EPS_QK)
        else:
            gpu_qk_norm_mojo(ctx, d_q, d_q_norm_w, S * l_nh, l_hd, RMS_EPS_QK)
            gpu_qk_norm_mojo(ctx, d_k_new, d_k_norm_w, S * l_nkv, l_hd, RMS_EPS_QK)
    else:
        gpu_rmsnorm_no_weight_mojo(ctx, d_q, S * l_nh, l_hd, RMS_EPS_QK, QK_NORM_Q_SCALE)
        gpu_rmsnorm_no_weight_mojo(ctx, d_k_new, S * l_nkv, l_hd, RMS_EPS_QK, 1.0)

    # Stage 26: selected Q row after Q norm, before RoPE.
    if debug_stage == 26 and debug_out != 0 and debug_row >= 0 and debug_row < S:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_q + UInt64(debug_row * l_qd * 4), l_qd * 4, 2,
        )

    # 5. RoPE (per-token absolute position)
    # Same profile rules as the single-token path. These two sites MUST stay in step:
    # a divergence is a prefill/decode split that only shows up as drift deep into a
    # long generation.
    @parameter
    if TARGET_ROPE_YARN:
        if is_full:
            gpu_yarn_rope_batched_mojo(
                ctx, d_q, base_pos, S, l_nh, l_hd, ROPE_THETA_FULL,
                TARGET_YARN_FACTOR, TARGET_YARN_ORIGINAL_MAX,
                TARGET_YARN_BETA_FAST, TARGET_YARN_BETA_SLOW,
                TARGET_YARN_ATTN_FACTOR,
            )
            gpu_yarn_rope_batched_mojo(
                ctx, d_k_new, base_pos, S, l_nkv, l_hd, ROPE_THETA_FULL,
                TARGET_YARN_FACTOR, TARGET_YARN_ORIGINAL_MAX,
                TARGET_YARN_BETA_FAST, TARGET_YARN_BETA_SLOW,
                TARGET_YARN_ATTN_FACTOR,
            )
        else:
            gpu_rope_batched_mojo(
                ctx, d_q, base_pos, S, l_nh, l_hd,
                ROPE_THETA_SLIDING, l_hd,
            )
            gpu_rope_batched_mojo(
                ctx, d_k_new, base_pos, S, l_nkv, l_hd,
                ROPE_THETA_SLIDING, l_hd,
            )
    else:
        if is_full:
            @parameter
            if not FULL_LAYERS_NOPE:
                var rdim = l_hd // 4 if ROPE_FULL_PARTIAL_DIM else l_hd
                gpu_rope_batched_mojo(ctx, d_q, base_pos, S, l_nh, l_hd, ROPE_THETA_FULL, rdim)
                gpu_rope_batched_mojo(ctx, d_k_new, base_pos, S, l_nkv, l_hd, ROPE_THETA_FULL, rdim)
        else:
            gpu_rope_batched_mojo(ctx, d_q, base_pos, S, l_nh, l_hd, ROPE_THETA_SLIDING, l_hd)
            gpu_rope_batched_mojo(ctx, d_k_new, base_pos, S, l_nkv, l_hd, ROPE_THETA_SLIDING, l_hd)
    if debug_out != 0 and debug_row >= 0 and debug_row < S:
        if debug_stage == 30:
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out),
                d_k_new + UInt64(debug_row * l_kvd * 4), l_kvd * 4, 2,
            )
        elif debug_stage == 31:
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out),
                d_v_new + UInt64(debug_row * l_kvd * 4), l_kvd * 4, 2,
            )
    return attn_gate_ready


def apply_output_and_mlp_batched(
    ctx: DeviceContext,
    d_x: UInt64,                    # [S, d] residual
    d_attn_out: UInt64,             # [S, l_qd] attention output (filled by caller)
    d_attn_in: UInt64,              # [S, d] input-normalized hidden used by attention gate
    d_attn_gate: UInt64,            # [S, l_qd] gate projection scratch
    d_o_out: UInt64,                # [S, d] scratch
    d_pn: UInt64,                   # [S, d] scratch
    d_gate: UInt64,                 # [S, ff]
    d_up: UInt64,                   # [S, ff]
    d_mlp_v: UInt64,                # [S, ff]
    d_down: UInt64,                 # [S, d]
    handle: UInt64,
    d_attn_gw: UInt64,
    d_ow: UInt64,
    d_post_attn_norm_w: UInt64,
    d_pre_ff_norm_w: UInt64,
    d_post_ff_norm_w: UInt64,
    d_gw: UInt64,
    d_uw: UInt64,
    d_dw: UInt64,
    d_scratch_in_bf16: UInt64,      # >= S*max(d,ff)*2 bytes
    layer_scalar: Float32,
    d: Int,
    ff: Int,
    l_qd: Int,
    S: Int,
    layer_idx: Int,
    d_q4_scratch: UInt64 = 0,       # weight-bf16 scratch; !=0 => d_ow/d_gw/d_uw/d_dw are Q4 blobs
    g_o: Float32 = 0.0,             # NVFP4 per-tensor global scales (0 = bf16/Q4)
    g_attn_g: Float32 = 0.0,
    g_g: Float32 = 0.0,
    g_u: Float32 = 0.0,
    g_d: Float32 = 0.0,
    ag_o: Float32 = 0.0,
    ag_attn_g: Float32 = 0.0,
    ag_g: Float32 = 0.0,
    ag_u: Float32 = 0.0,
    ag_d: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    debug_row: Int = -1,
    debug_stage: Int = -1,
    debug_out: Int64 = 0,
    prof_stages: Bool = False,
    attn_gate_ready: Bool = False,
) raises:
    """Batched steps 8-13 over S tokens. Matmuls are GEMM (M=S); norms are
    per-row batched; residual/gelu/mul/scalar are pure-elementwise over S*n."""
    @parameter
    if HAS_LINEAR_ATTENTION:
        if not attn_gate_ready:
            raise Error("Qwen softmax attention gate was not produced by q_proj")
        if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 0:
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out),
                d_attn_out + UInt64(debug_row * l_qd * 4), l_qd * 4, 2,
            )
        if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 27:
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out),
                d_attn_gate + UInt64(debug_row * l_qd * 4), l_qd * 4, 2,
            )
        gpu_sigmoid_mul_inplace_mojo(ctx, d_attn_out, d_attn_gate, S * l_qd)
        if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 28:
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out),
                d_attn_out + UInt64(debug_row * l_qd * 4), l_qd * 4, 2,
            )
        _mm_dev_batched(ctx, handle, d_o_out, d_attn_out, d_ow,
                        d_scratch_in_bf16, S, l_qd, d, d_q4_scratch,
                        g_o, ag_o, w4a4, layer_idx, "qwen_o_proj")
        if debug_out != 0 and debug_row >= 0 and debug_row < S and (
            debug_stage == 1 or debug_stage == 2
        ):
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out), d_o_out + UInt64(debug_row * d * 4), d * 4, 2
            )
        gpu_residual_add_mojo(ctx, d_x, d_o_out, S * d)
        if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 3:
            ctx.synchronize()
            cuda_memcpy(
                UInt64(debug_out), d_x + UInt64(debug_row * d * 4), d * 4, 2
            )
        apply_qwen_mlp_batched(
            ctx, d_x, d_pn, d_gate, d_up, d_mlp_v, d_down, handle,
            d_post_attn_norm_w, d_gw, d_uw, d_dw, d_scratch_in_bf16,
            S, d, ff, layer_idx, d_q4_scratch,
            g_g, g_u, g_d, ag_g, ag_u, ag_d, w4a4,
            debug_row, debug_stage, debug_out,
        )
        return

    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 0:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_attn_out + UInt64(debug_row * l_qd * 4), l_qd * 4, 2)

    var _pt0 = 0
    var t_o = 0; var t_small1 = 0; var t_gate = 0; var t_up = 0
    var t_act = 0; var t_down = 0; var t_small2 = 0
    if prof_stages:
        ctx.synchronize(); _pt0 = Int(perf_counter_ns())
    # ATTENTION GATE (profile capability, Muse only). Projected from the
    # input-normalized layer input, then sigmoid-multiplied into the attention output
    # BEFORE the O projection. Gemma-4 has no such gate, so on that profile this
    # compiles away entirely and the O projection consumes d_attn_out untouched —
    # which is why the Gemma-4 numbers must come out bit-identical.
    @parameter
    if HAS_ATTN_GATE:
        if not attn_gate_ready:
            _mm_dev_batched(ctx, handle, d_attn_gate, d_attn_in, d_attn_gw,
                            d_scratch_in_bf16, S, d, l_qd, d_q4_scratch,
                            g_attn_g, ag_attn_g, w4a4, layer_idx, "attention_gate")
        gpu_sigmoid_mul_inplace_mojo(ctx, d_attn_out, d_attn_gate, S * l_qd)
    # O projection → post-attn norm → residual
    _mm_dev_batched(ctx, handle, d_o_out, d_attn_out, d_ow, d_scratch_in_bf16, S, l_qd, d, d_q4_scratch, g_o, ag_o, w4a4, layer_idx, "o_proj")
    if prof_stages:
        ctx.synchronize(); t_o = Int(perf_counter_ns()) - _pt0; _pt0 = Int(perf_counter_ns())
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 1:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_o_out + UInt64(debug_row * d * 4), d * 4, 2)
    gpu_rmsnorm_inplace_batched_mojo(ctx, d_o_out, d_post_attn_norm_w, d, S, RMS_EPS_POST_ATTN)
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 2:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_o_out + UInt64(debug_row * d * 4), d * 4, 2)
    gpu_residual_add_mojo(ctx, d_x, d_o_out, S * d)
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 3:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_x + UInt64(debug_row * d * 4), d * 4, 2)

    if prof_stages:
        ctx.synchronize(); t_small1 = Int(perf_counter_ns()) - _pt0; _pt0 = Int(perf_counter_ns())
    # 10-12. Pre-FF norm → gate, up → GeGLU → down → post-FF norm → residual
    @parameter
    if NORM_STYLE_POST:
        cuda_memcpy(d_pn, d_x, S * d * 4, 3)
    else:
        gpu_rmsnorm_batched_mojo(ctx, d_x, d_pn, d_pre_ff_norm_w, d, S, RMS_EPS_PRE_FF)
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 4:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_pn + UInt64(debug_row * d * 4), d * 4, 2)
    if prof_stages:
        ctx.synchronize(); _pt0 = Int(perf_counter_ns())
    _mm_dev_batched(ctx, handle, d_gate, d_pn, d_gw, d_scratch_in_bf16, S, d, ff, d_q4_scratch, g_g, ag_g, w4a4, layer_idx, "mlp_gate")
    if prof_stages:
        ctx.synchronize(); t_gate = Int(perf_counter_ns()) - _pt0; _pt0 = Int(perf_counter_ns())
    _mm_dev_batched(ctx, handle, d_up, d_pn, d_uw, d_scratch_in_bf16, S, d, ff, d_q4_scratch, g_u, ag_u, w4a4, layer_idx, "mlp_up")
    if prof_stages:
        ctx.synchronize(); t_up = Int(perf_counter_ns()) - _pt0; _pt0 = Int(perf_counter_ns())
    @parameter
    if MLP_ACT_SILU:
        gpu_silu_mul_mojo(ctx, d_mlp_v, d_gate, d_up, S * ff)
    else:
        gpu_gelu_mul_mojo(ctx, d_mlp_v, d_gate, d_up, S * ff)
    if prof_stages:
        ctx.synchronize(); t_act = Int(perf_counter_ns()) - _pt0; _pt0 = Int(perf_counter_ns())
    _mm_dev_batched(ctx, handle, d_down, d_mlp_v, d_dw, d_scratch_in_bf16, S, ff, d, d_q4_scratch, g_d, ag_d, w4a4, layer_idx, "mlp_down")
    if prof_stages:
        ctx.synchronize(); t_down = Int(perf_counter_ns()) - _pt0; _pt0 = Int(perf_counter_ns())
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 5:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_down + UInt64(debug_row * d * 4), d * 4, 2)
    gpu_rmsnorm_inplace_batched_mojo(ctx, d_down, d_post_ff_norm_w, d, S, RMS_EPS_POST_FF)
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 6:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_down + UInt64(debug_row * d * 4), d * 4, 2)
    # 13. Residual + layer scalar.
    gpu_residual_add_scaled_mojo(ctx, d_x, d_down, S * d, layer_scalar)
    if prof_stages:
        ctx.synchronize(); t_small2 = Int(perf_counter_ns()) - _pt0
        print("[mlp-stages us] o=", t_o // 1000, " small1=", t_small1 // 1000,
              " gate=", t_gate // 1000, " up=", t_up // 1000, " act=", t_act // 1000,
              " down=", t_down // 1000, " small2=", t_small2 // 1000, " S=", S)
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 7:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_x + UInt64(debug_row * d * 4), d * 4, 2)


def apply_output_and_mlp_dev(
    ctx: DeviceContext,
    d_x: UInt64,                    # residual stream on device [d]
    d_attn_out: UInt64,             # attention output [l_qd]
    d_attn_in: UInt64,              # input-normalized hidden used by attention gate
    d_attn_gate: UInt64,            # [l_qd] gate projection scratch
    d_o_out: UInt64,                # O proj output [d]
    d_pn: UInt64,                   # pre-FF norm scratch [d]
    d_gate: UInt64,                 # gate output [ff]
    d_up: UInt64,                   # up output [ff]
    d_mlp_v: UInt64,                # gate * up [ff]
    d_down: UInt64,                 # down output [d]
    handle: UInt64,
    d_attn_gw: UInt64,
    d_ow: UInt64,
    d_post_attn_norm_w: UInt64,
    d_pre_ff_norm_w: UInt64,
    d_post_ff_norm_w: UInt64,
    d_gw: UInt64,
    d_uw: UInt64,
    d_dw: UInt64,
    d_scratch_in_bf16: UInt64,
    layer_scalar: Float32,
    d: Int,
    ff: Int,
    l_qd: Int,
    layer_idx: Int,
    d_q4_scratch: UInt64 = 0,
    g_o: Float32 = 0.0,            # NVFP4 per-tensor global scales (0 = bf16/Q4)
    g_attn_g: Float32 = 0.0,
    g_g: Float32 = 0.0,
    g_u: Float32 = 0.0,
    g_d: Float32 = 0.0,
    ag_o: Float32 = 0.0,
    ag_attn_g: Float32 = 0.0,
    ag_g: Float32 = 0.0,
    ag_u: Float32 = 0.0,
    ag_d: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    attn_gate_ready: Bool = False,
    debug_stage: Int = -1,
    debug_out: Int64 = 0,
    budget_prof: Bool = False,
) raises:
    """Steps 8-13 of a Gemma-4 layer forward pass, fully on device."""
    @parameter
    if HAS_LINEAR_ATTENTION:
        if not attn_gate_ready:
            raise Error("Qwen softmax attention gate was not produced by q_proj")
        if debug_out != 0 and debug_stage == 0:
            ctx.synchronize()
            cuda_memcpy(UInt64(debug_out), d_attn_out, l_qd * 4, 2)
        if debug_out != 0 and debug_stage == 27:
            ctx.synchronize()
            cuda_memcpy(UInt64(debug_out), d_attn_gate, l_qd * 4, 2)
        gpu_sigmoid_mul_inplace_mojo(ctx, d_attn_out, d_attn_gate, l_qd)
        if debug_out != 0 and debug_stage == 28:
            ctx.synchronize()
            cuda_memcpy(UInt64(debug_out), d_attn_out, l_qd * 4, 2)
        _mm_dev_batched(ctx, handle, d_o_out, d_attn_out, d_ow,
                        d_scratch_in_bf16, 1, l_qd, d, d_q4_scratch,
                        g_o, ag_o, w4a4, layer_idx, "qwen_o_proj")
        if debug_out != 0 and (debug_stage == 1 or debug_stage == 2):
            ctx.synchronize()
            cuda_memcpy(UInt64(debug_out), d_o_out, d * 4, 2)
        gpu_residual_add_mojo(ctx, d_x, d_o_out, d)
        if debug_out != 0 and debug_stage == 3:
            ctx.synchronize()
            cuda_memcpy(UInt64(debug_out), d_x, d * 4, 2)
        apply_qwen_mlp_batched(
            ctx, d_x, d_pn, d_gate, d_up, d_mlp_v, d_down, handle,
            d_post_attn_norm_w, d_gw, d_uw, d_dw, d_scratch_in_bf16,
            1, d, ff, layer_idx, d_q4_scratch,
            g_g, g_u, g_d, ag_g, ag_u, ag_d, w4a4,
            0, debug_stage, debug_out,
        )
        return

    if debug_out != 0 and debug_stage == 0:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_attn_out, l_qd * 4, 2)

    # ATTENTION GATE (profile capability, Muse only) — see the batched path above.
    # `attn_gate_ready` means the grouped QKV+gate GEMM already produced it, so the
    # standalone projection would be redundant work, not a correctness choice.
    @parameter
    if HAS_ATTN_GATE:
        if not attn_gate_ready:
            _mm_dev(ctx, handle, d_attn_gate, d_attn_in, d_attn_gw,
                    d_scratch_in_bf16, d, l_qd, d_q4_scratch,
                    g_attn_g, ag_attn_g, w4a4, layer_idx, "attention_gate")
        if budget_prof: cuda_budget_mark(ctx, 12)
        gpu_sigmoid_mul_inplace_mojo(ctx, d_attn_out, d_attn_gate, l_qd)
        if budget_prof: cuda_budget_mark(ctx, 13)
    # O projection → post-attn norm → residual
    _mm_dev(ctx, handle, d_o_out, d_attn_out, d_ow, d_scratch_in_bf16, l_qd, d, d_q4_scratch, g_o, ag_o, w4a4, layer_idx, "o_proj")
    if budget_prof: cuda_budget_mark(ctx, 3)
    if debug_out != 0 and debug_stage == 1:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_o_out, d * 4, 2)
    gpu_rmsnorm_inplace_mojo(ctx, d_o_out, d_post_attn_norm_w, d, RMS_EPS_POST_ATTN)
    if debug_out != 0 and debug_stage == 2:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_o_out, d * 4, 2)
    gpu_residual_add_mojo(ctx, d_x, d_o_out, d)
    if debug_out != 0 and debug_stage == 3:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_x, d * 4, 2)

    # 10-12. Pre-FF norm → gate, up → GeGLU → down → post-FF norm → residual
    @parameter
    if NORM_STYLE_POST:
        cuda_memcpy(d_pn, d_x, d * 4, 3)
    else:
        gpu_rmsnorm_mojo(ctx, d_x, d_pn, d_pre_ff_norm_w, d, RMS_EPS_PRE_FF)
    if budget_prof: cuda_budget_mark(ctx, 4)
    if debug_out != 0 and debug_stage == 4:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_pn, d * 4, 2)
    # gate/up SHARE d_pn → dp4a Q4 path quantizes once + gemv-only ×2 (drops 1 quant/layer).
    if act_precision() == 8 and d_q4_scratch != 0 and g_g == 0.0:
        gpu_q8_quantize_dev(ctx, d_pn, d_scratch_in_bf16, d)
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_gate, d_scratch_in_bf16, d_gw, d, ff)
        if budget_prof: cuda_budget_mark(ctx, 5)
        gpu_matmul_q4_dp4a_gemv_dev[4](ctx, d_up, d_scratch_in_bf16, d_uw, d, ff)
        if budget_prof: cuda_budget_mark(ctx, 6)
    elif (
        w4a4.on and not _is_sm100_dev(ctx)
        and g_g != 0.0 and g_u != 0.0 and ag_g == ag_u
    ):
        # W4A4 prequant dedup: gate/up share d_pn — quantize once, 2 prequant GEMMs
        # (drops 1 act-quant launch/layer). Same calibrated-global guard as q/k/v.
        gpu_quant_act_nvfp4_dec(ctx, d_pn, w4a4.packed, w4a4.bs, w4a4.glob, d, ag_g)
        gpu_matmul_nvfp4_w4a4_prequant_dev(
            ctx, d_gate, d_gw, g_g, w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad, 1, d, ff)
        if budget_prof: cuda_budget_mark(ctx, 5)
        gpu_matmul_nvfp4_w4a4_prequant_dev(
            ctx, d_up, d_uw, g_u, w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad, 1, d, ff)
        if budget_prof: cuda_budget_mark(ctx, 6)
    else:
        _mm_dev(ctx, handle, d_gate, d_pn, d_gw, d_scratch_in_bf16, d, ff, d_q4_scratch, g_g, ag_g, w4a4, layer_idx, "mlp_gate")
        if budget_prof: cuda_budget_mark(ctx, 5)
        _mm_dev(ctx, handle, d_up, d_pn, d_uw, d_scratch_in_bf16, d, ff, d_q4_scratch, g_u, ag_u, w4a4, layer_idx, "mlp_up")
        if budget_prof: cuda_budget_mark(ctx, 6)
    @parameter
    if MLP_ACT_SILU:
        gpu_silu_mul_mojo(ctx, d_mlp_v, d_gate, d_up, ff)
    else:
        gpu_gelu_mul_mojo(ctx, d_mlp_v, d_gate, d_up, ff)
    if budget_prof: cuda_budget_mark(ctx, 7)
    _mm_dev(ctx, handle, d_down, d_mlp_v, d_dw, d_scratch_in_bf16, ff, d, d_q4_scratch, g_d, ag_d, w4a4, layer_idx, "mlp_down")
    if budget_prof: cuda_budget_mark(ctx, 8)
    if debug_out != 0 and debug_stage == 5:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_down, d * 4, 2)
    gpu_rmsnorm_inplace_mojo(ctx, d_down, d_post_ff_norm_w, d, RMS_EPS_POST_FF)
    if debug_out != 0 and debug_stage == 6:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_down, d * 4, 2)
    # 13. Residual + layer scalar.
    gpu_residual_add_scaled_mojo(ctx, d_x, d_down, d, layer_scalar)
    if budget_prof: cuda_budget_mark(ctx, 9)
    if debug_out != 0 and debug_stage == 7:
        ctx.synchronize()
        cuda_memcpy(UInt64(debug_out), d_x, d * 4, 2)


def apply_qwen_mlp_batched(
    ctx: DeviceContext,
    d_x: UInt64,
    d_normed: UInt64,
    d_gate: UInt64,
    d_up: UInt64,
    d_mlp_v: UInt64,
    d_down: UInt64,
    handle: UInt64,
    d_post_attn_norm_w: UInt64,
    d_gw: UInt64,
    d_uw: UInt64,
    d_dw: UInt64,
    d_scratch_in_bf16: UInt64,
    S: Int,
    d: Int,
    ff: Int,
    layer_idx: Int,
    d_q4_scratch: UInt64 = 0,
    g_g: Float32 = 0.0,
    g_u: Float32 = 0.0,
    g_d: Float32 = 0.0,
    ag_g: Float32 = 0.0,
    ag_u: Float32 = 0.0,
    ag_d: Float32 = 0.0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    debug_row: Int = -1,
    debug_stage: Int = -1,
    debug_out: Int64 = 0,
) raises:
    """Qwen decoder MLP: pre-norm -> SwiGLU -> down -> residual.

    Unlike Gemma/Muse there is no post-feedforward output norm and no layer
    scalar.  Keeping this separate prevents the four-norm sandwich from being
    silently applied to Qwen's two-pre-norm decoder layer.
    """
    gpu_rmsnorm_batched_mojo(
        ctx, d_x, d_normed, d_post_attn_norm_w, d, S,
        RMS_EPS_POST_ATTN,
    )
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 4:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_normed + UInt64(debug_row * d * 4), d * 4, 2
        )
    _mm_dev_batched(
        ctx, handle, d_gate, d_normed, d_gw, d_scratch_in_bf16,
        S, d, ff, d_q4_scratch, g_g, ag_g, w4a4, layer_idx,
        "qwen_mlp_gate",
    )
    _mm_dev_batched(
        ctx, handle, d_up, d_normed, d_uw, d_scratch_in_bf16,
        S, d, ff, d_q4_scratch, g_u, ag_u, w4a4, layer_idx,
        "qwen_mlp_up",
    )
    gpu_silu_mul_mojo(ctx, d_mlp_v, d_gate, d_up, S * ff)
    _mm_dev_batched(
        ctx, handle, d_down, d_mlp_v, d_dw, d_scratch_in_bf16,
        S, ff, d, d_q4_scratch, g_d, ag_d, w4a4, layer_idx,
        "qwen_mlp_down",
    )
    if debug_out != 0 and debug_row >= 0 and debug_row < S and (
        debug_stage == 5 or debug_stage == 6
    ):
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_down + UInt64(debug_row * d * 4), d * 4, 2
        )
    gpu_residual_add_mojo(ctx, d_x, d_down, S * d)
    if debug_out != 0 and debug_row >= 0 and debug_row < S and debug_stage == 7:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_x + UInt64(debug_row * d * 4), d * 4, 2
        )
