"""prefill_batch — the batched 60-layer prefill that populates the KV cache,
extracted from GemmaEngine so the struct stays a thin orchestrator (file-size
rule). Free def taking `mut self: GemmaEngine`; verbatim move, behavior-identical.
"""
from lib.gemma4_engine import (
    GemmaEngine, _env_float,
    D, FF, FULL_HD, HD, MAX_PROBE_TOKENS, NH, NKV, SLIDING_WINDOW, TOTAL_LAYERS, VOCAB,
    FULL_LAYERS_V_EQ_K,
)
from lib.gemma4_ops import rmsnorm, rmsnorm_no_weight
from lib.model_config import (
    EMBED_RMSNORM, EMBED_SQRT_SCALE, RMS_EPS_INPUT, RMS_EPS_FINAL,
    HAS_LINEAR_ATTENTION, ATTENTION_SCORE_SCALE,
    GDN_CONV_DIM, GDN_VALUE_DIM, GDN_NUM_V_HEADS,
)
from lib.cuda import cuda_malloc, cuda_free, cuda_download, cuda_memcpy, cuda_upload
from lib.gemma4_layer import (
    prepare_qkv_batched, apply_output_and_mlp_batched,
    apply_qwen_mlp_batched, W4A4Scratch,
)
from lib.gdn_layer import gdn_forward_batched
from lib.ops_gpu_mojo_reductions import gpu_rmsnorm_batched_mojo
from lib.ops_gpu_mojo import gpu_residual_add_mojo, gpu_scalar_mul_mojo
from lib.kv_cache_quant import gpu_append_quant_kv_i8, gpu_dequant_kv_i8_layer
from lib.kv_cache_quant import gpu_append_quant_kv_i4, gpu_append_quant_kv_i4_with_q8, gpu_dequant_kv_i4_layer, gpu_dequant_kv_i4_to_q8_layer, debug_kv_i4_source_to_f32_dev
from lib.attention_gpu import append_kv_gpu_dev, attention_gpu_fp32_dev
from lib.attention_gpu import append_kv_gpu_bf16_dev, attention_gpu_bf16_dev
from lib.attention_gpu_int8 import attention_gpu_int8_dev, attention_decode_order_multirow_int8, debug_q_i8_to_f32_dev, debug_kv_i8_view_to_f32_dev, debug_qk_scores_int8_dev
from lib.attention_gpu_int8 import attention_prefill_batched_int8, attention_prefill_prefix_batched_int8
from lib.batched_attn_gpu import attention_prefill_batched_fp32, attention_prefill_batched_bf16
from lib.batched_attn_gpu import attention_prefill_prefix_batched_fp32, attention_prefill_prefix_batched_bf16
from lib.batched_attn_gpu import gpu_append_kv_batched_bf16
from lib.engine_init import _strict_q4, _flag_violation
from lib.engine_init import VIOL_BF16_ATTN_BATCHED, VIOL_FP32_ATTN_BATCHED, VIOL_BF16_ATTN_DECODE, VIOL_FP32_ATTN_DECODE
from lib.q4_gemv_dp4a import act_precision, gpu_matmul_q4_dp4a_dev, gpu_matmul_q4_s8_v4_gemv_dev
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_dev, _w4a4_sync_checkpoint
from lib.fp4_weights import gpu_matmul_nvfp4_fused_dev
from lib.fp4_gemv_v2 import (gpu_matmul_nvfp4_fused_v3_dev,
                            gpu_matmul_nvfp4_fused_v3_multirow_dev,
                            NVFP4_GEMV_SMAX)
from lib.cuda import cuda_sync
from std.ffi import external_call, c_size_t
from std.math import sqrt
from std.time import perf_counter_ns
from std.memory import UnsafePointer
from std.collections import List
from lib.dflash_drafter import DFlashDrafter, DFLASH_CTX_BATCH, DFLASH_TAPS, dflash_append_context_rows


def prefill_batch_impl(
    mut self: GemmaEngine,
    tokens: List[Int],
    start_pos: Int = 0,
    verify_g_out: Int64 = 0,
    eagle_tap_row: Int = -1,
    eagle_tap_rows: Int = 0,
    debug_qkv_layer: Int = -1,
    debug_qkv_row: Int = 0,
    debug_normed_out: Int64 = 0,
    debug_k_out: Int64 = 0,
    debug_v_out: Int64 = 0,
    debug_omlp_layer: Int = -1,
    debug_omlp_row: Int = 0,
    debug_omlp_stage: Int = -1,
    debug_omlp_out: Int64 = 0,
    dflash_context_append: Bool = False,
    chunk_prefix: Bool = False,
) raises:
    """Batched prefill: process `tokens` (the prompt MINUS its last token)
    through all 60 layers as a single [S,d] batch, populating the KV cache
    at positions start_pos..start_pos+S-1 and leaving cache_lens[layer] = start_pos+S.

    start_pos>0 is the KV-REUSE path (#2): append the suffix attending to the cached
    prefix [0,start_pos). Batched QKV/MLP GEMMs stay; attention falls to the per-token
    path (the batched S×S kernel is start_pos==0 only) at absolute positions start_pos+i,
    mirroring the decode loop. Supported on the kv_quant product path, SWA off. start_pos==0
    is byte-identical to the original (every offset term collapses to the old value).

    Produces logits only when verify_g_out is non-zero. A drafter can configure
    target mid-layer taps and fused verify can capture those hidden rows into
    the drafter's verify tap cache:
    eagle_tap_rows stores rows [0,eagle_tap_rows), and eagle_tap_row also arms
    the live [tap_count,D] tap sink directly.

    Scratch is allocated here sized to S and freed before return."""
    var S = len(tokens)
    if S <= 0:
        return
    var verify_prof = verify_g_out != 0 and _env_float("NOMOS_VERIFY_PROFILE", 0.0) > Float32(0.5)
    var t_verify_total0 = Int(perf_counter_ns()) if verify_prof else 0
    var t_alloc0 = t_verify_total0
    var t_alloc: Int = 0
    var t_embed: Int = 0

    # KV-reuse offset path (#2): only on the kv_quant product path, SWA off. The batched
    # S×S attention assumes start at 0; offset uses the per-token path below. Fail loud.
    if start_pos > 0 and not self.kv_quant:
        raise Error("offset prefill (KV reuse) requires kv_quant (product path)")
    if start_pos > 0 and self.kv_swa:
        raise Error("offset prefill + SWA not yet supported (needs #430/#431 ring truncation)")

    # SWA prompts longer than the ring use a transient linear quantized cache to compute
    # every query's exact sliding-window attention. The persistent ring is populated
    # directly from the final retained window, so each modulo slot is written once.

    # Worst-case per-layer dims (full-attention layers): qd=NH*FULL_HD, kvd=NKV*FULL_HD.
    alias QD_MAX = NH * FULL_HD     # 32*512 = 16384
    alias KVD_MAX = NKV * FULL_HD   # 16*512 = 8192
    var d_x_b = cuda_malloc(S * D * 4)
    var d_normed_b = cuda_malloc(S * D * 4)
    var d_q_b = cuda_malloc(S * QD_MAX * 4)
    var d_k_b = cuda_malloc(S * KVD_MAX * 4)
    var d_v_b = cuda_malloc(S * KVD_MAX * 4)
    var d_attn_b = cuda_malloc(S * QD_MAX * 4)
    var d_o_b = cuda_malloc(S * D * 4)
    var d_pn_b = cuda_malloc(S * D * 4)
    var d_gate_b = cuda_malloc(S * FF * 4)
    var d_up_b = cuda_malloc(S * FF * 4)
    var d_mlpv_b = cuda_malloc(S * FF * 4)
    var d_down_b = cuda_malloc(S * D * 4)
    var d_bf16_b = cuda_malloc(S * FF * 2)   # BF16 input scratch, max dim = FF
    if verify_prof:
        t_alloc = Int(perf_counter_ns()) - t_alloc0

    # Batched-attention path (NOMOS_BATCHED_PREFILL=1). Full-materialized causal
    # attention: the whole layer's attention in 5 launches instead of S per-token
    # calls. fp32-only for now (parity-gated vs the per-token fp32 path). Scratch:
    #   d_q_t / d_attn_t : [nkv,S,kvg*l_hd] = S*l_qd  (worst case S*QD_MAX)
    #   d_scores_big     : [nkv,S*kvg,S]    = NH*S*S  (the big one — 32*S^2 floats)
    var use_batched = _env_float("NOMOS_BATCHED_PREFILL", 0.0) > 0.5
    var use_int8_attn = self.kv_quant and self.kv_int4 and self.precision_bits <= 16 and act_precision() == 8
    # PREFIX-BATCHED attention engages for offset rows on two INDEPENDENT triggers:
    #  - verify's legacy route: NOMOS_VERIFY_BLOCK_ATTN env (unchanged semantics — for VERIFY,
    #    bf16-prefix is a measured approximation, 1/12; that env stays a verify-side decision).
    #  - chunk_prefix=True: the chunked-prefill driver's offset chunks (route A, 2026-08-03).
    #    EXPLICIT PARAMETER, never inferred from env, so verify calls into this same impl are
    #    untouched by NOMOS_PREFILL_CHUNK. For PREFILL the exactness bar is the shipped batched
    #    envelope (Gate B: the square-batched baseline itself diverges from per-token on 11/12,
    #    2 at token zero; chunked measured never-earlier, equal Hamming). int8 is EXCLUDED from
    #    the chunk trigger: attention_prefill_prefix_batched_int8's PV anchors quant blocks at
    #    the shared key base (Codex refutation, 2026-08-02) — int8 chunks keep the per-token
    #    fallback until route C (row-relative multirow port) lands.
    var use_prefix_block = (start_pos > 0 and _env_float("NOMOS_VERIFY_BLOCK_ATTN", 0.0) > 0.5) \
        or (start_pos > 0 and chunk_prefix and not use_int8_attn)
    var batched_fp32 = use_batched and self.precision_bits > 16
    var batched_int8 = use_batched and use_int8_attn
    var batched_bf16 = use_batched and self.precision_bits <= 16 and not use_int8_attn
    var prefix_fp32 = use_prefix_block and self.precision_bits > 16
    var prefix_int8 = use_prefix_block and use_int8_attn
    var prefix_bf16 = use_prefix_block and self.precision_bits <= 16 and not use_int8_attn
    var decode_order_verify_attn = prefix_int8 and _env_float("NOMOS_VERIFY_DECODE_ORDER_ATTN", 0.0) > Float32(0.5)
    var decode_order_multirow_attn = decode_order_verify_attn and _env_float("NOMOS_VERIFY_DECODE_ORDER_MULTIROW_ATTN", 0.0) > Float32(0.5)
    # Fast-exact Qwen verify: retain M-row weight reuse for the large
    # projections, but advance the recurrent/conv pools through the production
    # S=1 state boundary once per row.  Debug AB calls also use this flag so the
    # stage gate can adjudicate the candidate before the verifier enables it.
    var gdn_fast_exact = (
        HAS_LINEAR_ATTENTION
        and _env_float("NOMOS_VERIFY_GDN_FAST_EXACT", 0.0) > Float32(0.5)
        # Both the initial fused verify and a partial-commit replay are offset
        # forwards.  Replay intentionally has no logits/debug destination, so
        # output-pointer liveness is not a valid route discriminator.
        and start_pos > 0
    )
    var long_swa = self.kv_swa and S > SLIDING_WINDOW
    var need_block_scratch = (use_batched and not long_swa) or use_prefix_block
    var d_q_t = cuda_malloc(S * QD_MAX * 4) if need_block_scratch else UInt64(0)
    var d_attn_t = cuda_malloc(S * QD_MAX * 4) if need_block_scratch else UInt64(0)
    var d_scores_big = cuda_malloc(NH * S * S * 4) if (use_batched and not long_swa) else UInt64(0)
    var prefix_key_cap = start_pos + S
    var d_scores_prefix = cuda_malloc(NH * S * prefix_key_cap * 4) if use_prefix_block else UInt64(0)
    # bf16-batched extras: cast targets for Q and the scores matrix.
    var d_q_t_bf16 = cuda_malloc(S * QD_MAX * 2) if ((batched_bf16 and not long_swa) or prefix_bf16) else UInt64(0)
    var d_scores_bf16_big = cuda_malloc(NH * S * S * 2) if (batched_bf16 and not long_swa) else UInt64(0)
    var d_scores_bf16_prefix = cuda_malloc(NH * S * prefix_key_cap * 2) if prefix_bf16 else UInt64(0)
    # NOMOS_ROUTE_BANNER=1 -> print the RESOLVED prefill route. kv_swa silently selects a
    # different attention ALGORITHM here (long_swa), not just a KV layout, and on 2026-08-04 an
    # SWA A/B was read as a storage comparison when it was also square-batched vs per-token.
    # A benchmark must be able to prove which route ran instead of inferring it from a flag.
    if _env_float("NOMOS_ROUTE_BANNER", 0.0) > 0.5:
        var route = String("per-token")
        if use_batched and not long_swa: route = String("square-batched[NH*S*S]")
        elif use_prefix_block: route = String("prefix-block")
        elif long_swa: route = String("per-token(long_swa)")
        print("[prefill] route =", route, " S =", S, " start_pos =", start_pos,
              " long_swa =", long_swa, " use_batched =", use_batched,
              " scores_bytes =", NH * S * S * 4 if (use_batched and not long_swa) else 0)
    # A long sliding prefill needs historical K/V while producing intermediate query
    # rows, but that history must not become persistent state. Keep one layer's packed
    # cache transiently, reuse it across all 50 sliding layers, and commit only the final
    # SLIDING_WINDOW rows to the persistent ring.
    var d_swa_k_tmp = UInt64(0)
    var d_swa_v_tmp = UInt64(0)
    var d_swa_ks_tmp = UInt64(0)
    var d_swa_vs_tmp = UInt64(0)
    if long_swa:
        var tmp_slot = NKV * S * HD
        if self.kv_int4:
            tmp_slot = tmp_slot // 2
        d_swa_k_tmp = cuda_malloc(tmp_slot)
        d_swa_v_tmp = cuda_malloc(tmp_slot)
        var tmp_scale_blocks = (HD + 31) // 32 if self.kv_int4_block32 else 1
        var tmp_scale_rows = NKV * S
        var tmp_scale_bytes = (
            tmp_scale_rows * tmp_scale_blocks * 2 + tmp_scale_rows * 4
            if self.kv_int4_block32
            else tmp_scale_rows * 4
        )
        d_swa_ks_tmp = cuda_malloc(tmp_scale_bytes)
        d_swa_vs_tmp = cuda_malloc(tmp_scale_bytes)
    # NOTE the ABSENCE of `start_pos == 0` here (removed 2026-08-03, Codex root-cause). It made
    # only the FIRST chunk of a chunked prefill capture taps: chunks 1+ (start_pos>0) allocated no
    # tap buffer and appended nothing, so the drafter's context held just the first C rows of the
    # prompt and speculative acceptance flat-lined at depth (E=1.00 — the dead-drafter guard
    # caught it on the agent8k ceiling proof). The drafter append API owns its own contiguous
    # cursor (dflash_append_context_rows appends at current seq_len; capacity-checked), so
    # sequential chunk captures accumulate correctly WITHOUT any start_pos indexing here.
    # Continue-mode/KV-reuse offset prefill stays excluded by dflash_context_append itself, which
    # the caller only sets on fresh full-prompt requests.
    # Tap capture must match the drafter THIS BUILD was compiled for. A mismatch is a
    # configuration error, not a reason to quietly run without taps: with capture off the
    # drafter sees no target context, every block is rejected, and acceptance pins to E=1 —
    # which reads as "this drafter is bad", NOT "the taps never fired". That exact
    # misdiagnosis cost a full day on 2026-08-03, when dflash_capture was gated on
    # start_pos==0 and the drafter silently saw 256 of 5,871 tokens.
    var dflash_capture = (
        dflash_context_append
        and self.dflash_ptr != 0
        and self.drafter_tap_count == DFLASH_TAPS
    )
    if dflash_context_append and self.dflash_ptr != 0 and not dflash_capture:
        print("[engine] DFlash tap capture DISABLED: drafter reports",
              self.drafter_tap_count, "taps, this build expects", DFLASH_TAPS,
              "-- acceptance will pin to E=1. Rebuild with the matching model profile.")
    var d_dflash_prefill_taps = UInt64(0)
    if dflash_capture:
        d_dflash_prefill_taps = cuda_malloc(S * self.drafter_tap_count * D * 4)

    # ── Embeddings → host [S,d], profile-selected scaling ──────────
    var t_embed0 = Int(perf_counter_ns()) if verify_prof else 0
    var host_x = List[Float32](capacity=S * D)
    for _ in range(S * D):
        host_x.append(0.0)
    for i in range(S):
        var tok = tokens[i]
        _ = external_call["nomos_pread", c_size_t](
            self.embed_fd, UnsafePointer(to=host_x[i * D]),
            c_size_t(D * 4), c_size_t(tok * D * 4))
    @parameter
    if EMBED_RMSNORM:
        for i in range(S):
            rmsnorm_no_weight(host_x, i * D, D, RMS_EPS_INPUT)
    elif EMBED_SQRT_SCALE:
        var embed_scale = sqrt(Float32(D))
        for j in range(S * D):
            host_x[j] *= embed_scale
    cuda_upload(d_x_b, host_x)
    if verify_prof:
        cuda_sync()
        t_embed = Int(perf_counter_ns()) - t_embed0

    # ── per-phase profiling (NOMOS_PROFILE=1): qkv / kv-append / attn / mlp ──
    var prof = _env_float("NOMOS_PROFILE", 0.0) > Float32(0.5) or verify_prof
    var t_qkv: Int = 0
    var t_app: Int = 0
    var t_deq: Int = 0
    var t_attn: Int = 0
    var t_mlp: Int = 0
    var t_read_prep: Int = 0
    var t_read_lmhead: Int = 0
    var t_read_copy: Int = 0
    var route_square: Int = 0
    var route_prefix: Int = 0
    var route_pertoken: Int = 0
    var _t0: Int = 0

    # ── 60-layer batched forward, populating the KV cache ───────────
    for layer in range(TOTAL_LAYERS):
        var is_full = self.layer_is_full[layer]
        var l_nh = NH
        var l_nkv = self.layer_nkv[layer]
        var l_hd = self.layer_hd[layer]
        var i4_scale_blocks = (l_hd + 31) // 32 if self.kv_int4_block32 else 1
        var l_qd = self.layer_qd[layer]
        var l_kvd = self.layer_kvd[layer]
        var l_kvg = self.layer_kvg[layer]
        var ccap = self.layer_cache_cap[layer]   # #430: ring capacity (==max_seq unless SWA sliding layer)
        var v_eq_k = is_full and FULL_LAYERS_V_EQ_K
        var d_vw_or_zero: UInt64 = UInt64(0) if v_eq_k else self.d_vw[layer]
        var ring_prefill = long_swa and not is_full
        var d_k_compute_cache = UInt64(0)
        var d_v_compute_cache = UInt64(0)
        var d_ks_compute = UInt64(0)
        var d_vs_compute = UInt64(0)
        if self.kv_quant:
            d_k_compute_cache = d_swa_k_tmp if ring_prefill else self.d_k_cache_i8[layer]
            d_v_compute_cache = d_swa_v_tmp if ring_prefill else self.d_v_cache_i8[layer]
            d_ks_compute = d_swa_ks_tmp if ring_prefill else self.d_k_scales[layer]
            d_vs_compute = d_swa_vs_tmp if ring_prefill else self.d_v_scales[layer]
        var compute_cap = S if ring_prefill else ccap

        if HAS_LINEAR_ATTENTION and not is_full:
            var gdn_slot = self.gdn_state.gdn_slot_for_layer(layer)
            if gdn_slot < 0:
                raise Error("GDN layer missing ordered state slot")
            # Correctness-first per-layer work surfaces. Once parity is green,
            # hoist these to a reusable prefill workspace; allocation policy is
            # not part of the arithmetic contract.
            var d_gdn_qkv_raw = cuda_malloc(S * GDN_CONV_DIM * 4)
            var d_gdn_qkv_conv = cuda_malloc(S * GDN_CONV_DIM * 4)
            var d_gdn_z = cuda_malloc(S * GDN_VALUE_DIM * 4)
            var d_gdn_a = cuda_malloc(S * GDN_NUM_V_HEADS * 4)
            var d_gdn_b = cuda_malloc(S * GDN_NUM_V_HEADS * 4)
            var d_gdn_core = cuda_malloc(S * GDN_VALUE_DIM * 4)
            var d_gdn_out = cuda_malloc(S * D * 4)
            gpu_rmsnorm_batched_mojo(
                self.ctx, d_x_b, d_normed_b, self.d_in_norms[layer],
                D, S, RMS_EPS_INPUT,
            )
            if (layer == debug_omlp_layer and debug_omlp_stage == 18
                    and debug_omlp_out != 0):
                self.ctx.synchronize()
                cuda_memcpy(
                    UInt64(debug_omlp_out),
                    d_normed_b + UInt64(debug_omlp_row * D * 4), D * 4, 2,
                )
            gdn_forward_batched(
                self.ctx, self.handle, d_gdn_out, d_normed_b,
                d_gdn_qkv_raw, d_gdn_qkv_conv, d_gdn_z,
                d_gdn_a, d_gdn_b, d_gdn_core, d_bf16_b,
                self.gdn_weights.in_proj_qkv[gdn_slot],
                self.gdn_weights.in_proj_z[gdn_slot],
                self.gdn_weights.in_proj_a[gdn_slot],
                self.gdn_weights.in_proj_b[gdn_slot],
                self.gdn_weights.out_proj[gdn_slot],
                self.gdn_weights.conv1d[gdn_slot],
                self.gdn_weights.a_log[gdn_slot],
                self.gdn_weights.dt_bias[gdn_slot],
                self.gdn_weights.norm[gdn_slot],
                self.gdn_state.conv_ptr(gdn_slot),
                self.gdn_state.rec_ptr(gdn_slot), S,
                self.gdn_nvfp4,
                self.gdn_weights.qkv_gs[gdn_slot],
                self.gdn_weights.z_gs[gdn_slot],
                self.gdn_weights.out_gs[gdn_slot],
                self.gdn_weights.qkv_ags[gdn_slot],
                self.gdn_weights.z_ags[gdn_slot],
                self.gdn_weights.out_ags[gdn_slot],
                self.d_weight_bf16_scratch,
                W4A4Scratch(
                    self.d_w4a4_packed, self.d_w4a4_bs,
                    self.d_w4a4_global, self.d_w4a4_cpad,
                    self.w4a4_chunk_mpad, self.w4a4,
                    self.d_w4a4_bs_sf, self.d_w4a4_wbs_sf,
                ),
                debug_omlp_row if layer == debug_omlp_layer else -1,
                debug_omlp_stage if layer == debug_omlp_layer else -1,
                debug_omlp_out if layer == debug_omlp_layer else Int64(0),
                gdn_fast_exact,
            )
            # GDN stage surfaces reuse the existing O/MLP debug ABI.
            if layer == debug_omlp_layer and debug_omlp_out != 0:
                self.ctx.synchronize()
                if debug_omlp_stage == 13:
                    cuda_memcpy(
                        UInt64(debug_omlp_out),
                        d_gdn_core + UInt64(debug_omlp_row * GDN_VALUE_DIM * 4),
                        GDN_VALUE_DIM * 4, 2,
                    )
                elif debug_omlp_stage == 14:
                    cuda_memcpy(
                        UInt64(debug_omlp_out),
                        d_gdn_z + UInt64(debug_omlp_row * GDN_VALUE_DIM * 4),
                        GDN_VALUE_DIM * 4, 2,
                    )
                elif debug_omlp_stage == 15:
                    cuda_memcpy(
                        UInt64(debug_omlp_out),
                        d_gdn_out + UInt64(debug_omlp_row * D * 4),
                        D * 4, 2,
                    )
            gpu_residual_add_mojo(self.ctx, d_x_b, d_gdn_out, S * D)
            if (layer == debug_omlp_layer and debug_omlp_stage == 16
                    and debug_omlp_out != 0):
                self.ctx.synchronize()
                cuda_memcpy(
                    UInt64(debug_omlp_out),
                    d_x_b + UInt64(debug_omlp_row * D * 4), D * 4, 2,
                )
            apply_qwen_mlp_batched(
                self.ctx, d_x_b, d_pn_b, d_gate_b, d_up_b,
                d_mlpv_b, d_down_b, self.handle,
                self.d_post_attn_norms[layer], self.d_gw[layer],
                self.d_uw[layer], self.d_dw[layer], d_bf16_b,
                S, D, FF, layer, self.d_weight_bf16_scratch,
                self.d_gw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_uw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_dw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_gw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_uw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_dw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                W4A4Scratch(
                    self.d_w4a4_packed, self.d_w4a4_bs,
                    self.d_w4a4_global, self.d_w4a4_cpad,
                    self.w4a4_chunk_mpad, self.w4a4,
                    self.d_w4a4_bs_sf, self.d_w4a4_wbs_sf,
                ),
            )
            if (layer == debug_omlp_layer and debug_omlp_stage == 17
                    and debug_omlp_out != 0):
                self.ctx.synchronize()
                cuda_memcpy(
                    UInt64(debug_omlp_out),
                    d_x_b + UInt64(debug_omlp_row * D * 4), D * 4, 2,
                )

            # Keep the generic all-layer/debug tap contract valid for GDN
            # layers too.  This branch used to `continue` before the common
            # tap block below, leaving deterministic zeros that looked like a
            # layer-output failure even when the state-producing scan matched.
            if self.drafter_tap_count > 0:
                var tap_slot = self.drafter_tap_slot(layer)
                if tap_slot >= 0:
                    var rows_to_copy = eagle_tap_rows
                    if rows_to_copy < 0: rows_to_copy = 0
                    if rows_to_copy > S: rows_to_copy = S
                    var copy_live_tap = (
                        self.d_eagle3_tap_out != 0
                        and eagle_tap_row >= 0 and eagle_tap_row < S
                    )
                    var copy_row_taps = (
                        self.d_eagle3_tap_rows_out != 0 and rows_to_copy > 0
                    )
                    if copy_live_tap or copy_row_taps or dflash_capture:
                        self.ctx.synchronize()
                        if copy_live_tap:
                            cuda_memcpy(
                                self.d_eagle3_tap_out + UInt64(tap_slot * D * 4),
                                d_x_b + UInt64(eagle_tap_row * D * 4),
                                D * 4, 3,
                            )
                        if copy_row_taps:
                            for r in range(rows_to_copy):
                                cuda_memcpy(
                                    self.d_eagle3_tap_rows_out
                                    + UInt64((r * self.drafter_tap_count + tap_slot) * D * 4),
                                    d_x_b + UInt64(r * D * 4), D * 4, 3,
                                )
                        if dflash_capture:
                            for r in range(S):
                                cuda_memcpy(
                                    d_dflash_prefill_taps
                                    + UInt64((r * self.drafter_tap_count + tap_slot) * D * 4),
                                    d_x_b + UInt64(r * D * 4), D * 4, 3,
                                )
                        cuda_sync()
            cuda_free(d_gdn_qkv_raw); cuda_free(d_gdn_qkv_conv)
            cuda_free(d_gdn_z); cuda_free(d_gdn_a); cuda_free(d_gdn_b)
            cuda_free(d_gdn_core); cuda_free(d_gdn_out)
            continue

        if prof: cuda_sync()
        _t0 = Int(perf_counter_ns())
        var attn_gate_ready = prepare_qkv_batched(
            self.ctx, d_x_b, d_normed_b, d_q_b, d_k_b, d_v_b,
            d_up_b, d_gate_b, self.handle,
            self.d_in_norms[layer], self.d_q_norms[layer], self.d_k_norms[layer],
            self.d_qw[layer], self.d_kw[layer], d_vw_or_zero, d_bf16_b,
            is_full, D, l_nh, l_nkv, l_hd, l_qd, l_kvd, S, layer,
            start_pos,
            self.d_weight_bf16_scratch,
            self.d_qw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_kw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            Float32(0.0) if v_eq_k else (self.d_vw_gs[layer] if self.weight_nvfp4 else Float32(0.0)),
            self.d_qw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_kw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            Float32(0.0) if v_eq_k else (self.d_vw_ags[layer] if self.weight_nvfp4 else Float32(0.0)),
            W4A4Scratch(self.d_w4a4_packed, self.d_w4a4_bs, self.d_w4a4_global, self.d_w4a4_cpad, self.w4a4_chunk_mpad, self.w4a4, self.d_w4a4_bs_sf, self.d_w4a4_wbs_sf),
            debug_omlp_row if layer == debug_omlp_layer else -1,
            debug_omlp_stage if layer == debug_omlp_layer else -1,
            debug_omlp_out if layer == debug_omlp_layer else Int64(0),
        )
        if layer == debug_omlp_layer and (
            debug_omlp_stage == 8 or debug_omlp_stage == 26
        ):
            if debug_omlp_row < 0 or debug_omlp_row >= S:
                raise Error("debug_omlp_row out of range")
            self.ctx.synchronize()
            if debug_omlp_stage == 8 and debug_omlp_out != 0:
                cuda_memcpy(
                    UInt64(debug_omlp_out),
                    d_q_b + UInt64(debug_omlp_row * l_qd * 4),
                    l_qd * 4,
                    2,
                )
            cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
            cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
            cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
            cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
            cuda_free(d_bf16_b)
            if need_block_scratch:
                cuda_free(d_q_t); cuda_free(d_attn_t)
            if use_batched and not long_swa:
                cuda_free(d_scores_big)
            if use_prefix_block:
                cuda_free(d_scores_prefix)
            if batched_bf16 and not long_swa:
                cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
            elif prefix_bf16:
                cuda_free(d_q_t_bf16)
            if prefix_bf16:
                cuda_free(d_scores_bf16_prefix)
            if long_swa:
                cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
            return
        if layer == debug_omlp_layer and debug_omlp_stage == 9:
            if debug_omlp_row < 0 or debug_omlp_row >= S:
                raise Error("debug_omlp_row out of range")
            var q_dbg_len = l_qd + l_nh * ((l_hd + 31) // 32)
            if debug_omlp_out != 0:
                var d_q_dbg = cuda_malloc(q_dbg_len * 4)
                debug_q_i8_to_f32_dev(
                    self.ctx,
                    d_q_b + UInt64(debug_omlp_row * l_qd * 4),
                    d_q_dbg,
                    l_nh,
                    l_hd,
                )
                self.ctx.synchronize()
                cuda_memcpy(UInt64(debug_omlp_out), d_q_dbg, q_dbg_len * 4, 2)
                cuda_free(d_q_dbg)
            cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
            cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
            cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
            cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
            cuda_free(d_bf16_b)
            if need_block_scratch:
                cuda_free(d_q_t); cuda_free(d_attn_t)
            if use_batched and not long_swa:
                cuda_free(d_scores_big)
            if use_prefix_block:
                cuda_free(d_scores_prefix)
            if batched_bf16 and not long_swa:
                cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
            elif prefix_bf16:
                cuda_free(d_q_t_bf16)
            if prefix_bf16:
                cuda_free(d_scores_bf16_prefix)
            if long_swa:
                cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
            return
        if layer == debug_qkv_layer:
            if debug_qkv_row < 0 or debug_qkv_row >= S:
                raise Error("debug_qkv_row out of range")
            self.ctx.synchronize()
            if debug_normed_out != 0:
                cuda_memcpy(
                    UInt64(debug_normed_out),
                    d_normed_b + UInt64(debug_qkv_row * D * 4),
                    D * 4,
                    2,
                )
            if debug_k_out != 0:
                cuda_memcpy(
                    UInt64(debug_k_out),
                    d_k_b + UInt64(debug_qkv_row * l_kvd * 4),
                    l_kvd * 4,
                    2,
                )
            if debug_v_out != 0:
                cuda_memcpy(
                    UInt64(debug_v_out),
                    d_v_b + UInt64(debug_qkv_row * l_kvd * 4),
                    l_kvd * 4,
                    2,
                )
            cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
            cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
            cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
            cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
            cuda_free(d_bf16_b)
            if need_block_scratch:
                cuda_free(d_q_t); cuda_free(d_attn_t)
            if use_batched and not long_swa:
                cuda_free(d_scores_big)
            if use_prefix_block:
                cuda_free(d_scores_prefix)
            if batched_bf16 and not long_swa:
                cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
            elif prefix_bf16:
                cuda_free(d_q_t_bf16)
            if prefix_bf16:
                cuda_free(d_scores_bf16_prefix)
            if long_swa:
                cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
            return
        # Scale Q only after stages 26/8 have captured the HF-visible post-norm
        # and post-RoPE tensors. This is algebraically the score scaling applied
        # by HF eager_attention_forward and avoids touching K/V cache contents.
        @parameter
        if ATTENTION_SCORE_SCALE != 1.0:
            gpu_scalar_mul_mojo(
                self.ctx, d_q_b, ATTENTION_SCORE_SCALE, S * l_qd
            )
        if prof:
            cuda_sync()
            t_qkv += Int(perf_counter_ns()) - _t0
            _t0 = Int(perf_counter_ns())

        # Append all S tokens' K/V to the cache at positions 0..S-1.
        # The decode-order verify route below intentionally skips this batch append:
        # it appends row i immediately before row i's dequant+attention, matching decode.
        # BF16 path: ONE batched launch per layer (was S per-token launches —
        # ~78% of prefill time, the launch-overhead wall). 2026-06-15 perf.
        if not decode_order_verify_attn:
            if self.kv_quant:
                # INT8/INT4 KV: quantize+scatter all S new tokens at positions 0..S-1.
                if self.kv_int4:
                    gpu_append_quant_kv_i4(
                        self.ctx, d_k_b, d_v_b,
                        d_k_compute_cache, d_v_compute_cache,
                        d_ks_compute, d_vs_compute,
                        S, l_nkv, l_hd, l_kvd, compute_cap, l_hd, start_pos, i4_scale_blocks,
                    )
                else:
                    gpu_append_quant_kv_i8(
                        self.ctx, d_k_b, d_v_b,
                        d_k_compute_cache, d_v_compute_cache,
                        d_ks_compute, d_vs_compute,
                        S, l_nkv, l_hd, l_kvd, compute_cap, l_hd, start_pos,
                    )
                if ring_prefill:
                    var tail = S - ccap
                    if self.kv_int4:
                        gpu_append_quant_kv_i4(
                            self.ctx,
                            d_k_b + UInt64(tail * l_kvd * 4),
                            d_v_b + UInt64(tail * l_kvd * 4),
                            self.d_k_cache_i8[layer], self.d_v_cache_i8[layer],
                            self.d_k_scales[layer], self.d_v_scales[layer],
                            ccap, l_nkv, l_hd, l_kvd, ccap, l_hd,
                            start_pos + tail, i4_scale_blocks,
                        )
                    else:
                        gpu_append_quant_kv_i8(
                            self.ctx,
                            d_k_b + UInt64(tail * l_kvd * 4),
                            d_v_b + UInt64(tail * l_kvd * 4),
                            self.d_k_cache_i8[layer], self.d_v_cache_i8[layer],
                            self.d_k_scales[layer], self.d_v_scales[layer],
                            ccap, l_nkv, l_hd, l_kvd, ccap, l_hd,
                            start_pos + tail,
                        )
            elif self.precision_bits <= 16:
                gpu_append_kv_batched_bf16(
                    self.ctx, d_k_b, d_v_b,
                    self.d_k_cache[layer], self.d_v_cache[layer],
                    S, l_nkv, l_hd, l_kvd, self.max_seq, l_hd,
                )
            else:
                for i in range(S):
                    append_kv_gpu_dev(
                        self.d_k_cache[layer], self.d_v_cache[layer],
                        d_k_b + UInt64(i * l_kvd * 4), d_v_b + UInt64(i * l_kvd * 4),
                        l_nkv, l_hd, l_hd, self.max_seq, i,
                    )
        if not decode_order_verify_attn:
            self.cache_lens_fp32[layer] = start_pos + S
        if prof:
            cuda_sync()
            t_app += Int(perf_counter_ns()) - _t0
            _t0 = Int(perf_counter_ns())

        # Causal attention. Two paths, same result for keys 0..i:
        #   batched (NOMOS_BATCHED_PREFILL): whole layer in 5 launches.
        #   per-token (default/proven): token i attends to cache[0..i], output
        #     written DIRECTLY into d_attn_b[i] (no per-token D2D memcpy).
        # Gemma-4 sliding-window: local layers attend only to the last
        # SLIDING_WINDOW keys; full layers (layer%6==5) attend to all. window=0
        # means "no lower bound" (full attention).
        var l_window = 0 if is_full else SLIDING_WINDOW
        # Deq-trim (matches decode): sliding layers under SWA pack the deq scratch at
        # deq_slide_pitch (=SLIDING_WINDOW; S<=window is guaranteed for SWA prefill), full
        # layers at max_seq. attention reads the deq (kv_quant) at that pitch, else the raw
        # cache at max_seq. Byte-identical (same window keys, packed denser).
        var deq_pitch = self.max_seq if is_full else self.deq_slide_pitch
        var attn_pitch = deq_pitch if self.kv_quant else self.max_seq
        # INT4 KV in strict Q4 mode widens losslessly to int8 + per-row scale
        # for attention; legacy research paths still dequant to bf16/fp32.
        var d_k_read = self.d_k_cache[layer] if not self.kv_quant else self.d_k_deq
        var d_v_read = self.d_v_cache[layer] if not self.kv_quant else self.d_v_deq
        var d_k_scale_read = self.d_q_bf16
        var d_v_scale_read = self.d_scores_bf16
        if self.kv_quant and not decode_order_verify_attn and not ring_prefill:
            # #430: prefill window is the whole prompt (S<=SLIDING_WINDOW guaranteed
            # above for SWA), so klen=S, win_start=0 → ring slots [0,S) == linear [0,S).
            # ccap==max_seq off-SWA → byte-identical to the pre-#430 dequant.
            # Offset: dequant the FULL [0, start_pos+S) absolute range so the per-token
            # attention below can read the cached prefix keys. start_pos==0 → [0,S) as before.
            var deq_count = start_pos + S
            if use_int8_attn and not decode_order_verify_attn:
                gpu_dequant_kv_i4_to_q8_layer(self.ctx, d_k_compute_cache,
                    d_ks_compute, self.d_k_deq, d_k_scale_read, l_nkv, deq_count, l_hd, deq_pitch, compute_cap, 0, i4_scale_blocks)
                gpu_dequant_kv_i4_to_q8_layer(self.ctx, d_v_compute_cache,
                    d_vs_compute, self.d_v_deq, d_v_scale_read, l_nkv, deq_count, l_hd, deq_pitch, compute_cap, 0, i4_scale_blocks)
            elif self.kv_int4:
                gpu_dequant_kv_i4_layer(self.ctx, d_k_compute_cache,
                    d_ks_compute, self.d_k_deq, l_nkv, deq_count, l_hd, deq_pitch, compute_cap, 0, i4_scale_blocks)
                gpu_dequant_kv_i4_layer(self.ctx, d_v_compute_cache,
                    d_vs_compute, self.d_v_deq, l_nkv, deq_count, l_hd, deq_pitch, compute_cap, 0, i4_scale_blocks)
            else:
                gpu_dequant_kv_i8_layer(self.ctx, d_k_compute_cache,
                    d_ks_compute, self.d_k_deq, l_nkv, deq_count, l_hd, deq_pitch, compute_cap, 0)
                gpu_dequant_kv_i8_layer(self.ctx, d_v_compute_cache,
                    d_vs_compute, self.d_v_deq, l_nkv, deq_count, l_hd, deq_pitch, compute_cap, 0)
        if prof:
            cuda_sync()
            t_deq += Int(perf_counter_ns()) - _t0
            _t0 = Int(perf_counter_ns())
        # The normal batched S×S attention assumes queries START at position 0
        # (square causal). Offset/KV-reuse can optionally use the prefix-aware
        # block path for Unit F verify-forward; otherwise it falls to the proven
        # per-token cuBLAS oracle at absolute positions (mirrors decode).
        # #431 R2a: the batched S×S prefill attention is CAUSAL ([0,p]), NOT a per-query
        # sliding floor ([p-1023,p]) — fine for S<=window (causal==windowed), but for SWA
        # S>window it over-attends the early tokens (byte-parity gate caught this: diverged
        # @tok2). So SWA+S>window falls to the per-token oracle below, which windows per-query
        # (lo=apos-window+1) correctly. (Batched-windowed prefill = Round 2b, after the
        # per-query-mask verify.) S<=window keeps the fast batched path.
        var swa_pertoken = self.kv_swa and S > SLIDING_WINDOW
        if start_pos == 0 and batched_int8 and not swa_pertoken:
            if verify_prof: route_square += 1
            attention_prefill_batched_int8(
                self.ctx,
                d_q_b, d_q_t, d_scores_big, d_attn_t, d_attn_b,
                d_k_read, d_v_read, d_k_scale_read, d_v_scale_read,
                l_nh, l_nkv, l_hd, l_hd, attn_pitch, S, l_kvg, l_window,
            )
        elif start_pos == 0 and batched_bf16 and not swa_pertoken:
            if verify_prof: route_square += 1
            if _strict_q4(): _flag_violation(VIOL_BF16_ATTN_BATCHED)
            attention_prefill_batched_bf16(
                self.ctx, self.handle,
                d_q_b, d_q_t, d_q_t_bf16, d_scores_big, d_scores_bf16_big,
                d_attn_t, d_attn_b,
                d_k_read, d_v_read,
                l_nh, l_nkv, l_hd, l_hd, attn_pitch, S, l_kvg, l_window,
            )
        elif start_pos == 0 and batched_fp32 and not swa_pertoken:
            if verify_prof: route_square += 1
            if _strict_q4(): _flag_violation(VIOL_FP32_ATTN_BATCHED)
            attention_prefill_batched_fp32(
                self.ctx, self.handle,
                d_q_b, d_q_t, d_scores_big, d_attn_t, d_attn_b,
                d_k_read, d_v_read,
                l_nh, l_nkv, l_hd, l_hd, attn_pitch, S, l_kvg, l_window,
            )
        elif prefix_int8 and decode_order_multirow_attn:
            if verify_prof: route_prefix += 1
            if start_pos > 0:
                gpu_dequant_kv_i4_to_q8_layer(
                    self.ctx,
                    self.d_k_cache_i8[layer],
                    self.d_k_scales[layer],
                    self.d_k_deq,
                    d_k_scale_read,
                    l_nkv,
                    start_pos,
                    l_hd,
                    deq_pitch,
                    ccap,
                    0,
                    i4_scale_blocks,
                )
                gpu_dequant_kv_i4_to_q8_layer(
                    self.ctx,
                    self.d_v_cache_i8[layer],
                    self.d_v_scales[layer],
                    self.d_v_deq,
                    d_v_scale_read,
                    l_nkv,
                    start_pos,
                    l_hd,
                    deq_pitch,
                    ccap,
                    0,
                    i4_scale_blocks,
                )
            # ONE batched append, not S serial ones (Codex diagnosis, 2026-08-02). In THIS branch
            # no attention runs between appends — the single multirow attention call below fires
            # only after every row is published, and its per-row causal masking prevents any row
            # from reading a future row. The kernel is one warp per (seq, head) with the q8 row at
            # absolute base_pos+s, so each warp's quantization is byte-identical whether launched
            # alone or batched: n_s=S is EXACT by row independence.
            # The serial form cost S launches/layer instead of 1 — at VB=7 that is 6 extra x 60
            # layers = 360 extra launches per verify cycle (~2-3 ms), which measured as the
            # multirow route running SLOWER (38.57) than the inexact prefix route (40.88) despite
            # doing strictly less arithmetic. The per-row decode-order branch below keeps its
            # serial loop: there, attention genuinely runs after each append.
            gpu_append_quant_kv_i4_with_q8(
                self.ctx,
                d_k_b,
                d_v_b,
                self.d_k_cache_i8[layer],
                self.d_v_cache_i8[layer],
                self.d_k_scales[layer],
                self.d_v_scales[layer],
                self.d_k_deq,
                self.d_v_deq,
                d_k_scale_read,
                d_v_scale_read,
                S,
                l_nkv,
                l_hd,
                l_kvd,
                ccap,
                l_hd,
                start_pos,
                deq_pitch,
                i4_scale_blocks,
            )
            self.cache_lens_fp32[layer] = start_pos + S

            if layer == debug_omlp_layer and (
                debug_omlp_stage == 10 or debug_omlp_stage == 11 or debug_omlp_stage == 12
            ):
                if debug_omlp_row < 0 or debug_omlp_row >= S:
                    raise Error("debug_omlp_row out of range")
                var apos = start_pos + debug_omlp_row
                var lo = 0
                if l_window > 0 and apos - l_window + 1 > 0:
                    lo = apos - l_window + 1
                var klen = apos + 1 - lo
                var koff = UInt64(lo * l_hd)
                var scale_off = UInt64(lo * 4)
                if debug_omlp_stage == 10:
                    if debug_omlp_out != 0:
                        var kv_dbg_len = 2 * l_nkv * klen * l_hd + 2 * l_nkv * klen
                        var d_kv_dbg = cuda_malloc(kv_dbg_len * 4)
                        debug_kv_i8_view_to_f32_dev(
                            self.ctx,
                            d_k_read + koff,
                            d_v_read + koff,
                            d_k_scale_read + scale_off,
                            d_v_scale_read + scale_off,
                            d_kv_dbg,
                            l_nkv,
                            klen,
                            l_hd,
                            attn_pitch,
                        )
                        self.ctx.synchronize()
                        cuda_memcpy(UInt64(debug_omlp_out), d_kv_dbg, kv_dbg_len * 4, 2)
                        cuda_free(d_kv_dbg)
                elif debug_omlp_stage == 11:
                    if debug_omlp_out != 0:
                        var raw_len = 2 * l_nkv * klen * ((l_hd + 1) // 2) + 2 * l_nkv * klen
                        var d_raw_dbg = cuda_malloc(raw_len * 4)
                        debug_kv_i4_source_to_f32_dev(
                            self.ctx,
                            self.d_k_cache_i8[layer],
                            self.d_v_cache_i8[layer],
                            self.d_k_scales[layer],
                            self.d_v_scales[layer],
                            d_raw_dbg,
                            l_nkv,
                            klen,
                            l_hd,
                            ccap,
                            lo,
                        )
                        self.ctx.synchronize()
                        cuda_memcpy(UInt64(debug_omlp_out), d_raw_dbg, raw_len * 4, 2)
                        cuda_free(d_raw_dbg)
                else:
                    if debug_omlp_out != 0:
                        var score_len = l_nh * klen
                        var d_score_dbg = cuda_malloc(score_len * 4)
                        debug_qk_scores_int8_dev(
                            self.ctx,
                            d_q_b + UInt64(debug_omlp_row * l_qd * 4),
                            d_k_read + koff,
                            d_k_scale_read + scale_off,
                            d_score_dbg,
                            l_nh,
                            l_nkv,
                            l_hd,
                            l_hd,
                            attn_pitch,
                            klen,
                            l_kvg,
                        )
                        self.ctx.synchronize()
                        cuda_memcpy(UInt64(debug_omlp_out), d_score_dbg, score_len * 4, 2)
                        cuda_free(d_score_dbg)
                cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
                cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
                cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
                cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
                cuda_free(d_bf16_b)
                if need_block_scratch:
                    cuda_free(d_q_t); cuda_free(d_attn_t)
                if use_batched and not long_swa:
                    cuda_free(d_scores_big)
                if use_prefix_block:
                    cuda_free(d_scores_prefix)
                if batched_bf16 and not long_swa:
                    cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
                elif prefix_bf16:
                    cuda_free(d_q_t_bf16)
                if prefix_bf16:
                    cuda_free(d_scores_bf16_prefix)
                if long_swa:
                    cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                    cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
                return

            attention_decode_order_multirow_int8(
                self.ctx,
                d_q_b,
                d_scores_prefix,
                d_attn_b,
                d_k_read,
                d_v_read,
                d_k_scale_read,
                d_v_scale_read,
                l_nh,
                l_nkv,
                l_hd,
                l_hd,
                attn_pitch,
                S,
                prefix_key_cap,
                l_kvg,
                l_window,
                start_pos,
            )
        elif prefix_int8:
            if decode_order_verify_attn:
                if verify_prof: route_pertoken += 1
                if start_pos > 0:
                    gpu_dequant_kv_i4_to_q8_layer(
                        self.ctx,
                        self.d_k_cache_i8[layer],
                        self.d_k_scales[layer],
                        self.d_k_deq,
                        d_k_scale_read,
                        l_nkv,
                        start_pos,
                        l_hd,
                        deq_pitch,
                        ccap,
                        0,
                        i4_scale_blocks,
                    )
                    gpu_dequant_kv_i4_to_q8_layer(
                        self.ctx,
                        self.d_v_cache_i8[layer],
                        self.d_v_scales[layer],
                        self.d_v_deq,
                        d_v_scale_read,
                        l_nkv,
                        start_pos,
                        l_hd,
                        deq_pitch,
                        ccap,
                        0,
                        i4_scale_blocks,
                    )
                for i in range(S):
                    var apos = start_pos + i
                    gpu_append_quant_kv_i4_with_q8(
                        self.ctx,
                        d_k_b + UInt64(i * l_kvd * 4),
                        d_v_b + UInt64(i * l_kvd * 4),
                        self.d_k_cache_i8[layer],
                        self.d_v_cache_i8[layer],
                        self.d_k_scales[layer],
                        self.d_v_scales[layer],
                        self.d_k_deq,
                        self.d_v_deq,
                        d_k_scale_read,
                        d_v_scale_read,
                        1,
                        l_nkv,
                        l_hd,
                        l_kvd,
                        ccap,
                        l_hd,
                        apos,
                        deq_pitch,
                        i4_scale_blocks,
                    )
                    self.cache_lens_fp32[layer] = apos + 1
                    var lo = 0
                    if l_window > 0 and apos - l_window + 1 > 0:
                        lo = apos - l_window + 1
                    var klen = apos + 1 - lo
                    var koff = UInt64(lo * l_hd)
                    var scale_off = UInt64(lo * 4)
                    if layer == debug_omlp_layer and debug_omlp_stage == 10 and debug_omlp_row == i:
                        if debug_omlp_out != 0:
                            var kv_dbg_len = 2 * l_nkv * klen * l_hd + 2 * l_nkv * klen
                            var d_kv_dbg = cuda_malloc(kv_dbg_len * 4)
                            debug_kv_i8_view_to_f32_dev(
                                self.ctx,
                                d_k_read + koff,
                                d_v_read + koff,
                                d_k_scale_read + scale_off,
                                d_v_scale_read + scale_off,
                                d_kv_dbg,
                                l_nkv,
                                klen,
                                l_hd,
                                attn_pitch,
                            )
                            self.ctx.synchronize()
                            cuda_memcpy(UInt64(debug_omlp_out), d_kv_dbg, kv_dbg_len * 4, 2)
                            cuda_free(d_kv_dbg)
                        cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
                        cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
                        cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
                        cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
                        cuda_free(d_bf16_b)
                        if need_block_scratch:
                            cuda_free(d_q_t); cuda_free(d_attn_t)
                        if use_batched and not long_swa:
                            cuda_free(d_scores_big)
                        if use_prefix_block:
                            cuda_free(d_scores_prefix)
                        if batched_bf16 and not long_swa:
                            cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
                        elif prefix_bf16:
                            cuda_free(d_q_t_bf16)
                        if prefix_bf16:
                            cuda_free(d_scores_bf16_prefix)
                        if long_swa:
                            cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                            cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
                        return
                    if layer == debug_omlp_layer and debug_omlp_stage == 11 and debug_omlp_row == i:
                        if debug_omlp_out != 0:
                            var raw_len = 2 * l_nkv * klen * ((l_hd + 1) // 2) + 2 * l_nkv * klen
                            var d_raw_dbg = cuda_malloc(raw_len * 4)
                            debug_kv_i4_source_to_f32_dev(
                                self.ctx,
                                self.d_k_cache_i8[layer],
                                self.d_v_cache_i8[layer],
                                self.d_k_scales[layer],
                                self.d_v_scales[layer],
                                d_raw_dbg,
                                l_nkv,
                                klen,
                                l_hd,
                                ccap,
                                lo,
                            )
                            self.ctx.synchronize()
                            cuda_memcpy(UInt64(debug_omlp_out), d_raw_dbg, raw_len * 4, 2)
                            cuda_free(d_raw_dbg)
                        cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
                        cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
                        cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
                        cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
                        cuda_free(d_bf16_b)
                        if need_block_scratch:
                            cuda_free(d_q_t); cuda_free(d_attn_t)
                        if use_batched and not long_swa:
                            cuda_free(d_scores_big)
                        if use_prefix_block:
                            cuda_free(d_scores_prefix)
                        if batched_bf16 and not long_swa:
                            cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
                        elif prefix_bf16:
                            cuda_free(d_q_t_bf16)
                        if prefix_bf16:
                            cuda_free(d_scores_bf16_prefix)
                        if long_swa:
                            cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                            cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
                        return
                    if layer == debug_omlp_layer and debug_omlp_stage == 12 and debug_omlp_row == i:
                        if debug_omlp_out != 0:
                            var score_len = l_nh * klen
                            var d_score_dbg = cuda_malloc(score_len * 4)
                            debug_qk_scores_int8_dev(
                                self.ctx,
                                d_q_b + UInt64(i * l_qd * 4),
                                d_k_read + koff,
                                d_k_scale_read + scale_off,
                                d_score_dbg,
                                l_nh,
                                l_nkv,
                                l_hd,
                                l_hd,
                                attn_pitch,
                                klen,
                                l_kvg,
                            )
                            self.ctx.synchronize()
                            cuda_memcpy(UInt64(debug_omlp_out), d_score_dbg, score_len * 4, 2)
                            cuda_free(d_score_dbg)
                        cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
                        cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
                        cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
                        cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
                        cuda_free(d_bf16_b)
                        if need_block_scratch:
                            cuda_free(d_q_t); cuda_free(d_attn_t)
                        if use_batched and not long_swa:
                            cuda_free(d_scores_big)
                        if use_prefix_block:
                            cuda_free(d_scores_prefix)
                        if batched_bf16 and not long_swa:
                            cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
                        elif prefix_bf16:
                            cuda_free(d_q_t_bf16)
                        if prefix_bf16:
                            cuda_free(d_scores_bf16_prefix)
                        if long_swa:
                            cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                            cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
                        return
                    attention_gpu_int8_dev(
                        self.ctx,
                        d_q_b + UInt64(i * l_qd * 4),
                        d_k_read + koff,
                        d_v_read + koff,
                        d_k_scale_read + scale_off,
                        d_v_scale_read + scale_off,
                        self.d_scores_scratch,
                        d_attn_b + UInt64(i * l_qd * 4),
                        l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                    )
            else:
                if verify_prof: route_prefix += 1
                var key_base = 0
                if l_window > 0 and start_pos - l_window + 1 > 0:
                    key_base = start_pos - l_window + 1
                var key_len = start_pos + S - key_base
                var key_off = UInt64(key_base * l_hd)
                var scale_off = UInt64(key_base * 4)
                attention_prefill_prefix_batched_int8(
                    self.ctx,
                    d_q_b, d_q_t, d_scores_prefix, d_attn_t, d_attn_b,
                    d_k_read + key_off, d_v_read + key_off,
                    d_k_scale_read + scale_off, d_v_scale_read + scale_off,
                    l_nh, l_nkv, l_hd, l_hd, attn_pitch, S, key_len, l_kvg,
                    l_window, start_pos, key_base,
                )
        elif prefix_bf16:
            if verify_prof: route_prefix += 1
            var key_base = 0
            if l_window > 0 and start_pos - l_window + 1 > 0:
                key_base = start_pos - l_window + 1
            var key_len = start_pos + S - key_base
            var key_off = UInt64(key_base * l_hd * 2)
            if _strict_q4(): _flag_violation(VIOL_BF16_ATTN_BATCHED)
            attention_prefill_prefix_batched_bf16(
                self.ctx, self.handle,
                d_q_b, d_q_t, d_q_t_bf16, d_scores_prefix, d_scores_bf16_prefix,
                d_attn_t, d_attn_b,
                d_k_read + key_off, d_v_read + key_off,
                l_nh, l_nkv, l_hd, l_hd, attn_pitch, S, key_len, l_kvg,
                l_window, start_pos, key_base,
            )
        elif prefix_fp32:
            if verify_prof: route_prefix += 1
            var key_base = 0
            if l_window > 0 and start_pos - l_window + 1 > 0:
                key_base = start_pos - l_window + 1
            var key_len = start_pos + S - key_base
            var key_off = UInt64(key_base * l_hd * 4)
            if _strict_q4(): _flag_violation(VIOL_FP32_ATTN_BATCHED)
            attention_prefill_prefix_batched_fp32(
                self.ctx, self.handle,
                d_q_b, d_q_t, d_scores_prefix, d_attn_t, d_attn_b,
                d_k_read + key_off, d_v_read + key_off,
                l_nh, l_nkv, l_hd, l_hd, attn_pitch, S, key_len, l_kvg,
                l_window, start_pos, key_base,
            )
        else:
            if verify_prof: route_pertoken += 1
            var kvb = 1 if use_int8_attn else (2 if self.precision_bits <= 16 else 4)
            for i in range(S):
                # Sliding-window via cache offset: query at absolute pos attends keys [lo, pos].
                var apos = start_pos + i
                var lo = 0
                if l_window > 0 and apos - l_window + 1 > 0:
                    lo = apos - l_window + 1
                var klen = apos + 1 - lo
                # Long SWA prefills compute against the transient linear cache. Pack
                # exactly this query's logical window into the compact dequant scratch,
                # matching decode's ring read order byte-for-byte.
                if ring_prefill:
                    if use_int8_attn:
                        gpu_dequant_kv_i4_to_q8_layer(
                            self.ctx, d_k_compute_cache, d_ks_compute,
                            self.d_k_deq, d_k_scale_read, l_nkv, klen, l_hd,
                            deq_pitch, compute_cap, lo, i4_scale_blocks,
                        )
                        gpu_dequant_kv_i4_to_q8_layer(
                            self.ctx, d_v_compute_cache, d_vs_compute,
                            self.d_v_deq, d_v_scale_read, l_nkv, klen, l_hd,
                            deq_pitch, compute_cap, lo, i4_scale_blocks,
                        )
                    elif self.kv_int4:
                        gpu_dequant_kv_i4_layer(
                            self.ctx, d_k_compute_cache, d_ks_compute,
                            self.d_k_deq, l_nkv, klen, l_hd,
                            deq_pitch, compute_cap, lo, i4_scale_blocks,
                        )
                        gpu_dequant_kv_i4_layer(
                            self.ctx, d_v_compute_cache, d_vs_compute,
                            self.d_v_deq, l_nkv, klen, l_hd,
                            deq_pitch, compute_cap, lo, i4_scale_blocks,
                        )
                    else:
                        gpu_dequant_kv_i8_layer(
                            self.ctx, d_k_compute_cache, d_ks_compute,
                            self.d_k_deq, l_nkv, klen, l_hd,
                            deq_pitch, compute_cap, lo,
                        )
                        gpu_dequant_kv_i8_layer(
                            self.ctx, d_v_compute_cache, d_vs_compute,
                            self.d_v_deq, l_nkv, klen, l_hd,
                            deq_pitch, compute_cap, lo,
                        )
                var koff = UInt64(0) if ring_prefill else UInt64(lo * l_hd * kvb)
                if use_int8_attn:
                    var scale_off = UInt64(0) if ring_prefill else UInt64(lo * 4)
                    attention_gpu_int8_dev(
                        self.ctx,
                        d_q_b + UInt64(i * l_qd * 4),
                        d_k_read + koff, d_v_read + koff,
                        d_k_scale_read + scale_off, d_v_scale_read + scale_off,
                        self.d_scores_scratch, d_attn_b + UInt64(i * l_qd * 4),
                        l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                    )
                elif self.precision_bits <= 16:
                    if _strict_q4() and i == 0: _flag_violation(VIOL_BF16_ATTN_DECODE)
                    attention_gpu_bf16_dev(
                        self.ctx, self.handle,
                        d_q_b + UInt64(i * l_qd * 4),
                        d_k_read + koff, d_v_read + koff,
                        self.d_scores_scratch, self.d_q_bf16, self.d_scores_bf16,
                        d_attn_b + UInt64(i * l_qd * 4),
                        l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                    )
                else:
                    if _strict_q4() and i == 0: _flag_violation(VIOL_FP32_ATTN_DECODE)
                    attention_gpu_fp32_dev(
                        self.ctx, self.handle,
                        d_q_b + UInt64(i * l_qd * 4),
                        d_k_read + koff, d_v_read + koff,
                        self.d_scores_scratch, d_attn_b + UInt64(i * l_qd * 4),
                        l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                    )

        if prof:
            cuda_sync()
            t_attn += Int(perf_counter_ns()) - _t0
            _t0 = Int(perf_counter_ns())
        apply_output_and_mlp_batched(
            self.ctx, d_x_b, d_attn_b, d_normed_b,
            d_up_b if attn_gate_ready else d_q_b, d_o_b, d_pn_b,
            d_gate_b, d_up_b, d_mlpv_b, d_down_b, self.handle,
            self.d_attn_gw[layer], self.d_ow[layer],
            self.d_post_attn_norms[layer], self.d_pre_ff_norms[layer], self.d_post_ff_norms[layer],
            self.d_gw[layer], self.d_uw[layer], self.d_dw[layer], d_bf16_b,
            self.layer_scalars[layer], D, FF, l_qd, S, layer,
            self.d_weight_bf16_scratch,
            self.d_ow_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_attn_gw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_gw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_uw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_dw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_ow_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_attn_gw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_gw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_uw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            self.d_dw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
            W4A4Scratch(self.d_w4a4_packed, self.d_w4a4_bs, self.d_w4a4_global, self.d_w4a4_cpad, self.w4a4_chunk_mpad, self.w4a4, self.d_w4a4_bs_sf, self.d_w4a4_wbs_sf),
            debug_omlp_row if layer == debug_omlp_layer else -1,
            debug_omlp_stage if layer == debug_omlp_layer else -1,
            debug_omlp_out if layer == debug_omlp_layer else Int64(0),
            prof_stages=(verify_prof and (layer == 20 or layer == 40)),
            attn_gate_ready=attn_gate_ready,
        )
        if layer == debug_omlp_layer:
            cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
            cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
            cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
            cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
            cuda_free(d_bf16_b)
            if need_block_scratch:
                cuda_free(d_q_t); cuda_free(d_attn_t)
            if use_batched and not long_swa:
                cuda_free(d_scores_big)
            if use_prefix_block:
                cuda_free(d_scores_prefix)
            if batched_bf16 and not long_swa:
                cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
            elif prefix_bf16:
                cuda_free(d_q_t_bf16)
            if prefix_bf16:
                cuda_free(d_scores_bf16_prefix)
            if long_swa:
                cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
                cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)
            return
        if prof:
            cuda_sync()
            t_mlp += Int(perf_counter_ns()) - _t0

        if self.drafter_tap_count > 0:
            var tap_slot = self.drafter_tap_slot(layer)
            if tap_slot >= 0:
                var rows_to_copy = eagle_tap_rows
                if rows_to_copy < 0:
                    rows_to_copy = 0
                if rows_to_copy > S:
                    rows_to_copy = S
                var copy_live_tap = self.d_eagle3_tap_out != 0 and eagle_tap_row >= 0 and eagle_tap_row < S
                var copy_row_taps = self.d_eagle3_tap_rows_out != 0 and rows_to_copy > 0
                if copy_live_tap or copy_row_taps or dflash_capture:
                    self.ctx.synchronize()
                    if copy_live_tap:
                        cuda_memcpy(
                            self.d_eagle3_tap_out + UInt64(tap_slot * D * 4),
                            d_x_b + UInt64(eagle_tap_row * D * 4),
                            D * 4,
                            3,
                        )
                    if copy_row_taps:
                        for r in range(rows_to_copy):
                            cuda_memcpy(
                                self.d_eagle3_tap_rows_out + UInt64((r * self.drafter_tap_count + tap_slot) * D * 4),
                                d_x_b + UInt64(r * D * 4),
                                D * 4,
                                3,
                            )
                    if dflash_capture:
                        for r in range(S):
                            cuda_memcpy(
                                d_dflash_prefill_taps + UInt64((r * self.drafter_tap_count + tap_slot) * D * 4),
                                d_x_b + UInt64(r * D * 4),
                                D * 4,
                                3,
                            )
                    cuda_sync()

    if dflash_capture:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(self.dflash_ptr)
        )
        if dp[0].cache_len() + S > dp[0].max_ctx:
            raise Error("DFlash prefill context exceeds cache capacity")
        var off = 0
        while off < S:
            var chunk = S - off
            if chunk > DFLASH_CTX_BATCH:
                chunk = DFLASH_CTX_BATCH
            dflash_append_context_rows(
                self.ctx,
                dp[0],
                d_dflash_prefill_taps + UInt64(off * self.drafter_tap_count * D * 4),
                chunk,
            )
            off += chunk
        self.ctx.synchronize()

    # EAGLE-3 D3 (discrete NVFP4/W4A4) verify readout: when verify_g_out != 0 on the w4a4
    # path, emit per-position logits for ALL S rows via the DECODE-IDENTICAL lm-head:
    # per row, D2H the post-layer hidden -> HOST rmsnorm(final_norm) -> H2D d_lmhead_in
    # -> the SAME direct NVFP4-weight/fp32-activation GEMV engine_decode uses
    # -> D2H VOCAB row. SERIAL by design:
    # bit-exactness vs decode is the lossless-spec gate (landmine #5); batching the
    # lm-head (M=S) would change the GEMM route vs decode for ~3ms/cycle saved.
    # Raw pre-softcap logits (softcap is monotone -> host argmax unchanged).
    if verify_g_out != 0 and self.w4a4:
        var xs = List[Float32](capacity=D)
        for _ in range(D): xs.append(0.0)
        var normed = List[Float32](capacity=D)
        for _ in range(D): normed.append(0.0)
        # BATCHED READOUT. Each single-row GEMV re-streams the whole 749 MB embedding matrix to
        # produce one row of logits, so S rows cost S x that traffic. Measured on the RTX PRO 4000:
        # read_lmhead 23.22 ms/call = 31.6% of verify. The multirow kernel reads each weight
        # once per <=SMAX row chunk and reuses it across that chunk, and is bit-exact to v3 by
        # construction (row-independent weight decode hoisted; identical per-(s,n) expression,
        # accumulation order and warp reduction). The in-tree comment that priced batching at
        # "~3ms/cycle" costed launch overhead and missed the S-fold weight re-read.
        # S > SMAX is split into ceil(S/SMAX) row slices. Each slice re-streams the
        # weights once, instead of the serial fallback re-streaming them once per row.
        # NOMOS_VERIFY_BATCH_READOUT=0 retains that fallback as the A/B control.
        if _env_float("NOMOS_VERIFY_BATCH_READOUT", 1.0) > 0.5:
            var _rr0 = Int(perf_counter_ns()) if prof else 0
            var normed_rows = List[Float32](capacity=S * D)
            for _ in range(S * D): normed_rows.append(0.0)
            for s in range(S):
                cuda_download(xs, d_x_b + UInt64(s * D * 4), D)
                rmsnorm(normed, xs, self.final_norm, D, RMS_EPS_FINAL)
                for j in range(D):
                    normed_rows[s * D + j] = normed[j]
            cuda_upload(d_normed_b, normed_rows)
            cuda_sync()   # same raw H2D -> ctx GEMV seam as decode
            if prof:
                t_read_prep += Int(perf_counter_ns()) - _rr0
                _rr0 = Int(perf_counter_ns())
            var row0 = 0
            while row0 < S:
                var chunk_s = S - row0
                if chunk_s > NVFP4_GEMV_SMAX:
                    chunk_s = NVFP4_GEMV_SMAX
                gpu_matmul_nvfp4_fused_v3_multirow_dev(
                    self.ctx,
                    self.d_lmhead_logits + UInt64(row0 * VOCAB * 4),
                    d_normed_b + UInt64(row0 * D * 4),
                    self.d_embed_nvfp4,
                    self.embed_global,
                    D,
                    VOCAB,
                    chunk_s,
                )
                _w4a4_sync_checkpoint(
                    self.ctx,
                    "verify lmhead multirow NVFP4 GEMV chunk",
                    chunk_s,
                    VOCAB,
                    D,
                )
                row0 += chunk_s
            self.ctx.synchronize()
            if prof:
                t_read_lmhead += Int(perf_counter_ns()) - _rr0
                _rr0 = Int(perf_counter_ns())
            cuda_memcpy(UInt64(verify_g_out), self.d_lmhead_logits, S * VOCAB * 4, 2)
            if prof:
                t_read_copy += Int(perf_counter_ns()) - _rr0
        else:
            # Keep verify readout decode-identical on sm_100 and sm_120/121. Besides
            # preserving L1 losslessness, this avoids the same vocab-sized W4A4 fault.
            for s in range(S):
                # This branch had NO timers, which is why the phase profiler reported ~44.6 ms/call
                # (48%) unaccounted while every named phase looked reasonable. Mirror the timers the
                # act_precision()==8 branch already has, so the NVFP4 readout is priced too.
                var _rr0 = Int(perf_counter_ns()) if prof else 0
                cuda_download(xs, d_x_b + UInt64(s * D * 4), D)
                rmsnorm(normed, xs, self.final_norm, D, RMS_EPS_FINAL)
                cuda_upload(self.d_lmhead_in, normed)
                cuda_sync()   # raw H2D -> ctx GEMV boundary (same seam as decode)
                if prof:
                    t_read_prep += Int(perf_counter_ns()) - _rr0
                    _rr0 = Int(perf_counter_ns())
                # Must track engine_decode's lm-head route -- that IS the "decode-identical"
                # invariant above. Decode defaults to v3 (e837271); v3 is bit-identical to v1 by
                # construction (same per-lane block set, same sequential t=0..15 accumulation, only
                # the branchy E2M1 decode becomes branchless SIMD) and measured 12/12 byte-exact.
                # Leaving verify on v1 while decode ran v3 VIOLATED the invariant rather than
                # preserving it, and it cost S x 3.54 ms/cycle: the verify phase profiler showed
                # ~44.6 ms/call (48%) unaccounted, and S serial v1 lm-heads at 9.285 ms is 46.4 of it.
                if _env_float("NOMOS_LMHEAD_V3", 1.0) > 0.5:
                    gpu_matmul_nvfp4_fused_v3_dev(self.ctx, self.d_lmhead_logits, self.d_lmhead_in,
                                                  self.d_embed_nvfp4, self.embed_global, D, VOCAB)
                else:
                    gpu_matmul_nvfp4_fused_dev(self.ctx, self.d_lmhead_logits, self.d_lmhead_in,
                                               self.d_embed_nvfp4, self.embed_global, D, VOCAB)
                _w4a4_sync_checkpoint(
                    self.ctx, "verify lmhead direct NVFP4 GEMV", 1, VOCAB, D
                )
                self.ctx.synchronize()
                if prof:
                    t_read_lmhead += Int(perf_counter_ns()) - _rr0
                    _rr0 = Int(perf_counter_ns())
                cuda_memcpy(
                    UInt64(verify_g_out) + UInt64(s * VOCAB * 4),
                    self.d_lmhead_logits,
                    VOCAB * 4,
                    2,
                )
                if prof:
                    t_read_copy += Int(perf_counter_ns()) - _rr0
    # MTP spec-decode verify readout: when verify_g_out != 0, emit per-position logits for ALL
    # S batched positions (the prefill fast-path skips the LM-head on non-final positions; verify
    # needs every position's argmax). Mirrors the decode dp4a Q4 lm-head per position: host final
    # norm -> q4-dp4a lm-head -> copy VOCAB logits (D2H) into verify_g_out[s*VOCAB]. Q4·Q8 int8
    # path (verify == decode precision). d_x_b holds the post-layer hidden [S, D].
    elif verify_g_out != 0 and act_precision() == 8 and self.d_embed_q4 != 0:
        var _tr0 = Int(perf_counter_ns())
        var xs = List[Float32](capacity=D)
        for _ in range(D): xs.append(0.0)
        var normed = List[Float32](capacity=D)
        for _ in range(D): normed.append(0.0)
        if _env_float("NOMOS_VERIFY_SERIAL_READOUT", 0.0) > Float32(0.5):
            for s in range(S):
                var _rr0 = Int(perf_counter_ns()) if prof else 0
                cuda_download(xs, d_x_b + UInt64(s * D * 4), D)
                rmsnorm(normed, xs, self.final_norm, D, RMS_EPS_FINAL)
                cuda_upload(self.d_lmhead_in, normed)
                cuda_sync()   # raw H2D -> ctx GEMV boundary
                if prof:
                    t_read_prep += Int(perf_counter_ns()) - _rr0
                    _rr0 = Int(perf_counter_ns())
                gpu_matmul_q4_dp4a_dev[4](
                    self.ctx, self.d_lmhead_logits, self.d_lmhead_in,
                    self.d_embed_q4, self.d_weight_bf16_scratch, D, VOCAB,
                )
                self.ctx.synchronize()
                if prof:
                    t_read_lmhead += Int(perf_counter_ns()) - _rr0
                    _rr0 = Int(perf_counter_ns())
                cuda_memcpy(
                    UInt64(verify_g_out) + UInt64(s * VOCAB * 4),
                    self.d_lmhead_logits,
                    VOCAB * 4,
                    2,
                )
                if prof:
                    cuda_sync()
                    t_read_copy += Int(perf_counter_ns()) - _rr0
        else:
            var all_x = List[Float32](capacity=S * D)
            for _ in range(S * D): all_x.append(0.0)
            var normed_rows = List[Float32](capacity=S * D)
            for _ in range(S * D): normed_rows.append(0.0)
            var _rr0 = Int(perf_counter_ns()) if prof else 0
            cuda_download(all_x, d_x_b, S * D)
            for s in range(S):
                for j in range(D):
                    xs[j] = all_x[s * D + j]
                rmsnorm(normed, xs, self.final_norm, D, RMS_EPS_FINAL)
                for j in range(D):
                    normed_rows[s * D + j] = normed[j]
            cuda_upload(d_normed_b, normed_rows)
            cuda_sync()   # raw H2D -> ctx S8 lm-head boundary
            if prof:
                t_read_prep += Int(perf_counter_ns()) - _rr0
                _rr0 = Int(perf_counter_ns())
            gpu_matmul_q4_s8_v4_gemv_dev[8](
                self.ctx, self.d_lmhead_logits, d_normed_b, self.d_embed_q4,
                self.d_weight_bf16_scratch, D, VOCAB, S,
            )
            self.ctx.synchronize()   # sync ctx queue before the raw cuda_memcpy host read (lm-head GEMM race)
            if prof:
                t_read_lmhead += Int(perf_counter_ns()) - _rr0
                _rr0 = Int(perf_counter_ns())
            cuda_memcpy(UInt64(verify_g_out), self.d_lmhead_logits, S * VOCAB * 4, 2)
            if prof:
                cuda_sync()
                t_read_copy += Int(perf_counter_ns()) - _rr0
        if prof:
            cuda_sync()
            print("[prof] verify readout (ms):", (Int(perf_counter_ns()) - _tr0) // 1000000, " S=", S)

    if prof:
        print("[prof] prefill phases (ms): qkv=", t_qkv // 1000000,
              " append=", t_app // 1000000, " deq=", t_deq // 1000000,
              " attn=", t_attn // 1000000,
              " mlp=", t_mlp // 1000000, " S=", S)
    if verify_prof:
        cuda_sync()
        print("[verify-prof] total_ms=", (Int(perf_counter_ns()) - t_verify_total0) // 1000000,
              " alloc=", t_alloc // 1000000, " embed=", t_embed // 1000000,
              " qkv=", t_qkv // 1000000, " append=", t_app // 1000000,
              " deq=", t_deq // 1000000, " attn=", t_attn // 1000000,
              " mlp=", t_mlp // 1000000, " read_prep=", t_read_prep // 1000000,
              " read_lmhead=", t_read_lmhead // 1000000,
              " read_copy=", t_read_copy // 1000000, " S=", S,
              " start=", start_pos, " routes square/prefix/row=",
              route_square, "/", route_prefix, "/", route_pertoken)

    cuda_free(d_x_b); cuda_free(d_normed_b); cuda_free(d_q_b)
    cuda_free(d_k_b); cuda_free(d_v_b); cuda_free(d_attn_b)
    cuda_free(d_o_b); cuda_free(d_pn_b); cuda_free(d_gate_b)
    cuda_free(d_up_b); cuda_free(d_mlpv_b); cuda_free(d_down_b)
    cuda_free(d_bf16_b)
    if need_block_scratch:
        cuda_free(d_q_t); cuda_free(d_attn_t)
    if use_batched and not long_swa:
        cuda_free(d_scores_big)
    if use_prefix_block:
        cuda_free(d_scores_prefix)
    if batched_bf16 and not long_swa:
        cuda_free(d_q_t_bf16); cuda_free(d_scores_bf16_big)
    elif prefix_bf16:
        cuda_free(d_q_t_bf16)
    if prefix_bf16:
        cuda_free(d_scores_bf16_prefix)
    if d_dflash_prefill_taps != 0:
        cuda_free(d_dflash_prefill_taps)
    if long_swa:
        cuda_free(d_swa_k_tmp); cuda_free(d_swa_v_tmp)
        cuda_free(d_swa_ks_tmp); cuda_free(d_swa_vs_tmp)

# ─────────────────────────────────────────────────────────────────
