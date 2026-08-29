"""eagle3_drafter.mojo — the EAGLE-3 drafter for the Gemma-4-31B MTP spec-decode path.

EAGLE-3 forward (port of `RedHatAI/gemma-4-31B-it-speculator.eagle3`, parametric,
swarm-safe — TRAINING stays internal, only the forward body + FC + d2t are public).

The drafter is a SINGLE 1-layer Llama-style decoder that drafts the next K
tokens off the 31B backbone. The architecture is a verbatim port of
`max/pipelines/architectures/eagle_common/eagle_mha_draft.py` (Eagle3MHADraft)
adapted for Gemma 31B (DeepseekYarn -> standard rope θ1e4, hd256, GQA 32Q/16KV,
3-way taps at [L2, L30, L57], d2t LUT for the 32K→262K vocab extension).

Forward (matches `eagle_mha_draft.py` `__call__` exactly, line-by-line):

  Per step k:
    1) fused = fc @ cat[tap_L2, tap_L30, tap_L57]   [dB, 3*dB] @ [3*dB] -> [dB]
                                                  (step 1 only; steps 2..K skip fc
                                                   and use the drafter's prior hs
                                                   as the fused input — this is
                                                   the EAGLE-3 autoregressive
                                                   recurrence, fixing #4)
    2) h_embed = embed_table[token]                [DRAFT_VOCAB=32K, dB=5376]
    3) norm_embed = input_layernorm(h_embed)
    4) norm_fused = hidden_norm(fused)              in-place on s_fused
    5) concat_inputs = cat[norm_embed, norm_fused] [2*dB = 10752]   (gotcha F2: wide)
    6) qkv = qkv_proj @ concat_inputs              [Q+K+V, 2*dB=10752]
    7) q = rope(q, pos, θ=1e4, hd=256)              Gemma standard rope (NOT DeepseekYarn; NO q/k-norm — EAGLE-3 is llama-typed)
    8) attn_out = causal_GQA(q, k, v)               32Q / 16KV / hd=256; 1/sqrt(256)=0.0625 scale lives in the eagle3_attn
    9) attn_out = o_proj(attn_out)                 [Q, 2*dB] -> [dB=5376] (REPLACED narrow)
   10) hs = norm_fused + attn_out                   RESIDUAL BASE = hidden_NORM'D FUSED (NOT pre-norm;
                                                   norm_before_residual=TRUE, byte-verified vs
                                                   the Eagle3DraftModel checkpoint — MAX's
                                                   Eagle3MHADraft diverges here, checkpoint wins)
   11) norm_outs = post_attention_layernorm(hs)    post-attn rmsnorm
   12) mlp_outs = silu(gate(hs)) * up(hs); mlp = down(mlp_outs)   (silu, NOT gelu; fixes #2)
   13) hs = hs + mlp_outs                          RESIDUAL on hs
   14) last_token = hs[last_idx, :]                 (M=1, last idx = 0)
   15) norm_last_token = norm(last_token)          "norm" = the final post-FFW norm
   16) logits = lm_head(norm_last_token)            [dB=5376 -> 32K draft vocab]
   17) target_token = d2t_lut[argmax(logits)]       [32K] int64 -> int  (gotcha F3; fixes #3)
                                                    bare path: 1 indexed read; not a dense matmul

The 4 norms placed (per `eagle_mha_draft.py`):
  - input_layernorm   -> embed (pre-cat, on the embed half)
  - hidden_norm       -> fused (pre-cat, on the fused half)
  - post_attn_layernorm -> post-attn (before mlp)
  - norm              -> final (before lm_head)

d2t (dequant-to-token) LUT: simple [DRAFT_VOCAB=32K] int64 map
(d2t[draft_id] = target_id, 1 indexed read). NOT a dense 32Kx262K matrix.

Weights are Q8 (the law-allowed drafter concession), loaded via the .q8 int8
GEMV path (q8_weights.mojo). M=1 per step (K drafts are sequential) so every
matmul is a GEMV.

K-step recurrence: step 1's fused = fc(target_taps); steps 2..K's fused = the
prior step's `hs` (skip fc — hs is already dB-wide). This is the EAGLE-3
autoregressive h_next recurrence (fixes #4 — re-using seed taps drafts every
token from STALE seed context, which tanks acceptance depth).
"""
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import UnsafePointer
from std.ffi import external_call, c_int, c_size_t
from lib.q8_weights import load_to_gpu_q8, gpu_matmul_q8_fused_dev
from lib.io import load_to_gpu, read_f32, load_to_gpu_bf16
from lib.fp4_weights import load_to_gpu_nvfp4
from lib.fp4_act import gpu_quant_act_nvfp4_dec, gpu_matmul_nvfp4_w4a4_prequant_dev
from lib.cuda import cuda_malloc, cuda_memcpy, cuda_free, cuda_sync, cuda_upload
from lib.ops_gpu_mojo import (
    gpu_residual_add_mojo,
    gpu_elementwise_mul_mojo,
    gpu_silu_mojo,
    gpu_scalar_mul_mojo,
    gpu_rope_mojo,
    gpu_matmul_bf16w_gemv_dev,
)
from lib.ops_gpu_mojo_reductions import (
    gpu_rmsnorm_mojo,
    gpu_rmsnorm_inplace_mojo,
)
from lib.decode_argmax import device_decode_token
from lib.eagle3_attn import eagle3_own_kv_attend
from lib.eagle3_d2t import eagle3_argmax_d2t


comptime D2D = 3  # cudaMemcpyDeviceToDevice
comptime DRAFT_VOCAB = 32000   # real draft_vocab_size (was 32768 — padded tail → OOB d2t)
comptime TARGET_VOCAB = 262144
# Weight modes (D2, discrete NVFP4 port). NVFP4 = drafter-study/eagle3-flat blobs on the
# R2 W4A4 path (prequant dedup + fused postscale). GOLD = the same weights as EXACT
# checkpoint bf16 (fp32 blob -> bf16 device, lossless) + fp32-activation GEMV — the
# quant-BYPASS mode for the G2 wiring-parity gate vs the HF Eagle3DraftModel. Q8 is
# the legacy GB10 path (eagle3_* .q8 stems), kept for cross-box compatibility.
comptime EAGLE_MODE_Q8 = 0
comptime EAGLE_MODE_NVFP4 = 1
comptime EAGLE_MODE_GOLD = 2
comptime EAGLE_AQ_KMAX = 21504   # act-quant scratch K capacity = the widest GEMM INPUT.
                                 # That is the DOWN-proj (K = ff = 21504), NOT fc (3*dB =
                                 # 16128) — sizing to fc overflowed the 16-row quant scratch
                                 # by 43KB into adjacent drafter buffers every step (found
                                 # via per-op gold-diff: o_out cos 0.979 -> hidden_out 0.62).
comptime EAGLE_NMAX = 32000      # widest GEMM output (lm_head) for the c_pad fallback
comptime MAX_DRAFT_SEQ = 1024  # persistent drafter-KV depth: the draft attends the committed prefix
                               # (positions 0..seq_len-1) + its own draft tokens — the real EAGLE-3
                               # acceptance mechanism (bare-path K-local context = slot 0 craters at step 2)
comptime MAX_VERIFY_ROWS = 17   # fused verify rows: 16 draft candidates + committed/bonus row


# EAGLE-3 drafter struct: holds all weights + scratch buffers.
struct Eagle3Drafter(Movable):
    # dims (Gemma 31B)
    var dB: Int  # backbone dim (5376)
    var ff: Int  # ffn dim (14336) - matches target ratio ~2.55x
    var nh: Int  # 32 (Q heads)
    var nkv: Int  # 16 (KV heads)
    var hd: Int  # 256
    # 3-way aux tap config
    var n_taps: Int
    var tap_layers: List[Int]
    # weight mode (EAGLE_MODE_*): Q8 legacy / NVFP4 (R2 W4A4) / GOLD (bf16 wiring gate)
    var mode: Int
    # NVFP4 per-tensor global scales (parallel to the 9 GEMM weights; unused otherwise)
    var gs_fc: Float32
    var gs_q: Float32
    var gs_k: Float32
    var gs_v: Float32
    var gs_o: Float32
    var gs_gate: Float32
    var gs_up: Float32
    var gs_down: Float32
    var gs_lm_head: Float32
    # NVFP4 act-quant scratch (decode M=1/Mpad=16; K cap = 3*dB, N cap = 32000).
    # Drafter-owned so head GEMMs never share the engine's per-layer W4A4 scratch.
    var s_aq_packed: UInt64   # [16, EAGLE_AQ_KMAX/2] u8
    var s_aq_bs: UInt64       # [16, EAGLE_AQ_KMAX/16] u8
    var s_aq_global: UInt64   # [16] fp32
    var s_aq_cpad: UInt64     # [16, EAGLE_NMAX] fp32 (raw-MMA fallback when qa-fuse is off)
    # weights (device; format per mode)
    var w_fc: UInt64  # [dB, 3*dB]    fuse 3 aux taps into 1 hidden
    # NOTE: no w_embed — the draft embedding is bit-identical to the target's
    # (proven 2026-07-07, all 1.409B elements) and is read per step from the
    # engine's mmap'd fp32 table via embed_fd (raw row, NO sqrt(D) target scale).
    var w_in_norm_embed: UInt64  # [dB] input_layernorm (on embed half)
    var w_hidden_norm: UInt64  # [dB] hidden_norm (on fused half)
    var w_q: UInt64  # [nh*hd, 2*dB=10752]  q_proj (wide in)
    var w_k: UInt64  # [nkv*hd, 2*dB]       k_proj
    var w_v: UInt64  # [nkv*hd, 2*dB]       v_proj
    var w_o: UInt64  # [dB=5376, nh*hd]     o_proj (REPLACED NARROW OUT)
    var w_post_attn_norm: UInt64  # [dB] post_attention_layernorm
    var w_ffn_gate: UInt64  # [ff, dB]
    var w_ffn_up: UInt64  # [ff, dB]
    var w_ffn_down: UInt64  # [dB, ff]
    var w_norm: UInt64  # [dB] final post-FFW norm
    var w_lm_head: UInt64  # [DRAFT_VOCAB=32K, dB] (TIED to embed)
    # d2t LUT: simple [DRAFT_VOCAB=32K] int64 map (NOT dense; fixes #3)
    var d_d2t_lut: UInt64  # [DRAFT_VOCAB=32K] int64 (d2t[draft_id] = target_id)
    # scratch (device)
    var s_fused: UInt64  # [dB] hidden_norm'd fused (residual base; norm_before_residual=TRUE)
    var s_h_embed: UInt64  # [dB] input embed
    var s_concat_in: UInt64  # [2*dB=10752] wide concat input
    var s_q: UInt64  # [nh*hd]
    var s_k: UInt64  # [nkv*hd]
    var s_v: UInt64  # [nkv*hd]
    var s_attn: UInt64  # [dB=5376] (narrow o_proj output)
    var s_norm: UInt64  # [dB] (scratch for rmsnorm output)
    var s_gate: UInt64  # [ff]
    var s_up: UInt64  # [ff]
    var s_mlp: UInt64  # [ff]
    var s_hs: UInt64  # [dB] the residual state (= hs after each block)
    var s_logits: UInt64  # [DRAFT_VOCAB=32K] (post lm_head, pre d2t)
    var s_rope_cache: UInt64  # [hd/2] (rope freqs)
    var s_scores: UInt64  # [nh*MAX_DRAFT_SEQ] attend scores scratch
    var d_taps_buf: UInt64  # [3*dB] receives the engine L[1,29,56] tap (live spec path; armed at load)
    var d_verify_taps_buf: UInt64  # [MAX_VERIFY_ROWS, 3, dB] batched-verify tap rows for host accept select
    var last_verify_rows: Int  # valid rows currently stored in d_verify_taps_buf
    var seq_len: Int  # persistent drafter-KV length (committed prefix); reset via nomos_eagle3_reset
    var last_draft_base_len: Int  # committed prefix length before the latest speculative draft
    var next_seed_token: Int  # host-visible breadcrumb: token expected to seed the next draft
    # drafter's own paged KV cache (8 slots; grows by 1 per K-step).
    var s_drafter_k: UInt64  # [8, nkv*hd]
    var s_drafter_v: UInt64  # [8, nkv*hd]
    # gold-diff capture buffers (opt-in; default ON for the oracle).
    var cap_fc_out: UInt64  # [8, dB]      fused input per step (step 0 = post-fc; later = prior hs)
    var cap_attn_out: UInt64  # [8, nh*hd]   pre-o_proj
    var cap_o_out: UInt64  # [8, dB]        post-o_proj
    var cap_hidden_out: UInt64  # [8, dB]     pre-final-norm (= recurrent state)
    var cap_logits_out: UInt64  # [8, DRAFT_VOCAB] pre-d2t draft logits
    var cap_draft_ids: List[Int32]  # [8] host
    var cap_target_ids: List[Int64]  # [8] host
    var capture: Bool  # opt-in capture flag

    def __init__(out self, dir: String, mode: Int = EAGLE_MODE_NVFP4):
        self.dB = 5376
        self.ff = 21504   # intermediate_size (gate/up [21504,5376]) — was 14336; weight shape is ground truth
        self.nh = 32
        self.nkv = 16
        self.hd = 256
        self.n_taps = 3
        self.tap_layers = [2, 30, 57]   # config aux ids (the ENGINE arms decoder layers [1,29,56] — bc2daf6)
        self.mode = mode
        self.gs_fc = Float32(1.0)
        self.gs_q = Float32(1.0)
        self.gs_k = Float32(1.0)
        self.gs_v = Float32(1.0)
        self.gs_o = Float32(1.0)
        self.gs_gate = Float32(1.0)
        self.gs_up = Float32(1.0)
        self.gs_down = Float32(1.0)
        self.gs_lm_head = Float32(1.0)
        if mode == EAGLE_MODE_NVFP4:
            # drafter-study/eagle3-flat blobs (convert_eagle3.py stems). ~0.33 GB device.
            var _gs = List[Float32]()
            var _ags = List[Float32]()
            self.w_fc = load_to_gpu_nvfp4(dir + "eagle_fc_weight.nvfp4", _gs, _ags)
            self.w_q = load_to_gpu_nvfp4(dir + "eagle_layer_self_attn_q_proj_weight.nvfp4", _gs, _ags)
            self.w_k = load_to_gpu_nvfp4(dir + "eagle_layer_self_attn_k_proj_weight.nvfp4", _gs, _ags)
            self.w_v = load_to_gpu_nvfp4(dir + "eagle_layer_self_attn_v_proj_weight.nvfp4", _gs, _ags)
            self.w_o = load_to_gpu_nvfp4(dir + "eagle_layer_self_attn_o_proj_weight.nvfp4", _gs, _ags)
            self.w_ffn_gate = load_to_gpu_nvfp4(dir + "eagle_layer_mlp_gate_proj_weight.nvfp4", _gs, _ags)
            self.w_ffn_up = load_to_gpu_nvfp4(dir + "eagle_layer_mlp_up_proj_weight.nvfp4", _gs, _ags)
            self.w_ffn_down = load_to_gpu_nvfp4(dir + "eagle_layer_mlp_down_proj_weight.nvfp4", _gs, _ags)
            self.w_lm_head = load_to_gpu_nvfp4(dir + "eagle_lm_head_weight.nvfp4", _gs, _ags)
            self.gs_fc = _gs[0]
            self.gs_q = _gs[1]
            self.gs_k = _gs[2]
            self.gs_v = _gs[3]
            self.gs_o = _gs[4]
            self.gs_gate = _gs[5]
            self.gs_up = _gs[6]
            self.gs_down = _gs[7]
            self.gs_lm_head = _gs[8]
        elif mode == EAGLE_MODE_GOLD:
            # Quant-bypass wiring mode: fp32 blobs (widened bf16) -> bf16 device =
            # EXACT checkpoint weights. fp32 activation GEMV downstream.
            self.w_fc = load_to_gpu_bf16(dir + "eagle_fc_weight.f32.bin")
            self.w_q = load_to_gpu_bf16(dir + "eagle_layer_self_attn_q_proj_weight.f32.bin")
            self.w_k = load_to_gpu_bf16(dir + "eagle_layer_self_attn_k_proj_weight.f32.bin")
            self.w_v = load_to_gpu_bf16(dir + "eagle_layer_self_attn_v_proj_weight.f32.bin")
            self.w_o = load_to_gpu_bf16(dir + "eagle_layer_self_attn_o_proj_weight.f32.bin")
            self.w_ffn_gate = load_to_gpu_bf16(dir + "eagle_layer_mlp_gate_proj_weight.f32.bin")
            self.w_ffn_up = load_to_gpu_bf16(dir + "eagle_layer_mlp_up_proj_weight.f32.bin")
            self.w_ffn_down = load_to_gpu_bf16(dir + "eagle_layer_mlp_down_proj_weight.f32.bin")
            self.w_lm_head = load_to_gpu_bf16(dir + "eagle_lm_head_weight.f32.bin")
        else:
            # Q8 legacy (GB10 stems)
            self.w_fc = load_to_gpu_q8(dir + "eagle3_fc_weight.q8")  # [dB, 3*dB]
            self.w_q = load_to_gpu_q8(dir + "eagle3_q_proj_weight.q8")  # [nh*hd, 2*dB]
            self.w_k = load_to_gpu_q8(dir + "eagle3_k_proj_weight.q8")  # [nkv*hd, 2*dB]
            self.w_v = load_to_gpu_q8(dir + "eagle3_v_proj_weight.q8")  # [nkv*hd, 2*dB]
            self.w_o = load_to_gpu_q8(dir + "eagle3_o_proj_weight.q8")  # [dB, nh*hd]
            self.w_ffn_gate = load_to_gpu_q8(dir + "eagle3_ffn_gate_weight.q8")
            self.w_ffn_up = load_to_gpu_q8(dir + "eagle3_ffn_up_weight.q8")
            self.w_ffn_down = load_to_gpu_q8(dir + "eagle3_ffn_down_weight.q8")
            self.w_lm_head = load_to_gpu_q8(dir + "eagle3_lm_head_weight.q8")  # [32K, dB]
        # norms + d2t stay full precision in every mode (stems differ per converter era)
        if mode == EAGLE_MODE_Q8:
            self.w_in_norm_embed = load_to_gpu(dir + "eagle3_input_layernorm_weight.bin")
            self.w_hidden_norm = load_to_gpu(dir + "eagle3_hidden_norm_weight.bin")
            self.w_post_attn_norm = load_to_gpu(dir + "eagle3_post_attention_layernorm_weight.bin")
            self.w_norm = load_to_gpu(dir + "eagle3_norm_weight.bin")
            self.d_d2t_lut = load_to_gpu(dir + "eagle3_d2t.bin")  # [DRAFT_VOCAB] fp32 ADDITIVE offset LUT
        else:
            self.w_in_norm_embed = load_to_gpu(dir + "eagle_layer_input_layernorm_weight.bin")
            self.w_hidden_norm = load_to_gpu(dir + "eagle_layer_hidden_norm_weight.bin")
            self.w_post_attn_norm = load_to_gpu(dir + "eagle_layer_post_attention_layernorm_weight.bin")
            self.w_norm = load_to_gpu(dir + "eagle_norm_weight.bin")
            self.d_d2t_lut = load_to_gpu(dir + "eagle_d2t.bin")  # [DRAFT_VOCAB] fp32 ADDITIVE offset LUT
        # NVFP4 act-quant scratch (allocated in every mode — 2.3 MB, keeps release simple)
        self.s_aq_packed = cuda_malloc(16 * (EAGLE_AQ_KMAX // 2))
        self.s_aq_bs = cuda_malloc(16 * (EAGLE_AQ_KMAX // 16))
        self.s_aq_global = cuda_malloc(16 * 4)
        self.s_aq_cpad = cuda_malloc(16 * EAGLE_NMAX * 4)
        # scratch
        self.s_fused = cuda_malloc(
            self.dB * 4
        )  # hidden_norm'd fused (residual base)
        self.s_h_embed = cuda_malloc(self.dB * 4)
        self.s_concat_in = cuda_malloc(2 * self.dB * 4)
        self.s_q = cuda_malloc(self.nh * self.hd * 4)
        self.s_k = cuda_malloc(self.nkv * self.hd * 4)
        self.s_v = cuda_malloc(self.nkv * self.hd * 4)
        self.s_attn = cuda_malloc(self.dB * 4)
        self.s_norm = cuda_malloc(self.dB * 4)
        self.s_gate = cuda_malloc(self.ff * 4)
        self.s_up = cuda_malloc(self.ff * 4)
        self.s_mlp = cuda_malloc(self.ff * 4)
        self.s_hs = cuda_malloc(self.dB * 4)
        self.s_logits = cuda_malloc(DRAFT_VOCAB * 4)
        self.s_rope_cache = cuda_malloc((self.hd // 2) * 4)
        self.s_scores = cuda_malloc(
            self.nh * MAX_DRAFT_SEQ * 4
        )  # [nh, MAX_DRAFT_SEQ] attend scores scratch
        self.d_taps_buf = cuda_malloc(3 * self.dB * 4)  # [3*dB] live L[1,29,56] tap sink
        self.d_verify_taps_buf = cuda_malloc(MAX_VERIFY_ROWS * 3 * self.dB * 4)
        self.last_verify_rows = 0
        self.seq_len = 0   # empty prefix until nomos_eagle3_commit advances it (or reset)
        self.last_draft_base_len = 0
        self.next_seed_token = 0
        # drafter's own paged KV cache (growing by 1 each step). the
        # eagle3_own_kv_attend reads from this. M=1 path: only 1 K/V per
        # step; the cache grows across the K-step loop. Bare path: pre-alloc
        # K_CAP slots (caller's K=4 default; 8 for safety).
        self.s_drafter_k = cuda_malloc(
            self.nkv * self.hd * 4 * MAX_DRAFT_SEQ
        )  # [8, nkv*hd]
        self.s_drafter_v = cuda_malloc(
            self.nkv * self.hd * 4 * MAX_DRAFT_SEQ
        )  # [8, nkv*hd]
        # Capture buffers for the gold-diff (opt-in, all zeroed by default).
        # 4 captures per step × 8 step slots = 32 captures total.
        # d_fc_out[step]     = [dB]         fused input (post-fc for step 0; prior hs for later steps)
        # d_attn_out[step]   = [nh*hd]      pre-o_proj (the attended output)
        # d_o_out[step]      = [dB]         post-o_proj (narrow)
        # d_hidden_out[step] = [dB]         pre-final-norm (= r2+mlp; the recurrent state)
        # d_logits_out[step] = [DRAFT_VOCAB] pre-d2t draft logits
        self.cap_fc_out = cuda_malloc(self.dB * 4 * 8)  # 8 * dB fp32
        self.cap_attn_out = cuda_malloc(
            self.nh * self.hd * 4 * 8
        )  # 8 * nh*hd fp32
        self.cap_o_out = cuda_malloc(self.dB * 4 * 8)  # 8 * dB fp32
        self.cap_hidden_out = cuda_malloc(self.dB * 4 * 8)  # 8 * dB fp32
        self.cap_logits_out = cuda_malloc(DRAFT_VOCAB * 4 * 8)  # 8 * 32K fp32
        # d2t: capture per-step draft_id + target_id (host-side, 8 slots each).
        self.cap_draft_ids = List[Int32]()
        self.cap_target_ids = List[Int64]()
        for _ in range(8):   # size to K_CAP=8 (was empty → OOB host write at step 0 with capture on)
            self.cap_draft_ids.append(Int32(0))
            self.cap_target_ids.append(Int64(0))
        # opt-in capture flag (default: True; the gold-diff needs it on).
        self.capture = True


# ── D2 mode-dispatched GEMM path ─────────────────────────────────────────────
# NVFP4 runs the R2 W4A4 recipe: quantize each DISTINCT activation once
# (gpu_quant_act_nvfp4_dec -> the drafter-owned scratch), then 1..3 GEMMs share
# the prequantized bytes via the fused-postscale prequant entry (all 9 head
# shapes satisfy qa_fuse_route: K%256==0, N%32==0). GOLD = bf16-weight GEMV
# (exact checkpoint weights, fp32 act/accum). Q8 = the legacy fused GEMV.
def _eagle3_quant_act(ctx: DeviceContext, m: Eagle3Drafter, d_act: UInt64, K: Int) raises:
    """Prequant one distinct activation for the following _eagle3_mm calls (NVFP4 only)."""
    if m.mode == EAGLE_MODE_NVFP4:
        gpu_quant_act_nvfp4_dec(ctx, d_act, m.s_aq_packed, m.s_aq_bs, m.s_aq_global, K)


def _eagle3_mm(
    ctx: DeviceContext, m: Eagle3Drafter,
    d_out: UInt64, d_act: UInt64, w: UInt64, gs: Float32, K: Int, N: Int,
) raises:
    """out[N] = W[N,K] @ act[K]. NVFP4 requires a preceding _eagle3_quant_act(d_act, K)."""
    if m.mode == EAGLE_MODE_NVFP4:
        gpu_matmul_nvfp4_w4a4_prequant_dev(
            ctx, d_out, w, gs,
            m.s_aq_packed, m.s_aq_bs, m.s_aq_global, m.s_aq_cpad,
            1, K, N,
        )
    elif m.mode == EAGLE_MODE_GOLD:
        gpu_matmul_bf16w_gemv_dev(ctx, d_out, d_act, w, K, N)
    else:
        gpu_matmul_q8_fused_dev(ctx, d_out, d_act, w, K, N)


def _eagle3_load_embed_row(m: Eagle3Drafter, embed_fd: Int, token: Int) raises:
    """Host-pread one RAW fp32 embedding row [dB] from the TARGET's mmap'd table
    (bit-identical to the drafter checkpoint's embed_tokens — shared, 2026-07-07)
    and upload it to m.s_h_embed. NO sqrt(D) scale: that is the gemma-4 TARGET
    input convention; the llama-typed drafter consumes raw rows (b1ec92d lineage)."""
    var x = List[Float32](capacity=m.dB)
    for _ in range(m.dB):
        x.append(0.0)
    _ = external_call["nomos_pread", c_size_t](
        c_int(embed_fd), x.unsafe_ptr(), c_size_t(m.dB * 4), c_size_t(token * m.dB * 4))
    cuda_upload(m.s_h_embed, x)


# EAGLE-3 forward: one step. Mirrors `Eagle3MHADraft.__call__` line-by-line.
# Inputs:
#   d_tap_h_or_prior_hs: device pointer to [dB]. If step 1: the pre-norm FUSED
#     (= the post-fc result, 3-tap fused). If step > 1: the prior step's hs
#     (= the drafter's own autoregressive state). The caller distinguishes.
#   seed_pos: int (the position in cache to attend at)
#   seed_token: int (the last real (verified) token)
#   embed_fd: the ENGINE's mmap'd fp32 [262144, dB] embed table fd (shared with
#     the target — bit-identical to the drafter checkpoint's embed_tokens)
# Returns: (target_token: Int, next_hs: device pointer in m.s_hs)
def eagle3_draft_step(
    ctx: DeviceContext,
    mut m: Eagle3Drafter,
    d_tap_h_or_prior_hs: UInt64,  # [dB] = 3-tap fused (step 1) or prior hs (step > 1)
    seed_pos: Int,
    seed_token: Int,
    embed_fd: Int,  # target embed table fd (raw fp32 rows, target-vocab indexed)
    step: Int = 0,  # step index (0..K-1) for capture buffer slot
) raises -> Int:
    # 1) fused = the input (already dB-wide, either from fc or prior hs)
    cuda_memcpy(m.s_fused, d_tap_h_or_prior_hs, m.dB * 4, D2D)
    # capture: d_fc_out[step] = the per-step fused input. Step 0 is the post-fc
    # seed-tap fusion; steps 1..K-1 are the prior drafter hidden fed forward.
    if m.capture:
        cuda_memcpy(
            m.cap_fc_out + UInt64(step * m.dB * 4),
            m.s_fused,
            m.dB * 4,
            D2D,
        )
    # 2) h_embed = embed_table[token] — shared target table row via mmap pread
    #    (raw fp32, no sqrt(D) target scale), uploaded to m.s_h_embed.
    _eagle3_load_embed_row(m, embed_fd, seed_token)
    # 3) norm_embed = input_layernorm(h_embed)
    gpu_rmsnorm_inplace_mojo(ctx, m.s_h_embed, m.w_in_norm_embed, m.dB)
    # 4) norm_fused = hidden_norm(fused)  (in-place on m.s_fused; s_fused is now the normed fused)
    gpu_rmsnorm_inplace_mojo(ctx, m.s_fused, m.w_hidden_norm, m.dB)
    # 5) concat_inputs = cat[norm_embed, norm_fused]  [2*dB = 10752]
    cuda_memcpy(m.s_concat_in, m.s_h_embed, m.dB * 4, D2D)
    cuda_memcpy(m.s_concat_in + UInt64(m.dB * 4), m.s_fused, m.dB * 4, D2D)
    # 6) qkv = qkv_proj @ concat_inputs  [Q+K+V, 2*dB=10752]
    #    R2 dedup: q/k/v share concat_in — quantize ONCE, three prequant GEMMs.
    _eagle3_quant_act(ctx, m, m.s_concat_in, 2 * m.dB)
    # 7a) q = w_q @ concat_in
    _eagle3_mm(ctx, m, m.s_q, m.s_concat_in, m.w_q, m.gs_q, 2 * m.dB, m.nh * m.hd)
    # 7b) k = w_k @ concat_in
    _eagle3_mm(ctx, m, m.s_k, m.s_concat_in, m.w_k, m.gs_k, 2 * m.dB, m.nkv * m.hd)
    # 7c) v = w_v @ concat_in
    _eagle3_mm(ctx, m, m.s_v, m.s_concat_in, m.w_v, m.gs_v, 2 * m.dB, m.nkv * m.hd)
    # 7) RoPE BOTH q and k at the absolute position (θ=1e4, full hd=256; NO q/k-norm — llama-typed).
    #    k MUST be roped before it enters the own-KV cache — correct RoPE rotates both q and k, each at
    #    its own position; caching k unroped breaks relative-position attention (invisible at pos 0).
    var abs_slot = m.seq_len + step          # absolute position in the persistent drafter KV
    var rope_pos = abs_slot + 1              # 1-indexed rope phase (relative offsets preserved)
    gpu_rope_mojo(ctx, m.s_q, rope_pos, m.nh, m.hd, Float32(10000.0), m.hd)
    gpu_rope_mojo(ctx, m.s_k, rope_pos, m.nkv, m.hd, Float32(10000.0), m.hd)
    # 8) attn_out = causal_GQA(q, k, v)  32Q / 16KV / hd=256; 1/sqrt(256) scale lives in eagle3_attn.
    #    eagle3_own_kv_attend(ctx, q, k_new, v_new, kc, vc, scores, out, p, nh, nkv, hd, kmax, kvg):
    #    APPENDS k_new/v_new at slot p into the own-KV cache, then attends causally over [0,p].
    #    p = the DRAFT STEP (cache slot 0..K-1), NOT seed_pos (seed_pos is only the rope phase above).
    var s_attn_in = cuda_malloc(m.nh * m.hd * 4)
    eagle3_own_kv_attend(
        ctx,
        m.s_q,            # d_q (roped)
        m.s_k,            # d_k_new (roped) — appended at slot=step
        m.s_v,            # d_v_new         — appended at slot=step
        m.s_drafter_k,    # d_kc (own-KV cache)
        m.s_drafter_v,    # d_vc (own-KV cache)
        m.s_scores,       # d_scores scratch
        s_attn_in,        # d_out
        abs_slot,         # p = absolute slot in the persistent KV (causal [0,abs_slot] = prefix+drafts)
        m.nh, m.nkv, m.hd,
        MAX_DRAFT_SEQ,    # kmax (persistent cache stride)
        2,                # kvg (32Q / 16KV)
    )
    # capture: d_attn_out[step] = s_attn_in (pre-o_proj)
    if m.capture:
        cuda_memcpy(
            m.cap_attn_out + UInt64(step * m.nh * m.hd * 4),
            s_attn_in,
            m.nh * m.hd * 4,
            D2D,
        )
    # 9) attn_out = o_proj(attn_in)  [dB=5376, nh*hd] @ [nh*hd] -> [dB=5376] (narrow)
    _eagle3_quant_act(ctx, m, s_attn_in, m.nh * m.hd)
    _eagle3_mm(ctx, m, m.s_attn, s_attn_in, m.w_o, m.gs_o, m.nh * m.hd, m.dB)
    # capture: d_o_out[step] = m.s_attn (post-o_proj; s_attn is reused by ffn_down at line 227, so grab it BEFORE)
    if m.capture:
        cuda_memcpy(
            m.cap_o_out + UInt64(step * m.dB * 4),
            m.s_attn,
            m.dB * 4,
            D2D,
        )
    # 10) hs = norm_fused + attn_out  RESIDUAL BASE = hidden_NORM'D FUSED (NOT pre-norm;
    #     norm_before_residual=TRUE, byte-verified vs Eagle3DraftModel checkpoint).
    #     s_fused is already the normed fused (in-place rmsnorm at step 4).
    cuda_memcpy(m.s_hs, m.s_fused, m.dB * 4, D2D)
    gpu_residual_add_mojo(ctx, m.s_hs, m.s_attn, m.dB)
    # 11) post_attention_layernorm → SCRATCH s_norm, NOT in-place. The step-13 residual base is
    #     r2 = the PRE-norm post-attn hs; an in-place norm here destroys r2 (same class as the
    #     final-norm bug). spec: r2=hs; norm=post_attn_norm(hs); mlp=MLP(norm); hs=r2+mlp.
    gpu_rmsnorm_mojo(ctx, m.s_hs, m.s_norm, m.w_post_attn_norm, m.dB)  # sig=(in,out): s_norm=norm(s_hs), s_hs stays r2
    # 12) mlp on the NORMED hs (s_norm); m.s_hs stays = r2. Llama-style silu SwiGLU.
    #     R2 dedup: gate/up share s_norm — quantize ONCE, two prequant GEMMs.
    _eagle3_quant_act(ctx, m, m.s_norm, m.dB)
    _eagle3_mm(ctx, m, m.s_gate, m.s_norm, m.w_ffn_gate, m.gs_gate, m.dB, m.ff)
    _eagle3_mm(ctx, m, m.s_up, m.s_norm, m.w_ffn_up, m.gs_up, m.dB, m.ff)
    gpu_silu_mojo(ctx, m.s_gate, m.ff)  # silu (fixes #2)
    gpu_elementwise_mul_mojo(ctx, m.s_mlp, m.s_gate, m.s_up, m.ff)
    _eagle3_quant_act(ctx, m, m.s_mlp, m.ff)
    _eagle3_mm(ctx, m, m.s_attn, m.s_mlp, m.w_ffn_down, m.gs_down, m.ff, m.dB)
    # 13) hs = r2 + mlp_outs  (r2 = m.s_hs, untouched since the post-attn add at step 10)
    gpu_residual_add_mojo(ctx, m.s_hs, m.s_attn, m.dB)
    # capture: d_hidden_out[step] = m.s_hs (pre-final-norm; the recurrent state)
    if m.capture:
        cuda_memcpy(
            m.cap_hidden_out + UInt64(step * m.dB * 4),
            m.s_hs,
            m.dB * 4,
            D2D,
        )
    # 14) last_token = hs (M=1, the only token). m.s_hs holds the pre-final-norm
    #     r2+mlp output (= the recurrent state for the next step).
    # 15) norm_last_token = norm(last_token)  "norm" = the final post-FFW norm.
    #     RECURRENT-STATE NORM BUG FIX: use the s_norm scratch for the
    #     in-place-ELSE write; do NOT norm into m.s_hs (the in-place
    #     rmsnorm at this step would corrupt the recurrent state from
    #     step 2 on). s_norm = norm(m.s_hs, w_norm); m.s_hs stays
    #     as the pre-norm r2+mlp output for both the recurrence AND
    #     d_hidden_out capture. Matches the spec:
    #       d_hidden_out = pre-self.norm = r2+mlp
    gpu_rmsnorm_mojo(ctx, m.s_hs, m.s_norm, m.w_norm, m.dB)  # sig=(in,out): s_norm=norm(hidden); lm_head reads s_norm
    # 16) logits = lm_head(norm_last_token)  [DRAFT_VOCAB=32K, dB] @ [dB] -> [32K]
    #     Note: lm_head reads m.s_norm (the post-final-norm hidden), NOT m.s_hs.
    #     (s_norm changed since the gate/up prequant -> re-quantize.)
    _eagle3_quant_act(ctx, m, m.s_norm, m.dB)
    _eagle3_mm(ctx, m, m.s_logits, m.s_norm, m.w_lm_head, m.gs_lm_head, m.dB, DRAFT_VOCAB)
    if m.capture:
        cuda_memcpy(
            m.cap_logits_out + UInt64(step * DRAFT_VOCAB * 4),
            m.s_logits,
            DRAFT_VOCAB * 4,
            D2D,
        )
    # 17) target_token = d2t_lut[argmax(logits)]  Wire to the eagle3_argmax_d2t
    #     (f6dc73f: target = draft + d2t[draft]; returns the target id).
    var ids = eagle3_argmax_d2t(ctx, m.s_logits, m.d_d2t_lut, DRAFT_VOCAB)
    var draft_id = ids[0]   # argmax over the draft vocab (32000)
    var target_id = ids[1]  # = draft_id + d2t[draft_id] (additive offset)
    # capture: per-step draft_id (argmax) + target_id (post-d2t)
    if m.capture:
        m.cap_draft_ids[step] = Int32(draft_id)
        m.cap_target_ids[step] = Int64(target_id)
    cuda_free(s_attn_in)   # per-step scratch (robustness: no leak across rounds)
    return target_id


# K-token draft: autoregressive h-next recurrence (fixes #4).
# step 1: fused = fc(target_taps at seed pos).
# steps 2..K: fused = prior step's hs (the drafter's own hidden; skip fc).
# Returns: K draft tokens (target-vocab ids).
def eagle3_draft_k(
    ctx: DeviceContext,
    mut m: Eagle3Drafter,
    d_tap_h_first: UInt64,  # [3, dB] for step 1's fc input (3 aux taps at seed pos)
    seed_pos: Int,
    seed_token: Int,
    embed_fd: Int,
    k: Int,
    mut drafts: List[Int32],
) raises:
    # The K-step draft is speculative. Remember the committed length so
    # nomos_eagle3_commit can roll back the drafter KV before exact replay.
    m.last_draft_base_len = m.seq_len
    # scratch for the 3-tap -> fc result (step 1 only)
    var s_3tap = cuda_malloc(3 * m.dB * 4)
    cuda_memcpy(s_3tap, d_tap_h_first, 3 * m.dB * 4, D2D)
    var s_fused_step1 = cuda_malloc(m.dB * 4)  # the post-fc hidden for step 1
    _eagle3_quant_act(ctx, m, s_3tap, 3 * m.dB)
    _eagle3_mm(ctx, m, s_fused_step1, s_3tap, m.w_fc, m.gs_fc, 3 * m.dB, m.dB)
    var tok = seed_token
    var pos = seed_pos
    var kk = k if k <= 8 else 8   # capture buffers are sized 8; clamp the draft depth
    for step in range(kk):
        # The fused input is either the post-fc hidden (step 0) or the prior
        # step's hs (steps 1..K-1). The prior hs is in m.s_hs from the prior
        # eagle3_draft_step call.
        var d_fused_in: UInt64
        if step == 0:
            d_fused_in = s_fused_step1
        else:
            d_fused_in = m.s_hs  # prior step's hs (autoregressive; fixes #4)
        var draft = eagle3_draft_step(
            ctx, m, d_fused_in, pos, tok, embed_fd, step
        )
        # (own-KV append happens INSIDE eagle3_own_kv_attend at slot=step, on the ROPED k —
        #  no separate append here; a second copy would also re-append unroped K/V from m.s_k.)
        drafts.append(Int32(draft))
        tok = draft  # next input token
        pos = pos + 1  # next position in the drafter's own cache
    cuda_free(s_3tap)         # robustness: free the round's scratch (no leak across rounds)
    cuda_free(s_fused_step1)


def _as_i32_ptr(addr: UInt64) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(addr))


def eagle3_commit_prefix(
    ctx: DeviceContext,
    mut m: Eagle3Drafter,
    prefix_tokens_ptr: UInt64,
    d_prefix_taps: UInt64,
    prefix_count: Int,
    d_next_taps: UInt64,
    next_token: Int,
    embed_fd: Int,
) raises:
    """Rollback speculative drafter KV, replay accepted tokens from target taps,
    and install the raw target taps for the next draft seed.

    prefix_tokens/prefix_taps are the committed tokens that should become part
    of the drafter's persistent prefix. The next token is not appended here:
    nomos_eagle3_draft will append it speculatively as step 0 on the next round.
    """
    m.seq_len = m.last_draft_base_len
    m.next_seed_token = next_token
    if d_next_taps != UInt64(0):
        cuda_memcpy(m.d_taps_buf, d_next_taps, 3 * m.dB * 4, D2D)
    if prefix_count <= 0:
        return
    var old_capture = m.capture
    m.capture = False
    var toks = _as_i32_ptr(prefix_tokens_ptr)
    var s_fused = cuda_malloc(m.dB * 4)
    for i in range(prefix_count):
        var taps_i = d_prefix_taps + UInt64(i * 3 * m.dB * 4)
        _eagle3_quant_act(ctx, m, taps_i, 3 * m.dB)
        _eagle3_mm(ctx, m, s_fused, taps_i, m.w_fc, m.gs_fc, 3 * m.dB, m.dB)
        _ = eagle3_draft_step(
            ctx, m, s_fused, m.seq_len, Int(toks[i]), embed_fd, 0
        )
        m.seq_len += 1
    cuda_free(s_fused)
    m.capture = old_capture


def release_eagle3_drafter_handle(handle: UInt64):
    if handle == 0:
        return
    var ptr = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )
    if ptr[0].w_fc != 0: cuda_free(ptr[0].w_fc)
    if ptr[0].w_in_norm_embed != 0: cuda_free(ptr[0].w_in_norm_embed)
    if ptr[0].w_hidden_norm != 0: cuda_free(ptr[0].w_hidden_norm)
    if ptr[0].w_q != 0: cuda_free(ptr[0].w_q)
    if ptr[0].w_k != 0: cuda_free(ptr[0].w_k)
    if ptr[0].w_v != 0: cuda_free(ptr[0].w_v)
    if ptr[0].w_o != 0: cuda_free(ptr[0].w_o)
    if ptr[0].w_post_attn_norm != 0: cuda_free(ptr[0].w_post_attn_norm)
    if ptr[0].w_ffn_gate != 0: cuda_free(ptr[0].w_ffn_gate)
    if ptr[0].w_ffn_up != 0: cuda_free(ptr[0].w_ffn_up)
    if ptr[0].w_ffn_down != 0: cuda_free(ptr[0].w_ffn_down)
    if ptr[0].w_norm != 0: cuda_free(ptr[0].w_norm)
    if ptr[0].w_lm_head != 0: cuda_free(ptr[0].w_lm_head)
    if ptr[0].d_d2t_lut != 0: cuda_free(ptr[0].d_d2t_lut)
    if ptr[0].s_fused != 0: cuda_free(ptr[0].s_fused)
    if ptr[0].s_h_embed != 0: cuda_free(ptr[0].s_h_embed)
    if ptr[0].s_concat_in != 0: cuda_free(ptr[0].s_concat_in)
    if ptr[0].s_q != 0: cuda_free(ptr[0].s_q)
    if ptr[0].s_k != 0: cuda_free(ptr[0].s_k)
    if ptr[0].s_v != 0: cuda_free(ptr[0].s_v)
    if ptr[0].s_attn != 0: cuda_free(ptr[0].s_attn)
    if ptr[0].s_norm != 0: cuda_free(ptr[0].s_norm)
    if ptr[0].s_gate != 0: cuda_free(ptr[0].s_gate)
    if ptr[0].s_up != 0: cuda_free(ptr[0].s_up)
    if ptr[0].s_mlp != 0: cuda_free(ptr[0].s_mlp)
    if ptr[0].s_hs != 0: cuda_free(ptr[0].s_hs)
    if ptr[0].s_logits != 0: cuda_free(ptr[0].s_logits)
    if ptr[0].s_rope_cache != 0: cuda_free(ptr[0].s_rope_cache)
    if ptr[0].s_scores != 0: cuda_free(ptr[0].s_scores)
    if ptr[0].d_taps_buf != 0: cuda_free(ptr[0].d_taps_buf)
    if ptr[0].d_verify_taps_buf != 0: cuda_free(ptr[0].d_verify_taps_buf)
    if ptr[0].s_drafter_k != 0: cuda_free(ptr[0].s_drafter_k)
    if ptr[0].s_drafter_v != 0: cuda_free(ptr[0].s_drafter_v)
    if ptr[0].cap_fc_out != 0: cuda_free(ptr[0].cap_fc_out)
    if ptr[0].cap_attn_out != 0: cuda_free(ptr[0].cap_attn_out)
    if ptr[0].cap_o_out != 0: cuda_free(ptr[0].cap_o_out)
    if ptr[0].cap_hidden_out != 0: cuda_free(ptr[0].cap_hidden_out)
    if ptr[0].cap_logits_out != 0: cuda_free(ptr[0].cap_logits_out)
    if ptr[0].s_aq_packed != 0: cuda_free(ptr[0].s_aq_packed)
    if ptr[0].s_aq_bs != 0: cuda_free(ptr[0].s_aq_bs)
    if ptr[0].s_aq_global != 0: cuda_free(ptr[0].s_aq_global)
    if ptr[0].s_aq_cpad != 0: cuda_free(ptr[0].s_aq_cpad)
    cuda_sync()
    ptr.destroy_pointee()
    ptr.free()
