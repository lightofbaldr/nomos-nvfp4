# MODEL PROFILE — Qwen3.8-27B (qwen3_5 arch, text tower). Do not import this directly.
#
# The build copies ONE profile to lib/model_config.mojo, which the engine imports. See
# refresh_build.sh --model. Everything here is compile-time on purpose: these constants
# size ABI-visible buffers, KV caches, argmax storage and drafter surfaces.
#
# ┌─ STATUS: COMPLETE FOR TARGET INFERENCE. PROFILE_COMPLETE = 1. ────────────────┐
# │ The shared engine implements all 64 layers: 48 Gated-DeltaNet layers plus 16   │
# │ global softmax-attention layers. Prefill/decode, recurrent state, untied head, │
# │ and the debug/parity surfaces are wired. G1 passed on sm_120 NVFP4 (2026-08-17).│
# │ Speculative rollback for recurrent state remains a separate design task and is │
# │ not part of the target-inference completeness contract.                         │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# EVERY constant below is CONFIRMED from one of: unsloth/Qwen3.8-27B config.json,
# the MAX qwen3_5 source audit (weight_adapters/state_cache/gated_delta.mojo), or
# Kvasir's source audit of modeling_qwen3_5.py. Items still needing work are
# marked TODO where they describe deferred performance/serve work, not arithmetic gaps.

alias PROFILE_COMPLETE = 1        # GDN + softmax target prefill/decode passed deploy-card G1.
alias MODEL_ID = 3                # gemma4 = 1, muse = 2, qwen3_5 = 3 (no collision). nomos_model_id() exports this.
alias MODEL_NAME = "qwen3.8-27b"  # TODO: verify byte-exact against the weights manifest `model` field at convert time.

# TODO(serve): Qwen chat template / channel protocol. 1 = Gemma channels, 2 = Muse ATEM; qwen3_5 needs its own.
alias SERVE_PROTOCOL = 3
alias SERVE_MODEL_NAME = "nomos-qwen3.8-27b"

# ── Geometry (config.json text_config — confirmed exact vs the spec) ──────────
alias D = 5120           # hidden_size
alias NH = 24            # num_attention_heads (softmax layers)
alias NKV = 4            # num_key_value_heads (softmax layers, GQA 6:1)
alias HD = 256           # head_dim. NOTE: HD=256 > 128 → KV cache page_size MUST be >= 256 (MAX bumps
                         # kv_cache_page_size to head_dim; the engine's compile-time HD-sized layout
                         # preserves the full 256-wide softmax-layer K/V rows.
alias FF = 17408         # intermediate_size (MLP)
alias VOCAB = 248320
alias TOTAL_LAYERS = 64
alias FULL_HD = HD

# ── Layer types: 16 SOFTMAX + 48 GDN — the second-engine core ────────────────
# config full_attention_interval = 4. layer_types[i] = full_attention iff (i+1) % 4 == 0, else
# linear_attention. That is layers 3,7,...,63 = 16 SOFTMAX (global full-attention), the other 48 = GDN.
#
# Reusing Muse's BACKWARD period expression (TOTAL_LAYERS-1-layer) % FULL_LAYER_PERIOD == 0 gives
# i % 4 == 3 for TOTAL_LAYERS=64 — identical set to (i+1)%4==0. So FULL_LAYER_PERIOD marks the SOFTMAX
# layers correctly. THE DIFFERENCE FROM MUSE: the NON-full layers here are NOT sliding-softmax, they are
# GDN. The engine's sliding-vs-full binary cannot express this; a non-full qwen3_5 layer must route to
# the GDN path, never to a sliding-softmax path. The engine routes this through HAS_LINEAR_ATTENTION.
alias FULL_LAYER_PERIOD = 4
alias HAS_LINEAR_ATTENTION = 1        # NEW: the non-full layers are Gated-DeltaNet, not sliding-softmax.

# qwen3_5 softmax layers are GLOBAL full-attention — there is NO sliding-window softmax layer on this
# model. SLIDING_WINDOW is therefore N/A; the sliding KV cache is unused. Placeholder only.
alias SLIDING_WINDOW = 4096           # N/A on qwen3_5 — no layer routes to the sliding path.

# ── GDN (Gated-DeltaNet) parameters — NEW aliases, no prior model has these ───
# The linear-attention scan geometry (config + MAX audit). Kvasir owns the scan; the PROFILE owns
# the shapes. Recurrent state per GDN layer = [GDN_NUM_V_HEADS, GDN_KEY_HEAD_DIM, GDN_VALUE_HEAD_DIM]
# = [48,128,128] = 786432 elts; STORED bf16 (1.5 MiB/layer × 48 = 72 MiB/slot). Conv state
# = [conv_dim=10240, GDN_CONV_KERNEL] (HF cache keeps the full kernel width, 4).
alias GDN_NUM_K_HEADS = 16            # linear_num_key_heads
alias GDN_NUM_V_HEADS = 48           # linear_num_value_heads
alias GDN_KEY_HEAD_DIM = 128         # linear_key_head_dim
alias GDN_VALUE_HEAD_DIM = 128       # linear_value_head_dim
alias GDN_CONV_KERNEL = 4            # linear_conv_kernel_dim
alias GDN_KEY_DIM = 2048             # GDN_NUM_K_HEADS * GDN_KEY_HEAD_DIM
alias GDN_VALUE_DIM = 6144           # GDN_NUM_V_HEADS * GDN_VALUE_HEAD_DIM
alias GDN_CONV_DIM = 10240           # 2*GDN_KEY_DIM + GDN_VALUE_DIM (Q|K conv + V)
alias GDN_GQA_EXPAND = 3             # GDN_NUM_V_HEADS // GDN_NUM_K_HEADS — query/key repeat_interleave
# GDN dtype law: the SSM COMPUTE dtype is fp32 (config mamba_ssm_dtype=float32); the STORED state pool
# is bf16 (MAX GatedDeltaNetStateCache dtype = model dtype). Two different dtypes — a single-state-dtype
# kernel diverges silently. Reference goldens capture state as fp16 (NOT byte-exact bf16 — calibrate
# thresholds as fp16). The kernel is dtype-parametric: work_dtype fp32, state_dtype bf16.
alias GDN_STATE_DTYPE_IS_BF16 = 1
alias GDN_COMPUTE_FP32 = 1
# GDN scan silent-fail deltas (Kvasir owns the impl; recorded here so the contract is one place):
#   - use_qk_l2norm_in_kernel = True  (q,k L2-normalized INSIDE the scan, not before)
#   - g = -A_log.float().exp() * softplus(a.float() + dt_bias)   (decay gate, fp32)
#   - beta = b.sigmoid()
#   - readout = norm(core_attn_out, z)  — gated RMSNorm, z the gate (direct weight, offset 0.0)
#   - output_gate_type = swish

# ── Softmax-layer capabilities (the 16 full-attention layers) ────────────────
# Partial RoPE: only partial_rotary_factor * head_dim = 0.25 * 256 = 64 dims are rotated (interleaved
# pattern forced). rope_theta = 1e7. These layers DO rope (NOT NoPE — unlike Muse's full layers).
# CONFIRMED from text_config.rope_parameters = {mrope_interleaved:True, mrope_section:[11,11,10],
# rope_theta:10000000, rope_type:'default', partial_rotary_factor:0.25} (nested there, not top-level).
alias ROPE_THETA_FULL = 10000000.0
alias ROPE_THETA_SLIDING = 10000000.0   # N/A (no sliding layers) — set equal to avoid a stray 0.
alias ROPE_FULL_PARTIAL_DIM = 64        # 0.25 * 256; only these dims rotate.
alias FULL_LAYERS_NOPE = 0              # qwen3_5 full layers ARE positionally encoded (partial RoPE).
# mRoPE: mrope_interleaved=True, mrope_section=[11,11,10]. Reads as vision-only and WILL get dropped when
# the tower is de-scoped — then it silently corrupts text-path positions (spec §3 trap 1). It STAYS on
# the text path. Verified present in the goldens capture and wired as standard partial RoPE for text.
alias MROPE_SECTION_T = 11
alias MROPE_SECTION_H = 11
alias MROPE_SECTION_W = 10

# Attention output gate: q_proj is DOUBLE-WIDTH — out_dim = head_dim*NH*2, interleaved per head as
# [h0_query, h0_gate, h1_query, h1_gate, ...]; the second half is a sigmoid output gate applied before
# o_proj. Distinct from Muse's separate gate_proj — here the gate rides inside q_proj. Assuming a single-
# width q_proj mis-reads every head. The softmax route splits query|gate after the projection.
alias HAS_ATTN_GATE = 1
alias ATTN_GATE_IN_QPROJ = 1            # NEW: gate is the 2nd half of a double-width q_proj, not a separate weight.

# Learned q_norm / k_norm on the softmax layers (RMSNorm over head_dim, (1+weight) offset — Qwen HAS
# these weights, unlike Muse's parameterless QK norm). Do NOT apply Muse's fixed Q scale here.
alias HAS_LEARNED_QK_NORM = 1
alias QK_NORM_FULL_VECTOR = 0
alias QK_NORM_Q_SCALE = 1.0             # unused while HAS_LEARNED_QK_NORM = 1 (learned norm carries it).
alias ATTENTION_SCORE_SCALE = 1.0
alias NORM_STYLE_POST = 0
alias NORM_ADD_ONE = 0
alias TARGET_ROPE_YARN = 0
alias TARGET_YARN_FACTOR = 1.0
alias TARGET_YARN_ORIGINAL_MAX = 0
alias TARGET_YARN_BETA_FAST = 0.0
alias TARGET_YARN_BETA_SLOW = 0.0
alias TARGET_YARN_ATTN_FACTOR = 1.0
# Qwen normalizes Q AND K only — NOT V. The shared HAS_LEARNED_QK_NORM path V-normalizes (correct for
# Gemma), which would be a silent part-whole lift on Qwen. (Codex find.)
alias HAS_V_NORM = 0

# attention_bias = False (config) → softmax q/k/v/o projections have NO bias. Assuming bias mis-loads
# the weight layout. (The GDN in_proj_{qkv,z,b,a} are also bias-free per the MAX weight adapter.)
alias ATTENTION_BIAS = 0

# Full-attention (softmax) layer KV geometry — shared engine / gemma4_layer import these on every profile.
# qwen3_5 has NO sliding layers (all 16 softmax are global full-attention), so FULL_NKV == NKV == 4.
# V is projected DISTINCT from K on the full layers (v_proj is its own tensor), so V != K.
alias FULL_NKV = NKV                 # 4 — num_key_value_heads on the softmax layers
alias FULL_LAYERS_V_EQ_K = 0         # V distinct from K (separate v_proj), NOT reused from K

# ── Embedding / LM head / output scalars ─────────────────────────────────────
alias UNTIED_LM_HEAD = 1                # tie_word_embeddings = False → distinct lm_head weights (Codex audit).
alias EMBED_RMSNORM = 0                 # hidden_states[0] = RAW embed_tokens output, NO embedding norm (Codex audit).
alias EMBED_SQRT_SCALE = 0              # Qwen uses the raw lookup: no norm and no Gemma sqrt(D) scale.
alias MLP_ACT_SILU = 1                  # hidden_act = silu.
# NO output multiplier, NO logit softcap on qwen3_5 (bare lm_head; none of output_multiplier /
# final_logit_softcapping / logit_scale in config — Codex audit). Disabled explicitly.
alias TARGET_SOFTCAP = 0.0              # 0 = disabled.
alias TARGET_OUTPUT_MULTIPLIER = 1.0    # 1 = no scaling.
alias TARGET_BOS_ID = 248044           # config bos_token_id (== eos_token_id 248044; pad_token_id = None).

# ── RMS-norm epsilon (config rms_norm_eps = 1e-6, uniform — unlike Muse's per-site split) ─
alias RMS_EPS_INPUT = 0.000001
alias RMS_EPS_QK = 0.000001
alias RMS_EPS_POST_ATTN = 0.000001
alias RMS_EPS_PRE_FF = 0.000001
alias RMS_EPS_POST_FF = 0.000001
alias RMS_EPS_FINAL = 0.000001
alias GDN_NORM_EPS = 0.000001          # gated RMSNorm eps (verify vs config).

# ── DRAFTER (DSpark) — RadixArk/Qwen3.8-27B-DSpark, GATED behind the #518 roofline ──
# Decode drafter (Adam 2026-08-16): a 1.36B plain-qwen3 head, NOT the native MTP head, NOT a trained
# DFlash. Our convert_dspark_q8 family. License "other" → internal only. ALL drafter constants below are
# TODO — the drafter is downstream of the raw-GDN-decode roofline gate (#518) and is not on the critical
# path yet. Interface constraints (like Muse's): DRAFTER_HIDDEN must == D, DRAFTER_VOCAB must == VOCAB.
alias DRAFTER_TAPS = 5
alias DRAFTER_TAP_0 = 4
alias DRAFTER_TAP_1 = 16
alias DRAFTER_TAP_2 = 28
alias DRAFTER_TAP_3 = 40
alias DRAFTER_TAP_4 = 52
alias DRAFTER_TAP_5 = -1
alias DRAFTER_MLP = 10240
alias DRAFTER_Q_HEADS = 40
alias DRAFTER_KV_HEADS = 8
alias DRAFTER_BLOCK = 7
alias DRAFTER_CANDIDATES = 7
alias DRAFTER_READOUT_SKIP_ROWS = 0
alias DRAFTER_WEIGHTS_BF16 = 1
alias DRAFTER_ROPE_YARN = 1
alias DRAFTER_YARN_FACTOR = 32.0
alias DRAFTER_YARN_ORIGINAL_MAX = 8192
alias DRAFTER_YARN_BETA_FAST = 32.0
alias DRAFTER_YARN_BETA_SLOW = 1.0
alias DRAFTER_MARKOV_RANK = 256
alias DRAFTER_MASK_ELISION_SINGLE_BLOCK = 1
alias DRAFTER_MASK_TOKEN_ID = 248077
alias DRAFTER_ROPE_THETA = 10000000.0  # TODO(drafter): verify off the assistant.
alias DRAFTER_RMS_EPS = 0.000001       # TODO(drafter)
alias DRAFTER_SOFTCAP = 0.0            # TODO(drafter)
alias DRAFTER_SLIDING_WINDOW = 262144
alias DRAFTER_SLIDING_LAYERS = 0       # TODO(drafter)
alias DRAFTER_READOUT_V4 = 1          # TODO(drafter): measured route per assistant, not a default.
alias DRAFTER_EMBED_SQRT_SCALE = 0     # TODO(drafter)

# No Eagle3 drafter on this model.
alias EAGLE3_TAPS = 0
alias EAGLE3_TAP_0 = -1
alias EAGLE3_TAP_1 = -1
alias EAGLE3_TAP_2 = -1

alias MAX_PROBE_TOKENS = 128
alias PROBE_TOPK = 32
