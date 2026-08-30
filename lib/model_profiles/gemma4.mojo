# MODEL PROFILE — Gemma-4-31B. Do not import this directly.
#
# The build copies ONE profile to lib/model_config.mojo, which is what the engine
# imports. See refresh_build.sh --model. Everything here is compile-time on purpose:
# these constants size ABI-visible buffers, KV caches, argmax storage and drafter
# surfaces, and several are static layout shapes. Making them runtime is a separate,
# measured piece of work — see docs/MODEL_PROFILES.md.
#
# MODEL_NAME / MODEL_ID are not decoration. The kernel exports MODEL_ID via
# nomos_model_id(), and tools/check_model_identity.py compares it against the weights
# manifest BEFORE the weights are loaded, refusing a mismatch at the front door.
# Before that check existed, pointing a Muse-built .so at Gemma weights failed deep
# inside nomos_init while the banner cheerfully printed the COMPILED-IN model name —
# the error told you nothing about the real cause (2026-08-15, cost real time).

# Stable id for the runtime identity check. The kernel exports this via
# nomos_model_id(); tools/check_model_identity.py maps the weights manifest's
# `model` field to the same number and refuses a mismatch BEFORE loading.
# STATUS: complete — this is the profile main has always built.
alias PROFILE_COMPLETE = 1
alias MODEL_ID = 1
alias MODEL_NAME = "gemma-4-31b"

# Serve policy is selected from the same profile as kernel geometry. Python resolves
# the actual marker IDs from the paired tokenizer and fails loud if they are absent.
# Protocol 1 = Gemma thinking channels; protocol 2 = Muse ATEM recipient routing.
alias SERVE_PROTOCOL = 1
alias SERVE_MODEL_NAME = "nomos-gemma4-31b"

alias D = 5376         # hidden_size
alias NH = 32          # num attention heads
alias NKV = 16         # num KV heads (sliding)
alias HD = 256         # head dim (sliding)
alias FF = 21504       # intermediate (MLP) size
alias VOCAB = 262144
alias TOTAL_LAYERS = 60
alias FULL_HD = 512    # head dim (full attention) — differs from sliding on this model

# Full-attention layers, counted BACKWARD from the last layer:
#   (TOTAL_LAYERS - 1 - layer) % FULL_LAYER_PERIOD == 0
# For Gemma-4 that is every 6th (layers 5, 11, ..., 59 — i.e. the old `layer % 6 == 5`,
# which is the same set written forwards). Expressing both models in the backward form
# is what lets one engine serve either: the forward phase differs, the backward one
# does not.
alias FULL_LAYER_PERIOD = 6

# Sliding layers attend only to the last SLIDING_WINDOW keys. Without this bound they
# over-attend past ~window tokens -> garbage at long context (confirmed 2026-06-14:
# coherent <2.4k, gibberish >5k). Value from the HF config ("sliding_window": 1024).
alias SLIDING_WINDOW = 1024

# ── Capability flags — what this architecture DOES, not just how big it is ───
# Geometry was the easy half. These are the parts where the two models compute
# DIFFERENT THINGS, and they are why a profile can be geometrically correct and still
# produce wrong tokens (see PROFILE_COMPLETE).

# Attention gate: attn_out *= sigmoid(attn_in @ attn_gw) before the O projection.
# Gemma-4 has no such gate.
alias HAS_ATTN_GATE = 0

# RoPE. Gemma-4 ropes BOTH layer types, with different thetas, and the full layers
# rope only a PARTIAL head dim (rdim) rather than the whole head.
alias FULL_LAYERS_NOPE = 0          # 1 = full layers get no positional encoding at all
alias ROPE_THETA_FULL = 1000000.0
alias ROPE_THETA_SLIDING = 10000.0
alias ROPE_FULL_PARTIAL_DIM = 1     # 1 = full layers rope `rdim`, not the full head dim

# LM head. Gemma-4 TIES its head to the embedding table, so there is no lm_head_weight
# file at all and embed_tokens_weight serves as the head. Muse ships a genuinely
# UNTIED head (its manifest reports cos-vs-embed ~= -0.0009, i.e. unrelated weights).
# Getting this wrong does not degrade quality — it fails to load, or silently decodes
# through the wrong matrix.
alias UNTIED_LM_HEAD = 0

# ── Architecture divergences found by the Muse port, 2026-08-15 ──────────────
# FIVE of these six are SILENT: get them wrong and the kernel builds, runs, and emits
# WRONG TOKENS. Only the V-projection one crashes, and only because a null pointer
# reached a GEMM. That asymmetry is why PROFILE_COMPLETE exists and why the gate is
# acceptance rates, not losslessness.

# Full-attention layers take V = K (device-to-device copy); there is no V-projection
# weight on those layers AT ALL -- the files do not exist on disk. Muse projects V on
# every layer.
alias FULL_LAYERS_V_EQ_K = 1

# Full layers use a DIFFERENT head dim and KV-head count from the sliding layers.
alias FULL_NKV = 4

# Learned per-layer Q/K norm weights + a layer scalar, read from the weight tree.
# Muse instead uses parameterless RMSNorm and a fixed Q multiplier.
alias HAS_LEARNED_QK_NORM = 1
alias QK_NORM_FULL_VECTOR = 0
alias QK_NORM_Q_SCALE = 0.0     # unused when HAS_LEARNED_QK_NORM = 1
alias ATTENTION_SCORE_SCALE = 1.0
alias NORM_STYLE_POST = 0
# Norm blobs are already stored as the effective multiplicative scale.
alias NORM_ADD_ONE = 0
alias TARGET_ROPE_YARN = 0
alias TARGET_YARN_FACTOR = 1.0
alias TARGET_YARN_ORIGINAL_MAX = 0
alias TARGET_YARN_BETA_FAST = 0.0
alias TARGET_YARN_BETA_SLOW = 0.0
alias TARGET_YARN_ATTN_FACTOR = 1.0

# Embedding scaling: Gemma multiplies embeddings by sqrt(D); Muse RMS-normalizes them.
alias EMBED_RMSNORM = 0
alias EMBED_SQRT_SCALE = 1              # Gemma scales token embeddings by sqrt(D).

# MLP activation: Gemma is GeGLU, Muse is SiLU-mul.
alias MLP_ACT_SILU = 0

# ── DRAFTER (DFlash assistant) — a SECOND MODEL, not a reshape of the target ──
# My first pass derived the drafter's shape from the TARGET's constants
# (DFLASH_MLP = FF). That was wrong and Codex caught it: Gemma-4's assistant has
# MLP 10752 while the target's FF is 21504, so the gate_up GEMM ran N = 2*21504
# against a 2*10752 weight -> OOB in q4_gemv_dp4a.
#
# What IS legitimately derived, and why: the assistant manifest says
# "head_and_embed = NONE by design; target embed in, target head out". The drafter
# consumes target hidden states and proposes target tokens, so DFLASH_HIDDEN must
# equal D and DFLASH_VOCAB must equal VOCAB — those are INTERFACE CONSTRAINTS. Its
# depth, width and tap count are its own and cannot be inferred from the target.
alias DRAFTER_MLP = 10752
alias DRAFTER_TAPS = 6
alias DRAFTER_Q_HEADS = 64
alias DRAFTER_KV_HEADS = 8
alias DRAFTER_BLOCK = 16
alias DRAFTER_CANDIDATES = 15
alias DRAFTER_READOUT_SKIP_ROWS = 1
alias DRAFTER_WEIGHTS_BF16 = 0
alias DRAFTER_ROPE_YARN = 0
alias DRAFTER_YARN_FACTOR = 1.0
alias DRAFTER_YARN_ORIGINAL_MAX = 0
alias DRAFTER_YARN_BETA_FAST = 0.0
alias DRAFTER_YARN_BETA_SLOW = 0.0
alias DRAFTER_MARKOV_RANK = 0
alias DRAFTER_MASK_ELISION_SINGLE_BLOCK = 0

# Assistant SCALARS. These are per-assistant, and they are profile constants rather
# than manifest reads for a measured reason: the two assistants carry ASYMMETRIC
# metadata. The Gemma assistant ships a config.json with intermediate_size, rope_theta,
# rms_norm_eps, sliding_window, num_attention_heads and final_logit_softcapping -- but
# NO mask_token_id. The Muse assistant ships no config.json at all, only a
# convert_manifest.json that has mask_token_id but none of the rest. Reading from disk
# would therefore work for one model and fail for the other. Unifying those two schemas
# is worth doing, and it is a separate job from this port.
#
# tools/check_model_identity.py's lesson applies here too: where metadata DOES exist,
# validate these against it at load and fail loud. Do not silently prefer either side.
alias DRAFTER_MASK_TOKEN_ID = 4
alias DRAFTER_ROPE_THETA = 1000000.0
alias DRAFTER_RMS_EPS = 0.000001
alias DRAFTER_SOFTCAP = 30.0
alias DRAFTER_SLIDING_WINDOW = 2048

# ── Drafter TAP LAYERS — explicit IDs, deliberately NOT derived ──────────────
# Which target layers the drafter reads hidden states from. These are a property of
# the TRAINED ASSISTANT -- whoever trained it chose them -- not of the target's
# geometry, so they cannot be computed from TOTAL_LAYERS.
#
# I checked whether a formula fits, because one nearly does:
#     tap_i = 1 + round(i * (TOTAL_LAYERS - 4) / (DRAFTER_TAPS - 1))
# reproduces BOTH models exactly (gemma4 1,12,23,35,46,57 and muse 1,13,25,37,49).
# It is still recorded as literals, and that is the point: a formula that fits the two
# models you happen to have is not a contract. That is precisely how DFLASH_MLP = FF
# passed -- correct on Muse by coincidence, an OOB the moment the other profile built.
# If you add a third profile, read its tap IDs off the trained assistant. Do not trust
# the formula above; it is an observation, not a rule.
#
# TAP_ vs EAGLE3_TAP_: two DIFFERENT drafters with different tap sets, both live in
# nomos_ffi.mojo (nomos_dflash_load and nomos_eagle3_load). Unused slots are -1 and are
# bounded by the COUNT alias, never by scanning for the sentinel.
alias DRAFTER_TAP_0 = 1
alias DRAFTER_TAP_1 = 12
alias DRAFTER_TAP_2 = 23
alias DRAFTER_TAP_3 = 35
alias DRAFTER_TAP_4 = 46
alias DRAFTER_TAP_5 = 57

alias EAGLE3_TAPS = 3
alias EAGLE3_TAP_0 = 1
alias EAGLE3_TAP_1 = 29
alias EAGLE3_TAP_2 = 56

# ── RMS-norm epsilon, PER SITE ───────────────────────────────────────────────
# Not one value per model: Muse uses DIFFERENT epsilons at different norms (1e-5 at the
# input / QK / pre-FF / final norms, 1e-8 at post-attn and post-FF). Gemma-4 uses the
# shared library default at every site.
#
# Gemma's sites pass NO eps argument at all today and inherit EPS_RMSNORM = 1e-6 from
# lib/ops_gpu_mojo_reductions.mojo:31. Naming 1e-6 explicitly here does not change its
# behaviour; it stops the value being invisible, and it means a future change to that
# library default cannot silently move Gemma-4's numerics. Wire every site to these
# aliases even where the value equals the default.
#
# Do NOT couple these to HAS_LEARNED_QK_NORM or any other flag. Epsilon and norm-style
# are independent facts that happen to differ together across the two models we have --
# that is the DFLASH_MLP = FF trap wearing a different hat.
alias RMS_EPS_INPUT = 0.000001
alias RMS_EPS_QK = 0.000001
alias RMS_EPS_POST_ATTN = 0.000001
alias RMS_EPS_PRE_FF = 0.000001
alias RMS_EPS_POST_FF = 0.000001
alias RMS_EPS_FINAL = 0.000001

# ── Drafter INTERFACE + target output scalars ────────────────────────────────
# All trained-artifact properties. None are derivable from vocab size or model id --
# Codex refused to branch on those and was right: vocab CORRELATES with the correct
# choice across our two models without CAUSING it.

# Readout route. Measured (branch commit 8da87a6): the relaxed MMQ route is NOT correct
# for Muse's 15x202048 target-head readout -- it collapses onto the special-token tail,
# logit cosine 0.017, while the v4 Q4 path gives 0.969 against the HF reference.
# Gemma-4 uses the relaxed MMQ route, which is its validated path. This is a
# CORRECTNESS route per assistant, not a perf preference.
alias DRAFTER_READOUT_V4 = 0

# Drafter attention: 4 sliding layers then a full one. Confirmed by the assistant's own
# config.json: layer_types = [sliding x4, full_attention].
alias DRAFTER_SLIDING_LAYERS = 4

# The drafter consumes target embeddings scaled by sqrt(DFLASH_HIDDEN).
alias DRAFTER_EMBED_SQRT_SCALE = 1

# Target-side output scalars.
alias TARGET_SOFTCAP = 30.0
alias TARGET_BOS_ID = 2
alias TARGET_EOS_ID_0 = 1
alias TARGET_EOS_ID_1 = 106

# Target output multiplier, applied to logits before softcap.
#
# THIS IS THE ONE THING THE ACCEPTANCE GATE CANNOT SEE, and Codex refused to mark the
# profile complete without it. A uniform positive scale is monotonic, and softcap
# (cap * tanh(x/cap)) is monotonic too, so scale-then-softcap preserves ORDER: greedy
# argmax is identical either way, and acceptance rates therefore cannot detect a wrong
# value. Gemma-4 passes its exact gate with this set wrong.
#
# It still matters, because raw logit VALUES change: sampling and temperature,
# logit-based interp probes, calibration, and any published logit number. So this needs
# a gate whose reference sits outside greedy decode -- a raw-logit comparison against
# the HF reference -- which we do not currently run. Recording that limit here rather
# than letting a passing acceptance gate imply more than it proves.
alias TARGET_OUTPUT_MULTIPLIER = 1.0

alias MAX_PROBE_TOKENS = 128  # per-token buffer bound (reserved)
alias PROBE_TOPK = 32         # top-K logits kept per probed position (reserved)

# ── GDN (Gated-DeltaNet) capability contract — Gemma-4 is NOT a linear-attention hybrid ──
# The shared engine + lib/gdn_state.mojo reference these, so EVERY profile must define them or the
# GDN code will not compile for this build (same reason muse's HAS_ATTN_GATE also lives here). 0 =
# no GDN layers: GdnStatePools is never constructed and gdn_slot_for_layer() returns -1 everywhere.
alias HAS_LINEAR_ATTENTION = 0
alias GDN_NUM_V_HEADS = 0
alias GDN_NUM_K_HEADS = 0
alias GDN_KEY_HEAD_DIM = 0
alias GDN_VALUE_HEAD_DIM = 0
alias GDN_KEY_DIM = 0        # GDN_NUM_K_HEADS * GDN_KEY_HEAD_DIM
alias GDN_VALUE_DIM = 0      # GDN_NUM_V_HEADS * GDN_VALUE_HEAD_DIM
alias GDN_GQA_EXPAND = 0     # GDN_NUM_V_HEADS // GDN_NUM_K_HEADS
alias GDN_CONV_DIM = 0
alias GDN_CONV_KERNEL = 0

# Softmax-attn capability contract (shared engine imports on every profile; Codex compile finds):
alias ATTN_GATE_IN_QPROJ = 0   # 1 = q_proj is [NH,2*HD] interleaved query|gate (qwen); 0 = separate/none
alias HAS_V_NORM = 1           # Gemma V-normalizes under the learned-QK-norm path; qwen/muse do not
