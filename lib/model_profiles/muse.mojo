# MODEL PROFILE — Muse-Glimmer-30B (text tower). Do not import this directly.
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
# STATUS: COMPLETE for the KERNEL. Verified 2026-08-15 on the 5090: 6/6
# identity=EXACT, base ~28.9 tok/s, and Gemma-4 unchanged at .317/.534/.298/.613.
#
# Muse-Glimmer is not merely a differently-shaped Gemma-4; the architecture below is
# what took seventeen capabilities to express, and ALL of it has landed (61c0538):
#   - the attention gate (attn *= sigmoid(gate_proj(layer_input)); GemmaEngine.d_attn_gw)
#   - NoPE on the full layers (FULL_LAYERS_NOPE)
#   - the DFlash drafter's own shape (DRAFTER_MLP / _TAPS / _Q_HEADS)
#   - logit softcapping / output multiplier (TARGET_SOFTCAP / TARGET_OUTPUT_MULTIPLIER)
#
# SERVE POLICY is complete too (2026-08-16). The compiled profile ID selects ATEM;
# pyhost resolves <|start|>/<|message|>/<|eot|>/<|eom|> from the paired tokenizer,
# preserves <|eom|> as the self->user channel transition, and stops on <|eot|>. The
# shipped template owns prompt rendering; no Gemma <|channel> suffix is appended.
#
# Do NOT read a serve-layer failure as a kernel failure, and do not read fluent output as
# proof of kernel correctness either — tools/muse_parity.py + the goldens adjudicate the
# numerics, and that is a separate question from whether a turn ends.
#
# WHY THIS COMMENT WAS WRONG UNTIL 2026-08-16: it said GEOMETRY-ONLY and claimed
# refresh_build.sh REFUSES --model muse, directly above PROFILE_COMPLETE = 1. The commit
# that landed the architecture flipped the flag and left the prose. nano caught it: "a
# comment saying broken next to a flag saying fine is how someone trusts the wrong one." 
alias PROFILE_COMPLETE = 1
alias MODEL_ID = 2
alias MODEL_NAME = "muse-glimmer-30b"

# Protocol 1 = Gemma thinking channels; protocol 2 = Muse ATEM recipient routing.
alias SERVE_PROTOCOL = 2
alias SERVE_MODEL_NAME = "nomos-muse-glimmer-30b"

alias D = 6656         # hidden_size
alias NH = 32          # num attention heads
alias NKV = 2          # num KV heads (uniform, GQA 16:1)
alias HD = 128         # head dim (uniform)
alias FF = 19968       # intermediate (MLP) size
alias VOCAB = 202048
alias TOTAL_LAYERS = 52
alias FULL_HD = HD     # uniform on this model — full and sliding share head dim

# Full-attention layers, counted BACKWARD from the last layer:
#   (TOTAL_LAYERS - 1 - layer) % FULL_LAYER_PERIOD == 0
# For Muse that is every 4th (layers 3, 7, ..., 51) — 13 full, 39 sliding, the 3:1
# pattern. These full layers are also NoPE (layer_rope_theta = 0). Writing the rule
# backward is what lets one engine serve either model; Gemma-4 uses the same
# expression with period 6.
alias FULL_LAYER_PERIOD = 4

# Sliding layers attend only to the last SLIDING_WINDOW keys. Muse's window is 2048
# (twice Gemma-4's). NOTE: the mask is a MODEL PROPERTY — kv_swa does not change it.
alias SLIDING_WINDOW = 2048

# ── Capability flags — what this architecture DOES, not just how big it is ───
# Geometry was the easy half. These are the parts where the two models compute
# DIFFERENT THINGS, and they are why a profile can be geometrically correct and still
# produce wrong tokens (see PROFILE_COMPLETE).

# Attention gate: attn_out *= sigmoid(attn_in @ attn_gw) before the O projection.
# This is Muse's distinguishing feature and needs a real weight (GemmaEngine.d_attn_gw)
# plus an extra GEMM per layer — it cannot be emulated by reshaping anything.
alias HAS_ATTN_GATE = 1

# RoPE. Muse ropes ONLY the sliding layers, at theta 500000 over the whole head dim.
# Its full layers are NoPE (layer_rope_theta = 0 in the HF config) — they carry no
# positional encoding, which is what makes them the long-range path.
alias FULL_LAYERS_NOPE = 1
alias ROPE_THETA_FULL = 0.0         # unused while FULL_LAYERS_NOPE = 1
alias ROPE_THETA_SLIDING = 500000.0
alias ROPE_FULL_PARTIAL_DIM = 0

# LM head. Gemma-4 TIES its head to the embedding table, so there is no lm_head_weight
# file at all and embed_tokens_weight serves as the head. Muse ships a genuinely
# UNTIED head (its manifest reports cos-vs-embed ~= -0.0009, i.e. unrelated weights).
# Getting this wrong does not degrade quality — it fails to load, or silently decodes
# through the wrong matrix.
alias UNTIED_LM_HEAD = 1

# ── Architecture divergences found by the Muse port, 2026-08-15 ──────────────
# See the gemma4 profile for why these matter: five of the six are SILENT, producing
# wrong tokens rather than a crash.

# Muse projects V on EVERY layer, including the full/NoPE ones.
alias FULL_LAYERS_V_EQ_K = 0

# Uniform across both routes on this model.
alias FULL_NKV = NKV

# Parameterless RMSNorm on Q/K plus a fixed Q multiplier, rather than learned norms.
alias HAS_LEARNED_QK_NORM = 0
alias QK_NORM_FULL_VECTOR = 0
alias QK_NORM_Q_SCALE = 0.34206290543186163   # = 3.87 / sqrt(128)
alias ATTENTION_SCORE_SCALE = 1.0
alias NORM_STYLE_POST = 0
alias NORM_ADD_ONE = 0
alias TARGET_ROPE_YARN = 0
alias TARGET_YARN_FACTOR = 1.0
alias TARGET_YARN_ORIGINAL_MAX = 0
alias TARGET_YARN_BETA_FAST = 0.0
alias TARGET_YARN_BETA_SLOW = 0.0
alias TARGET_YARN_ATTN_FACTOR = 1.0

alias EMBED_RMSNORM = 1
alias EMBED_SQRT_SCALE = 0              # Muse uses its embedding RMSNorm instead.
alias MLP_ACT_SILU = 1

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
alias DRAFTER_MLP = 19968        # coincidentally equals Muse's FF; NOT derived from it
alias DRAFTER_TAPS = 5
alias DRAFTER_Q_HEADS = 32
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
alias DRAFTER_MASK_TOKEN_ID = 201818
alias DRAFTER_ROPE_THETA = 500000.0
alias DRAFTER_RMS_EPS = 0.00001
alias DRAFTER_SOFTCAP = 20.0
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
alias DRAFTER_TAP_1 = 13
alias DRAFTER_TAP_2 = 25
alias DRAFTER_TAP_3 = 37
alias DRAFTER_TAP_4 = 49
alias DRAFTER_TAP_5 = -1        # unused: DRAFTER_TAPS = 5

# No Eagle3 drafter on this model.
alias EAGLE3_TAPS = 0
alias EAGLE3_TAP_0 = -1
alias EAGLE3_TAP_1 = -1
alias EAGLE3_TAP_2 = -1

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
alias RMS_EPS_INPUT = 0.00001
alias RMS_EPS_QK = 0.00001
alias RMS_EPS_POST_ATTN = 0.00000001
alias RMS_EPS_PRE_FF = 0.00001
alias RMS_EPS_POST_FF = 0.00000001
alias RMS_EPS_FINAL = 0.00001

# ── Drafter INTERFACE + target output scalars ────────────────────────────────
# All trained-artifact properties. None are derivable from vocab size or model id --
# Codex refused to branch on those and was right: vocab CORRELATES with the correct
# choice across our two models without CAUSING it.

# Readout route. Measured (branch commit 8da87a6): the relaxed MMQ route is NOT correct
# for Muse's 15x202048 target-head readout -- it collapses onto the special-token tail,
# logit cosine 0.017, while the v4 Q4 path gives 0.969 against the HF reference.
# Gemma-4 uses the relaxed MMQ route, which is its validated path. This is a
# CORRECTNESS route per assistant, not a perf preference.
alias DRAFTER_READOUT_V4 = 1

# All 5 drafter layers slide on this assistant.
alias DRAFTER_SLIDING_LAYERS = 5

# Muse's DFlash consumes the target's RAW lookup rows: upstream deliberately bypasses
# MuseGlimmerTextNormedEmbedding with F.embedding, so apply NEITHER the target
# embedding RMSNorm NOR a sqrt(D) scale.
alias DRAFTER_EMBED_SQRT_SCALE = 0

alias TARGET_SOFTCAP = 20.0
alias TARGET_BOS_ID = 200000

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
        # = 1/sqrt(26), from the model config
alias TARGET_OUTPUT_MULTIPLIER = 0.19611613513818404

alias MAX_PROBE_TOKENS = 128  # per-token buffer bound (reserved)
alias PROBE_TOPK = 32         # top-K logits kept per probed position (reserved)

# ── GDN (Gated-DeltaNet) capability contract — Muse-Glimmer is NOT a linear-attention hybrid ──
# The shared engine + lib/gdn_state.mojo reference these, so EVERY profile must define them or the
# GDN code will not compile for this build. 0 = no GDN layers: GdnStatePools is never constructed
# and gdn_slot_for_layer() returns -1 everywhere.
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
alias ATTN_GATE_IN_QPROJ = 0   # muse uses a separate gate_proj (HAS_ATTN_GATE), not q_proj-embedded
alias HAS_V_NORM = 0           # muse QK-norm is parameterless Q/K only, no V norm
