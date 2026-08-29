# MODEL PROFILE — allenai/Olmo-3.1-32B-Instruct.
# Reference contract: HF main / OLMo-core training semantics. Transformers
# 5.12.1 incorrectly applies full-attention YaRN to sliding layers as well.
alias PROFILE_COMPLETE = 1
alias MODEL_ID = 4
alias MODEL_NAME = "olmo-3.1-32b-instruct"
alias SERVE_PROTOCOL = 0
alias SERVE_MODEL_NAME = "allenai/Olmo-3.1-32B-Instruct"

alias D = 5120
alias NH = 40
alias NKV = 8
alias HD = 128
alias FULL_HD = HD
alias FULL_NKV = NKV
alias FF = 27648
alias VOCAB = 100278
alias TOTAL_LAYERS = 64
alias FULL_LAYER_PERIOD = 4
alias SLIDING_WINDOW = 4096

alias HAS_ATTN_GATE = 0
alias ATTN_GATE_IN_QPROJ = 0
alias FULL_LAYERS_NOPE = 0
alias ROPE_THETA_FULL = 500000.0
alias ROPE_THETA_SLIDING = 500000.0
alias ROPE_FULL_PARTIAL_DIM = 0
alias TARGET_ROPE_YARN = 1
alias TARGET_YARN_FACTOR = 8.0
alias TARGET_YARN_ORIGINAL_MAX = 8192
alias TARGET_YARN_BETA_FAST = 32.0
alias TARGET_YARN_BETA_SLOW = 1.0
alias TARGET_YARN_ATTN_FACTOR = 1.2079441541679836

alias UNTIED_LM_HEAD = 1
alias FULL_LAYERS_V_EQ_K = 0
alias HAS_LEARNED_QK_NORM = 1
alias QK_NORM_FULL_VECTOR = 1
alias QK_NORM_Q_SCALE = 1.0
# HF eager_attention_forward applies module.scaling = head_dim**-0.5 to QK.
# The inherited Gemma/Muse attention core is deliberately unscaled because
# those models carry their scaling in Q normalization; OLMo does not.
alias ATTENTION_SCORE_SCALE = 0.08838834764831845
alias HAS_V_NORM = 0

# OLMo2/3 applies post-attention/post-FF norms to branch outputs before the
# residual adds. Attention and MLP consume the raw residual stream.
alias NORM_STYLE_POST = 1
# Checkpoint norm tensors are direct multiplicative weights (not 1+w).
alias NORM_ADD_ONE = 0
alias EMBED_RMSNORM = 0
alias EMBED_SQRT_SCALE = 0
alias MLP_ACT_SILU = 1

alias RMS_EPS_INPUT = 0.000001
alias RMS_EPS_QK = 0.000001
alias RMS_EPS_POST_ATTN = 0.000001
alias RMS_EPS_PRE_FF = 0.000001
alias RMS_EPS_POST_FF = 0.000001
alias RMS_EPS_FINAL = 0.000001

alias TARGET_SOFTCAP = 0.0
alias TARGET_OUTPUT_MULTIPLIER = 1.0
alias TARGET_BOS_ID = 1
alias MAX_PROBE_TOKENS = 128
alias PROBE_TOPK = 32

# Dense-only target. These zero aliases keep the shared hybrid engine contract
# explicit and compile-time dead for this profile.
alias HAS_LINEAR_ATTENTION = 0
alias GDN_CONV_DIM = 0
alias GDN_VALUE_DIM = 0
alias GDN_NUM_V_HEADS = 0
alias GDN_CONV_KERNEL = 0
alias GDN_KEY_HEAD_DIM = 0
alias GDN_VALUE_HEAD_DIM = 0

# No drafter is part of the target-inference port. Nonzero geometry prevents
# dormant debug/spec allocation code from acquiring zero-sized static shapes.
alias DRAFTER_MLP = FF
alias DRAFTER_TAPS = 0
alias DRAFTER_Q_HEADS = NH
alias DRAFTER_KV_HEADS = NKV
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
alias DRAFTER_MASK_TOKEN_ID = 0
alias DRAFTER_ROPE_THETA = 500000.0
alias DRAFTER_RMS_EPS = 0.000001
alias DRAFTER_SOFTCAP = 0.0
alias DRAFTER_SLIDING_WINDOW = 4096
alias DRAFTER_SLIDING_LAYERS = 0
alias DRAFTER_READOUT_V4 = 0
alias DRAFTER_EMBED_SQRT_SCALE = 0
alias DRAFTER_TAP_0 = -1
alias DRAFTER_TAP_1 = -1
alias DRAFTER_TAP_2 = -1
alias DRAFTER_TAP_3 = -1
alias DRAFTER_TAP_4 = -1
alias DRAFTER_TAP_5 = -1
alias EAGLE3_TAPS = 0
alias EAGLE3_TAP_0 = -1
alias EAGLE3_TAP_1 = -1
alias EAGLE3_TAP_2 = -1
