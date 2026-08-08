"""DFlash block drafter for Gemma-4-31B.

The implemented path follows z-lab's Qwen3-style DFlash forward over synthetic
or host-provided tensors: hidden_norm(fc(raw 6-tap stream)), five block layers,
bidirectional attention over [context + block], and final norm. Live target
embed/lm-head drafting is wired after this staged math path gates.
"""

from std.collections import List
from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from std.math import exp, sqrt
from layout import row_major
from layout import stack_allocation

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy, cuda_sync
from lib.io import load_f32_file_to_gpu
from lib.ops_gpu_mojo import gpu_residual_add_mojo, gpu_rope_batched_mojo
from lib.ops_gpu_mojo_reductions import gpu_qk_norm_mojo, gpu_rmsnorm_batched_mojo
from lib.q4_gemv_dp4a import (
    gpu_matmul_q4_dp4a_dev,
    gpu_matmul_q4_mmq_relaxed_verify_dev,
    gpu_matmul_q4_s8_v4_gemv_dev,
)
from lib.q4_weights import load_to_gpu_q4, load_to_gpu_q4_concat2, load_to_gpu_q4_concat3
from lib.q8_weights import load_to_gpu_q8, gpu_matmul_q8_rows_dev
from lib.softmax_gpu_mojo import gpu_softmax_over_heads_mojo
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_dev
from std.ffi import external_call
from std.gpu.host import DeviceContext


comptime DFLASH_HIDDEN = 5376
comptime DFLASH_TAPS = 6
comptime DFLASH_FC_IN = DFLASH_TAPS * DFLASH_HIDDEN
comptime DFLASH_BLOCK = 16
comptime DFLASH_CANDIDATES = 15
comptime DFLASH_LAYERS = 5
comptime DFLASH_Q_HEADS = 64
comptime DFLASH_KV_HEADS = 8
comptime DFLASH_HEAD_DIM = 128
comptime DFLASH_Q_DIM = DFLASH_Q_HEADS * DFLASH_HEAD_DIM
comptime DFLASH_KV_DIM = DFLASH_KV_HEADS * DFLASH_HEAD_DIM
comptime DFLASH_QKV_DIM = DFLASH_Q_DIM + 2 * DFLASH_KV_DIM
comptime DFLASH_MLP = 10752
comptime DFLASH_VOCAB = 262144
comptime DFLASH_MAX_VERIFY_ROWS = 17
comptime DFLASH_MAX_CTX = 8192
comptime DFLASH_CTX_BATCH = DFLASH_MAX_VERIFY_ROWS
comptime DFLASH_MAX_SEQ = DFLASH_MAX_CTX + DFLASH_BLOCK
comptime DFLASH_Q4_SCRATCH_BYTES = 16 * ((DFLASH_FC_IN + 31) // 32) * 36
comptime DFLASH_ARGMAX_TPB = 1024


def _dflash_env_flag(name: String) -> Bool:
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()):
        nb.append(name.as_bytes()[i])
    nb.append(0)
    var raw = external_call["getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]](
        nb.unsafe_ptr()
    )
    if raw == 0:
        return False
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    return Int(p[0]) == ord("1")


def _dflash_env_path(name: String) -> String:
    """Read an env var as a path. Empty String when unset."""
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()):
        nb.append(name.as_bytes()[i])
    nb.append(0)
    var raw = external_call["getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]](
        nb.unsafe_ptr()
    )
    if raw == 0:
        return String("")
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    var out = String("")
    var i = 0
    while p[i] != 0 and i < 4096:
        out += chr(Int(p[i]))
        i += 1
    return out


def _dflash_f32(stem: String) -> UInt64:
    var ptr = load_f32_file_to_gpu(stem + ".bin")
    if ptr == 0:
        ptr = load_f32_file_to_gpu(stem + ".f32bin")
    if ptr == 0:
        print("[WARN] Empty or missing DFlash f32:", stem, ".bin/.f32bin")
    return ptr


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def _threads_for_cols(cols: Int) -> Int:
    var t = 32
    while t < cols:
        t *= 2
    if t > 1024:
        t = 1024
    return t


def _dflash_q4_batched(
    ctx: DeviceContext,
    d_out: UInt64,
    d_in: UInt64,
    d_w: UInt64,
    d_scratch: UInt64,
    K: Int,
    N: Int,
    S: Int,
    force_q4: Bool = False,
) raises:
    if not force_q4 and _dflash_env_flag("NOMOS_DFLASH_DRAFTER_Q8"):
        gpu_matmul_q8_rows_dev(ctx, d_out, d_in, d_w, K, N, S)
        return
    # Drafter-side proposals do not carry the target byte-law constraint. Use
    # the staged MMQ body at rows where the synthetic crossover is favorable,
    # and keep smaller/17-row cases on the v4 route.
    #
    # NOMOS_DFLASH_DRAFTER_V4=1 forces the v4 GEMV route at every row count. Kept as an A/B
    # lever only -- the MMQ routing above is CORRECT and this flag measures WORSE.
    #
    # The hypothesis it was built to test (2026-08-03) was: the drafter body is weight-stationary
    # and memory-bound at S=16 (arithmetic intensity ~57 flop/byte vs a ~416 flop/byte machine
    # balance), so MMQ's tensor-core advantage is unspendable and the bandwidth-cheaper GEMV
    # route should win. MEASURED, GB10 short bar, 12/12 lossless both arms, E identical at 3.10:
    #     MMQ (this route)  draft 10.02 ms/cycle   15.99 tok/s
    #     v4 (flag on)      draft 38.29 ms/cycle   13.90 tok/s   <- 3.8x SLOWER
    #
    # Why the hypothesis was wrong: that AI figure assumes each weight is read once and reused
    # across all 16 rows, and THAT REUSE IS A PROPERTY OF THE KERNEL, NOT THE SHAPE. MMQ stages a
    # weight tile once and serves all 16 rows from it; the GEMV route does not attain that reuse at
    # S=16, so its effective AI is far lower and it moves much more traffic. Ideal AI is an upper
    # bound on an implementation, not a description of one. (The 166-vs-208 GB/s figures used to
    # rank the routes were also measured at the VERIFY shapes, S=5..8, then misapplied here.)
    #
    # Leave this OFF. (a kernel property, not a shape one).
    if S >= 9 and S <= 16 and not _dflash_env_flag("NOMOS_DFLASH_DRAFTER_V4"):
        if K <= 8192 and N > 8192:
            gpu_matmul_q4_mmq_relaxed_verify_dev[8](
                ctx, d_out, d_in, d_w, d_scratch, K, N, S
            )
        elif K > 8192 and N <= 8192:
            gpu_matmul_q4_mmq_relaxed_verify_dev[2](
                ctx, d_out, d_in, d_w, d_scratch, K, N, S
            )
        else:
            gpu_matmul_q4_mmq_relaxed_verify_dev[4](
                ctx, d_out, d_in, d_w, d_scratch, K, N, S
            )
    else:
        gpu_matmul_q4_s8_v4_gemv_dev[4](ctx, d_out, d_in, d_w, d_scratch, K, N, S)


struct DFlashDrafter(Movable):
    var dB: Int
    var n_taps: Int
    var block_size: Int
    var mask_token_id: Int
    var num_layers: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var mlp_dim: Int
    var sliding_window: Int
    var rope_theta: Float32
    var rms_eps: Float32
    var softcap: Float32
    var max_ctx: Int
    var seq_len: Int
    var use_q8: Bool

    # Converted weights. Q/K/V and gate/up are loaded concatenated along output N.
    var w_fc: UInt64
    var w_hidden_norm: UInt64
    var w_norm: UInt64
    var w_qkv: List[UInt64]
    var w_o: List[UInt64]
    var w_gup: List[UInt64]
    var w_down: List[UInt64]
    var w_in_norm: List[UInt64]
    var w_post_attn_norm: List[UInt64]
    var w_q_norm: List[UInt64]
    var w_k_norm: List[UInt64]

    # Target-tap buffers wired through GemmaEngine.configure_drafter_taps.
    var d_taps_buf: UInt64
    var d_verify_taps_buf: UInt64
    var last_verify_rows: Int

    # Gold-diff and forward scratch. Block-forward use waits for goldens.
    var d_context: UInt64
    var d_target_hidden: UInt64
    var d_fc_out: UInt64
    var d_ctx_fused: UInt64
    var d_ctx_qkv: UInt64
    var d_k_ctx: UInt64
    var d_v_ctx: UInt64
    var d_ctx_k_cache: List[UInt64]
    var d_ctx_v_cache: List[UInt64]
    var d_k_all: UInt64
    var d_v_all: UInt64
    var d_block_h: UInt64
    var d_residual: UInt64
    var d_normed: UInt64
    var d_qkv: UInt64
    var d_q: UInt64
    var d_k: UInt64
    var d_v: UInt64
    var d_attn: UInt64
    var d_o: UInt64
    var d_gate_up: UInt64
    var d_mlp: UInt64
    var d_layer_out: UInt64
    var d_final_norm: UInt64
    var d_logits: UInt64
    # Optional draft-vocab restriction. 0 = disabled (full 262144 argmax, bit-identical
    # to pre-change behaviour). Non-zero = f32[VOCAB] keep-mask; the drafter may only
    # PROPOSE tokens with mask!=0. Lossless by construction on the verify side: the
    # target still scores the full vocabulary, so a restricted proposal set can only
    # cost acceptance, never correctness.
    var d_vocab_mask: UInt64
    var d_decode_ids: UInt64
    var d_q4_scratch: UInt64
    var d_scores: UInt64

    def __init__(out self, dir: String):
        self.dB = DFLASH_HIDDEN
        self.n_taps = DFLASH_TAPS
        self.block_size = DFLASH_BLOCK
        self.mask_token_id = 4
        self.num_layers = DFLASH_LAYERS
        self.num_q_heads = DFLASH_Q_HEADS
        self.num_kv_heads = DFLASH_KV_HEADS
        self.head_dim = DFLASH_HEAD_DIM
        self.mlp_dim = DFLASH_MLP
        self.sliding_window = 2048
        self.rope_theta = Float32(1000000.0)
        self.rms_eps = Float32(0.000001)
        self.softcap = Float32(30.0)
        self.max_ctx = DFLASH_MAX_CTX
        self.seq_len = 0

        self.use_q8 = _dflash_env_flag("NOMOS_DFLASH_DRAFTER_Q8")
        if self.use_q8:
            print("[dflash] drafter weights: Q8")
            self.w_fc = load_to_gpu_q8(dir + "fc_weight.q8")
        else:
            print("[dflash] drafter weights: Q4")
            self.w_fc = load_to_gpu_q4(dir + "fc_weight.q4")
        self.w_hidden_norm = _dflash_f32(dir + "hidden_norm_weight")
        self.w_norm = _dflash_f32(dir + "norm_weight")

        self.w_qkv = List[UInt64]()
        self.w_o = List[UInt64]()
        self.w_gup = List[UInt64]()
        self.w_down = List[UInt64]()
        self.w_in_norm = List[UInt64]()
        self.w_post_attn_norm = List[UInt64]()
        self.w_q_norm = List[UInt64]()
        self.w_k_norm = List[UInt64]()

        for layer in range(DFLASH_LAYERS):
            var p = dir + "layers_" + String(layer) + "_"
            if self.use_q8:
                self.w_qkv.append(load_to_gpu_q8(p + "self_attn_qkv_weight.q8"))
                self.w_o.append(load_to_gpu_q8(p + "self_attn_o_proj_weight.q8"))
                self.w_gup.append(load_to_gpu_q8(p + "mlp_gate_up_weight.q8"))
                self.w_down.append(load_to_gpu_q8(p + "mlp_down_proj_weight.q8"))
            else:
                self.w_qkv.append(load_to_gpu_q4_concat3(
                    p + "self_attn_q_proj_weight.q4",
                    p + "self_attn_k_proj_weight.q4",
                    p + "self_attn_v_proj_weight.q4",
                ))
                self.w_o.append(load_to_gpu_q4(p + "self_attn_o_proj_weight.q4"))
                self.w_gup.append(load_to_gpu_q4_concat2(
                    p + "mlp_gate_proj_weight.q4",
                    p + "mlp_up_proj_weight.q4",
                ))
                self.w_down.append(load_to_gpu_q4(p + "mlp_down_proj_weight.q4"))
            self.w_in_norm.append(_dflash_f32(p + "input_layernorm_weight"))
            self.w_post_attn_norm.append(_dflash_f32(p + "post_attention_layernorm_weight"))
            self.w_q_norm.append(_dflash_f32(p + "self_attn_q_norm_weight"))
            self.w_k_norm.append(_dflash_f32(p + "self_attn_k_norm_weight"))

        self.d_taps_buf = cuda_malloc(DFLASH_FC_IN * 4)
        self.d_verify_taps_buf = cuda_malloc(DFLASH_MAX_VERIFY_ROWS * DFLASH_FC_IN * 4)
        self.last_verify_rows = 0

        self.d_context = cuda_malloc(DFLASH_HIDDEN * 4)
        self.d_target_hidden = cuda_malloc(DFLASH_CTX_BATCH * DFLASH_FC_IN * 4)
        self.d_fc_out = cuda_malloc(DFLASH_CTX_BATCH * DFLASH_HIDDEN * 4)
        self.d_ctx_fused = cuda_malloc(DFLASH_CTX_BATCH * DFLASH_HIDDEN * 4)
        self.d_ctx_qkv = cuda_malloc(DFLASH_CTX_BATCH * DFLASH_QKV_DIM * 4)
        self.d_k_ctx = cuda_malloc(DFLASH_CTX_BATCH * DFLASH_KV_DIM * 4)
        self.d_v_ctx = cuda_malloc(DFLASH_CTX_BATCH * DFLASH_KV_DIM * 4)
        self.d_ctx_k_cache = List[UInt64]()
        self.d_ctx_v_cache = List[UInt64]()
        for _ in range(DFLASH_LAYERS):
            self.d_ctx_k_cache.append(cuda_malloc(DFLASH_MAX_CTX * DFLASH_KV_DIM * 4))
            self.d_ctx_v_cache.append(cuda_malloc(DFLASH_MAX_CTX * DFLASH_KV_DIM * 4))
        self.d_k_all = cuda_malloc(DFLASH_MAX_SEQ * DFLASH_KV_DIM * 4)
        self.d_v_all = cuda_malloc(DFLASH_MAX_SEQ * DFLASH_KV_DIM * 4)
        self.d_block_h = cuda_malloc(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        self.d_residual = cuda_malloc(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        self.d_normed = cuda_malloc(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        self.d_qkv = cuda_malloc(DFLASH_BLOCK * DFLASH_QKV_DIM * 4)
        self.d_q = cuda_malloc(DFLASH_BLOCK * DFLASH_Q_DIM * 4)
        self.d_k = cuda_malloc(DFLASH_BLOCK * DFLASH_KV_DIM * 4)
        self.d_v = cuda_malloc(DFLASH_BLOCK * DFLASH_KV_DIM * 4)
        self.d_attn = cuda_malloc(DFLASH_BLOCK * DFLASH_Q_DIM * 4)
        self.d_o = cuda_malloc(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        self.d_gate_up = cuda_malloc(DFLASH_BLOCK * 2 * DFLASH_MLP * 4)
        self.d_mlp = cuda_malloc(DFLASH_BLOCK * DFLASH_MLP * 4)
        self.d_layer_out = cuda_malloc(DFLASH_LAYERS * DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        self.d_final_norm = cuda_malloc(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        self.d_logits = cuda_malloc(DFLASH_CANDIDATES * DFLASH_VOCAB * 4)
        # NOMOS_DRAFT_VOCAB_MASK=<path to f32[262144] of 1.0/0.0>. f32 (not u8) so it reuses
        # the existing, tested load_f32_file_to_gpu path rather than adding a byte loader.
        #
        # *** MEASURED NULL ON GB10 (2026-08-05). KEPT so nobody re-runs it hoping. ***
        # Premise: our 262144-wide draft head is 1.41B params = 48% of ALL draft compute (vs 6%
        # for RedHatAI's DSpark at draft_vocab_size=32000). Restricting proposals to a
        # frequency-selected 33,043-token subset (RedHat's learned d2t U repo-corpus top-4096)
        # cuts draft compute ~42%. LOSSLESS BY CONSTRUCTION on the verify side -- the target still
        # scores the full vocabulary -- and that held: 12/12 bit-identical on every arm.
        #
        # 4 interleaved arms, GB10 Q4_0, VB=9, NTOK=48, 12-prompt bar:
        #     none_a  24.49 tok/s   allones 24.82   union33k 24.72   none_b 24.66
        #     machinery gate (allones vs none_a): +1.35%   drift (none_b vs none_a): +0.69%
        #     EFFECT (union33k vs mean baseline): +0.59%  <- INSIDE the drift band
        # Acceptance cost was small and localized: mean E 3.684 -> 3.638 (-1.2%), and only two
        # buckets moved at all (code2 3.36->3.13, mem3 2.88->2.56); the other ten were IDENTICAL.
        # The all-ones control reproduced the unmasked baseline per-bucket to three decimals, so
        # the machinery is correct and the null is about the LEVER, not the implementation.
        #
        # WHY IT CANNOT PAY HERE -- Amdahl, and I should have checked it first: draft is
        # 9.9-10.1 ms against ~129 ms of verify = 7.2% of the cycle. Cutting 42% of 7.2% is ~3%
        # of cycle time, which is under the measured drift floor. #52 has said "verify is ~2x too
        # slow" for weeks; I did the draft-side arithmetic carefully and never checked draft's
        # SHARE. Compute the quantity a change is supposed to move AND confirm it can move it.
        #
        # WHERE IT WOULD PAY: a drafter whose forward is a large share of the cycle. DSpark spends
        # 2.39B on its backbone vs our 1.36B and funds that by shrinking the vocab head -- the
        # 42% is not the prize, it is the BUDGET for a bigger transformer.
        self.d_vocab_mask = 0
        var _mask_path = _dflash_env_path("NOMOS_DRAFT_VOCAB_MASK")
        if _mask_path.byte_length() > 0:
            self.d_vocab_mask = load_f32_file_to_gpu(_mask_path)
            if self.d_vocab_mask == 0:
                print("[dflash] WARNING: NOMOS_DRAFT_VOCAB_MASK set but not loadable:",
                      _mask_path, "-- running UNMASKED (full 262144 vocab)")
            else:
                print("[dflash] draft-vocab mask ACTIVE:", _mask_path,
                      "-- drafter may only propose masked tokens; verify is unaffected")
        self.d_decode_ids = cuda_malloc(DFLASH_CANDIDATES * 4)
        self.d_q4_scratch = cuda_malloc(DFLASH_Q4_SCRATCH_BYTES)
        self.d_scores = cuda_malloc(DFLASH_BLOCK * DFLASH_Q_HEADS * DFLASH_MAX_SEQ * 4)
        cuda_sync()

    def set_cache_len(mut self, n: Int):
        if n < 0:
            self.seq_len = 0
        elif n > self.max_ctx:
            self.seq_len = self.max_ctx
        else:
            self.seq_len = n

    def cache_len(self) -> Int:
        return self.seq_len

    def reset(mut self):
        self.seq_len = 0


def dflash_split_block_qkv_kernel(
    qkv: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = DFLASH_BLOCK * DFLASH_QKV_DIM
    if idx >= total:
        return
    var row = idx // DFLASH_QKV_DIM
    var col = idx - row * DFLASH_QKV_DIM
    if col < DFLASH_Q_DIM:
        q[row * DFLASH_Q_DIM + col] = qkv[idx]
    elif col < DFLASH_Q_DIM + DFLASH_KV_DIM:
        k[row * DFLASH_KV_DIM + (col - DFLASH_Q_DIM)] = qkv[idx]
    else:
        v[row * DFLASH_KV_DIM + (col - DFLASH_Q_DIM - DFLASH_KV_DIM)] = qkv[idx]


def dflash_split_ctx_qkv_kernel(
    qkv: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    ctx_len_arg: Int32,
):
    var ctx_len = Int(ctx_len_arg)
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = ctx_len * DFLASH_KV_DIM
    if idx >= total:
        return
    var row = idx // DFLASH_KV_DIM
    var col = idx - row * DFLASH_KV_DIM
    var qkv_base = row * DFLASH_QKV_DIM
    k[idx] = qkv[qkv_base + DFLASH_Q_DIM + col]
    v[idx] = qkv[qkv_base + DFLASH_Q_DIM + DFLASH_KV_DIM + col]


def dflash_concat_kv_kernel(
    k_ctx: UnsafePointer[Float32, MutAnyOrigin],
    v_ctx: UnsafePointer[Float32, MutAnyOrigin],
    k_blk: UnsafePointer[Float32, MutAnyOrigin],
    v_blk: UnsafePointer[Float32, MutAnyOrigin],
    k_all: UnsafePointer[Float32, MutAnyOrigin],
    v_all: UnsafePointer[Float32, MutAnyOrigin],
    ctx_len_arg: Int32,
):
    var ctx_len = Int(ctx_len_arg)
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = (ctx_len + DFLASH_BLOCK) * DFLASH_KV_DIM
    if idx >= total:
        return
    var row = idx // DFLASH_KV_DIM
    var col = idx - row * DFLASH_KV_DIM
    if row < ctx_len:
        k_all[idx] = k_ctx[row * DFLASH_KV_DIM + col]
        v_all[idx] = v_ctx[row * DFLASH_KV_DIM + col]
    else:
        var brow = row - ctx_len
        k_all[idx] = k_blk[brow * DFLASH_KV_DIM + col]
        v_all[idx] = v_blk[brow * DFLASH_KV_DIM + col]


def dflash_swiglu_kernel(
    gate_up: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = DFLASH_BLOCK * DFLASH_MLP
    if idx >= total:
        return
    var row = idx // DFLASH_MLP
    var col = idx - row * DFLASH_MLP
    var base = row * (2 * DFLASH_MLP)
    var gate = gate_up[base + col]
    var up = gate_up[base + DFLASH_MLP + col]
    outp[idx] = (gate / (1.0 + exp(-gate))) * up


def dflash_qk_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k_all: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    key_len_arg: Int32,
    scale: Float32,
):
    var key_len = Int(key_len_arg)
    var row = block_idx.x
    if row >= DFLASH_BLOCK * DFLASH_Q_HEADS:
        return
    var tid = thread_idx.x
    var n_threads = block_dim.x
    var tok = row // DFLASH_Q_HEADS
    var qh = row - tok * DFLASH_Q_HEADS
    var kvh = qh // (DFLASH_Q_HEADS // DFLASH_KV_HEADS)
    var key = tid
    while key < key_len:
        var acc = Float32(0.0)
        for d in range(DFLASH_HEAD_DIM):
            acc += q[(tok * DFLASH_Q_HEADS + qh) * DFLASH_HEAD_DIM + d] * k_all[(key * DFLASH_KV_HEADS + kvh) * DFLASH_HEAD_DIM + d]
        scores[row * key_len + key] = acc * scale
        key += n_threads


def dflash_pv_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],
    v_all: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    key_len_arg: Int32,
):
    var key_len = Int(key_len_arg)
    var row = block_idx.x
    if row >= DFLASH_BLOCK * DFLASH_Q_HEADS:
        return
    var tid = thread_idx.x
    var n_threads = block_dim.x
    var tok = row // DFLASH_Q_HEADS
    var qh = row - tok * DFLASH_Q_HEADS
    var kvh = qh // (DFLASH_Q_HEADS // DFLASH_KV_HEADS)
    var d = tid
    while d < DFLASH_HEAD_DIM:
        var acc = Float32(0.0)
        for key in range(key_len):
            acc += scores[row * key_len + key] * v_all[(key * DFLASH_KV_HEADS + kvh) * DFLASH_HEAD_DIM + d]
        outp[(tok * DFLASH_Q_HEADS + qh) * DFLASH_HEAD_DIM + d] = acc
        d += n_threads


def dflash_copy_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = src[i]


def dflash_argmax_rows_kernel(
    out_ids: UnsafePointer[Int32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    n_rows_arg: Int32,
    vocab_arg: Int32,
    mask: UnsafePointer[Float32, MutAnyOrigin],
    use_mask_arg: Int32,
):
    var n_rows = Int(n_rows_arg)
    var vocab = Int(vocab_arg)
    var row = block_idx.x
    if row >= n_rows:
        return
    var score_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[DFLASH_ARGMAX_TPB]())
    var idx_s = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[DFLASH_ARGMAX_TPB]())
    var tid = thread_idx.x
    var best = Float32(-3.4e38)
    var best_i = Int32(0)
    var i = tid
    var row_off = row * vocab
    # Two loops rather than a per-element branch: the unmasked path must stay
    # bit-identical AND pay nothing for a feature it is not using.
    if use_mask_arg != Int32(0):
        while i < vocab:
            if mask[i] != Float32(0):
                var v = logits[row_off + i]
                if v > best:
                    best = v
                    best_i = Int32(i)
            i += DFLASH_ARGMAX_TPB
    else:
        while i < vocab:
            var v = logits[row_off + i]
            if v > best:
                best = v
                best_i = Int32(i)
            i += DFLASH_ARGMAX_TPB
    score_s[tid] = best
    idx_s[tid] = best_i
    barrier()
    var stride = DFLASH_ARGMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            if score_s[tid + stride] > score_s[tid]:
                score_s[tid] = score_s[tid + stride]
                idx_s[tid] = idx_s[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_ids[row] = idx_s[0]


def _dflash_copy(ctx: DeviceContext, dst: UInt64, src: UInt64, n: Int) raises:
    var threads = 256
    var k = ctx.compile_function[dflash_copy_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(dst), _as_f32_ptr(src), Int32(n),
        grid_dim=(n + threads - 1) // threads,
        block_dim=threads,
    )


def _dflash_readout_argmax(ctx: DeviceContext, mut m: DFlashDrafter, d_lmhead_q4: UInt64) raises:
    _dflash_q4_batched(
        ctx,
        m.d_logits,
        m.d_final_norm + UInt64(DFLASH_HIDDEN * 4),
        d_lmhead_q4,
        m.d_q4_scratch,
        DFLASH_HIDDEN,
        DFLASH_VOCAB,
        DFLASH_CANDIDATES,
        True,
    )
    var k = ctx.compile_function[dflash_argmax_rows_kernel]()
    ctx.enqueue_function(
        k,
        UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(m.d_decode_ids)),
        _as_f32_ptr(m.d_logits),
        Int32(DFLASH_CANDIDATES),
        Int32(DFLASH_VOCAB),
        _as_f32_ptr(m.d_vocab_mask),
        Int32(1) if m.d_vocab_mask != 0 else Int32(0),
        grid_dim=DFLASH_CANDIDATES,
        block_dim=DFLASH_ARGMAX_TPB,
    )


def dflash_project_context(ctx: DeviceContext, mut m: DFlashDrafter, d_taps: UInt64) raises:
    """Project one six-tap context row through hidden_norm(fc(raw_concat))."""
    if m.use_q8:
        gpu_matmul_q8_rows_dev(
            ctx, m.d_fc_out, d_taps, m.w_fc, DFLASH_FC_IN, DFLASH_HIDDEN, 1
        )
    else:
        gpu_matmul_q4_dp4a_dev[4](
            ctx,
            m.d_fc_out,
            d_taps,
            m.w_fc,
            m.d_q4_scratch,
            DFLASH_FC_IN,
            DFLASH_HIDDEN,
        )
    gpu_rmsnorm_batched_mojo(ctx, m.d_fc_out, m.d_context, m.w_hidden_norm, DFLASH_HIDDEN, 1)


def dflash_project_context_stream(ctx: DeviceContext, mut m: DFlashDrafter, ctx_len: Int) raises:
    if ctx_len <= 0 or ctx_len > DFLASH_CTX_BATCH:
        return
    _dflash_q4_batched(
        ctx,
        m.d_fc_out,
        m.d_target_hidden,
        m.w_fc,
        m.d_q4_scratch,
        DFLASH_FC_IN,
        DFLASH_HIDDEN,
        ctx_len,
    )
    gpu_rmsnorm_batched_mojo(ctx, m.d_fc_out, m.d_ctx_fused, m.w_hidden_norm, DFLASH_HIDDEN, ctx_len)


def dflash_append_context_rows(
    ctx: DeviceContext, mut m: DFlashDrafter, d_taps_rows: UInt64, n_rows: Int
) raises:
    if n_rows <= 0 or n_rows > DFLASH_CTX_BATCH:
        return
    if m.seq_len + n_rows > m.max_ctx:
        return
    _dflash_q4_batched(
        ctx,
        m.d_fc_out,
        d_taps_rows,
        m.w_fc,
        m.d_q4_scratch,
        DFLASH_FC_IN,
        DFLASH_HIDDEN,
        n_rows,
    )
    gpu_rmsnorm_batched_mojo(
        ctx, m.d_fc_out, m.d_ctx_fused, m.w_hidden_norm, DFLASH_HIDDEN, n_rows
    )
    var base_pos = m.seq_len
    for layer in range(DFLASH_LAYERS):
        _dflash_q4_batched(
            ctx,
            m.d_ctx_qkv,
            m.d_ctx_fused,
            m.w_qkv[layer],
            m.d_q4_scratch,
            DFLASH_HIDDEN,
            DFLASH_QKV_DIM,
            n_rows,
        )
        _dflash_split_ctx_qkv(ctx, m, n_rows)
        gpu_qk_norm_mojo(
            ctx, m.d_k_ctx, m.w_k_norm[layer], n_rows * DFLASH_KV_HEADS, DFLASH_HEAD_DIM
        )
        gpu_rope_batched_mojo(
            ctx, m.d_k_ctx, base_pos, n_rows, DFLASH_KV_HEADS, DFLASH_HEAD_DIM,
            m.rope_theta, DFLASH_HEAD_DIM,
        )
        cuda_memcpy(
            m.d_ctx_k_cache[layer] + UInt64(base_pos * DFLASH_KV_DIM * 4),
            m.d_k_ctx,
            n_rows * DFLASH_KV_DIM * 4,
            3,
        )
        cuda_memcpy(
            m.d_ctx_v_cache[layer] + UInt64(base_pos * DFLASH_KV_DIM * 4),
            m.d_v_ctx,
            n_rows * DFLASH_KV_DIM * 4,
            3,
        )
    m.seq_len += n_rows


def _dflash_split_block_qkv(ctx: DeviceContext, mut m: DFlashDrafter) raises:
    var threads = 256
    var total = DFLASH_BLOCK * DFLASH_QKV_DIM
    var k = ctx.compile_function[dflash_split_block_qkv_kernel]()
    ctx.enqueue_function(
        k,
        _as_f32_ptr(m.d_qkv),
        _as_f32_ptr(m.d_q),
        _as_f32_ptr(m.d_k),
        _as_f32_ptr(m.d_v),
        grid_dim=(total + threads - 1) // threads,
        block_dim=threads,
    )


def _dflash_split_ctx_qkv(ctx: DeviceContext, mut m: DFlashDrafter, ctx_len: Int) raises:
    if ctx_len <= 0:
        return
    var threads = 256
    var total = ctx_len * DFLASH_KV_DIM
    var k = ctx.compile_function[dflash_split_ctx_qkv_kernel]()
    ctx.enqueue_function(
        k,
        _as_f32_ptr(m.d_ctx_qkv),
        _as_f32_ptr(m.d_k_ctx),
        _as_f32_ptr(m.d_v_ctx),
        Int32(ctx_len),
        grid_dim=(total + threads - 1) // threads,
        block_dim=threads,
    )


def _dflash_concat_kv(ctx: DeviceContext, mut m: DFlashDrafter, ctx_len: Int) raises:
    var threads = 256
    var total = (ctx_len + DFLASH_BLOCK) * DFLASH_KV_DIM
    var k = ctx.compile_function[dflash_concat_kv_kernel]()
    ctx.enqueue_function(
        k,
        _as_f32_ptr(m.d_k_ctx),
        _as_f32_ptr(m.d_v_ctx),
        _as_f32_ptr(m.d_k),
        _as_f32_ptr(m.d_v),
        _as_f32_ptr(m.d_k_all),
        _as_f32_ptr(m.d_v_all),
        Int32(ctx_len),
        grid_dim=(total + threads - 1) // threads,
        block_dim=threads,
    )


def _dflash_swiglu(ctx: DeviceContext, mut m: DFlashDrafter) raises:
    var threads = 256
    var total = DFLASH_BLOCK * DFLASH_MLP
    var k = ctx.compile_function[dflash_swiglu_kernel]()
    ctx.enqueue_function(
        k,
        _as_f32_ptr(m.d_gate_up),
        _as_f32_ptr(m.d_mlp),
        grid_dim=(total + threads - 1) // threads,
        block_dim=threads,
    )


def _dflash_attention(
    ctx: DeviceContext, mut m: DFlashDrafter, key_start: Int, key_len: Int
) raises:
    var rows = DFLASH_BLOCK * DFLASH_Q_HEADS
    var scale = Float32(1.0) / sqrt(Float32(DFLASH_HEAD_DIM))
    var d_k_read = m.d_k_all + UInt64(key_start * DFLASH_KV_DIM * 4)
    var d_v_read = m.d_v_all + UInt64(key_start * DFLASH_KV_DIM * 4)
    var qk = ctx.compile_function[dflash_qk_kernel]()
    ctx.enqueue_function(
        qk,
        _as_f32_ptr(m.d_q),
        _as_f32_ptr(d_k_read),
        _as_f32_ptr(m.d_scores),
        Int32(key_len),
        scale,
        grid_dim=rows,
        block_dim=_threads_for_cols(key_len),
    )
    gpu_softmax_over_heads_mojo(ctx, m.d_scores, rows, key_len)
    var pv = ctx.compile_function[dflash_pv_kernel]()
    ctx.enqueue_function(
        pv,
        _as_f32_ptr(m.d_scores),
        _as_f32_ptr(d_v_read),
        _as_f32_ptr(m.d_attn),
        Int32(key_len),
        grid_dim=rows,
        block_dim=_threads_for_cols(DFLASH_HEAD_DIM),
    )


def dflash_forward_block_embeddings(ctx: DeviceContext, mut m: DFlashDrafter, ctx_len: Int) raises:
    if ctx_len < 0 or ctx_len > DFLASH_CTX_BATCH:
        return
    dflash_project_context_stream(ctx, m, ctx_len)
    for layer in range(DFLASH_LAYERS):
        _dflash_copy(ctx, m.d_residual, m.d_block_h, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_rmsnorm_batched_mojo(
            ctx, m.d_block_h, m.d_normed, m.w_in_norm[layer], DFLASH_HIDDEN, DFLASH_BLOCK
        )

        _dflash_q4_batched(
            ctx,
            m.d_qkv,
            m.d_normed,
            m.w_qkv[layer],
            m.d_q4_scratch,
            DFLASH_HIDDEN,
            DFLASH_QKV_DIM,
            DFLASH_BLOCK,
        )
        if ctx_len > 0:
            _dflash_q4_batched(
                ctx,
                m.d_ctx_qkv,
                m.d_ctx_fused,
                m.w_qkv[layer],
                m.d_q4_scratch,
                DFLASH_HIDDEN,
                DFLASH_QKV_DIM,
                ctx_len,
            )
        _dflash_split_block_qkv(ctx, m)
        _dflash_split_ctx_qkv(ctx, m, ctx_len)
        _dflash_concat_kv(ctx, m, ctx_len)

        gpu_qk_norm_mojo(ctx, m.d_q, m.w_q_norm[layer], DFLASH_BLOCK * DFLASH_Q_HEADS, DFLASH_HEAD_DIM)
        gpu_qk_norm_mojo(ctx, m.d_k_all, m.w_k_norm[layer], (ctx_len + DFLASH_BLOCK) * DFLASH_KV_HEADS, DFLASH_HEAD_DIM)
        gpu_rope_batched_mojo(
            ctx, m.d_q, ctx_len, DFLASH_BLOCK, DFLASH_Q_HEADS, DFLASH_HEAD_DIM, m.rope_theta, DFLASH_HEAD_DIM
        )
        gpu_rope_batched_mojo(
            ctx, m.d_k_all, 0, ctx_len + DFLASH_BLOCK, DFLASH_KV_HEADS, DFLASH_HEAD_DIM, m.rope_theta, DFLASH_HEAD_DIM
        )

        var key_start = 0
        var key_len = ctx_len + DFLASH_BLOCK
        if layer < 4 and key_len > m.sliding_window:
            key_start = key_len - m.sliding_window
            key_len = m.sliding_window
        _dflash_attention(ctx, m, key_start, key_len)
        _dflash_q4_batched(
            ctx,
            m.d_o,
            m.d_attn,
            m.w_o[layer],
            m.d_q4_scratch,
            DFLASH_Q_DIM,
            DFLASH_HIDDEN,
            DFLASH_BLOCK,
        )
        _dflash_copy(ctx, m.d_block_h, m.d_residual, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_residual_add_mojo(ctx, m.d_block_h, m.d_o, DFLASH_BLOCK * DFLASH_HIDDEN)

        _dflash_copy(ctx, m.d_residual, m.d_block_h, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_rmsnorm_batched_mojo(
            ctx, m.d_block_h, m.d_normed, m.w_post_attn_norm[layer], DFLASH_HIDDEN, DFLASH_BLOCK
        )
        _dflash_q4_batched(
            ctx,
            m.d_gate_up,
            m.d_normed,
            m.w_gup[layer],
            m.d_q4_scratch,
            DFLASH_HIDDEN,
            2 * DFLASH_MLP,
            DFLASH_BLOCK,
        )
        _dflash_swiglu(ctx, m)
        _dflash_q4_batched(
            ctx,
            m.d_o,
            m.d_mlp,
            m.w_down[layer],
            m.d_q4_scratch,
            DFLASH_MLP,
            DFLASH_HIDDEN,
            DFLASH_BLOCK,
        )
        _dflash_copy(ctx, m.d_block_h, m.d_residual, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_residual_add_mojo(ctx, m.d_block_h, m.d_o, DFLASH_BLOCK * DFLASH_HIDDEN)
        _dflash_copy(
            ctx,
            m.d_layer_out + UInt64(layer * DFLASH_BLOCK * DFLASH_HIDDEN * 4),
            m.d_block_h,
            DFLASH_BLOCK * DFLASH_HIDDEN,
        )

    gpu_rmsnorm_batched_mojo(ctx, m.d_block_h, m.d_final_norm, m.w_norm, DFLASH_HIDDEN, DFLASH_BLOCK)


def dflash_forward_block_cached(ctx: DeviceContext, mut m: DFlashDrafter, abs_start_pos: Int) raises:
    var ctx_len = m.seq_len
    if ctx_len < 0 or ctx_len > m.max_ctx:
        return
    for layer in range(DFLASH_LAYERS):
        _dflash_copy(ctx, m.d_residual, m.d_block_h, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_rmsnorm_batched_mojo(
            ctx, m.d_block_h, m.d_normed, m.w_in_norm[layer], DFLASH_HIDDEN, DFLASH_BLOCK
        )

        _dflash_q4_batched(
            ctx,
            m.d_qkv,
            m.d_normed,
            m.w_qkv[layer],
            m.d_q4_scratch,
            DFLASH_HIDDEN,
            DFLASH_QKV_DIM,
            DFLASH_BLOCK,
        )
        _dflash_split_block_qkv(ctx, m)

        gpu_qk_norm_mojo(ctx, m.d_q, m.w_q_norm[layer], DFLASH_BLOCK * DFLASH_Q_HEADS, DFLASH_HEAD_DIM)
        gpu_qk_norm_mojo(ctx, m.d_k, m.w_k_norm[layer], DFLASH_BLOCK * DFLASH_KV_HEADS, DFLASH_HEAD_DIM)
        gpu_rope_batched_mojo(
            ctx, m.d_q, abs_start_pos, DFLASH_BLOCK, DFLASH_Q_HEADS, DFLASH_HEAD_DIM,
            m.rope_theta, DFLASH_HEAD_DIM,
        )
        gpu_rope_batched_mojo(
            ctx, m.d_k, abs_start_pos, DFLASH_BLOCK, DFLASH_KV_HEADS, DFLASH_HEAD_DIM,
            m.rope_theta, DFLASH_HEAD_DIM,
        )

        if ctx_len > 0:
            cuda_memcpy(
                m.d_k_all,
                m.d_ctx_k_cache[layer],
                ctx_len * DFLASH_KV_DIM * 4,
                3,
            )
            cuda_memcpy(
                m.d_v_all,
                m.d_ctx_v_cache[layer],
                ctx_len * DFLASH_KV_DIM * 4,
                3,
            )
        cuda_memcpy(
            m.d_k_all + UInt64(ctx_len * DFLASH_KV_DIM * 4),
            m.d_k,
            DFLASH_BLOCK * DFLASH_KV_DIM * 4,
            3,
        )
        cuda_memcpy(
            m.d_v_all + UInt64(ctx_len * DFLASH_KV_DIM * 4),
            m.d_v,
            DFLASH_BLOCK * DFLASH_KV_DIM * 4,
            3,
        )

        var key_start = 0
        var key_len = ctx_len + DFLASH_BLOCK
        if layer < 4 and key_len > m.sliding_window:
            key_start = key_len - m.sliding_window
            key_len = m.sliding_window
        _dflash_attention(ctx, m, key_start, key_len)
        _dflash_q4_batched(
            ctx,
            m.d_o,
            m.d_attn,
            m.w_o[layer],
            m.d_q4_scratch,
            DFLASH_Q_DIM,
            DFLASH_HIDDEN,
            DFLASH_BLOCK,
        )
        _dflash_copy(ctx, m.d_block_h, m.d_residual, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_residual_add_mojo(ctx, m.d_block_h, m.d_o, DFLASH_BLOCK * DFLASH_HIDDEN)

        _dflash_copy(ctx, m.d_residual, m.d_block_h, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_rmsnorm_batched_mojo(
            ctx, m.d_block_h, m.d_normed, m.w_post_attn_norm[layer], DFLASH_HIDDEN, DFLASH_BLOCK
        )
        _dflash_q4_batched(
            ctx,
            m.d_gate_up,
            m.d_normed,
            m.w_gup[layer],
            m.d_q4_scratch,
            DFLASH_HIDDEN,
            2 * DFLASH_MLP,
            DFLASH_BLOCK,
        )
        _dflash_swiglu(ctx, m)
        _dflash_q4_batched(
            ctx,
            m.d_o,
            m.d_mlp,
            m.w_down[layer],
            m.d_q4_scratch,
            DFLASH_MLP,
            DFLASH_HIDDEN,
            DFLASH_BLOCK,
        )
        _dflash_copy(ctx, m.d_block_h, m.d_residual, DFLASH_BLOCK * DFLASH_HIDDEN)
        gpu_residual_add_mojo(ctx, m.d_block_h, m.d_o, DFLASH_BLOCK * DFLASH_HIDDEN)

    gpu_rmsnorm_batched_mojo(ctx, m.d_block_h, m.d_final_norm, m.w_norm, DFLASH_HIDDEN, DFLASH_BLOCK)


def dflash_draft_block(
    ctx: DeviceContext, mut m: DFlashDrafter, d_lmhead_q4: UInt64, abs_start_pos: Int
) raises:
    dflash_forward_block_cached(ctx, m, abs_start_pos)
    _dflash_readout_argmax(ctx, m, d_lmhead_q4)


def _dflash_readout_argmax_nvfp4(
    ctx: DeviceContext, mut m: DFlashDrafter,
    handle: UInt64, d_embed_nvfp4: UInt64, embed_global: Float32,
    d_w4a4_packed: UInt64, d_w4a4_bs: UInt64, d_w4a4_global: UInt64,
    d_w4a4_bs_sf: UInt64, d_w4a4_wbs_sf: UInt64, d_lmhead_cpad: UInt64,
) raises:
    """Read out the 15 block-draft rows through the TARGET's NVFP4 W4A4 lm-head.

    Discrete-VRAM re-wire (Gold RTX PRO 4000, sm_120): the GB10 build read out
    through the Q4 dp4a lm-head (d_embed_q4), which is not populated on the NVFP4
    weight build (d_embed_q4 == 0). This routes the readout through the SAME
    d_embed_nvfp4 W4A4 batched lm-head base decode / eagle3-verify use (the M-row
    fused MMA is row-independent -> per-row argmax is bit-identical), so DFlash
    drafts on the discrete card without loading the Q4 lm-head or the dp4a path.
    """
    gpu_matmul_nvfp4_w4a4_dev(
        ctx, handle,
        m.d_logits,
        m.d_final_norm + UInt64(DFLASH_HIDDEN * 4),
        d_embed_nvfp4, embed_global, Float32(0.0),
        d_w4a4_packed, d_w4a4_bs, d_w4a4_global,
        d_w4a4_bs_sf, d_w4a4_wbs_sf, d_lmhead_cpad,
        DFLASH_CANDIDATES, 16, DFLASH_HIDDEN, DFLASH_VOCAB,
    )
    var k = ctx.compile_function[dflash_argmax_rows_kernel]()
    ctx.enqueue_function(
        k,
        UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(m.d_decode_ids)),
        _as_f32_ptr(m.d_logits),
        Int32(DFLASH_CANDIDATES),
        Int32(DFLASH_VOCAB),
        _as_f32_ptr(m.d_vocab_mask),
        Int32(1) if m.d_vocab_mask != 0 else Int32(0),
        grid_dim=DFLASH_CANDIDATES,
        block_dim=DFLASH_ARGMAX_TPB,
    )


def dflash_draft_block_nvfp4(
    ctx: DeviceContext, mut m: DFlashDrafter,
    handle: UInt64, d_embed_nvfp4: UInt64, embed_global: Float32,
    d_w4a4_packed: UInt64, d_w4a4_bs: UInt64, d_w4a4_global: UInt64,
    d_w4a4_bs_sf: UInt64, d_w4a4_wbs_sf: UInt64, d_lmhead_cpad: UInt64,
    abs_start_pos: Int,
) raises:
    """NVFP4/discrete-card twin of dflash_draft_block: same block forward, NVFP4
    lm-head readout instead of the Q4 dp4a one."""
    dflash_forward_block_cached(ctx, m, abs_start_pos)
    _dflash_readout_argmax_nvfp4(
        ctx, m, handle, d_embed_nvfp4, embed_global,
        d_w4a4_packed, d_w4a4_bs, d_w4a4_global,
        d_w4a4_bs_sf, d_w4a4_wbs_sf, d_lmhead_cpad,
    )


def release_dflash_drafter_handle(handle: UInt64):
    if handle == 0:
        return
    var ptr = UnsafePointer[DFlashDrafter, MutExternalOrigin](
        unsafe_from_address=Int(handle)
    )
    if ptr[0].w_fc != 0: cuda_free(ptr[0].w_fc)
    if ptr[0].w_hidden_norm != 0: cuda_free(ptr[0].w_hidden_norm)
    if ptr[0].w_norm != 0: cuda_free(ptr[0].w_norm)
    for layer in range(DFLASH_LAYERS):
        if ptr[0].w_qkv[layer] != 0: cuda_free(ptr[0].w_qkv[layer])
        if ptr[0].w_o[layer] != 0: cuda_free(ptr[0].w_o[layer])
        if ptr[0].w_gup[layer] != 0: cuda_free(ptr[0].w_gup[layer])
        if ptr[0].w_down[layer] != 0: cuda_free(ptr[0].w_down[layer])
        if ptr[0].w_in_norm[layer] != 0: cuda_free(ptr[0].w_in_norm[layer])
        if ptr[0].w_post_attn_norm[layer] != 0: cuda_free(ptr[0].w_post_attn_norm[layer])
        if ptr[0].w_q_norm[layer] != 0: cuda_free(ptr[0].w_q_norm[layer])
        if ptr[0].w_k_norm[layer] != 0: cuda_free(ptr[0].w_k_norm[layer])
    if ptr[0].d_taps_buf != 0: cuda_free(ptr[0].d_taps_buf)
    if ptr[0].d_verify_taps_buf != 0: cuda_free(ptr[0].d_verify_taps_buf)
    if ptr[0].d_context != 0: cuda_free(ptr[0].d_context)
    if ptr[0].d_target_hidden != 0: cuda_free(ptr[0].d_target_hidden)
    if ptr[0].d_fc_out != 0: cuda_free(ptr[0].d_fc_out)
    if ptr[0].d_ctx_fused != 0: cuda_free(ptr[0].d_ctx_fused)
    if ptr[0].d_ctx_qkv != 0: cuda_free(ptr[0].d_ctx_qkv)
    if ptr[0].d_k_ctx != 0: cuda_free(ptr[0].d_k_ctx)
    if ptr[0].d_v_ctx != 0: cuda_free(ptr[0].d_v_ctx)
    for layer in range(DFLASH_LAYERS):
        if ptr[0].d_ctx_k_cache[layer] != 0: cuda_free(ptr[0].d_ctx_k_cache[layer])
        if ptr[0].d_ctx_v_cache[layer] != 0: cuda_free(ptr[0].d_ctx_v_cache[layer])
    if ptr[0].d_k_all != 0: cuda_free(ptr[0].d_k_all)
    if ptr[0].d_v_all != 0: cuda_free(ptr[0].d_v_all)
    if ptr[0].d_block_h != 0: cuda_free(ptr[0].d_block_h)
    if ptr[0].d_residual != 0: cuda_free(ptr[0].d_residual)
    if ptr[0].d_normed != 0: cuda_free(ptr[0].d_normed)
    if ptr[0].d_qkv != 0: cuda_free(ptr[0].d_qkv)
    if ptr[0].d_q != 0: cuda_free(ptr[0].d_q)
    if ptr[0].d_k != 0: cuda_free(ptr[0].d_k)
    if ptr[0].d_v != 0: cuda_free(ptr[0].d_v)
    if ptr[0].d_attn != 0: cuda_free(ptr[0].d_attn)
    if ptr[0].d_o != 0: cuda_free(ptr[0].d_o)
    if ptr[0].d_gate_up != 0: cuda_free(ptr[0].d_gate_up)
    if ptr[0].d_mlp != 0: cuda_free(ptr[0].d_mlp)
    if ptr[0].d_layer_out != 0: cuda_free(ptr[0].d_layer_out)
    if ptr[0].d_final_norm != 0: cuda_free(ptr[0].d_final_norm)
    if ptr[0].d_logits != 0: cuda_free(ptr[0].d_logits)
    if ptr[0].d_vocab_mask != 0: cuda_free(ptr[0].d_vocab_mask)
    if ptr[0].d_decode_ids != 0: cuda_free(ptr[0].d_decode_ids)
    if ptr[0].d_q4_scratch != 0: cuda_free(ptr[0].d_q4_scratch)
    if ptr[0].d_scores != 0: cuda_free(ptr[0].d_scores)
    cuda_sync()
    ptr.destroy_pointee()
    ptr.free()
