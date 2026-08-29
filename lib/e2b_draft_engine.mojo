"""Gemma-4-E2B draft-model engine for Unit F.

Dedicated small LM draft tower.  This intentionally does not parameterize or
mutate the production 31B ``GemmaEngine``: target verify stays on the green
31B path, while this engine owns its weights, KV cache, and rollback state.
"""

from std.collections import List
from std.gpu.primitives import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import sqrt, tanh
from std.memory import UnsafePointer, alloc

from lib.attention_gpu import append_kv_gpu_dev_async, attention_gpu_fp32_dev
from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy, cuda_sync
from lib.cublas import cublas_create, cublas_set_stream
from lib.decode_argmax import (
    device_decode_token_into_stream,
    device_decode_token_conf_into,
)
from lib.engine_init import _read_env_bytes
from lib.io import (
    file_size_bytes,
    load_bf16_file_to_gpu,
    load_f32_file_to_gpu,
    read_f32,
)
from lib.ops_gpu_mojo import (
    gpu_embed_load_bf16_mojo,
    gpu_embed_load_mojo,
    gpu_gelu_mul_mojo,
    gpu_residual_add_scaled_mojo,
    gpu_scalar_mul_mojo,
    gpu_rope_mojo,
)
from lib.ops_gpu_mojo_reductions import (
    gpu_qk_norm_mojo,
    gpu_rmsnorm_batched_mojo,
    gpu_rmsnorm_mojo,
    gpu_rmsnorm_no_weight_mojo,
    gpu_rmsnorm_residual_add_scaled_mojo,
)
from lib.q4_gemv_dp4a import (
    gpu_matmul_q4_dp4a_dev,
    gpu_matmul_q4_dp4a_gemv_dev,
    gpu_q8_quantize_dev,
)
from lib.q4_weights import (
    load_to_gpu_q4,
    load_to_gpu_q4_concat2,
    load_to_gpu_q4_concat3,
)
from lib.spec_draft_e2b import (
    E2B_HEAD_DIM,
    E2B_HIDDEN,
    E2B_GLOBAL_HEAD_DIM,
    E2B_INTERMEDIATE,
    E2B_MAX_HEAD_DIM,
    E2B_NUM_HEADS,
    E2B_NUM_KV_HEADS,
    E2B_NUM_KV_SHARED_LAYERS,
    E2B_NUM_LAYERS,
    E2B_SLIDING_WINDOW,
    E2B_VOCAB,
    E2B_VOCAB_SIZE_PER_LAYER_INPUT,
    e2b_layer_intermediate_size,
)


comptime D2D = 3
comptime E2B_PLE_DIM = 256
comptime E2B_PLE_PACKED = E2B_NUM_LAYERS * E2B_PLE_DIM
comptime E2B_FINAL_LOGIT_SOFTCAP = Float32(30.0)


def _e2b_layer_is_full(layer: Int) -> Bool:
    """Gemma-4-E2B layer_types: four sliding layers, then one full layer."""
    return ((layer + 1) % 5 == 0) or (layer == E2B_NUM_LAYERS - 1)


def _e2b_layer_head_dim(layer: Int) -> Int:
    if _e2b_layer_is_full(layer):
        return E2B_GLOBAL_HEAD_DIM
    return E2B_HEAD_DIM


def _e2b_is_shared_layer(layer: Int) -> Bool:
    return layer >= E2B_NUM_LAYERS - E2B_NUM_KV_SHARED_LAYERS


def _e2b_shared_source_layer(layer: Int) -> Int:
    """Last non-shared layer of the same attention type."""
    if _e2b_layer_is_full(layer):
        return 14
    return 13


def _host_write_i32_4(ptr: UInt64, a: Int, b: Int, c: Int, d: Int):
    if ptr == 0:
        return
    var out = UnsafePointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=Int(ptr)
    )
    out[0] = Int32(a)
    out[1] = Int32(b)
    out[2] = Int32(c)
    out[3] = Int32(d)


def _host_write_i32_8(
    ptr: UInt64,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    e: Int,
    f: Int,
    g: Int,
    h: Int,
):
    if ptr == 0:
        return
    var out = UnsafePointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=Int(ptr)
    )
    out[0] = Int32(a)
    out[1] = Int32(b)
    out[2] = Int32(c)
    out[3] = Int32(d)
    out[4] = Int32(e)
    out[5] = Int32(f)
    out[6] = Int32(g)
    out[7] = Int32(h)


def _load_e2b_f32(stem: String) -> UInt64:
    """Load either the legacy .bin fp32 form or the .f32bin form."""
    var ptr = load_f32_file_to_gpu(stem + ".bin")
    if ptr == 0:
        ptr = load_f32_file_to_gpu(stem + ".f32bin")
    if ptr == 0:
        print("[WARN] Empty or missing:", stem, ".bin/.f32bin")
        return UInt64(0)
    return ptr


def _try_load_e2b_bf16(stem: String, elems: Int) -> UInt64:
    var ptr = load_bf16_file_to_gpu(stem + ".bf16bin")
    if ptr == 0:
        ptr = load_bf16_file_to_gpu(stem + ".bf16")
    if ptr == 0 and file_size_bytes(stem + ".bin") == elems * 2:
        ptr = load_bf16_file_to_gpu(stem + ".bin")
    return ptr


def _e2b_bf16_gather_enabled() -> Bool:
    var b = _read_env_bytes("NOMOS_E2B_BF16_GATHER")
    return not (len(b) == 1 and b[0] == UInt8(ord("0")))


def _e2b_fused_gemv_enabled() -> Bool:
    var b = _read_env_bytes("NOMOS_E2B_FUSED_GEMV")
    return not (len(b) == 1 and b[0] == UInt8(ord("0")))


def _read_e2b_f32(stem: String) -> List[Float32]:
    var data = read_f32(stem + ".bin")
    if len(data) == 0:
        data = read_f32(stem + ".f32bin")
    return data^


def _e2b_as_f32_ptr(p: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(p))


def _softcap_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
    cap: Float32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        x[i] = tanh(x[i] / cap) * cap


def _gpu_softcap(ctx: DeviceContext, d_x: UInt64, n: Int, cap: Float32) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[_softcap_kernel]()
    ctx.enqueue_function(
        k,
        _e2b_as_f32_ptr(d_x),
        Int32(n),
        cap,
        grid_dim=blocks,
        block_dim=threads,
    )


struct E2BDraftEngine(Movable):
    var ctx: DeviceContext
    var handle: UInt64
    var max_seq: Int
    var seq_len: Int

    # Embeddings / output.
    var d_embed: UInt64
    var d_embed_q4: UInt64
    var d_embed_per_layer: UInt64
    var embed_bf16: Bool
    var embed_per_layer_bf16: Bool
    var d_norm: UInt64
    var d_per_layer_model_projection: UInt64
    var d_per_layer_projection_norm: UInt64

    # Per-layer weights.
    var d_qw: List[UInt64]
    var d_kw: List[UInt64]
    var d_vw: List[UInt64]
    var d_ow: List[UInt64]
    var d_gw: List[UInt64]
    var d_uw: List[UInt64]
    var d_dw: List[UInt64]
    var d_ple_gate: List[UInt64]
    var d_ple_projection: List[UInt64]
    var d_qkvw: List[UInt64]
    var d_gupw: List[UInt64]

    var d_in_norms: List[UInt64]
    var d_post_attn_norms: List[UInt64]
    var d_pre_ff_norms: List[UInt64]
    var d_post_ff_norms: List[UInt64]
    var d_q_norms: List[UInt64]
    var d_k_norms: List[UInt64]
    var d_post_ple_norms: List[UInt64]
    var layer_scalars: List[Float32]

    # KV cache: [layer][1 kv head, max_seq, 256] fp32.
    var d_k_cache: List[UInt64]
    var d_v_cache: List[UInt64]

    # Per-token scratch.
    var d_x: UInt64
    var d_normed: UInt64
    var d_q: UInt64
    var d_qkv: UInt64
    var d_k_new: UInt64
    var d_v_new: UInt64
    var d_attn: UInt64
    var d_o: UInt64
    var d_pn: UInt64
    var d_gate: UInt64
    var d_up: UInt64
    var d_gate_up: UInt64
    var d_mlp: UInt64
    var d_down: UInt64
    var d_ple_token: UInt64
    var d_ple_proj: UInt64
    var d_ple_gate_s: UInt64
    var d_ple_out: UInt64
    var d_logits: UInt64
    var d_q4_scratch: UInt64
    var d_scores: UInt64
    var d_attn_scratch: UInt64
    var d_lmhead_in: UInt64
    var d_decode_token: UInt64
    var d_decode_stats: UInt64

    def __init__(
        out self, dir: String, max_seq: Int, ctx: DeviceContext
    ) raises:
        self.ctx = ctx
        self.handle = cublas_create()
        cublas_set_stream(self.handle, self.ctx)
        self.max_seq = max_seq
        if self.max_seq < E2B_SLIDING_WINDOW:
            self.max_seq = E2B_SLIDING_WINDOW
        self.seq_len = 0

        var use_bf16_gather = _e2b_bf16_gather_enabled()
        var use_fused_gemv = _e2b_fused_gemv_enabled()
        self.d_embed = UInt64(0)
        self.d_embed_per_layer = UInt64(0)
        self.embed_bf16 = False
        self.embed_per_layer_bf16 = False

        if use_bf16_gather:
            self.d_embed = _try_load_e2b_bf16(
                dir + "embed_tokens_weight",
                E2B_VOCAB * E2B_HIDDEN,
            )
            self.embed_bf16 = self.d_embed != 0
        if self.d_embed == 0:
            self.d_embed = _load_e2b_f32(dir + "embed_tokens_weight")
        self.d_embed_q4 = load_to_gpu_q4(dir + "embed_tokens_weight.q4")
        if use_bf16_gather:
            self.d_embed_per_layer = _try_load_e2b_bf16(
                dir + "embed_tokens_per_layer_weight",
                E2B_VOCAB_SIZE_PER_LAYER_INPUT * E2B_PLE_PACKED,
            )
            self.embed_per_layer_bf16 = self.d_embed_per_layer != 0
        if self.d_embed_per_layer == 0:
            self.d_embed_per_layer = _load_e2b_f32(
                dir + "embed_tokens_per_layer_weight"
            )
        self.d_norm = _load_e2b_f32(dir + "norm_weight")
        self.d_per_layer_model_projection = load_to_gpu_q4(
            dir + "per_layer_model_projection_weight.q4"
        )
        self.d_per_layer_projection_norm = _load_e2b_f32(
            dir + "per_layer_projection_norm_weight"
        )

        self.d_qw = List[UInt64]()
        self.d_kw = List[UInt64]()
        self.d_vw = List[UInt64]()
        self.d_ow = List[UInt64]()
        self.d_gw = List[UInt64]()
        self.d_uw = List[UInt64]()
        self.d_dw = List[UInt64]()
        self.d_ple_gate = List[UInt64]()
        self.d_ple_projection = List[UInt64]()
        self.d_qkvw = List[UInt64]()
        self.d_gupw = List[UInt64]()
        self.d_in_norms = List[UInt64]()
        self.d_post_attn_norms = List[UInt64]()
        self.d_pre_ff_norms = List[UInt64]()
        self.d_post_ff_norms = List[UInt64]()
        self.d_q_norms = List[UInt64]()
        self.d_k_norms = List[UInt64]()
        self.d_post_ple_norms = List[UInt64]()
        self.layer_scalars = List[Float32]()

        for layer in range(E2B_NUM_LAYERS):
            var p = dir + "layers_" + String(layer) + "_"
            var q_path = p + "self_attn_q_proj_weight.q4"
            var k_path = p + "self_attn_k_proj_weight.q4"
            var v_path = p + "self_attn_v_proj_weight.q4"
            var g_path = p + "mlp_gate_proj_weight.q4"
            var u_path = p + "mlp_up_proj_weight.q4"
            var use_layer_qkv = use_fused_gemv and not _e2b_is_shared_layer(layer)
            if use_layer_qkv:
                var qkv = load_to_gpu_q4_concat3(q_path, k_path, v_path)
                if qkv != 0:
                    self.d_qw.append(UInt64(0))
                    self.d_qkvw.append(qkv)
                else:
                    self.d_qw.append(load_to_gpu_q4(q_path))
                    self.d_qkvw.append(UInt64(0))
                    use_layer_qkv = False
            else:
                self.d_qw.append(load_to_gpu_q4(q_path))
                self.d_qkvw.append(UInt64(0))
            if _e2b_is_shared_layer(layer):
                self.d_kw.append(UInt64(0))
                self.d_vw.append(UInt64(0))
                self.d_k_norms.append(UInt64(0))
            else:
                if use_layer_qkv:
                    self.d_kw.append(UInt64(0))
                    self.d_vw.append(UInt64(0))
                else:
                    self.d_kw.append(load_to_gpu_q4(k_path))
                    self.d_vw.append(load_to_gpu_q4(v_path))
                self.d_k_norms.append(
                    _load_e2b_f32(p + "self_attn_k_norm_weight")
                )
            self.d_ow.append(load_to_gpu_q4(p + "self_attn_o_proj_weight.q4"))
            if use_fused_gemv:
                var gup = load_to_gpu_q4_concat2(g_path, u_path)
                if gup != 0:
                    self.d_gupw.append(gup)
                    self.d_gw.append(UInt64(0))
                    self.d_uw.append(UInt64(0))
                else:
                    self.d_gupw.append(UInt64(0))
                    self.d_gw.append(load_to_gpu_q4(g_path))
                    self.d_uw.append(load_to_gpu_q4(u_path))
            else:
                self.d_gupw.append(UInt64(0))
                self.d_gw.append(load_to_gpu_q4(g_path))
                self.d_uw.append(load_to_gpu_q4(u_path))
            self.d_dw.append(load_to_gpu_q4(p + "mlp_down_proj_weight.q4"))
            self.d_ple_gate.append(
                load_to_gpu_q4(p + "per_layer_input_gate_weight.q4")
            )
            self.d_ple_projection.append(
                load_to_gpu_q4(p + "per_layer_projection_weight.q4")
            )
            self.d_in_norms.append(_load_e2b_f32(p + "input_layernorm_weight"))
            self.d_post_attn_norms.append(
                _load_e2b_f32(p + "post_attention_layernorm_weight")
            )
            self.d_pre_ff_norms.append(
                _load_e2b_f32(p + "pre_feedforward_layernorm_weight")
            )
            self.d_post_ff_norms.append(
                _load_e2b_f32(p + "post_feedforward_layernorm_weight")
            )
            self.d_q_norms.append(_load_e2b_f32(p + "self_attn_q_norm_weight"))
            self.d_post_ple_norms.append(
                _load_e2b_f32(p + "post_per_layer_input_norm_weight")
            )
            var scalar_data = _read_e2b_f32(p + "layer_scalar")
            if len(scalar_data) > 0:
                self.layer_scalars.append(scalar_data[0])
            else:
                self.layer_scalars.append(Float32(1.0))

        self.d_k_cache = List[UInt64]()
        self.d_v_cache = List[UInt64]()
        for layer in range(E2B_NUM_LAYERS):
            var l_hd = _e2b_layer_head_dim(layer)
            self.d_k_cache.append(cuda_malloc(self.max_seq * l_hd * 4))
            self.d_v_cache.append(cuda_malloc(self.max_seq * l_hd * 4))

        self.d_x = cuda_malloc(E2B_HIDDEN * 4)
        self.d_normed = cuda_malloc(E2B_HIDDEN * 4)
        self.d_q = cuda_malloc(E2B_NUM_HEADS * E2B_MAX_HEAD_DIM * 4)
        self.d_qkv = cuda_malloc(
            (E2B_NUM_HEADS + 2 * E2B_NUM_KV_HEADS) * E2B_MAX_HEAD_DIM * 4
        )
        self.d_k_new = cuda_malloc(E2B_NUM_KV_HEADS * E2B_MAX_HEAD_DIM * 4)
        self.d_v_new = cuda_malloc(E2B_NUM_KV_HEADS * E2B_MAX_HEAD_DIM * 4)
        self.d_attn = cuda_malloc(E2B_NUM_HEADS * E2B_MAX_HEAD_DIM * 4)
        self.d_o = cuda_malloc(E2B_HIDDEN * 4)
        self.d_pn = cuda_malloc(E2B_HIDDEN * 4)
        self.d_gate = cuda_malloc(E2B_INTERMEDIATE * 4)
        self.d_up = cuda_malloc(E2B_INTERMEDIATE * 4)
        self.d_gate_up = cuda_malloc(2 * E2B_INTERMEDIATE * 4)
        self.d_mlp = cuda_malloc(E2B_INTERMEDIATE * 4)
        self.d_down = cuda_malloc(E2B_HIDDEN * 4)
        self.d_ple_token = cuda_malloc(E2B_PLE_PACKED * 4)
        self.d_ple_proj = cuda_malloc(E2B_PLE_PACKED * 4)
        self.d_ple_gate_s = cuda_malloc(E2B_PLE_DIM * 4)
        self.d_ple_out = cuda_malloc(E2B_HIDDEN * 4)
        self.d_logits = cuda_malloc(E2B_VOCAB * 4)
        self.d_lmhead_in = cuda_malloc(E2B_HIDDEN * 4)
        self.d_decode_token = cuda_malloc(4)
        self.d_decode_stats = cuda_malloc(8)
        # q8_1 activation scratch.  E2B_INTERMEDIATE is the max FF width.
        self.d_q4_scratch = cuda_malloc(E2B_INTERMEDIATE * 4)
        self.d_scores = cuda_malloc(E2B_NUM_HEADS * self.max_seq * 4)
        self.d_attn_scratch = cuda_malloc(E2B_NUM_HEADS * E2B_MAX_HEAD_DIM * 4)
        cuda_sync()

    def _load_token_embedding(mut self, token: Int) raises:
        if self.embed_bf16:
            gpu_embed_load_bf16_mojo(
                self.ctx,
                self.d_x,
                self.d_embed,
                token,
                E2B_HIDDEN,
                sqrt(Float32(E2B_HIDDEN)),
            )
        else:
            gpu_embed_load_mojo(
                self.ctx,
                self.d_x,
                self.d_embed,
                token,
                E2B_HIDDEN,
                sqrt(Float32(E2B_HIDDEN)),
            )

    def _load_per_layer_embedding(mut self, token: Int) raises:
        if self.embed_per_layer_bf16:
            gpu_embed_load_bf16_mojo(
                self.ctx,
                self.d_ple_token,
                self.d_embed_per_layer,
                token,
                E2B_PLE_PACKED,
                sqrt(Float32(E2B_PLE_DIM)),
            )
        else:
            gpu_embed_load_mojo(
                self.ctx,
                self.d_ple_token,
                self.d_embed_per_layer,
                token,
                E2B_PLE_PACKED,
                sqrt(Float32(E2B_PLE_DIM)),
            )

    def _compute_per_layer_inputs(mut self, token: Int) raises:
        self._load_per_layer_embedding(token)
        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_ple_proj,
            self.d_x,
            self.d_per_layer_model_projection,
            self.d_q4_scratch,
            E2B_HIDDEN,
            E2B_PLE_PACKED,
        )
        gpu_scalar_mul_mojo(
            self.ctx,
            self.d_ple_proj,
            Float32(1.0) / sqrt(Float32(E2B_HIDDEN)),
            E2B_PLE_PACKED,
        )
        gpu_rmsnorm_batched_mojo(
            self.ctx,
            self.d_ple_proj,
            self.d_ple_proj,
            self.d_per_layer_projection_norm,
            E2B_PLE_DIM,
            E2B_NUM_LAYERS,
        )
        gpu_residual_add_scaled_mojo(
            self.ctx,
            self.d_ple_proj,
            self.d_ple_token,
            E2B_PLE_PACKED,
            Float32(0.70710678118),
        )

    def _run_layer(
        mut self,
        layer: Int,
        pos: Int,
        trace_layer: Int = -1,
        out_phase_ptr: UInt64 = 0,
        out_meta_ptr: UInt64 = 0,
        out_attn_ptr: UInt64 = 0,
        out_attn_meta_ptr: UInt64 = 0,
    ) raises:
        var is_shared = _e2b_is_shared_layer(layer)
        var is_full = _e2b_layer_is_full(layer)
        var l_hd = _e2b_layer_head_dim(layer)
        var l_qd = E2B_NUM_HEADS * l_hd
        var l_kvd = E2B_NUM_KV_HEADS * l_hd
        var trace_this = out_phase_ptr != 0 and layer == trace_layer
        if trace_this:
            self.ctx.synchronize()
            cuda_memcpy(out_phase_ptr, self.d_x, E2B_HIDDEN * 4, 2)

        gpu_rmsnorm_mojo(
            self.ctx,
            self.d_x,
            self.d_normed,
            self.d_in_norms[layer],
            E2B_HIDDEN,
        )

        gpu_q8_quantize_dev(
            self.ctx, self.d_normed, self.d_q4_scratch, E2B_HIDDEN
        )
        var d_q_cur = self.d_q
        var d_k_cur = self.d_k_new
        var d_v_cur = self.d_v_new
        if self.d_qkvw[layer] != 0:
            var qkv_n = l_qd + 2 * l_kvd
            gpu_matmul_q4_dp4a_gemv_dev[4](
                self.ctx,
                self.d_qkv,
                self.d_q4_scratch,
                self.d_qkvw[layer],
                E2B_HIDDEN,
                qkv_n,
            )
            d_q_cur = self.d_qkv
            d_k_cur = self.d_qkv + UInt64(l_qd * 4)
            d_v_cur = self.d_qkv + UInt64((l_qd + l_kvd) * 4)
        else:
            gpu_matmul_q4_dp4a_gemv_dev[4](
                self.ctx,
                self.d_q,
                self.d_q4_scratch,
                self.d_qw[layer],
                E2B_HIDDEN,
                l_qd,
            )

            if not is_shared:
                gpu_matmul_q4_dp4a_gemv_dev[4](
                    self.ctx,
                    self.d_k_new,
                    self.d_q4_scratch,
                    self.d_kw[layer],
                    E2B_HIDDEN,
                    l_kvd,
                )
                gpu_matmul_q4_dp4a_gemv_dev[4](
                    self.ctx,
                    self.d_v_new,
                    self.d_q4_scratch,
                    self.d_vw[layer],
                    E2B_HIDDEN,
                    l_kvd,
                )

        gpu_qk_norm_mojo(
            self.ctx,
            d_q_cur,
            self.d_q_norms[layer],
            E2B_NUM_HEADS,
            l_hd,
        )
        if is_full:
            gpu_rope_mojo(
                self.ctx,
                d_q_cur,
                pos,
                E2B_NUM_HEADS,
                l_hd,
                1000000.0,
                l_hd // 4,
            )
        else:
            gpu_rope_mojo(
                self.ctx,
                d_q_cur,
                pos,
                E2B_NUM_HEADS,
                l_hd,
                10000.0,
                l_hd,
            )

        var read_layer = layer
        if is_shared:
            read_layer = _e2b_shared_source_layer(layer)
        else:
            gpu_rmsnorm_no_weight_mojo(
                self.ctx, d_v_cur, E2B_NUM_KV_HEADS, l_hd
            )
            gpu_qk_norm_mojo(
                self.ctx,
                d_k_cur,
                self.d_k_norms[layer],
                E2B_NUM_KV_HEADS,
                l_hd,
            )
            if is_full:
                gpu_rope_mojo(
                    self.ctx,
                    d_k_cur,
                    pos,
                    E2B_NUM_KV_HEADS,
                    l_hd,
                    1000000.0,
                    l_hd // 4,
                )
            else:
                gpu_rope_mojo(
                    self.ctx,
                    d_k_cur,
                    pos,
                    E2B_NUM_KV_HEADS,
                    l_hd,
                    10000.0,
                    l_hd,
                )
            append_kv_gpu_dev_async(
                self.ctx,
                self.d_k_cache[layer],
                self.d_v_cache[layer],
                d_k_cur,
                d_v_cur,
                E2B_NUM_KV_HEADS,
                l_hd,
                l_hd,
                self.max_seq,
                pos,
            )

        var read_hd = _e2b_layer_head_dim(read_layer)
        if layer == trace_layer:
            var shared_flag = 0
            if is_shared:
                shared_flag = 1
            _host_write_i32_4(
                out_meta_ptr, read_layer, l_hd, read_hd, shared_flag
            )

        var attn_len = pos + 1
        var win_start = 0
        if (not is_full) and attn_len > E2B_SLIDING_WINDOW:
            win_start = attn_len - E2B_SLIDING_WINDOW
            attn_len = E2B_SLIDING_WINDOW
        var d_k_read = self.d_k_cache[read_layer] + UInt64(win_start * read_hd * 4)
        var d_v_read = self.d_v_cache[read_layer] + UInt64(win_start * read_hd * 4)
        if trace_this and out_attn_ptr != 0:
            cuda_sync()
            var dump_elems = 16
            var last_off = UInt64((attn_len - 1) * read_hd * 4)
            cuda_memcpy(out_attn_ptr, d_q_cur, dump_elems * 4, 2)
            cuda_memcpy(
                out_attn_ptr + UInt64(dump_elems * 4),
                d_k_read,
                dump_elems * 4,
                2,
            )
            cuda_memcpy(
                out_attn_ptr + UInt64(2 * dump_elems * 4),
                d_v_read,
                dump_elems * 4,
                2,
            )
            cuda_memcpy(
                out_attn_ptr + UInt64(3 * dump_elems * 4),
                d_k_read + last_off,
                dump_elems * 4,
                2,
            )
            cuda_memcpy(
                out_attn_ptr + UInt64(4 * dump_elems * 4),
                d_v_read + last_off,
                dump_elems * 4,
                2,
            )
            _host_write_i32_8(
                out_attn_meta_ptr,
                read_layer,
                pos + 1,
                attn_len,
                win_start,
                l_hd,
                read_hd,
                E2B_NUM_HEADS // E2B_NUM_KV_HEADS,
                self.max_seq,
            )
        attention_gpu_fp32_dev(
            self.ctx,
            self.handle,
            d_q_cur,
            d_k_read,
            d_v_read,
            self.d_scores,
            self.d_attn,
            E2B_NUM_HEADS,
            E2B_NUM_KV_HEADS,
            l_hd,
            read_hd,
            self.max_seq,
            attn_len,
            E2B_NUM_HEADS // E2B_NUM_KV_HEADS,
        )

        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_o,
            self.d_attn,
            self.d_ow[layer],
            self.d_q4_scratch,
            l_qd,
            E2B_HIDDEN,
        )
        gpu_rmsnorm_residual_add_scaled_mojo(
            self.ctx,
            self.d_x,
            self.d_o,
            self.d_post_attn_norms[layer],
            E2B_HIDDEN,
            Float32(1.0),
        )
        if trace_this:
            self.ctx.synchronize()
            cuda_memcpy(
                out_phase_ptr + UInt64(E2B_HIDDEN * 4),
                self.d_x,
                E2B_HIDDEN * 4,
                2,
            )

        gpu_rmsnorm_mojo(
            self.ctx,
            self.d_x,
            self.d_pn,
            self.d_pre_ff_norms[layer],
            E2B_HIDDEN,
        )
        var ff = e2b_layer_intermediate_size(layer)
        gpu_q8_quantize_dev(self.ctx, self.d_pn, self.d_q4_scratch, E2B_HIDDEN)
        var d_gate_cur = self.d_gate
        var d_up_cur = self.d_up
        if self.d_gupw[layer] != 0:
            gpu_matmul_q4_dp4a_gemv_dev[4](
                self.ctx,
                self.d_gate_up,
                self.d_q4_scratch,
                self.d_gupw[layer],
                E2B_HIDDEN,
                2 * ff,
            )
            d_gate_cur = self.d_gate_up
            d_up_cur = self.d_gate_up + UInt64(ff * 4)
        else:
            gpu_matmul_q4_dp4a_gemv_dev[4](
                self.ctx,
                self.d_gate,
                self.d_q4_scratch,
                self.d_gw[layer],
                E2B_HIDDEN,
                ff,
            )
            gpu_matmul_q4_dp4a_gemv_dev[4](
                self.ctx,
                self.d_up,
                self.d_q4_scratch,
                self.d_uw[layer],
                E2B_HIDDEN,
                ff,
            )
        gpu_gelu_mul_mojo(
            self.ctx, self.d_mlp, d_gate_cur, d_up_cur, ff
        )
        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_down,
            self.d_mlp,
            self.d_dw[layer],
            self.d_q4_scratch,
            ff,
            E2B_HIDDEN,
        )
        gpu_rmsnorm_residual_add_scaled_mojo(
            self.ctx,
            self.d_x,
            self.d_down,
            self.d_post_ff_norms[layer],
            E2B_HIDDEN,
            Float32(1.0),
        )
        if trace_this:
            self.ctx.synchronize()
            cuda_memcpy(
                out_phase_ptr + UInt64(2 * E2B_HIDDEN * 4),
                self.d_x,
                E2B_HIDDEN * 4,
                2,
            )

        var ple_slice = self.d_ple_proj + UInt64(layer * E2B_PLE_DIM * 4)
        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_ple_gate_s,
            self.d_x,
            self.d_ple_gate[layer],
            self.d_q4_scratch,
            E2B_HIDDEN,
            E2B_PLE_DIM,
        )
        gpu_gelu_mul_mojo(
            self.ctx,
            self.d_ple_gate_s,
            self.d_ple_gate_s,
            ple_slice,
            E2B_PLE_DIM,
        )
        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_ple_out,
            self.d_ple_gate_s,
            self.d_ple_projection[layer],
            self.d_q4_scratch,
            E2B_PLE_DIM,
            E2B_HIDDEN,
        )
        gpu_rmsnorm_residual_add_scaled_mojo(
            self.ctx,
            self.d_x,
            self.d_ple_out,
            self.d_post_ple_norms[layer],
            E2B_HIDDEN,
            self.layer_scalars[layer],
        )
        if trace_this:
            self.ctx.synchronize()
            cuda_memcpy(
                out_phase_ptr + UInt64(3 * E2B_HIDDEN * 4),
                self.d_x,
                E2B_HIDDEN * 4,
                2,
            )

    def _finish_logits(mut self) raises -> Int:
        gpu_rmsnorm_mojo(
            self.ctx, self.d_x, self.d_lmhead_in, self.d_norm, E2B_HIDDEN
        )
        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_logits,
            self.d_lmhead_in,
            self.d_embed_q4,
            self.d_q4_scratch,
            E2B_HIDDEN,
            E2B_VOCAB,
        )
        _gpu_softcap(
            self.ctx, self.d_logits, E2B_VOCAB, E2B_FINAL_LOGIT_SOFTCAP
        )
        return device_decode_token_into_stream(
            self.ctx,
            self.d_logits,
            E2B_VOCAB,
            self.d_decode_token,
            Float32(0.0),
        )

    def _finish_logits_conf(
        mut self, mut confs: List[Float32], mut gaps: List[Float32]
    ) raises -> Int:
        gpu_rmsnorm_mojo(
            self.ctx, self.d_x, self.d_lmhead_in, self.d_norm, E2B_HIDDEN
        )
        gpu_matmul_q4_dp4a_dev[4](
            self.ctx,
            self.d_logits,
            self.d_lmhead_in,
            self.d_embed_q4,
            self.d_q4_scratch,
            E2B_HIDDEN,
            E2B_VOCAB,
        )
        _gpu_softcap(
            self.ctx, self.d_logits, E2B_VOCAB, E2B_FINAL_LOGIT_SOFTCAP
        )
        return device_decode_token_conf_into(
            self.ctx,
            self.d_logits,
            E2B_VOCAB,
            self.d_decode_token,
            self.d_decode_stats,
            confs,
            gaps,
            Float32(0.0),
        )

    def step_token(mut self, token: Int) raises -> Int:
        if self.seq_len >= self.max_seq:
            raise Error("E2B draft KV cache full")
        var pos = self.seq_len
        self._load_token_embedding(token)
        self._compute_per_layer_inputs(token)
        for layer in range(E2B_NUM_LAYERS):
            self._run_layer(layer, pos)
        var tok = self._finish_logits()
        self.seq_len += 1
        return tok

    def step_token_conf(
        mut self, token: Int, mut confs: List[Float32], mut gaps: List[Float32]
    ) raises -> Int:
        if self.seq_len >= self.max_seq:
            raise Error("E2B draft KV cache full")
        var pos = self.seq_len
        self._load_token_embedding(token)
        self._compute_per_layer_inputs(token)
        for layer in range(E2B_NUM_LAYERS):
            self._run_layer(layer, pos)
        var tok = self._finish_logits_conf(confs, gaps)
        self.seq_len += 1
        return tok

    def step_token_trace_last(
        mut self,
        token: Int,
        out_hs_ptr: UInt64,
        out_logits_ptr: UInt64,
    ) raises -> Int:
        if self.seq_len >= self.max_seq:
            raise Error("E2B draft KV cache full")
        var pos = self.seq_len
        self._load_token_embedding(token)
        if out_hs_ptr != 0:
            self.ctx.synchronize()
            cuda_memcpy(out_hs_ptr, self.d_x, E2B_HIDDEN * 4, 2)
        self._compute_per_layer_inputs(token)
        for layer in range(E2B_NUM_LAYERS):
            self._run_layer(layer, pos)
            if out_hs_ptr != 0:
                self.ctx.synchronize()
                cuda_memcpy(
                    out_hs_ptr + UInt64((layer + 1) * E2B_HIDDEN * 4),
                    self.d_x,
                    E2B_HIDDEN * 4,
                    2,
                )
        var tok = self._finish_logits()
        if out_logits_ptr != 0:
            self.ctx.synchronize()
            cuda_memcpy(out_logits_ptr, self.d_logits, E2B_VOCAB * 4, 2)
        self.seq_len += 1
        return tok

    def step_token_trace_layer(
        mut self,
        token: Int,
        trace_layer: Int,
        out_phase_ptr: UInt64,
        out_meta_ptr: UInt64,
        out_attn_ptr: UInt64 = 0,
        out_attn_meta_ptr: UInt64 = 0,
    ) raises -> Int:
        if self.seq_len >= self.max_seq:
            raise Error("E2B draft KV cache full")
        var pos = self.seq_len
        self._load_token_embedding(token)
        self._compute_per_layer_inputs(token)
        for layer in range(E2B_NUM_LAYERS):
            self._run_layer(
                layer,
                pos,
                trace_layer,
                out_phase_ptr,
                out_meta_ptr,
                out_attn_ptr,
                out_attn_meta_ptr,
            )
        var tok = self._finish_logits()
        self.seq_len += 1
        return tok

    def prefill(mut self, tokens: List[Int]) raises -> Int:
        self.set_cache_len(0)
        var out_tok = 0
        for i in range(len(tokens)):
            out_tok = self.step_token(tokens[i])
        return out_tok

    def prefill_debug_last(
        mut self,
        tokens: List[Int],
        out_hs_ptr: UInt64,
        out_logits_ptr: UInt64,
    ) raises -> Int:
        self.set_cache_len(0)
        var n = len(tokens)
        if n <= 0:
            raise Error("E2B debug prefill requires at least one token")
        var out_tok = 0
        for i in range(n):
            if i == n - 1:
                out_tok = self.step_token_trace_last(
                    tokens[i], out_hs_ptr, out_logits_ptr
                )
            else:
                out_tok = self.step_token(tokens[i])
        return out_tok

    def prefill_debug_layer(
        mut self,
        tokens: List[Int],
        trace_layer: Int,
        out_phase_ptr: UInt64,
        out_meta_ptr: UInt64,
        out_attn_ptr: UInt64 = 0,
        out_attn_meta_ptr: UInt64 = 0,
    ) raises -> Int:
        self.set_cache_len(0)
        var n = len(tokens)
        if n <= 0:
            raise Error("E2B debug prefill requires at least one token")
        if trace_layer < 0 or trace_layer >= E2B_NUM_LAYERS:
            raise Error("E2B debug layer out of range")
        var out_tok = 0
        for i in range(n):
            if i == n - 1:
                out_tok = self.step_token_trace_layer(
                    tokens[i],
                    trace_layer,
                    out_phase_ptr,
                    out_meta_ptr,
                    out_attn_ptr,
                    out_attn_meta_ptr,
                )
            else:
                out_tok = self.step_token(tokens[i])
        return out_tok

    def draft_k(
        mut self, seed_token: Int, k: Int, mut drafts: List[Int32]
    ) raises -> Int:
        var tok = seed_token
        var n = 0
        for _ in range(k):
            tok = self.step_token(tok)
            drafts.append(Int32(tok))
            n += 1
        return n

    def draft_k_conf(
        mut self,
        seed_token: Int,
        k: Int,
        mut drafts: List[Int32],
        mut confs: List[Float32],
        mut gaps: List[Float32],
    ) raises -> Int:
        var tok = seed_token
        var n = 0
        for _ in range(k):
            tok = self.step_token_conf(tok, confs, gaps)
            drafts.append(Int32(tok))
            n += 1
        return n

    def draft_until_conf(
        mut self,
        seed_token: Int,
        k_max: Int,
        p_min: Float32,
        mut drafts: List[Int32],
        mut confs: List[Float32],
    ) raises -> Int:
        var tok = seed_token
        var gaps = List[Float32]()
        var n = 0
        for _ in range(k_max):
            var conf_i = len(confs)
            tok = self.step_token_conf(tok, confs, gaps)
            if confs[conf_i] < p_min:
                break
            drafts.append(Int32(tok))
            n += 1
        return n

    def set_cache_len(mut self, n: Int):
        if n < 0:
            self.seq_len = 0
        elif n > self.max_seq:
            self.seq_len = self.max_seq
        else:
            self.seq_len = n

    def reset(mut self):
        self.seq_len = 0

    def cache_len(self) -> Int:
        return self.seq_len


def init_e2b_draft_handle(
    weights_dir: String, max_seq: Int = 1024
) raises -> UInt64:
    var ctx = DeviceContext()
    return init_e2b_draft_handle_with_context(weights_dir, max_seq, ctx)


def init_e2b_draft_handle_with_context(
    weights_dir: String, max_seq: Int, ctx: DeviceContext
) raises -> UInt64:
    var ptr = alloc[E2BDraftEngine](1)
    ptr.init_pointee_move(E2BDraftEngine(weights_dir, max_seq, ctx))
    return UInt64(Int(ptr))


def release_e2b_draft_handle(handle: UInt64):
    if handle == 0:
        return
    var ptr = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )

    if ptr[0].d_embed != 0:
        cuda_free(ptr[0].d_embed)
    if ptr[0].d_embed_q4 != 0:
        cuda_free(ptr[0].d_embed_q4)
    if ptr[0].d_embed_per_layer != 0:
        cuda_free(ptr[0].d_embed_per_layer)
    if ptr[0].d_norm != 0:
        cuda_free(ptr[0].d_norm)
    if ptr[0].d_per_layer_model_projection != 0:
        cuda_free(ptr[0].d_per_layer_model_projection)
    if ptr[0].d_per_layer_projection_norm != 0:
        cuda_free(ptr[0].d_per_layer_projection_norm)

    for layer in range(E2B_NUM_LAYERS):
        if ptr[0].d_qw[layer] != 0:
            cuda_free(ptr[0].d_qw[layer])
        if ptr[0].d_qkvw[layer] != 0:
            cuda_free(ptr[0].d_qkvw[layer])
        if ptr[0].d_kw[layer] != 0:
            cuda_free(ptr[0].d_kw[layer])
        if ptr[0].d_vw[layer] != 0:
            cuda_free(ptr[0].d_vw[layer])
        if ptr[0].d_ow[layer] != 0:
            cuda_free(ptr[0].d_ow[layer])
        if ptr[0].d_gw[layer] != 0:
            cuda_free(ptr[0].d_gw[layer])
        if ptr[0].d_gupw[layer] != 0:
            cuda_free(ptr[0].d_gupw[layer])
        if ptr[0].d_uw[layer] != 0:
            cuda_free(ptr[0].d_uw[layer])
        if ptr[0].d_dw[layer] != 0:
            cuda_free(ptr[0].d_dw[layer])
        if ptr[0].d_ple_gate[layer] != 0:
            cuda_free(ptr[0].d_ple_gate[layer])
        if ptr[0].d_ple_projection[layer] != 0:
            cuda_free(ptr[0].d_ple_projection[layer])
        if ptr[0].d_in_norms[layer] != 0:
            cuda_free(ptr[0].d_in_norms[layer])
        if ptr[0].d_post_attn_norms[layer] != 0:
            cuda_free(ptr[0].d_post_attn_norms[layer])
        if ptr[0].d_pre_ff_norms[layer] != 0:
            cuda_free(ptr[0].d_pre_ff_norms[layer])
        if ptr[0].d_post_ff_norms[layer] != 0:
            cuda_free(ptr[0].d_post_ff_norms[layer])
        if ptr[0].d_q_norms[layer] != 0:
            cuda_free(ptr[0].d_q_norms[layer])
        if ptr[0].d_k_norms[layer] != 0:
            cuda_free(ptr[0].d_k_norms[layer])
        if ptr[0].d_post_ple_norms[layer] != 0:
            cuda_free(ptr[0].d_post_ple_norms[layer])
        if ptr[0].d_k_cache[layer] != 0:
            cuda_free(ptr[0].d_k_cache[layer])
        if ptr[0].d_v_cache[layer] != 0:
            cuda_free(ptr[0].d_v_cache[layer])

    cuda_free(ptr[0].d_x)
    cuda_free(ptr[0].d_normed)
    cuda_free(ptr[0].d_q)
    cuda_free(ptr[0].d_qkv)
    cuda_free(ptr[0].d_k_new)
    cuda_free(ptr[0].d_v_new)
    cuda_free(ptr[0].d_attn)
    cuda_free(ptr[0].d_o)
    cuda_free(ptr[0].d_pn)
    cuda_free(ptr[0].d_gate)
    cuda_free(ptr[0].d_up)
    cuda_free(ptr[0].d_gate_up)
    cuda_free(ptr[0].d_mlp)
    cuda_free(ptr[0].d_down)
    cuda_free(ptr[0].d_ple_token)
    cuda_free(ptr[0].d_ple_proj)
    cuda_free(ptr[0].d_ple_gate_s)
    cuda_free(ptr[0].d_ple_out)
    cuda_free(ptr[0].d_logits)
    cuda_free(ptr[0].d_q4_scratch)
    cuda_free(ptr[0].d_scores)
    cuda_free(ptr[0].d_attn_scratch)
    cuda_free(ptr[0].d_lmhead_in)
    cuda_free(ptr[0].d_decode_token)
    cuda_free(ptr[0].d_decode_stats)
    cuda_sync()
    ptr.destroy_pointee()
    ptr.free()
