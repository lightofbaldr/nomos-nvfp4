"""Tier-0 Muse vision attention and single-layer runner.

Correctness-first implementation for M1.  Q/K/V are projected from BF16
weights, 2D RoPE is applied in Muse's [w,h,w,h] layout, and non-causal
attention is evaluated independently for every cu_seqlens segment.  Window
layers therefore become a block-diagonal sequence of ordinary attentions;
full layers pass a single [0,S] segment.  This deliberately favors a small,
auditable implementation over fusion.
"""

from std.ffi import external_call, c_int
from std.gpu import block_idx, thread_idx, block_dim
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from std.math import cos, sin, exp, sqrt, erf
from std.memory import UnsafePointer, alloc
from max.gpu.host import DeviceContext
from layout import row_major, stack_allocation

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.io import load_bf16_file_to_gpu
from lib.cublas import (
    cublas_create, cublas_set_stream, gpu_matmul_bf16_dev_batched,
)


comptime VISION_D = 1536
comptime VISION_FF = 8960
comptime VISION_NH = 16
comptime VISION_HD = 96
comptime VISION_ROPE_THETA = 10000.0
comptime VISION_LN_EPS = 1.0e-5
comptime VISION_LAYERS = 50
comptime PATCH_D = 1176
comptime PROJECTOR_D = 4096
comptime TEXT_D = 6656
comptime PERCEPTION_RMS_EPS = 1.0e-5


def _f32(ptr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ptr))


def _bf16(ptr: UInt64) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(ptr))


def _i32(ptr: UInt64) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(ptr))


def _round_bf16(x: Float32) -> Float32:
    return Float32(x.cast[DType.bfloat16]())


def add_bias_round_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows_arg: Int32, cols_arg: Int32,
):
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < rows * cols:
        x[i] = _round_bf16(x[i] + Float32(bias[i % cols]))


def gpu_add_bias_round(
    ctx: DeviceContext, x: UInt64, bias: UInt64, rows: Int, cols: Int,
) raises:
    var n = rows * cols
    var threads = 256
    var k = ctx.compile_function[add_bias_round_kernel]()
    ctx.enqueue_function(
        k, _f32(x), _bf16(bias), Int32(rows), Int32(cols),
        grid_dim=(n + threads - 1) // threads, block_dim=threads,
    )


def layernorm_bf16_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    rows_arg: Int32, cols_arg: Int32, eps: Float32,
):
    # One correctness-first block per row.  Thread zero performs the same FP32
    # definition as torch LayerNorm; parallel reduction is a later perf tier.
    if Int(thread_idx.x) != 0:
        return
    var row = Int(block_idx.x)
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    if row >= rows:
        return
    var base = row * cols
    var mean = Float32(0.0)
    for j in range(cols):
        mean += x[base + j]
    mean /= Float32(cols)
    var variance = Float32(0.0)
    for j in range(cols):
        var d = x[base + j] - mean
        variance += d * d
    variance /= Float32(cols)
    var inv = Float32(1.0) / sqrt(variance + eps)
    for j in range(cols):
        var y = (x[base + j] - mean) * inv
        y = y * Float32(weight[j]) + Float32(bias[j])
        output[base + j] = _round_bf16(y)


def gpu_layernorm_bf16(
    ctx: DeviceContext, x: UInt64, weight: UInt64, bias: UInt64,
    output: UInt64, rows: Int, cols: Int,
) raises:
    var k = ctx.compile_function[layernorm_bf16_kernel]()
    ctx.enqueue_function(
        k, _f32(x), _bf16(weight), _bf16(bias), _f32(output),
        Int32(rows), Int32(cols), Float32(VISION_LN_EPS),
        grid_dim=rows, block_dim=1,
    )


def vision_rope_2d_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Int32, MutAnyOrigin],
    S_arg: Int32,
):
    var S = Int(S_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    # One thread owns a rotate-half PAIR.  An earlier in-place elementwise
    # version let the two mate threads overwrite values while the other still
    # read them, a deterministic-looking race that inflated attention output.
    var half = VISION_HD // 2
    var total = S * VISION_NH * half
    if idx >= total:
        return
    var d = idx % half
    var t = idx // half
    var token = t // VISION_NH
    # cos/sin layout is [freq_w(24),freq_h(24),freq_w(24),freq_h(24)].
    var axis_w = d < 24
    var freq_i = d if axis_w else d - 24
    var exponent = Float32(2 * freq_i) / Float32(VISION_HD // 2)
    var inv_freq = exp(-exponent * Float32(9.210340371976184)) # ln(10000)
    var p = Float32(pos[token * 2 + (0 if axis_w else 1)])
    var angle = p * inv_freq
    var c = cos(angle)
    var s = sin(angle)
    var base = t * VISION_HD
    var lo = base + d
    var hi = lo + half
    var qa = q[lo]; var qb = q[hi]
    var ka = k[lo]; var kb = k[hi]
    q[lo] = _round_bf16(qa * c - qb * s)
    q[hi] = _round_bf16(qb * c + qa * s)
    k[lo] = _round_bf16(ka * c - kb * s)
    k[hi] = _round_bf16(kb * c + ka * s)


def gpu_vision_rope_2d(
    ctx: DeviceContext, q: UInt64, k: UInt64, pos: UInt64, S: Int,
) raises:
    var n = S * VISION_NH * (VISION_HD // 2)
    var threads = 256
    var kern = ctx.compile_function[vision_rope_2d_kernel]()
    ctx.enqueue_function(
        kern, _f32(q), _f32(k), _i32(pos), Int32(S),
        grid_dim=(n + threads - 1) // threads, block_dim=threads,
    )


def transpose_s_h_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    S_arg: Int32,
):
    var S = Int(S_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < S * VISION_D:
        var d = idx % VISION_HD
        var t = idx // VISION_HD
        var h = t % VISION_NH
        var s = t // VISION_NH
        dst[(h * S + s) * VISION_HD + d] = src[idx]


def transpose_h_s_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    S_arg: Int32,
):
    var S = Int(S_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < S * VISION_D:
        var d = idx % VISION_HD
        var t = idx // VISION_HD
        var h = t % VISION_NH
        var s = t // VISION_NH
        dst[(s * VISION_NH + h) * VISION_HD + d] = src[idx]


def scale_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32, scale: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        x[i] *= scale


def vision_segment_attention_kernel(
    qh: UnsafePointer[Float32, MutAnyOrigin],
    kh: UnsafePointer[Float32, MutAnyOrigin],
    vh: UnsafePointer[Float32, MutAnyOrigin],
    out_h: UnsafePointer[Float32, MutAnyOrigin],
    S_arg: Int32, start_arg: Int32, length_arg: Int32,
):
    """One block per (query, head), explicit non-causal softmax·V.

    MAX segment length is the 32x32 window (1024) for window layers and
    Tier-0's single-image full layer is currently 2552.  A fixed 4096 score
    tile covers both and keeps the segment boundary completely explicit.
    """
    var S = Int(S_arg); var start = Int(start_arg); var length = Int(length_arg)
    var bh = Int(block_idx.x)
    var q_local = bh // VISION_NH
    var head = bh % VISION_NH
    if q_local >= length or length > 4096:
        return
    var tid = Int(thread_idx.x)
    var scores = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[4096]())
    var stats = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[2]())
    var q_abs = start + q_local
    var q_base = (head * S + q_abs) * VISION_HD

    var key = tid
    while key < length:
        var k_base = (head * S + start + key) * VISION_HD
        var dot = Float32(0.0)
        for d in range(VISION_HD):
            dot += qh[q_base + d] * kh[k_base + d]
        scores[key] = dot
        key += Int(block_dim.x)
    barrier()

    if tid == 0:
        var mx = Float32(-3.4e38)
        for j in range(length):
            if scores[j] > mx: mx = scores[j]
        var denom = Float32(0.0)
        for j in range(length):
            denom += exp(scores[j] - mx)
        stats[0] = mx; stats[1] = denom
    barrier()

    if tid < VISION_HD:
        var acc = Float32(0.0)
        for j in range(length):
            var prob = exp(scores[j] - stats[0]) / stats[1]
            acc += prob * vh[(head * S + start + j) * VISION_HD + tid]
        # Write the model's native [S, heads, hd] layout directly.  Keeping a
        # second head-major output transpose here is both unnecessary and an
        # easy source of a norm-preserving permutation bug.
        out_h[(q_abs * VISION_NH + head) * VISION_HD + tid] = _round_bf16(acc)


def gpu_vision_segment_attention(
    ctx: DeviceContext, qh: UInt64, kh: UInt64, vh: UInt64, out_h: UInt64,
    S: Int, start: Int, length: Int,
) raises:
    if length <= 0 or length > 4096:
        raise Error("vision M1 segment length outside Tier-0 [1,4096]")
    var kern = ctx.compile_function[vision_segment_attention_kernel]()
    ctx.enqueue_function(
        kern, _f32(qh), _f32(kh), _f32(vh), _f32(out_h),
        Int32(S), Int32(start), Int32(length),
        grid_dim=length * VISION_NH, block_dim=128,
    )


def residual_round_kernel(
    residual: UnsafePointer[Float32, MutAnyOrigin],
    update: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        output[i] = _round_bf16(residual[i] + update[i])


def gelu_exact_round_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        var v = x[i]
        x[i] = _round_bf16(Float32(0.5) * v * (Float32(1.0) + erf(v / Float32(1.4142135623730951))))


def unorder_kernel(
    reordered: UnsafePointer[Float32, MutAnyOrigin],
    original: UnsafePointer[Float32, MutAnyOrigin],
    window_index: UnsafePointer[Int32, MutAnyOrigin], S_arg: Int32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var S = Int(S_arg)
    if idx < S * VISION_D:
        var row = idx // VISION_D
        var col = idx % VISION_D
        original[Int(window_index[row]) * VISION_D + col] = reordered[idx]


def reorder_kernel(
    original: UnsafePointer[Float32, MutAnyOrigin],
    reordered: UnsafePointer[Float32, MutAnyOrigin],
    window_index: UnsafePointer[Int32, MutAnyOrigin], S_arg: Int32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var S = Int(S_arg)
    if idx < S * VISION_D:
        var row = idx // VISION_D
        var col = idx % VISION_D
        reordered[idx] = original[Int(window_index[row]) * VISION_D + col]


def round_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg): x[i] = _round_bf16(x[i])


def add_pos_round_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    # HF casts host-resampled position rows to the patch-embedding dtype
    # before addition. Keep that BF16 boundary explicit at the ABI edge.
    if i < Int(n_arg): x[i] = _round_bf16(x[i] + _round_bf16(pos[i]))


def pixel_shuffle_2x2_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    t_arg: Int32, h_arg: Int32, w_arg: Int32,
):
    var t = Int(t_arg); var h = Int(h_arg); var w = Int(w_arg)
    var out_rows = t * (h // 2) * (w // 2)
    var out_d = VISION_D * 4
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < out_rows * out_d:
        var c = idx % out_d
        var row = idx // out_d
        # HF: [h/2,2,w/2,2] -> [h/2,w/2,2,2], then
        # [out,4,D] -> [out,D,4].  The sub-patch lane is therefore the
        # fastest-moving output channel inside each original feature.
        var d = c // 4
        var lane = c % 4
        var dh = lane // 2; var dw = lane % 2
        var per_frame = (h // 2) * (w // 2)
        var frame = row // per_frame
        var spatial = row % per_frame
        var oh = spatial // (w // 2); var ow = spatial % (w // 2)
        var src_row = frame * h * w + (oh * 2 + dh) * w + (ow * 2 + dw)
        y[idx] = x[src_row * VISION_D + d]


def scaleless_rmsnorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin], rows_arg: Int32, eps: Float32,
):
    if Int(thread_idx.x) != 0: return
    var row = Int(block_idx.x); var rows = Int(rows_arg)
    if row >= rows: return
    var off = row * TEXT_D
    var ss = Float32(0.0)
    for d in range(TEXT_D):
        var v = x[off + d]
        ss += v * v
    var inv = Float32(1.0) / sqrt(ss / Float32(TEXT_D) + eps)
    for d in range(TEXT_D):
        x[off + d] = _round_bf16(x[off + d] * inv)


def _launch_1d(ctx: DeviceContext, n: Int, kind: Int, a: UInt64, b: UInt64 = 0, c: UInt64 = 0) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    if kind == 0:
        var kern = ctx.compile_function[transpose_s_h_kernel]()
        ctx.enqueue_function(kern, _f32(a), _f32(b), Int32(n // VISION_D), grid_dim=blocks, block_dim=threads)
    elif kind == 1:
        var kern = ctx.compile_function[transpose_h_s_kernel]()
        ctx.enqueue_function(kern, _f32(a), _f32(b), Int32(n // VISION_D), grid_dim=blocks, block_dim=threads)
    elif kind == 2:
        var kern = ctx.compile_function[residual_round_kernel]()
        ctx.enqueue_function(kern, _f32(a), _f32(b), _f32(c), Int32(n), grid_dim=blocks, block_dim=threads)
    else:
        var kern = ctx.compile_function[gelu_exact_round_kernel]()
        ctx.enqueue_function(kern, _f32(a), Int32(n), grid_dim=blocks, block_dim=threads)


struct VisionM1(Movable):
    var ctx: DeviceContext
    var handle: UInt64
    var qw: List[UInt64]; var qb: List[UInt64]
    var kw: List[UInt64]; var kb: List[UInt64]
    var vw: List[UInt64]; var vb: List[UInt64]
    var ow: List[UInt64]; var ob: List[UInt64]
    var fc1w: List[UInt64]; var fc1b: List[UInt64]
    var fc2w: List[UInt64]; var fc2b: List[UInt64]
    var n1w: List[UInt64]; var n1b: List[UInt64]
    var n2w: List[UInt64]; var n2b: List[UInt64]
    var ln_post_w: UInt64; var ln_post_b: UInt64
    var adapter_fc1_w: UInt64; var adapter_fc2_w: UInt64
    var projection_w: UInt64
    var patch_w: UInt64; var ln_pre_w: UInt64; var ln_pre_b: UInt64

    def __init__(out self, weights_dir: String) raises:
        self.ctx = DeviceContext()
        self.handle = cublas_create()
        cublas_set_stream(self.handle, self.ctx)
        self.qw = List[UInt64](); self.qb = List[UInt64]()
        self.kw = List[UInt64](); self.kb = List[UInt64]()
        self.vw = List[UInt64](); self.vb = List[UInt64]()
        self.ow = List[UInt64](); self.ob = List[UInt64]()
        self.fc1w = List[UInt64](); self.fc1b = List[UInt64]()
        self.fc2w = List[UInt64](); self.fc2b = List[UInt64]()
        self.n1w = List[UInt64](); self.n1b = List[UInt64]()
        self.n2w = List[UInt64](); self.n2b = List[UInt64]()
        var base = weights_dir if weights_dir.endswith("/") else weights_dir + "/"
        for layer in range(VISION_LAYERS):
            var p = base + "vision_L" + String(layer) + "_"
            self.qw.append(load_bf16_file_to_gpu(p + "attn_q_w.bf16")); self.qb.append(load_bf16_file_to_gpu(p + "attn_q_b.bf16"))
            self.kw.append(load_bf16_file_to_gpu(p + "attn_k_w.bf16")); self.kb.append(load_bf16_file_to_gpu(p + "attn_k_b.bf16"))
            self.vw.append(load_bf16_file_to_gpu(p + "attn_v_w.bf16")); self.vb.append(load_bf16_file_to_gpu(p + "attn_v_b.bf16"))
            self.ow.append(load_bf16_file_to_gpu(p + "attn_o_w.bf16")); self.ob.append(load_bf16_file_to_gpu(p + "attn_o_b.bf16"))
            self.fc1w.append(load_bf16_file_to_gpu(p + "mlp_fc1_w.bf16")); self.fc1b.append(load_bf16_file_to_gpu(p + "mlp_fc1_b.bf16"))
            self.fc2w.append(load_bf16_file_to_gpu(p + "mlp_fc2_w.bf16")); self.fc2b.append(load_bf16_file_to_gpu(p + "mlp_fc2_b.bf16"))
            self.n1w.append(load_bf16_file_to_gpu(p + "norm1_w.bf16")); self.n1b.append(load_bf16_file_to_gpu(p + "norm1_b.bf16"))
            self.n2w.append(load_bf16_file_to_gpu(p + "norm2_w.bf16")); self.n2b.append(load_bf16_file_to_gpu(p + "norm2_b.bf16"))
        self.ln_post_w = load_bf16_file_to_gpu(base + "vision_ln_post_w.bf16")
        self.ln_post_b = load_bf16_file_to_gpu(base + "vision_ln_post_b.bf16")
        self.adapter_fc1_w = load_bf16_file_to_gpu(base + "vision_adapter_fc1_w.bf16")
        self.adapter_fc2_w = load_bf16_file_to_gpu(base + "vision_adapter_fc2_w.bf16")
        self.projection_w = load_bf16_file_to_gpu(base + "vision_projection_w.bf16")
        self.patch_w = load_bf16_file_to_gpu(base + "vision_patch_embed_w.bf16")
        self.ln_pre_w = load_bf16_file_to_gpu(base + "vision_ln_pre_w.bf16")
        self.ln_pre_b = load_bf16_file_to_gpu(base + "vision_ln_pre_b.bf16")

    def run_frontend(
        mut self, pixels_host: UInt64, pos_rows_host: UInt64,
        window_index_host: UInt64, S: Int, patch_host: UInt64,
        ln_pre_host: UInt64, reordered_host: UInt64,
    ) raises:
        if S <= 0: raise Error("invalid vision frontend length")
        var elems = S * VISION_D
        var d_pixels = cuda_malloc(S * PATCH_D * 4)
        var d_patch = cuda_malloc(elems * 4)
        var d_pos = cuda_malloc(elems * 4)
        var d_ln = cuda_malloc(elems * 4)
        var d_reordered = cuda_malloc(elems * 4)
        var d_index = cuda_malloc(S * 4)
        var d_bf16 = cuda_malloc(S * PATCH_D * 2)
        cuda_memcpy(d_pixels, pixels_host, S * PATCH_D * 4, 1)
        cuda_memcpy(d_pos, pos_rows_host, elems * 4, 1)
        cuda_memcpy(d_index, window_index_host, S * 4, 1)
        gpu_matmul_bf16_dev_batched(
            self.ctx, self.handle, d_patch, d_pixels, self.patch_w, d_bf16,
            S, PATCH_D, VISION_D,
        )
        var threads = 256
        var round_k = self.ctx.compile_function[round_kernel]()
        self.ctx.enqueue_function(round_k, _f32(d_patch), Int32(elems), grid_dim=(elems+threads-1)//threads, block_dim=threads)
        var add_k = self.ctx.compile_function[add_pos_round_kernel]()
        self.ctx.enqueue_function(add_k, _f32(d_patch), _f32(d_pos), Int32(elems), grid_dim=(elems+threads-1)//threads, block_dim=threads)
        self.ctx.synchronize()
        # HF's patch_embedder hook is after both the linear and positional add.
        cuda_memcpy(patch_host, d_patch, elems * 4, 2)
        gpu_layernorm_bf16(self.ctx, d_patch, self.ln_pre_w, self.ln_pre_b, d_ln, S, VISION_D)
        var reorder_k = self.ctx.compile_function[reorder_kernel]()
        self.ctx.enqueue_function(reorder_k, _f32(d_ln), _f32(d_reordered), _i32(d_index), Int32(S), grid_dim=(elems+threads-1)//threads, block_dim=threads)
        self.ctx.synchronize()
        cuda_memcpy(ln_pre_host, d_ln, elems * 4, 2)
        cuda_memcpy(reordered_host, d_reordered, elems * 4, 2)
        cuda_free(d_bf16); cuda_free(d_index); cuda_free(d_reordered)
        cuda_free(d_ln); cuda_free(d_pos); cuda_free(d_patch); cuda_free(d_pixels)

    def run_layer(
        mut self, layer: Int, hidden_host: UInt64, pos_host: UInt64,
        cu_host: UInt64, n_segments: Int, S: Int, out_host: UInt64,
        debug_stage: Int = 0, debug_host: UInt64 = 0,
    ) raises:
        if layer < 0 or layer >= VISION_LAYERS or S <= 0 or n_segments <= 0:
            raise Error("invalid M1 vision layer request")
        var elems = S * VISION_D
        var d_x = cuda_malloc(elems * 4)
        var d_norm = cuda_malloc(elems * 4)
        var d_q = cuda_malloc(elems * 4); var d_k = cuda_malloc(elems * 4); var d_v = cuda_malloc(elems * 4)
        var d_qh = cuda_malloc(elems * 4); var d_kh = cuda_malloc(elems * 4); var d_vh = cuda_malloc(elems * 4)
        var d_ah = cuda_malloc(elems * 4); var d_attn = cuda_malloc(elems * 4)
        var d_proj = cuda_malloc(elems * 4); var d_mlp = cuda_malloc(S * VISION_FF * 4)
        var d_bf16 = cuda_malloc(S * VISION_FF * 2)
        var d_pos = cuda_malloc(S * 2 * 4)
        cuda_memcpy(d_x, hidden_host, elems * 4, 1)
        cuda_memcpy(d_pos, pos_host, S * 2 * 4, 1)

        gpu_layernorm_bf16(self.ctx, d_x, self.n1w[layer], self.n1b[layer], d_norm, S, VISION_D)
        if debug_stage == 1 and debug_host != 0: cuda_memcpy(debug_host, d_norm, elems * 4, 2)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_q, d_norm, self.qw[layer], d_bf16, S, VISION_D, VISION_D)
        gpu_add_bias_round(self.ctx, d_q, self.qb[layer], S, VISION_D)
        if debug_stage == 2 and debug_host != 0: cuda_memcpy(debug_host, d_q, elems * 4, 2)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_k, d_norm, self.kw[layer], d_bf16, S, VISION_D, VISION_D)
        gpu_add_bias_round(self.ctx, d_k, self.kb[layer], S, VISION_D)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_v, d_norm, self.vw[layer], d_bf16, S, VISION_D, VISION_D)
        gpu_add_bias_round(self.ctx, d_v, self.vb[layer], S, VISION_D)
        if debug_stage == 12 and debug_host != 0: cuda_memcpy(debug_host, d_v, elems * 4, 2)
        gpu_vision_rope_2d(self.ctx, d_q, d_k, d_pos, S)
        if debug_stage == 3 and debug_host != 0: cuda_memcpy(debug_host, d_q, elems * 4, 2)
        var scale = Float32(1.0) / sqrt(Float32(VISION_HD))
        var threads = 256
        var sk = self.ctx.compile_function[scale_kernel]()
        self.ctx.enqueue_function(sk, _f32(d_q), Int32(elems), scale, grid_dim=(elems + threads - 1) // threads, block_dim=threads)
        _launch_1d(self.ctx, elems, 0, d_q, d_qh)
        _launch_1d(self.ctx, elems, 0, d_k, d_kh)
        _launch_1d(self.ctx, elems, 0, d_v, d_vh)
        if debug_stage == 13 and debug_host != 0: cuda_memcpy(debug_host, d_vh, elems * 4, 2)
        if debug_stage == 14 and debug_host != 0: cuda_memcpy(debug_host, d_qh, elems * 4, 2)
        if debug_stage == 15 and debug_host != 0: cuda_memcpy(debug_host, d_kh, elems * 4, 2)

        var cu = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(cu_host))
        for seg in range(n_segments):
            var start = Int(cu[seg]); var L = Int(cu[seg + 1]) - start
            if L <= 0: continue
            gpu_vision_segment_attention(self.ctx, d_qh, d_kh, d_vh, d_ah, S, start, L)
        cuda_memcpy(d_attn, d_ah, elems * 4, 3)
        if debug_stage == 4 and debug_host != 0: cuda_memcpy(debug_host, d_attn, elems * 4, 2)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_proj, d_attn, self.ow[layer], d_bf16, S, VISION_D, VISION_D)
        gpu_add_bias_round(self.ctx, d_proj, self.ob[layer], S, VISION_D)
        if debug_stage == 5 and debug_host != 0: cuda_memcpy(debug_host, d_proj, elems * 4, 2)
        _launch_1d(self.ctx, elems, 2, d_x, d_proj, d_x)
        if debug_stage == 6 and debug_host != 0: cuda_memcpy(debug_host, d_x, elems * 4, 2)

        gpu_layernorm_bf16(self.ctx, d_x, self.n2w[layer], self.n2b[layer], d_norm, S, VISION_D)
        if debug_stage == 7 and debug_host != 0: cuda_memcpy(debug_host, d_norm, elems * 4, 2)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_mlp, d_norm, self.fc1w[layer], d_bf16, S, VISION_D, VISION_FF)
        gpu_add_bias_round(self.ctx, d_mlp, self.fc1b[layer], S, VISION_FF)
        _launch_1d(self.ctx, S * VISION_FF, 3, d_mlp)
        if debug_stage == 8 and debug_host != 0: cuda_memcpy(debug_host, d_mlp, S * VISION_FF * 4, 2)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_proj, d_mlp, self.fc2w[layer], d_bf16, S, VISION_FF, VISION_D)
        gpu_add_bias_round(self.ctx, d_proj, self.fc2b[layer], S, VISION_D)
        if debug_stage == 9 and debug_host != 0: cuda_memcpy(debug_host, d_proj, elems * 4, 2)
        _launch_1d(self.ctx, elems, 2, d_x, d_proj, d_x)
        self.ctx.synchronize()
        cuda_memcpy(out_host, d_x, elems * 4, 2)

        cuda_free(d_pos); cuda_free(d_bf16); cuda_free(d_mlp); cuda_free(d_proj)
        cuda_free(d_attn); cuda_free(d_ah); cuda_free(d_vh); cuda_free(d_kh); cuda_free(d_qh)
        cuda_free(d_v); cuda_free(d_k); cuda_free(d_q); cuda_free(d_norm); cuda_free(d_x)

    def run_m2(
        mut self, hidden_reordered_host: UInt64, pos_host: UInt64,
        cu_window_host: UInt64, n_window_segments: Int, cu_full_host: UInt64,
        window_index_host: UInt64, S: Int, t: Int, h: Int, w: Int,
        layer49_host: UInt64, ln_post_host: UInt64,
        encoder_hidden_host: UInt64, features_host: UInt64,
    ) raises:
        if S != t * h * w or h % 2 != 0 or w % 2 != 0:
            raise Error("invalid vision M2 grid")
        var elems = S * VISION_D
        var scratch_a = alloc[Float32](elems)
        var scratch_b = alloc[Float32](elems)
        var src = UInt64(hidden_reordered_host)
        var dst = UInt64(Int(scratch_a))
        for layer in range(VISION_LAYERS):
            var is_full = ((layer + 1) % 4 == 0) or layer == VISION_LAYERS - 1
            self.run_layer(
                layer, src, pos_host,
                cu_full_host if is_full else cu_window_host,
                1 if is_full else n_window_segments, S, dst,
            )
            src = dst
            dst = UInt64(Int(scratch_b)) if dst == UInt64(Int(scratch_a)) else UInt64(Int(scratch_a))
        var final_src = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(src))
        var layer49 = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(layer49_host))
        for i in range(elems): layer49[i] = final_src[i]

        var out_rows = t * (h // 2) * (w // 2)
        var d_reordered = cuda_malloc(elems * 4)
        var d_original = cuda_malloc(elems * 4)
        var d_index = cuda_malloc(S * 4)
        var d_ln = cuda_malloc(elems * 4)
        var d_shuffle = cuda_malloc(out_rows * VISION_D * 4 * 4)
        var d_fc1 = cuda_malloc(out_rows * PROJECTOR_D * 4)
        var d_fc2 = cuda_malloc(out_rows * PROJECTOR_D * 4)
        var d_features = cuda_malloc(out_rows * TEXT_D * 4)
        var d_bf16 = cuda_malloc(out_rows * VISION_D * 4 * 2)
        cuda_memcpy(d_reordered, src, elems * 4, 1)
        cuda_memcpy(d_index, window_index_host, S * 4, 1)
        var threads = 256
        var uk = self.ctx.compile_function[unorder_kernel]()
        self.ctx.enqueue_function(uk, _f32(d_reordered), _f32(d_original), _i32(d_index), Int32(S), grid_dim=(elems+threads-1)//threads, block_dim=threads)
        gpu_layernorm_bf16(self.ctx, d_original, self.ln_post_w, self.ln_post_b, d_ln, S, VISION_D)
        self.ctx.synchronize()
        cuda_memcpy(ln_post_host, d_ln, elems * 4, 2)

        var pk = self.ctx.compile_function[pixel_shuffle_2x2_kernel]()
        var shuffled_elems = out_rows * VISION_D * 4
        self.ctx.enqueue_function(pk, _f32(d_ln), _f32(d_shuffle), Int32(t), Int32(h), Int32(w), grid_dim=(shuffled_elems+threads-1)//threads, block_dim=threads)
        self.ctx.synchronize()
        cuda_memcpy(encoder_hidden_host, d_shuffle, shuffled_elems * 4, 2)

        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_fc1, d_shuffle, self.adapter_fc1_w, d_bf16, out_rows, VISION_D*4, PROJECTOR_D)
        _launch_1d(self.ctx, out_rows * PROJECTOR_D, 3, d_fc1)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_fc2, d_fc1, self.adapter_fc2_w, d_bf16, out_rows, PROJECTOR_D, PROJECTOR_D)
        _launch_1d(self.ctx, out_rows * PROJECTOR_D, 3, d_fc2)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_features, d_fc2, self.projection_w, d_bf16, out_rows, PROJECTOR_D, TEXT_D)
        var rk = self.ctx.compile_function[scaleless_rmsnorm_kernel]()
        self.ctx.enqueue_function(rk, _f32(d_features), Int32(out_rows), Float32(PERCEPTION_RMS_EPS), grid_dim=out_rows, block_dim=1)
        self.ctx.synchronize()
        cuda_memcpy(features_host, d_features, out_rows * TEXT_D * 4, 2)

        cuda_free(d_bf16); cuda_free(d_features); cuda_free(d_fc2); cuda_free(d_fc1)
        cuda_free(d_shuffle); cuda_free(d_ln); cuda_free(d_index); cuda_free(d_original); cuda_free(d_reordered)
        scratch_b.free(); scratch_a.free()

    def free(mut self):
        for layer in range(VISION_LAYERS):
            cuda_free(self.qw[layer]); cuda_free(self.qb[layer]); cuda_free(self.kw[layer]); cuda_free(self.kb[layer])
            cuda_free(self.vw[layer]); cuda_free(self.vb[layer]); cuda_free(self.ow[layer]); cuda_free(self.ob[layer])
            cuda_free(self.fc1w[layer]); cuda_free(self.fc1b[layer]); cuda_free(self.fc2w[layer]); cuda_free(self.fc2b[layer])
            cuda_free(self.n1w[layer]); cuda_free(self.n1b[layer]); cuda_free(self.n2w[layer]); cuda_free(self.n2b[layer])
        cuda_free(self.ln_post_w); cuda_free(self.ln_post_b)
        cuda_free(self.adapter_fc1_w); cuda_free(self.adapter_fc2_w); cuda_free(self.projection_w)
        cuda_free(self.patch_w); cuda_free(self.ln_pre_w); cuda_free(self.ln_pre_b)
        self.qw.clear()
        if self.handle != 0:
            _ = external_call["cublasDestroy_v2", c_int, UInt64](self.handle)
            self.handle = UInt64(0)
