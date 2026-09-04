"""Correctness-first Qwen3TTSTokenizerV2 decoder for Breeze-TTS-2.

The production Breeze runtime requires the bundled Qwen tokenizer decoder;
the top-level Mimi codec is a training-only fallback and is deliberately not
implemented here.  M1 is staged at HF-visible boundaries so every new
machinery class is independently comparable before the waveform stack lands.
"""

from std.gpu import block_idx, block_dim, thread_idx
from std.collections import List
from std.math import exp, log, cos, sin, sqrt
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from layout import row_major, stack_allocation
from std.memory import UnsafePointer, alloc
from max.gpu.host import DeviceContext

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.io import file_size_bytes, load_f32_file_to_gpu
from lib.cublas import cublas_create, cublas_set_stream, cublas_sgemm


comptime CODEBOOKS = 16
comptime CODEBOOK_SIZE = 2048
comptime VQ_DIM = 256
comptime CODE_DIM = 512
comptime LATENT_DIM = 1024
comptime PRE_KERNEL = 3
comptime TRANSFORMER_D = 512
comptime TRANSFORMER_FF = 1024
comptime TRANSFORMER_QD = 1024
comptime TRANSFORMER_LAYERS = 8
comptime TRANSFORMER_HEADS = 16
comptime TRANSFORMER_HD = 64
comptime TRANSFORMER_WINDOW = 72
comptime TRANSFORMER_EPS = 1.0e-5
comptime TRANSFORMER_THETA = 10000.0


def _f32(ptr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(ptr))


def _i64(ptr: UInt64) -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(ptr))


def codec_rvq_lookup_kernel(
    codes: UnsafePointer[Int64, MutAnyOrigin],       # [16,T]
    tables: UnsafePointer[UInt64, MutAnyOrigin],     # 16 x [2048,256]
    semantic: UnsafePointer[Float32, MutAnyOrigin],  # [256,T]
    acoustic: UnsafePointer[Float32, MutAnyOrigin],  # [256,T]
    frames_arg: Int32,
):
    var frames = Int(frames_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= VQ_DIM * frames:
        return
    var channel = i // frames
    var frame = i % frames

    # Match SplitResidualVectorQuantizer.decode exactly: semantic codebook 0,
    # a separate left-to-right sum for acoustic codebooks 1..15, then add the
    # two projected branches.  The converter emits each codebook's effective
    # embedding_sum / clamp(cluster_usage) rows; the branch projections remain
    # explicit so their accumulation is independently gateable.
    var semantic_id = Int(codes[frame])
    if semantic_id < 0 or semantic_id >= CODEBOOK_SIZE:
        semantic[i] = 0.0
        acoustic[i] = 0.0
        return
    semantic[i] = _f32(tables[0])[semantic_id * VQ_DIM + channel]
    var acoustic_sum = Float32(0.0)
    for q in range(1, CODEBOOKS):
        var token = Int(codes[q * frames + frame])
        if token < 0 or token >= CODEBOOK_SIZE:
            acoustic[i] = 0.0
            return
        acoustic_sum += _f32(tables[q])[token * VQ_DIM + channel]
    acoustic[i] = acoustic_sum


def codec_rvq_project_kernel(
    semantic: UnsafePointer[Float32, MutAnyOrigin],  # [256,T]
    acoustic: UnsafePointer[Float32, MutAnyOrigin],  # [256,T]
    semantic_weight: UnsafePointer[Float32, MutAnyOrigin], # [512,256]
    acoustic_weight: UnsafePointer[Float32, MutAnyOrigin], # [512,256]
    output: UnsafePointer[Float32, MutAnyOrigin],    # [512,T]
    frames_arg: Int32,
):
    var frames = Int(frames_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= CODE_DIM * frames:
        return
    var channel = i // frames
    var frame = i % frames
    var first = Float32(0.0)
    var rest = Float32(0.0)
    for d in range(VQ_DIM):
        first += semantic[d * frames + frame] * semantic_weight[channel * VQ_DIM + d]
        rest += acoustic[d * frames + frame] * acoustic_weight[channel * VQ_DIM + d]
    output[i] = first + rest


def codec_causal_preconv_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],     # [512,T]
    weight: UnsafePointer[Float32, MutAnyOrigin],    # [1024,512,3]
    bias: UnsafePointer[Float32, MutAnyOrigin],      # [1024]
    output: UnsafePointer[Float32, MutAnyOrigin],    # [1024,T]
    frames_arg: Int32,
):
    var frames = Int(frames_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= LATENT_DIM * frames:
        return
    var out_channel = i // frames
    var frame = i % frames
    var acc = bias[out_channel]
    for in_channel in range(CODE_DIM):
        for tap in range(PRE_KERNEL):
            var source_frame = frame + tap - (PRE_KERNEL - 1)
            if source_frame >= 0:
                acc += (
                    input[in_channel * frames + source_frame]
                    * weight[(out_channel * CODE_DIM + in_channel) * PRE_KERNEL + tap]
                )
    output[i] = acc


def codec_transpose_preconv_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],   # [1024,T]
    output: UnsafePointer[Float32, MutAnyOrigin],  # [T,1024]
    frames_arg: Int32,
):
    var frames = Int(frames_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < LATENT_DIM * frames:
        var channel = i // frames
        var frame = i % frames
        output[frame * LATENT_DIM + channel] = input[i]


def codec_add_bias_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    bias: UnsafePointer[Float32, MutAnyOrigin],
    rows_arg: Int32,
    cols_arg: Int32,
):
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < rows * cols:
        x[i] += bias[i % cols]


def codec_rmsnorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    rows_arg: Int32,
):
    if Int(thread_idx.x) != 0:
        return
    var row = Int(block_idx.x)
    if row >= Int(rows_arg):
        return
    var off = row * TRANSFORMER_D
    var ss = Float32(0.0)
    for d in range(TRANSFORMER_D):
        var v = x[off + d]
        ss += v * v
    var inv = Float32(1.0) / sqrt(ss / Float32(TRANSFORMER_D) + Float32(TRANSFORMER_EPS))
    for d in range(TRANSFORMER_D):
        output[off + d] = x[off + d] * inv * weight[d]


def codec_rope_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    frames_arg: Int32,
):
    var frames = Int(frames_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var half = TRANSFORMER_HD // 2
    var total = frames * TRANSFORMER_HEADS * half
    if i >= total:
        return
    var pair = i % half
    var th = i // half
    var frame = th // TRANSFORMER_HEADS
    var off = th * TRANSFORMER_HD
    var inv = exp(-(Float32(2 * pair) / Float32(TRANSFORMER_HD)) * log(Float32(TRANSFORMER_THETA)))
    var angle = Float32(frame) * inv
    var c = cos(angle)
    var s = sin(angle)
    var lo = off + pair
    var hi = lo + half
    var qa = q[lo]
    var qb = q[hi]
    var ka = k[lo]
    var kb = k[hi]
    q[lo] = qa * c - qb * s
    q[hi] = qb * c + qa * s
    k[lo] = ka * c - kb * s
    k[hi] = kb * c + ka * s


def codec_sliding_attention_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    frames_arg: Int32,
):
    var frames = Int(frames_arg)
    var query_head = Int(block_idx.x)
    var frame = query_head // TRANSFORMER_HEADS
    var head = query_head % TRANSFORMER_HEADS
    if frame >= frames:
        return
    var tid = Int(thread_idx.x)
    var first = frame - (TRANSFORMER_WINDOW - 1)
    if first < 0:
        first = 0
    var count = frame - first + 1
    var scores = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TRANSFORMER_WINDOW]())
    var stats = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[2]())
    if tid < count:
        var key_frame = first + tid
        var qoff = (frame * TRANSFORMER_HEADS + head) * TRANSFORMER_HD
        var koff = (key_frame * TRANSFORMER_HEADS + head) * TRANSFORMER_HD
        var dot = Float32(0.0)
        for d in range(TRANSFORMER_HD):
            dot += q[qoff + d] * k[koff + d]
        scores[tid] = dot / sqrt(Float32(TRANSFORMER_HD))
    barrier()
    if tid == 0:
        var mx = Float32(-3.4e38)
        for j in range(count):
            if scores[j] > mx:
                mx = scores[j]
        var den = Float32(0.0)
        for j in range(count):
            den += exp(scores[j] - mx)
        stats[0] = mx
        stats[1] = den
    barrier()
    if tid < TRANSFORMER_HD:
        var acc = Float32(0.0)
        for j in range(count):
            var prob = exp(scores[j] - stats[0]) / stats[1]
            var voff = ((first + j) * TRANSFORMER_HEADS + head) * TRANSFORMER_HD
            acc += prob * v[voff + tid]
        output[(frame * TRANSFORMER_HEADS + head) * TRANSFORMER_HD + tid] = acc


def codec_scaled_residual_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    update: UnsafePointer[Float32, MutAnyOrigin],
    scale: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        x[i] += update[i] * scale[i % TRANSFORMER_D]


def codec_swiglu_kernel(
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        var v = gate[i]
        gate[i] = (v / (Float32(1.0) + exp(-v))) * up[i]


struct BreezeCodec:
    var ctx: DeviceContext
    var cublas: UInt64
    var tables_host: UnsafePointer[UInt64, MutUntrackedOrigin]
    var semantic_proj: UInt64
    var acoustic_proj: UInt64
    var preconv_weight: UInt64
    var preconv_bias: UInt64
    var transformer_input_weight: UInt64
    var transformer_input_bias: UInt64
    var transformer_q: List[UInt64]
    var transformer_k: List[UInt64]
    var transformer_v: List[UInt64]
    var transformer_o: List[UInt64]
    var transformer_gate: List[UInt64]
    var transformer_up: List[UInt64]
    var transformer_down: List[UInt64]
    var transformer_in_norm: List[UInt64]
    var transformer_post_norm: List[UInt64]
    var transformer_attn_scale: List[UInt64]
    var transformer_mlp_scale: List[UInt64]
    var transformer_final_norm: UInt64
    var transformer_output_weight: UInt64
    var transformer_output_bias: UInt64

    def __init__(out self, weights_dir: String) raises:
        self.ctx = DeviceContext()
        self.cublas = cublas_create()
        cublas_set_stream(self.cublas, self.ctx)
        self.tables_host = alloc[UInt64](CODEBOOKS)
        var prefix = weights_dir
        if not prefix.endswith("/"):
            prefix += "/"
        for q in range(CODEBOOKS):
            var path = prefix + "codec_codebook_" + String(q) + ".f32"
            if file_size_bytes(path) != CODEBOOK_SIZE * VQ_DIM * 4:
                raise Error("missing/wrong-sized effective codec codebook " + String(q))
            self.tables_host[q] = load_f32_file_to_gpu(path)
            if self.tables_host[q] == 0:
                raise Error("failed loading effective codec codebook " + String(q))
        var spath = prefix + "codec_rvq_first_output_proj_weight.f32"
        var apath = prefix + "codec_rvq_rest_output_proj_weight.f32"
        if file_size_bytes(spath) != CODE_DIM * VQ_DIM * 4:
            raise Error("missing/wrong-sized semantic RVQ output projection")
        if file_size_bytes(apath) != CODE_DIM * VQ_DIM * 4:
            raise Error("missing/wrong-sized acoustic RVQ output projection")
        self.semantic_proj = load_f32_file_to_gpu(spath)
        self.acoustic_proj = load_f32_file_to_gpu(apath)
        var wpath = prefix + "codec_pre_conv_weight.f32"
        var bpath = prefix + "codec_pre_conv_bias.f32"
        if file_size_bytes(wpath) != LATENT_DIM * CODE_DIM * PRE_KERNEL * 4:
            raise Error("missing/wrong-sized codec pre-conv weight")
        if file_size_bytes(bpath) != LATENT_DIM * 4:
            raise Error("missing/wrong-sized codec pre-conv bias")
        self.preconv_weight = load_f32_file_to_gpu(wpath)
        self.preconv_bias = load_f32_file_to_gpu(bpath)
        var tiw = prefix + "codec_pre_transformer_input_proj_weight.f32"
        var tib = prefix + "codec_pre_transformer_input_proj_bias.f32"
        if file_size_bytes(tiw) != TRANSFORMER_D * LATENT_DIM * 4:
            raise Error("missing/wrong-sized transformer input projection weight")
        if file_size_bytes(tib) != TRANSFORMER_D * 4:
            raise Error("missing/wrong-sized transformer input projection bias")
        self.transformer_input_weight = load_f32_file_to_gpu(tiw)
        self.transformer_input_bias = load_f32_file_to_gpu(tib)
        self.transformer_q = List[UInt64]()
        self.transformer_k = List[UInt64]()
        self.transformer_v = List[UInt64]()
        self.transformer_o = List[UInt64]()
        self.transformer_gate = List[UInt64]()
        self.transformer_up = List[UInt64]()
        self.transformer_down = List[UInt64]()
        self.transformer_in_norm = List[UInt64]()
        self.transformer_post_norm = List[UInt64]()
        self.transformer_attn_scale = List[UInt64]()
        self.transformer_mlp_scale = List[UInt64]()
        for layer in range(TRANSFORMER_LAYERS):
            var lp = prefix + "codec_pre_transformer_layers_" + String(layer) + "_"
            self.transformer_q.append(load_f32_file_to_gpu(lp + "self_attn_q_proj_weight.f32"))
            self.transformer_k.append(load_f32_file_to_gpu(lp + "self_attn_k_proj_weight.f32"))
            self.transformer_v.append(load_f32_file_to_gpu(lp + "self_attn_v_proj_weight.f32"))
            self.transformer_o.append(load_f32_file_to_gpu(lp + "self_attn_o_proj_weight.f32"))
            self.transformer_gate.append(load_f32_file_to_gpu(lp + "mlp_gate_proj_weight.f32"))
            self.transformer_up.append(load_f32_file_to_gpu(lp + "mlp_up_proj_weight.f32"))
            self.transformer_down.append(load_f32_file_to_gpu(lp + "mlp_down_proj_weight.f32"))
            self.transformer_in_norm.append(load_f32_file_to_gpu(lp + "input_layernorm_weight.f32"))
            self.transformer_post_norm.append(load_f32_file_to_gpu(lp + "post_attention_layernorm_weight.f32"))
            self.transformer_attn_scale.append(load_f32_file_to_gpu(lp + "self_attn_layer_scale_scale.f32"))
            self.transformer_mlp_scale.append(load_f32_file_to_gpu(lp + "mlp_layer_scale_scale.f32"))
        self.transformer_final_norm = load_f32_file_to_gpu(prefix + "codec_pre_transformer_norm_weight.f32")
        self.transformer_output_weight = load_f32_file_to_gpu(prefix + "codec_pre_transformer_output_proj_weight.f32")
        self.transformer_output_bias = load_f32_file_to_gpu(prefix + "codec_pre_transformer_output_proj_bias.f32")

    def run_frontend(
        mut self,
        codes_host: UInt64,
        frames: Int,
        quantized_host: UInt64,
        preconv_host: UInt64,
    ) raises:
        if frames <= 0 or frames > 8000:
            raise Error("codec frame count outside [1,8000]")
        var codes_bytes = CODEBOOKS * frames * 8
        var d_codes = cuda_malloc(codes_bytes)
        var d_tables = cuda_malloc(CODEBOOKS * 8)
        var d_semantic = cuda_malloc(VQ_DIM * frames * 4)
        var d_acoustic = cuda_malloc(VQ_DIM * frames * 4)
        var d_quantized = cuda_malloc(CODE_DIM * frames * 4)
        var d_preconv = cuda_malloc(LATENT_DIM * frames * 4)
        cuda_memcpy(d_codes, codes_host, codes_bytes, 1)
        cuda_memcpy(d_tables, UInt64(Int(self.tables_host)), CODEBOOKS * 8, 1)
        var threads = 256
        var lookup_elems = VQ_DIM * frames
        self.ctx.enqueue_function[codec_rvq_lookup_kernel](
            _i64(d_codes),
            UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(d_tables)),
            _f32(d_semantic), _f32(d_acoustic), Int32(frames),
            grid_dim=(lookup_elems + threads - 1) // threads, block_dim=threads,
        )
        var q_elems = CODE_DIM * frames
        self.ctx.enqueue_function[codec_rvq_project_kernel](
            _f32(d_semantic), _f32(d_acoustic),
            _f32(self.semantic_proj), _f32(self.acoustic_proj),
            _f32(d_quantized), Int32(frames),
            grid_dim=(q_elems + threads - 1) // threads, block_dim=threads,
        )
        var p_elems = LATENT_DIM * frames
        self.ctx.enqueue_function[codec_causal_preconv_kernel](
            _f32(d_quantized), _f32(self.preconv_weight), _f32(self.preconv_bias),
            _f32(d_preconv), Int32(frames),
            grid_dim=(p_elems + threads - 1) // threads, block_dim=threads,
        )
        self.ctx.synchronize()
        cuda_memcpy(quantized_host, d_quantized, q_elems * 4, 2)
        cuda_memcpy(preconv_host, d_preconv, p_elems * 4, 2)
        cuda_free(d_codes); cuda_free(d_tables)
        cuda_free(d_semantic); cuda_free(d_acoustic)
        cuda_free(d_quantized); cuda_free(d_preconv)

    def run_transformer_input(
        mut self,
        preconv_host: UInt64,
        frames: Int,
        output_host: UInt64,
    ) raises:
        """Run the canonical [1024,T] -> [T,512] transformer input seam."""
        if frames <= 0 or frames > 8000:
            raise Error("codec frame count outside [1,8000]")
        var in_elems = LATENT_DIM * frames
        var out_elems = TRANSFORMER_D * frames
        var d_channel_major = cuda_malloc(in_elems * 4)
        var d_row_major = cuda_malloc(in_elems * 4)
        var d_output = cuda_malloc(out_elems * 4)
        cuda_memcpy(d_channel_major, preconv_host, in_elems * 4, 1)
        var threads = 256
        var transpose_fn = self.ctx.compile_function[codec_transpose_preconv_kernel]()
        self.ctx.enqueue_function(
            transpose_fn, _f32(d_channel_major), _f32(d_row_major), Int32(frames),
            grid_dim=(in_elems + threads - 1) // threads, block_dim=threads,
        )
        cublas_sgemm(
            self.cublas, d_row_major, self.transformer_input_weight, d_output,
            frames, LATENT_DIM, TRANSFORMER_D,
        )
        var bias_fn = self.ctx.compile_function[codec_add_bias_kernel]()
        self.ctx.enqueue_function(
            bias_fn, _f32(d_output), _f32(self.transformer_input_bias),
            Int32(frames), Int32(TRANSFORMER_D),
            grid_dim=(out_elems + threads - 1) // threads, block_dim=threads,
        )
        self.ctx.synchronize()
        cuda_memcpy(output_host, d_output, out_elems * 4, 2)
        cuda_free(d_output)
        cuda_free(d_row_major)
        cuda_free(d_channel_major)

    def run_transformer(
        mut self,
        input_host: UInt64,
        frames: Int,
        layer0_host: UInt64,
        layer7_host: UInt64,
        output_host: UInt64,
    ) raises:
        """Run the 8-layer causal sliding transformer from [T,512]."""
        if frames <= 0 or frames > 8000:
            raise Error("codec frame count outside [1,8000]")
        var threads = 256
        var hidden_elems = frames * TRANSFORMER_D
        var wide_elems = frames * TRANSFORMER_QD
        var d_x = cuda_malloc(hidden_elems * 4)
        var d_norm = cuda_malloc(hidden_elems * 4)
        var d_q = cuda_malloc(wide_elems * 4)
        var d_k = cuda_malloc(wide_elems * 4)
        var d_v = cuda_malloc(wide_elems * 4)
        var d_attn = cuda_malloc(wide_elems * 4)
        var d_update = cuda_malloc(hidden_elems * 4)
        var d_gate = cuda_malloc(wide_elems * 4)
        var d_up = cuda_malloc(wide_elems * 4)
        cuda_memcpy(d_x, input_host, hidden_elems * 4, 1)
        var rms_fn = self.ctx.compile_function[codec_rmsnorm_kernel]()
        var rope_fn = self.ctx.compile_function[codec_rope_kernel]()
        var attn_fn = self.ctx.compile_function[codec_sliding_attention_kernel]()
        var residual_fn = self.ctx.compile_function[codec_scaled_residual_kernel]()
        var swiglu_fn = self.ctx.compile_function[codec_swiglu_kernel]()
        for layer in range(TRANSFORMER_LAYERS):
            self.ctx.enqueue_function(
                rms_fn, _f32(d_x), _f32(self.transformer_in_norm[layer]), _f32(d_norm),
                Int32(frames), grid_dim=frames, block_dim=1,
            )
            cublas_sgemm(self.cublas, d_norm, self.transformer_q[layer], d_q, frames, TRANSFORMER_D, TRANSFORMER_QD)
            cublas_sgemm(self.cublas, d_norm, self.transformer_k[layer], d_k, frames, TRANSFORMER_D, TRANSFORMER_QD)
            cublas_sgemm(self.cublas, d_norm, self.transformer_v[layer], d_v, frames, TRANSFORMER_D, TRANSFORMER_QD)
            self.ctx.enqueue_function(
                rope_fn, _f32(d_q), _f32(d_k), Int32(frames),
                grid_dim=(frames * TRANSFORMER_HEADS * (TRANSFORMER_HD // 2) + threads - 1) // threads,
                block_dim=threads,
            )
            self.ctx.enqueue_function(
                attn_fn, _f32(d_q), _f32(d_k), _f32(d_v), _f32(d_attn), Int32(frames),
                grid_dim=frames * TRANSFORMER_HEADS, block_dim=TRANSFORMER_WINDOW,
            )
            cublas_sgemm(self.cublas, d_attn, self.transformer_o[layer], d_update, frames, TRANSFORMER_QD, TRANSFORMER_D)
            self.ctx.enqueue_function(
                residual_fn, _f32(d_x), _f32(d_update), _f32(self.transformer_attn_scale[layer]),
                Int32(hidden_elems), grid_dim=(hidden_elems + threads - 1) // threads, block_dim=threads,
            )
            self.ctx.enqueue_function(
                rms_fn, _f32(d_x), _f32(self.transformer_post_norm[layer]), _f32(d_norm),
                Int32(frames), grid_dim=frames, block_dim=1,
            )
            cublas_sgemm(self.cublas, d_norm, self.transformer_gate[layer], d_gate, frames, TRANSFORMER_D, TRANSFORMER_FF)
            cublas_sgemm(self.cublas, d_norm, self.transformer_up[layer], d_up, frames, TRANSFORMER_D, TRANSFORMER_FF)
            self.ctx.enqueue_function(
                swiglu_fn, _f32(d_gate), _f32(d_up), Int32(wide_elems),
                grid_dim=(wide_elems + threads - 1) // threads, block_dim=threads,
            )
            cublas_sgemm(self.cublas, d_gate, self.transformer_down[layer], d_update, frames, TRANSFORMER_FF, TRANSFORMER_D)
            self.ctx.enqueue_function(
                residual_fn, _f32(d_x), _f32(d_update), _f32(self.transformer_mlp_scale[layer]),
                Int32(hidden_elems), grid_dim=(hidden_elems + threads - 1) // threads, block_dim=threads,
            )
            self.ctx.synchronize()
            if layer == 0:
                cuda_memcpy(layer0_host, d_x, hidden_elems * 4, 2)
            if layer == 7:
                cuda_memcpy(layer7_host, d_x, hidden_elems * 4, 2)
        self.ctx.enqueue_function(
            rms_fn, _f32(d_x), _f32(self.transformer_final_norm), _f32(d_norm),
            Int32(frames), grid_dim=frames, block_dim=1,
        )
        cublas_sgemm(
            self.cublas, d_norm, self.transformer_output_weight, d_attn,
            frames, TRANSFORMER_D, LATENT_DIM,
        )
        var bias_fn = self.ctx.compile_function[codec_add_bias_kernel]()
        self.ctx.enqueue_function(
            bias_fn, _f32(d_attn), _f32(self.transformer_output_bias),
            Int32(frames), Int32(LATENT_DIM),
            grid_dim=(frames * LATENT_DIM + threads - 1) // threads, block_dim=threads,
        )
        self.ctx.synchronize()
        cuda_memcpy(output_host, d_attn, frames * LATENT_DIM * 4, 2)
        cuda_free(d_up)
        cuda_free(d_gate)
        cuda_free(d_update)
        cuda_free(d_attn)
        cuda_free(d_v)
        cuda_free(d_k)
        cuda_free(d_q)
        cuda_free(d_norm)
        cuda_free(d_x)

    def free(mut self):
        for q in range(CODEBOOKS):
            if self.tables_host[q] != 0:
                cuda_free(self.tables_host[q])
        self.tables_host.free()
        if self.semantic_proj != 0:
            cuda_free(self.semantic_proj)
        if self.acoustic_proj != 0:
            cuda_free(self.acoustic_proj)
        if self.preconv_weight != 0:
            cuda_free(self.preconv_weight)
        if self.preconv_bias != 0:
            cuda_free(self.preconv_bias)
        if self.transformer_input_weight != 0:
            cuda_free(self.transformer_input_weight)
        if self.transformer_input_bias != 0:
            cuda_free(self.transformer_input_bias)
        for layer in range(TRANSFORMER_LAYERS):
            cuda_free(self.transformer_q[layer])
            cuda_free(self.transformer_k[layer])
            cuda_free(self.transformer_v[layer])
            cuda_free(self.transformer_o[layer])
            cuda_free(self.transformer_gate[layer])
            cuda_free(self.transformer_up[layer])
            cuda_free(self.transformer_down[layer])
            cuda_free(self.transformer_in_norm[layer])
            cuda_free(self.transformer_post_norm[layer])
            cuda_free(self.transformer_attn_scale[layer])
            cuda_free(self.transformer_mlp_scale[layer])
        cuda_free(self.transformer_final_norm)
        cuda_free(self.transformer_output_weight)
        cuda_free(self.transformer_output_bias)
