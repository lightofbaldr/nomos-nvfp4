"""Correctness-first Qwen3TTSTokenizerV2 decoder for Breeze-TTS-2.

The production Breeze runtime requires the bundled Qwen tokenizer decoder;
the top-level Mimi codec is a training-only fallback and is deliberately not
implemented here.  M1 is staged at HF-visible boundaries so every new
machinery class is independently comparable before the waveform stack lands.
"""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import UnsafePointer, alloc
from max.gpu.host import DeviceContext

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.io import file_size_bytes, load_f32_file_to_gpu


comptime CODEBOOKS = 16
comptime CODEBOOK_SIZE = 2048
comptime VQ_DIM = 256
comptime CODE_DIM = 512
comptime LATENT_DIM = 1024
comptime PRE_KERNEL = 3


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


struct BreezeCodec:
    var ctx: DeviceContext
    var tables_host: UnsafePointer[UInt64, MutUntrackedOrigin]
    var semantic_proj: UInt64
    var acoustic_proj: UInt64
    var preconv_weight: UInt64
    var preconv_bias: UInt64

    def __init__(out self, weights_dir: String) raises:
        self.ctx = DeviceContext()
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
