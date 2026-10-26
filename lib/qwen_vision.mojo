"""Tier-0 Qwen3.5/Qwen3.8 vision encoder.

Correctness-first BF16 implementation: patch Linear + host positional rows,
27 full bidirectional ViT blocks, 2D vision RoPE, and 2x2 patch merger.
"""

from std.gpu import block_idx, thread_idx, block_dim
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List
from std.math import exp, log, cos, sin, sqrt, erf, tanh
from layout import row_major, stack_allocation

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.io import load_bf16_file_to_gpu
from lib.cublas import cublas_create, cublas_set_stream, gpu_matmul_bf16_dev_batched


comptime QV_D = 1152
comptime QV_FF = 4304
comptime QV_NH = 16
comptime QV_HD = 72
comptime QV_LAYERS = 27
comptime QV_PATCH_D = 1536
comptime QV_MERGED_D = 4608
comptime QV_OUT_D = 5120
comptime QV_LN_EPS = 1.0e-6
comptime QV_ROPE_THETA = 10000.0


def _f32(p: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(p))

def _bf16(p: UInt64) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(p))

def _i32(p: UInt64) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(p))

def _round_bf16(x: Float32) -> Float32:
    return Float32(x.cast[DType.bfloat16]())


def qv_add_bias_round_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows_arg: Int32, cols_arg: Int32,
):
    var rows = Int(rows_arg); var cols = Int(cols_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < rows * cols:
        x[i] = _round_bf16(x[i] + Float32(bias[i % cols]))


def qv_add_pos_round_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        x[i] = _round_bf16(x[i] + _round_bf16(pos[i]))


def qv_layernorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    rows_arg: Int32, cols_arg: Int32,
):
    if Int(thread_idx.x) != 0: return
    var row = Int(block_idx.x); var rows = Int(rows_arg); var cols = Int(cols_arg)
    if row >= rows: return
    var off = row * cols; var mean = Float32(0.0)
    for j in range(cols): mean += x[off + j]
    mean /= Float32(cols)
    var variance = Float32(0.0)
    for j in range(cols):
        var d = x[off + j] - mean
        variance += d * d
    variance /= Float32(cols)
    var inv = Float32(1.0) / sqrt(variance + Float32(QV_LN_EPS))
    for j in range(cols):
        output[off + j] = _round_bf16(
            (x[off + j] - mean) * inv * Float32(weight[j]) + Float32(bias[j])
        )


def qv_split_qkv_kernel(
    qkv: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    S_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var elems = Int(S_arg) * QV_D
    if i < elems:
        var row = i // QV_D; var col = i % QV_D
        var base = row * 3 * QV_D
        q[i] = qkv[base + col]
        k[i] = qkv[base + QV_D + col]
        v[i] = qkv[base + 2 * QV_D + col]


def qv_rope_2d_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Int32, MutAnyOrigin],
    S_arg: Int32,
):
    var S = Int(S_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var half = QV_HD // 2
    var total = S * QV_NH * half
    if idx >= total: return
    var d = idx % half; var t = idx // half
    var token = t // QV_NH
    # rotary_pos_emb(head_dim//2=36) receives [h,w], yielding 18 freqs
    # per axis; cat(emb,emb) then rotate_half pairs the 36-d halves.
    var axis = 0 if d < 18 else 1
    var freq = d if d < 18 else d - 18
    var inv = exp(-(Float32(2 * freq) / Float32(half)) * log(Float32(QV_ROPE_THETA)))
    var angle = Float32(pos[token * 2 + axis]) * inv
    var c = cos(angle); var s = sin(angle)
    var off = t * QV_HD; var lo = off + d; var hi = lo + half
    var qa = q[lo]; var qb = q[hi]; var ka = k[lo]; var kb = k[hi]
    q[lo] = _round_bf16(qa * c - qb * s)
    q[hi] = _round_bf16(qb * c + qa * s)
    k[lo] = _round_bf16(ka * c - kb * s)
    k[hi] = _round_bf16(kb * c + ka * s)


def qv_scale_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32, scale: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg): x[i] *= scale


def qv_attention_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    start_arg: Int32, length_arg: Int32,
):
    var start = Int(start_arg); var length = Int(length_arg)
    var bh = Int(block_idx.x); var q_local = bh // QV_NH; var head = bh % QV_NH
    if q_local >= length or length > 4096: return
    var tid = Int(thread_idx.x)
    var scores = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[4096]())
    var stats = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[2]())
    var q_abs = start + q_local; var qoff = (q_abs * QV_NH + head) * QV_HD
    var j = tid
    while j < length:
        var koff = ((start + j) * QV_NH + head) * QV_HD
        var dot = Float32(0.0)
        for d in range(QV_HD): dot += q[qoff + d] * k[koff + d]
        scores[j] = dot
        j += Int(block_dim.x)
    barrier()
    if tid == 0:
        var mx = Float32(-3.4e38)
        for p in range(length):
            if scores[p] > mx: mx = scores[p]
        var den = Float32(0.0)
        for p in range(length): den += exp(scores[p] - mx)
        stats[0] = mx; stats[1] = den
    barrier()
    if tid < QV_HD:
        var acc = Float32(0.0)
        for p in range(length):
            var prob = exp(scores[p] - stats[0]) / stats[1]
            acc += prob * v[((start + p) * QV_NH + head) * QV_HD + tid]
        output[(q_abs * QV_NH + head) * QV_HD + tid] = _round_bf16(acc)


def qv_residual_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    update: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg): x[i] = _round_bf16(x[i] + update[i])


def qv_gelu_tanh_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        var v = x[i]
        var inner = Float32(0.7978845608028654) * (v + Float32(0.044715) * v * v * v)
        x[i] = _round_bf16(Float32(0.5) * v * (Float32(1.0) + tanh(inner)))


def qv_gelu_exact_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        var v = x[i]
        x[i] = _round_bf16(Float32(0.5) * v * (Float32(1.0) + erf(v / Float32(1.4142135623730951))))


def _launch(ctx: DeviceContext, kind: Int, a: UInt64, b: UInt64, n: Int) raises:
    var threads = 256; var blocks = (n + threads - 1) // threads
    if kind == 0:
        var kern = ctx.compile_function[qv_residual_kernel]()
        ctx.enqueue_function(kern, _f32(a), _f32(b), Int32(n), grid_dim=blocks, block_dim=threads)
    elif kind == 1:
        var kern = ctx.compile_function[qv_gelu_tanh_kernel]()
        ctx.enqueue_function(kern, _f32(a), Int32(n), grid_dim=blocks, block_dim=threads)
    else:
        var kern = ctx.compile_function[qv_gelu_exact_kernel]()
        ctx.enqueue_function(kern, _f32(a), Int32(n), grid_dim=blocks, block_dim=threads)


struct QwenVision(Movable):
    var ctx: DeviceContext
    var handle: UInt64
    var patch_w: UInt64; var patch_b: UInt64
    var qkv_w: List[UInt64]; var qkv_b: List[UInt64]
    var o_w: List[UInt64]; var o_b: List[UInt64]
    var fc1_w: List[UInt64]; var fc1_b: List[UInt64]
    var fc2_w: List[UInt64]; var fc2_b: List[UInt64]
    var n1_w: List[UInt64]; var n1_b: List[UInt64]
    var n2_w: List[UInt64]; var n2_b: List[UInt64]
    var merge_nw: UInt64; var merge_nb: UInt64
    var merge_f1w: UInt64; var merge_f1b: UInt64
    var merge_f2w: UInt64; var merge_f2b: UInt64

    def __init__(out self, weights_dir: String) raises:
        self.ctx = DeviceContext(); self.handle = cublas_create(); cublas_set_stream(self.handle, self.ctx)
        var base = weights_dir if weights_dir.endswith("/") else weights_dir + "/"
        self.patch_w = load_bf16_file_to_gpu(base + "patch_embed_w.bf16")
        self.patch_b = load_bf16_file_to_gpu(base + "patch_embed_b.bf16")
        self.qkv_w = List[UInt64](); self.qkv_b = List[UInt64]()
        self.o_w = List[UInt64](); self.o_b = List[UInt64]()
        self.fc1_w = List[UInt64](); self.fc1_b = List[UInt64]()
        self.fc2_w = List[UInt64](); self.fc2_b = List[UInt64]()
        self.n1_w = List[UInt64](); self.n1_b = List[UInt64]()
        self.n2_w = List[UInt64](); self.n2_b = List[UInt64]()
        for layer in range(QV_LAYERS):
            var p = base + "vision_L" + String(layer) + "_"
            self.qkv_w.append(load_bf16_file_to_gpu(p + "attn_qkv_w.bf16")); self.qkv_b.append(load_bf16_file_to_gpu(p + "attn_qkv_b.bf16"))
            self.o_w.append(load_bf16_file_to_gpu(p + "attn_o_w.bf16")); self.o_b.append(load_bf16_file_to_gpu(p + "attn_o_b.bf16"))
            self.fc1_w.append(load_bf16_file_to_gpu(p + "mlp_fc1_w.bf16")); self.fc1_b.append(load_bf16_file_to_gpu(p + "mlp_fc1_b.bf16"))
            self.fc2_w.append(load_bf16_file_to_gpu(p + "mlp_fc2_w.bf16")); self.fc2_b.append(load_bf16_file_to_gpu(p + "mlp_fc2_b.bf16"))
            self.n1_w.append(load_bf16_file_to_gpu(p + "norm1_w.bf16")); self.n1_b.append(load_bf16_file_to_gpu(p + "norm1_b.bf16"))
            self.n2_w.append(load_bf16_file_to_gpu(p + "norm2_w.bf16")); self.n2_b.append(load_bf16_file_to_gpu(p + "norm2_b.bf16"))
        self.merge_nw = load_bf16_file_to_gpu(base + "merger_norm_w.bf16"); self.merge_nb = load_bf16_file_to_gpu(base + "merger_norm_b.bf16")
        self.merge_f1w = load_bf16_file_to_gpu(base + "merger_fc1_w.bf16"); self.merge_f1b = load_bf16_file_to_gpu(base + "merger_fc1_b.bf16")
        self.merge_f2w = load_bf16_file_to_gpu(base + "merger_fc2_w.bf16"); self.merge_f2b = load_bf16_file_to_gpu(base + "merger_fc2_b.bf16")

    def run_frontend(
        mut self, pixels_host: UInt64, pos_host: UInt64, S: Int,
        patch_host: UInt64, block_in_host: UInt64,
    ) raises:
        if S <= 0: raise Error("invalid qwen vision frontend length")
        var elems = S * QV_D; var threads = 256
        var d_pix = cuda_malloc(S * QV_PATCH_D * 4); var d_x = cuda_malloc(elems * 4)
        var d_pos = cuda_malloc(elems * 4); var d_bf = cuda_malloc(S * QV_PATCH_D * 2)
        cuda_memcpy(d_pix, pixels_host, S * QV_PATCH_D * 4, 1); cuda_memcpy(d_pos, pos_host, elems * 4, 1)
        gpu_matmul_bf16_dev_batched(self.ctx, self.handle, d_x, d_pix, self.patch_w, d_bf, S, QV_PATCH_D, QV_D)
        var bias_fn = self.ctx.compile_function[qv_add_bias_round_kernel]()
        self.ctx.enqueue_function(bias_fn, _f32(d_x), _bf16(self.patch_b), Int32(S), Int32(QV_D), grid_dim=(elems+threads-1)//threads, block_dim=threads)
        self.ctx.synchronize(); cuda_memcpy(patch_host, d_x, elems * 4, 2)
        var pos_fn = self.ctx.compile_function[qv_add_pos_round_kernel]()
        self.ctx.enqueue_function(pos_fn, _f32(d_x), _f32(d_pos), Int32(elems), grid_dim=(elems+threads-1)//threads, block_dim=threads)
        self.ctx.synchronize(); cuda_memcpy(block_in_host, d_x, elems * 4, 2)
        cuda_free(d_bf); cuda_free(d_pos); cuda_free(d_x); cuda_free(d_pix)

    def run_encoder(
        mut self, hidden_host: UInt64, pos_host: UInt64, cu_host: UInt64,
        n_segments: Int, S: Int, out0: UInt64, out13: UInt64, out26: UInt64,
        features_host: UInt64,
    ) raises:
        if S <= 0 or S % 4 != 0 or n_segments <= 0: raise Error("invalid qwen vision encoder shape")
        var elems = S * QV_D; var rows = S // 4; var threads = 256
        var d_x = cuda_malloc(elems*4); var d_norm = cuda_malloc(elems*4)
        var d_qkv = cuda_malloc(S*3*QV_D*4); var d_q = cuda_malloc(elems*4); var d_k = cuda_malloc(elems*4); var d_v = cuda_malloc(elems*4)
        var d_attn = cuda_malloc(elems*4); var d_proj = cuda_malloc(elems*4); var d_mlp = cuda_malloc(S*QV_FF*4)
        var scratch_elems = S*QV_PATCH_D
        if S*QV_FF > scratch_elems: scratch_elems = S*QV_FF
        if rows*QV_MERGED_D > scratch_elems: scratch_elems = rows*QV_MERGED_D
        var d_bf = cuda_malloc(scratch_elems*2); var d_pos = cuda_malloc(S*2*4)
        cuda_memcpy(d_x, hidden_host, elems*4, 1); cuda_memcpy(d_pos, pos_host, S*2*4, 1)
        var cu = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(cu_host))
        for layer in range(QV_LAYERS):
            var ln1 = self.ctx.compile_function[qv_layernorm_kernel]()
            self.ctx.enqueue_function(ln1, _f32(d_x), _bf16(self.n1_w[layer]), _bf16(self.n1_b[layer]), _f32(d_norm), Int32(S), Int32(QV_D), grid_dim=S, block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,d_qkv,d_norm,self.qkv_w[layer],d_bf,S,QV_D,3*QV_D)
            var bias_fn = self.ctx.compile_function[qv_add_bias_round_kernel]()
            self.ctx.enqueue_function(bias_fn,_f32(d_qkv),_bf16(self.qkv_b[layer]),Int32(S),Int32(3*QV_D),grid_dim=(S*3*QV_D+threads-1)//threads,block_dim=threads)
            var split_fn = self.ctx.compile_function[qv_split_qkv_kernel]()
            self.ctx.enqueue_function(split_fn,_f32(d_qkv),_f32(d_q),_f32(d_k),_f32(d_v),Int32(S),grid_dim=(elems+threads-1)//threads,block_dim=threads)
            var rope_fn = self.ctx.compile_function[qv_rope_2d_kernel]()
            self.ctx.enqueue_function(rope_fn,_f32(d_q),_f32(d_k),_i32(d_pos),Int32(S),grid_dim=(S*QV_NH*(QV_HD//2)+threads-1)//threads,block_dim=threads)
            var scale_fn = self.ctx.compile_function[qv_scale_kernel]()
            self.ctx.enqueue_function(scale_fn,_f32(d_q),Int32(elems),Float32(1.0)/sqrt(Float32(QV_HD)),grid_dim=(elems+threads-1)//threads,block_dim=threads)
            for seg in range(n_segments):
                var start=Int(cu[seg]); var length=Int(cu[seg+1])-start
                if length > 0:
                    var attn_fn = self.ctx.compile_function[qv_attention_kernel]()
                    self.ctx.enqueue_function(attn_fn,_f32(d_q),_f32(d_k),_f32(d_v),_f32(d_attn),Int32(start),Int32(length),grid_dim=length*QV_NH,block_dim=128)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,d_proj,d_attn,self.o_w[layer],d_bf,S,QV_D,QV_D)
            self.ctx.enqueue_function(bias_fn,_f32(d_proj),_bf16(self.o_b[layer]),Int32(S),Int32(QV_D),grid_dim=(elems+threads-1)//threads,block_dim=threads)
            _launch(self.ctx,0,d_x,d_proj,elems)
            var ln2 = self.ctx.compile_function[qv_layernorm_kernel]()
            self.ctx.enqueue_function(ln2,_f32(d_x),_bf16(self.n2_w[layer]),_bf16(self.n2_b[layer]),_f32(d_norm),Int32(S),Int32(QV_D),grid_dim=S,block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,d_mlp,d_norm,self.fc1_w[layer],d_bf,S,QV_D,QV_FF)
            self.ctx.enqueue_function(bias_fn,_f32(d_mlp),_bf16(self.fc1_b[layer]),Int32(S),Int32(QV_FF),grid_dim=(S*QV_FF+threads-1)//threads,block_dim=threads)
            _launch(self.ctx,1,d_mlp,0,S*QV_FF)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,d_proj,d_mlp,self.fc2_w[layer],d_bf,S,QV_FF,QV_D)
            self.ctx.enqueue_function(bias_fn,_f32(d_proj),_bf16(self.fc2_b[layer]),Int32(S),Int32(QV_D),grid_dim=(elems+threads-1)//threads,block_dim=threads)
            _launch(self.ctx,0,d_x,d_proj,elems)
            self.ctx.synchronize()
            if layer == 0: cuda_memcpy(out0,d_x,elems*4,2)
            if layer == 13: cuda_memcpy(out13,d_x,elems*4,2)
            if layer == 26: cuda_memcpy(out26,d_x,elems*4,2)

        # Merger norm is applied per patch before contiguous groups of four
        # patches are viewed as [rows,4608].
        var mln = self.ctx.compile_function[qv_layernorm_kernel]()
        self.ctx.enqueue_function(mln,_f32(d_x),_bf16(self.merge_nw),_bf16(self.merge_nb),_f32(d_norm),Int32(S),Int32(QV_D),grid_dim=S,block_dim=1)
        var d_merge = cuda_malloc(rows*QV_MERGED_D*4); var d_features = cuda_malloc(rows*QV_OUT_D*4)
        gpu_matmul_bf16_dev_batched(self.ctx,self.handle,d_merge,d_norm,self.merge_f1w,d_bf,rows,QV_MERGED_D,QV_MERGED_D)
        var bias_fn2 = self.ctx.compile_function[qv_add_bias_round_kernel]()
        self.ctx.enqueue_function(bias_fn2,_f32(d_merge),_bf16(self.merge_f1b),Int32(rows),Int32(QV_MERGED_D),grid_dim=(rows*QV_MERGED_D+threads-1)//threads,block_dim=threads)
        _launch(self.ctx,2,d_merge,0,rows*QV_MERGED_D)
        gpu_matmul_bf16_dev_batched(self.ctx,self.handle,d_features,d_merge,self.merge_f2w,d_bf,rows,QV_MERGED_D,QV_OUT_D)
        self.ctx.enqueue_function(bias_fn2,_f32(d_features),_bf16(self.merge_f2b),Int32(rows),Int32(QV_OUT_D),grid_dim=(rows*QV_OUT_D+threads-1)//threads,block_dim=threads)
        self.ctx.synchronize(); cuda_memcpy(features_host,d_features,rows*QV_OUT_D*4,2)
        cuda_free(d_features); cuda_free(d_merge); cuda_free(d_pos); cuda_free(d_bf); cuda_free(d_mlp); cuda_free(d_proj); cuda_free(d_attn)
        cuda_free(d_v); cuda_free(d_k); cuda_free(d_q); cuda_free(d_qkv); cuda_free(d_norm); cuda_free(d_x)

    def free(mut self):
        cuda_free(self.patch_w); cuda_free(self.patch_b)
        for i in range(QV_LAYERS):
            cuda_free(self.qkv_w[i]); cuda_free(self.qkv_b[i]); cuda_free(self.o_w[i]); cuda_free(self.o_b[i])
            cuda_free(self.fc1_w[i]); cuda_free(self.fc1_b[i]); cuda_free(self.fc2_w[i]); cuda_free(self.fc2_b[i])
            cuda_free(self.n1_w[i]); cuda_free(self.n1_b[i]); cuda_free(self.n2_w[i]); cuda_free(self.n2_b[i])
        cuda_free(self.merge_nw); cuda_free(self.merge_nb); cuda_free(self.merge_f1w); cuda_free(self.merge_f1b); cuda_free(self.merge_f2w); cuda_free(self.merge_f2b)
