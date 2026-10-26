"""Tier-0 Gemma-4 vision encoder.

Correctness-first BF16 implementation of the Gemma-style 27-layer tower,
dynamic 3x3 spatial pooling, standardization, and text-space projection.
"""

from std.gpu import block_idx, thread_idx, block_dim
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.collections import List
from std.math import exp, log, cos, sin, sqrt, tanh
from layout import row_major, stack_allocation

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.io import load_bf16_file_to_gpu
from lib.cublas import cublas_create, cublas_set_stream, gpu_matmul_bf16_dev_batched


comptime GV_D = 1152
comptime GV_FF = 4304
comptime GV_NH = 16
comptime GV_HD = 72
comptime GV_LAYERS = 27
comptime GV_PATCH_D = 768
comptime GV_OUT_D = 5376
comptime GV_POOL_K = 3
comptime GV_EPS = 1.0e-6
comptime GV_ROPE_THETA = 100.0
comptime GV_POS_SIZE = 10240


def _f32(p: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(p))

def _bf16(p: UInt64) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(p))

def _i32(p: UInt64) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(p))

def _round_bf16(x: Float32) -> Float32:
    return Float32(x.cast[DType.bfloat16]())


def gv_prepare_pixels_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        dst[i] = _round_bf16(Float32(2.0) * (src[i] - Float32(0.5)))


def gv_add_position_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    pos_table: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    positions: UnsafePointer[Int32, MutAnyOrigin], S_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var S = Int(S_arg)
    if i < S * GV_D:
        var row = i // GV_D
        var col = i % GV_D
        var px = Int(positions[row * 2])
        var py = Int(positions[row * 2 + 1])
        # Padding positions suppress only the positional lookup.  The reference
        # retains input_proj(2*(zero_pixel-.5)) until the encoder pool mask.
        if px >= 0 and py >= 0:
            var p = Float32(pos_table[px * GV_D + col])
            p += Float32(pos_table[(GV_POS_SIZE + py) * GV_D + col])
            x[i] = _round_bf16(x[i] + p)


def gv_rmsnorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin], rows_arg: Int32, cols_arg: Int32,
):
    if Int(thread_idx.x) != 0: return
    var row = Int(block_idx.x)
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    if row >= rows: return
    var off = row * cols
    var ss = Float32(0.0)
    for j in range(cols): ss += x[off + j] * x[off + j]
    var inv = Float32(1.0) / sqrt(ss / Float32(cols) + Float32(GV_EPS))
    for j in range(cols):
        # Gemma4Vision's checkpoint stores the effective RMS scale directly
        # (unlike the text-tower converter's zero-centered 1+w convention).
        output[off + j] = _round_bf16(x[off + j] * inv * Float32(weight[j]))


def gv_head_rmsnorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    rows_arg: Int32, with_scale: Int32,
):
    if Int(thread_idx.x) != 0: return
    var hrow = Int(block_idx.x)
    if hrow >= Int(rows_arg) * GV_NH: return
    var off = hrow * GV_HD
    var ss = Float32(0.0)
    for d in range(GV_HD): ss += x[off + d] * x[off + d]
    var inv = Float32(1.0) / sqrt(ss / Float32(GV_HD) + Float32(GV_EPS))
    for d in range(GV_HD):
        var scale = Float32(1.0)
        if with_scale != 0: scale = Float32(weight[d])
        x[off + d] = _round_bf16(x[off + d] * inv * scale)


def gv_rope_2d_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Int32, MutAnyOrigin], S_arg: Int32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var S = Int(S_arg)
    # Each spatial dimension owns 36 channels and rotates two 18-channel halves.
    var total = S * GV_NH * 36
    if idx >= total: return
    var pair = idx % 36
    var t = idx // 36
    var token = t // GV_NH
    var axis = pair // 18
    var freq = pair % 18
    var inv = exp(-(Float32(2 * freq) / Float32(36)) * log(Float32(GV_ROPE_THETA)))
    var angle = Float32(pos[token * 2 + axis]) * inv
    var c = cos(angle); var s = sin(angle)
    var base = t * GV_HD + axis * 36
    var lo = base + freq; var hi = lo + 18
    var qa = q[lo]; var qb = q[hi]; var ka = k[lo]; var kb = k[hi]
    q[lo] = _round_bf16(qa * c - qb * s)
    q[hi] = _round_bf16(qb * c + qa * s)
    k[lo] = _round_bf16(ka * c - kb * s)
    k[hi] = _round_bf16(kb * c + ka * s)


def gv_attention_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Int32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin], S_arg: Int32,
):
    var S = Int(S_arg)
    var bh = Int(block_idx.x)
    var query = bh // GV_NH
    var head = bh % GV_NH
    if query >= S or S > 4096: return
    var tid = Int(thread_idx.x)
    var scores = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[4096]())
    var stats = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[2]())
    var qoff = (query * GV_NH + head) * GV_HD
    var j = tid
    while j < S:
        if pos[j * 2] < 0:
            scores[j] = Float32(-3.4e38)
        else:
            var koff = (j * GV_NH + head) * GV_HD
            var dot = Float32(0.0)
            for d in range(GV_HD): dot += q[qoff + d] * k[koff + d]
            scores[j] = dot
        j += Int(block_dim.x)
    barrier()
    if tid == 0:
        var mx = Float32(-3.4e38)
        for p in range(S):
            if scores[p] > mx: mx = scores[p]
        var den = Float32(0.0)
        for p in range(S): den += exp(scores[p] - mx)
        stats[0] = mx; stats[1] = den
    barrier()
    if tid < GV_HD:
        var acc = Float32(0.0)
        for p in range(S):
            var prob = exp(scores[p] - stats[0]) / stats[1]
            acc += prob * v[(p * GV_NH + head) * GV_HD + tid]
        output[qoff + tid] = _round_bf16(acc)


def gv_postnorm_residual_kernel(
    residual: UnsafePointer[Float32, MutAnyOrigin],
    update: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg): residual[i] = _round_bf16(residual[i] + update[i])


def gv_gated_gelu_kernel(
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_arg):
        var x = gate[i]
        var inner = Float32(0.7978845608028654) * (x + Float32(0.044715) * x*x*x)
        gate[i] = _round_bf16(Float32(0.5) * x * (Float32(1.0) + tanh(inner)) * up[i])


def gv_pool_standardize_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    pos: UnsafePointer[Int32, MutAnyOrigin],
    pool_ids: UnsafePointer[Int32, MutAnyOrigin],
    scale: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin], S_arg: Int32, rows_arg: Int32, width_arg: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rows = Int(rows_arg)
    if i >= rows * GV_D: return
    var row = i // GV_D; var col = i % GV_D
    var pool_id = Int(pool_ids[row]); var width = Int(width_arg)
    var acc = Float32(0.0)
    for p in range(Int(S_arg)):
        var px = Int(pos[p*2]); var py = Int(pos[p*2+1])
        if px >= 0 and py >= 0:
            var pid = (px // GV_POOL_K) + width * (py // GV_POOL_K)
            if pid == pool_id: acc += x[p*GV_D+col]
    acc = acc / Float32(9.0) * sqrt(Float32(GV_D))
    output[i] = _round_bf16((acc - Float32(bias[col])) * Float32(scale[col]))


def gv_scaleless_rmsnorm_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin], rows_arg: Int32,
):
    if Int(thread_idx.x) != 0: return
    var row = Int(block_idx.x)
    if row >= Int(rows_arg): return
    var off = row * GV_D; var ss = Float32(0.0)
    for d in range(GV_D): ss += x[off+d] * x[off+d]
    var inv = Float32(1.0) / sqrt(ss / Float32(GV_D) + Float32(GV_EPS))
    for d in range(GV_D): x[off+d] = _round_bf16(x[off+d] * inv)


struct Gemma4Vision(Movable):
    var ctx: DeviceContext
    var handle: UInt64
    var patch_w: UInt64; var pos_emb: UInt64
    var std_scale: UInt64; var std_bias: UInt64; var embed_w: UInt64
    var qw: List[UInt64]; var kw: List[UInt64]; var vw: List[UInt64]; var ow: List[UInt64]
    var qn: List[UInt64]; var kn: List[UInt64]
    var ni: List[UInt64]; var npa: List[UInt64]; var npf: List[UInt64]; var npo: List[UInt64]
    var gw: List[UInt64]; var uw: List[UInt64]; var dw: List[UInt64]

    def __init__(out self, weights_dir: String) raises:
        self.ctx = DeviceContext(); self.handle = cublas_create(); cublas_set_stream(self.handle,self.ctx)
        var base = weights_dir if weights_dir.endswith("/") else weights_dir+"/"
        self.patch_w=load_bf16_file_to_gpu(base+"patch_embed_w.bf16")
        self.pos_emb=load_bf16_file_to_gpu(base+"pos_emb.bf16")
        self.std_scale=load_bf16_file_to_gpu(base+"std_scale.bf16")
        self.std_bias=load_bf16_file_to_gpu(base+"std_bias.bf16")
        self.embed_w=load_bf16_file_to_gpu(base+"embed_projection_w.bf16")
        self.qw=List[UInt64](); self.kw=List[UInt64](); self.vw=List[UInt64](); self.ow=List[UInt64]()
        self.qn=List[UInt64](); self.kn=List[UInt64](); self.ni=List[UInt64](); self.npa=List[UInt64](); self.npf=List[UInt64](); self.npo=List[UInt64]()
        self.gw=List[UInt64](); self.uw=List[UInt64](); self.dw=List[UInt64]()
        for layer in range(GV_LAYERS):
            var p=base+"vision_L"+String(layer)+"_"
            self.qw.append(load_bf16_file_to_gpu(p+"attn_q_w.bf16")); self.kw.append(load_bf16_file_to_gpu(p+"attn_k_w.bf16"))
            self.vw.append(load_bf16_file_to_gpu(p+"attn_v_w.bf16")); self.ow.append(load_bf16_file_to_gpu(p+"attn_o_w.bf16"))
            self.qn.append(load_bf16_file_to_gpu(p+"attn_q_norm.bf16")); self.kn.append(load_bf16_file_to_gpu(p+"attn_k_norm.bf16"))
            self.ni.append(load_bf16_file_to_gpu(p+"norm_input.bf16")); self.npa.append(load_bf16_file_to_gpu(p+"norm_post_attn.bf16"))
            self.npf.append(load_bf16_file_to_gpu(p+"norm_pre_ff.bf16")); self.npo.append(load_bf16_file_to_gpu(p+"norm_post_ff.bf16"))
            self.gw.append(load_bf16_file_to_gpu(p+"mlp_gate_w.bf16")); self.uw.append(load_bf16_file_to_gpu(p+"mlp_up_w.bf16")); self.dw.append(load_bf16_file_to_gpu(p+"mlp_down_w.bf16"))

    def frontend(mut self, pixels_h: UInt64, pos_h: UInt64, S: Int, patch_h: UInt64) raises:
        if S<=0: raise Error("invalid Gemma4 vision frontend length")
        var n=S*GV_D; var threads=256
        var dp=cuda_malloc(S*GV_PATCH_D*4); var dprep=cuda_malloc(S*GV_PATCH_D*4); var dx=cuda_malloc(n*4); var dpos=cuda_malloc(S*2*4); var dbf=cuda_malloc(S*GV_PATCH_D*2)
        cuda_memcpy(dp,pixels_h,S*GV_PATCH_D*4,1); cuda_memcpy(dpos,pos_h,S*2*4,1)
        var prep=self.ctx.compile_function[gv_prepare_pixels_kernel]()
        self.ctx.enqueue_function(prep,_f32(dp),_f32(dprep),Int32(S*GV_PATCH_D),grid_dim=(S*GV_PATCH_D+threads-1)//threads,block_dim=threads)
        gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dx,dprep,self.patch_w,dbf,S,GV_PATCH_D,GV_D)
        var add=self.ctx.compile_function[gv_add_position_kernel]()
        self.ctx.enqueue_function(add,_f32(dx),_bf16(self.pos_emb),_i32(dpos),Int32(S),grid_dim=(n+threads-1)//threads,block_dim=threads)
        self.ctx.synchronize(); cuda_memcpy(patch_h,dx,n*4,2)
        cuda_free(dbf); cuda_free(dpos); cuda_free(dx); cuda_free(dprep); cuda_free(dp)

    def run(mut self, hidden_h: UInt64, pos_h: UInt64, S: Int, out0: UInt64, out13: UInt64, out26: UInt64, features_h: UInt64, nvis_out: UInt64) raises:
        if S<=0 or S>4096 or S%9!=0: raise Error("invalid Gemma4 vision encoder shape")
        var hostpos=UnsafePointer[Int32,MutUntrackedOrigin](unsafe_from_address=Int(pos_h))
        var maxx=-1
        for p in range(S):
            if Int(hostpos[p*2])>maxx: maxx=Int(hostpos[p*2])
        if maxx<0: raise Error("Gemma4 vision has no valid patches")
        var width=(maxx+1)//GV_POOL_K; var maxrows=S//9
        var ids=List[Int32](); var seen=List[Bool](capacity=maxrows)
        for i in range(maxrows): seen.append(False)
        for p in range(S):
            var px=Int(hostpos[p*2]); var py=Int(hostpos[p*2+1])
            if px>=0 and py>=0:
                var pid=(px//3)+width*(py//3)
                if pid>=0 and pid<maxrows and not seen[pid]: seen[pid]=True
        for pid in range(maxrows):
            if seen[pid]: ids.append(Int32(pid))
        var rows=len(ids)
        if rows<=0: raise Error("Gemma4 vision pooling produced no rows")
        var hout=UnsafePointer[Int32,MutUntrackedOrigin](unsafe_from_address=Int(nvis_out)); hout[0]=Int32(rows)
        var elems=S*GV_D; var threads=256
        var dx=cuda_malloc(elems*4); var dn=cuda_malloc(elems*4); var dq=cuda_malloc(elems*4); var dk=cuda_malloc(elems*4); var dv=cuda_malloc(elems*4)
        var da=cuda_malloc(elems*4); var dp=cuda_malloc(elems*4); var dg=cuda_malloc(S*GV_FF*4); var du=cuda_malloc(S*GV_FF*4)
        var scratch=S*GV_FF
        if S*GV_PATCH_D>scratch: scratch=S*GV_PATCH_D
        var dbf=cuda_malloc(scratch*2); var dpos=cuda_malloc(S*2*4); var dids=cuda_malloc(rows*4)
        cuda_memcpy(dx,hidden_h,elems*4,1); cuda_memcpy(dpos,pos_h,S*2*4,1); cuda_memcpy(dids,UInt64(Int(ids.unsafe_ptr())),rows*4,1)
        var norm=self.ctx.compile_function[gv_rmsnorm_kernel](); var hnorm=self.ctx.compile_function[gv_head_rmsnorm_kernel]()
        var rope=self.ctx.compile_function[gv_rope_2d_kernel](); var attn=self.ctx.compile_function[gv_attention_kernel]()
        var residual=self.ctx.compile_function[gv_postnorm_residual_kernel](); var gated=self.ctx.compile_function[gv_gated_gelu_kernel]()
        for layer in range(GV_LAYERS):
            self.ctx.enqueue_function(norm,_f32(dx),_bf16(self.ni[layer]),_f32(dn),Int32(S),Int32(GV_D),grid_dim=S,block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dq,dn,self.qw[layer],dbf,S,GV_D,GV_D)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dk,dn,self.kw[layer],dbf,S,GV_D,GV_D)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dv,dn,self.vw[layer],dbf,S,GV_D,GV_D)
            self.ctx.enqueue_function(hnorm,_f32(dq),_bf16(self.qn[layer]),Int32(S),Int32(1),grid_dim=S*GV_NH,block_dim=1)
            self.ctx.enqueue_function(hnorm,_f32(dk),_bf16(self.kn[layer]),Int32(S),Int32(1),grid_dim=S*GV_NH,block_dim=1)
            self.ctx.enqueue_function(hnorm,_f32(dv),_bf16(self.qn[layer]),Int32(S),Int32(0),grid_dim=S*GV_NH,block_dim=1)
            self.ctx.enqueue_function(rope,_f32(dq),_f32(dk),_i32(dpos),Int32(S),grid_dim=(S*GV_NH*36+threads-1)//threads,block_dim=threads)
            self.ctx.enqueue_function(attn,_f32(dq),_f32(dk),_f32(dv),_i32(dpos),_f32(da),Int32(S),grid_dim=S*GV_NH,block_dim=128)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dp,da,self.ow[layer],dbf,S,GV_D,GV_D)
            self.ctx.enqueue_function(norm,_f32(dp),_bf16(self.npa[layer]),_f32(dn),Int32(S),Int32(GV_D),grid_dim=S,block_dim=1)
            self.ctx.enqueue_function(residual,_f32(dx),_f32(dn),Int32(elems),grid_dim=(elems+threads-1)//threads,block_dim=threads)
            self.ctx.enqueue_function(norm,_f32(dx),_bf16(self.npf[layer]),_f32(dn),Int32(S),Int32(GV_D),grid_dim=S,block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dg,dn,self.gw[layer],dbf,S,GV_D,GV_FF)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,du,dn,self.uw[layer],dbf,S,GV_D,GV_FF)
            self.ctx.enqueue_function(gated,_f32(dg),_f32(du),Int32(S*GV_FF),grid_dim=(S*GV_FF+threads-1)//threads,block_dim=threads)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dp,dg,self.dw[layer],dbf,S,GV_FF,GV_D)
            self.ctx.enqueue_function(norm,_f32(dp),_bf16(self.npo[layer]),_f32(dn),Int32(S),Int32(GV_D),grid_dim=S,block_dim=1)
            self.ctx.enqueue_function(residual,_f32(dx),_f32(dn),Int32(elems),grid_dim=(elems+threads-1)//threads,block_dim=threads)
            self.ctx.synchronize()
            if layer==0: cuda_memcpy(out0,dx,elems*4,2)
            if layer==13: cuda_memcpy(out13,dx,elems*4,2)
            if layer==26: cuda_memcpy(out26,dx,elems*4,2)
        var dpool=cuda_malloc(rows*GV_D*4); var dfeat=cuda_malloc(rows*GV_OUT_D*4)
        var pool=self.ctx.compile_function[gv_pool_standardize_kernel]()
        self.ctx.enqueue_function(pool,_f32(dx),_i32(dpos),_i32(dids),_bf16(self.std_scale),_bf16(self.std_bias),_f32(dpool),Int32(S),Int32(rows),Int32(width),grid_dim=(rows*GV_D+threads-1)//threads,block_dim=threads)
        var sn=self.ctx.compile_function[gv_scaleless_rmsnorm_kernel]()
        self.ctx.enqueue_function(sn,_f32(dpool),Int32(rows),grid_dim=rows,block_dim=1)
        gpu_matmul_bf16_dev_batched(self.ctx,self.handle,dfeat,dpool,self.embed_w,dbf,rows,GV_D,GV_OUT_D)
        self.ctx.synchronize(); cuda_memcpy(features_h,dfeat,rows*GV_OUT_D*4,2)
        cuda_free(dfeat); cuda_free(dpool); cuda_free(dids); cuda_free(dpos); cuda_free(dbf); cuda_free(du); cuda_free(dg); cuda_free(dp); cuda_free(da); cuda_free(dv); cuda_free(dk); cuda_free(dq); cuda_free(dn); cuda_free(dx)

    def free(mut self):
        cuda_free(self.patch_w); cuda_free(self.pos_emb); cuda_free(self.std_scale); cuda_free(self.std_bias); cuda_free(self.embed_w)
        for i in range(GV_LAYERS):
            cuda_free(self.qw[i]); cuda_free(self.kw[i]); cuda_free(self.vw[i]); cuda_free(self.ow[i]); cuda_free(self.qn[i]); cuda_free(self.kn[i])
            cuda_free(self.ni[i]); cuda_free(self.npa[i]); cuda_free(self.npf[i]); cuda_free(self.npo[i]); cuda_free(self.gw[i]); cuda_free(self.uw[i]); cuda_free(self.dw[i])
