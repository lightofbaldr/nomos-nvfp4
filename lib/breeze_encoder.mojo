"""Correctness-first Breeze T5Gemma2 text encoder."""

from std.gpu import block_idx, block_dim, thread_idx
from std.collections import List
from std.memory import UnsafePointer
from std.math import exp, log, cos, sin, sqrt, tanh
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from layout import row_major, stack_allocation

from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy
from lib.io import load_bf16_file_to_gpu
from lib.cublas import cublas_create, cublas_set_stream, gpu_matmul_bf16_dev_batched

comptime BE_D = 1152
comptime BE_FF = 6912
comptime BE_QD = 1024
comptime BE_KVD = 256
comptime BE_HD = 256
comptime BE_NH = 4
comptime BE_LAYERS = 26
comptime BE_VOCAB = 262158
comptime BE_EOI = 256000
comptime BE_WINDOW = 512
comptime BE_EPS = 1.0e-6

def _ef32(p: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(p))
def _ebf16(p: UInt64) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(p))
def _ei64(p: UInt64) -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(p))
def _eround(x: Float32) -> Float32:
    return Float32(x.cast[DType.bfloat16]())

def be_embed_kernel(ids: UnsafePointer[Int64, MutAnyOrigin], table: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], eoi: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], output: UnsafePointer[Float32, MutAnyOrigin], S_arg: Int32):
    var S=Int(S_arg); var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < S*BE_D:
        var row=i//BE_D; var d=i%BE_D; var token=Int(ids[row])
        var v=Float32(0.0)
        if token == BE_EOI: v=Float32(eoi[d])
        elif token >= 0 and token < BE_VOCAB: v=Float32(table[token*BE_D+d])
        output[i]=_eround(v*sqrt(Float32(BE_D)))

def be_round_kernel(x: UnsafePointer[Float32, MutAnyOrigin], n_arg: Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n_arg): x[i]=_eround(x[i])

def be_rmsnorm_kernel(x: UnsafePointer[Float32, MutAnyOrigin], w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], output: UnsafePointer[Float32, MutAnyOrigin], rows_arg:Int32, cols_arg:Int32):
    if Int(thread_idx.x)!=0: return
    var row=Int(block_idx.x); var rows=Int(rows_arg); var cols=Int(cols_arg)
    if row>=rows: return
    var off=row*cols; var ss=Float32(0.0)
    for d in range(cols):
        var v=x[off+d]; ss+=v*v
    var inv=Float32(1.0)/sqrt(ss/Float32(cols)+Float32(BE_EPS))
    for d in range(cols): output[off+d]=_eround(x[off+d]*inv*(Float32(1.0)+Float32(w[d])))

def be_rope_kernel(x: UnsafePointer[Float32, MutAnyOrigin], S_arg:Int32, heads_arg:Int32, theta:Float32, factor:Float32):
    var S=Int(S_arg); var heads=Int(heads_arg); var half=BE_HD//2
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=S*heads*half: return
    var pair=i%half; var th=i//half; var pos=th//heads; var off=th*BE_HD
    var inv=exp(-(Float32(2*pair)/Float32(BE_HD))*log(theta))/factor
    var angle=Float32(pos)*inv; var c=cos(angle); var s=sin(angle)
    var lo=off+pair; var hi=lo+half; var a=x[lo]; var b=x[hi]
    x[lo]=_eround(a*c-b*s); x[hi]=_eround(b*c+a*s)

def be_attention_kernel(q:UnsafePointer[Float32,MutAnyOrigin], k:UnsafePointer[Float32,MutAnyOrigin], v:UnsafePointer[Float32,MutAnyOrigin], output:UnsafePointer[Float32,MutAnyOrigin], S_arg:Int32, sliding_arg:Int32):
    var S=Int(S_arg); var sliding=Int(sliding_arg); var qh=Int(block_idx.x); var row=qh//BE_NH; var head=qh%BE_NH
    if row>=S:return
    var first=0; var last=S
    if sliding!=0:
        first=row-255
        if first<0:first=0
        last=row+257
        if last>S:last=S
    var count=last-first; var tid=Int(thread_idx.x)
    var scores=stack_allocation[DType.float32,address_space=AddressSpace.SHARED](row_major[2048]())
    var stats=stack_allocation[DType.float32,address_space=AddressSpace.SHARED](row_major[2]())
    var j=tid
    while j<count:
        var dot=Float32(0.0); var qo=(row*BE_NH+head)*BE_HD; var ko=(first+j)*BE_HD
        for d in range(BE_HD): dot+=q[qo+d]*k[ko+d]
        scores[j]=dot/Float32(16.0); j+=Int(block_dim.x)
    barrier()
    if tid==0:
        var mx=Float32(-3.4e38)
        for p in range(count):
            if scores[p]>mx:mx=scores[p]
        var den=Float32(0.0)
        for p in range(count):den+=exp(scores[p]-mx)
        stats[0]=mx;stats[1]=den
    barrier()
    if tid<BE_HD:
        var acc=Float32(0.0)
        for p in range(count): acc+=(exp(scores[p]-stats[0])/stats[1])*v[(first+p)*BE_HD+tid]
        output[(row*BE_NH+head)*BE_HD+tid]=_eround(acc)

def be_add_kernel(x:UnsafePointer[Float32,MutAnyOrigin], u:UnsafePointer[Float32,MutAnyOrigin], n_arg:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n_arg):x[i]=_eround(x[i]+u[i])

def be_gated_gelu_kernel(g:UnsafePointer[Float32,MutAnyOrigin], u:UnsafePointer[Float32,MutAnyOrigin], n_arg:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n_arg):
        var x=g[i]; var inner=Float32(0.7978845608028654)*(x+Float32(0.044715)*x*x*x)
        g[i]=_eround(Float32(0.5)*x*(Float32(1.0)+tanh(inner))*u[i])

struct BreezeEncoder(Movable):
    var ctx:DeviceContext; var handle:UInt64; var embed:UInt64; var eoi:UInt64
    var q:List[UInt64];var k:List[UInt64];var v:List[UInt64];var o:List[UInt64]
    var qn:List[UInt64];var kn:List[UInt64]
    var nprea:List[UInt64];var nposta:List[UInt64];var npref:List[UInt64];var npostf:List[UInt64]
    var gate:List[UInt64];var up:List[UInt64];var down:List[UInt64];var final_norm:UInt64
    def __init__(out self,weights_dir:String) raises:
        self.ctx=DeviceContext();self.handle=cublas_create();cublas_set_stream(self.handle,self.ctx)
        var b=weights_dir if weights_dir.endswith("/") else weights_dir+"/"
        self.embed=load_bf16_file_to_gpu(b+"text_encoder_embed_tokens_weight.bf16");self.eoi=load_bf16_file_to_gpu(b+"text_encoder_embed_tokens_eoi_embedding.bf16")
        self.q=List[UInt64]();self.k=List[UInt64]();self.v=List[UInt64]();self.o=List[UInt64]();self.qn=List[UInt64]();self.kn=List[UInt64]()
        self.nprea=List[UInt64]();self.nposta=List[UInt64]();self.npref=List[UInt64]();self.npostf=List[UInt64]()
        self.gate=List[UInt64]();self.up=List[UInt64]();self.down=List[UInt64]()
        for l in range(BE_LAYERS):
            var p=b+"text_encoder_layers_"+String(l)+"_"
            self.q.append(load_bf16_file_to_gpu(p+"self_attn_q_proj_weight.bf16"));self.k.append(load_bf16_file_to_gpu(p+"self_attn_k_proj_weight.bf16"));self.v.append(load_bf16_file_to_gpu(p+"self_attn_v_proj_weight.bf16"));self.o.append(load_bf16_file_to_gpu(p+"self_attn_o_proj_weight.bf16"))
            self.qn.append(load_bf16_file_to_gpu(p+"self_attn_q_norm_weight.bf16"));self.kn.append(load_bf16_file_to_gpu(p+"self_attn_k_norm_weight.bf16"))
            self.nprea.append(load_bf16_file_to_gpu(p+"pre_self_attn_layernorm_weight.bf16"));self.nposta.append(load_bf16_file_to_gpu(p+"post_self_attn_layernorm_weight.bf16"));self.npref.append(load_bf16_file_to_gpu(p+"pre_feedforward_layernorm_weight.bf16"));self.npostf.append(load_bf16_file_to_gpu(p+"post_feedforward_layernorm_weight.bf16"))
            self.gate.append(load_bf16_file_to_gpu(p+"mlp_gate_proj_weight.bf16"));self.up.append(load_bf16_file_to_gpu(p+"mlp_up_proj_weight.bf16"));self.down.append(load_bf16_file_to_gpu(p+"mlp_down_proj_weight.bf16"))
        self.final_norm=load_bf16_file_to_gpu(b+"text_encoder_norm_weight.bf16")

    def run(mut self,ids_host:UInt64,S:Int,layers_host:UInt64,final_host:UInt64) raises:
        if S<=0 or S>2048:raise Error("invalid Breeze encoder length")
        var threads=256;var he=S*BE_D;var qe=S*BE_QD;var ke=S*BE_KVD;var fe=S*BE_FF
        var di=cuda_malloc(S*8);var x=cuda_malloc(he*4);var n=cuda_malloc(he*4);var q=cuda_malloc(qe*4);var k=cuda_malloc(ke*4);var v=cuda_malloc(ke*4);var a=cuda_malloc(qe*4);var u=cuda_malloc(he*4);var g=cuda_malloc(fe*4);var upv=cuda_malloc(fe*4);var bf=cuda_malloc(fe*2)
        cuda_memcpy(di,ids_host,S*8,1)
        var ef=self.ctx.compile_function[be_embed_kernel]();self.ctx.enqueue_function(ef,_ei64(di),_ebf16(self.embed),_ebf16(self.eoi),_ef32(x),Int32(S),grid_dim=(he+threads-1)//threads,block_dim=threads)
        var rf=self.ctx.compile_function[be_rmsnorm_kernel]();var rof=self.ctx.compile_function[be_rope_kernel]();var af=self.ctx.compile_function[be_attention_kernel]();var addf=self.ctx.compile_function[be_add_kernel]();var gf=self.ctx.compile_function[be_gated_gelu_kernel]();var roundf=self.ctx.compile_function[be_round_kernel]()
        for l in range(BE_LAYERS):
            self.ctx.enqueue_function(rf,_ef32(x),_ebf16(self.nprea[l]),_ef32(n),Int32(S),Int32(BE_D),grid_dim=S,block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,q,n,self.q[l],bf,S,BE_D,BE_QD);gpu_matmul_bf16_dev_batched(self.ctx,self.handle,k,n,self.k[l],bf,S,BE_D,BE_KVD);gpu_matmul_bf16_dev_batched(self.ctx,self.handle,v,n,self.v[l],bf,S,BE_D,BE_KVD)
            self.ctx.enqueue_function(rf,_ef32(q),_ebf16(self.qn[l]),_ef32(q),Int32(S*BE_NH),Int32(BE_HD),grid_dim=S*BE_NH,block_dim=1);self.ctx.enqueue_function(rf,_ef32(k),_ebf16(self.kn[l]),_ef32(k),Int32(S),Int32(BE_HD),grid_dim=S,block_dim=1)
            var full=1 if (l+1)%6==0 else 0;var theta=Float32(1000000.0) if full==1 else Float32(10000.0);var factor=Float32(8.0) if full==1 else Float32(1.0)
            self.ctx.enqueue_function(rof,_ef32(q),Int32(S),Int32(BE_NH),theta,factor,grid_dim=(S*BE_NH*(BE_HD//2)+threads-1)//threads,block_dim=threads);self.ctx.enqueue_function(rof,_ef32(k),Int32(S),Int32(1),theta,factor,grid_dim=(S*(BE_HD//2)+threads-1)//threads,block_dim=threads)
            self.ctx.enqueue_function(af,_ef32(q),_ef32(k),_ef32(v),_ef32(a),Int32(S),Int32(0 if full==1 else 1),grid_dim=S*BE_NH,block_dim=256)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,u,a,self.o[l],bf,S,BE_QD,BE_D);self.ctx.enqueue_function(rf,_ef32(u),_ebf16(self.nposta[l]),_ef32(u),Int32(S),Int32(BE_D),grid_dim=S,block_dim=1);self.ctx.enqueue_function(addf,_ef32(x),_ef32(u),Int32(he),grid_dim=(he+threads-1)//threads,block_dim=threads)
            self.ctx.enqueue_function(rf,_ef32(x),_ebf16(self.npref[l]),_ef32(n),Int32(S),Int32(BE_D),grid_dim=S,block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,g,n,self.gate[l],bf,S,BE_D,BE_FF);gpu_matmul_bf16_dev_batched(self.ctx,self.handle,upv,n,self.up[l],bf,S,BE_D,BE_FF);self.ctx.enqueue_function(gf,_ef32(g),_ef32(upv),Int32(fe),grid_dim=(fe+threads-1)//threads,block_dim=threads)
            gpu_matmul_bf16_dev_batched(self.ctx,self.handle,u,g,self.down[l],bf,S,BE_FF,BE_D);self.ctx.enqueue_function(rf,_ef32(u),_ebf16(self.npostf[l]),_ef32(u),Int32(S),Int32(BE_D),grid_dim=S,block_dim=1);self.ctx.enqueue_function(addf,_ef32(x),_ef32(u),Int32(he),grid_dim=(he+threads-1)//threads,block_dim=threads)
            self.ctx.synchronize();cuda_memcpy(layers_host+UInt64(l*he*4),x,he*4,2)
        self.ctx.enqueue_function(rf,_ef32(x),_ebf16(self.final_norm),_ef32(n),Int32(S),Int32(BE_D),grid_dim=S,block_dim=1);self.ctx.synchronize();cuda_memcpy(final_host,n,he*4,2)
        cuda_free(bf);cuda_free(upv);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n);cuda_free(x);cuda_free(di)

    def free(mut self):
        cuda_free(self.embed);cuda_free(self.eoi)
        for l in range(BE_LAYERS):
            cuda_free(self.q[l]);cuda_free(self.k[l]);cuda_free(self.v[l]);cuda_free(self.o[l]);cuda_free(self.qn[l]);cuda_free(self.kn[l]);cuda_free(self.nprea[l]);cuda_free(self.nposta[l]);cuda_free(self.npref[l]);cuda_free(self.npostf[l]);cuda_free(self.gate[l]);cuda_free(self.up[l]);cuda_free(self.down[l])
        cuda_free(self.final_norm)
