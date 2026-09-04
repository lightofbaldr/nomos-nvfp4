"""Correctness-first Breeze backbone + depth decoder (two independent lanes)."""

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

comptime BB_D=2048
comptime BB_FF=6144
comptime BB_QD=2048
comptime BB_KVD=1024
comptime BB_HD=128
comptime BB_NH=16
comptime BB_NKV=8
comptime BB_L=28
comptime BB_V=2052
comptime DD_D=1024
comptime DD_FF=8192
comptime DD_QD=1024
comptime DD_KVD=256
comptime DD_HD=128
comptime DD_NH=8
comptime DD_NKV=2
comptime DD_L=12
comptime DD_V=2051
comptime CB=16
comptime MAX_SEQ=256

def _mf32(p:UInt64)->UnsafePointer[Float32,MutAnyOrigin]: return UnsafePointer[Float32,MutAnyOrigin](unsafe_from_address=Int(p))
def _mbf16(p:UInt64)->UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin]: return UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin](unsafe_from_address=Int(p))
def _mi64(p:UInt64)->UnsafePointer[Int64,MutAnyOrigin]: return UnsafePointer[Int64,MutAnyOrigin](unsafe_from_address=Int(p))
def _mr(x:Float32)->Float32: return Float32(x.cast[DType.bfloat16]())

def bm_copy_f32(src:UnsafePointer[Float32,MutAnyOrigin],dst:UnsafePointer[Float32,MutAnyOrigin],n:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n): dst[i]=src[i]

def bm_rms(x:UnsafePointer[Float32,MutAnyOrigin],w:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin],rows:Int32,cols:Int32,eps:Float32):
    if Int(thread_idx.x)!=0:return
    var r=Int(block_idx.x);var C=Int(cols)
    if r>=Int(rows):return
    var ss=Float32(0);var o=r*C
    for d in range(C):var v=x[o+d];ss+=v*v
    var iv=Float32(1)/sqrt(ss/Float32(C)+eps)
    for d in range(C):output[o+d]=_mr(x[o+d]*iv*Float32(w[d]))

def bm_qknorm(x:UnsafePointer[Float32,MutAnyOrigin],w:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],rows:Int32,eps:Float32):
    if Int(thread_idx.x)!=0:return
    var r=Int(block_idx.x)
    if r>=Int(rows):return
    var ss=Float32(0);var o=r*BB_HD
    for d in range(BB_HD):var v=x[o+d];ss+=v*v
    var iv=Float32(1)/sqrt(ss/Float32(BB_HD)+eps)
    for d in range(BB_HD):x[o+d]=_mr(x[o+d]*iv*Float32(w[d]))

def bm_rope(x:UnsafePointer[Float32,MutAnyOrigin],rows:Int32,heads:Int32,start:Int32,theta:Float32,factor:Float32):
    var half=BB_HD//2;var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=Int(rows)*Int(heads)*half:return
    var p=i//(Int(heads)*half);var hpair=i%(Int(heads)*half);var h=hpair//half;var d=hpair%half;var o=(p*Int(heads)+h)*BB_HD
    var inv=exp(-(Float32(2*d)/Float32(BB_HD))*log(theta))/factor;var a=Float32(Int(start)+p)*inv;var c=cos(a);var s=sin(a);var lo=o+d;var hi=lo+half;var u=x[lo];var v=x[hi]
    x[lo]=_mr(u*c-v*s);x[hi]=_mr(v*c+u*s)

def bm_store_kv(k:UnsafePointer[Float32,MutAnyOrigin],v:UnsafePointer[Float32,MutAnyOrigin],kc:UnsafePointer[Float32,MutAnyOrigin],vc:UnsafePointer[Float32,MutAnyOrigin],S:Int32,start:Int32,kvd:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x);var n=Int(S)*Int(kvd)
    if i<n:var r=i//Int(kvd);var d=i%Int(kvd);kc[(Int(start)+r)*Int(kvd)+d]=k[i];vc[(Int(start)+r)*Int(kvd)+d]=v[i]

def bm_attn(q:UnsafePointer[Float32,MutAnyOrigin],kc:UnsafePointer[Float32,MutAnyOrigin],vc:UnsafePointer[Float32,MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin],S:Int32,start:Int32,nh:Int32,nkv:Int32):
    var qh=Int(block_idx.x);var row=qh//Int(nh);var head=qh%Int(nh)
    if row>=Int(S):return
    var kh=head/(Int(nh)//Int(nkv));var count=Int(start)+row+1;var tid=Int(thread_idx.x)
    var sc=stack_allocation[DType.float32,address_space=AddressSpace.SHARED](row_major[512]());var st=stack_allocation[DType.float32,address_space=AddressSpace.SHARED](row_major[2]())
    var j=tid
    while j<count:
        var z=Float32(0);var qo=(row*Int(nh)+head)*BB_HD;var ko=(j*Int(nkv)+kh)*BB_HD
        for d in range(BB_HD):z+=q[qo+d]*kc[ko+d]
        sc[j]=z/sqrt(Float32(BB_HD));j+=Int(block_dim.x)
    barrier()
    if tid==0:
        var mx=Float32(-3.4e38)
        for p in range(count):
            if sc[p]>mx:mx=sc[p]
        var den=Float32(0)
        for p in range(count):den+=exp(sc[p]-mx)
        st[0]=mx;st[1]=den
    barrier()
    if tid<BB_HD:
        var z=Float32(0)
        for p in range(count):z+=exp(sc[p]-st[0])/st[1]*vc[(p*Int(nkv)+kh)*BB_HD+tid]
        output[(row*Int(nh)+head)*BB_HD+tid]=_mr(z)

def bm_add(x:UnsafePointer[Float32,MutAnyOrigin],u:UnsafePointer[Float32,MutAnyOrigin],n:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n):x[i]=_mr(x[i]+u[i])

def bm_swiglu(g:UnsafePointer[Float32,MutAnyOrigin],u:UnsafePointer[Float32,MutAnyOrigin],n:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n):var x=g[i];g[i]=_mr((x/(Float32(1)+exp(-x)))*u[i])

def bm_embed_frame(ids:UnsafePointer[Int64,MutAnyOrigin],table:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin]):
    var d=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if d<BB_D:
        var z=Float32(0)
        for c in range(CB):var t=Int(ids[c]);z+=Float32(table[(c*DD_V+t)*BB_D+d])
        output[d]=_mr(z)

def bm_depth_inputs(ids:UnsafePointer[Int64,MutAnyOrigin],backbone:UnsafePointer[Float32,MutAnyOrigin],table:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin]):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<CB*BB_D:
        var p=i//BB_D;var d=i%BB_D
        if p==0:output[i]=backbone[d]
        else:var tok=Int(ids[p-1]);output[i]=Float32(table[((p-1)*DD_V+tok)*BB_D+d])

def bm_depth_rope(x:UnsafePointer[Float32,MutAnyOrigin],rows:Int32,heads:Int32):
    var half=DD_HD//2;var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=Int(rows)*Int(heads)*half:return
    var p=i//(Int(heads)*half);var hp=i%(Int(heads)*half);var h=hp//half;var d=hp%half;var o=(p*Int(heads)+h)*DD_HD
    var base=exp(-(Float32(2*d)/Float32(DD_HD))*log(Float32(500000)));var wavelength=Float32(6.283185307179586)/base;var low=Float32(16)/Float32(0.001953125);var high=Float32(16)/Float32(0.0078125);var inv=base
    if wavelength>low:inv=base/Float32(32)
    elif wavelength>=high:
        var smooth=(Float32(16)/wavelength-Float32(0.001953125))/(Float32(0.0078125)-Float32(0.001953125));inv=(Float32(1)-smooth)*base/Float32(32)+smooth*base
    var angle=Float32(p)*inv;var c=cos(angle);var s=sin(angle);var lo=o+d;var hi=lo+half;var u=x[lo];var v=x[hi];x[lo]=_mr(u*c-v*s);x[hi]=_mr(v*c+u*s)

def bm_depth_head(hidden:UnsafePointer[Float32,MutAnyOrigin],weights:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin]):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<15*DD_V:
        var cb=i//DD_V;var token=i%DD_V;var z=Float32(0);var ho=(cb+1)*DD_D;var wo=(cb*DD_D*DD_V)+token
        for d in range(DD_D):z+=hidden[ho+d]*Float32(weights[wo+d*DD_V])
        output[i]=z

struct BMWeights(Movable):
    var q:List[UInt64];var k:List[UInt64];var v:List[UInt64];var o:List[UInt64];var qn:List[UInt64];var kn:List[UInt64];var ni:List[UInt64];var np:List[UInt64];var g:List[UInt64];var u:List[UInt64];var d:List[UInt64];var norm:UInt64;var head:UInt64;var audio:UInt64
    def __init__(out self,b:String) raises:
        self.q=List[UInt64]();self.k=List[UInt64]();self.v=List[UInt64]();self.o=List[UInt64]();self.qn=List[UInt64]();self.kn=List[UInt64]();self.ni=List[UInt64]();self.np=List[UInt64]();self.g=List[UInt64]();self.u=List[UInt64]();self.d=List[UInt64]()
        for l in range(BB_L):
            var p=b+"backbone_model_layers_"+String(l)+"_"
            self.q.append(load_bf16_file_to_gpu(p+"self_attn_q_proj_weight.bf16"));self.k.append(load_bf16_file_to_gpu(p+"self_attn_k_proj_weight.bf16"));self.v.append(load_bf16_file_to_gpu(p+"self_attn_v_proj_weight.bf16"));self.o.append(load_bf16_file_to_gpu(p+"self_attn_o_proj_weight.bf16"));self.qn.append(load_bf16_file_to_gpu(p+"self_attn_q_norm_weight.bf16"));self.kn.append(load_bf16_file_to_gpu(p+"self_attn_k_norm_weight.bf16"));self.ni.append(load_bf16_file_to_gpu(p+"input_layernorm_weight.bf16"));self.np.append(load_bf16_file_to_gpu(p+"post_attention_layernorm_weight.bf16"));self.g.append(load_bf16_file_to_gpu(p+"mlp_gate_proj_weight.bf16"));self.u.append(load_bf16_file_to_gpu(p+"mlp_up_proj_weight.bf16"));self.d.append(load_bf16_file_to_gpu(p+"mlp_down_proj_weight.bf16"))
        self.norm=load_bf16_file_to_gpu(b+"backbone_model_norm_weight.bf16");self.head=load_bf16_file_to_gpu(b+"lm_head_weight.bf16");self.audio=load_bf16_file_to_gpu(b+"depth_decoder_model_embed_tokens_weight.bf16")

struct DDWeights(Movable):
    var q:List[UInt64];var k:List[UInt64];var v:List[UInt64];var o:List[UInt64];var ni:List[UInt64];var np:List[UInt64];var g:List[UInt64];var u:List[UInt64];var d:List[UInt64];var project:UInt64;var norm:UInt64;var head:UInt64
    def __init__(out self,b:String) raises:
        self.q=List[UInt64]();self.k=List[UInt64]();self.v=List[UInt64]();self.o=List[UInt64]();self.ni=List[UInt64]();self.np=List[UInt64]();self.g=List[UInt64]();self.u=List[UInt64]();self.d=List[UInt64]()
        for l in range(DD_L):
            var p=b+"depth_decoder_model_layers_"+String(l)+"_"
            self.q.append(load_bf16_file_to_gpu(p+"self_attn_q_proj_weight.bf16"));self.k.append(load_bf16_file_to_gpu(p+"self_attn_k_proj_weight.bf16"));self.v.append(load_bf16_file_to_gpu(p+"self_attn_v_proj_weight.bf16"));self.o.append(load_bf16_file_to_gpu(p+"self_attn_o_proj_weight.bf16"));self.ni.append(load_bf16_file_to_gpu(p+"input_layernorm_weight.bf16"));self.np.append(load_bf16_file_to_gpu(p+"post_attention_layernorm_weight.bf16"));self.g.append(load_bf16_file_to_gpu(p+"mlp_gate_proj_weight.bf16"));self.u.append(load_bf16_file_to_gpu(p+"mlp_up_proj_weight.bf16"));self.d.append(load_bf16_file_to_gpu(p+"mlp_down_proj_weight.bf16"))
        self.project=load_bf16_file_to_gpu(b+"depth_decoder_model_inputs_embeds_projector_weight.bf16");self.norm=load_bf16_file_to_gpu(b+"depth_decoder_model_norm_weight.bf16");self.head=load_bf16_file_to_gpu(b+"depth_decoder_codebooks_head_weight.bf16")

struct BMLane(Movable):
    var length:Int;var kc:List[UInt64];var vc:List[UInt64];var last_hidden:UInt64
    def __init__(out self):
        self.length=0;self.kc=List[UInt64]();self.vc=List[UInt64]();self.last_hidden=cuda_malloc(BB_D*4)
        for _ in range(BB_L):self.kc.append(cuda_malloc(MAX_SEQ*BB_KVD*4));self.vc.append(cuda_malloc(MAX_SEQ*BB_KVD*4))

struct BreezeModel(Movable):
    var ctx:DeviceContext;var h:UInt64;var w:BMWeights;var dw:DDWeights;var lanes:List[BMLane]
    def __init__(out self,path:String) raises:
        self.ctx=DeviceContext();self.h=cublas_create();cublas_set_stream(self.h,self.ctx);var b=path if path.endswith("/") else path+"/";self.w=BMWeights(b);self.dw=DDWeights(b);self.lanes=List[BMLane]();self.lanes.append(BMLane());self.lanes.append(BMLane())

    def _run(mut self,lane:Int,x:UInt64,S:Int,layers_host:UInt64,final_host:UInt64,logits_host:UInt64,last_dev:UInt64) raises:
        var start=self.lanes[lane].length;var T=256;var he=S*BB_D;var qe=S*BB_QD;var ke=S*BB_KVD;var fe=S*BB_FF
        var n=cuda_malloc(he*4);var q=cuda_malloc(qe*4);var k=cuda_malloc(ke*4);var v=cuda_malloc(ke*4);var a=cuda_malloc(qe*4);var u=cuda_malloc(he*4);var g=cuda_malloc(fe*4);var up=cuda_malloc(fe*4);var bf=cuda_malloc(fe*2);var lo=cuda_malloc(BB_V*4)
        var rms=self.ctx.compile_function[bm_rms]();var qkn=self.ctx.compile_function[bm_qknorm]();var rope=self.ctx.compile_function[bm_rope]();var skv=self.ctx.compile_function[bm_store_kv]();var att=self.ctx.compile_function[bm_attn]();var add=self.ctx.compile_function[bm_add]();var sw=self.ctx.compile_function[bm_swiglu]()
        for l in range(BB_L):
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.w.ni[l]),_mf32(n),Int32(S),Int32(BB_D),Float32(1e-6),grid_dim=S,block_dim=1)
            gpu_matmul_bf16_dev_batched(self.ctx,self.h,q,n,self.w.q[l],bf,S,BB_D,BB_QD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,k,n,self.w.k[l],bf,S,BB_D,BB_KVD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,v,n,self.w.v[l],bf,S,BB_D,BB_KVD)
            self.ctx.enqueue_function(qkn,_mf32(q),_mbf16(self.w.qn[l]),Int32(S*BB_NH),Float32(1e-6),grid_dim=S*BB_NH,block_dim=1);self.ctx.enqueue_function(qkn,_mf32(k),_mbf16(self.w.kn[l]),Int32(S*BB_NKV),Float32(1e-6),grid_dim=S*BB_NKV,block_dim=1)
            self.ctx.enqueue_function(rope,_mf32(q),Int32(S),Int32(BB_NH),Int32(start),Float32(1e6),Float32(1),grid_dim=(S*BB_NH*64+T-1)//T,block_dim=T);self.ctx.enqueue_function(rope,_mf32(k),Int32(S),Int32(BB_NKV),Int32(start),Float32(1e6),Float32(1),grid_dim=(S*BB_NKV*64+T-1)//T,block_dim=T)
            self.ctx.enqueue_function(skv,_mf32(k),_mf32(v),_mf32(self.lanes[lane].kc[l]),_mf32(self.lanes[lane].vc[l]),Int32(S),Int32(start),Int32(BB_KVD),grid_dim=(ke+T-1)//T,block_dim=T);self.ctx.enqueue_function(att,_mf32(q),_mf32(self.lanes[lane].kc[l]),_mf32(self.lanes[lane].vc[l]),_mf32(a),Int32(S),Int32(start),Int32(BB_NH),Int32(BB_NKV),grid_dim=S*BB_NH,block_dim=128)
            gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,a,self.w.o[l],bf,S,BB_QD,BB_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.w.np[l]),_mf32(n),Int32(S),Int32(BB_D),Float32(1e-6),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,g,n,self.w.g[l],bf,S,BB_D,BB_FF);gpu_matmul_bf16_dev_batched(self.ctx,self.h,up,n,self.w.u[l],bf,S,BB_D,BB_FF);self.ctx.enqueue_function(sw,_mf32(g),_mf32(up),Int32(fe),grid_dim=(fe+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,g,self.w.d[l],bf,S,BB_FF,BB_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
            if layers_host!=0:self.ctx.synchronize();cuda_memcpy(layers_host+UInt64(l*he*4),x,he*4,2)
        self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.w.norm),_mf32(n),Int32(S),Int32(BB_D),Float32(1e-6),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,lo,n+UInt64((S-1)*BB_D*4),self.w.head,bf,1,BB_D,BB_V);self.ctx.synchronize()
        if final_host!=0:cuda_memcpy(final_host,n,he*4,2)
        if logits_host!=0:cuda_memcpy(logits_host,lo,BB_V*4,2)
        cuda_memcpy(self.lanes[lane].last_hidden,n+UInt64((S-1)*BB_D*4),BB_D*4,3)
        if last_dev!=0:cuda_memcpy(last_dev,self.lanes[lane].last_hidden,BB_D*4,3)
        self.lanes[lane].length=start+S
        cuda_free(lo);cuda_free(bf);cuda_free(up);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n)

    def prefill_lane(mut self,lane:Int,embeds_host:UInt64,S:Int,layers_host:UInt64,final_host:UInt64,logits_host:UInt64) raises:
        if lane<0 or lane>1 or S<=0 or S>MAX_SEQ:raise Error("invalid Breeze lane/prefill length")
        self.lanes[lane].length=0;var x=cuda_malloc(S*BB_D*4);cuda_memcpy(x,embeds_host,S*BB_D*4,1);self._run(lane,x,S,layers_host,final_host,logits_host,0);cuda_free(x)

    def step_backbone(mut self,lane:Int,codes_host:UInt64,logits_host:UInt64) raises:
        var ids=cuda_malloc(CB*8);var x=cuda_malloc(BB_D*4);cuda_memcpy(ids,codes_host,CB*8,1);var ef=self.ctx.compile_function[bm_embed_frame]();self.ctx.enqueue_function(ef,_mi64(ids),_mbf16(self.w.audio),_mf32(x),grid_dim=(BB_D+255)//256,block_dim=256);self._run(lane,x,1,0,0,logits_host,0);cuda_free(x);cuda_free(ids)

    def step(mut self,lane:Int,codes_host:UInt64,lm_host:UInt64,depth_host:UInt64) raises:
        if lane<0 or lane>1:raise Error("invalid Breeze lane")
        self.depth(codes_host,self.lanes[lane].last_hidden,depth_host)
        var ids=cuda_malloc(CB*8);var x=cuda_malloc(BB_D*4);cuda_memcpy(ids,codes_host,CB*8,1);var ef=self.ctx.compile_function[bm_embed_frame]();self.ctx.enqueue_function(ef,_mi64(ids),_mbf16(self.w.audio),_mf32(x),grid_dim=(BB_D+255)//256,block_dim=256);self._run(lane,x,1,0,0,lm_host,0);cuda_free(x);cuda_free(ids)

    def depth(mut self,codes_host:UInt64,backbone_host:UInt64,logits_host:UInt64) raises:
        var S=CB;var T=256;var he=S*DD_D;var qe=S*DD_QD;var ke=S*DD_KVD;var fe=S*DD_FF
        var ids=cuda_malloc(CB*8);var bh=cuda_malloc(BB_D*4);var raw=cuda_malloc(CB*BB_D*4);var x=cuda_malloc(he*4);var n=cuda_malloc(he*4);var q=cuda_malloc(qe*4);var k=cuda_malloc(ke*4);var v=cuda_malloc(ke*4);var a=cuda_malloc(qe*4);var u=cuda_malloc(he*4);var g=cuda_malloc(fe*4);var up=cuda_malloc(fe*4);var bf=cuda_malloc(fe*2);var kc=cuda_malloc(S*DD_KVD*4);var vc=cuda_malloc(S*DD_KVD*4);var lo=cuda_malloc(15*DD_V*4)
        cuda_memcpy(ids,codes_host,CB*8,1);cuda_memcpy(bh,backbone_host,BB_D*4,1);var inf=self.ctx.compile_function[bm_depth_inputs]();self.ctx.enqueue_function(inf,_mi64(ids),_mf32(bh),_mbf16(self.w.audio),_mf32(raw),grid_dim=(CB*BB_D+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,x,raw,self.dw.project,bf,S,BB_D,DD_D)
        var rms=self.ctx.compile_function[bm_rms]();var rope=self.ctx.compile_function[bm_depth_rope]();var skv=self.ctx.compile_function[bm_store_kv]();var att=self.ctx.compile_function[bm_attn]();var add=self.ctx.compile_function[bm_add]();var sw=self.ctx.compile_function[bm_swiglu]()
        for l in range(DD_L):
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.ni[l]),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,q,n,self.dw.q[l],bf,S,DD_D,DD_QD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,k,n,self.dw.k[l],bf,S,DD_D,DD_KVD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,v,n,self.dw.v[l],bf,S,DD_D,DD_KVD);self.ctx.enqueue_function(rope,_mf32(q),Int32(S),Int32(DD_NH),grid_dim=(S*DD_NH*64+T-1)//T,block_dim=T);self.ctx.enqueue_function(rope,_mf32(k),Int32(S),Int32(DD_NKV),grid_dim=(S*DD_NKV*64+T-1)//T,block_dim=T);self.ctx.enqueue_function(skv,_mf32(k),_mf32(v),_mf32(kc),_mf32(vc),Int32(S),Int32(0),Int32(DD_KVD),grid_dim=(ke+T-1)//T,block_dim=T);self.ctx.enqueue_function(att,_mf32(q),_mf32(kc),_mf32(vc),_mf32(a),Int32(S),Int32(0),Int32(DD_NH),Int32(DD_NKV),grid_dim=S*DD_NH,block_dim=128);gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,a,self.dw.o[l],bf,S,DD_QD,DD_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T);self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.np[l]),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,g,n,self.dw.g[l],bf,S,DD_D,DD_FF);gpu_matmul_bf16_dev_batched(self.ctx,self.h,up,n,self.dw.u[l],bf,S,DD_D,DD_FF);self.ctx.enqueue_function(sw,_mf32(g),_mf32(up),Int32(fe),grid_dim=(fe+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,g,self.dw.d[l],bf,S,DD_FF,DD_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
        self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.norm),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);var hf=self.ctx.compile_function[bm_depth_head]();self.ctx.enqueue_function(hf,_mf32(n),_mbf16(self.dw.head),_mf32(lo),grid_dim=(15*DD_V+T-1)//T,block_dim=T);self.ctx.synchronize();cuda_memcpy(logits_host,lo,15*DD_V*4,2)
        cuda_free(lo);cuda_free(vc);cuda_free(kc);cuda_free(bf);cuda_free(up);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n);cuda_free(x);cuda_free(raw);cuda_free(bh);cuda_free(ids)

    def free(mut self):
        for l in range(BB_L):
            cuda_free(self.w.q[l]);cuda_free(self.w.k[l]);cuda_free(self.w.v[l]);cuda_free(self.w.o[l]);cuda_free(self.w.qn[l]);cuda_free(self.w.kn[l]);cuda_free(self.w.ni[l]);cuda_free(self.w.np[l]);cuda_free(self.w.g[l]);cuda_free(self.w.u[l]);cuda_free(self.w.d[l])
        cuda_free(self.w.norm);cuda_free(self.w.head);cuda_free(self.w.audio)
        for l in range(DD_L):
            cuda_free(self.dw.q[l]);cuda_free(self.dw.k[l]);cuda_free(self.dw.v[l]);cuda_free(self.dw.o[l]);cuda_free(self.dw.ni[l]);cuda_free(self.dw.np[l]);cuda_free(self.dw.g[l]);cuda_free(self.dw.u[l]);cuda_free(self.dw.d[l])
        cuda_free(self.dw.project);cuda_free(self.dw.norm);cuda_free(self.dw.head)
        for lane in range(2):
            for l in range(BB_L):cuda_free(self.lanes[lane].kc[l]);cuda_free(self.lanes[lane].vc[l])
            cuda_free(self.lanes[lane].last_hidden)
