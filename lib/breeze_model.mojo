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
from lib.fp4_weights import load_to_gpu_nvfp4
from lib.gemma4_layer import _mm_dev_batched, W4A4Scratch

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

def bm_depth_rope(x:UnsafePointer[Float32,MutAnyOrigin],rows:Int32,heads:Int32,start:Int32):
    var half=DD_HD//2;var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=Int(rows)*Int(heads)*half:return
    var p=i//(Int(heads)*half);var hp=i%(Int(heads)*half);var h=hp//half;var d=hp%half;var o=(p*Int(heads)+h)*DD_HD
    var base=exp(-(Float32(2*d)/Float32(DD_HD))*log(Float32(500000)));var wavelength=Float32(6.283185307179586)/base;var low=Float32(16)/Float32(0.001953125);var high=Float32(16)/Float32(0.0078125);var inv=base
    if wavelength>low:inv=base/Float32(32)
    elif wavelength>=high:
        var smooth=(Float32(16)/wavelength-Float32(0.001953125))/(Float32(0.0078125)-Float32(0.001953125));inv=(Float32(1)-smooth)*base/Float32(32)+smooth*base
    var angle=Float32(Int(start)+p)*inv;var c=cos(angle);var s=sin(angle);var lo=o+d;var hi=lo+half;var u=x[lo];var v=x[hi];x[lo]=_mr(u*c-v*s);x[hi]=_mr(v*c+u*s)

def bm_depth_two_inputs(ids:UnsafePointer[Int64,MutAnyOrigin],backbone:UnsafePointer[Float32,MutAnyOrigin],table:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin]):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<2*BB_D:
        var p=i//BB_D;var d=i%BB_D
        if p==0:output[i]=backbone[d]
        else:output[i]=Float32(table[(Int(ids[0]))*BB_D+d])

def bm_depth_one_input(ids:UnsafePointer[Int64,MutAnyOrigin],cb:Int32,table:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin]):
    var d=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if d<BB_D:output[d]=Float32(table[(Int(cb)*DD_V+Int(ids[Int(cb)]))*BB_D+d])

def bm_depth_head(hidden:UnsafePointer[Float32,MutAnyOrigin],weights:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],output:UnsafePointer[Float32,MutAnyOrigin]):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<15*DD_V:
        var cb=i//DD_V;var token=i%DD_V;var z=Float32(0);var ho=(cb+1)*DD_D;var wo=(cb*DD_D*DD_V)+token
        for d in range(DD_D):z+=hidden[ho+d]*Float32(weights[wo+d*DD_V])
        output[i]=z

def bm_depth_head_one(hidden:UnsafePointer[Float32,MutAnyOrigin],weights:UnsafePointer[Scalar[DType.bfloat16],MutAnyOrigin],cb:Int32,output:UnsafePointer[Float32,MutAnyOrigin]):
    var token=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if token<DD_V:
        var z=Float32(0);var wo=Int(cb)*DD_D*DD_V+token
        for d in range(DD_D):z+=hidden[d]*Float32(weights[wo+d*DD_V])
        output[token]=z

def _bm_load_projection(stem:String,mut gs:List[Float32],mut ags:List[Float32]) raises -> UInt64:
    var w=load_to_gpu_nvfp4(stem+".nvfp4",gs,ags)
    if w!=0:return w
    w=load_bf16_file_to_gpu(stem+".bf16")
    if w==0:raise Error("missing Breeze backbone projection: "+stem+".{nvfp4,bf16}")
    return w

struct BMWeights(Movable):
    var q:List[UInt64];var k:List[UInt64];var v:List[UInt64];var o:List[UInt64];var qn:List[UInt64];var kn:List[UInt64];var ni:List[UInt64];var np:List[UInt64];var g:List[UInt64];var u:List[UInt64];var d:List[UInt64];var norm:UInt64;var head:UInt64;var audio:UInt64
    var qgs:List[Float32];var qags:List[Float32];var kgs:List[Float32];var kags:List[Float32];var vgs:List[Float32];var vags:List[Float32];var ogs:List[Float32];var oags:List[Float32];var ggs:List[Float32];var gags:List[Float32];var ugs:List[Float32];var uags:List[Float32];var dgs:List[Float32];var dags:List[Float32]
    def __init__(out self,b:String) raises:
        self.q=List[UInt64]();self.k=List[UInt64]();self.v=List[UInt64]();self.o=List[UInt64]();self.qn=List[UInt64]();self.kn=List[UInt64]();self.ni=List[UInt64]();self.np=List[UInt64]();self.g=List[UInt64]();self.u=List[UInt64]();self.d=List[UInt64]()
        self.qgs=List[Float32]();self.qags=List[Float32]();self.kgs=List[Float32]();self.kags=List[Float32]();self.vgs=List[Float32]();self.vags=List[Float32]();self.ogs=List[Float32]();self.oags=List[Float32]();self.ggs=List[Float32]();self.gags=List[Float32]();self.ugs=List[Float32]();self.uags=List[Float32]();self.dgs=List[Float32]();self.dags=List[Float32]()
        for l in range(BB_L):
            var p=b+"backbone_model_layers_"+String(l)+"_"
            self.q.append(_bm_load_projection(p+"self_attn_q_proj_weight",self.qgs,self.qags));self.k.append(_bm_load_projection(p+"self_attn_k_proj_weight",self.kgs,self.kags));self.v.append(_bm_load_projection(p+"self_attn_v_proj_weight",self.vgs,self.vags));self.o.append(_bm_load_projection(p+"self_attn_o_proj_weight",self.ogs,self.oags));self.g.append(_bm_load_projection(p+"mlp_gate_proj_weight",self.ggs,self.gags));self.u.append(_bm_load_projection(p+"mlp_up_proj_weight",self.ugs,self.uags));self.d.append(_bm_load_projection(p+"mlp_down_proj_weight",self.dgs,self.dags))
            self.qn.append(load_bf16_file_to_gpu(p+"self_attn_q_norm_weight.bf16"));self.kn.append(load_bf16_file_to_gpu(p+"self_attn_k_norm_weight.bf16"));self.ni.append(load_bf16_file_to_gpu(p+"input_layernorm_weight.bf16"));self.np.append(load_bf16_file_to_gpu(p+"post_attention_layernorm_weight.bf16"))
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
    var length:Int;var kc:List[UInt64];var vc:List[UInt64];var last_hidden:UInt64;var depth_kc:UInt64;var depth_vc:UInt64;var depth_pos:Int
    def __init__(out self):
        self.length=0;self.kc=List[UInt64]();self.vc=List[UInt64]();self.last_hidden=cuda_malloc(BB_D*4);self.depth_kc=cuda_malloc(DD_L*CB*DD_KVD*4);self.depth_vc=cuda_malloc(DD_L*CB*DD_KVD*4);self.depth_pos=0
        for _ in range(BB_L):self.kc.append(cuda_malloc(MAX_SEQ*BB_KVD*4));self.vc.append(cuda_malloc(MAX_SEQ*BB_KVD*4))

struct BreezeModel(Movable):
    var ctx:DeviceContext;var h:UInt64;var w:BMWeights;var dw:DDWeights;var lanes:List[BMLane]
    def __init__(out self,path:String) raises:
        self.ctx=DeviceContext();self.h=cublas_create();cublas_set_stream(self.h,self.ctx);var b=path if path.endswith("/") else path+"/";self.w=BMWeights(b);self.dw=DDWeights(b);self.lanes=List[BMLane]();self.lanes.append(BMLane());self.lanes.append(BMLane())

    def _bbmm(mut self,dst:UInt64,src:UInt64,weight:UInt64,bf:UInt64,ws:UInt64,S:Int,K:Int,N:Int,gs:Float32,ags:Float32,w4:W4A4Scratch,l:Int,p:String) raises:
        if gs!=0.0:_mm_dev_batched(self.ctx,self.h,dst,src,weight,bf,S,K,N,ws,gs,ags,w4,l,p)
        else:gpu_matmul_bf16_dev_batched(self.ctx,self.h,dst,src,weight,bf,S,K,N)

    def _run(mut self,lane:Int,x:UInt64,S:Int,layers_host:UInt64,final_host:UInt64,logits_host:UInt64,last_dev:UInt64,paired:Bool=False) raises:
        var start=self.lanes[lane].length;var count=2 if paired else 1;var rows=1 if paired else S;var T=256;var he=S*BB_D;var qe=S*BB_QD;var ke=S*BB_KVD;var fe=S*BB_FF
        var n=cuda_malloc(he*4);var q=cuda_malloc(qe*4);var k=cuda_malloc(ke*4);var v=cuda_malloc(ke*4);var a=cuda_malloc(qe*4);var u=cuda_malloc(he*4);var g=cuda_malloc(fe*4);var up=cuda_malloc(fe*4);var bf=cuda_malloc(fe*2);var lo=cuda_malloc(count*BB_V*4);var ws=cuda_malloc(BB_D*BB_FF*2);var w4=W4A4Scratch(0,0,0,0,0,False,0,0)
        var rms=self.ctx.compile_function[bm_rms]();var qkn=self.ctx.compile_function[bm_qknorm]();var rope=self.ctx.compile_function[bm_rope]();var skv=self.ctx.compile_function[bm_store_kv]();var att=self.ctx.compile_function[bm_attn]();var add=self.ctx.compile_function[bm_add]();var sw=self.ctx.compile_function[bm_swiglu]()
        for l in range(BB_L):
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.w.ni[l]),_mf32(n),Int32(S),Int32(BB_D),Float32(1e-6),grid_dim=S,block_dim=1)
            self._bbmm(q,n,self.w.q[l],bf,ws,S,BB_D,BB_QD,self.w.qgs[l],self.w.qags[l],w4,l,"q");self._bbmm(k,n,self.w.k[l],bf,ws,S,BB_D,BB_KVD,self.w.kgs[l],self.w.kags[l],w4,l,"k");self._bbmm(v,n,self.w.v[l],bf,ws,S,BB_D,BB_KVD,self.w.vgs[l],self.w.vags[l],w4,l,"v")
            self.ctx.enqueue_function(qkn,_mf32(q),_mbf16(self.w.qn[l]),Int32(S*BB_NH),Float32(1e-6),grid_dim=S*BB_NH,block_dim=1);self.ctx.enqueue_function(qkn,_mf32(k),_mbf16(self.w.kn[l]),Int32(S*BB_NKV),Float32(1e-6),grid_dim=S*BB_NKV,block_dim=1)
            # Lane-major GEMM rows; each attention domain keeps its own prefix.
            for ln in range(count):
                var which=ln if paired else lane
                var pos=self.lanes[which].length
                var qo=q+UInt64(ln*rows*BB_QD*4);var ko=k+UInt64(ln*rows*BB_KVD*4);var vo=v+UInt64(ln*rows*BB_KVD*4);var ao=a+UInt64(ln*rows*BB_QD*4)
                self.ctx.enqueue_function(rope,_mf32(qo),Int32(rows),Int32(BB_NH),Int32(pos),Float32(1e6),Float32(1),grid_dim=(rows*BB_NH*64+T-1)//T,block_dim=T)
                self.ctx.enqueue_function(rope,_mf32(ko),Int32(rows),Int32(BB_NKV),Int32(pos),Float32(1e6),Float32(1),grid_dim=(rows*BB_NKV*64+T-1)//T,block_dim=T)
                self.ctx.enqueue_function(skv,_mf32(ko),_mf32(vo),_mf32(self.lanes[which].kc[l]),_mf32(self.lanes[which].vc[l]),Int32(rows),Int32(pos),Int32(BB_KVD),grid_dim=(rows*BB_KVD+T-1)//T,block_dim=T)
                self.ctx.enqueue_function(att,_mf32(qo),_mf32(self.lanes[which].kc[l]),_mf32(self.lanes[which].vc[l]),_mf32(ao),Int32(rows),Int32(pos),Int32(BB_NH),Int32(BB_NKV),grid_dim=rows*BB_NH,block_dim=128)
            self._bbmm(u,a,self.w.o[l],bf,ws,S,BB_QD,BB_D,self.w.ogs[l],self.w.oags[l],w4,l,"o");self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.w.np[l]),_mf32(n),Int32(S),Int32(BB_D),Float32(1e-6),grid_dim=S,block_dim=1);self._bbmm(g,n,self.w.g[l],bf,ws,S,BB_D,BB_FF,self.w.ggs[l],self.w.gags[l],w4,l,"gate");self._bbmm(up,n,self.w.u[l],bf,ws,S,BB_D,BB_FF,self.w.ugs[l],self.w.uags[l],w4,l,"up");self.ctx.enqueue_function(sw,_mf32(g),_mf32(up),Int32(fe),grid_dim=(fe+T-1)//T,block_dim=T);self._bbmm(u,g,self.w.d[l],bf,ws,S,BB_FF,BB_D,self.w.dgs[l],self.w.dags[l],w4,l,"down");self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
            if layers_host!=0:self.ctx.synchronize();cuda_memcpy(layers_host+UInt64(l*he*4),x,he*4,2)
        self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.w.norm),_mf32(n),Int32(S),Int32(BB_D),Float32(1e-6),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,lo,n+UInt64((0 if paired else S-1)*BB_D*4),self.w.head,bf,count,BB_D,BB_V);self.ctx.synchronize()
        if final_host!=0:cuda_memcpy(final_host,n,he*4,2)
        if logits_host!=0:cuda_memcpy(logits_host,lo,count*BB_V*4,2)
        for ln in range(count):
            var which=ln if paired else lane
            cuda_memcpy(self.lanes[which].last_hidden,n+UInt64((ln if paired else S-1)*BB_D*4),BB_D*4,3)
            self.lanes[which].length+=rows
        if last_dev!=0:cuda_memcpy(last_dev,self.lanes[lane].last_hidden,BB_D*4,3)
        cuda_free(ws);cuda_free(lo);cuda_free(bf);cuda_free(up);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n)

    def prefill_lane(mut self,lane:Int,embeds_host:UInt64,S:Int,layers_host:UInt64,final_host:UInt64,logits_host:UInt64) raises:
        if lane<0 or lane>1 or S<=0 or S>MAX_SEQ:raise Error("invalid Breeze lane/prefill length")
        self.lanes[lane].length=0;var x=cuda_malloc(S*BB_D*4);cuda_memcpy(x,embeds_host,S*BB_D*4,1);self._run(lane,x,S,layers_host,final_host,logits_host,0);cuda_free(x)

    def step_backbone(mut self,lane:Int,codes_host:UInt64,logits_host:UInt64) raises:
        var ids=cuda_malloc(CB*8);var x=cuda_malloc(BB_D*4);cuda_memcpy(ids,codes_host,CB*8,1);var ef=self.ctx.compile_function[bm_embed_frame]();self.ctx.enqueue_function(ef,_mi64(ids),_mbf16(self.w.audio),_mf32(x),grid_dim=(BB_D+255)//256,block_dim=256);self._run(lane,x,1,0,0,logits_host,0);cuda_free(x);cuda_free(ids)

    def step(mut self,lane:Int,codes_host:UInt64,lm_host:UInt64,depth_host:UInt64) raises:
        if lane<0 or lane>1:raise Error("invalid Breeze lane")
        self.depth_sequential(codes_host,self.lanes[lane].last_hidden,depth_host)
        var ids=cuda_malloc(CB*8);var x=cuda_malloc(BB_D*4);cuda_memcpy(ids,codes_host,CB*8,1);var ef=self.ctx.compile_function[bm_embed_frame]();self.ctx.enqueue_function(ef,_mi64(ids),_mbf16(self.w.audio),_mf32(x),grid_dim=(BB_D+255)//256,block_dim=256);self._run(lane,x,1,0,0,lm_host,0);cuda_free(x);cuda_free(ids)

    def depth_lane(mut self,lane:Int,codes_host:UInt64,depth_host:UInt64) raises:
        if lane<0 or lane>1:raise Error("invalid Breeze lane")
        self.depth_sequential(codes_host,self.lanes[lane].last_hidden,depth_host)

    def _depth_increment(mut self,raw:UInt64,S:Int,start:Int,head_index:Int,out_host:UInt64,kcbase:UInt64,vcbase:UInt64,paired:Bool=False) raises:
        var count=2 if paired else 1;var rows=S//count;var T=256;var x=cuda_malloc(S*DD_D*4);var n=cuda_malloc(S*DD_D*4);var q=cuda_malloc(S*DD_QD*4);var k=cuda_malloc(S*DD_KVD*4);var v=cuda_malloc(S*DD_KVD*4);var a=cuda_malloc(S*DD_QD*4);var u=cuda_malloc(S*DD_D*4);var g=cuda_malloc(S*DD_FF*4);var up=cuda_malloc(S*DD_FF*4);var bf=cuda_malloc(S*DD_FF*2);var one=cuda_malloc(count*DD_V*4)
        gpu_matmul_bf16_dev_batched(self.ctx,self.h,x,raw,self.dw.project,bf,S,BB_D,DD_D);self._depth_pass(x,S,start,kcbase,vcbase,n,q,k,v,a,u,g,up,bf,paired)
        var head=self.ctx.compile_function[bm_depth_head_one]()
        for ln in range(count):
            self.ctx.enqueue_function(head,_mf32(n+UInt64(((ln+1)*rows-1)*DD_D*4)),_mbf16(self.dw.head),Int32(head_index),_mf32(one+UInt64(ln*DD_V*4)),grid_dim=(DD_V+T-1)//T,block_dim=T)
        self.ctx.synchronize();cuda_memcpy(out_host,one,count*DD_V*4,2)
        cuda_free(one);cuda_free(bf);cuda_free(up);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n);cuda_free(x)

    def depth_begin(mut self,lane:Int,codes_host:UInt64,out_host:UInt64) raises:
        if lane<0 or lane>1:raise Error("invalid Breeze depth begin")
        var T=256;var ids=cuda_malloc(CB*8);cuda_memcpy(ids,codes_host,CB*8,1);var raw=cuda_malloc(2*BB_D*4);var inf=self.ctx.compile_function[bm_depth_two_inputs]();self.ctx.enqueue_function(inf,_mi64(ids),_mf32(self.lanes[lane].last_hidden),_mbf16(self.w.audio),_mf32(raw),grid_dim=(2*BB_D+T-1)//T,block_dim=T);self._depth_increment(raw,2,0,0,out_host,self.lanes[lane].depth_kc,self.lanes[lane].depth_vc);self.lanes[lane].depth_pos=2;cuda_free(raw);cuda_free(ids)

    def depth_advance(mut self,lane:Int,input_codebook:Int,codes_host:UInt64,out_host:UInt64) raises:
        if lane<0 or lane>1 or input_codebook<1 or input_codebook>=15 or self.lanes[lane].depth_pos!=input_codebook+1:raise Error("invalid Breeze depth advance")
        var T=256;var ids=cuda_malloc(CB*8);cuda_memcpy(ids,codes_host,CB*8,1);var raw=cuda_malloc(BB_D*4);var onein=self.ctx.compile_function[bm_depth_one_input]();self.ctx.enqueue_function(onein,_mi64(ids),Int32(input_codebook),_mbf16(self.w.audio),_mf32(raw),grid_dim=(BB_D+T-1)//T,block_dim=T);self._depth_increment(raw,1,input_codebook+1,input_codebook,out_host,self.lanes[lane].depth_kc,self.lanes[lane].depth_vc);self.lanes[lane].depth_pos=input_codebook+2;cuda_free(raw);cuda_free(ids)


    def step_backbone2(mut self,codes_host:UInt64,logits_host:UInt64) raises:
        # Prefills remain separate (e.g. 23 cond / 10 uncond tokens).
        for ln in range(2):
            if self.lanes[ln].length<=0 or self.lanes[ln].length>=MAX_SEQ:raise Error("paired backbone requires two live prefixes with cache capacity")
        var ids=cuda_malloc(CB*8);var x=cuda_malloc(2*BB_D*4)
        cuda_memcpy(ids,codes_host,CB*8,1)
        var ef=self.ctx.compile_function[bm_embed_frame]()
        for ln in range(2):
            self.ctx.enqueue_function(ef,_mi64(ids),_mbf16(self.w.audio),_mf32(x+UInt64(ln*BB_D*4)),grid_dim=(BB_D+255)//256,block_dim=256)
        self._run(0,x,2,0,0,logits_host,0,True)
        cuda_free(x);cuda_free(ids)

    def depth_begin2(mut self,codes_host:UInt64,out_host:UInt64) raises:
        for ln in range(2):
            if self.lanes[ln].length<=0:raise Error("paired depth requires two live prefixes")
        var ids=cuda_malloc(CB*8);var raw=cuda_malloc(4*BB_D*4)
        cuda_memcpy(ids,codes_host,CB*8,1)
        var inf=self.ctx.compile_function[bm_depth_two_inputs]()
        for ln in range(2):
            self.ctx.enqueue_function(inf,_mi64(ids),_mf32(self.lanes[ln].last_hidden),_mbf16(self.w.audio),_mf32(raw+UInt64(ln*2*BB_D*4)),grid_dim=(2*BB_D+255)//256,block_dim=256)
        self._depth_increment(raw,4,0,0,out_host,0,0,True)
        for ln in range(2):self.lanes[ln].depth_pos=2
        cuda_free(raw);cuda_free(ids)

    def depth_advance2(mut self,input_codebook:Int,codes_host:UInt64,out_host:UInt64) raises:
        if input_codebook<1 or input_codebook>=15:raise Error("invalid paired codebook")
        for ln in range(2):
            if self.lanes[ln].depth_pos!=input_codebook+1:raise Error("paired depth cache positions disagree")
        var ids=cuda_malloc(CB*8);var raw=cuda_malloc(2*BB_D*4)
        cuda_memcpy(ids,codes_host,CB*8,1)
        var inf=self.ctx.compile_function[bm_depth_one_input]()
        for ln in range(2):
            self.ctx.enqueue_function(inf,_mi64(ids),Int32(input_codebook),_mbf16(self.w.audio),_mf32(raw+UInt64(ln*BB_D*4)),grid_dim=(BB_D+255)//256,block_dim=256)
        self._depth_increment(raw,2,input_codebook+1,input_codebook,out_host,0,0,True)
        for ln in range(2):self.lanes[ln].depth_pos=input_codebook+2
        cuda_free(raw);cuda_free(ids)

    def depth(mut self,codes_host:UInt64,backbone_host:UInt64,logits_host:UInt64) raises:
        var S=CB;var T=256;var he=S*DD_D;var qe=S*DD_QD;var ke=S*DD_KVD;var fe=S*DD_FF
        var ids=cuda_malloc(CB*8);var bh=cuda_malloc(BB_D*4);var raw=cuda_malloc(CB*BB_D*4);var x=cuda_malloc(he*4);var n=cuda_malloc(he*4);var q=cuda_malloc(qe*4);var k=cuda_malloc(ke*4);var v=cuda_malloc(ke*4);var a=cuda_malloc(qe*4);var u=cuda_malloc(he*4);var g=cuda_malloc(fe*4);var up=cuda_malloc(fe*4);var bf=cuda_malloc(fe*2);var kc=cuda_malloc(S*DD_KVD*4);var vc=cuda_malloc(S*DD_KVD*4);var lo=cuda_malloc(15*DD_V*4)
        cuda_memcpy(ids,codes_host,CB*8,1);cuda_memcpy(bh,backbone_host,BB_D*4,1);var inf=self.ctx.compile_function[bm_depth_inputs]();self.ctx.enqueue_function(inf,_mi64(ids),_mf32(bh),_mbf16(self.w.audio),_mf32(raw),grid_dim=(CB*BB_D+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,x,raw,self.dw.project,bf,S,BB_D,DD_D)
        var rms=self.ctx.compile_function[bm_rms]();var rope=self.ctx.compile_function[bm_depth_rope]();var skv=self.ctx.compile_function[bm_store_kv]();var att=self.ctx.compile_function[bm_attn]();var add=self.ctx.compile_function[bm_add]();var sw=self.ctx.compile_function[bm_swiglu]()
        for l in range(DD_L):
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.ni[l]),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,q,n,self.dw.q[l],bf,S,DD_D,DD_QD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,k,n,self.dw.k[l],bf,S,DD_D,DD_KVD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,v,n,self.dw.v[l],bf,S,DD_D,DD_KVD);self.ctx.enqueue_function(rope,_mf32(q),Int32(S),Int32(DD_NH),Int32(0),grid_dim=(S*DD_NH*64+T-1)//T,block_dim=T);self.ctx.enqueue_function(rope,_mf32(k),Int32(S),Int32(DD_NKV),Int32(0),grid_dim=(S*DD_NKV*64+T-1)//T,block_dim=T);self.ctx.enqueue_function(skv,_mf32(k),_mf32(v),_mf32(kc),_mf32(vc),Int32(S),Int32(0),Int32(DD_KVD),grid_dim=(ke+T-1)//T,block_dim=T);self.ctx.enqueue_function(att,_mf32(q),_mf32(kc),_mf32(vc),_mf32(a),Int32(S),Int32(0),Int32(DD_NH),Int32(DD_NKV),grid_dim=S*DD_NH,block_dim=128);gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,a,self.dw.o[l],bf,S,DD_QD,DD_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T);self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.np[l]),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,g,n,self.dw.g[l],bf,S,DD_D,DD_FF);gpu_matmul_bf16_dev_batched(self.ctx,self.h,up,n,self.dw.u[l],bf,S,DD_D,DD_FF);self.ctx.enqueue_function(sw,_mf32(g),_mf32(up),Int32(fe),grid_dim=(fe+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,g,self.dw.d[l],bf,S,DD_FF,DD_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
        self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.norm),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);var hf=self.ctx.compile_function[bm_depth_head]();self.ctx.enqueue_function(hf,_mf32(n),_mbf16(self.dw.head),_mf32(lo),grid_dim=(15*DD_V+T-1)//T,block_dim=T);self.ctx.synchronize();cuda_memcpy(logits_host,lo,15*DD_V*4,2)
        cuda_free(lo);cuda_free(vc);cuda_free(kc);cuda_free(bf);cuda_free(up);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n);cuda_free(x);cuda_free(raw);cuda_free(bh);cuda_free(ids)

    def _depth_pass(mut self,x:UInt64,S:Int,start:Int,kcbase:UInt64,vcbase:UInt64,n:UInt64,q:UInt64,k:UInt64,v:UInt64,a:UInt64,u:UInt64,g:UInt64,up:UInt64,bf:UInt64,paired:Bool=False) raises:
        var count=2 if paired else 1;var rows=S//count;var T=256;var he=S*DD_D;var qe=S*DD_QD;var ke=S*DD_KVD;var fe=S*DD_FF
        var rms=self.ctx.compile_function[bm_rms]();var rope=self.ctx.compile_function[bm_depth_rope]();var skv=self.ctx.compile_function[bm_store_kv]();var att=self.ctx.compile_function[bm_attn]();var add=self.ctx.compile_function[bm_add]();var sw=self.ctx.compile_function[bm_swiglu]()
        for l in range(DD_L):
            var kc=kcbase+UInt64(l*CB*DD_KVD*4);var vc=vcbase+UInt64(l*CB*DD_KVD*4)
            self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.ni[l]),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,q,n,self.dw.q[l],bf,S,DD_D,DD_QD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,k,n,self.dw.k[l],bf,S,DD_D,DD_KVD);gpu_matmul_bf16_dev_batched(self.ctx,self.h,v,n,self.dw.v[l],bf,S,DD_D,DD_KVD)
            for ln in range(count):
                var lkc=(self.lanes[ln].depth_kc if paired else kcbase)+UInt64(l*CB*DD_KVD*4)
                var lvc=(self.lanes[ln].depth_vc if paired else vcbase)+UInt64(l*CB*DD_KVD*4)
                var qo=q+UInt64(ln*rows*DD_QD*4);var ko=k+UInt64(ln*rows*DD_KVD*4);var vo=v+UInt64(ln*rows*DD_KVD*4);var ao=a+UInt64(ln*rows*DD_QD*4)
                self.ctx.enqueue_function(rope,_mf32(qo),Int32(rows),Int32(DD_NH),Int32(start),grid_dim=(rows*DD_NH*64+T-1)//T,block_dim=T)
                self.ctx.enqueue_function(rope,_mf32(ko),Int32(rows),Int32(DD_NKV),Int32(start),grid_dim=(rows*DD_NKV*64+T-1)//T,block_dim=T)
                self.ctx.enqueue_function(skv,_mf32(ko),_mf32(vo),_mf32(lkc),_mf32(lvc),Int32(rows),Int32(start),Int32(DD_KVD),grid_dim=(rows*DD_KVD+T-1)//T,block_dim=T)
                self.ctx.enqueue_function(att,_mf32(qo),_mf32(lkc),_mf32(lvc),_mf32(ao),Int32(rows),Int32(start),Int32(DD_NH),Int32(DD_NKV),grid_dim=rows*DD_NH,block_dim=128)
            gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,a,self.dw.o[l],bf,S,DD_QD,DD_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T);self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.np[l]),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1);gpu_matmul_bf16_dev_batched(self.ctx,self.h,g,n,self.dw.g[l],bf,S,DD_D,DD_FF);gpu_matmul_bf16_dev_batched(self.ctx,self.h,up,n,self.dw.u[l],bf,S,DD_D,DD_FF);self.ctx.enqueue_function(sw,_mf32(g),_mf32(up),Int32(fe),grid_dim=(fe+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,u,g,self.dw.d[l],bf,S,DD_FF,DD_D);self.ctx.enqueue_function(add,_mf32(x),_mf32(u),Int32(he),grid_dim=(he+T-1)//T,block_dim=T)
        self.ctx.enqueue_function(rms,_mf32(x),_mbf16(self.dw.norm),_mf32(n),Int32(S),Int32(DD_D),Float32(1e-5),grid_dim=S,block_dim=1)

    def depth_sequential(mut self,codes_host:UInt64,backbone_dev:UInt64,logits_host:UInt64) raises:
        var T=256;var ids=cuda_malloc(CB*8);cuda_memcpy(ids,codes_host,CB*8,1)
        var raw=cuda_malloc(2*BB_D*4);var x=cuda_malloc(2*DD_D*4);var n=cuda_malloc(2*DD_D*4);var q=cuda_malloc(2*DD_QD*4);var k=cuda_malloc(2*DD_KVD*4);var v=cuda_malloc(2*DD_KVD*4);var a=cuda_malloc(2*DD_QD*4);var u=cuda_malloc(2*DD_D*4);var g=cuda_malloc(2*DD_FF*4);var up=cuda_malloc(2*DD_FF*4);var bf=cuda_malloc(2*DD_FF*2);var one=cuda_malloc(DD_V*4)
        var kcbase=cuda_malloc(DD_L*CB*DD_KVD*4);var vcbase=cuda_malloc(DD_L*CB*DD_KVD*4)
        var head=self.ctx.compile_function[bm_depth_head_one]();var inf=self.ctx.compile_function[bm_depth_two_inputs]();self.ctx.enqueue_function(inf,_mi64(ids),_mf32(backbone_dev),_mbf16(self.w.audio),_mf32(raw),grid_dim=(2*BB_D+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,x,raw,self.dw.project,bf,2,BB_D,DD_D);self._depth_pass(x,2,0,kcbase,vcbase,n,q,k,v,a,u,g,up,bf);self.ctx.enqueue_function(head,_mf32(n+UInt64(DD_D*4)),_mbf16(self.dw.head),Int32(0),_mf32(one),grid_dim=(DD_V+T-1)//T,block_dim=T);self.ctx.synchronize();cuda_memcpy(logits_host,one,DD_V*4,2)
        var onein=self.ctx.compile_function[bm_depth_one_input]()
        for cb in range(1,15):
            self.ctx.enqueue_function(onein,_mi64(ids),Int32(cb),_mbf16(self.w.audio),_mf32(raw),grid_dim=(BB_D+T-1)//T,block_dim=T);gpu_matmul_bf16_dev_batched(self.ctx,self.h,x,raw,self.dw.project,bf,1,BB_D,DD_D);self._depth_pass(x,1,cb+1,kcbase,vcbase,n,q,k,v,a,u,g,up,bf);self.ctx.enqueue_function(head,_mf32(n),_mbf16(self.dw.head),Int32(cb),_mf32(one),grid_dim=(DD_V+T-1)//T,block_dim=T);self.ctx.synchronize();cuda_memcpy(logits_host+UInt64(cb*DD_V*4),one,DD_V*4,2)
        cuda_free(vcbase);cuda_free(kcbase);cuda_free(one);cuda_free(bf);cuda_free(up);cuda_free(g);cuda_free(u);cuda_free(a);cuda_free(v);cuda_free(k);cuda_free(q);cuda_free(n);cuda_free(x);cuda_free(raw);cuda_free(ids)

    def free(mut self):
        for l in range(BB_L):
            cuda_free(self.w.q[l]);cuda_free(self.w.k[l]);cuda_free(self.w.v[l]);cuda_free(self.w.o[l]);cuda_free(self.w.qn[l]);cuda_free(self.w.kn[l]);cuda_free(self.w.ni[l]);cuda_free(self.w.np[l]);cuda_free(self.w.g[l]);cuda_free(self.w.u[l]);cuda_free(self.w.d[l])
        cuda_free(self.w.norm);cuda_free(self.w.head);cuda_free(self.w.audio)
        for l in range(DD_L):
            cuda_free(self.dw.q[l]);cuda_free(self.dw.k[l]);cuda_free(self.dw.v[l]);cuda_free(self.dw.o[l]);cuda_free(self.dw.ni[l]);cuda_free(self.dw.np[l]);cuda_free(self.dw.g[l]);cuda_free(self.dw.u[l]);cuda_free(self.dw.d[l])
        cuda_free(self.dw.project);cuda_free(self.dw.norm);cuda_free(self.dw.head)
        for lane in range(2):
            for l in range(BB_L):cuda_free(self.lanes[lane].kc[l]);cuda_free(self.lanes[lane].vc[l])
            cuda_free(self.lanes[lane].last_hidden);cuda_free(self.lanes[lane].depth_kc);cuda_free(self.lanes[lane].depth_vc)
