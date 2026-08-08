"""mtp_drafter.mojo — the Gemma-4-31B MTP speculative-decode drafter (assistant head).

A 4-layer gemma4-style MTP head (internal 1024) that drafts K tokens off the 31B's
backbone. Forward is the EXACT llama.cpp gemma4-assistant graph (src/models/gemma4-assistant.cpp):

  per draft step k:
    x  = 31B_tok_embd[token] * sqrt(5376)          # target embedding, gemma-scaled
    xh = concat(x, inp_h)                           # [target_embed(5376), backbone_hidden(5376)] = 10752
    cur = pre_proj @ xh                             # 10752 -> 1024
    for il in 0..4:                                 # gemma4 4-norm sandwich, per-layer head_dim
        n   = rmsnorm(cur, attn_norm[il])
        a   = ATTN(n)                               # q_proj -> q_norm -> rope -> shared-31B-KV attend -> o_proj
        a   = rmsnorm(a, attn_post_norm[il]); attn_out = a + cur
        f   = rmsnorm(attn_out, ffn_norm[il]); f = geglu(f); f = rmsnorm(f, ffn_post_norm[il])
        cur = (f + attn_out) * out_scale[il]
    cur     = rmsnorm(cur, output_norm)
    logits  = tok_embd^T @ cur                      # 1024 -> 262144 (TIED, full vocab — no d2t)
    draft_k = argmax(logits)
    inp_h   = post_proj @ cur                       # 1024 -> 5376  (the K-recurrence carrier: next step's hidden)
    token   = draft_k

Weights are Q8 (the law-allowed drafter concession), loaded via the .q8 int8 GEMV path
(q8_weights.mojo). M=1 per step (K drafts are sequential) so every matmul is a GEMV.

⚠️ The ATTN block (q_norm + rope + shared-KV windowed attend) is the co-design piece —
the assistant has NO own K/V; it attends the 31B's shared INT4 KV with the 31B's window-floor
map (#431/#432). It is STUBBED here (q_proj + o_proj are real; the attend is zeroed and
clearly marked) so the skeleton builds + runs end-to-end. `_mtp_attn` is filled in separately.
"""
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import UnsafePointer
from lib.q8_weights import load_to_gpu_q8, gpu_matmul_q8_fused_dev
from lib.io import load_to_gpu, read_f32
from lib.cuda import cuda_malloc, cuda_memcpy, cuda_free, cuda_sync
from lib.ops_gpu_mojo import gpu_residual_add_mojo, gpu_elementwise_mul_mojo, gpu_gelu_mojo, gpu_scalar_mul_mojo, gpu_rope_mojo
from lib.ops_gpu_mojo_reductions import gpu_rmsnorm_mojo, gpu_rmsnorm_inplace_mojo, gpu_qk_norm_mojo
from lib.decode_argmax import device_decode_token
from lib.gemma4_engine import GemmaEngine
from lib.engine_decode import engine_attend_shared

comptime D2D = 3   # cudaMemcpyDeviceToDevice


struct MTPDrafter(Movable):
    # dims
    var dC: Int        # internal dim (1024)
    var dB: Int        # backbone dim (5376)
    var nl: Int        # n layers (4)
    var ff: Int        # ffn dim (8192)
    var vocab: Int     # 262144
    var nhead: Int     # 32
    # per-layer
    var hd: List[Int]          # head_dim [256,256,256,512]
    var qdim: List[Int]        # attn_q out dim = nhead*head_dim [8192,8192,8192,16384]
    var swa: List[Bool]        # sliding-window pattern [T,T,T,F]
    var oscale: List[Float32]  # layer_output_scale (host scalar per layer)
    # global weights (device)
    var w_pre: UInt64          # pre_projection .q8  [1024, 10752]
    var w_post: UInt64         # post_projection .q8 [5376, 1024]
    var w_tok: UInt64          # token_embd .q8 [262144, 1024] (tied output)
    var n_out: UInt64          # output_norm .bin (fp32) [1024]
    var w_rope: UInt64         # rope_freqs .bin (fp32) [head_dim/2] (global layer)
    # per-layer weights (device)
    var w_q: List[UInt64]
    var w_o: List[UInt64]
    var w_g: List[UInt64]
    var w_u: List[UInt64]
    var w_d: List[UInt64]
    var n_an: List[UInt64]     # attn_norm
    var n_qn: List[UInt64]     # attn_q_norm
    var n_pan: List[UInt64]    # attn_post_norm (post_attention_norm)
    var n_fn: List[UInt64]     # ffn_norm
    var n_fpn: List[UInt64]    # ffn_post_norm (post_ffw_norm)
    # scratch (device)
    var s_cur: UInt64
    var s_norm: UInt64
    var s_q: UInt64
    var s_attn: UInt64
    var s_gate: UInt64
    var s_up: UInt64
    var s_mlp: UInt64
    var s_down: UInt64
    var s_xh: UInt64
    var s_hnext: UInt64
    var s_logits: UInt64
    var s_embed: UInt64        # [dB] input-token embedding * sqrt(dB), per draft step
    var s_aqd: UInt64          # [max qdim] attention output (separate from s_q — no aliasing)
    var engine_ptr: UInt64     # GemmaEngine address (the shared-KV read); set by nomos_mtp_draft

    def __init__(out self, dir: String):
        self.dC = 1024
        self.dB = 5376
        self.nl = 4
        self.ff = 8192
        self.vocab = 262144
        self.nhead = 32
        self.hd = [256, 256, 256, 512]
        self.qdim = [8192, 8192, 8192, 16384]
        self.swa = [True, True, True, False]
        # global weights
        self.w_pre = load_to_gpu_q8(dir + "nextn_pre_projection_weight.q8")
        self.w_post = load_to_gpu_q8(dir + "nextn_post_projection_weight.q8")
        self.w_tok = load_to_gpu_q8(dir + "token_embd_weight.q8")
        self.n_out = load_to_gpu(dir + "output_norm_weight.bin")
        self.w_rope = load_to_gpu(dir + "rope_freqs_weight.bin")
        # per-layer
        self.w_q = List[UInt64]()
        self.w_o = List[UInt64]()
        self.w_g = List[UInt64]()
        self.w_u = List[UInt64]()
        self.w_d = List[UInt64]()
        self.n_an = List[UInt64]()
        self.n_qn = List[UInt64]()
        self.n_pan = List[UInt64]()
        self.n_fn = List[UInt64]()
        self.n_fpn = List[UInt64]()
        self.oscale = List[Float32]()
        for il in range(4):
            var b = dir + "blk_" + String(il) + "_"
            self.w_q.append(load_to_gpu_q8(b + "attn_q_weight.q8"))
            self.w_o.append(load_to_gpu_q8(b + "attn_output_weight.q8"))
            self.w_g.append(load_to_gpu_q8(b + "ffn_gate_weight.q8"))
            self.w_u.append(load_to_gpu_q8(b + "ffn_up_weight.q8"))
            self.w_d.append(load_to_gpu_q8(b + "ffn_down_weight.q8"))
            self.n_an.append(load_to_gpu(b + "attn_norm_weight.bin"))
            self.n_qn.append(load_to_gpu(b + "attn_q_norm_weight.bin"))
            self.n_pan.append(load_to_gpu(b + "post_attention_norm_weight.bin"))
            self.n_fn.append(load_to_gpu(b + "ffn_norm_weight.bin"))
            self.n_fpn.append(load_to_gpu(b + "post_ffw_norm_weight.bin"))
            var os = read_f32(b + "layer_output_scale_weight.bin")
            self.oscale.append(os[0] if len(os) > 0 else Float32(1.0))
        # scratch
        self.s_cur = cuda_malloc(1024 * 4)
        self.s_norm = cuda_malloc(1024 * 4)
        self.s_q = cuda_malloc(16384 * 4)
        self.s_attn = cuda_malloc(1024 * 4)
        self.s_gate = cuda_malloc(8192 * 4)
        self.s_up = cuda_malloc(8192 * 4)
        self.s_mlp = cuda_malloc(8192 * 4)
        self.s_down = cuda_malloc(1024 * 4)
        self.s_xh = cuda_malloc(10752 * 4)
        self.s_hnext = cuda_malloc(5376 * 4)
        self.s_logits = cuda_malloc(262144 * 4)
        self.s_embed = cuda_malloc(5376 * 4)
        self.s_aqd = cuda_malloc(16384 * 4)
        self.engine_ptr = UInt64(0)


# ── ATTN block — the co-design piece (shared-31B-KV windowed attention) ───
# Input  d_in:  [dC] post-input-norm hidden.
# Output d_out: [dC] post-o_proj attention contribution.
# REAL: q_proj (wq @ d_in -> [qdim]) and o_proj (wo @ attended -> [dC]).
# STUB: the attended value is zeroed (no q_norm / rope / shared-KV softmax yet), so the
#   skeleton runs end-to-end and o_proj(0)=0 -> attn_out == inpL. Replace the marked
#   middle with: reshape q -> [head_dim, nhead]; rmsnorm(q, attn_q_norm[il]); rope (rope_freqs
#   on the global layer only); attend the 31B int4 KV with the 31B window-floor map; -> [qdim].
def _mtp_attn(ctx: DeviceContext, mut m: MTPDrafter, il: Int, d_in: UInt64, d_out: UInt64) raises:
    var qd = m.qdim[il]
    var l_hd = m.hd[il]                  # 256 (il0-2) / 512 (il3) == the attended 31B bank's head_dim
    var l31b = 59 if il == 3 else 58     # FACT 2: SWA il0-2 -> 31B L58, global il3 -> 31B L59
    # 1. q_proj (assistant): q = wq @ d_in  [dC -> qd], qd = 32 heads * l_hd
    gpu_matmul_q8_fused_dev(ctx, m.s_q, d_in, m.w_q[il], m.dC, qd)
    # 2. q_norm (assistant, per-head; FACT 4: pre-attn scale folded into q_norm, no extra 1/sqrt(hd))
    gpu_qk_norm_mojo(ctx, m.s_q, m.n_qn[il], 32, l_hd)
    # 3. RoPE at the FIXED draft position (FACT 1), basis matched to the cached K (FACT 3):
    var eng = UnsafePointer[GemmaEngine, MutAnyOrigin](unsafe_from_address=Int(m.engine_ptr))
    var pos = eng[0].cache_lens_fp32[l31b] - 1
    if il == 3:
        gpu_rope_mojo(ctx, m.s_q, pos, 32, l_hd, Float32(1000000.0), l_hd // 4)  # global θ=1e6, partial 128
    else:
        gpu_rope_mojo(ctx, m.s_q, pos, 32, l_hd, Float32(10000.0), l_hd)         # sliding θ=1e4, full
    # 4. attend the shared 31B int4 KV (engine owns dequant+window+int8-attn) -> s_aqd
    engine_attend_shared(eng[0], l31b, m.s_q, 32, m.s_aqd)
    # 5. o_proj: o = wo @ attended  [qd -> dC]
    gpu_matmul_q8_fused_dev(ctx, d_out, m.s_aqd, m.w_o[il], qd, m.dC)


# ── one draft step: (inp_h, inp_embed, token) -> draft token; leaves next-hidden in s_hnext ──
# d_inp_h:     [dB] the backbone hidden for this step (step 0 = the 31B tap; later = prev s_hnext).
# d_inp_embed: [dB] the target embedding of the input token, ALREADY * sqrt(dB) (caller supplies).
def mtp_draft_step(ctx: DeviceContext, mut m: MTPDrafter, d_inp_h: UInt64, d_inp_embed: UInt64) raises -> Int:
    # xh = concat(embed, hidden)  [embed first — matches ggml_concat(x, inp_h, 0)]
    cuda_memcpy(m.s_xh, d_inp_embed, m.dB * 4, D2D)
    cuda_memcpy(m.s_xh + UInt64(m.dB * 4), d_inp_h, m.dB * 4, D2D)
    # pre_proj: 10752 -> 1024
    gpu_matmul_q8_fused_dev(ctx, m.s_cur, m.s_xh, m.w_pre, 2 * m.dB, m.dC)
    # 4 gemma4 layers
    for il in range(m.nl):
        gpu_rmsnorm_mojo(ctx, m.s_cur, m.s_norm, m.n_an[il], m.dC)         # input norm -> s_norm
        _mtp_attn(ctx, m, il, m.s_norm, m.s_attn)                         # attn -> s_attn (post o_proj)
        gpu_rmsnorm_inplace_mojo(ctx, m.s_attn, m.n_pan[il], m.dC)         # post-attn norm
        gpu_residual_add_mojo(ctx, m.s_attn, m.s_cur, m.dC)               # attn_out = s_attn + inpL(s_cur)
        # FFN on attn_out (s_attn)
        gpu_rmsnorm_mojo(ctx, m.s_attn, m.s_norm, m.n_fn[il], m.dC)        # pre-ff norm -> s_norm
        gpu_matmul_q8_fused_dev(ctx, m.s_gate, m.s_norm, m.w_g[il], m.dC, m.ff)
        gpu_matmul_q8_fused_dev(ctx, m.s_up, m.s_norm, m.w_u[il], m.dC, m.ff)
        gpu_gelu_mojo(ctx, m.s_gate, m.ff)                                # gelu(gate)
        gpu_elementwise_mul_mojo(ctx, m.s_mlp, m.s_gate, m.s_up, m.ff)    # gate*up
        gpu_matmul_q8_fused_dev(ctx, m.s_down, m.s_mlp, m.w_d[il], m.ff, m.dC)
        gpu_rmsnorm_inplace_mojo(ctx, m.s_down, m.n_fpn[il], m.dC)         # post-ff norm
        gpu_residual_add_mojo(ctx, m.s_down, m.s_attn, m.dC)             # cur = s_down + attn_out
        gpu_scalar_mul_mojo(ctx, m.s_down, m.oscale[il], m.dC)            # * layer_output_scale
        cuda_memcpy(m.s_cur, m.s_down, m.dC * 4, D2D)                     # inpL = cur for next layer
    # output norm
    gpu_rmsnorm_inplace_mojo(ctx, m.s_cur, m.n_out, m.dC)
    # logits = tok_embd^T @ cur : 1024 -> 262144 (tied, full vocab)
    gpu_matmul_q8_fused_dev(ctx, m.s_logits, m.s_cur, m.w_tok, m.dC, m.vocab)
    var pen = List[Int32]()
    var token = device_decode_token(ctx, m.s_logits, m.vocab, pen, Float32(0.0))
    # h_next = post_proj @ cur : 1024 -> 5376 (next step's backbone hidden)
    gpu_matmul_q8_fused_dev(ctx, m.s_hnext, m.s_cur, m.w_post, m.dC, m.dB)
    return token


# ── K-token draft: run the head K times, threading (token, h_next) ───────────
# d_tap_h:       [dB] the 31B's final backbone hidden at the seed position (the tap).
# seed_token:    the last real (verified) token — the first input to the head.
# d_embed_table: the 31B's fp32 tied embed [vocab, dB] (engine's d_embed_lmhead).
# Appends K draft token ids to `drafts`. The 31B then batch-verifies them (the MMQ GEMM).
def mtp_draft_k(ctx: DeviceContext, mut m: MTPDrafter, d_tap_h: UInt64, seed_token: Int,
                d_embed_table: UInt64, k: Int, mut drafts: List[Int32]) raises:
    var scale = sqrt(Float32(m.dB))
    var h = d_tap_h
    var tok = seed_token
    for _ in range(k):
        # x = embed[tok] * sqrt(dB), built from already-linked ops (no embed_load_op C shim):
        # D2D-copy the fp32 embed row [tok*dB, +dB), then scale in place.
        cuda_memcpy(m.s_embed, d_embed_table + UInt64(tok * m.dB * 4), m.dB * 4, D2D)
        gpu_scalar_mul_mojo(ctx, m.s_embed, scale, m.dB)
        var draft = mtp_draft_step(ctx, m, h, m.s_embed)
        drafts.append(Int32(draft))
        h = m.s_hnext        # next step's hidden = this step's post_proj output
        tok = draft          # next step's input token = this draft


def release_mtp_drafter_handle(handle: UInt64):
    if handle == 0:
        return
    var ptr = UnsafePointer[MTPDrafter, MutExternalOrigin](
        unsafe_from_address=Int(handle)
    )
    if ptr[0].w_pre != 0: cuda_free(ptr[0].w_pre)
    if ptr[0].w_post != 0: cuda_free(ptr[0].w_post)
    if ptr[0].w_tok != 0: cuda_free(ptr[0].w_tok)
    if ptr[0].n_out != 0: cuda_free(ptr[0].n_out)
    if ptr[0].w_rope != 0: cuda_free(ptr[0].w_rope)
    for il in range(ptr[0].nl):
        if ptr[0].w_q[il] != 0: cuda_free(ptr[0].w_q[il])
        if ptr[0].w_o[il] != 0: cuda_free(ptr[0].w_o[il])
        if ptr[0].w_g[il] != 0: cuda_free(ptr[0].w_g[il])
        if ptr[0].w_u[il] != 0: cuda_free(ptr[0].w_u[il])
        if ptr[0].w_d[il] != 0: cuda_free(ptr[0].w_d[il])
        if ptr[0].n_an[il] != 0: cuda_free(ptr[0].n_an[il])
        if ptr[0].n_qn[il] != 0: cuda_free(ptr[0].n_qn[il])
        if ptr[0].n_pan[il] != 0: cuda_free(ptr[0].n_pan[il])
        if ptr[0].n_fn[il] != 0: cuda_free(ptr[0].n_fn[il])
        if ptr[0].n_fpn[il] != 0: cuda_free(ptr[0].n_fpn[il])
    if ptr[0].s_cur != 0: cuda_free(ptr[0].s_cur)
    if ptr[0].s_norm != 0: cuda_free(ptr[0].s_norm)
    if ptr[0].s_q != 0: cuda_free(ptr[0].s_q)
    if ptr[0].s_attn != 0: cuda_free(ptr[0].s_attn)
    if ptr[0].s_gate != 0: cuda_free(ptr[0].s_gate)
    if ptr[0].s_up != 0: cuda_free(ptr[0].s_up)
    if ptr[0].s_mlp != 0: cuda_free(ptr[0].s_mlp)
    if ptr[0].s_down != 0: cuda_free(ptr[0].s_down)
    if ptr[0].s_xh != 0: cuda_free(ptr[0].s_xh)
    if ptr[0].s_hnext != 0: cuda_free(ptr[0].s_hnext)
    if ptr[0].s_logits != 0: cuda_free(ptr[0].s_logits)
    if ptr[0].s_embed != 0: cuda_free(ptr[0].s_embed)
    if ptr[0].s_aqd != 0: cuda_free(ptr[0].s_aqd)
    cuda_sync()
    ptr.destroy_pointee()
    ptr.free()
