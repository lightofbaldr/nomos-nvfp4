"""kv_cache_quant.mojo — GPU-resident KV-cache quantization (FP4 Step 5).

The live K/V cache is the BIGGER memory cost than the weights at long context
(bf16 KV @16k ≈ 32 GB >> 16.5 GB NVFP4 weights), so quantizing it is the gate to
fit the 31B on a 24 GB edge card (RTX PRO 4000). Design: store the cache
QUANTIZED, dequant-on-read into a one-layer bf16 scratch, then the existing cuBLAS
attention runs unchanged (parity-safe — no attention-kernel rewrite).

Granularity: per (head, position) ROW of `dim` values — one scale per token-head.
This is per-token absmax (handles per-token KV outliers), matching
the host/on-disk codecs (which are separate; these
are their GPU live-cache twins). INT8 here (2×, near-lossless); INT4 + rotation is
the follow-up for the full 4× / 24 GB fit. Mojo 1.0.0b3 + sm_12xa.

  gpu_quant_kv_i8(ctx, in_bf16, out_i8, out_scales, n_rows, dim)   one warp/row
  gpu_dequant_kv_i8(ctx, in_i8, in_scales, out_bf16, n_rows, dim)  elementwise

Sliding-window (#430): for Gemma-4's 50 sliding layers the cache is allocated as a
RING of `cache_cap` (= SLIDING_WINDOW) slots, not `max_seq`. The slot for absolute
position q is `q % cache_cap`. The append kernels write at the ring slot; the
dequant-layer kernels read the active window [win_start, win_start+klen) from the
ring and write it LINEARLY into the (still max_seq-strided) bf16 scratch, so the
cuBLAS attention is unchanged. When cache_cap == max_seq and win_start == 0 (the
non-SWA path, and all full/global layers) the modulo and remap are no-ops and the
output is byte-identical to the pre-#430 absolute-position codec.
"""
from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from max.gpu import barrier
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer


def _round_i8(x: Float32) -> Int:
    """Round-to-nearest (half away from zero) into the symmetric INT8 range."""
    var r = Int(x + Float32(0.5)) if x >= Float32(0.0) else Int(x - Float32(0.5))
    if r > 127:
        r = 127
    if r < -127:
        r = -127
    return r


def quant_kv_i8_kernel(
    in_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],  # [n_rows, dim]
    out_i8: UnsafePointer[Int8, MutAnyOrigin],                     # [n_rows, dim]
    out_scales: UnsafePointer[Float32, MutAnyOrigin],             # [n_rows]
    n_rows_arg: Int32, dim_arg: Int32,
):
    var n_rows = Int(n_rows_arg)
    var dim = Int(dim_arg)
    var r = Int(block_idx.x)            # one warp per (head,position) row
    if r >= n_rows:
        return
    var lane = Int(thread_idx.x)
    var base = r * dim

    # per-row absmax (warp-shuffle max reduce over dim)
    var amax = Float32(0.0)
    var i = lane
    while i < dim:
        var v = abs(Float32(in_bf16[base + i]))
        if v > amax:
            amax = v
        i += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        var other = shuffle_xor(amax, UInt32(off))
        if other > amax:
            amax = other
        off //= 2

    var scale = amax / Float32(127.0)
    if scale == Float32(0.0):
        scale = Float32(1.0)
    if lane == 0:
        out_scales[r] = scale
    var inv = Float32(1.0) / scale

    var j = lane
    while j < dim:
        out_i8[base + j] = Int8(_round_i8(Float32(in_bf16[base + j]) * inv))
        j += WARP_SIZE


def dequant_kv_i8_kernel(
    in_i8: UnsafePointer[Int8, MutAnyOrigin],                      # [n_rows, dim]
    in_scales: UnsafePointer[Float32, MutAnyOrigin],              # [n_rows]
    out_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin], # [n_rows, dim]
    n_rows_arg: Int32, dim_arg: Int32,
):
    var n_rows = Int(n_rows_arg)
    var dim = Int(dim_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < n_rows * dim:
        var r = idx // dim
        var v = Float32(in_i8[idx]) * in_scales[r]
        out_bf16[idx] = v.cast[DType.bfloat16]()


def gpu_quant_kv_i8(
    ctx: DeviceContext, in_bf16: UInt64, out_i8: UInt64, out_scales: UInt64,
    n_rows: Int, dim: Int,
) raises:
    """Quantize a [n_rows, dim] bf16 buffer -> per-row symmetric INT8 (one warp/row)."""
    var kern = ctx.compile_function[quant_kv_i8_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(in_bf16)),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(out_i8)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_scales)),
        Int32(n_rows), Int32(dim),
        grid_dim=n_rows, block_dim=WARP_SIZE,
    )


def gpu_dequant_kv_i8(
    ctx: DeviceContext, in_i8: UInt64, in_scales: UInt64, out_bf16: UInt64,
    n_rows: Int, dim: Int,
) raises:
    """Dequant a [n_rows, dim] INT8 buffer (+ per-row scales) -> bf16."""
    var kern = ctx.compile_function[dequant_kv_i8_kernel]()
    var threads = 256
    var blocks = (n_rows * dim + threads - 1) // threads
    ctx.enqueue_function(
        kern,
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(in_i8)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(in_scales)),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(out_bf16)),
        Int32(n_rows), Int32(dim),
        grid_dim=blocks, block_dim=threads,
    )


# ─────────────────────────────────────────────────────────────────────────
# Live-cache integration (mirrors batched_attn_gpu.append_kv_batch_bf16_kernel
# layout EXACTLY): src K/V is [n_s, nkv, l_hd] (seq stride l_kvd); the cache is
# [nkv, cache_cap, cache_hd] with the row for (head h, abs-pos p) at the RING slot
# (h*cache_cap + p % cache_cap)*cache_hd and only the first l_hd of cache_hd real
# (the rest zero-padded). One scale per (head, slot) row over the l_hd real values
# — per-token granularity. cache_cap == max_seq for non-SWA / full layers
# (modulo is then a no-op since p < max_seq).
# ─────────────────────────────────────────────────────────────────────────

def append_quant_kv_i8_kernel(
    src_k: UnsafePointer[Float32, MutAnyOrigin],            # [n_s, nkv, l_hd]
    src_v: UnsafePointer[Float32, MutAnyOrigin],
    dst_k_i8: UnsafePointer[Int8, MutAnyOrigin],           # [nkv, cache_cap, cache_hd]
    dst_v_i8: UnsafePointer[Int8, MutAnyOrigin],
    k_scales: UnsafePointer[Float32, MutAnyOrigin],        # [nkv, cache_cap]
    v_scales: UnsafePointer[Float32, MutAnyOrigin],
    n_s_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, l_kvd_arg: Int32, cache_cap_arg: Int32, cache_hd_arg: Int32, base_pos_arg: Int32,
):
    var n_s = Int(n_s_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var l_kvd = Int(l_kvd_arg)
    var cache_cap = Int(cache_cap_arg)
    var cache_hd = Int(cache_hd_arg)
    var base_pos = Int(base_pos_arg)
    var w = Int(block_idx.x)            # one warp per (seq, head)
    if w >= n_s * nkv:
        return
    var h = w % nkv
    var s = w // nkv
    var lane = Int(thread_idx.x)
    var src_base = s * l_kvd + h * l_hd
    var cache_row = h * cache_cap + ((base_pos + s) % cache_cap)   # ring slot
    var dst_base = cache_row * cache_hd

    # K: per-row absmax over l_hd -> int8 + scale; zero the [l_hd, cache_hd) pad
    var amk = Float32(0.0)
    var i = lane
    while i < l_hd:
        var v = abs(src_k[src_base + i])
        if v > amk:
            amk = v
        i += WARP_SIZE
    var ok = WARP_SIZE // 2
    while ok > 0:
        var o = shuffle_xor(amk, UInt32(ok))
        if o > amk:
            amk = o
        ok //= 2
    var sck = amk / Float32(127.0)
    if sck == Float32(0.0):
        sck = Float32(1.0)
    if lane == 0:
        k_scales[cache_row] = sck
    var ik = Float32(1.0) / sck
    var jk = lane
    while jk < cache_hd:
        dst_k_i8[dst_base + jk] = Int8(_round_i8(src_k[src_base + jk] * ik)) if jk < l_hd else Int8(0)
        jk += WARP_SIZE

    # V: same
    var amv = Float32(0.0)
    var iv = lane
    while iv < l_hd:
        var v = abs(src_v[src_base + iv])
        if v > amv:
            amv = v
        iv += WARP_SIZE
    var ov = WARP_SIZE // 2
    while ov > 0:
        var o = shuffle_xor(amv, UInt32(ov))
        if o > amv:
            amv = o
        ov //= 2
    var scv = amv / Float32(127.0)
    if scv == Float32(0.0):
        scv = Float32(1.0)
    if lane == 0:
        v_scales[cache_row] = scv
    var ivd = Float32(1.0) / scv
    var jv = lane
    while jv < cache_hd:
        dst_v_i8[dst_base + jv] = Int8(_round_i8(src_v[src_base + jv] * ivd)) if jv < l_hd else Int8(0)
        jv += WARP_SIZE


def dequant_kv_i8_layer_kernel(
    src_i8: UnsafePointer[Int8, MutAnyOrigin],             # [nkv, cache_cap, cache_hd]
    scales: UnsafePointer[Float32, MutAnyOrigin],          # [nkv, cache_cap]
    dst_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],  # [nkv, max_seq, cache_hd]
    nkv_arg: Int32, klen_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, cache_cap_arg: Int32, win_start_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var cache_cap = Int(cache_cap_arg)
    var win_start = Int(win_start_arg)
    # Dequant the active window [win_start, win_start+klen) from the RING cache into
    # the max_seq-strided bf16 scratch, written LINEARLY at linear index j in [0,klen)
    # so the existing cuBLAS attention reads it with identical strides. Ring slot for
    # absolute pos (win_start+j) is (win_start+j) % cache_cap. For the non-SWA path
    # (cache_cap==max_seq, win_start==0, klen==seq_len) ring==j and src_row==dst_row,
    # so this is byte-identical to the old absolute-position dequant.
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nkv * klen * cache_hd:
        var d = idx % cache_hd
        var t = idx // cache_hd
        var j = t % klen
        var h = t // klen
        var ring = (win_start + j) % cache_cap
        var src_row = h * cache_cap + ring
        var dst_row = h * max_seq + j
        var v = Float32(src_i8[src_row * cache_hd + d]) * scales[src_row]
        dst_bf16[dst_row * cache_hd + d] = v.cast[DType.bfloat16]()


def gpu_append_quant_kv_i8(
    ctx: DeviceContext, src_k: UInt64, src_v: UInt64,
    dst_k_i8: UInt64, dst_v_i8: UInt64, k_scales: UInt64, v_scales: UInt64,
    n_s: Int, nkv: Int, l_hd: Int, l_kvd: Int, cache_cap: Int, cache_hd: Int, base_pos: Int,
) raises:
    """Quantize+scatter n_s new tokens' K/V (fp32 [n_s,nkv,l_hd]) into the INT8 ring
    cache at abs positions base_pos.. (ring slot p%cache_cap) — the INT8 twin of
    gpu_append_kv_batched_bf16 (one warp per (seq,head))."""
    var kern = ctx.compile_function[append_quant_kv_i8_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_k)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_v)),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_k_i8)),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_v_i8)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(k_scales)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(v_scales)),
        Int32(n_s), Int32(nkv), Int32(l_hd), Int32(l_kvd), Int32(cache_cap), Int32(cache_hd), Int32(base_pos),
        grid_dim=n_s * nkv, block_dim=WARP_SIZE,
    )


def gpu_dequant_kv_i8_layer(
    ctx: DeviceContext, src_i8: UInt64, scales: UInt64, dst_bf16: UInt64,
    nkv: Int, klen: Int, cache_hd: Int, max_seq: Int, cache_cap: Int, win_start: Int,
) raises:
    """Dequant the window [win_start, win_start+klen) of the ring cache -> bf16 scratch
    (max_seq-strided, written linearly at [0,klen), attention-ready)."""
    var kern = ctx.compile_function[dequant_kv_i8_layer_kernel]()
    var threads = 256
    var blocks = (nkv * klen * cache_hd + threads - 1) // threads
    ctx.enqueue_function(
        kern,
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(src_i8)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scales)),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(dst_bf16)),
        Int32(nkv), Int32(klen), Int32(cache_hd), Int32(max_seq), Int32(cache_cap), Int32(win_start),
        grid_dim=blocks, block_dim=threads,
    )


# ─────────────────────────────────────────────────────────────────────────
# INT4 live-cache codec (Feature B) — 2 nibbles/byte, /7 absmax, SAME per-(head,
# slot) row granularity as the INT8 codec above. Packed cache is [nkv, cache_cap,
# cache_hd/2] UInt8; dequant unpacks + sign-extends 4-bit two's complement into the
# same bf16 scratch the cuBLAS attention reads. 4× vs bf16 / 2× vs INT8. The ring
# (cache_cap) + window→linear remap match the INT8 codec exactly.
# ─────────────────────────────────────────────────────────────────────────

def _round_i4(x: Float32) -> Int:
    """Round-to-nearest (half away from zero) into the symmetric INT4 range [-7,7]."""
    var r = Int(x + Float32(0.5)) if x >= Float32(0.0) else Int(x - Float32(0.5))
    if r > 7:
        r = 7
    if r < -7:
        r = -7
    return r


def append_quant_kv_i4_kernel(
    src_k: UnsafePointer[Float32, MutAnyOrigin],            # [n_s, nkv, l_hd]
    src_v: UnsafePointer[Float32, MutAnyOrigin],
    dst_k_i4: UnsafePointer[UInt8, MutAnyOrigin],           # [nkv, cache_cap, cache_hd/2]
    dst_v_i4: UnsafePointer[UInt8, MutAnyOrigin],
    k_scales: UnsafePointer[Float32, MutAnyOrigin],        # [nkv, cache_cap]
    v_scales: UnsafePointer[Float32, MutAnyOrigin],
    n_s_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, l_kvd_arg: Int32, cache_cap_arg: Int32, cache_hd_arg: Int32, base_pos_arg: Int32,
):
    var n_s = Int(n_s_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var l_kvd = Int(l_kvd_arg)
    var cache_cap = Int(cache_cap_arg)
    var cache_hd = Int(cache_hd_arg)
    var base_pos = Int(base_pos_arg)
    var w = Int(block_idx.x)            # one warp per (seq, head)
    if w >= n_s * nkv:
        return
    var h = w % nkv
    var s = w // nkv
    var lane = Int(thread_idx.x)
    var src_base = s * l_kvd + h * l_hd
    var cache_row = h * cache_cap + ((base_pos + s) % cache_cap)   # ring slot
    var hcache = cache_hd // 2
    var dst_base = cache_row * hcache

    # K: per-row absmax over l_hd -> int4 + scale (/7); pack 2 nibbles/byte
    var amk = Float32(0.0)
    var i = lane
    while i < l_hd:
        var v = abs(src_k[src_base + i])
        if v > amk:
            amk = v
        i += WARP_SIZE
    var ok = WARP_SIZE // 2
    while ok > 0:
        var o = shuffle_xor(amk, UInt32(ok))
        if o > amk:
            amk = o
        ok //= 2
    var sck = amk / Float32(7.0)
    if sck == Float32(0.0):
        sck = Float32(1.0)
    if lane == 0:
        k_scales[cache_row] = sck
    var ik = Float32(1.0) / sck
    var jb = lane
    while jb < hcache:
        var j0 = 2 * jb
        var j1 = j0 + 1
        var q0 = _round_i4(src_k[src_base + j0] * ik) if j0 < l_hd else 0
        var q1 = _round_i4(src_k[src_base + j1] * ik) if j1 < l_hd else 0
        dst_k_i4[dst_base + jb] = UInt8((q0 & 0xF) | ((q1 & 0xF) << 4))
        jb += WARP_SIZE

    # V: same
    var amv = Float32(0.0)
    var iv = lane
    while iv < l_hd:
        var vv = abs(src_v[src_base + iv])
        if vv > amv:
            amv = vv
        iv += WARP_SIZE
    var ov = WARP_SIZE // 2
    while ov > 0:
        var o = shuffle_xor(amv, UInt32(ov))
        if o > amv:
            amv = o
        ov //= 2
    var scv = amv / Float32(7.0)
    if scv == Float32(0.0):
        scv = Float32(1.0)
    if lane == 0:
        v_scales[cache_row] = scv
    var ivd = Float32(1.0) / scv
    var jbv = lane
    while jbv < hcache:
        var j0v = 2 * jbv
        var j1v = j0v + 1
        var q0v = _round_i4(src_v[src_base + j0v] * ivd) if j0v < l_hd else 0
        var q1v = _round_i4(src_v[src_base + j1v] * ivd) if j1v < l_hd else 0
        dst_v_i4[dst_base + jbv] = UInt8((q0v & 0xF) | ((q1v & 0xF) << 4))
        jbv += WARP_SIZE


def append_quant_kv_i4_with_q8_kernel(
    src_k: UnsafePointer[Float32, MutAnyOrigin],            # [n_s, nkv, l_hd]
    src_v: UnsafePointer[Float32, MutAnyOrigin],
    dst_k_i4: UnsafePointer[UInt8, MutAnyOrigin],           # [nkv, cache_cap, cache_hd/2]
    dst_v_i4: UnsafePointer[UInt8, MutAnyOrigin],
    k_scales: UnsafePointer[Float32, MutAnyOrigin],         # [nkv, cache_cap]
    v_scales: UnsafePointer[Float32, MutAnyOrigin],
    dst_k_q8: UnsafePointer[Int8, MutAnyOrigin],            # [nkv, q8_pitch, cache_hd]
    dst_v_q8: UnsafePointer[Int8, MutAnyOrigin],
    k_q8_scales: UnsafePointer[Float32, MutAnyOrigin],      # [nkv, q8_pitch]
    v_q8_scales: UnsafePointer[Float32, MutAnyOrigin],
    n_s_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, l_kvd_arg: Int32, cache_cap_arg: Int32,
    cache_hd_arg: Int32, base_pos_arg: Int32, q8_pitch_arg: Int32,
):
    var n_s = Int(n_s_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var l_kvd = Int(l_kvd_arg)
    var cache_cap = Int(cache_cap_arg)
    var cache_hd = Int(cache_hd_arg)
    var base_pos = Int(base_pos_arg)
    var q8_pitch = Int(q8_pitch_arg)
    var w = Int(block_idx.x)            # one warp per (seq, head)
    if w >= n_s * nkv:
        return
    var h = w % nkv
    var s = w // nkv
    var lane = Int(thread_idx.x)
    var apos = base_pos + s
    var src_base = s * l_kvd + h * l_hd
    var cache_row = h * cache_cap + (apos % cache_cap)   # packed cache ring slot
    var hcache = cache_hd // 2
    var dst_base = cache_row * hcache
    var q8_row = h * q8_pitch + apos                     # decode-order linear view
    var q8_base = q8_row * cache_hd

    # K: one quantization feeds both the packed INT4 cache and the q8 attention view.
    var amk = Float32(0.0)
    var i = lane
    while i < l_hd:
        var v = abs(src_k[src_base + i])
        if v > amk:
            amk = v
        i += WARP_SIZE
    var ok = WARP_SIZE // 2
    while ok > 0:
        var o = shuffle_xor(amk, UInt32(ok))
        if o > amk:
            amk = o
        ok //= 2
    var sck = amk / Float32(7.0)
    if sck == Float32(0.0):
        sck = Float32(1.0)
    if lane == 0:
        k_scales[cache_row] = sck
        k_q8_scales[q8_row] = sck
    var ik = Float32(1.0) / sck
    var jb = lane
    while jb < hcache:
        var j0 = 2 * jb
        var j1 = j0 + 1
        var q0 = _round_i4(src_k[src_base + j0] * ik) if j0 < l_hd else 0
        var q1 = _round_i4(src_k[src_base + j1] * ik) if j1 < l_hd else 0
        dst_k_i4[dst_base + jb] = UInt8((q0 & 0xF) | ((q1 & 0xF) << 4))
        dst_k_q8[q8_base + j0] = Int8(q0)
        if j1 < cache_hd:
            dst_k_q8[q8_base + j1] = Int8(q1)
        jb += WARP_SIZE

    # V: same single producer for packed cache + q8 view.
    var amv = Float32(0.0)
    var iv = lane
    while iv < l_hd:
        var vv = abs(src_v[src_base + iv])
        if vv > amv:
            amv = vv
        iv += WARP_SIZE
    var ov = WARP_SIZE // 2
    while ov > 0:
        var o = shuffle_xor(amv, UInt32(ov))
        if o > amv:
            amv = o
        ov //= 2
    var scv = amv / Float32(7.0)
    if scv == Float32(0.0):
        scv = Float32(1.0)
    if lane == 0:
        v_scales[cache_row] = scv
        v_q8_scales[q8_row] = scv
    var ivd = Float32(1.0) / scv
    var jbv = lane
    while jbv < hcache:
        var j0v = 2 * jbv
        var j1v = j0v + 1
        var q0v = _round_i4(src_v[src_base + j0v] * ivd) if j0v < l_hd else 0
        var q1v = _round_i4(src_v[src_base + j1v] * ivd) if j1v < l_hd else 0
        dst_v_i4[dst_base + jbv] = UInt8((q0v & 0xF) | ((q1v & 0xF) << 4))
        dst_v_q8[q8_base + j0v] = Int8(q0v)
        if j1v < cache_hd:
            dst_v_q8[q8_base + j1v] = Int8(q1v)
        jbv += WARP_SIZE


# Q4_0-style codec: one fp16 scale per 32 values. Kept as separate kernels so
# the shipped one-fp32-scale-per-row layout remains byte-for-byte untouched.
def append_quant_kv_i4_block32_kernel(
    src_k: UnsafePointer[Float32, MutAnyOrigin],
    src_v: UnsafePointer[Float32, MutAnyOrigin],
    dst_k_i4: UnsafePointer[UInt8, MutAnyOrigin],
    dst_v_i4: UnsafePointer[UInt8, MutAnyOrigin],
    k_scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    n_s_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, l_kvd_arg: Int32, cache_cap_arg: Int32, cache_hd_arg: Int32, base_pos_arg: Int32,
):
    var n_s = Int(n_s_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var l_kvd = Int(l_kvd_arg)
    var cache_cap = Int(cache_cap_arg)
    var cache_hd = Int(cache_hd_arg)
    var base_pos = Int(base_pos_arg)
    var w = Int(block_idx.x)
    if w >= n_s * nkv:
        return
    var h = w % nkv
    var s = w // nkv
    var lane = Int(thread_idx.x)
    var src_base = s * l_kvd + h * l_hd
    var cache_row = h * cache_cap + ((base_pos + s) % cache_cap)
    var hcache = cache_hd // 2
    var dst_base = cache_row * hcache
    var nblocks = (l_hd + 31) // 32
    var k_q8_cache = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(k_scales) + nkv * cache_cap * nblocks * 2
    )
    var v_q8_cache = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(v_scales) + nkv * cache_cap * nblocks * 2
    )

    var row_recon_k = Float32(0.0)
    var row_recon_v = Float32(0.0)
    for b in range(nblocks):
        var d = b * 32 + lane
        var vk = src_k[src_base + d] if d < l_hd else Float32(0.0)
        var ak = abs(vk)
        var vv = src_v[src_base + d] if d < l_hd else Float32(0.0)
        var av = abs(vv)
        var off = WARP_SIZE // 2
        while off > 0:
            var ok = shuffle_xor(ak, UInt32(off))
            if ok > ak: ak = ok
            var ov = shuffle_xor(av, UInt32(off))
            if ov > av: av = ov
            off //= 2
        var sk = ak / Float32(7.0) if ak > 0 else Float32(1.0)
        var sv = av / Float32(7.0) if av > 0 else Float32(1.0)
        # Quantize with the rounded stored scale, matching q4_0 semantics.
        var sk16 = sk.cast[DType.float16]()
        var sv16 = sv.cast[DType.float16]()
        var skf = Float32(sk16)
        var svf = Float32(sv16)
        var recon_k = skf * Float32(7.0)
        var recon_v = svf * Float32(7.0)
        if recon_k > row_recon_k:
            row_recon_k = recon_k
        if recon_v > row_recon_v:
            row_recon_v = recon_v
        if lane == 0:
            k_scales[cache_row * nblocks + b] = sk16
            v_scales[cache_row * nblocks + b] = sv16
        if lane < 16:
            var d0 = b * 32 + 2 * lane
            var d1 = d0 + 1
            var qk0 = _round_i4(src_k[src_base + d0] / skf) if d0 < l_hd else 0
            var qk1 = _round_i4(src_k[src_base + d1] / skf) if d1 < l_hd else 0
            var qv0 = _round_i4(src_v[src_base + d0] / svf) if d0 < l_hd else 0
            var qv1 = _round_i4(src_v[src_base + d1] / svf) if d1 < l_hd else 0
            dst_k_i4[dst_base + d0 // 2] = UInt8((qk0 & 0xF) | ((qk1 & 0xF) << 4))
            dst_v_i4[dst_base + d0 // 2] = UInt8((qv0 & 0xF) | ((qv1 & 0xF) << 4))
    if lane == 0:
        k_q8_cache[cache_row] = row_recon_k / Float32(127.0) if row_recon_k > 0 else Float32(1.0)
        v_q8_cache[cache_row] = row_recon_v / Float32(127.0) if row_recon_v > 0 else Float32(1.0)


def append_quant_kv_i4_block32_with_q8_kernel(
    src_k: UnsafePointer[Float32, MutAnyOrigin],
    src_v: UnsafePointer[Float32, MutAnyOrigin],
    dst_k_i4: UnsafePointer[UInt8, MutAnyOrigin],
    dst_v_i4: UnsafePointer[UInt8, MutAnyOrigin],
    k_scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    dst_k_q8: UnsafePointer[Int8, MutAnyOrigin],
    dst_v_q8: UnsafePointer[Int8, MutAnyOrigin],
    k_q8_scales: UnsafePointer[Float32, MutAnyOrigin],
    v_q8_scales: UnsafePointer[Float32, MutAnyOrigin],
    n_s_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, l_kvd_arg: Int32, cache_cap_arg: Int32,
    cache_hd_arg: Int32, base_pos_arg: Int32, q8_pitch_arg: Int32,
):
    var n_s = Int(n_s_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var l_kvd = Int(l_kvd_arg)
    var cache_cap = Int(cache_cap_arg)
    var cache_hd = Int(cache_hd_arg)
    var base_pos = Int(base_pos_arg)
    var q8_pitch = Int(q8_pitch_arg)
    var w = Int(block_idx.x)
    if w >= n_s * nkv:
        return
    var h = w % nkv
    var s = w // nkv
    var lane = Int(thread_idx.x)
    var apos = base_pos + s
    var src_base = s * l_kvd + h * l_hd
    var cache_row = h * cache_cap + (apos % cache_cap)
    var dst_base = cache_row * (cache_hd // 2)
    var q8_row = h * q8_pitch + apos
    var q8_base = q8_row * cache_hd
    var nblocks = (l_hd + 31) // 32
    var k_q8_cache = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(k_scales) + nkv * cache_cap * nblocks * 2
    )
    var v_q8_cache = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(v_scales) + nkv * cache_cap * nblocks * 2
    )

    # First materialize the only representation retained by the cache: packed
    # q4 values plus fp16 block scales.  Accumulate the row Q8 scale from those
    # rounded fp16 scales in the same block order as block32_i4_q8_scales_kernel.
    var row_recon_k = Float32(0.0)
    var row_recon_v = Float32(0.0)
    for b in range(nblocks):
        var d = b * 32 + lane
        var vk = src_k[src_base + d] if d < l_hd else Float32(0.0)
        var ak = abs(vk)
        var vv = src_v[src_base + d] if d < l_hd else Float32(0.0)
        var av = abs(vv)
        var off = WARP_SIZE // 2
        while off > 0:
            var ok = shuffle_xor(ak, UInt32(off))
            if ok > ak:
                ak = ok
            var ov = shuffle_xor(av, UInt32(off))
            if ov > av:
                av = ov
            off //= 2
        var sk16 = (ak / Float32(7.0) if ak > 0 else Float32(1.0)).cast[DType.float16]()
        var sv16 = (av / Float32(7.0) if av > 0 else Float32(1.0)).cast[DType.float16]()
        var skf = Float32(sk16)
        var svf = Float32(sv16)
        var recon_k = skf * Float32(7.0)
        var recon_v = svf * Float32(7.0)
        if recon_k > row_recon_k:
            row_recon_k = recon_k
        if recon_v > row_recon_v:
            row_recon_v = recon_v
        if lane == 0:
            k_scales[cache_row * nblocks + b] = sk16
            v_scales[cache_row * nblocks + b] = sv16
        if lane < 16:
            var d0 = b * 32 + 2 * lane
            var d1 = d0 + 1
            var qk0 = _round_i4(src_k[src_base + d0] / skf) if d0 < l_hd else 0
            var qk1 = _round_i4(src_k[src_base + d1] / skf) if d1 < l_hd else 0
            var qv0 = _round_i4(src_v[src_base + d0] / svf) if d0 < l_hd else 0
            var qv1 = _round_i4(src_v[src_base + d1] / svf) if d1 < l_hd else 0
            dst_k_i4[dst_base + d0 // 2] = UInt8((qk0 & 0xF) | ((qk1 & 0xF) << 4))
            dst_v_i4[dst_base + d0 // 2] = UInt8((qv0 & 0xF) | ((qv1 & 0xF) << 4))

    var q8sk = row_recon_k / Float32(127.0) if row_recon_k > 0 else Float32(1.0)
    var q8sv = row_recon_v / Float32(127.0) if row_recon_v > 0 else Float32(1.0)
    if lane == 0:
        k_q8_scales[q8_row] = q8sk
        v_q8_scales[q8_row] = q8sv
        k_q8_cache[cache_row] = q8sk
        v_q8_cache[cache_row] = q8sv

    # Reconstruct the Q8 view from the cached q4 bytes and cached fp16 scales,
    # exactly as dequant_kv_i4_block32_to_q8_kernel will on a later decode.
    barrier()
    var d = lane
    while d < l_hd:
        var byte = Int(dst_k_i4[dst_base + d // 2])
        var nib = (byte & 0xF) if d % 2 == 0 else ((byte >> 4) & 0xF)
        var qk = nib - 16 if nib >= 8 else nib
        byte = Int(dst_v_i4[dst_base + d // 2])
        nib = (byte & 0xF) if d % 2 == 0 else ((byte >> 4) & 0xF)
        var qv = nib - 16 if nib >= 8 else nib
        var skf = Float32(k_scales[cache_row * nblocks + d // 32])
        var svf = Float32(v_scales[cache_row * nblocks + d // 32])
        dst_k_q8[q8_base + d] = Int8(_round_i8(Float32(qk) * skf / q8sk))
        dst_v_q8[q8_base + d] = Int8(_round_i8(Float32(qv) * svf / q8sv))
        d += WARP_SIZE


def block32_i4_q8_scales_kernel(
    scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    dst_scale: UnsafePointer[Float32, MutAnyOrigin],
    nkv: Int, klen: Int, nblocks: Int, max_seq: Int, cache_cap: Int, win_start: Int,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nkv * klen:
        var j = idx % klen; var h = idx // klen
        var src_row = h * cache_cap + ((win_start + j) % cache_cap)
        var am = Float32(0.0)
        for b in range(nblocks):
            var s = Float32(scales[src_row * nblocks + b]) * Float32(7.0)
            if s > am: am = s
        dst_scale[h * max_seq + j] = am / Float32(127.0) if am > 0 else Float32(1.0)


def dequant_kv_i4_block32_to_q8_kernel(
    src_i4: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    cached_q8_scales: UnsafePointer[Float32, MutAnyOrigin],
    dst_i8: UnsafePointer[Int8, MutAnyOrigin],
    dst_scale: UnsafePointer[Float32, MutAnyOrigin],
    nkv_arg: Int32, klen_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, cache_cap_arg: Int32, win_start_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var cache_cap = Int(cache_cap_arg)
    var win_start = Int(win_start_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nkv * klen * cache_hd:
        var d = idx % cache_hd; var t = idx // cache_hd
        var j = t % klen; var h = t // klen
        var src_row = h * cache_cap + ((win_start + j) % cache_cap)
        var dst_row = h * max_seq + j
        var byte = Int(src_i4[src_row * (cache_hd // 2) + d // 2])
        var nib = (byte & 0xF) if d % 2 == 0 else ((byte >> 4) & 0xF)
        var q4 = nib - 16 if nib >= 8 else nib
        var nblocks = (cache_hd + 31) // 32
        var bs = Float32(scales[src_row * nblocks + d // 32])
        var qs = cached_q8_scales[src_row]
        dst_i8[dst_row * cache_hd + d] = Int8(_round_i8(Float32(q4) * bs / qs))
        if d == 0:
            dst_scale[dst_row] = cached_q8_scales[src_row]


def dequant_kv_i4_layer_kernel(
    src_i4: UnsafePointer[UInt8, MutAnyOrigin],            # [nkv, cache_cap, cache_hd/2]
    scales: UnsafePointer[Float32, MutAnyOrigin],          # [nkv, cache_cap]
    dst_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],  # [nkv, max_seq, cache_hd]
    nkv_arg: Int32, klen_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, cache_cap_arg: Int32, win_start_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var cache_cap = Int(cache_cap_arg)
    var win_start = Int(win_start_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nkv * klen * cache_hd:
        var d = idx % cache_hd
        var t = idx // cache_hd
        var j = t % klen
        var h = t // klen
        var ring = (win_start + j) % cache_cap
        var src_row = h * cache_cap + ring
        var dst_row = h * max_seq + j
        var hcache = cache_hd // 2
        var b = Int(src_i4[src_row * hcache + (d // 2)])
        var nib = (b & 0xF) if (d % 2 == 0) else ((b >> 4) & 0xF)
        var q = nib - 16 if nib >= 8 else nib     # sign-extend 4-bit two's complement
        dst_bf16[dst_row * cache_hd + d] = (Float32(q) * scales[src_row]).cast[DType.bfloat16]()


# ── block-32 twin of dequant_kv_i4_layer_kernel. The append side (append_quant_kv_i4_block32_kernel)
#    writes one FLOAT16 scale per 32 values at [nkv, cache_cap, nblocks]; the block-1 kernel above
#    reads a FLOAT32 scale per row at [nkv, cache_cap]. Reading the former with the latter does not
#    merely lose granularity — it reinterprets two fp16 scales as one fp32, i.e. garbage. Before
#    this kernel existed, gpu_dequant_kv_i4_layer had no way to read a block-32 cache at all, so
#    NOMOS_KV_I4_BLOCK=32 was only correct when int8 attention happened to be selected. ──
def dequant_kv_i4_block32_layer_kernel(
    src_i4: UnsafePointer[UInt8, MutAnyOrigin],                     # [nkv, cache_cap, cache_hd/2]
    scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],     # [nkv, cache_cap, nblocks]
    dst_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],  # [nkv, max_seq, cache_hd]
    nkv_arg: Int32, klen_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, cache_cap_arg: Int32, win_start_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var cache_cap = Int(cache_cap_arg)
    var win_start = Int(win_start_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nkv * klen * cache_hd:
        var d = idx % cache_hd
        var t = idx // cache_hd
        var j = t % klen
        var h = t // klen
        var ring = (win_start + j) % cache_cap
        var src_row = h * cache_cap + ring
        var dst_row = h * max_seq + j
        var hcache = cache_hd // 2
        var b = Int(src_i4[src_row * hcache + (d // 2)])
        var nib = (b & 0xF) if (d % 2 == 0) else ((b >> 4) & 0xF)
        var q = nib - 16 if nib >= 8 else nib     # sign-extend 4-bit two's complement
        var nblocks = (cache_hd + 31) // 32
        var bs = Float32(scales[src_row * nblocks + (d // 32)])
        dst_bf16[dst_row * cache_hd + d] = (Float32(q) * bs).cast[DType.bfloat16]()


def gpu_append_quant_kv_i4(
    ctx: DeviceContext, src_k: UInt64, src_v: UInt64,
    dst_k_i4: UInt64, dst_v_i4: UInt64, k_scales: UInt64, v_scales: UInt64,
    n_s: Int, nkv: Int, l_hd: Int, l_kvd: Int, cache_cap: Int, cache_hd: Int, base_pos: Int,
    scale_blocks: Int = 1,
) raises:
    """INT4 twin of gpu_append_quant_kv_i8 (packs 2 nibbles/byte; cache is cache_hd/2
    bytes; ring slot p%cache_cap)."""
    if scale_blocks > 1:
        var bk = ctx.compile_function[append_quant_kv_i4_block32_kernel]()
        ctx.enqueue_function(
            bk,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_k)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_v)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_k_i4)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_v_i4)),
            UnsafePointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=Int(k_scales)),
            UnsafePointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=Int(v_scales)),
            Int32(n_s), Int32(nkv), Int32(l_hd), Int32(l_kvd), Int32(cache_cap), Int32(cache_hd), Int32(base_pos),
            grid_dim=n_s * nkv, block_dim=WARP_SIZE,
        )
        return
    var kern = ctx.compile_function[append_quant_kv_i4_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_k)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_v)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_k_i4)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_v_i4)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(k_scales)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(v_scales)),
        Int32(n_s), Int32(nkv), Int32(l_hd), Int32(l_kvd), Int32(cache_cap), Int32(cache_hd), Int32(base_pos),
        grid_dim=n_s * nkv, block_dim=WARP_SIZE,
    )


def gpu_append_quant_kv_i4_with_q8(
    ctx: DeviceContext, src_k: UInt64, src_v: UInt64,
    dst_k_i4: UInt64, dst_v_i4: UInt64, k_scales: UInt64, v_scales: UInt64,
    dst_k_q8: UInt64, dst_v_q8: UInt64, k_q8_scales: UInt64, v_q8_scales: UInt64,
    n_s: Int, nkv: Int, l_hd: Int, l_kvd: Int, cache_cap: Int,
    cache_hd: Int, base_pos: Int, q8_pitch: Int, scale_blocks: Int = 1,
) raises:
    """Append INT4 KV and simultaneously publish the same quantized row as int8 for
    decode-order verify attention. The q8 row is absolute (`base_pos+s`), so prefix
    rows can be dequantized once and newly appended rows become visible immediately."""
    if scale_blocks > 1:
        var bk = ctx.compile_function[append_quant_kv_i4_block32_with_q8_kernel]()
        ctx.enqueue_function(
            bk,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_k)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_v)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_k_i4)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_v_i4)),
            UnsafePointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=Int(k_scales)),
            UnsafePointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=Int(v_scales)),
            UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_k_q8)),
            UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_v_q8)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(k_q8_scales)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(v_q8_scales)),
            Int32(n_s), Int32(nkv), Int32(l_hd), Int32(l_kvd), Int32(cache_cap), Int32(cache_hd), Int32(base_pos), Int32(q8_pitch),
            grid_dim=n_s * nkv, block_dim=WARP_SIZE,
        )
        return
    var kern = ctx.compile_function[append_quant_kv_i4_with_q8_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_k)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src_v)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_k_i4)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst_v_i4)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(k_scales)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(v_scales)),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_k_q8)),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_v_q8)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(k_q8_scales)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(v_q8_scales)),
        Int32(n_s), Int32(nkv), Int32(l_hd), Int32(l_kvd), Int32(cache_cap), Int32(cache_hd), Int32(base_pos), Int32(q8_pitch),
        grid_dim=n_s * nkv, block_dim=WARP_SIZE,
    )


def gpu_dequant_kv_i4_layer(
    ctx: DeviceContext, src_i4: UInt64, scales: UInt64, dst_bf16: UInt64,
    nkv: Int, klen: Int, cache_hd: Int, max_seq: Int, cache_cap: Int, win_start: Int,
    scale_blocks: Int = 1,
) raises:
    """INT4 twin of gpu_dequant_kv_i8_layer (unpacks 2 nibbles/byte; window->linear).

    scale_blocks>1 selects the block-32 scale layout. The append side always writes whichever
    layout the engine configured, so this MUST be passed through wherever gpu_append_quant_kv_i4
    is given a non-1 i4_scale_blocks — otherwise fp16 block scales get read as fp32 row scales,
    which is garbage KV rather than degraded KV, and it fails silently."""
    var threads = 256
    if scale_blocks > 1:
        var bk = ctx.compile_function[dequant_kv_i4_block32_layer_kernel]()
        var bblocks = (nkv * klen * cache_hd + threads - 1) // threads
        ctx.enqueue_function(
            bk,
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(src_i4)),
            UnsafePointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=Int(scales)),
            UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(dst_bf16)),
            Int32(nkv), Int32(klen), Int32(cache_hd), Int32(max_seq), Int32(cache_cap), Int32(win_start),
            grid_dim=bblocks, block_dim=threads,
        )
        return
    var kern = ctx.compile_function[dequant_kv_i4_layer_kernel]()
    var blocks = (nkv * klen * cache_hd + threads - 1) // threads
    ctx.enqueue_function(
        kern,
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(src_i4)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scales)),
        UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(dst_bf16)),
        Int32(nkv), Int32(klen), Int32(cache_hd), Int32(max_seq), Int32(cache_cap), Int32(win_start),
        grid_dim=blocks, block_dim=threads,
    )


# ── LAW-compliant int8 KV read (the path A, int8 variant): sign-extend the INT4 nibble to int8
#    (lossless — int4 ⊂ int8) + carry the per-row scale, mirroring the i4 dequant's window/ring
#    indexing so the int8 attention reads a clean linear [nkv, klen, cache_hd] buffer. No bf16/fp32
#    intermediate, no SWA fork in the attention kernel. ──
def dequant_kv_i4_to_q8_layer_kernel(
    src_i4: UnsafePointer[UInt8, MutAnyOrigin],            # [nkv, cache_cap, cache_hd/2]
    scales: UnsafePointer[Float32, MutAnyOrigin],          # [nkv, cache_cap]
    dst_i8: UnsafePointer[Int8, MutAnyOrigin],             # [nkv, max_seq, cache_hd] int8 (sign-ext nibble)
    dst_scale: UnsafePointer[Float32, MutAnyOrigin],       # [nkv, max_seq] per-row scale (linear, windowed)
    nkv_arg: Int32, klen_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, cache_cap_arg: Int32, win_start_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var cache_cap = Int(cache_cap_arg)
    var win_start = Int(win_start_arg)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < nkv * klen * cache_hd:
        var d = idx % cache_hd
        var t = idx // cache_hd
        var j = t % klen
        var h = t // klen
        var ring = (win_start + j) % cache_cap
        var src_row = h * cache_cap + ring
        var dst_row = h * max_seq + j
        var hcache = cache_hd // 2
        var b = Int(src_i4[src_row * hcache + (d // 2)])
        var nib = (b & 0xF) if (d % 2 == 0) else ((b >> 4) & 0xF)
        var q = nib - 16 if nib >= 8 else nib     # sign-extend 4-bit two's complement -> -8..7
        dst_i8[dst_row * cache_hd + d] = Int8(q)
        if d == 0:
            dst_scale[dst_row] = scales[src_row]


def gpu_dequant_kv_i4_to_q8_layer(
    ctx: DeviceContext, src_i4: UInt64, scales: UInt64, dst_i8: UInt64, dst_scale: UInt64,
    nkv: Int, klen: Int, cache_hd: Int, max_seq: Int, cache_cap: Int, win_start: Int,
    scale_blocks: Int = 1,
) raises:
    """INT4 KV -> int8 (lossless widen) + linear per-row scales — the law-compliant KV read for the
    int8 attention (vs gpu_dequant_kv_i4_layer which emits bf16, a precision-law violation)."""
    var threads = 256
    if scale_blocks > 1:
        var bk = ctx.compile_function[dequant_kv_i4_block32_to_q8_kernel]()
        var bblocks = (nkv * klen * cache_hd + threads - 1) // threads
        var cached_scale_addr = scales + UInt64(nkv * cache_cap * scale_blocks * 2)
        ctx.enqueue_function(
            bk,
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(src_i4)),
            UnsafePointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=Int(scales)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(cached_scale_addr)),
            UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_i8)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(dst_scale)),
            Int32(nkv), Int32(klen), Int32(cache_hd), Int32(max_seq), Int32(cache_cap), Int32(win_start),
            grid_dim=bblocks, block_dim=threads,
        )
        return
    var kern = ctx.compile_function[dequant_kv_i4_to_q8_layer_kernel]()
    var blocks = (nkv * klen * cache_hd + threads - 1) // threads
    ctx.enqueue_function(
        kern,
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(src_i4)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scales)),
        UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(dst_i8)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(dst_scale)),
        Int32(nkv), Int32(klen), Int32(cache_hd), Int32(max_seq), Int32(cache_cap), Int32(win_start),
        grid_dim=blocks, block_dim=threads,
    )


def debug_kv_i4_source_to_f32_kernel(
    k_i4: UnsafePointer[UInt8, MutAnyOrigin],      # [nkv, cache_cap, l_hd/2]
    v_i4: UnsafePointer[UInt8, MutAnyOrigin],
    k_scales: UnsafePointer[Float32, MutAnyOrigin], # [nkv, cache_cap]
    v_scales: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],    # Kraw, Vraw, Kscale, Vscale
    nkv_arg: Int32,
    klen_arg: Int32,
    l_hd_arg: Int32,
    cache_cap_arg: Int32,
    win_start_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var l_hd = Int(l_hd_arg)
    var cache_cap = Int(cache_cap_arg)
    var win_start = Int(win_start_arg)
    var hcache = (l_hd + 1) // 2
    var q_count = nkv * klen * hcache
    var s_count = nkv * klen
    var total = 2 * q_count + 2 * s_count
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= total:
        return
    if idx < q_count:
        var b = idx % hcache
        var t = idx // hcache
        var key = t % klen
        var h = t // klen
        var ring = (win_start + key) % cache_cap
        var row = h * cache_cap + ring
        outp[idx] = Float32(Int(k_i4[row * hcache + b]))
    elif idx < 2 * q_count:
        var rel = idx - q_count
        var b = rel % hcache
        var t = rel // hcache
        var key = t % klen
        var h = t // klen
        var ring = (win_start + key) % cache_cap
        var row = h * cache_cap + ring
        outp[idx] = Float32(Int(v_i4[row * hcache + b]))
    elif idx < 2 * q_count + s_count:
        var rel = idx - 2 * q_count
        var key = rel % klen
        var h = rel // klen
        var ring = (win_start + key) % cache_cap
        var row = h * cache_cap + ring
        outp[idx] = k_scales[row]
    else:
        var rel = idx - 2 * q_count - s_count
        var key = rel % klen
        var h = rel // klen
        var ring = (win_start + key) % cache_cap
        var row = h * cache_cap + ring
        outp[idx] = v_scales[row]


def debug_kv_i4_source_to_f32_dev(
    ctx: DeviceContext,
    d_k_i4: UInt64,
    d_v_i4: UInt64,
    d_k_scales: UInt64,
    d_v_scales: UInt64,
    d_out: UInt64,
    nkv: Int,
    klen: Int,
    l_hd: Int,
    cache_cap: Int,
    win_start: Int,
) raises:
    if nkv <= 0 or klen <= 0 or l_hd <= 0 or cache_cap <= 0:
        return
    var hcache = (l_hd + 1) // 2
    var q_count = nkv * klen * hcache
    var s_count = nkv * klen
    var total = 2 * q_count + 2 * s_count
    var threads = 256
    var blocks = (total + threads - 1) // threads
    var kern = ctx.compile_function[debug_kv_i4_source_to_f32_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(d_k_i4)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(d_v_i4)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_k_scales)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_v_scales)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out)),
        Int32(nkv),
        Int32(klen),
        Int32(l_hd),
        Int32(cache_cap),
        Int32(win_start),
        grid_dim=blocks,
        block_dim=threads,
    )
