"""Law-compliant INT8 attention for the Q4 perf path.

This mirrors the existing materialized attention structure:
  1. FP32 Q -> q8 per 32-lane block, int8 Q·K -> FP32 raw scores.
  2. Existing FP32 softmax kernels.
  3. Quantize p*Vscale per 32 keys, int8 scores·V -> FP32 output.

There is deliberately no 1/sqrt(head_dim) and no attention softcap here. The
engine's existing bf16/fp32 attention is raw GEMM -> raw softmax -> GEMM.
"""

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from std.math import round, exp
from layout import row_major
from layout import stack_allocation

from lib.softmax_gpu_mojo import gpu_softmax_over_heads_mojo
from lib.batched_attn_gpu import gpu_block_transpose, gpu_softmax_causal, gpu_softmax_causal_prefix

comptime MAX_WARPS = 32


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_i8_ptr(addr: UInt64) -> UnsafePointer[Int8, MutAnyOrigin]:
    return UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=Int(addr))


def _clamp_i8_symmetric(x: Int) -> Int8:
    if x > 127:
        return Int8(127)
    if x < -127:
        return Int8(-127)
    return Int8(x)


def q_to_i8(x: Float32, scale: Float32) -> Int8:
    var qi = Int(round(x / scale))
    return _clamp_i8_symmetric(qi)


def qk_decode_int8_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],       # [nh, l_hd]
    k_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, max_seq, cache_hd]
    k_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, max_seq]
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [nh, seq_len]
    nh_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, seq_len_arg: Int32, kvg_arg: Int32,
):
    var nh = Int(nh_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var seq_len = Int(seq_len_arg)
    var kvg = Int(kvg_arg)
    var h = Int(block_idx.x)
    if h >= nh:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var kvh = h // kvg
    var nqb = (l_hd + 31) // 32
    var key = tid
    while key < seq_len:
        var acc = Float32(0.0)
        for b in range(nqb):
            var amax = Float32(0.0)
            for ii in range(32):
                var d = b * 32 + ii
                if d < l_hd:
                    var v = q[h * l_hd + d]
                    var av = v if v >= 0 else -v
                    if av > amax:
                        amax = av
            var dq = amax / 127.0 if amax > 0 else Float32(1.0)
            var isum = Int32(0)
            for ii in range(32):
                var d = b * 32 + ii
                if d < l_hd:
                    var q8 = q_to_i8(q[h * l_hd + d], dq)
                    var kval = k_i8[(kvh * max_seq + key) * cache_hd + d]
                    isum += Int32(Int(q8)) * Int32(Int(kval))
            acc += dq * k_scale[kvh * max_seq + key] * Float32(isum)
        scores[h * seq_len + key] = acc
        key += n_threads


def pv_decode_int8_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [nh, seq_len], post-softmax
    v_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, max_seq, cache_hd]
    v_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, max_seq]
    outp: UnsafePointer[Float32, MutAnyOrigin],    # [nh, l_hd]
    nh_arg: Int32, nkv_arg: Int32, l_hd_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, seq_len_arg: Int32, kvg_arg: Int32,
):
    var nh = Int(nh_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var seq_len = Int(seq_len_arg)
    var kvg = Int(kvg_arg)
    var h = Int(block_idx.x)
    if h >= nh:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var kvh = h // kvg
    var nkb = (seq_len + 31) // 32
    var d = tid
    while d < l_hd:
        var acc = Float32(0.0)
        for b in range(nkb):
            var amax = Float32(0.0)
            for ii in range(32):
                var key = b * 32 + ii
                if key < seq_len:
                    var pv = scores[h * seq_len + key] * v_scale[kvh * max_seq + key]
                    var apv = pv if pv >= 0 else -pv
                    if apv > amax:
                        amax = apv
            var dp = amax / 127.0 if amax > 0 else Float32(1.0)
            var isum = Int32(0)
            for ii in range(32):
                var key = b * 32 + ii
                if key < seq_len:
                    var q8p = q_to_i8(scores[h * seq_len + key] * v_scale[kvh * max_seq + key], dp)
                    var vv = v_i8[(kvh * max_seq + key) * cache_hd + d]
                    isum += Int32(Int(q8p)) * Int32(Int(vv))
            acc += dp * Float32(isum)
        outp[h * l_hd + d] = acc
        d += n_threads


def debug_q_i8_to_f32_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],       # [nh, l_hd]
    outp: UnsafePointer[Float32, MutAnyOrigin],    # [nh*l_hd q8-as-f32][nh*nqb dq]
    nh_arg: Int32,
    l_hd_arg: Int32,
):
    var nh = Int(nh_arg)
    var l_hd = Int(l_hd_arg)
    var nqb = (l_hd + 31) // 32
    var hb = Int(block_idx.x)
    if hb >= nh * nqb:
        return
    if Int(thread_idx.x) != 0:
        return
    var h = hb // nqb
    var b = hb % nqb
    var amax = Float32(0.0)
    for ii in range(32):
        var d = b * 32 + ii
        if d < l_hd:
            var v = q[h * l_hd + d]
            var av = v if v >= 0 else -v
            if av > amax:
                amax = av
    var dq = amax / 127.0 if amax > 0 else Float32(1.0)
    outp[nh * l_hd + hb] = dq
    for ii in range(32):
        var d = b * 32 + ii
        if d < l_hd:
            var q8 = q_to_i8(q[h * l_hd + d], dq)
            outp[h * l_hd + d] = Float32(Int(q8))


def debug_kv_i8_view_to_f32_kernel(
    k_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, pitch, l_hd], possibly offset to lo
    v_i8: UnsafePointer[Int8, MutAnyOrigin],
    k_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, pitch], possibly offset to lo
    v_scale: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],    # Kq8, Vq8, Kscale, Vscale
    nkv_arg: Int32,
    klen_arg: Int32,
    l_hd_arg: Int32,
    pitch_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var klen = Int(klen_arg)
    var l_hd = Int(l_hd_arg)
    var pitch = Int(pitch_arg)
    var q_count = nkv * klen * l_hd
    var s_count = nkv * klen
    var total = 2 * q_count + 2 * s_count
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= total:
        return
    if idx < q_count:
        var d = idx % l_hd
        var t = idx // l_hd
        var key = t % klen
        var h = t // klen
        outp[idx] = Float32(Int(k_i8[(h * pitch + key) * l_hd + d]))
    elif idx < 2 * q_count:
        var rel = idx - q_count
        var d = rel % l_hd
        var t = rel // l_hd
        var key = t % klen
        var h = t // klen
        outp[idx] = Float32(Int(v_i8[(h * pitch + key) * l_hd + d]))
    elif idx < 2 * q_count + s_count:
        var rel = idx - 2 * q_count
        var key = rel % klen
        var h = rel // klen
        outp[idx] = k_scale[h * pitch + key]
    else:
        var rel = idx - 2 * q_count - s_count
        var key = rel % klen
        var h = rel // klen
        outp[idx] = v_scale[h * pitch + key]


def qk_batched_int8_kernel(
    q_t: UnsafePointer[Float32, MutAnyOrigin],     # [nkv, S, kvg*l_hd]
    k_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, max_seq, cache_hd]
    k_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, max_seq]
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [nkv, S*kvg, key_len]
    nkv_arg: Int32, l_hd_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, S_arg: Int32, key_len_arg: Int32, kvg_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var S = Int(S_arg)
    var key_len = Int(key_len_arg)
    var kvg = Int(kvg_arg)
    var row = Int(block_idx.x)
    var total_rows = nkv * S * kvg
    if row >= total_rows:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var rows_per_kv = S * kvg
    var kvh = row // rows_per_kv
    var r = row % rows_per_kv
    var s = r // kvg
    var g = r % kvg
    var q_base = (kvh * S + s) * (kvg * l_hd) + g * l_hd
    var nqb = (l_hd + 31) // 32
    var key = tid
    while key < key_len:
        var acc = Float32(0.0)
        for b in range(nqb):
            var amax = Float32(0.0)
            for ii in range(32):
                var d = b * 32 + ii
                if d < l_hd:
                    var v = q_t[q_base + d]
                    var av = v if v >= 0 else -v
                    if av > amax:
                        amax = av
            var dq = amax / 127.0 if amax > 0 else Float32(1.0)
            var isum = Int32(0)
            for ii in range(32):
                var d = b * 32 + ii
                if d < l_hd:
                    var q8 = q_to_i8(q_t[q_base + d], dq)
                    var kval = k_i8[(kvh * max_seq + key) * cache_hd + d]
                    isum += Int32(Int(q8)) * Int32(Int(kval))
            acc += dq * k_scale[kvh * max_seq + key] * Float32(isum)
        scores[row * key_len + key] = acc
        key += n_threads


def pv_batched_int8_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [nkv, S*kvg, key_len]
    v_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, max_seq, cache_hd]
    v_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, max_seq]
    out_t: UnsafePointer[Float32, MutAnyOrigin],   # [nkv, S, kvg*l_hd]
    nkv_arg: Int32, l_hd_arg: Int32, cache_hd_arg: Int32, max_seq_arg: Int32, S_arg: Int32, key_len_arg: Int32, kvg_arg: Int32,
):
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var S = Int(S_arg)
    var key_len = Int(key_len_arg)
    var kvg = Int(kvg_arg)
    var row = Int(block_idx.x)
    var total_rows = nkv * S * kvg
    if row >= total_rows:
        return
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var rows_per_kv = S * kvg
    var kvh = row // rows_per_kv
    var r = row % rows_per_kv
    var s = r // kvg
    var g = r % kvg
    var nkb = (key_len + 31) // 32
    var d = tid
    while d < l_hd:
        var acc = Float32(0.0)
        for b in range(nkb):
            var amax = Float32(0.0)
            for ii in range(32):
                var key = b * 32 + ii
                if key < key_len:
                    var pv = scores[row * key_len + key] * v_scale[kvh * max_seq + key]
                    var apv = pv if pv >= 0 else -pv
                    if apv > amax:
                        amax = apv
            var dp = amax / 127.0 if amax > 0 else Float32(1.0)
            var isum = Int32(0)
            for ii in range(32):
                var key = b * 32 + ii
                if key < key_len:
                    var q8p = q_to_i8(scores[row * key_len + key] * v_scale[kvh * max_seq + key], dp)
                    var vv = v_i8[(kvh * max_seq + key) * cache_hd + d]
                    isum += Int32(Int(q8p)) * Int32(Int(vv))
            acc += dp * Float32(isum)
        out_t[(kvh * S + s) * (kvg * l_hd) + g * l_hd + d] = acc
        d += n_threads


def qk_decode_multirow_int8_kernel(
    q_b: UnsafePointer[Float32, MutAnyOrigin],     # [S, nh, l_hd]
    k_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, max_seq, cache_hd]
    k_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, max_seq]
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [S, nh, score_stride]
    nh_arg: Int32,
    nkv_arg: Int32,
    l_hd_arg: Int32,
    cache_hd_arg: Int32,
    max_seq_arg: Int32,
    S_arg: Int32,
    score_stride_arg: Int32,
    kvg_arg: Int32,
    window_arg: Int32,
    query_base_pos_arg: Int32,
):
    var nh = Int(nh_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var S = Int(S_arg)
    var score_stride = Int(score_stride_arg)
    var kvg = Int(kvg_arg)
    var window = Int(window_arg)
    var query_base_pos = Int(query_base_pos_arg)
    var row_head = Int(block_idx.x)
    var total = S * nh
    if row_head >= total:
        return
    var row_i = row_head // nh
    var h = row_head % nh
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var kvh = h // kvg
    var apos = query_base_pos + row_i
    var lo = 0
    if window > 0 and apos - window + 1 > 0:
        lo = apos - window + 1
    var klen = apos + 1 - lo
    if klen <= 0:
        return

    var nqb = (l_hd + 31) // 32
    var q_base = row_i * nh * l_hd + h * l_hd
    var row_off = row_head * score_stride
    var key = tid
    while key < klen:
        var abs_key = lo + key
        var acc = Float32(0.0)
        for b in range(nqb):
            var amax = Float32(0.0)
            for ii in range(32):
                var d = b * 32 + ii
                if d < l_hd:
                    var v = q_b[q_base + d]
                    var av = v if v >= 0 else -v
                    if av > amax:
                        amax = av
            var dq = amax / 127.0 if amax > 0 else Float32(1.0)
            var isum = Int32(0)
            for ii in range(32):
                var d = b * 32 + ii
                if d < l_hd:
                    var q8 = q_to_i8(q_b[q_base + d], dq)
                    var kval = k_i8[(kvh * max_seq + abs_key) * cache_hd + d]
                    isum += Int32(Int(q8)) * Int32(Int(kval))
            acc += dq * k_scale[kvh * max_seq + abs_key] * Float32(isum)
        scores[row_off + key] = acc
        key += n_threads


def softmax_decode_multirow_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [S, nh, score_stride]
    total_rows_arg: Int32,
    nh_arg: Int32,
    score_stride_arg: Int32,
    window_arg: Int32,
    query_base_pos_arg: Int32,
):
    var total_rows = Int(total_rows_arg)
    var nh = Int(nh_arg)
    var score_stride = Int(score_stride_arg)
    var window = Int(window_arg)
    var query_base_pos = Int(query_base_pos_arg)
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[MAX_WARPS]())

    var row_head = Int(block_idx.x)
    if row_head >= total_rows:
        return
    var tid = Int(thread_idx.x)
    var row_i = row_head // nh
    var apos = query_base_pos + row_i
    var lo = 0
    if window > 0 and apos - window + 1 > 0:
        lo = apos - window + 1
    var klen = apos + 1 - lo
    if klen <= 0:
        return

    # Match gpu_softmax_over_heads_mojo's per-row launch geometry even though
    # this batched kernel has one physical block size for all rows.
    var active_threads = 256
    if klen < 256:
        active_threads = ((klen + 31) // 32) * 32
        if active_threads < 32:
            active_threads = 32
    if active_threads > 1024:
        active_threads = 1024
    if active_threads > Int(block_dim.x):
        active_threads = Int(block_dim.x)
    var active = tid < active_threads
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n_warps = (active_threads + WARP_SIZE - 1) // WARP_SIZE
    var row_off = row_head * score_stride

    var local_max = Scalar[DType.float32](-3.4e38)
    if active:
        var i = tid
        while i < klen:
            var v = Scalar[DType.float32](scores[row_off + i])
            if v > local_max:
                local_max = v
            i += active_threads

    var warp_off = WARP_SIZE // 2
    while warp_off > 0:
        var other = shuffle_xor(local_max, UInt32(warp_off))
        if other > local_max:
            local_max = other
        warp_off //= 2

    if active and lane == 0:
        smem[wid] = local_max
    barrier()

    if wid == 0:
        var partial = Scalar[DType.float32](-3.4e38)
        if lane < n_warps:
            partial = smem[lane]
        var off2 = WARP_SIZE // 2
        while off2 > 0:
            var other = shuffle_xor(partial, UInt32(off2))
            if other > partial:
                partial = other
            off2 //= 2
        if lane == 0:
            smem[0] = partial
    barrier()
    var row_max = smem[0]

    var local_sum = Scalar[DType.float32](0.0)
    if active:
        var j = tid
        while j < klen:
            var v = exp(Scalar[DType.float32](scores[row_off + j]) - row_max)
            scores[row_off + j] = Float32(v)
            local_sum += v
            j += active_threads

    var warp_total = local_sum
    var so = WARP_SIZE // 2
    while so > 0:
        warp_total += shuffle_xor(warp_total, UInt32(so))
        so //= 2
    if active and lane == 0:
        smem[wid] = warp_total
    barrier()
    if wid == 0:
        var partial2 = Scalar[DType.float32](0.0)
        if lane < n_warps:
            partial2 = smem[lane]
        var so2 = WARP_SIZE // 2
        while so2 > 0:
            partial2 += shuffle_xor(partial2, UInt32(so2))
            so2 //= 2
        if lane == 0:
            smem[0] = partial2
    barrier()
    var row_sum = smem[0]
    var inv = Scalar[DType.float32](1.0) / row_sum

    if active:
        var k = tid
        while k < klen:
            scores[row_off + k] = Float32(Scalar[DType.float32](scores[row_off + k]) * inv)
            k += active_threads


def pv_decode_multirow_int8_kernel(
    scores: UnsafePointer[Float32, MutAnyOrigin],  # [S, nh, score_stride]
    v_i8: UnsafePointer[Int8, MutAnyOrigin],       # [nkv, max_seq, cache_hd]
    v_scale: UnsafePointer[Float32, MutAnyOrigin], # [nkv, max_seq]
    outp: UnsafePointer[Float32, MutAnyOrigin],    # [S, nh, l_hd]
    nh_arg: Int32,
    nkv_arg: Int32,
    l_hd_arg: Int32,
    cache_hd_arg: Int32,
    max_seq_arg: Int32,
    S_arg: Int32,
    score_stride_arg: Int32,
    kvg_arg: Int32,
    window_arg: Int32,
    query_base_pos_arg: Int32,
):
    var nh = Int(nh_arg)
    var nkv = Int(nkv_arg)
    var l_hd = Int(l_hd_arg)
    var cache_hd = Int(cache_hd_arg)
    var max_seq = Int(max_seq_arg)
    var S = Int(S_arg)
    var score_stride = Int(score_stride_arg)
    var kvg = Int(kvg_arg)
    var window = Int(window_arg)
    var query_base_pos = Int(query_base_pos_arg)
    var row_head = Int(block_idx.x)
    var total = S * nh
    if row_head >= total:
        return
    var row_i = row_head // nh
    var h = row_head % nh
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)
    var kvh = h // kvg
    var apos = query_base_pos + row_i
    var lo = 0
    if window > 0 and apos - window + 1 > 0:
        lo = apos - window + 1
    var klen = apos + 1 - lo
    if klen <= 0:
        return

    var nkb = (klen + 31) // 32
    var row_off = row_head * score_stride
    var out_base = row_i * nh * l_hd + h * l_hd
    var d = tid
    while d < l_hd:
        var acc = Float32(0.0)
        for b in range(nkb):
            var amax = Float32(0.0)
            for ii in range(32):
                var key = b * 32 + ii
                if key < klen:
                    var abs_key = lo + key
                    var pv = scores[row_off + key] * v_scale[kvh * max_seq + abs_key]
                    var apv = pv if pv >= 0 else -pv
                    if apv > amax:
                        amax = apv
            var dp = amax / 127.0 if amax > 0 else Float32(1.0)
            var isum = Int32(0)
            for ii in range(32):
                var key = b * 32 + ii
                if key < klen:
                    var abs_key = lo + key
                    var q8p = q_to_i8(scores[row_off + key] * v_scale[kvh * max_seq + abs_key], dp)
                    var vv = v_i8[(kvh * max_seq + abs_key) * cache_hd + d]
                    isum += Int32(Int(q8p)) * Int32(Int(vv))
            acc += dp * Float32(isum)
        outp[out_base + d] = acc
        d += n_threads


def _threads_for_cols(n: Int) -> Int:
    var threads = 256
    if n < 256:
        threads = ((n + 31) // 32) * 32
        if threads < 32:
            threads = 32
    if threads > 1024:
        threads = 1024
    return threads


def attention_gpu_int8_dev(
    ctx: DeviceContext,
    d_q_gpu: UInt64,
    d_k_i8: UInt64,
    d_v_i8: UInt64,
    d_k_scale: UInt64,
    d_v_scale: UInt64,
    d_scores_scratch: UInt64,
    d_attn_out_scratch: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    seq_len: Int,
    kvg: Int,
) raises:
    if nh <= 0 or seq_len <= 0:
        return
    var qk = ctx.compile_function[qk_decode_int8_kernel]()
    ctx.enqueue_function(
        qk, _as_f32_ptr(d_q_gpu), _as_i8_ptr(d_k_i8), _as_f32_ptr(d_k_scale),
        _as_f32_ptr(d_scores_scratch),
        Int32(nh), Int32(nkv), Int32(l_hd), Int32(cache_hd), Int32(max_seq), Int32(seq_len), Int32(kvg),
        grid_dim=nh, block_dim=_threads_for_cols(seq_len),
    )
    gpu_softmax_over_heads_mojo(ctx, d_scores_scratch, nh, seq_len)
    var pv = ctx.compile_function[pv_decode_int8_kernel]()
    ctx.enqueue_function(
        pv, _as_f32_ptr(d_scores_scratch), _as_i8_ptr(d_v_i8), _as_f32_ptr(d_v_scale),
        _as_f32_ptr(d_attn_out_scratch),
        Int32(nh), Int32(nkv), Int32(l_hd), Int32(cache_hd), Int32(max_seq), Int32(seq_len), Int32(kvg),
        grid_dim=nh, block_dim=_threads_for_cols(l_hd),
    )


def attention_decode_order_multirow_int8(
    ctx: DeviceContext,
    d_q_b: UInt64,
    d_scores: UInt64,
    d_attn_b: UInt64,
    d_k_i8: UInt64,
    d_v_i8: UInt64,
    d_k_scale: UInt64,
    d_v_scale: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    score_stride: Int,
    kvg: Int,
    window: Int,
    query_base_pos: Int,
) raises:
    if nh <= 0 or S <= 0 or score_stride <= 0:
        return
    var total_rows = nh * S
    var qk = ctx.compile_function[qk_decode_multirow_int8_kernel]()
    ctx.enqueue_function(
        qk,
        _as_f32_ptr(d_q_b),
        _as_i8_ptr(d_k_i8),
        _as_f32_ptr(d_k_scale),
        _as_f32_ptr(d_scores),
        Int32(nh),
        Int32(nkv),
        Int32(l_hd),
        Int32(cache_hd),
        Int32(max_seq),
        Int32(S),
        Int32(score_stride),
        Int32(kvg),
        Int32(window),
        Int32(query_base_pos),
        grid_dim=total_rows,
        block_dim=_threads_for_cols(score_stride),
    )
    var sm = ctx.compile_function[softmax_decode_multirow_kernel]()
    ctx.enqueue_function(
        sm,
        _as_f32_ptr(d_scores),
        Int32(total_rows),
        Int32(nh),
        Int32(score_stride),
        Int32(window),
        Int32(query_base_pos),
        grid_dim=total_rows,
        block_dim=_threads_for_cols(score_stride),
    )
    var pv = ctx.compile_function[pv_decode_multirow_int8_kernel]()
    ctx.enqueue_function(
        pv,
        _as_f32_ptr(d_scores),
        _as_i8_ptr(d_v_i8),
        _as_f32_ptr(d_v_scale),
        _as_f32_ptr(d_attn_b),
        Int32(nh),
        Int32(nkv),
        Int32(l_hd),
        Int32(cache_hd),
        Int32(max_seq),
        Int32(S),
        Int32(score_stride),
        Int32(kvg),
        Int32(window),
        Int32(query_base_pos),
        grid_dim=total_rows,
        block_dim=_threads_for_cols(l_hd),
    )


def debug_qk_scores_int8_dev(
    ctx: DeviceContext,
    d_q_gpu: UInt64,
    d_k_i8: UInt64,
    d_k_scale: UInt64,
    d_scores_out: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    seq_len: Int,
    kvg: Int,
) raises:
    if nh <= 0 or seq_len <= 0:
        return
    var qk = ctx.compile_function[qk_decode_int8_kernel]()
    ctx.enqueue_function(
        qk,
        _as_f32_ptr(d_q_gpu),
        _as_i8_ptr(d_k_i8),
        _as_f32_ptr(d_k_scale),
        _as_f32_ptr(d_scores_out),
        Int32(nh),
        Int32(nkv),
        Int32(l_hd),
        Int32(cache_hd),
        Int32(max_seq),
        Int32(seq_len),
        Int32(kvg),
        grid_dim=nh,
        block_dim=_threads_for_cols(seq_len),
    )


def debug_q_i8_to_f32_dev(
    ctx: DeviceContext,
    d_q_gpu: UInt64,
    d_out: UInt64,
    nh: Int,
    l_hd: Int,
) raises:
    var nqb = (l_hd + 31) // 32
    if nh <= 0 or l_hd <= 0:
        return
    var kern = ctx.compile_function[debug_q_i8_to_f32_kernel]()
    ctx.enqueue_function(
        kern,
        _as_f32_ptr(d_q_gpu),
        _as_f32_ptr(d_out),
        Int32(nh),
        Int32(l_hd),
        grid_dim=nh * nqb,
        block_dim=32,
    )


def debug_kv_i8_view_to_f32_dev(
    ctx: DeviceContext,
    d_k_i8: UInt64,
    d_v_i8: UInt64,
    d_k_scale: UInt64,
    d_v_scale: UInt64,
    d_out: UInt64,
    nkv: Int,
    klen: Int,
    l_hd: Int,
    pitch: Int,
) raises:
    if nkv <= 0 or klen <= 0 or l_hd <= 0:
        return
    var q_count = nkv * klen * l_hd
    var s_count = nkv * klen
    var total = 2 * q_count + 2 * s_count
    var threads = 256
    var blocks = (total + threads - 1) // threads
    var kern = ctx.compile_function[debug_kv_i8_view_to_f32_kernel]()
    ctx.enqueue_function(
        kern,
        _as_i8_ptr(d_k_i8),
        _as_i8_ptr(d_v_i8),
        _as_f32_ptr(d_k_scale),
        _as_f32_ptr(d_v_scale),
        _as_f32_ptr(d_out),
        Int32(nkv),
        Int32(klen),
        Int32(l_hd),
        Int32(pitch),
        grid_dim=blocks,
        block_dim=threads,
    )


def attention_prefill_batched_int8(
    ctx: DeviceContext,
    d_q_b: UInt64,
    d_q_t: UInt64,
    d_scores: UInt64,
    d_attn_t: UInt64,
    d_attn_b: UInt64,
    d_k_i8: UInt64,
    d_v_i8: UInt64,
    d_k_scale: UInt64,
    d_v_scale: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    kvg: Int,
    window: Int,
) raises:
    var blk = kvg * l_hd
    gpu_block_transpose(ctx, d_q_b, d_q_t, S, nkv, blk)
    var qk = ctx.compile_function[qk_batched_int8_kernel]()
    ctx.enqueue_function(
        qk, _as_f32_ptr(d_q_t), _as_i8_ptr(d_k_i8), _as_f32_ptr(d_k_scale),
        _as_f32_ptr(d_scores),
        Int32(nkv), Int32(l_hd), Int32(cache_hd), Int32(max_seq), Int32(S), Int32(S), Int32(kvg),
        grid_dim=nkv * S * kvg, block_dim=_threads_for_cols(S),
    )
    gpu_softmax_causal(ctx, d_scores, nh * S, S, kvg, window)
    var pv = ctx.compile_function[pv_batched_int8_kernel]()
    ctx.enqueue_function(
        pv, _as_f32_ptr(d_scores), _as_i8_ptr(d_v_i8), _as_f32_ptr(d_v_scale),
        _as_f32_ptr(d_attn_t),
        Int32(nkv), Int32(l_hd), Int32(cache_hd), Int32(max_seq), Int32(S), Int32(S), Int32(kvg),
        grid_dim=nkv * S * kvg, block_dim=_threads_for_cols(l_hd),
    )
    gpu_block_transpose(ctx, d_attn_t, d_attn_b, nkv, S, blk)


def attention_prefill_prefix_batched_int8(
    ctx: DeviceContext,
    d_q_b: UInt64,
    d_q_t: UInt64,
    d_scores: UInt64,
    d_attn_t: UInt64,
    d_attn_b: UInt64,
    d_k_i8: UInt64,
    d_v_i8: UInt64,
    d_k_scale: UInt64,
    d_v_scale: UInt64,
    nh: Int,
    nkv: Int,
    l_hd: Int,
    cache_hd: Int,
    max_seq: Int,
    S: Int,
    key_len: Int,
    kvg: Int,
    window: Int,
    query_base_pos: Int,
    key_base_pos: Int,
) raises:
    var blk = kvg * l_hd
    gpu_block_transpose(ctx, d_q_b, d_q_t, S, nkv, blk)
    var qk = ctx.compile_function[qk_batched_int8_kernel]()
    ctx.enqueue_function(
        qk, _as_f32_ptr(d_q_t), _as_i8_ptr(d_k_i8), _as_f32_ptr(d_k_scale),
        _as_f32_ptr(d_scores),
        Int32(nkv), Int32(l_hd), Int32(cache_hd), Int32(max_seq), Int32(S), Int32(key_len), Int32(kvg),
        grid_dim=nkv * S * kvg, block_dim=_threads_for_cols(key_len),
    )
    gpu_softmax_causal_prefix(
        ctx, d_scores, nh * S, S, key_len, kvg, window,
        query_base_pos, key_base_pos,
    )
    var pv = ctx.compile_function[pv_batched_int8_kernel]()
    ctx.enqueue_function(
        pv, _as_f32_ptr(d_scores), _as_i8_ptr(d_v_i8), _as_f32_ptr(d_v_scale),
        _as_f32_ptr(d_attn_t),
        Int32(nkv), Int32(l_hd), Int32(cache_hd), Int32(max_seq), Int32(S), Int32(key_len), Int32(kvg),
        grid_dim=nkv * S * kvg, block_dim=_threads_for_cols(l_hd),
    )
    gpu_block_transpose(ctx, d_attn_t, d_attn_b, nkv, S, blk)
