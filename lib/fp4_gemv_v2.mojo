"""Unit A (GLM) — NVFP4 M=1 wave-front GEMV v2 + persistent + lm_head greedy fusion.

Tightening of lib/fp4_weights.nvfp4_gemv_kernel toward the weight-HBM roofline, plus the
lm_head "path" (large-N, persistent, sampling-fused) for the decode loop. Targets the
deployment box: GB10 / sm_121 (warp-MMA, unified mem) — NOT sm_100/tcgen05 (that is an
optional rented-B200 path, guarded elsewhere, deprioritized 2026-06-24).

Decode is HARD weight-HBM-bound (compute ~2% of FP32 peak). So the levers here are:
  (1) COALESCED, MINIMAL weight read — the existing kernel already nails this (256 B
      contiguous / warp-step). v2 keeps it bit-for-bit.
  (2) Shared-memory activation tiling — the existing kernel reads the K-vector activation
      STRIDED (lane l reads inp[l*16+t] → 32 lanes at 64 B stride = uncoalesced per FMA).
      v2 stages the activation in shared once per block (cooperative coalesced load) and
      serves all WARPS_PER_BLOCK rows from it. Fixes the only uncoalesced traffic. On GB10
      unified memory this is a real latency win.
  (3) Persistent grid-stride (lm_head) — fixed grid instead of N/WPB blocks → less block
      dispatch + tail, and CUDA-graph-capturable for the ~300 GEMVs/token loop (Unit E).
  (4) lm_head greedy fusion — GEMV + softcap + per-block max → tiny scratch → 1-block
      reduce. Kills the [1,262144]=1 MB logit write+read AND the host sampling round-trip.
      Bit-identical GREEDY token (argmax over the same dot products). Dense-logit fallback
      for true stochastic top_p (pending the A<->D gpu_sample seam, under adjudication).

PARITY: every variant computes each output row's dot product with the SAME per-lane
scale-block assignment (b = lane, lane+32, ...) and SAME FMA order as
nvfp4_gemv_kernel — only the activation source (shared vs global) and dispatch shape
differ. Floating-point addition is non-associative, so the per-lane block set + order is
preserved exactly → bit-identical output. The greedy fusion preserves argmax (max is
associative; ties broken toward the lower index, deterministic).

Same device-address ABI + NVFP4 blob layout as lib/fp4_weights (dequant math reused via
import — single source of truth for the codec). Requires Mojo >=1.0.0b3 +
--target-accelerator sm_121a.
"""
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer, bitcast
from layout import row_major
from layout import stack_allocation

# Reuse the codec + ptr helpers from fp4_weights — ONE source of truth (parity).
from lib.fp4_weights import (
    e4m3_decode,
    e2m1_mag,
    FP4_GROUP,
    NVFP4_WARPS_PER_BLOCK,
    _as_u8_ptr,
    _as_f32_ptr,
)
from lib.engine_init import (
    _env_flag_cached, ENV_NVFP4_V3, ENV_NVFP4_V3_TREE,
)


# ─────────────────────────────────────────────────────────────────────────
# Compile-time tuning knobs (baked per arch — no runtime autotune dispatch).
# ─────────────────────────────────────────────────────────────────────────
comptime WPB_V2: Int = 4           # warps per block (occupancy: matches NVFP4_WARPS_PER_BLOCK)
comptime KILE: Int = 4096          # K-tile in elements (mult of 16 + 32; 4096*4 B = 16 KiB smem)
comptime PERSISTENT_BLOCKS: Int = 64   # fixed grid for the persistent / lm_head variant
# lm_head greedy fusion knobs (scaffold; pending A<->D gpu_sample seam).
comptime GREEDY_CHUNK: Int = 256       # output rows reduced per block before emitting a max
comptime GREEDY_WPB: Int = 8           # warps per greedy block (GREEDY_CHUNK/GREEDY_WPB rows/warp)


# ═════════════════════════════════════════════════════════════════════════════
# Persistent grid-stride GEMV (lm_head / large-N path). Fixed grid =
# PERSISTENT_BLOCKS, blocks stride over output rows. Same per-row math →
# bit-identical. Fewer blocks → less dispatch + tail; CUDA-graph-capturable.
# ═════════════════════════════════════════════════════════════════════════════
def nvfp4_gemv_persistent_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    global_scale: Float32,
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var xs = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[KILE]())

    var ntiles = (K + KILE - 1) // KILE
    # Grid-stride over row-chunks of WPB_V2 rows. Each chunk: full tile loop (x staged
    # once per tile, shared across the chunk's rows), then write.
    var chunk_base = Int(block_idx.x) * WPB_V2
    while chunk_base < N:
        var n = chunk_base + warp_id
        var local = Scalar[DType.float32](0.0)
        if n < N:
            var row_block0 = (n * K) // FP4_GROUP
            var packed_base = nb + (n * K) // 2
            for tile in range(ntiles):
                var kt0 = tile * KILE
                var kt1 = kt0 + KILE
                if kt1 > K:
                    kt1 = K
                var klen = kt1 - kt0
                var t = tid
                while t < klen:
                    xs[t] = inp[kt0 + t]
                    t += Int(block_dim.x)
                barrier()
                var blk_lo = kt0 // FP4_GROUP
                var blk_hi = kt1 // FP4_GROUP
                var b = blk_lo + lane
                while b < blk_hi:
                    var eff = global_scale * e4m3_decode(Int(blob[row_block0 + b]))
                    var pk = (blob + (packed_base + b * (FP4_GROUP // 2))).bitcast[UInt64]()[0]
                    var xb = b * FP4_GROUP - kt0
                    for tt in range(FP4_GROUP):
                        var nib = Int((pk >> UInt64(4 * tt)) & 0xF)
                        var sign = Float32(-1.0) if (nib & 8) != 0 else Float32(1.0)
                        local += eff * sign * e2m1_mag(nib & 7) * xs[xb + tt]
                    b += WARP_SIZE
                barrier()
        var off = WARP_SIZE // 2
        while off > 0:
            local += shuffle_xor(local, UInt32(off))
            off //= 2
        if lane == 0 and n < N:
            outp[n] = Float32(local)
        chunk_base += Int(grid_dim.x) * WPB_V2


def gpu_matmul_nvfp4_persistent_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_nvfp4: UInt64,
    global_scale: Float32,
    K: Int,
    N: Int,
) raises:
    """Persistent grid-stride GEMV dispatch (lm_head / large N). Bit-identical to the v1
    fused path; fixed grid = PERSISTENT_BLOCKS. Best for the D->VOCAB=262144 lm_head and
    for CUDA-graph capture of the decode loop (coordinate Unit E)."""
    var nb = (K * N) // FP4_GROUP
    var kern = ctx.compile_function[nvfp4_gemv_persistent_kernel]()
    var blocks = PERSISTENT_BLOCKS if N > PERSISTENT_BLOCKS * WPB_V2 else (N + WPB_V2 - 1) // WPB_V2
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_nvfp4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        global_scale, Int32(K), Int32(N), Int32(nb),
        grid_dim=blocks, block_dim=WPB_V2 * WARP_SIZE,
    )


# ═════════════════════════════════════════════════════════════════════════════
# lm_head GREEDY fusion (scaffold — pending the A<->D gpu_sample seam, under
# adjudication with Unit D). GEMV body + softcap fused in epilogue + per-block max
# -> scratch[num_chunks] -> 1-block reduce -> token id. Kills the [1,VOCAB]=1 MB
# logit write+read + the host sampling round-trip. Bit-identical GREEDY token.
# Dense-logit fallback (true stochastic top_p) uses gpu_matmul_nvfp4_persistent_dev.
#
# rep_penalty window + full top_p stay on Unit D's side of the seam; this kernel
# accepts an optional rep-window ptr + penalty (unused=0/1.0) so the signature is
# ready, but the greedy path here applies softcap only (the common reasoning case).
# ═════════════════════════════════════════════════════════════════════════════

# Greedy scratch: two parallel arrays per chunk — scratch_score[num_chunks] (Float32)
# and scratch_idx[num_chunks] (Int32), holding each block's (max logit, argmax row).
# Plain shared-memory linear scan for the argmax (no packed-Int64 trick — keeps the
# tie-break toward the lower index explicit + obviously correct, GPU-untested-safe).


def nvfp4_lm_head_greedy_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # weight [N,K] NVFP4
    inp: UnsafePointer[Float32, MutAnyOrigin],    # activation, K elems
    scratch_score: UnsafePointer[Float32, MutAnyOrigin],  # [num_chunks] block-max logit
    scratch_idx: UnsafePointer[Int32, MutAnyOrigin],      # [num_chunks] block-argmax row
    global_scale: Float32,
    softcap: Float32,            # logit softcap (tanh) magnitude; <=0 disables
    rep_window: UnsafePointer[Int32, MutAnyOrigin],   # last-N token ids (0 ptr = unused)
    rep_len_arg: Int32,                # # ids in rep_window (0 = unused)
    rep_penalty: Float32,        # multiplier applied to window-token logits (1.0 = none)
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
):
    """Per-block: compute GREEDY_CHUNK rows' dot products (shared-x tiled GEMV body),
    apply softcap (+ rep_penalty to window tokens), keep the block max (score, idx) via a
    shared-memory scan, emit ONE (score, idx) to scratch[block]. Bit-identical greedy
    token vs dense argmax-over-softcapped-logits (max is associative; ties → lower idx)."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var rep_len = Int(rep_len_arg)
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var chunk = Int(block_idx.x)
    var chunk_start = chunk * GREEDY_CHUNK
    var gscores = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[GREEDY_CHUNK]())
    var gidx = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[GREEDY_CHUNK]())
    var stage_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1024]())   # block_dim upper bound (256 here); per-thread best staging
    var stage_i = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[1024]())

    # Each warp grid-strides within [chunk_start, chunk_start+GREEDY_CHUNK), one row at a
    # time (r = warp_id, warp_id+GREEDY_WPB, ...). All warps do exactly
    # GREEDY_CHUNK/GREEDY_WPB iterations so the trailing barrier stays aligned.
    var nbk = K // FP4_GROUP
    var r = warp_id
    while r < GREEDY_CHUNK:
        var n = chunk_start + r
        var local = Scalar[DType.float32](0.0)
        if n < N:
            var row_block0 = (n * K) // FP4_GROUP
            var packed_base = nb + (n * K) // 2
            # v3-style GEMV body: full-K reduction, DIRECT global activation read (the K-vector
            # is ~21 KB -> L2-resident, so per-row re-read is cheap and the shared-x tile +
            # barrier overhead is not worth it), VECTORIZED E2M1 decode, sequential t=0..15
            # accumulation (parity: bit-identical to v1/v3).
            var b = lane
            while b < nbk:
                var eff = global_scale * e4m3_decode(Int(blob[row_block0 + b]))
                var pbase = packed_base + b * (FP4_GROUP // 2)
                var ibase = b * FP4_GROUP
                var pk = (blob + pbase).bitcast[UInt64]()[0]
                var wvec = _e2m1_nibbles_to_fp32(pk)
                var act = (inp + ibase).load[width=FP4_GROUP]()
                var contrib = (wvec * eff) * act
                @parameter
                for t in range(FP4_GROUP):
                    local += contrib[t]
                b += WARP_SIZE
            # Warp-reduce the dot product.
            var off = WARP_SIZE // 2
            while off > 0:
                local += shuffle_xor(local, UInt32(off))
                off //= 2
        # Epilogue (lane 0): softcap + rep_penalty + store (score, idx) into gscores/gidx[r].
        if lane == 0:
            var score = Float32(local)
            if n < N:
                if softcap > Float32(0.0):
                    # tanh softcap: score = softcap * tanh(score / softcap). Padé approx.
                    var z = score / softcap
                    var z2 = z * z
                    var tanh_z = z * (Float32(27.0) + z2) / (Float32(27.0) + Float32(9.0) * z2)
                    if tanh_z > Float32(1.0):
                        tanh_z = Float32(1.0)
                    elif tanh_z < Float32(-1.0):
                        tanh_z = Float32(-1.0)
                    score = softcap * tanh_z
                if rep_len > 0 and rep_penalty != Float32(1.0):
                    var hit = False
                    for w in range(rep_len):
                        if Int(rep_window[w]) == n:
                            hit = True
                            break
                    if hit:
                        score *= rep_penalty
            else:
                score = Float32(-3.0e38)   # out-of-range row: never the argmax
            gscores[r] = score
            gidx[r] = Int32(n)
        barrier()   # gscores/gidx[r] visible before the block-max scan; also ends the tile xs use
        r += GREEDY_WPB

    # Block argmax over gscores[0:live] / gidx[0:live]. Ties → lower idx (strict >).
    var live = GREEDY_CHUNK
    if chunk_start + live > N:
        live = N - chunk_start
    var my_score = Float32(-3.0e38)
    var my_idx = Int32(-1)
    var i = tid
    while i < live:
        var s = gscores[i]
        if s > my_score or (s == my_score and gidx[i] < my_idx):
            my_score = s
            my_idx = gidx[i]
        i += Int(block_dim.x)
    # Stage per-thread bests; one warp scans for the block argmax.
    stage_s[tid] = my_score
    stage_i[tid] = my_idx
    barrier()
    var wid = tid // WARP_SIZE
    var lane2 = tid % WARP_SIZE
    var nthr = Int(block_dim.x)
    if wid == 0:
        var bs = Float32(-3.0e38)
        var bi = Int32(-1)
        var j = lane2
        while j < nthr:
            var s = stage_s[j]
            if s > bs or (s == bs and stage_i[j] < bi):
                bs = s
                bi = stage_i[j]
            j += WARP_SIZE
        # Cross-lane butterfly reduce of the 32 lane-local maxes into lane 0
        # (the strided scan above only partitions elements; without this, lane 0 sees
        # just indices 0,32,64,... and the true max on another lane is lost).
        var off = WARP_SIZE // 2
        while off > 0:
            var os = shuffle_xor(bs, UInt32(off))
            var oi = shuffle_xor(bi, UInt32(off))
            if os > bs or (os == bs and oi < bi):
                bs = os
                bi = oi
            off //= 2
        if lane2 == 0:
            scratch_score[chunk] = bs
            scratch_idx[chunk] = bi


def greedy_reduce_kernel(
    scratch_score: UnsafePointer[Float32, MutAnyOrigin],  # [num_chunks]
    scratch_idx: UnsafePointer[Int32, MutAnyOrigin],      # [num_chunks]
    out_token: UnsafePointer[Int32, MutAnyOrigin],        # [1] argmax token id
    num_chunks_arg: Int32,
):
    """1-block reduce: argmax over the per-chunk (score, idx) -> global greedy token id."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var num_chunks = Int(num_chunks_arg)
    var tid = Int(thread_idx.x)
    var my_score = Float32(-3.0e38)
    var my_idx = Int32(-1)
    var i = tid
    while i < num_chunks:
        var s = scratch_score[i]
        if s > my_score or (s == my_score and scratch_idx[i] < my_idx):
            my_score = s
            my_idx = scratch_idx[i]
        i += Int(block_dim.x)
    var stage_s = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[1024]())
    var stage_i = stack_allocation[
        dtype=DType.int32, address_space=AddressSpace.SHARED
    ](row_major[1024]())
    stage_s[tid] = my_score
    stage_i[tid] = my_idx
    barrier()
    var wid = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var nthr = Int(block_dim.x)
    if wid == 0:
        var bs = Float32(-3.0e38)
        var bi = Int32(-1)
        var j = lane
        while j < nthr:
            var s = stage_s[j]
            if s > bs or (s == bs and stage_i[j] < bi):
                bs = s
                bi = stage_i[j]
            j += WARP_SIZE
        # Cross-lane butterfly reduce of the 32 lane-local maxes into lane 0
        # (same fix as the block-argmax scan above).
        var off = WARP_SIZE // 2
        while off > 0:
            var os = shuffle_xor(bs, UInt32(off))
            var oi = shuffle_xor(bi, UInt32(off))
            if os > bs or (os == bs and oi < bi):
                bs = os
                bi = oi
            off //= 2
        if lane == 0:
            out_token[0] = bi


def gpu_lm_head_greedy_dev(
    ctx: DeviceContext,
    d_in_fp32: UInt64,         # activation K elems
    d_w_nvfp4: UInt64,         # weight [N,K] NVFP4
    d_scratch_score: UInt64,   # [num_chunks] Float32 scratch (caller-allocated, reused)
    d_scratch_idx: UInt64,     # [num_chunks] Int32 scratch (caller-allocated, reused)
    d_out_token: UInt64,       # [1] Int32 token id
    global_scale: Float32,
    softcap: Float32,          # <=0 disables
    d_rep_window: UInt64,      # 0 = unused
    rep_len: Int,
    rep_penalty: Float32,
    K: Int,
    N: Int,
) raises:
    """lm_head greedy fusion: GEMV + softcap + per-block max -> scratch -> reduce -> token.
    Bit-identical greedy token vs dense argmax-over-softcapped-logits. SCAFFOLD: the
    rep_penalty window + true top_p stay on Unit D's side of the seam.
    Scratch = num_chunks*4 bytes each (score, idx); caller-owned (reuse across tokens)."""
    var nb = (K * N) // FP4_GROUP
    var num_chunks = (N + GREEDY_CHUNK - 1) // GREEDY_CHUNK
    var rep_ptr = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(0))
    if d_rep_window != 0:
        rep_ptr = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(d_rep_window))
    var kern = ctx.compile_function[nvfp4_lm_head_greedy_kernel]()
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_nvfp4),
        _as_f32_ptr(d_in_fp32),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_scratch_score)),
        UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(d_scratch_idx)),
        global_scale, softcap, rep_ptr, Int32(rep_len), rep_penalty,
        Int32(K), Int32(N), Int32(nb),
        grid_dim=num_chunks, block_dim=GREEDY_WPB * WARP_SIZE,
    )
    var rk = ctx.compile_function[greedy_reduce_kernel]()
    ctx.enqueue_function(
        rk,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_scratch_score)),
        UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(d_scratch_idx)),
        UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(d_out_token)),
        Int32(num_chunks),
        grid_dim=1, block_dim=128,
    )


# ═════════════════════════════════════════════════════════════════════════════
# v3 GEMV: VECTORIZED ARITHMETIC E2M1 DECODE (the real raw-kernel lever).
#
# v1/v2 measured ~3.3-3.6x OVER the HBM roofline on every HBM-bound shape (UNIT_A_RESULTS_v1.md):
# the bottleneck is NOT memory (weight read is already coalesced+minimal) but the PER-ELEMENT
# SCALAR DEQUANT in v1's inner loop (16 scalar iters/block: nibble shift+mask, sign branch,
# e2m1_mag branch). v3 vectorizes that decode via a BRANCHLESS ARITHMETIC E2M1->fp32 bit-decode
# (no LUT gather, no hardware cvt.rn.f16x2.e2m1x2 — both unavailable on sm_121/GB10).
#
# Why arithmetic, not a LUT: E2M1 is NON-AFFINE (levels 0,±.5,±1,±1.5,±2,±3,±4,±6), so the
# affine (q-zp)*scale INT4 trick does not apply. And a 16-entry register-vector LUT gather does
# NOT vectorize on sm_121 (no vector gather; the per-element LUT compiles to a slow select-chain).
# The arithmetic decode extracts sign/exp(2)/mant(1) as SIMD bitfields and constructs the fp32 by
# SIMD shift/OR into the exponent field + a single branchless select for the exp==0 subnormal/
# zero case. Structure mirrors MAX per_channel_grouped_4bit._decode_bits (vectorized whole-SIMD);
# decode math replaces its affine scale*(q-8) with the E2M1 bit-decode.
#
# E2M1 -> fp32 (verified all 8 magnitudes): for normal (exp>=1), value=(2+m)*2^(e-2) ->
#   fp32_bits = sign | ((126+e)<<23) | (m<<22)   [bias 127; (2+m)=2*(1+m/2), m/2 -> mant bit 22]
# For subnormal (exp==0): value=m*0.5 -> m=1: 0x3F000000 (0.5); m=0: 0 (signed zero).
#
# PARITY (greedy-lossless gate): the per-lane scale-block assignment (b=lane,lane+32,...) AND the
# sequential t=0..15 accumulation order are BOTH identical to v1's nvfp4_gemv_kernel, so the
# output is BIT-IDENTICAL. The elementwise product (eff*wvec)*act per element is bit-identical to
# v1's ((eff*sign)*mag)*inp (sign-multiply is exact, so the reassociation is exact). Only the
# per-element BRANCHY decode is replaced by branchless SIMD — same fp32 results, same order.
# ═════════════════════════════════════════════════════════════════════════════

@always_inline
def _e2m1_nibbles_to_fp32(pk: UInt64) -> SIMD[DType.float32, FP4_GROUP]:
    """Branchless arithmetic decode of 16 packed E2M1 nibbles (one UInt64) -> SIMD[f32,16].
    Bit-identical to v1's per-element (sign*e2m1_mag) for all 16 values."""
    # Unpack 16 nibbles (comptime-unrolled shift+mask -> SIMD[uint8,16], then widen to uint32).
    var nib_u8 = SIMD[DType.uint8, FP4_GROUP]()
    @parameter
    for t in range(FP4_GROUP):
        nib_u8[t] = UInt8((pk >> UInt64(4 * t)) & 0xF)
    var nib = nib_u8.cast[DType.uint32]()
    # Bitfields: bit3=sign, bits2:1=exp(2), bit0=mant(1).
    var s = (nib >> UInt32(3)) & UInt32(1)
    var e = (nib >> UInt32(1)) & UInt32(3)
    var m = nib & UInt32(1)
    var sign_bits = s << UInt32(31)
    # Normal (e>=1): fp32 = sign | ((126+e)<<23) | (m<<22).
    var normal_bits = sign_bits | ((e + UInt32(126)) << UInt32(23)) | (m << UInt32(22))
    # Subnormal (e==0): m=1 -> 0.5 (0x3F000000), m=0 -> 0. Use MAX's .lt(scalar).select idiom
    # (== returns a scalar Bool on this toolchain; .lt returns SIMD[bool] with .select).
    var sub_half = m.lt(UInt32(1)).select(
        SIMD[DType.uint32, FP4_GROUP](0), SIMD[DType.uint32, FP4_GROUP](0x3F000000)
    )
    var sub_bits = sign_bits | sub_half
    var bits = e.lt(UInt32(1)).select(sub_bits, normal_bits)
    return bitcast[DType.float32, FP4_GROUP](bits)


def nvfp4_gemv_v3_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [e4m3 scales : nb][E2M1 packed : n/2]
    inp: UnsafePointer[Float32, MutAnyOrigin],    # activation, K elems, fp32
    outp: UnsafePointer[Float32, MutAnyOrigin],   # output, N elems, fp32
    global_scale: Float32,
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,                                       # scale blocks = N*K/16
    row_base_arg: Int32 = 0,                             # first output row in this launch
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var row_base = Int(row_base_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    # row_base mirrors v1: N=VOCAB=262144 at WPB=4 is exactly 65536 blocks, and that grid
    # illegal-addresses on the sm_120a path while ordinary projection sizes pass. Cover the
    # vocabulary with row-offset launches instead. Row math and output are unchanged, so this
    # stays bit-identical to v1 for every shape.
    var n = row_base + Int(block_idx.x) * NVFP4_WARPS_PER_BLOCK + warp_id
    if n >= N:
        return
    var row_block0 = (n * K) // FP4_GROUP
    var packed_base = nb + (n * K) // 2
    var nbk = K // FP4_GROUP                   # 16-weight blocks per row (K % 16 == 0)

    var local = Scalar[DType.float32](0.0)
    var b = lane
    while b < nbk:
        var eff = global_scale * e4m3_decode(Int(blob[row_block0 + b]))   # ONCE / 16 weights
        var pbase = packed_base + b * (FP4_GROUP // 2)                     # 16 nibbles = 8 bytes
        var ibase = b * FP4_GROUP
        var pk = (blob + pbase).bitcast[UInt64]()[0]
        # VECTORIZED branchless E2M1 decode -> 16 fp32 weights.
        var wvec = _e2m1_nibbles_to_fp32(pk)
        # VECTOR LOAD of the 16 contiguous activation values (coalesced: 64 B/warp-step).
        var act = (inp + ibase).load[width=FP4_GROUP]()
        # Elementwise product (vectorized): (eff * wvec) * act.
        var contrib = (wvec * eff) * act
        # SEQUENTIAL t=0..15 accumulation (parity: same order as v1 -> bit-identical).
        @parameter
        for t in range(FP4_GROUP):
            local += contrib[t]
        b += WARP_SIZE

    # Warp-shuffle butterfly reduce (identical to v1).
    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


comptime NVFP4_GEMV_SMAX = 8   # multirow verify readout: rows batched per launch
# OUTPUT rows per warp. NCU on Gold/sm_120 (Codex, 095224b1) says both this kernel and the S=1
# v3 are L1/TEX load-path bound — lg_throttle is 66%/73% of the entire issue gap, DRAM is only
# 4.97%/22.35% and compute 14.57%/32.70%, with zero spills. The limiter is activation load
# INSTRUCTIONS, not bytes: every output warp re-traverses all K activations, so 704 MiB (S=1) /
# 3.44 GiB (S=5) of logical activation traffic against ~99 MiB of weights. It caches, so DRAM
# stays idle while the LSU queue saturates.
# One warp owning R output rows loads each activation block ONCE and applies it to R weight
# rows, cutting activation load instructions per output by R. Weight loads per output are
# unchanged, and every (s,n) keeps its exact b/t accumulation order -> still bit-exact.
# R is capped by registers: ptxas reports 48/thread today, and R weight vectors stay live.
comptime NVFP4_GEMV_ROWS = 4


def nvfp4_gemv_v3_multirow_kernel[SMAX: Int](
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [e4m3 scales : nb][E2M1 packed : n/2]
    inp: UnsafePointer[Float32, MutAnyOrigin],    # [S, K] fp32 activations (row-major)
    outp: UnsafePointer[Float32, MutAnyOrigin],   # [S, N] fp32 outputs
    global_scale: Float32,
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
    S_arg: Int32,                                 # live rows, 1..SMAX
    row_base_arg: Int32 = 0,
):
    """S-row GEMV that reads each weight ONCE and reuses it across all S activations.

    WHY: the NVFP4 verify readout called the single-row GEMV once per row, and each call
    re-streams the WHOLE 749 MB embedding matrix to produce one row of logits. At S=5 that
    is ~3.7 GB of weight traffic for 5 rows. Measured on Gold: read_lmhead = 23.22 ms/call,
    31.6% of the whole verify. Reading the weights once cuts that by ~S.

    BIT-EXACT vs the single-row v3 kernel, by construction and not by hope:
      - the weight decode (eff, pk, wvec) is row-INDEPENDENT, so hoisting it out of the row
        loop changes no value, only how many times it is computed;
      - the per-(s,n) arithmetic is the identical expression `(wvec * eff) * act` followed by
        the identical SEQUENTIAL t=0..15 accumulation, over the identical ascending-b block
        sequence, reduced by the identical warp-shuffle butterfly.
    So each output element sees exactly the operations, in exactly the order, that v3 gives it
    -- which is what made v3 safe to swap in for v1 in the first place.
    """
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var S = Int(S_arg)
    var row_base = Int(row_base_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    # This warp owns NVFP4_GEMV_ROWS consecutive output rows starting at n0.
    var n0 = row_base + (Int(block_idx.x) * NVFP4_WARPS_PER_BLOCK + warp_id) * NVFP4_GEMV_ROWS
    if n0 >= N:
        return
    var nbk = K // FP4_GROUP

    # accumulators: [row r][activation s]
    var local = InlineArray[SIMD[DType.float32, SMAX], NVFP4_GEMV_ROWS](uninitialized=True)
    @parameter
    for r in range(NVFP4_GEMV_ROWS):
        local[r] = SIMD[DType.float32, SMAX](0.0)
    var b = lane
    while b < nbk:
        var ibase = b * FP4_GROUP
        # decode this block's weights for each of the R output rows this warp owns
        var weff = InlineArray[SIMD[DType.float32, FP4_GROUP], NVFP4_GEMV_ROWS](
            uninitialized=True)
        @parameter
        for r in range(NVFP4_GEMV_ROWS):
            var nr = n0 + r
            if nr < N:
                var eff = global_scale * e4m3_decode(Int(blob[((nr * K) // FP4_GROUP) + b]))
                var pk = (blob + (nb + (nr * K) // 2) + b * (FP4_GROUP // 2)
                          ).bitcast[UInt64]()[0]
                weff[r] = _e2m1_nibbles_to_fp32(pk) * eff   # == v3's (wvec * eff)
        # ONE activation load per (s, block), applied to all R rows -> R x fewer LG loads/output
        @parameter
        for s in range(SMAX):
            if s < S:
                var act = (inp + s * K + ibase).load[width=FP4_GROUP]()
                @parameter
                for r in range(NVFP4_GEMV_ROWS):
                    if n0 + r < N:
                        var contrib = weff[r] * act      # == v3's (wvec * eff) * act
                        @parameter
                        for t in range(FP4_GROUP):
                            local[r][s] += contrib[t]    # same sequential t order as v3
        b += WARP_SIZE

    @parameter
    for r in range(NVFP4_GEMV_ROWS):
        if n0 + r < N:
            @parameter
            for s in range(SMAX):
                if s < S:
                    var acc = local[r][s]
                    var off = WARP_SIZE // 2
                    while off > 0:
                        acc += shuffle_xor(acc, UInt32(off))
                        off //= 2
                    if lane == 0:
                        outp[s * N + (n0 + r)] = Float32(acc)


def gpu_matmul_nvfp4_fused_v3_multirow_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,     # [S, N]
    d_in_fp32: UInt64,      # [S, K]
    d_w_nvfp4: UInt64,
    global_scale: Float32,
    K: Int,
    N: Int,
    S: Int,
) raises:
    """Launch the S-row GEMV. Same row-offset chunking as v1/v3 so the vocab-sized launch
    cannot hit the 65536-block sm_120a fault. Caller must ensure 1 <= S <= NVFP4_GEMV_SMAX."""
    var nb = (K * N) // FP4_GROUP
    var kern = ctx.compile_function[nvfp4_gemv_v3_multirow_kernel[NVFP4_GEMV_SMAX]]()
    var rows_per_launch = 32768
    var row_base = 0
    while row_base < N:
        var rows = N - row_base
        if rows > rows_per_launch:
            rows = rows_per_launch
        var rows_per_blk = NVFP4_WARPS_PER_BLOCK * NVFP4_GEMV_ROWS
        var blocks = (rows + rows_per_blk - 1) // rows_per_blk
        ctx.enqueue_function(
            kern,
            _as_u8_ptr(d_w_nvfp4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
            global_scale, Int32(K), Int32(N), Int32(nb), Int32(S), Int32(row_base),
            grid_dim=blocks, block_dim=NVFP4_WARPS_PER_BLOCK * WARP_SIZE,
        )
        row_base += rows


def gpu_matmul_nvfp4_fused_v3_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_nvfp4: UInt64,
    global_scale: Float32,
    K: Int,
    N: Int,
) raises:
    """v3 fused GEMV dispatch: vectorized arithmetic E2M1 decode. Bit-identical to v1
    (same per-lane block set + same sequential t=0..15 accumulation); only the branchy
    per-element decode is replaced by branchless SIMD. This is the raw-kernel lever."""
    var nb = (K * N) // FP4_GROUP
    var kern = ctx.compile_function[nvfp4_gemv_v3_kernel]()
    # Same row-offset chunking as v1 (gpu_matmul_nvfp4_fused_dev): keep grid.x in the proven
    # <=8192 range so the vocab-sized launch cannot hit the 65536-block sm_120a fault. For
    # ordinary projection sizes N <= 32768 this is a single launch — behaviour unchanged.
    var rows_per_launch = 32768
    var row_base = 0
    while row_base < N:
        var rows = N - row_base
        if rows > rows_per_launch:
            rows = rows_per_launch
        var blocks = (rows + NVFP4_WARPS_PER_BLOCK - 1) // NVFP4_WARPS_PER_BLOCK
        ctx.enqueue_function(
            kern,
            _as_u8_ptr(d_w_nvfp4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
            global_scale, Int32(K), Int32(N), Int32(nb), Int32(row_base),
            grid_dim=blocks, block_dim=NVFP4_WARPS_PER_BLOCK * WARP_SIZE,
        )
        row_base += rows


# ═════════════════════════════════════════════════════════════════════════════
# v3-tree: multi-warp-per-row tree reduction of the K-block partials.
# Same vectorized E2M1 decode as v3, BUT WPB warps collaborate on each output row:
# each (warp,lane) owns a DISJOINT set of 16-weight blocks (b = warp*32+lane, +WPB*32),
# so the per-lane sequential accumulation chain shrinks from nbk/32 to nbk/(WPB*32).
# Then warp-shuffle within each warp + a FIXED-ORDER shared-mem tree across the WPB
# warps. DETERMINISTIC (fixed warp order) but NOT bit-identical to v1/v3 (different
# accumulation order -> within FP4 tolerance of an fp32 reference; near-tie argmax
# flips are semantically free). Accumulator stays fp32; data stays fp4. This is the
# lever for HIGH-K shapes (FF_D, nbk=1344 -> chain 42 -> 42/WPB).
# ═════════════════════════════════════════════════════════════════════════════
def nvfp4_gemv_v3_tree_kernel[
    WPB: Int,    # warps per block collaborating on ONE output row (1,2,4,8)
](
    blob: UnsafePointer[UInt8, MutAnyOrigin],
    inp: UnsafePointer[Float32, MutAnyOrigin],
    outp: UnsafePointer[Float32, MutAnyOrigin],
    global_scale: Float32,
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var n = Int(block_idx.x)                 # one block per output row
    if n >= N:
        return
    var row_block0 = (n * K) // FP4_GROUP
    var packed_base = nb + (n * K) // 2
    var nbk = K // FP4_GROUP

    var local = Scalar[DType.float32](0.0)
    # Disjoint block ownership: stride = WPB*32 so the WPB warps tile nbk without overlap.
    var b = warp_id * WARP_SIZE + lane
    var stride = WPB * WARP_SIZE
    while b < nbk:
        var eff = global_scale * e4m3_decode(Int(blob[row_block0 + b]))
        var pbase = packed_base + b * (FP4_GROUP // 2)
        var ibase = b * FP4_GROUP
        var pk = (blob + pbase).bitcast[UInt64]()[0]
        var wvec = _e2m1_nibbles_to_fp32(pk)
        var act = (inp + ibase).load[width=FP4_GROUP]()
        var contrib = (wvec * eff) * act
        @parameter
        for t in range(FP4_GROUP):
            local += contrib[t]
        b += stride

    # Warp-shuffle butterfly reduce WITHIN this warp (fixed order -> deterministic).
    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2

    # Cross-warp tree reduce via shared memory (FIXED ORDER: warp 0,1,...,WPB-1).
    # Single thread (warp 0 lane 0) sums the WPB warp-partials sequentially -> deterministic.
    var partials = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[8]())   # up to WPB=8
    if lane == 0:
        partials[warp_id] = Float32(local)
    barrier()
    if warp_id == 0 and lane == 0:
        var s = Float32(0.0)
        @parameter
        for w in range(WPB):
            s += partials[w]
        outp[n] = s


def gpu_matmul_nvfp4_fused_v3_tree_dev[
    WPB: Int,
](
    ctx: DeviceContext,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_nvfp4: UInt64,
    global_scale: Float32,
    K: Int,
    N: Int,
) raises:
    """v3-tree dispatch: WPB warps/row tree-reduce the K-block partials. Deterministic
    (fixed warp order) but not bit-identical to v1/v3 (different accumulation order).
    Choose WPB by shape: HIGH-K (FF_D, nbk large) wants larger WPB (shorter per-lane
    chain); N-saturated shapes (D_VOC) keep WPB small (more rows in flight)."""
    var nb = (K * N) // FP4_GROUP
    var kern = ctx.compile_function[nvfp4_gemv_v3_tree_kernel[WPB]]()
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_nvfp4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        global_scale, Int32(K), Int32(N), Int32(nb),
        grid_dim=N, block_dim=WPB * WARP_SIZE,
    )


def _nvfp4_use_v3() -> Bool:
    """NOMOS_NVFP4_V3 toggle — default ON. v3 = arithmetic-E2M1 branchless fused GEMV
    (~1.8x vs v2, parity-clean, FP4-tolerance gated). Set =0 to fall back to the prior
    gpu_matmul_nvfp4_fused_dev for the v2-vs-v3 A/B in one binary (GB10 validation)."""
    # Cached in the C shim (read once/process, not per-GEMV). Default ON; only "0" disables.
    return _env_flag_cached(ENV_NVFP4_V3) != 0


def _nvfp4_use_v3_tree() -> Bool:
    """NOMOS_NVFP4_V3_TREE toggle — default ON. Per-shape v3-tree: WPB warps collaborate
    per row over disjoint K-blocks + a fixed-order shared-mem tree reduce (det, no atomics;
    fp32 accumulator, ~1 ULP, within-FP4-tol of fp32 — validated). Shortens the per-lane
    K-chain -> lifts high-K per-layer shapes to ~57-59% roofline (FF_D laggard 50->58%).
    Set =0 to force plain v3 everywhere (the v3-vs-v3-tree A/B in one binary)."""
    # Cached in the C shim (read once/process, not per-GEMV). Default ON; only "0" disables.
    return _env_flag_cached(ENV_NVFP4_V3_TREE) != 0
