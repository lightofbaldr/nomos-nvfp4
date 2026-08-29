"""Per-token activation NVFP4 quantizer (FP4 Step 3, W4A4).

Quantizes a [M,K] fp32 activation to NVFP4 on the fly so it can feed lib/fp4_gemm's
native FP4 tensor-core MMA (both operands FP4). ONE WARP per token: a warp-shuffle amax
reduction over K gives the per-token fp32 global, then per-16-element e4m3 block scales
(stored as raw uint8 bytes, same as the weight path) + E2M1-packed values. Same two-level
scheme as the weight encoder (tools/quantize_nvfp4.py) so query-space == storage-space.
The per-token global is applied AFTER the MMA (× weight_global). e4m3 is encoded manually
(b3 has no f32→f8e4m3fn cast); decode reuses lib/fp4_weights.e4m3_decode. Mojo 1.0.0b3 +
--target-accelerator sm_121a. Layout matches lib/fp4_gemm: packed low nibble=elem 2j, high=2j+1.
"""
from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim, grid_dim
from max.gpu import barrier
from max.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer, stack_allocation
from std.ffi import external_call
from lib.fp4_weights import e4m3_decode
from lib.fp4_gemm import gpu_fp4_gemm, gpu_fp4_gemm_ps, gpu_fp4_gemm_ps_grouped4, qa_fuse_route
from lib.fp4_gemm import gpu_fp4_gemm_ps_m, qa_fuse_route_m
from lib.fp4_gemm import _qa_fuse_env_off, _fp4_old_kernel_forced
# NVFP4 encode helpers live in lib/fp4_gemm since R2 Phase B (the in-GEMM A-tile
# encode needs them there; fp4_act imports fp4_gemm so they moved to avoid a cycle).
from lib.fp4_gemm import E2M1_MAX, E4M3_MAX, FP4_SMEM_WARPS, FP4_TK_STAGE
from lib.fp4_gemm import _f32_to_e4m3_byte, _e2m1_nibble
from lib.fp4_gemm_sm100 import gpu_fp4_gemm_sm100, gpu_nvfp4_sf_scatter


def _w4a4_debug_sync() -> Bool:
    """NOMOS_W4A4_DEBUG_SYNC=1 synchronizes each W4A4 sub-launch."""
    var name = String("NOMOS_W4A4_DEBUG_SYNC")
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()):
        nb.append(name.as_bytes()[i])
    nb.append(0)
    var raw = external_call[
        "getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]
    ](nb.unsafe_ptr())
    if raw == 0:
        return False
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    return p[0] == UInt8(49)  # '1'


def _w4a4_sync_checkpoint(
    ctx: DeviceContext, stage: String, M: Int, N: Int, K: Int
) raises:
    if not _w4a4_debug_sync():
        return
    # Print before synchronizing: if the queued launch faulted, this is the last
    # marker visible before ctx.synchronize raises CUDA_ERROR_ILLEGAL_ADDRESS.
    print("[W4A4 sync] checking", stage, "M=", M, "N=", N, "K=", K)
    ctx.synchronize()


def quant_act_nvfp4_kernel(
    act: UnsafePointer[Float32, MutAnyOrigin],      # [M, K] fp32
    out_packed: UnsafePointer[UInt8, MutAnyOrigin], # [M, K/2] E2M1
    out_bs: UnsafePointer[UInt8, MutAnyOrigin],     # [M, K/16] ue4m3 bytes
    out_global: UnsafePointer[Float32, MutAnyOrigin],  # [M]
    M_arg: Int32, Mreal_arg: Int32, K_arg: Int32, calibrated_global: Float32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var M = Int(M_arg)
    var Mreal = Int(Mreal_arg)
    var K = Int(K_arg)
    var m = Int(block_idx.x)            # one warp per token (M = padded rows)
    if m >= M:
        return
    var lane = Int(thread_idx.x)        # 0..31
    var KB = K // 2
    var NB = K // 16
    var base = m * K

    # Pad rows (m >= Mreal): emit exact zeros (global=1, packed/bs all 0) so the
    # MMA contributes 0 for them and their (discarded) output rows are clean.
    if m >= Mreal:
        if lane == 0:
            out_global[m] = Float32(1.0)
        var zp = lane
        while zp < KB:
            out_packed[m * KB + zp] = UInt8(0)
            zp += WARP_SIZE
        var zs = lane
        while zs < NB:
            out_bs[m * NB + zs] = UInt8(0)
            zs += WARP_SIZE
        return

    # ── Phase 1: per-token amax over K (warp-shuffle max reduce) ──
    var amax = Float32(0.0)
    var i = lane
    while i < K:
        var v = abs(act[base + i])
        if v > amax:
            amax = v
        i += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        var other = shuffle_xor(amax, UInt32(off))
        if other > amax:
            amax = other
        off //= 2
    var gscale = calibrated_global
    if gscale <= Float32(0.0):
        gscale = amax / (E2M1_MAX * E4M3_MAX)
    if gscale == Float32(0.0):
        gscale = Float32(1.0)
    if lane == 0:
        out_global[m] = gscale

    # ── Phase 2: per-16-block e4m3 scale + E2M1 pack (blocks strided across the warp) ──
    var b = lane
    while b < NB:
        var boff = base + b * 16
        var bamax = Float32(0.0)
        for e in range(16):
            var v = abs(act[boff + e])
            if v > bamax:
                bamax = v
        var bs_byte = _f32_to_e4m3_byte(bamax / (E2M1_MAX * gscale))
        out_bs[m * NB + b] = UInt8(bs_byte)
        var denom = gscale * e4m3_decode(bs_byte)
        if denom == Float32(0.0):
            denom = Float32(1.0)
        for j in range(8):
            var n0 = _e2m1_nibble(act[boff + 2 * j] / denom)
            var n1 = _e2m1_nibble(act[boff + 2 * j + 1] / denom)
            out_packed[m * KB + b * 8 + j] = UInt8((n1 << 4) | n0)
        b += WARP_SIZE


def gpu_quant_act_nvfp4(
    ctx: DeviceContext, act_ptr: UInt64, packed_ptr: UInt64, bs_ptr: UInt64,
    global_ptr: UInt64, M: Int, Mreal: Int, K: Int,
    calibrated_global: Float32 = 0.0,
) raises:
    """Quantize a [Mreal,K] fp32 activation -> NVFP4 over M (>=Mreal) MMA rows (one warp
    per row; rows >=Mreal emit zeros). Caller-owned buffers: packed [M,K/2] u8,
    bs [M,K/16] ue4m3 u8, global [M] fp32. K must be %16==0."""
    var kern = ctx.compile_function[quant_act_nvfp4_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_ptr)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(packed_ptr)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(bs_ptr)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(global_ptr)),
        Int32(M), Int32(Mreal), Int32(K), calibrated_global,
        grid_dim=M, block_dim=WARP_SIZE,
    )


# ─────────────────────────────────────────────────────────────────────────
# W4A4 GEMM (FP4 Step 3b) — the SPEED path: quantize activation -> native FP4
# tensor-core MMA (lib/fp4_gemm) -> post-scale. No bf16 dequant. out[M,N] =
# act[M,K] @ weight[N,K]^T, both operands FP4.
# ─────────────────────────────────────────────────────────────────────────

def postscale_w4a4_kernel(
    out_: UnsafePointer[Float32, MutAnyOrigin],        # [M, N] result
    c_pad: UnsafePointer[Float32, MutAnyOrigin],       # [Mpad, N] raw MMA output
    act_global: UnsafePointer[Float32, MutAnyOrigin],  # [Mpad] per-token global
    weight_global: Float32,                            # per-tensor weight global
    M_arg: Int32, N_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var M = Int(M_arg)
    var N = Int(N_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < M * N:
        var m = i // N
        out_[i] = c_pad[i] * act_global[m] * weight_global


def act_amax_gscale_kernel(
    act: UnsafePointer[Float32, MutAnyOrigin],          # [K] fp32 (decode row)
    out_global: UnsafePointer[Float32, MutAnyOrigin],   # [>=16] per-row globals
    K_arg: Int32, calibrated_global: Float32,
):
    """R2 Phase B: per-token gscale WITHOUT the full quant — 1 block x 256 threads,
    grid-stride abs-max + shuffle/smem reduce. fp32 max is order-independent, so
    out_global[0] is BIT-IDENTICAL to quant_act_nvfp4_kernel's gscale (same abs,
    same compare, same divide). Pads [1..15] = 1.0 (quant-kernel pad semantics)."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var tid = Int(thread_idx.x)          # single block
    if calibrated_global > Float32(0.0):
        if tid == 0:
            out_global[0] = calibrated_global
        elif tid < 16:
            out_global[tid] = Float32(1.0)
        return
    var amax = Float32(0.0)
    var i = tid
    while i < K:
        var v = abs(act[i])
        if v > amax:
            amax = v
        i += 256
    var off = WARP_SIZE // 2             # warp-level max reduce
    while off > 0:
        var other = shuffle_xor(amax, UInt32(off))
        if other > amax:
            amax = other
        off //= 2
    var sm = stack_allocation[
        8, Float32, address_space = AddressSpace.SHARED
    ]()
    if (tid & 31) == 0:
        sm[tid >> 5] = amax
    barrier()
    if tid == 0:
        var m = sm[0]
        for wi in range(1, 8):
            if sm[wi] > m:
                m = sm[wi]
        var gscale = m / (E2M1_MAX * E4M3_MAX)
        if gscale == Float32(0.0):
            gscale = Float32(1.0)
        out_global[0] = gscale
    elif tid < 16:
        out_global[tid] = Float32(1.0)


def gpu_act_amax_gscale(
    ctx: DeviceContext, act_ptr: UInt64, global_ptr: UInt64, K: Int,
    calibrated_global: Float32 = 0.0,
) raises:
    """Write the per-token NVFP4 global scale of a [1,K] fp32 activation into
    global_ptr[0] (pads [1..15]=1.0) — the only A-side precompute the QA-fused
    GEMM (gpu_fp4_gemm_qa) needs. One launch per DISTINCT activation."""
    var kern = ctx.compile_function[act_amax_gscale_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_ptr)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(global_ptr)),
        Int32(K), calibrated_global,
        grid_dim=1, block_dim=256,
    )


def quant_act_encode_kernel(
    act: UnsafePointer[Float32, MutAnyOrigin],      # [K] fp32 (decode row 0)
    out_packed: UnsafePointer[UInt8, MutAnyOrigin], # [16, K/2] E2M1
    out_bs: UnsafePointer[UInt8, MutAnyOrigin],     # [16, K/16] ue4m3 bytes
    out_global: UnsafePointer[Float32, MutAnyOrigin],  # [0] = gscale (amax kernel)
    K_arg: Int32,
):
    """R2 Phase B: grid-PARALLEL NVFP4 encode of a decode activation — one thread per
    16-el block instead of one warp for the whole row (the one-warp
    quant_act_nvfp4_kernel serializes ~NB/32 heavy encodes per lane -> ~63us at
    gemma-4 K; this spreads them over NB threads -> ~2-4us). Per-block math, loop
    order and denominators are VERBATIM quant_act_nvfp4_kernel -> bytes identical.
    Pad rows 1..15 zero-filled word-wise (requires K%64==0). gscale is read from
    out_global[0] (gpu_act_amax_gscale — bit-identical to the one-warp reduce)."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var KB = K // 2
    var NB = K // 16
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nthreads = Int(grid_dim.x) * Int(block_dim.x)
    var gscale = out_global[0]
    var b = t
    while b < NB:
        var boff = b * 16
        var bamax = Float32(0.0)
        for e in range(16):
            var v = abs(act[boff + e])
            if v > bamax:
                bamax = v
        var bs_byte = _f32_to_e4m3_byte(bamax / (E2M1_MAX * gscale))
        out_bs[b] = UInt8(bs_byte)
        var denom = gscale * e4m3_decode(bs_byte)
        if denom == Float32(0.0):
            denom = Float32(1.0)
        for j in range(8):
            var n0 = _e2m1_nibble(act[boff + 2 * j] / denom)
            var n1 = _e2m1_nibble(act[boff + 2 * j + 1] / denom)
            out_packed[b * 8 + j] = UInt8((n1 << 4) | n0)
        b += nthreads
    # Pad rows 1..15: exact zeros (same bytes the one-warp kernel emits), word-wise.
    var zw = t
    var pw = (15 * KB) // 4
    while zw < pw:
        (out_packed + KB + zw * 4).bitcast[UInt32]()[0] = UInt32(0)
        zw += nthreads
    var zs = t
    var sw = (15 * NB) // 4
    while zs < sw:
        (out_bs + NB + zs * 4).bitcast[UInt32]()[0] = UInt32(0)
        zs += nthreads


def gpu_quant_act_nvfp4_fast(
    ctx: DeviceContext, act_ptr: UInt64, packed_ptr: UInt64, bs_ptr: UInt64,
    global_ptr: UInt64, K: Int, calibrated_global: Float32 = 0.0,
) raises:
    """R2 Phase B: decode (M=1/Mpad=16) act quant, byte-identical output to
    gpu_quant_act_nvfp4(..., 16, 1, K) but split into the amax micro-kernel (gscale;
    fp32 max is order-independent -> bit-identical) + the grid-parallel encode.
    ~63us -> ~5-7us per distinct activation at gemma-4 K. Requires K%64==0."""
    gpu_act_amax_gscale(ctx, act_ptr, global_ptr, K, calibrated_global)
    var kern = ctx.compile_function[quant_act_encode_kernel]()
    var blocks = ((K // 16) + 255) // 256
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_ptr)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(packed_ptr)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(bs_ptr)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(global_ptr)),
        Int32(K),
        grid_dim=blocks, block_dim=256,
    )


def gpu_quant_act_nvfp4_dec(
    ctx: DeviceContext, act_ptr: UInt64, packed_ptr: UInt64, bs_ptr: UInt64,
    global_ptr: UInt64, K: Int, calibrated_global: Float32 = 0.0,
) raises:
    """Decode (M=1/Mpad=16) act-quant prep for the W4A4 dedup call sites: routes to
    the parallel fast quant (R2 Phase B) unless NOMOS_FP4_QAFUSE=0 or K%64!=0, else
    the original one-warp quant kernel. Output bytes are IDENTICAL either way, so
    any downstream GEMM (plain or fused-postscale) composes with either prep."""
    if K % 64 == 0 and not _qa_fuse_env_off():
        gpu_quant_act_nvfp4_fast(
            ctx, act_ptr, packed_ptr, bs_ptr, global_ptr, K, calibrated_global
        )
    else:
        gpu_quant_act_nvfp4(
            ctx, act_ptr, packed_ptr, bs_ptr, global_ptr, 16, 1, K,
            calibrated_global,
        )


# ─────────────────────────────────────────────────────────────────────────
# Spec-verify M-row parallel act quant (byte-identical to the one-warp
# quant_act_nvfp4_kernel over Mpad rows, but parallel). The one-warp kernel
# serializes each row's ~NB/32 heavy encodes on a single warp (~63us/row = the
# 30ms/verify-forward bottleneck at 410 launches). This splits it exactly like
# the M=1 fast path: a per-row amax micro-kernel (fp32 max is order-independent
# -> bit-identical gscale) + a grid-parallel encode over ALL Mreal*NB blocks.
# Per-block math/loop-order/denominators are VERBATIM quant_act_nvfp4_kernel, so
# packed + scales + globals bytes match the one-warp output EXACTLY (incl the
# Mreal..Mpad-1 pad rows: global=1.0, packed=0, bs=0).
# ─────────────────────────────────────────────────────────────────────────

def act_amax_gscale_m_kernel(
    act: UnsafePointer[Float32, MutAnyOrigin],          # [Mreal, K] fp32
    out_global: UnsafePointer[Float32, MutAnyOrigin],   # [Mpad] per-row globals
    Mpad_arg: Int32, Mreal_arg: Int32, K_arg: Int32, calibrated_global: Float32,
):
    """One block per MMA row (grid=Mpad, block=256): block m grid-strides row m's K,
    warp+smem max-reduce -> gscale = amax/(E2M1_MAX*E4M3_MAX) (bit-identical to
    quant_act_nvfp4_kernel's per-row reduce). Pad rows (m>=Mreal) write global=1.0."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var Mpad = Int(Mpad_arg)
    var Mreal = Int(Mreal_arg)
    var K = Int(K_arg)
    var m = Int(block_idx.x)
    if m >= Mpad:
        return
    var tid = Int(thread_idx.x)          # single block, 256 threads
    if m >= Mreal:
        if tid == 0:
            out_global[m] = Float32(1.0)     # pad-row global (quant-kernel semantics)
        return
    if calibrated_global > Float32(0.0):
        if tid == 0:
            out_global[m] = calibrated_global
        return
    var base = m * K
    var amax = Float32(0.0)
    var i = tid
    while i < K:
        var v = abs(act[base + i])
        if v > amax:
            amax = v
        i += 256
    var off = WARP_SIZE // 2             # warp-level max reduce
    while off > 0:
        var other = shuffle_xor(amax, UInt32(off))
        if other > amax:
            amax = other
        off //= 2
    var sm = stack_allocation[
        8, Float32, address_space = AddressSpace.SHARED
    ]()
    if (tid & 31) == 0:
        sm[tid >> 5] = amax
    barrier()
    if tid == 0:
        var mm = sm[0]
        for wi in range(1, 8):
            if sm[wi] > mm:
                mm = sm[wi]
        var gscale = mm / (E2M1_MAX * E4M3_MAX)
        if gscale == Float32(0.0):
            gscale = Float32(1.0)
        out_global[m] = gscale


def quant_act_encode_m_kernel(
    act: UnsafePointer[Float32, MutAnyOrigin],      # [Mreal, K] fp32
    out_packed: UnsafePointer[UInt8, MutAnyOrigin], # [Mpad, K/2] E2M1
    out_bs: UnsafePointer[UInt8, MutAnyOrigin],     # [Mpad, K/16] ue4m3 bytes
    out_global: UnsafePointer[Float32, MutAnyOrigin],  # [Mpad] gscale per row (amax kern)
    Mpad_arg: Int32, Mreal_arg: Int32, K_arg: Int32,
):
    """Grid-parallel encode over ALL Mreal*NB 16-el blocks (thread -> flattened
    (row,block) index), reading its row's gscale from out_global. Per-block math =
    quant_act_nvfp4_kernel VERBATIM. Pad rows Mreal..Mpad-1 zero-filled word-wise
    (packed=0, bs=0). Requires K%64==0 (KB,NB both %4)."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var Mpad = Int(Mpad_arg)
    var Mreal = Int(Mreal_arg)
    var K = Int(K_arg)
    var KB = K // 2
    var NB = K // 16
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nthreads = Int(grid_dim.x) * Int(block_dim.x)
    var total = Mreal * NB
    var fb = t
    while fb < total:
        var row = fb // NB
        var b = fb % NB
        var gscale = out_global[row]
        var boff = row * K + b * 16
        var bamax = Float32(0.0)
        for e in range(16):
            var v = abs(act[boff + e])
            if v > bamax:
                bamax = v
        var bs_byte = _f32_to_e4m3_byte(bamax / (E2M1_MAX * gscale))
        out_bs[row * NB + b] = UInt8(bs_byte)
        var denom = gscale * e4m3_decode(bs_byte)
        if denom == Float32(0.0):
            denom = Float32(1.0)
        for j in range(8):
            var n0 = _e2m1_nibble(act[boff + 2 * j] / denom)
            var n1 = _e2m1_nibble(act[boff + 2 * j + 1] / denom)
            out_packed[row * KB + b * 8 + j] = UInt8((n1 << 4) | n0)
        fb += nthreads
    # Pad rows Mreal..Mpad-1: exact zeros (same bytes the one-warp kernel emits).
    var npad = Mpad - Mreal
    var zw = t
    var pw = (npad * KB) // 4
    while zw < pw:
        (out_packed + Mreal * KB + zw * 4).bitcast[UInt32]()[0] = UInt32(0)
        zw += nthreads
    var zs = t
    var sw = (npad * NB) // 4
    while zs < sw:
        (out_bs + Mreal * NB + zs * 4).bitcast[UInt32]()[0] = UInt32(0)
        zs += nthreads


def gpu_quant_act_nvfp4_fast_m(
    ctx: DeviceContext, act_ptr: UInt64, packed_ptr: UInt64, bs_ptr: UInt64,
    global_ptr: UInt64, Mpad: Int, Mreal: Int, K: Int,
    calibrated_global: Float32 = 0.0,
) raises:
    """Spec-verify (1<Mreal<=Mpad, Mpad mult of 16) act quant: byte-identical output to
    gpu_quant_act_nvfp4(..., Mpad, Mreal, K) but parallel (per-row amax micro-kernel +
    grid-parallel encode) instead of one-warp-per-row serial. ~63us/row -> ~2-4us total.
    Requires K%64==0."""
    var amk = ctx.compile_function[act_amax_gscale_m_kernel]()
    ctx.enqueue_function(
        amk,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_ptr)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(global_ptr)),
        Int32(Mpad), Int32(Mreal), Int32(K), calibrated_global,
        grid_dim=Mpad, block_dim=256,
    )
    var enk = ctx.compile_function[quant_act_encode_m_kernel]()
    var total = Mreal * (K // 16)
    var blocks = (total + 255) // 256
    if blocks < 1:
        blocks = 1
    ctx.enqueue_function(
        enk,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_ptr)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(packed_ptr)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(bs_ptr)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(global_ptr)),
        Int32(Mpad), Int32(Mreal), Int32(K),
        grid_dim=blocks, block_dim=256,
    )


def _w4a4_prequant_off() -> Bool:
    """NOMOS_FP4_PREQUANT=0 -> disable the R2 Phase-A call-site quant dedup (A/B switch);
    default ON. Same getenv pattern as lib/fp4_gemm._fp4_old_kernel_forced."""
    var name = String("NOMOS_FP4_PREQUANT")
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()):
        nb.append(name.as_bytes()[i])
    nb.append(0)
    var raw = external_call[
        "getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]
    ](nb.unsafe_ptr())
    if raw == 0:
        return False
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    return p[0] == UInt8(48)     # '0'


def gpu_matmul_nvfp4_w4a4_prequant_dev(
    ctx: DeviceContext,
    d_out: UInt64,            # [M, N] fp32 output
    d_w_nvfp4: UInt64,        # weight blob [e4m3 scales : nb_w][E2M1 packed]
    weight_global: Float32,
    d_act_packed: UInt64, d_act_bs: UInt64, d_act_global: UInt64,  # PRE-quantized act NVFP4
    d_c_pad: UInt64,          # [cap, N] fp32 raw-MMA scratch
    M: Int, K: Int, N: Int,
) raises:
    """W4A4 GEMM over a PRE-quantized activation: FP4 MMA + post-scale, NO act-quant
    launch. R2 Phase-A call-site dedup — q/k/v share d_normed and gate/up share d_pn,
    so the caller quantizes ONCE into the shared w4a4 scratch and the 2nd/3rd GEMMs
    skip the byte-identical re-quantization (mirrors the q8/dp4a prequant pattern in
    gemma4_layer prepare_qkv_dev / apply_output_and_mlp_dev).

    R2 Phase B: when qa_fuse_route allows (decode M==1, smem-tileable shape,
    NOMOS_FP4_QAFUSE!=0), the GEMM runs with the fused-postscale epilogue
    (gpu_fp4_gemm_ps: identical staging+MMA, row-0 scaled write in the epilogue —
    bit-exact, no c_pad round-trip, no postscale launch). Otherwise kernel calls +
    argument bytes are IDENTICAL to gpu_matmul_nvfp4_w4a4_dev's single-chunk path ->
    bit-exact by construction. Single chunk only (M <= the rows already quantized
    into the scratch); sm_120/121 only — callers guard exactly like _mm_dev's W4A4
    branch (the GEMM entries raise on sm_100)."""
    var nb_w = (N * K) // 16
    if qa_fuse_route(M, K, N):
        gpu_fp4_gemm_ps(ctx, d_out, d_act_packed, d_w_nvfp4 + UInt64(nb_w), d_act_bs,
                        d_w_nvfp4, d_act_global, weight_global, N, K)
        return
    var cmpad = ((M + 15) // 16) * 16       # pad rows to the warp-MMA m-dim (16)
    gpu_fp4_gemm(ctx, d_c_pad, d_act_packed, d_w_nvfp4 + UInt64(nb_w), d_act_bs,
                 d_w_nvfp4, cmpad, N, K)
    var kern = ctx.compile_function[postscale_w4a4_kernel]()
    var threads = 256
    var blocks = (M * N + threads - 1) // threads
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_c_pad)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_act_global)),
        weight_global, Int32(M), Int32(N),
        grid_dim=blocks, block_dim=threads,
    )


def w4a4_prequant_grouped4_route(
    K: Int, N0: Int, N1: Int, N2: Int, N3: Int,
) -> Bool:
    """Whether all four segments fit the exact NS=3/TK=4 fused-postscale base tile.

    Environment kill switches are resolved once by engine_decode before the layer loop;
    this per-layer predicate stays pure geometry so the experiment does not add repeated
    getenv/list-allocation overhead.
    """
    return (
        K % (FP4_TK_STAGE * 64) == 0
        and N0 % (8 * FP4_SMEM_WARPS) == 0
        and N1 % (8 * FP4_SMEM_WARPS) == 0
        and N2 % (8 * FP4_SMEM_WARPS) == 0
        and N3 % (8 * FP4_SMEM_WARPS) == 0
    )


def w4a4_prequant_grouped4_env_enabled() -> Bool:
    """Honor the existing fused-postscale and old-kernel kill switches exactly once."""
    return not _qa_fuse_env_off() and not _fp4_old_kernel_forced()


def gpu_matmul_nvfp4_w4a4_prequant_grouped4_dev(
    ctx: DeviceContext,
    d_out0: UInt64, d_w0: UInt64, weight_global0: Float32, N0: Int,
    d_out1: UInt64, d_w1: UInt64, weight_global1: Float32, N1: Int,
    d_out2: UInt64, d_w2: UInt64, weight_global2: Float32, N2: Int,
    d_out3: UInt64, d_w3: UInt64, weight_global3: Float32, N3: Int,
    d_act_packed: UInt64, d_act_bs: UInt64, d_act_global: UInt64,
    K: Int,
) raises:
    """Grouped decode projection over four separately encoded NVFP4 weight blobs.

    Each blob keeps its own scale-byte base, packed-byte offset, global multiplier,
    and output pointer. The shared activation was already quantized once by the caller.
    """
    var nb0 = (N0 * K) // 16
    var nb1 = (N1 * K) // 16
    var nb2 = (N2 * K) // 16
    var nb3 = (N3 * K) // 16
    gpu_fp4_gemm_ps_grouped4(
        ctx,
        d_out0, d_out1, d_out2, d_out3,
        d_act_packed, d_act_bs, d_act_global,
        d_w0 + UInt64(nb0), d_w1 + UInt64(nb1),
        d_w2 + UInt64(nb2), d_w3 + UInt64(nb3),
        d_w0, d_w1, d_w2, d_w3,
        weight_global0, weight_global1, weight_global2, weight_global3,
        N0, N1, N2, N3, K,
    )


def gpu_matmul_nvfp4_w4a4_dev(
    ctx: DeviceContext, handle: UInt64,
    d_out: UInt64,            # [M, N] fp32 output
    d_in: UInt64,             # [M, K] fp32 activation (M real rows; on-GPU, no padding)
    d_w_nvfp4: UInt64,        # weight blob [e4m3 scales : nb_w][E2M1 packed]
    weight_global: Float32,
    input_global: Float32,
    d_act_packed: UInt64, d_act_bs: UInt64, d_act_global: UInt64,  # act NVFP4 scratch [cap,*]
    d_act_bs_sf: UInt64,      # sm_100: SF-atom-scattered activation scales scratch (0 on sm_120)
    d_w_bs_sf: UInt64,        # sm_100: SF-atom-scattered weight scales scratch (0 on sm_120)
    d_c_pad: UInt64,          # [cap, N] fp32 raw-MMA scratch
    M: Int, cap: Int, K: Int, N: Int,
) raises:
    """W4A4 native-MMA GEMM: quantize the M-row activation -> NVFP4, run the FP4 tensor-core
    MMA vs the NVFP4 weight, post-scale by act_global[m] x weight_global.

    #431: the M rows are processed in ROW-CHUNKS of `cap` (the scratch row capacity, a
    multiple of 16), so the act/MMA scratch (packed/bs/global/c_pad) is sized to ONE chunk
    instead of max_seq — the cpad alone is N*max_seq*4 (≈22GB @256k) when unchunked. The
    GEMM is row-independent (per-row act-quant -> MMA vs the shared weight -> per-row
    post-scale), so chunking is BIT-IDENTICAL to one big call. Decode (M=1, cap>=16) and any
    prompt <= cap run as a single chunk (the original path). Weight is MMA-shaped
    (Gemma N%8==0, K%64==0); each chunk pads its rows to a multiple of 16 (the MMA m-dim)."""
    var nb_w = (N * K) // 16
    # Datacenter Blackwell (sm_100/B200): the warp-level mxf4nvf4 MMA was dropped; route to the
    # tcgen05 GEMM (lib/fp4_gemm_sm100). It needs (a) M padded to the MMA m-dim 128 (not 16) and
    # (b) scales in the tcgen05 SF-atom layout (gpu_nvfp4_sf_scatter), so scatter act+weight
    # scales per chunk into the caller-provided SF-atom scratch. compute 10.x == sm_100.
    var is_sm100 = (Float64(ctx.default_device_info.compute) >= 10.0
                    and Float64(ctx.default_device_info.compute) < 12.0)
    # R2 Phase B (decode, M=1, sm_120/121): parallel fast quant (amax micro-kernel +
    # grid-parallel encode; bytes identical to the one-warp quant, ~63us -> ~5-7us)
    # + fused-postscale GEMM (identical staging/MMA, row-0 scaled write in the
    # epilogue — no c_pad round-trip, no postscale launch). Bit-exact end to end.
    # NOMOS_FP4_QAFUSE=0 (or NOMOS_FP4_OLD=1 / off-shape) falls back below.
    if not is_sm100 and qa_fuse_route(M, K, N):
        gpu_quant_act_nvfp4_dec(
            ctx, d_in, d_act_packed, d_act_bs, d_act_global, K, input_global
        )
        _w4a4_sync_checkpoint(ctx, "decode act-quant", M, N, K)
        gpu_fp4_gemm_ps(ctx, d_out, d_act_packed, d_w_nvfp4 + UInt64(nb_w), d_act_bs,
                        d_w_nvfp4, d_act_global, weight_global, N, K)
        _w4a4_sync_checkpoint(ctx, "decode fused-postscale GEMM", M, N, K)
        return
    # Spec-verify (1<M<=16, sm_120/121): the SAME fast fused speedup as decode, now
    # for the M=k+1 verify rows. Parallel M-row act quant (byte-identical to the
    # one-warp quant, ~63us/row -> ~2-4us total) + fused-postscale M-row GEMM
    # (identical staging/MMA to the bare GEMM, per-row scaled epilogue -> no c_pad
    # round-trip, no postscale launch). Single 16-row m-tile (Mpad=16). Bit-exact
    # to the OLD chain below. NOMOS_FP4_QAFUSE=0 / NOMOS_FP4_OLD=1 -> OLD chain.
    if not is_sm100 and qa_fuse_route_m(M, K, N):
        var mpad = ((M + 15) // 16) * 16       # == 16 for the verify M<=16
        gpu_quant_act_nvfp4_fast_m(ctx, d_in, d_act_packed, d_act_bs, d_act_global,
                                   mpad, M, K, input_global)
        _w4a4_sync_checkpoint(ctx, "verify act-quant", M, N, K)
        gpu_fp4_gemm_ps_m(ctx, d_out, d_act_packed, d_w_nvfp4 + UInt64(nb_w), d_act_bs,
                          d_w_nvfp4, d_act_global, weight_global, M, N, K)
        _w4a4_sync_checkpoint(ctx, "verify fused-postscale GEMM", M, N, K)
        return
    var kern = ctx.compile_function[postscale_w4a4_kernel]()
    var threads = 256
    var mdim = 128 if is_sm100 else 16
    if is_sm100:
        # weight scales are static per call -> scatter once per call (flat [N,K/16] -> SF-atom).
        gpu_nvfp4_sf_scatter(ctx, d_w_nvfp4, d_w_bs_sf, N, K // 16)
    var m0 = 0
    while m0 < M:
        var cm = M - m0
        if cm > cap:
            cm = cap
        var cmpad = ((cm + mdim - 1) // mdim) * mdim  # pad THIS chunk's rows to the MMA m-dim
        var in_off = UInt64(m0 * K * 4)             # fp32 row stride
        var out_off = UInt64(m0 * N * 4)
        gpu_quant_act_nvfp4(
            ctx, d_in + in_off, d_act_packed, d_act_bs, d_act_global,
            cmpad, cm, K, input_global,
        )
        _w4a4_sync_checkpoint(ctx, "chunk act-quant", cm, N, K)
        # fp4_gemm: A = activation (a_bytes=packed, a_sc=bs), B = weight (b_bytes=blob+nb_w, b_sc=blob)
        if is_sm100:
            gpu_nvfp4_sf_scatter(ctx, d_act_bs, d_act_bs_sf, cmpad, K // 16)
            gpu_fp4_gemm_sm100(ctx, d_c_pad, d_act_packed, d_w_nvfp4 + UInt64(nb_w),
                               d_act_bs_sf, d_w_bs_sf, cmpad, N, K)
        else:
            gpu_fp4_gemm(ctx, d_c_pad, d_act_packed, d_w_nvfp4 + UInt64(nb_w), d_act_bs, d_w_nvfp4, cmpad, N, K)
        _w4a4_sync_checkpoint(ctx, "chunk raw GEMM", cm, N, K)
        var blocks = (cm * N + threads - 1) // threads
        ctx.enqueue_function(
            kern,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out + out_off)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_c_pad)),
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_act_global)),
            weight_global, Int32(cm), Int32(N),
            grid_dim=blocks, block_dim=threads,
        )
        _w4a4_sync_checkpoint(ctx, "chunk postscale", cm, N, K)
        m0 += cap
