"""NVFP4 W4A4 GEMM (in-tree) — FP4 Step 1.

C[M,N] = A[M,K] @ B[N,K]ᵀ with per-16-element e4m3 block scales, on the Blackwell
NVFP4 tensor core (`mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale`). A is the
activation, B the weight; both packed E2M1 (FP4, 2 values/byte) plus one e4m3 scale per
16-element block. This is the *same* MMA validated bit-exact in
`tools/fp4_nvfp4_stock_test.mojo`, generalized from the 64×64×256 spike to real Gemma-4
projection shapes (D=5376, FF=21504).

Tiling: one warp (32 threads) computes one 16(M)×8(N) output tile, looping K in steps of
64 (the MMA's k-depth). Grid = (N//8, M//16).

Edges: K is always a multiple of 64 for Gemma-4 → NO K masking. M-edge (M%16) and N-edge
(N%8) are handled by the CALLER padding A→[ceil(M/16)*16, K], B→[ceil(N/8)*8, K],
C→[Mpad, Npad]; the kernel then runs fully in-bounds and the caller reads only the live
[M,N] sub-block. Each output C[m,n] = Σ_k A[m,k]·B[n,k] depends only on its own row/col,
so padded rows/cols never perturb live results.

Toolchain: requires Mojo ≥1.0.0b3 + `--target-accelerator sm_121a` (the b3 NVPTX backend
stamps PTX `.target sm_121a`; older nightlies stamp `sm_121` and the driver rejects the
FP4 block-scale features). See the design notes (2026-06-16).
"""
from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx
from max.gpu import barrier
from max.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.sys.info import _is_sm_120x_or_newer
from std.memory import UnsafePointer, stack_allocation
from std.ffi import external_call
from lib.fp4_weights import e4m3_decode


# ── NVFP4 encode helpers (moved from lib/fp4_act.mojo for R2 Phase B: fp4_act
# imports fp4_gemm, so the in-GEMM A-tile encode must live HERE to avoid a cycle;
# fp4_act re-imports them — math unchanged, byte-identical encodes). ──
comptime E2M1_MAX = Float32(6.0)
comptime E4M3_MAX = Float32(448.0)


def _pow2(e: Int) -> Float32:
    if e >= 0:
        return Float32(1 << e)
    return Float32(1.0) / Float32(1 << (-e))


def _f32_to_e4m3_byte(v: Float32) -> Int:
    """Round a positive fp32 to a ue4m3 byte (0..126), round-to-nearest. Inverse of
    e4m3_decode; matches the encoder's exact LUT to within rounding."""
    if v <= Float32(0.0):
        return 0
    var x = v
    if x > E4M3_MAX:
        x = E4M3_MAX
    if x < Float32(0.015625):                       # < 2^-6 → subnormal: mant * 2^-9
        var mant = Int(x * Float32(512.0) + Float32(0.5))
        if mant > 7:
            mant = 7
        return mant                                 # exp field = 0
    var e = -6                                       # find e: 2^e <= x < 2^(e+1)
    while e < 8 and x >= _pow2(e + 1):
        e += 1
    var mant = Int((x / _pow2(e) - Float32(1.0)) * Float32(8.0) + Float32(0.5))
    if mant == 8:                                    # mantissa carry
        e += 1
        mant = 0
    var exp_field = e + 7
    if exp_field > 15:
        return (15 << 3) | 6                         # clamp to 448
    if exp_field == 15 and mant > 6:
        mant = 6
    return (exp_field << 3) | mant


def _e2m1_nibble(x: Float32) -> Int:
    """Round a value (scaled to ~[-6,6]) to an E2M1 nibble: (sign<<3) | magnitude-index."""
    var s = 8 if x < Float32(0.0) else 0
    var a = abs(x)
    var idx = 0
    if a >= Float32(0.25): idx = 1
    if a >= Float32(0.75): idx = 2
    if a >= Float32(1.25): idx = 3
    if a >= Float32(1.75): idx = 4
    if a >= Float32(2.5): idx = 5
    if a >= Float32(3.5): idx = 6
    if a >= Float32(5.0): idx = 7
    return s | idx


def fp4_gemm_kernel(
    c_out: UnsafePointer[Float32, MutAnyOrigin],   # [Mpad, Npad] fp32, row-major
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Mpad, K/2] packed E2M1 (activation)
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Npad, K/2] packed E2M1 (weight)
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Mpad, K/16] e4m3 block scales (A)
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Npad, K/16] e4m3 block scales (B)
    M_arg: Int32, N_arg: Int32, K_arg: Int32,                        # PADDED dims: M%16==0, N%8==0, K%64==0
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)
    var KB = K // 2          # FP4 bytes per row
    var NB = K // 16         # e4m3 scale blocks per row
    var wm = Int(block_idx.y)
    var wn = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var gid = lane >> 2      # 0..7  → row within the 16-row tile (gid and gid+8)
    var tid = lane & 3       # 0..3  → k-quad selector / which 2 output columns
    var mrow = wm * 16
    var ncol = wn * 8

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)
    for k0 in range(0, K, 64):
        var ka = (k0 + tid * 8) >> 1          # byte offset of the first k-quad
        var kb = (k0 + tid * 8 + 32) >> 1     # byte offset of the second k-quad
        var a0 = (a_bytes + ((mrow + gid) * KB + ka)).bitcast[UInt32]()[0]
        var a1 = (a_bytes + ((mrow + gid + 8) * KB + ka)).bitcast[UInt32]()[0]
        var a2 = (a_bytes + ((mrow + gid) * KB + kb)).bitcast[UInt32]()[0]
        var a3 = (a_bytes + ((mrow + gid + 8) * KB + kb)).bitcast[UInt32]()[0]
        var b0 = (b_bytes + ((ncol + gid) * KB + ka)).bitcast[UInt32]()[0]
        var b1 = (b_bytes + ((ncol + gid) * KB + kb)).bitcast[UInt32]()[0]

        # block-scale operands: each contributing lane's b32 = that row/col's 4 e4m3
        # block-scales (one byte each) for this k64 chunk. SF_A from lanes tid∈{0,1}
        # (tid=0 → row gid, tid=1 → row gid+8); SF_B from lane tid==0 (col gid).
        var blk = k0 // 16
        var sca = UInt32(0)
        if tid == 0:
            sca = (a_sc + ((mrow + gid) * NB + blk)).bitcast[UInt32]()[0]
        elif tid == 1:
            sca = (a_sc + ((mrow + gid + 8) * NB + blk)).bitcast[UInt32]()[0]
        var scb = UInt32(0)
        if tid == 0:
            scb = (b_sc + ((ncol + gid) * NB + blk)).bitcast[UInt32]()[0]

        # The warp-level `mma.sync...kind::mxf4nvf4.block_scale` exists ONLY on
        # consumer/workstation Blackwell (sm_120a/sm_121a, compute>=12.0). Datacenter
        # Blackwell (sm_100a, B200/B300) dropped it — block-scaled NVFP4 is tcgen05-only
        # there, so ptxas rejects this instruction for sm_100a. Guard it at comptime so
        # the kernel still BUILDS on sm_100 (the asm is simply not emitted); on sm_100 the
        # W4A4 path is never dispatched (the engine runs W4A16 instead — see the host-side
        # guard in gpu_fp4_gemm), so the zeroed accumulators here are never consumed.
        comptime if _is_sm_120x_or_newer():
            var r = inlined_assembly[
                (
                    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                    ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                    " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                    " {$15}, {0, 0};"
                ),
                _RegisterPackType[Float32, Float32, Float32, Float32],
                constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
            ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
            c0 = r[0]
            c1 = r[1]
            c2 = r[2]
            c3 = r[3]

    c_out[(mrow + gid) * N + ncol + 2 * tid] = c0
    c_out[(mrow + gid) * N + ncol + 2 * tid + 1] = c1
    c_out[(mrow + gid + 8) * N + ncol + 2 * tid] = c2
    c_out[(mrow + gid + 8) * N + ncol + 2 * tid + 1] = c3


# ─────────────────────────────────────────────────────────────────────────────
# R1: cooperative shared-memory tile staging (cp.async double-buffered).
#
# WHY (ncu, 2026-07-06): the 1-warp/block kernel above is latency-bound — long
# scoreboard stalls 74-82% of issue slots, DRAM at ~16-35% of peak, tensor pipe 5%.
# Each warp issues 6 serial dependent 4B global loads per k64 then waits. The fix
# (identical stall signature fixed on the GB10 sprint) is to widen the block to 4
# warps that COOPERATIVELY stage the K-tile operands into shared memory with
# cp.async, double-buffered so tile kt+1 streams in while tile kt computes: global
# loads become wide (16B), batched (whole stage issued back-to-back) and
# overlapped, instead of narrow+serial per warp.
#
# Tiling: block = 128 threads = 4 warps; each warp keeps the proven 16(M)x8(N)
# MMA tile -> block covers 16x32 of C. Grid = (N//32, M//16). K is staged in
# chunks of FP4_TK_STAGE k64-tiles (256 k-values) x FP4_NSTAGES buffers.
# Smem/stage: A packed 16x128B + B packed 32x128B + A scales 16x16B + B scales
# 32x16B = 6912B -> 13.5KB/block at 2 stages (~7 blocks/SM on sm_120's 100KB).
#
# Math is IDENTICAL to fp4_gemm_kernel: same MMA, same k-ascending accumulation
# order per warp -> bit-identical output. The old kernel stays callable
# (NOMOS_FP4_OLD=1, or any shape with N%32!=0 / K%256!=0 falls back).
# ─────────────────────────────────────────────────────────────────────────────
# RTX PRO 6000 (Blackwell, 188 SMs, ~1.79 TB/s) re-tune sweep — HONEST NULL
# (2026-07-08, inst 44270246, all L1 12/12): the discrete NS3/TK4/WARPS4 geometry
# stays optimal on the 188-SM card. Engine base decode is flat across geometry:
#   NS3(ship) 48.5 | NS2 48.2 | WARPS8 48.2 tok/s ; NS4/TK6 worse (isolated probe,
#   gateup regresses). The wave-quant hypothesis (NS4 wins on more SMs) did NOT
#   pan out. Base decode does not respond to GEMM geometry because it is ENGINE-
#   OVERHEAD-bound, not GEMM-geometry-bound: the isolated fp4_gemm already runs at
#   ~64% of the 6000's DRAM peak (on par w/ llama.cpp's realized eff.), but the
#   engine only realizes ~45% — the ~30% delta is per-token overhead (~412 GEMM
#   launches + separate act-quant/attn/KV/layernorm kernels + launch gaps) vs
#   llama.cpp's fused decode. Gap is STRUCTURAL (needs fusion/launch-reduction),
#   not a tunable param. Spec-stack VB optimum = 5 (89.8; VB3/4 = 79/86, VB8 = 88.6).
# ─────────────────────────────────────────────────────────────────────────────
comptime FP4_SMEM_WARPS = 4        # warps/block = 8-col N-tiles per block
comptime FP4_TK_STAGE = 4          # k64 chunks per stage (256 k-values)
comptime FP4_NSTAGES = 3           # pipeline depth (2 = double buffer)
comptime _APK = 16 * FP4_TK_STAGE * 32                   # 2048B A packed / stage
comptime _BPK = 8 * FP4_SMEM_WARPS * FP4_TK_STAGE * 32   # 4096B B packed / stage
comptime _ASC = 16 * FP4_TK_STAGE * 4                    # 256B A scales / stage
comptime _BSC = 8 * FP4_SMEM_WARPS * FP4_TK_STAGE * 4    # 512B B scales / stage
comptime _STAGE_B = _APK + _BPK + _ASC + _BSC            # 6912B
comptime _OFF_BPK = _APK
comptime _OFF_ASC = _APK + _BPK
comptime _OFF_BSC = _APK + _BPK + _ASC
comptime _BLK_THREADS = 32 * FP4_SMEM_WARPS
comptime _A_UNITS = 16 * FP4_TK_STAGE * 2                # 16B copy units, A packed
comptime _B_UNITS = 8 * FP4_SMEM_WARPS * FP4_TK_STAGE * 2
comptime _ASC_UNITS = 16 * FP4_TK_STAGE                  # 4B copy units, A scales
comptime _BSC_UNITS = 8 * FP4_SMEM_WARPS * FP4_TK_STAGE


@always_inline
def _stage_ktile[
    smem_origin: MutOrigin, //, TK: Int = FP4_TK_STAGE
](
    sm: UnsafePointer[UInt8, smem_origin, address_space = AddressSpace.SHARED],
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],
    tid: Int, mrow: Int, ncol_blk: Int, kbase: Int, KB: Int, NB: Int,
):
    """Cooperatively cp.async one K-stage (TK k64 chunks) into smem.
    Packed operands move as 16B units (row-chunk halves), scales as 4B units;
    units are strided across the block's threads (generic over TK/warps —
    divisions are by comptime constants, strength-reduced). At TK=4, 4 warps:
    A packed 128 units (1/thread), B 256 (2/thread), A sc 64, B sc 128.
    R3: TK is a comptime parameter (default keeps the R1/R2 TK=4 layout)."""
    comptime OFF_BPK = 16 * TK * 32
    comptime OFF_ASC = OFF_BPK + 8 * FP4_SMEM_WARPS * TK * 32
    comptime OFF_BSC = OFF_ASC + 16 * TK * 4
    comptime A_UNITS = 16 * TK * 2
    comptime B_UNITS = 8 * FP4_SMEM_WARPS * TK * 2
    comptime ASC_UNITS = 16 * TK
    comptime BSC_UNITS = 8 * FP4_SMEM_WARPS * TK
    var kb2 = kbase >> 1     # packed-byte offset of the stage
    var kb16 = kbase >> 4    # e4m3 scale-byte offset of the stage

    # A packed: unit u -> row u//(2TK), chunk c=(u%(2TK))>>1, half h=(u&1)*16
    var ua = tid
    while ua < A_UNITS:
        var ar = ua // (2 * TK)
        var arem = ua % (2 * TK)
        var ac = arem >> 1
        var ah = (arem & 1) * 16
        async_copy[16](
            (a_bytes + ((mrow + ar) * KB + kb2 + ac * 32 + ah))
                .address_space_cast[AddressSpace.GLOBAL](),
            sm + ac * (16 * 32) + ar * 32 + ah,
        )
        ua += _BLK_THREADS
    # B packed: same split over 8*WARPS rows
    var ub = tid
    while ub < B_UNITS:
        var br = ub // (2 * TK)
        var brem = ub % (2 * TK)
        var bc = brem >> 1
        var bh = (brem & 1) * 16
        async_copy[16](
            (b_bytes + ((ncol_blk + br) * KB + kb2 + bc * 32 + bh))
                .address_space_cast[AddressSpace.GLOBAL](),
            sm + OFF_BPK + bc * (8 * FP4_SMEM_WARPS * 32) + br * 32 + bh,
        )
        ub += _BLK_THREADS
    # A scales: unit u -> row u//TK, chunk u%TK, 4B each
    var us = tid
    while us < ASC_UNITS:
        var sr = us // TK
        var sc = us % TK
        async_copy[4](
            (a_sc + ((mrow + sr) * NB + kb16 + sc * 4))
                .address_space_cast[AddressSpace.GLOBAL](),
            sm + OFF_ASC + sc * (16 * 4) + sr * 4,
        )
        us += _BLK_THREADS
    # B scales: same split over 8*WARPS rows
    var ut = tid
    while ut < BSC_UNITS:
        var tr = ut // TK
        var tc = ut % TK
        async_copy[4](
            (b_sc + ((ncol_blk + tr) * NB + kb16 + tc * 4))
                .address_space_cast[AddressSpace.GLOBAL](),
            sm + OFF_BSC + tc * (8 * FP4_SMEM_WARPS * 4) + tr * 4,
        )
        ut += _BLK_THREADS


def fp4_gemm_kernel_smem(
    c_out: UnsafePointer[Float32, MutAnyOrigin],   # [Mpad, Npad] fp32, row-major
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Mpad, K/2] packed E2M1 (activation)
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Npad, K/2] packed E2M1 (weight)
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Mpad, K/16] e4m3 block scales (A)
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Npad, K/16] e4m3 block scales (B)
    M_arg: Int32, N_arg: Int32, K_arg: Int32,                        # PADDED: M%16==0, N%32==0, K%256==0
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)
    var KB = K // 2
    var NB = K // 16
    var tid = Int(thread_idx.x)      # 0..127
    var w = tid >> 5                 # warp 0..3 -> which 8-col N-tile
    var lane = tid & 31
    var gid = lane >> 2              # 0..7 -> row within the 16-row tile
    var q = lane & 3                 # 0..3 -> k-quad selector / 2 output columns
    var mrow = Int(block_idx.y) * 16
    var ncol_blk = Int(block_idx.x) * (8 * FP4_SMEM_WARPS)
    var ncol = ncol_blk + w * 8

    var sm = stack_allocation[
        FP4_NSTAGES * _STAGE_B, UInt8,
        address_space = AddressSpace.SHARED, alignment=16,
    ]()

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    var NT = K // (FP4_TK_STAGE * 64)    # K-stages
    # Fill the pipeline: stages 0..NSTAGES-2 in flight before compute starts.
    var npre = FP4_NSTAGES - 1
    if npre > NT:
        npre = NT
    for s in range(npre):
        _stage_ktile(sm + s * _STAGE_B, a_bytes, b_bytes, a_sc, b_sc,
                     tid, mrow, ncol_blk, s * (FP4_TK_STAGE * 64), KB, NB)
        async_copy_commit_group()

    for kt in range(NT):
        var pf = kt + FP4_NSTAGES - 1    # prefetch distance = NSTAGES-1
        if pf < NT:
            _stage_ktile(
                sm + (pf % FP4_NSTAGES) * _STAGE_B,
                a_bytes, b_bytes, a_sc, b_sc,
                tid, mrow, ncol_blk, pf * (FP4_TK_STAGE * 64), KB, NB,
            )
            async_copy_commit_group()
            async_copy_wait_group(FP4_NSTAGES - 1)   # stage kt landed
        else:
            # Drain: cp.async.wait_group needs an IMMEDIATE count — a runtime
            # (NT-kt-1) fails to lower. Waiting all remaining groups only
            # over-waits the final NSTAGES-1 iterations; negligible.
            async_copy_wait_group(0)
        barrier()

        var st = sm + (kt % FP4_NSTAGES) * _STAGE_B
        comptime for cc in range(FP4_TK_STAGE):
            var a0 = (st + cc * (16 * 32) + gid * 32 + q * 4).bitcast[UInt32]()[0]
            var a1 = (st + cc * (16 * 32) + (gid + 8) * 32 + q * 4).bitcast[UInt32]()[0]
            var a2 = (st + cc * (16 * 32) + gid * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var a3 = (st + cc * (16 * 32) + (gid + 8) * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var b0 = (st + _OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + q * 4)
                .bitcast[UInt32]()[0]
            var b1 = (st + _OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + 16 + q * 4)
                .bitcast[UInt32]()[0]

            var sca = UInt32(0)
            if q == 0:
                sca = (st + _OFF_ASC + cc * (16 * 4) + gid * 4).bitcast[UInt32]()[0]
            elif q == 1:
                sca = (st + _OFF_ASC + cc * (16 * 4) + (gid + 8) * 4).bitcast[UInt32]()[0]
            var scb = UInt32(0)
            if q == 0:
                scb = (st + _OFF_BSC + cc * (8 * FP4_SMEM_WARPS * 4) + (w * 8 + gid) * 4)
                    .bitcast[UInt32]()[0]

            # Same guard rationale as fp4_gemm_kernel (sm_120/121-only warp MMA).
            comptime if _is_sm_120x_or_newer():
                var r = inlined_assembly[
                    (
                        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                        " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                        " {$15}, {0, 0};"
                    ),
                    _RegisterPackType[Float32, Float32, Float32, Float32],
                    constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
                ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
                c0 = r[0]
                c1 = r[1]
                c2 = r[2]
                c3 = r[3]
        barrier()    # stage (kt&1) fully consumed before kt+2 overwrites it

    c_out[(mrow + gid) * N + ncol + 2 * q] = c0
    c_out[(mrow + gid) * N + ncol + 2 * q + 1] = c1
    c_out[(mrow + gid + 8) * N + ncol + 2 * q] = c2
    c_out[(mrow + gid + 8) * N + ncol + 2 * q + 1] = c3


# ─────────────────────────────────────────────────────────────────────────────
# R2 Phase B (SHIPPED half): fused-postscale M=1 GEMM epilogue.
#
# fp4_gemm_kernel_smem_ps == fp4_gemm_kernel_smem verbatim (staging, barriers, MMA)
# except the epilogue: instead of spilling all 16 padded rows to c_pad and running
# a separate postscale launch, lanes holding the LIVE row-0 outputs (gid==0) write
# d_out[n] = c * act_global[0] * weight_global directly — same multiply order as
# postscale_w4a4_kernel -> bit-exact — killing the postscale launch AND the whole
# c_pad round-trip (16xNx4B written + Nx4B reread per GEMM). Kill-switch:
# NOMOS_FP4_QAFUSE=0 -> R1 smem GEMM + separate postscale.
#
# R3 (GEMM geometry): pipeline depth NS and stage K-width TK are comptime
# parameters. R1 diagnostics flagged prefetch depth (NS=4 -> 1.93x) and stage
# width (TK=6 -> 2.02x) as the gateup levers — both measured on the R1 kernel
# whose epilogue still wrote c_pad. On THIS kernel the R3 interleaved probe
# found NS=4 ships nothing (qslide regresses 1.09x via wave quantization —
# 27.6KB/block = 3 blocks/SM pushes 256 blocks past the 280-block single wave;
# everything else ties within noise). TK=6 (NS=3, 31.1KB/block, K%384 shapes
# only — per-shape dispatch, oslide/ofull keep TK=4, NO old-kernel fallback)
# cuts stage loops 21->14 on the K=5376 heavies. K-ascending accumulation order
# per warp is UNCHANGED for any NS/TK -> bit-identical output.
# async_copy_wait_group(NS-1) stays a comptime immediate per instantiation
# (runtime counts fail to lower). Decode-only: the bare smem kernel (prefill,
# M>=16 tiles) keeps FP4_NSTAGES=3 / FP4_TK_STAGE=4.
# ─────────────────────────────────────────────────────────────────────────────


def fp4_gemm_kernel_smem_ps[NS: Int, TK: Int = FP4_TK_STAGE](
    d_out: UnsafePointer[Float32, MutAnyOrigin],   # [N] fp32 POSTSCALED output (row 0)
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [16, K/2] packed E2M1 (act, M=1 pad)
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Npad, K/2] packed E2M1 (weight)
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [16, K/16] e4m3 block scales (A)
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Npad, K/16] e4m3 block scales (B)
    act_global: UnsafePointer[Float32, MutAnyOrigin],  # [0] = per-token gscale
    weight_global: Float32,
    N_arg: Int32, K_arg: Int32,                                # N%32==0, K%(TK*64)==0; M==1 (grid.y==1)
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var N = Int(N_arg)
    var K = Int(K_arg)
    comptime STAGE_B = (16 + 8 * FP4_SMEM_WARPS) * TK * 32 \
        + (16 + 8 * FP4_SMEM_WARPS) * TK * 4       # packed + scales, TK-generic
    comptime OFF_BPK = 16 * TK * 32
    comptime OFF_ASC = OFF_BPK + 8 * FP4_SMEM_WARPS * TK * 32
    comptime OFF_BSC = OFF_ASC + 16 * TK * 4
    var KB = K // 2
    var NB = K // 16
    var tid = Int(thread_idx.x)      # 0..127
    var w = tid >> 5
    var lane = tid & 31
    var gid = lane >> 2
    var q = lane & 3
    var mrow = 0                     # M=1: single 16-row MMA tile (grid.y == 1)
    var ncol_blk = Int(block_idx.x) * (8 * FP4_SMEM_WARPS)
    var ncol = ncol_blk + w * 8

    var sm = stack_allocation[
        NS * STAGE_B, UInt8,
        address_space = AddressSpace.SHARED, alignment=16,
    ]()

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    var NT = K // (TK * 64)
    var npre = NS - 1
    if npre > NT:
        npre = NT
    for s in range(npre):
        _stage_ktile[TK](sm + s * STAGE_B, a_bytes, b_bytes, a_sc, b_sc,
                         tid, mrow, ncol_blk, s * (TK * 64), KB, NB)
        async_copy_commit_group()

    for kt in range(NT):
        var pf = kt + NS - 1
        if pf < NT:
            _stage_ktile[TK](
                sm + (pf % NS) * STAGE_B,
                a_bytes, b_bytes, a_sc, b_sc,
                tid, mrow, ncol_blk, pf * (TK * 64), KB, NB,
            )
            async_copy_commit_group()
            async_copy_wait_group(Int32(NS - 1))     # stage kt landed (comptime NS
                                                     # -> folds to an immediate)
        else:
            async_copy_wait_group(0)                 # drain (same rationale as R1)
        barrier()

        var st = sm + (kt % NS) * STAGE_B
        comptime for cc in range(TK):
            var a0 = (st + cc * (16 * 32) + gid * 32 + q * 4).bitcast[UInt32]()[0]
            var a1 = (st + cc * (16 * 32) + (gid + 8) * 32 + q * 4).bitcast[UInt32]()[0]
            var a2 = (st + cc * (16 * 32) + gid * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var a3 = (st + cc * (16 * 32) + (gid + 8) * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var b0 = (st + OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + q * 4)
                .bitcast[UInt32]()[0]
            var b1 = (st + OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + 16 + q * 4)
                .bitcast[UInt32]()[0]

            var sca = UInt32(0)
            if q == 0:
                sca = (st + OFF_ASC + cc * (16 * 4) + gid * 4).bitcast[UInt32]()[0]
            elif q == 1:
                sca = (st + OFF_ASC + cc * (16 * 4) + (gid + 8) * 4).bitcast[UInt32]()[0]
            var scb = UInt32(0)
            if q == 0:
                scb = (st + OFF_BSC + cc * (8 * FP4_SMEM_WARPS * 4) + (w * 8 + gid) * 4)
                    .bitcast[UInt32]()[0]

            # Same guard rationale as fp4_gemm_kernel (sm_120/121-only warp MMA).
            comptime if _is_sm_120x_or_newer():
                var r = inlined_assembly[
                    (
                        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                        " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                        " {$15}, {0, 0};"
                    ),
                    _RegisterPackType[Float32, Float32, Float32, Float32],
                    constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
                ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
                c0 = r[0]
                c1 = r[1]
                c2 = r[2]
                c3 = r[3]
        barrier()    # stage fully consumed before it is overwritten

    # Fused postscale epilogue (M=1): only row 0 lives (rows 1..15 = zero pads, their
    # c is discarded — no c_pad round-trip). Multiply order matches
    # postscale_w4a4_kernel (c * act_global[m] * weight_global) -> bit-exact.
    if gid == 0:
        var ag0 = act_global[0]
        d_out[ncol + 2 * q] = c0 * ag0 * weight_global
        d_out[ncol + 2 * q + 1] = c1 * ag0 * weight_global


def fp4_gemm_kernel_smem_ps_grouped4[NS: Int, TK: Int = FP4_TK_STAGE](
    d_out0: UnsafePointer[Float32, MutAnyOrigin],
    d_out1: UnsafePointer[Float32, MutAnyOrigin],
    d_out2: UnsafePointer[Float32, MutAnyOrigin],
    d_out3: UnsafePointer[Float32, MutAnyOrigin],
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],
    act_global: UnsafePointer[Float32, MutAnyOrigin],
    b_bytes0: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes1: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes2: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes3: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc0: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc1: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc2: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc3: UnsafePointer[UInt8, MutAnyOrigin],
    weight_global0: Float32,
    weight_global1: Float32,
    weight_global2: Float32,
    weight_global3: Float32,
    N0_arg: Int32,
    N1_arg: Int32,
    N2_arg: Int32,
    N3_arg: Int32,
    K_arg: Int32,
):
    """Four-pointer variant of fp4_gemm_kernel_smem_ps for one shared A row.

    block_idx.x selects one block-uniform output segment, then the body executes the
    identical 32-column tile, K-stage traversal, MMA sequence, and fused-postscale
    epilogue as fp4_gemm_kernel_smem_ps. Each segment retains its own packed weights,
    scale bytes, weight global, and output buffer; no weight repack or rescale exists.
    """
    var N0 = Int(N0_arg)
    var N1 = Int(N1_arg)
    var N2 = Int(N2_arg)
    var N3 = Int(N3_arg)
    var K = Int(K_arg)
    comptime TILE_N = 8 * FP4_SMEM_WARPS
    var blocks0 = N0 // TILE_N
    var blocks1 = N1 // TILE_N
    var blocks2 = N2 // TILE_N
    var blocks3 = N3 // TILE_N
    var global_block = Int(block_idx.x)
    if global_block >= blocks0 + blocks1 + blocks2 + blocks3:
        return
    var local_block = global_block
    var d_out = d_out0
    var b_bytes = b_bytes0
    var b_sc = b_sc0
    var weight_global = weight_global0
    if global_block >= blocks0:
        if global_block < blocks0 + blocks1:
            local_block = global_block - blocks0
            d_out = d_out1
            b_bytes = b_bytes1
            b_sc = b_sc1
            weight_global = weight_global1
        elif global_block < blocks0 + blocks1 + blocks2:
            local_block = global_block - blocks0 - blocks1
            d_out = d_out2
            b_bytes = b_bytes2
            b_sc = b_sc2
            weight_global = weight_global2
        else:
            local_block = global_block - blocks0 - blocks1 - blocks2
            d_out = d_out3
            b_bytes = b_bytes3
            b_sc = b_sc3
            weight_global = weight_global3

    # From this point through the epilogue, this is fp4_gemm_kernel_smem_ps verbatim
    # except that local_block replaces block_idx.x after the block-uniform segment map.
    comptime STAGE_B = (16 + 8 * FP4_SMEM_WARPS) * TK * 32 \
        + (16 + 8 * FP4_SMEM_WARPS) * TK * 4
    comptime OFF_BPK = 16 * TK * 32
    comptime OFF_ASC = OFF_BPK + 8 * FP4_SMEM_WARPS * TK * 32
    comptime OFF_BSC = OFF_ASC + 16 * TK * 4
    var KB = K // 2
    var NB = K // 16
    var tid = Int(thread_idx.x)
    var w = tid >> 5
    var lane = tid & 31
    var gid = lane >> 2
    var q = lane & 3
    var mrow = 0
    var ncol_blk = local_block * TILE_N
    var ncol = ncol_blk + w * 8

    var sm = stack_allocation[
        NS * STAGE_B, UInt8,
        address_space = AddressSpace.SHARED, alignment=16,
    ]()

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    var NT = K // (TK * 64)
    var npre = NS - 1
    if npre > NT:
        npre = NT
    for s in range(npre):
        _stage_ktile[TK](sm + s * STAGE_B, a_bytes, b_bytes, a_sc, b_sc,
                         tid, mrow, ncol_blk, s * (TK * 64), KB, NB)
        async_copy_commit_group()

    for kt in range(NT):
        var pf = kt + NS - 1
        if pf < NT:
            _stage_ktile[TK](
                sm + (pf % NS) * STAGE_B,
                a_bytes, b_bytes, a_sc, b_sc,
                tid, mrow, ncol_blk, pf * (TK * 64), KB, NB,
            )
            async_copy_commit_group()
            async_copy_wait_group(Int32(NS - 1))
        else:
            async_copy_wait_group(0)
        barrier()

        var st = sm + (kt % NS) * STAGE_B
        comptime for cc in range(TK):
            var a0 = (st + cc * (16 * 32) + gid * 32 + q * 4).bitcast[UInt32]()[0]
            var a1 = (st + cc * (16 * 32) + (gid + 8) * 32 + q * 4).bitcast[UInt32]()[0]
            var a2 = (st + cc * (16 * 32) + gid * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var a3 = (st + cc * (16 * 32) + (gid + 8) * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var b0 = (st + OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + q * 4)
                .bitcast[UInt32]()[0]
            var b1 = (st + OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + 16 + q * 4)
                .bitcast[UInt32]()[0]

            var sca = UInt32(0)
            if q == 0:
                sca = (st + OFF_ASC + cc * (16 * 4) + gid * 4).bitcast[UInt32]()[0]
            elif q == 1:
                sca = (st + OFF_ASC + cc * (16 * 4) + (gid + 8) * 4).bitcast[UInt32]()[0]
            var scb = UInt32(0)
            if q == 0:
                scb = (st + OFF_BSC + cc * (8 * FP4_SMEM_WARPS * 4) + (w * 8 + gid) * 4)
                    .bitcast[UInt32]()[0]

            comptime if _is_sm_120x_or_newer():
                var r = inlined_assembly[
                    (
                        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                        " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                        " {$15}, {0, 0};"
                    ),
                    _RegisterPackType[Float32, Float32, Float32, Float32],
                    constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
                ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
                c0 = r[0]
                c1 = r[1]
                c2 = r[2]
                c3 = r[3]
        barrier()

    if gid == 0:
        var ag0 = act_global[0]
        d_out[ncol + 2 * q] = c0 * ag0 * weight_global
        d_out[ncol + 2 * q + 1] = c1 * ag0 * weight_global


def _launch_ps[NS: Int, TK: Int = FP4_TK_STAGE](
    ctx: DeviceContext,
    d_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    N: Int, K: Int,
) raises:
    """Enqueue one ps-GEMM instantiation at comptime depth NS / stage width TK."""
    var kern = ctx.compile_function[fp4_gemm_kernel_smem_ps[NS, TK]]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_sc)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_global)),
        weight_global, Int32(N), Int32(K),
        grid_dim=(N // (8 * FP4_SMEM_WARPS), 1),
        block_dim=32 * FP4_SMEM_WARPS,
    )


def _launch_ps_grouped4[NS: Int, TK: Int = FP4_TK_STAGE](
    ctx: DeviceContext,
    d_out0: UInt64, d_out1: UInt64, d_out2: UInt64, d_out3: UInt64,
    a_bytes: UInt64, a_sc: UInt64, act_global: UInt64,
    b_bytes0: UInt64, b_bytes1: UInt64, b_bytes2: UInt64, b_bytes3: UInt64,
    b_sc0: UInt64, b_sc1: UInt64, b_sc2: UInt64, b_sc3: UInt64,
    weight_global0: Float32, weight_global1: Float32,
    weight_global2: Float32, weight_global3: Float32,
    N0: Int, N1: Int, N2: Int, N3: Int, K: Int,
) raises:
    var kern = ctx.compile_function[fp4_gemm_kernel_smem_ps_grouped4[NS, TK]]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out0)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out1)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out2)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out3)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_sc)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_global)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes0)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes1)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes2)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes3)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc0)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc1)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc2)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc3)),
        weight_global0, weight_global1, weight_global2, weight_global3,
        Int32(N0), Int32(N1), Int32(N2), Int32(N3), Int32(K),
        grid_dim=((N0 + N1 + N2 + N3) // (8 * FP4_SMEM_WARPS), 1),
        block_dim=32 * FP4_SMEM_WARPS,
    )


def _fp4_geo_base_forced() -> Bool:
    """NOMOS_FP4_GEOBASE=1 -> force the R2 base geometry (NS=3/TK=4) on the ps
    kernel (geometry A/B kill-switch)."""
    var name = String("NOMOS_FP4_GEOBASE")
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
    return p[0] == UInt8(49)     # '1'


def _ps_tk6_shape(N: Int, K: Int) -> Bool:
    """R3 shape dispatch: which decode shapes run the TK=6 (NS=3) ps kernel.
    ANSWER (R3, measured): NONE — base geometry (NS=3/TK=4) ships everywhere.
    The 10-shape interleaved L2-cold probe (2 full runs) found ONE reproducible
    isolated win: gateup (N=21504, K=5376) at 0.89x/0.93x vs base. But the e2e
    A/B/A sandwich REJECTED it: TK6-gateup dispatch 20.61/20.75 tok/s vs base
    21.11 (same session, every bucket agrees) — the isolated-probe win does not
    transfer to engine context (probe = L2-cold GEMM-only at saturated steady
    clocks; engine interleaves quant/attention and runs gate+up back-to-back on
    a shared A row; TK6's 31.1KB/block = 3 blocks/SM occupancy cost evidently
    dominates there). NS=4 was rejected at the probe already (qslide 1.09-1.11x
    regression via wave quantization, no reproducible win anywhere). Keep FALSE
    unless a future round shows an E2E win; the probe lanes + explicit-geometry
    launcher (gpu_fp4_gemm_ps_geo) stay as the instrument."""
    _ = N
    _ = K
    return False


def gpu_fp4_gemm_ps(
    ctx: DeviceContext,
    d_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    N: Int, K: Int,
) raises:
    """Launch the fused-postscale W4A4 GEMM (decode, M=1): NVFP4 A scratch in,
    postscaled fp32 row out — no c_pad, no postscale launch. Callers gate on
    qa_fuse_route(); shape must satisfy N%32==0, K%256==0. R3: stage geometry
    picked per shape (_ps_tk6_shape); NOMOS_FP4_GEOBASE=1 forces R2 geometry."""
    if Float64(ctx.default_device_info.compute) < 12.0:
        raise Error(
            "NVFP4 W4A4 fused-postscale GEMM requires sm_120/sm_121 (compute>=12.0);"
            " run W4A16 (NOMOS_W4A4=0) on datacenter Blackwell."
        )
    if _ps_tk6_shape(N, K) and not _fp4_geo_base_forced():
        _launch_ps[3, 6](ctx, d_out, a_bytes, b_bytes, a_sc, b_sc,
                         act_global, weight_global, N, K)
        return
    _launch_ps[3](ctx, d_out, a_bytes, b_bytes, a_sc, b_sc,
                  act_global, weight_global, N, K)


def gpu_fp4_gemm_ps_grouped4(
    ctx: DeviceContext,
    d_out0: UInt64, d_out1: UInt64, d_out2: UInt64, d_out3: UInt64,
    a_bytes: UInt64, a_sc: UInt64, act_global: UInt64,
    b_bytes0: UInt64, b_bytes1: UInt64, b_bytes2: UInt64, b_bytes3: UInt64,
    b_sc0: UInt64, b_sc1: UInt64, b_sc2: UInt64, b_sc3: UInt64,
    weight_global0: Float32, weight_global1: Float32,
    weight_global2: Float32, weight_global3: Float32,
    N0: Int, N1: Int, N2: Int, N3: Int, K: Int,
) raises:
    """One launch over four independent B/output segments sharing one quantized A row.

    This is the production NS=3/TK=4 fp4_gemm_kernel_smem_ps geometry. Callers must
    require each N%32==0, K%256==0, and the ordinary qa_fuse_route kill switches.
    """
    if Float64(ctx.default_device_info.compute) < 12.0:
        raise Error(
            "NVFP4 W4A4 grouped fused-postscale GEMM requires sm_120/sm_121 "
            "(compute>=12.0); run the ordinary projection dispatch on this device."
        )
    _launch_ps_grouped4[3](
        ctx,
        d_out0, d_out1, d_out2, d_out3,
        a_bytes, a_sc, act_global,
        b_bytes0, b_bytes1, b_bytes2, b_bytes3,
        b_sc0, b_sc1, b_sc2, b_sc3,
        weight_global0, weight_global1, weight_global2, weight_global3,
        N0, N1, N2, N3, K,
    )


def gpu_fp4_gemm_ps_geo(
    ctx: DeviceContext,
    d_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    N: Int, K: Int, ns: Int, tk: Int,
) raises:
    """Probe-only explicit-geometry launcher (A/B instrument): (ns, tk) in
    {(3,4), (4,4), (3,6)}. TK=6 requires K % 384 == 0 (caller checks)."""
    if Float64(ctx.default_device_info.compute) < 12.0:
        raise Error("NVFP4 W4A4 ps GEMM requires sm_120/sm_121 (compute>=12.0).")
    if tk == 6:
        _launch_ps[3, 6](ctx, d_out, a_bytes, b_bytes, a_sc, b_sc,
                         act_global, weight_global, N, K)
        return
    if ns == 4:
        _launch_ps[4](ctx, d_out, a_bytes, b_bytes, a_sc, b_sc,
                      act_global, weight_global, N, K)
        return
    _launch_ps[3](ctx, d_out, a_bytes, b_bytes, a_sc, b_sc,
                  act_global, weight_global, N, K)


# ─────────────────────────────────────────────────────────────────────────────
# Spec-verify M>1 fused-postscale GEMM (extends the M=1 ps kernel to M rows).
#
# WHY: eagle3 spec-decode VERIFY runs M=k+1 (=4) rows. The M=1 fused path
# (qa_fuse_route -> gpu_fp4_gemm_ps) only fires for M==1, so verify fell back to
# the OLD chain: one-warp-per-row quant_act_nvfp4_kernel (~63us/row, the 30ms/fwd
# bottleneck) + bare fp4_gemm into c_pad + a separate postscale launch. This
# kernel is fp4_gemm_kernel_smem_ps VERBATIM (staging, barriers, MMA, k-order)
# except the epilogue writes the M LIVE rows (rows 0..M-1) directly from the same
# 16-row MMA m-tile — NO c_pad round-trip, NO postscale launch. The M-row GEMM is
# row-independent (each output C[m,n] = Σ_k A[m,k]·B[n,k] depends only on row m),
# so a single 16-row m-tile (1<M<=16, pad rows M..15 = quant zeros) covers verify.
# Multiply order matches postscale_w4a4_kernel (c * act_global[m] * weight_global)
# -> BIT-EXACT to the OLD M-row path. The M=1 decode kernel above is UNCHANGED.
# ─────────────────────────────────────────────────────────────────────────────


def fp4_gemm_kernel_smem_ps_m[NS: Int, TK: Int = FP4_TK_STAGE](
    d_out: UnsafePointer[Float32, MutAnyOrigin],   # [M, N] fp32 POSTSCALED output
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [16, K/2] packed E2M1 (act, Mpad=16)
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Npad, K/2] packed E2M1 (weight)
    a_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [16, K/16] e4m3 block scales (A)
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Npad, K/16] e4m3 block scales (B)
    act_global: UnsafePointer[Float32, MutAnyOrigin],  # [16] per-row gscale
    weight_global: Float32,
    M_arg: Int32, N_arg: Int32, K_arg: Int32,                        # 1<M<=16; N%32==0, K%(TK*64)==0
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)
    comptime STAGE_B = (16 + 8 * FP4_SMEM_WARPS) * TK * 32 \
        + (16 + 8 * FP4_SMEM_WARPS) * TK * 4       # packed + scales, TK-generic
    comptime OFF_BPK = 16 * TK * 32
    comptime OFF_ASC = OFF_BPK + 8 * FP4_SMEM_WARPS * TK * 32
    comptime OFF_BSC = OFF_ASC + 16 * TK * 4
    var KB = K // 2
    var NB = K // 16
    var tid = Int(thread_idx.x)      # 0..127
    var w = tid >> 5
    var lane = tid & 31
    var gid = lane >> 2
    var q = lane & 3
    var mrow = 0                     # single 16-row MMA tile (grid.y == 1)
    var ncol_blk = Int(block_idx.x) * (8 * FP4_SMEM_WARPS)
    var ncol = ncol_blk + w * 8

    var sm = stack_allocation[
        NS * STAGE_B, UInt8,
        address_space = AddressSpace.SHARED, alignment=16,
    ]()

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    var NT = K // (TK * 64)
    var npre = NS - 1
    if npre > NT:
        npre = NT
    for s in range(npre):
        _stage_ktile[TK](sm + s * STAGE_B, a_bytes, b_bytes, a_sc, b_sc,
                         tid, mrow, ncol_blk, s * (TK * 64), KB, NB)
        async_copy_commit_group()

    for kt in range(NT):
        var pf = kt + NS - 1
        if pf < NT:
            _stage_ktile[TK](
                sm + (pf % NS) * STAGE_B,
                a_bytes, b_bytes, a_sc, b_sc,
                tid, mrow, ncol_blk, pf * (TK * 64), KB, NB,
            )
            async_copy_commit_group()
            async_copy_wait_group(Int32(NS - 1))     # stage kt landed (comptime NS
                                                     # -> folds to an immediate)
        else:
            async_copy_wait_group(0)                 # drain (same rationale as R1)
        barrier()

        var st = sm + (kt % NS) * STAGE_B
        comptime for cc in range(TK):
            var a0 = (st + cc * (16 * 32) + gid * 32 + q * 4).bitcast[UInt32]()[0]
            var a1 = (st + cc * (16 * 32) + (gid + 8) * 32 + q * 4).bitcast[UInt32]()[0]
            var a2 = (st + cc * (16 * 32) + gid * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var a3 = (st + cc * (16 * 32) + (gid + 8) * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var b0 = (st + OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + q * 4)
                .bitcast[UInt32]()[0]
            var b1 = (st + OFF_BPK + cc * (8 * FP4_SMEM_WARPS * 32) + (w * 8 + gid) * 32 + 16 + q * 4)
                .bitcast[UInt32]()[0]

            var sca = UInt32(0)
            if q == 0:
                sca = (st + OFF_ASC + cc * (16 * 4) + gid * 4).bitcast[UInt32]()[0]
            elif q == 1:
                sca = (st + OFF_ASC + cc * (16 * 4) + (gid + 8) * 4).bitcast[UInt32]()[0]
            var scb = UInt32(0)
            if q == 0:
                scb = (st + OFF_BSC + cc * (8 * FP4_SMEM_WARPS * 4) + (w * 8 + gid) * 4)
                    .bitcast[UInt32]()[0]

            # Same guard rationale as fp4_gemm_kernel (sm_120/121-only warp MMA).
            comptime if _is_sm_120x_or_newer():
                var r = inlined_assembly[
                    (
                        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                        " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                        " {$15}, {0, 0};"
                    ),
                    _RegisterPackType[Float32, Float32, Float32, Float32],
                    constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
                ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
                c0 = r[0]
                c1 = r[1]
                c2 = r[2]
                c3 = r[3]
        barrier()    # stage fully consumed before it is overwritten

    # ── fused postscale epilogue (M rows): the 16-row MMA tile holds rows 0..7 in
    # c0/c1 (lanes gid) and rows 8..15 in c2/c3 (lanes gid+8). Write only the M live
    # rows, each scaled by ITS OWN act_global[row]; pad rows M..15 are discarded (no
    # c_pad round-trip). Multiply order = postscale_w4a4_kernel -> bit-exact. ──
    if gid < M:
        var ag0 = act_global[gid]
        d_out[gid * N + ncol + 2 * q] = c0 * ag0 * weight_global
        d_out[gid * N + ncol + 2 * q + 1] = c1 * ag0 * weight_global
    if gid + 8 < M:
        var ag8 = act_global[gid + 8]
        d_out[(gid + 8) * N + ncol + 2 * q] = c2 * ag8 * weight_global
        d_out[(gid + 8) * N + ncol + 2 * q + 1] = c3 * ag8 * weight_global


def _launch_ps_m[NS: Int, TK: Int = FP4_TK_STAGE](
    ctx: DeviceContext,
    d_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    M: Int, N: Int, K: Int,
) raises:
    """Enqueue one M-row ps-GEMM instantiation at comptime depth NS / stage width TK."""
    var kern = ctx.compile_function[fp4_gemm_kernel_smem_ps_m[NS, TK]]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_sc)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_global)),
        weight_global, Int32(M), Int32(N), Int32(K),
        grid_dim=(N // (8 * FP4_SMEM_WARPS), 1),
        block_dim=32 * FP4_SMEM_WARPS,
    )


def gpu_fp4_gemm_ps_m(
    ctx: DeviceContext,
    d_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    M: Int, N: Int, K: Int,
) raises:
    """Launch the M-row fused-postscale W4A4 GEMM (spec-verify, 1<M<=16): PRE-quantized
    NVFP4 activation [16,*] in (Mpad=16), postscaled fp32 [M,N] out — no c_pad, no
    postscale launch. Callers gate on qa_fuse_route_m(); shape must satisfy N%32==0,
    K%256==0. Base geometry NS=3/TK=4 (the R3 TK6/NS4 probe found no E2E win)."""
    if Float64(ctx.default_device_info.compute) < 12.0:
        raise Error(
            "NVFP4 W4A4 M-row fused-postscale GEMM requires sm_120/sm_121 (compute>=12.0);"
            " run W4A16 (NOMOS_W4A4=0) on datacenter Blackwell."
        )
    _launch_ps_m[3](ctx, d_out, a_bytes, b_bytes, a_sc, b_sc,
                    act_global, weight_global, M, N, K)


# ─────────────────────────────────────────────────────────────────────────────
# R2 Phase B (EXPERIMENTAL half, NOT dispatched): in-GEMM A-tile quantization.
#
# The original Phase-B design staged the RAW fp32 A row and encoded it NVFP4 in
# smem between the stage wait and the MMA, killing the standalone quant launch.
# It is BIT-EXACT (probe parity lane passes on all shapes) but FAILED the
# pre-registered B-perf gate (1.8-5.5x the bare GEMM, gate <=1.05x): every one of
# the N/32 blocks redundantly re-encodes the same A row per stage, and the encode
# chain (~25 fp32 divides on 16 lanes between two barriers) exposes ~2.5-3us per
# k-stage. The redundancy is structural to the N-parallel decomposition, so the
# shipped Phase B instead uses (a) the parallel fast quant (lib/fp4_act,
# gpu_quant_act_nvfp4_fast: amax micro-kernel + grid-parallel encode, one encode
# per DISTINCT activation) and (b) fp4_gemm_kernel_smem_ps above. This kernel is
# kept for R3 iteration; only tools/fp4_gemm_probe.mojo exercises it (parity +
# timing lanes). Seam precedent: q4_mmq_s16_q8stage_verify_kernel (plain smem
# stores mixed with cp.async stage fills under one barrier).
# ─────────────────────────────────────────────────────────────────────────────
comptime _QA_ARAW = FP4_TK_STAGE * 64 * 4        # 1024B raw fp32 A (row 0) / stage
comptime _QA_STAGE_B = _STAGE_B + _QA_ARAW       # 7936B/stage -> 23.25KB @ 3 stages
comptime _QA_OFF_ARAW = _STAGE_B                 # raw A appended at the stage tail
comptime _QA_ARAW_UNITS = _QA_ARAW // 16         # 64 x 16B cp.async units
comptime _QA_BLOCKS = FP4_TK_STAGE * 4           # 16-el e4m3 blocks per stage (16)


@always_inline
def _stage_ktile_qa[
    smem_origin: MutOrigin, //,
    W: Int = FP4_SMEM_WARPS, TK: Int = FP4_TK_STAGE,
](
    sm: UnsafePointer[UInt8, smem_origin, address_space = AddressSpace.SHARED],
    a_raw8: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],
    tid: Int, ncol_blk: Int, kbase: Int, KB: Int, NB: Int,
):
    """QA stage loader: B packed/scales EXACTLY as _stage_ktile; A arrives as the RAW
    fp32 K-slice of row 0 (k-contiguous, 16B units -> QA_OFF_ARAW). A-packed/A-scale
    smem is NOT loaded from global — the in-kernel encode writes row 0 and rows 1..15
    keep the kernel-start zeros (= quant-kernel pad-row bytes).

    W/TK-generic: the layout constants below are the file-scope _APK/_BPK/... aliases
    re-derived from the parameters, so a swept config lays out shared memory the same
    way the alias version does at W=FP4_SMEM_WARPS, TK=FP4_TK_STAGE (defaults)."""
    comptime BLK_THREADS = 32 * W
    comptime B_UNITS = 8 * W * TK * 2
    comptime BSC_UNITS = 8 * W * TK
    comptime OFF_BPK = 16 * TK * 32                                  # = _APK
    comptime OFF_BSC = OFF_BPK + 8 * W * TK * 32 + 16 * TK * 4       # = _APK+_BPK+_ASC
    comptime STAGE_B = OFF_BSC + 8 * W * TK * 4                      # + _BSC
    comptime QA_OFF_ARAW = STAGE_B                                   # raw A at the stage tail
    comptime QA_ARAW_UNITS = (TK * 64 * 4) // 16
    var kb2 = kbase >> 1
    var kb16 = kbase >> 4
    # B packed: same split as _stage_ktile
    var ub = tid
    while ub < B_UNITS:
        var br = ub // (2 * TK)
        var brem = ub % (2 * TK)
        var bc = brem >> 1
        var bh = (brem & 1) * 16
        async_copy[16](
            (b_bytes + ((ncol_blk + br) * KB + kb2 + bc * 32 + bh))
                .address_space_cast[AddressSpace.GLOBAL](),
            sm + OFF_BPK + bc * (8 * W * 32) + br * 32 + bh,
        )
        ub += BLK_THREADS
    # B scales: same split as _stage_ktile
    var ut = tid
    while ut < BSC_UNITS:
        var tr = ut // TK
        var tc = ut % TK
        async_copy[4](
            (b_sc + ((ncol_blk + tr) * NB + kb16 + tc * 4))
                .address_space_cast[AddressSpace.GLOBAL](),
            sm + OFF_BSC + tc * (8 * W * 4) + tr * 4,
        )
        ut += BLK_THREADS
    # A raw fp32 (row 0 only): stage's 256 k-values = 1KB, k-contiguous
    var ur = tid
    while ur < QA_ARAW_UNITS:
        async_copy[16](
            (a_raw8 + (kbase * 4 + ur * 16)).address_space_cast[AddressSpace.GLOBAL](),
            sm + QA_OFF_ARAW + ur * 16,
        )
        ur += BLK_THREADS


def fp4_gemm_kernel_smem_qa[
    W: Int = FP4_SMEM_WARPS, TK: Int = FP4_TK_STAGE, NS: Int = FP4_NSTAGES
](
    d_out: UnsafePointer[Float32, MutAnyOrigin],   # [N] fp32 POSTSCALED output (row 0)
    a_raw: UnsafePointer[Float32, MutAnyOrigin],   # [K] fp32 raw activation (decode row)
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],   # [Npad, K/2] packed E2M1 (weight)
    b_sc: UnsafePointer[UInt8, MutAnyOrigin],      # [Npad, K/16] e4m3 scales (weight)
    act_global: UnsafePointer[Float32, MutAnyOrigin],  # [0] = per-token gscale
    weight_global: Float32,
    N_arg: Int32, K_arg: Int32,                    # N%(8*W)==0, K%(TK*64)==0; M==1 (grid.y==1)
):
    # Tile config is comptime so shared memory can be laid out statically. Defaults
    # reproduce the alias version exactly; NOMOS_FP4_QA_CFG selects a swept arm.
    # Staging is prefetch only — the k-loop order is identical for every (W,TK,NS),
    # so every arm must stay BYTE-EXACT. A parity break here is a bug, not a tradeoff.
    comptime APK = 16 * TK * 32
    comptime ASC = 16 * TK * 4
    comptime OFF_BPK = APK
    comptime OFF_ASC = APK + 8 * W * TK * 32
    comptime OFF_BSC = OFF_ASC + ASC
    comptime STAGE_B = OFF_BSC + 8 * W * TK * 4
    comptime QA_OFF_ARAW = STAGE_B                 # raw fp32 A appended at the stage tail
    comptime QA_STAGE_B = STAGE_B + TK * 64 * 4
    comptime BLK_THREADS = 32 * W
    comptime QA_BLOCKS = TK * 4                    # 16-el e4m3 blocks per stage
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var N = Int(N_arg)
    var K = Int(K_arg)
    var KB = K // 2
    var NB = K // 16
    var tid = Int(thread_idx.x)      # 0..127
    var w = tid >> 5
    var lane = tid & 31
    var gid = lane >> 2
    var q = lane & 3
    var ncol_blk = Int(block_idx.x) * (8 * W)
    var ncol = ncol_blk + w * 8
    var a_raw8 = a_raw.bitcast[UInt8]()

    var sm = stack_allocation[
        NS * QA_STAGE_B, UInt8,
        address_space = AddressSpace.SHARED, alignment=16,
    ]()

    # Zero the A-packed + A-scale regions of ALL stages ONCE. The encode below only
    # ever writes row 0; rows 1..15 keep these zeros == quant_act_nvfp4_kernel's
    # pad-row bytes (packed=0, bs=0 -> the MMA contributes exact 0). The first
    # post-wait barrier makes them visible to every thread before any MMA read.
    comptime for s in range(NS):
        var zi = tid
        while zi < APK // 4:
            (sm + s * QA_STAGE_B + zi * 4).bitcast[UInt32]()[0] = UInt32(0)
            zi += BLK_THREADS
        var zj = tid
        while zj < ASC // 4:
            (sm + s * QA_STAGE_B + OFF_ASC + zj * 4).bitcast[UInt32]()[0] = UInt32(0)
            zj += BLK_THREADS

    var gscale = act_global[0]       # per-token global (gpu_act_amax_gscale)

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    var NT = K // (TK * 64)
    var npre = NS - 1
    if npre > NT:
        npre = NT
    for s in range(npre):
        _stage_ktile_qa[W, TK](sm + s * QA_STAGE_B, a_raw8, b_bytes, b_sc,
                        tid, ncol_blk, s * (TK * 64), KB, NB)
        async_copy_commit_group()

    for kt in range(NT):
        var pf = kt + NS - 1
        if pf < NT:
            _stage_ktile_qa[W, TK](
                sm + (pf % NS) * QA_STAGE_B, a_raw8, b_bytes, b_sc,
                tid, ncol_blk, pf * (TK * 64), KB, NB,
            )
            async_copy_commit_group()
            async_copy_wait_group(NS - 1)   # stage kt landed (immediate count)
        else:
            async_copy_wait_group(0)                 # drain (same rationale as R1)
        barrier()

        var st = sm + (kt % NS) * QA_STAGE_B
        # ── in-place A encode: threads 0..15 quantize ONE 16-el block each of the raw
        # fp32 stage. Same helpers + loop order + denominators as
        # fp4_act.quant_act_nvfp4_kernel -> byte-identical A-packed/A-scale smem.
        # Safe vs the pipeline: this stage's raw A landed (wait+barrier above); its
        # A regions were last read by the MMA 3 iterations ago (>= 2 barriers back);
        # in-flight cp.async targets OTHER stage buffers / disjoint regions. ──
        if tid < QA_BLOCKS:
            var araw = (st + QA_OFF_ARAW).bitcast[Float32]()
            var boff = tid * 16
            var bamax = Float32(0.0)
            for e in range(16):
                var v = abs(araw[boff + e])
                if v > bamax:
                    bamax = v
            var bs_byte = _f32_to_e4m3_byte(bamax / (E2M1_MAX * gscale))
            var cc = tid >> 2            # k64 chunk within the stage
            var sb = tid & 3             # 16-el block within the chunk
            (st + OFF_ASC + cc * (16 * 4) + sb)[0] = UInt8(bs_byte)    # row 0 scale
            var denom = gscale * e4m3_decode(bs_byte)
            if denom == Float32(0.0):
                denom = Float32(1.0)
            for j in range(8):
                var n0 = _e2m1_nibble(araw[boff + 2 * j] / denom)
                var n1 = _e2m1_nibble(araw[boff + 2 * j + 1] / denom)
                (st + cc * (16 * 32) + sb * 8 + j)[0] = UInt8((n1 << 4) | n0)  # row 0
        barrier()    # encoded A visible before the MMA reads it

        comptime for cc in range(TK):
            var a0 = (st + cc * (16 * 32) + gid * 32 + q * 4).bitcast[UInt32]()[0]
            var a1 = (st + cc * (16 * 32) + (gid + 8) * 32 + q * 4).bitcast[UInt32]()[0]
            var a2 = (st + cc * (16 * 32) + gid * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var a3 = (st + cc * (16 * 32) + (gid + 8) * 32 + 16 + q * 4).bitcast[UInt32]()[0]
            var b0 = (st + OFF_BPK + cc * (8 * W * 32) + (w * 8 + gid) * 32 + q * 4)
                .bitcast[UInt32]()[0]
            var b1 = (st + OFF_BPK + cc * (8 * W * 32) + (w * 8 + gid) * 32 + 16 + q * 4)
                .bitcast[UInt32]()[0]

            var sca = UInt32(0)
            if q == 0:
                sca = (st + OFF_ASC + cc * (16 * 4) + gid * 4).bitcast[UInt32]()[0]
            elif q == 1:
                sca = (st + OFF_ASC + cc * (16 * 4) + (gid + 8) * 4).bitcast[UInt32]()[0]
            var scb = UInt32(0)
            if q == 0:
                scb = (st + OFF_BSC + cc * (8 * W * 4) + (w * 8 + gid) * 4)
                    .bitcast[UInt32]()[0]

            # Same guard rationale as fp4_gemm_kernel (sm_120/121-only warp MMA).
            comptime if _is_sm_120x_or_newer():
                var r = inlined_assembly[
                    (
                        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                        " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                        " {$15}, {0, 0};"
                    ),
                    _RegisterPackType[Float32, Float32, Float32, Float32],
                    constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
                ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
                c0 = r[0]
                c1 = r[1]
                c2 = r[2]
                c3 = r[3]
        barrier()    # stage fully consumed before it is overwritten

    # ── fused postscale epilogue (M=1): only row 0 lives (rows 1..15 = zero pads,
    # their c is discarded — no c_pad round-trip). Lanes gid==0 hold row 0's outputs
    # in c0/c1; multiply order matches postscale_w4a4_kernel
    # (c * act_global[m] * weight_global) -> bit-exact. ──
    if gid == 0:
        d_out[ncol + 2 * q] = c0 * gscale * weight_global
        d_out[ncol + 2 * q + 1] = c1 * gscale * weight_global


def _qa_fuse_env_off() -> Bool:
    """NOMOS_FP4_QAFUSE=0 -> disable the R2 Phase-B fused act-quant GEMM (A/B switch);
    default ON. Same getenv pattern as _fp4_old_kernel_forced."""
    var name = String("NOMOS_FP4_QAFUSE")
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


def qa_fuse_route(M: Int, K: Int, N: Int) -> Bool:
    """R2 Phase-B dispatch predicate (fast quant + fused-postscale GEMM):
    decode-shaped (M==1) + R1-smem-tileable shape + kill-switch. NOMOS_FP4_OLD=1
    also disables (the ps kernel builds on the smem kernel). True for every
    gemma-4 decode projection incl the lm-head.

    Divisibility is checked against the SELECTED arm's W/TK, not the file-scope
    aliases: the kernel that will actually run is the one whose layout has to
    accept this shape, and a route/kernel disagreement is an OOB, not a slowdown."""
    return (M == 1 and N % (8 * _qa_arm_w()) == 0
            and K % (_qa_arm_tk() * 64) == 0
            and not _qa_fuse_env_off() and not _fp4_old_kernel_forced())


def qa_fuse_route_m(M: Int, K: Int, N: Int) -> Bool:
    """Spec-verify M>1 variant of qa_fuse_route (parallel M-row fast quant + fused-
    postscale M-row GEMM). Fires for 1<M<=16 (a single 16-row MMA m-tile covers the
    k+1<=8 verify rows), R1-smem-tileable shape, same kill-switches as qa_fuse_route
    (NOMOS_FP4_QAFUSE=0 / NOMOS_FP4_OLD=1 -> OLD one-warp-quant + c_pad chain, the
    A/B baseline for the bit-exact seam gate)."""
    return (M > 1 and M <= 16 and N % (8 * FP4_SMEM_WARPS) == 0
            and K % (FP4_TK_STAGE * 64) == 0
            and not _qa_fuse_env_off() and not _fp4_old_kernel_forced())


def gpu_fp4_gemm_qa(
    ctx: DeviceContext,
    d_out: UInt64, a_raw: UInt64, b_bytes: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    N: Int, K: Int,
) raises:
    """Launch the fused act-quant W4A4 GEMM (decode, M=1): raw fp32 activation in,
    postscaled fp32 row out. act_global[0] must already hold the per-token gscale
    (lib/fp4_act.gpu_act_amax_gscale — one launch per DISTINCT activation). Callers
    gate on qa_fuse_route(); shape must satisfy N%(8*W)==0, K%(TK*64)==0."""
    if Float64(ctx.default_device_info.compute) < 12.0:
        raise Error(
            "NVFP4 W4A4 QA-fused GEMM requires sm_120/sm_121 (compute>=12.0);"
            " run W4A16 (NOMOS_W4A4=0) on datacenter Blackwell."
        )
    var arm = _qa_cfg_arm()
    if arm == 1:
        _launch_qa[4, 2, 3](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    elif arm == 2:
        _launch_qa[8, 2, 3](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    elif arm == 3:
        _launch_qa[16, 2, 3](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    elif arm == 4:
        _launch_qa[8, 2, 4](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    elif arm == 5:
        _launch_qa[4, 2, 6](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    elif arm == 6:
        _launch_qa[4, 4, 4](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    elif arm == 7:
        # W=2 UNDERFILL ARM (2026-08-04, Codex). At the shipped W=4 the decode projections
        # with N=5376 (o_proj, down) launch N/(8*4) = 168 blocks onto a 188-SM PRO 6000 —
        # ONE PARTIAL WAVE, >=20 SMs idle, and those two categories are 6.4 ms/tok = 27% of
        # the token. W=2 doubles blocks to 336 (1.79 waves) at half the warps/block. TK and NS
        # are pinned to the shipped 4/3 so this arm isolates W ALONE.
        # NOT a predicted win: halving the block halves A/weight reuse per block, so a null or
        # a regression is a real outcome. Standalone timing must be confirmed by an E2E A/B/A —
        # an isolated win reversing end-to-end has bitten us before (TK6).
        _launch_qa[2, 4, 3](ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)
    else:
        _launch_qa[FP4_SMEM_WARPS, FP4_TK_STAGE, FP4_NSTAGES](
            ctx, d_out, a_raw, b_bytes, b_sc, act_global, weight_global, N, K)


def _launch_qa[W: Int, TK: Int, NS: Int](
    ctx: DeviceContext,
    d_out: UInt64, a_raw: UInt64, b_bytes: UInt64, b_sc: UInt64,
    act_global: UInt64, weight_global: Float32,
    N: Int, K: Int,
) raises:
    """Enqueue one QA-GEMM instantiation at comptime (W, TK, NS). Grid/block derive from
    W so they cannot drift from the kernel's own layout."""
    var kern = ctx.compile_function[fp4_gemm_kernel_smem_qa[W, TK, NS]]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(d_out)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(a_raw)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc)),
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(act_global)),
        weight_global, Int32(N), Int32(K),
        grid_dim=(N // (8 * W), 1),
        block_dim=32 * W,
    )


# ── Decode-GEMM tile sweep (NOMOS_FP4_QA_CFG) ────────────────────────────────
# fp4_gemm is ~71% of the decode token and runs at ~430 GB/s against a MEASURED 621
# GB/s achievable on sm_120 (RTX PRO 4000). Prime suspect: the shipped config puts
# only 25% occupancy on an SM (TK=4 makes a stage 7936 B, so NS=3 costs 23.2 KB/block
# and shared caps it at 4 blocks x 128 threads). Halving TK buys depth AND occupancy.
#
# TK=8 is ILLEGAL — K%(TK*64)==0 fails for K=D=5376 (5376 % 512 == 256), which would
# silently de-route q/k/v/gate/up to a slower path and read as a config result.
# Legal TK = {2,4}; legal W = {4,8,16} (every Gemma-4 N is a multiple of 128).
#
#   arm  W   TK  NS   smem/blk   blocks/SM   threads/SM   occupancy
#    0   4   4   3     23.2 KB       4           512         25%   <- shipped default
#    1   4   2   3     11.6 KB       8          1024         50%
#    2   8   2   3     18.4 KB       5          1280         62%
#    3  16   2   3     31.9 KB       3          1536         75%
#    4   8   2   4     24.5 KB       4          1024         50%   (deeper pipeline)
#    5   4   2   6     23.2 KB       4           512         25%   (isolates depth: same
#                                                                   smem as arm 0, 2x deep)
#    6   4   4   4     31.0 KB       3           384         19%   (deeper + fat)
#    7   2   4   3     23.2 KB       4           256         12%   (UNDERFILL PROBE — see below)
#
# ARM 7 IS A MEASURED NULL. KEPT ON PURPOSE so nobody re-runs the experiment hoping.
# Hypothesis: at W=4 the N=5376 shapes (o_proj, down) launch 5376/(8*4) = 168 blocks onto the
# PRO 6000's 188 SMs — ONE PARTIAL WAVE, >=20 SMs idle — and those two categories are 27% of the
# decode token. W=2 doubles that to 336 blocks (1.79 waves) and fills the machine.
#
# Measured 2026-08-04, Vast RTX PRO 6000 Max-Q (188 SM), sha 84899ea, interleaved A/B/A/B/A:
#     W=4  41.05 / 41.31 / 41.11   (mean 41.157, spread 0.63%)
#     W=2         41.13 / 41.22    (mean 41.175)
#   => +0.04%, INSIDE the control's own run-to-run spread. Filling the idle SMs changed nothing.
#
# CONCLUSION: grid underfill is NOT binding for these shapes. The 168 resident blocks already
# saturate the limiting resource, so the idle SMs were never the cost. (This rules out an
# occupancy-by-blocks explanation ONLY — it does not by itself establish WHICH resource binds;
# bandwidth-bound and latency/issue-bound both predict this same null.)
# Correctness note: W=2 output is BYTE-IDENTICAL to W=4 (sha 317f9f81..., 292 B) — expected, since
# warps partition OUTPUT COLUMNS (8 each), so each dot product accumulates entirely within one warp
# and W cannot change the reduction order. A parity FAIL here would mean a real W-dependency bug.
#
# _qa_arm_w / _qa_arm_tk below MUST stay in step with the _launch_qa[...] dispatch in
# gpu_fp4_gemm_qa — they are the same table written twice (once as comptime params the
# kernel is built with, once as runtime values the route predicate checks against).


def _qa_cfg_arm() -> Int:
    """NOMOS_FP4_QA_CFG=<0..7> -> decode-GEMM tile arm. Unset or unrecognised returns 0
    (the shipped default), so a typo can never silently change what ships."""
    var name = String("NOMOS_FP4_QA_CFG")
    var nb = List[UInt8](capacity=name.byte_length() + 1)
    for i in range(name.byte_length()):
        nb.append(name.as_bytes()[i])
    nb.append(0)
    var raw = external_call[
        "getenv", Int64, UnsafePointer[UInt8, origin_of(nb)]
    ](nb.unsafe_ptr())
    if raw == 0:
        return 0
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw))
    var d = Int(p[0]) - 48                       # '0'..'7'
    if d < 0 or d > 7:
        return 0
    return d


def _qa_arm_w() -> Int:
    """Runtime W of the selected arm — qa_fuse_route MUST use this, not FP4_SMEM_WARPS.
    If the route predicate checked N%32 while the kernel was built for W=16 (needs
    N%128), a shape would pass the gate and hit a kernel that cannot address it."""
    var a = _qa_cfg_arm()
    if a == 2 or a == 4:
        return 8
    if a == 3:
        return 16
    if a == 7:
        return 2                   # W=2 underfill arm — route predicate needs N%16, not N%32
    return 4                       # arms 0, 1, 5, 6


def _qa_arm_tk() -> Int:
    """Runtime TK of the selected arm — same agreement requirement as _qa_arm_w."""
    var a = _qa_cfg_arm()
    if a == 0 or a == 6 or a == 7:
        return 4
    return 2                       # arms 1..5


def _fp4_old_kernel_forced() -> Bool:
    """NOMOS_FP4_OLD=1 -> force the pre-R1 single-warp kernel (A/B switch)."""
    var name = String("NOMOS_FP4_OLD")
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
    return p[0] == UInt8(49)     # '1'


def gpu_fp4_gemm(
    ctx: DeviceContext,
    c_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    M: Int, N: Int, K: Int,
) raises:
    """Launch the NVFP4 GEMM (device-address ABI matching the kernel's other _dev ops).
    M/N MUST already be padded to multiples of 16/8 and K a multiple of 64 (true for
    every Gemma-4 projection). This is the Step-3 (W4A4) integration entry; Step-1
    correctness is proven standalone in tools/fp4_gemm_test.mojo."""
    # Datacenter Blackwell (sm_100, B200/B300) has no warp-level mxf4nvf4.block_scale MMA
    # (tcgen05-only). The kernel's comptime guard makes it BUILD there but emit no MMA, so
    # dispatching W4A4 on sm_100 would silently produce zeros. Fail loudly instead — run
    # W4A16 (NOMOS_W4A4=0) on datacenter Blackwell until the tcgen05 path lands.
    if Float64(ctx.default_device_info.compute) < 12.0:
        raise Error(
            "NVFP4 W4A4 (mxf4nvf4 block-scale MMA) requires sm_120/sm_121 (compute>=12.0);"
            " this device is sm_100/datacenter Blackwell (tcgen05-only). Run W4A16 (NOMOS_W4A4=0)."
        )
    # R1 dispatch: cooperative smem-staged kernel when the shape allows
    # (N%32, K%256 — true for every gemma-4 projection); NOMOS_FP4_OLD=1 or an
    # off-shape falls back to the proven single-warp kernel.
    if N % (8 * FP4_SMEM_WARPS) == 0 and K % (FP4_TK_STAGE * 64) == 0 \
            and not _fp4_old_kernel_forced():
        var kern_s = ctx.compile_function[fp4_gemm_kernel_smem]()
        ctx.enqueue_function(
            kern_s,
            UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c_out)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_bytes)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_sc)),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc)),
            Int32(M), Int32(N), Int32(K),
            grid_dim=(N // (8 * FP4_SMEM_WARPS), M // 16),
            block_dim=32 * FP4_SMEM_WARPS,
        )
        return
    var kern = ctx.compile_function[fp4_gemm_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c_out)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_sc)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc)),
        Int32(M), Int32(N), Int32(K),
        grid_dim=(N // 8, M // 16), block_dim=32,
    )
