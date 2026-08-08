"""NVFP4 W4A4 GEMM for DATACENTER Blackwell (sm_100, B200/B300) — tcgen05 path.

The warp-level `mma.sync...kind::mxf4nvf4.block_scale` in lib/fp4_gemm.mojo is sm_120a/sm_121a
ONLY (consumer/workstation Blackwell). Datacenter Blackwell (sm_100) dropped it — block-scaled
NVFP4 is tcgen05-only there (async, accumulator + scales in tensor memory). This is the
standalone tcgen05 reimplementation, stdlib intrinsics only (no MAX import, no raw PTX).

OPERAND LOAD = TMA (matches MAX's block_scaled_matmul_kernel, the reference example): A/B are
loaded global->SMEM by the TMA engine with a swizzle descriptor, so the HARDWARE writes the
swizzled bytes — no hand-rolled swizzle XOR. Each MMA_K=64 (=32 packed bytes) K-slice is its own
SWIZZLE_32B tile (32-byte swizzle atom == the slice width), so the MMA descriptor reads offset 0
with no swizzle-offset math. The 8x32B FP4 k-atom (`_select_k_atom_bits[SWIZZLE_32B]` upcast to
4-bit) exactly matches one MMA_K slice.

Design + recipe: docs/TCGEN05_W4A4_SM100_DESIGN.md ; docs/tcgen05_w4a4_design_findings.json
Dispatch: gpu_matmul_nvfp4_w4a4_dev (lib/fp4_act.mojo) routes here when _is_sm_100x().

The TMADescriptor kernel params MUST carry `@__llvm_arg_metadata(<p>, `nvvm.grid_constant`)`
(below) — without it the descriptor has no valid const-mem address for the TMA tensormap
operand and every load faults with ILLEGAL_ADDRESS.

STATUS: BIT-EXACT on B200 — tools/fp4_gemm_sm100_test.mojo (all-ones) AND _test2.mojo
(varying A/B, per-value E2M1 dequant) both pass bad=0/maxerr=0. Remaining before go-live:
vary-SCALES test (SF-atom TMEM staging), then wire the sm_100 dispatch in
gpu_matmul_nvfp4_w4a4_dev. Until dispatched, the engine still runs W4A16 on sm_100.
"""
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim, barrier, warp_id, lane_id
from std.gpu.memory import AddressSpace, cp_async_bulk_tensor_shared_cluster_global
from std.gpu.sync import (
    mbarrier_init,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_try_wait_parity_shared,
)
from std.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
)
from std.memory import UnsafePointer, stack_allocation
from std.utils.index import Index, IndexList

from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAKind,
    UMMAInsDescriptor,
    MMASmemDescriptor,
    mma,
    mma_arrive,
)
from std.sys.info import _is_sm_120x_or_newer
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_cp,
    tcgen05_release_allocation_lock,
    tcgen05_load_wait,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 (prerequisite): scale-factor repack — our flat [MN, K/16] e4m3 block
# scales → the tcgen05 "SF-atom" layout. Pure PERMUTATION (no requantize; values
# are byte-identical ue4m3). The tcgen05 block-scale MMA reads scales from TMEM,
# staged from this SF-atom buffer. Index map (design doc): element (mn, kblk=k//16)
# → 5D [mn//128, kblk//4, mn%32, (mn%128)//32, kblk%4], dims [ceil(MN/128),
# ceil(K/64), 32, 4, 4]. Run once per weight (at load) + once per activation quant.
# ─────────────────────────────────────────────────────────────────────────────
def nvfp4_sf_scatter_kernel(
    src: UnsafePointer[UInt8, MutAnyOrigin],   # flat [MN, NB] e4m3 (NB = K/16)
    dst: UnsafePointer[UInt8, MutAnyOrigin],   # SF-atom [ceil(MN/128), ceil(NB/4), 32, 4, 4]
    MN_arg: Int32, NB_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var MN = Int(MN_arg)
    var NB = Int(NB_arg)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if gid >= MN * NB:
        return
    var mn = gid // NB
    var kblk = gid % NB
    var nk = (NB + 3) // 4                       # ceil(NB/4) = ceil(K/64)
    var d0 = mn // 128
    var d1 = kblk // 4
    var d2 = mn % 32
    var d3 = (mn % 128) // 32
    var d4 = kblk % 4
    var off = ((((d0 * nk + d1) * 32 + d2) * 4 + d3) * 4 + d4)
    dst[off] = src[mn * NB + kblk]


def gpu_nvfp4_sf_scatter(
    ctx: DeviceContext, src: UInt64, dst: UInt64, MN: Int, NB: Int,
) raises:
    """Permute flat [MN, NB] e4m3 block scales into the tcgen05 SF-atom layout.
    Caller sizes dst = ceil(MN/128)*ceil(NB/4)*32*4*4 bytes (zero-pad the M/K tails)."""
    var total = MN * NB
    var bs = 256
    var grid = (total + bs - 1) // bs
    var kern = ctx.compile_function[nvfp4_sf_scatter_kernel]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(src)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(dst)),
        Int32(MN), Int32(NB),
        grid_dim=grid, block_dim=bs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The tcgen05 block-scaled NVFP4 GEMM kernel (single-SM, cta_group=1). One CTA
# computes one 128(M) × MMA_N output tile. A/B come in via TMA, one SWIZZLE_32B
# tile per MMA_K=64 K-slice (32 bytes wide). One mma[1] per slice; accumulator +
# scales live in TMEM. Store map (4-warp × 16-row-half × 8×4 lane grid) is the
# all-ones-proven path.
# ─────────────────────────────────────────────────────────────────────────────
@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
def fp4_gemm_sm100_kernel[MMA_N: Int](
    c_out: UnsafePointer[Float32, MutAnyOrigin],   # [M, N] padded, row-major
    a_tma: TMADescriptor,                          # A [M, K/2] uint8, box [128, 32], SWIZZLE_32B
    b_tma: TMADescriptor,                          # B [N, K/2] uint8, box [MMA_N, 32], SWIZZLE_32B
    a_sf: UnsafePointer[UInt8, MutAnyOrigin],      # A scales, SF-atom layout (gpu_nvfp4_sf_scatter)
    b_sf: UnsafePointer[UInt8, MutAnyOrigin],      # B scales, SF-atom layout
    M_arg: Int32, N_arg: Int32, K_arg: Int32,                        # padded: M%128==0, N%MMA_N==0, K%64==0
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)
    comptime if not _is_sm_120x_or_newer():
        # tcgen05 block-scaled NVFP4 W4A4 exists ONLY on sm_100a/sm_101a (datacenter
        # Blackwell, B200/B300). On sm_120x+ (consumer/workstation Blackwell) this kernel
        # is never dispatched (host-side is_sm100 guard in gpu_matmul_nvfp4_w4a4_dev runs
        # the sm_120 mma.sync path instead), and tcgen05 is unavailable there, so the body
        # is comptime-elided and the kernel compiles as a no-op stub.
        comptime MMA_M = 128
        comptime MMA_K = 64               # mxf4nvf4 HW k-depth per issue (B200: 64 FP4 elems)
        comptime KSB = MMA_K // 2         # packed FP4 bytes per K-slice (=32 = SWIZZLE_32B atom width)
        comptime NSF = MMA_K // 16        # e4m3 scale blocks per K-slice (=4)

        var m_tile = Int(block_idx.y)     # which 128-row M tile
        var n_tile = Int(block_idx.x)     # which MMA_N-col N tile
        var tid = Int(thread_idx.x)

        # ---- SMEM: A/B FP4 slice tiles, SF tiles, mbarrier, tmem-addr slot ----------
        var a_smem = stack_allocation[MMA_M * KSB, Scalar[DType.uint8],
            address_space=AddressSpace.SHARED, alignment=1024]()
        var b_smem = stack_allocation[MMA_N * KSB, Scalar[DType.uint8],
            address_space=AddressSpace.SHARED, alignment=1024]()
        var sfa_smem = stack_allocation[MMA_M * NSF, Scalar[DType.uint8],
            address_space=AddressSpace.SHARED]()
        var sfb_smem = stack_allocation[MMA_N * NSF, Scalar[DType.uint8],
            address_space=AddressSpace.SHARED]()
        var load_mbar = stack_allocation[1, Scalar[DType.int64],
            address_space=AddressSpace.SHARED]()
        var mma_mbar = stack_allocation[1, Scalar[DType.int64],
            address_space=AddressSpace.SHARED]()
        var ptr_tmem_addr = stack_allocation[1, Scalar[DType.uint32],
            address_space=AddressSpace.SHARED]()

        if tid == 0:
            mbarrier_init(load_mbar, 1)
            mbarrier_init(mma_mbar, 1)

        # ---- TMEM alloc (full 512 cols; sub-range used) ----------------------------
        var elect_warp = warp_id() == 0
        if elect_warp:
            tcgen05_alloc[1](ptr_tmem_addr, 512)
        barrier()
        var tmem = ptr_tmem_addr[0]
        var c_tmem = tmem                              # accumulator: MMA_N cols at base
        var sfa_tmem = c_tmem + UInt32(MMA_N)          # A scales region
        var sfb_tmem = sfa_tmem + UInt32(4)            # B scales region (+4 cols/k-slice)

        # ---- MMA instruction descriptor (K implied by KIND_MXF4NVF4) ----------------
        var idesc = UMMAInsDescriptor[UMMAKind.KIND_MXF4NVF4].create[
            DType.float32, DType.uint8, DType.uint8, DType.float8_e4m3fn,
            Index[dtype = DType.uint32](MMA_M, MMA_N), transpose_b=True,
        ]()

        comptime a_tx = MMA_M * KSB                    # bytes per A-slice TMA load
        comptime b_tx = MMA_N * KSB                    # bytes per B-slice TMA load
        # Descriptors are @grid_constant params (see decorators) -> valid const-mem address
        # for the TMA tensormap operand. Take the pointer directly (matches stdlib test_tma).
        var a_desc_ptr = UnsafePointer(to=a_tma).bitcast[NoneType]()
        var b_desc_ptr = UnsafePointer(to=b_tma).bitcast[NoneType]()

        var lphase: UInt32 = 0
        var mphase: UInt32 = 0
        var n_slices = K // MMA_K
        comptime SF_ATOM_BYTES = 32 * 4 * 4   # one (d0,d1) SF-atom group = 128 rows x 4 kblk = 512B
        for it in range(n_slices):
            var k0b = it * KSB                         # byte col of this K-slice
            # ---- TMA load A/B K-slice (hardware swizzles) --------------------------
            if tid == 0:
                mbarrier_arrive_expect_tx_shared(load_mbar, Int32(a_tx + b_tx))
                cp_async_bulk_tensor_shared_cluster_global(
                    a_smem, a_desc_ptr, load_mbar, Index(k0b, m_tile * MMA_M)
                )
                cp_async_bulk_tensor_shared_cluster_global(
                    b_smem, b_desc_ptr, load_mbar, Index(k0b, n_tile * MMA_N)
                )
            # SF slice: each MMA_K=64 slice == exactly ONE SF-atom (d0=mn_tile, d1=it) group,
            # a CONTIGUOUS 512B block in the SF-atom buffer (gpu_nvfp4_sf_scatter order). Copy
            # it straight through -> tcgen05_cp -> TMEM. (Flat [MN,K/16] indexing was the bug:
            # a_sf is SF-atom-permuted, not flat, so per-row scales landed on wrong rows.)
            var a_sf_base = (m_tile * n_slices + it) * SF_ATOM_BYTES
            var b_sf_base = (n_tile * n_slices + it) * SF_ATOM_BYTES
            for r in range(tid, SF_ATOM_BYTES, Int(block_dim.x)):
                sfa_smem[r] = a_sf[a_sf_base + r]
                sfb_smem[r] = b_sf[b_sf_base + r]
            mbarrier_try_wait_parity_shared(load_mbar, Int32(lphase), Int32(10000000))
            lphase ^= 1
            barrier()

            if tid == 0:
                # scales SMEM->TMEM (one tcgen05_cp per 128-row group) ----------------
                var sfa_desc = MMASmemDescriptor.create[8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE](sfa_smem)
                var sfb_desc = MMASmemDescriptor.create[8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE](sfb_smem)
                tcgen05_cp[cta_group=Int32(1), datapaths=32, bits=128, multicast="warpx4"](sfa_tmem, sfa_desc)
                tcgen05_cp[cta_group=Int32(1), datapaths=32, bits=128, multicast="warpx4"](sfb_tmem, sfb_desc)
                # one block-scaled MMA over this K-slice (descriptors at offset 0) -----
                var a_desc = MMASmemDescriptor.create[8 * 32, 16, TensorMapSwizzle.SWIZZLE_32B](a_smem)
                var b_desc = MMASmemDescriptor.create[8 * 32, 16, TensorMapSwizzle.SWIZZLE_32B](b_smem)
                var c_scale: UInt32 = 0 if it == 0 else 1
                mma[1](
                    a_desc, b_desc, c_tmem, idesc,
                    sfa_tmem, sfb_tmem, c_scale=c_scale,
                )
                mma_arrive[1](mma_mbar)
            mbarrier_try_wait_parity_shared(mma_mbar, Int32(mphase), Int32(10000000))
            mphase ^= 1
            barrier()

        # ---- read fp32 accumulator out of TMEM (16-row datapath groups) -------------
        comptime LDW = MMA_N // 2          # width = (repeat*256*16)//1024 = MMA_N//2
        var cf_up = tcgen05_ld[
            datapaths=16, bits=256, repeat = MMA_N // 8,
            dtype = DType.float32, pack=False, width=LDW,
        ](c_tmem)
        var cf_lo = tcgen05_ld[
            datapaths=16, bits=256, repeat = MMA_N // 8,
            dtype = DType.float32, pack=False, width=LDW,
        ](c_tmem + (UInt32(16) << 16))
        tcgen05_load_wait()
        if elect_warp:
            tcgen05_release_allocation_lock[1]()
            tcgen05_dealloc[1](tmem, 512)

        # Store -> C[m_tile, n_tile]. Standard tcgen05 C layout: 4 warps (32 rows each)
        # × 16-row halves × an 8×4 lane grid (lane owns rows {lrow, lrow+8}, 2 cols/vec).
        var warp = tid // 32
        var lane = tid % 32
        var lrow = lane // 4
        var lcol = lane % 4
        comptime nvm = 16 // 8             # rows-per-lane per 16-row half (=2)
        comptime nvn = MMA_N // 8          # vec-cols-per-lane
        var rbase = m_tile * MMA_M + warp * 32
        var cbase = n_tile * MMA_N
        comptime for half in range(2):
            for m_vec in range(nvm):
                var row = rbase + half * 16 + lrow + 8 * m_vec
                for n_vec in range(nvn):
                    var col = cbase + 2 * (lcol + 4 * n_vec)
                    var iv = n_vec * nvm + m_vec
                    var v0 = cf_up[2 * iv] if half == 0 else cf_lo[2 * iv]
                    var v1 = cf_up[2 * iv + 1] if half == 0 else cf_lo[2 * iv + 1]
                    c_out[row * N + col] = v0
                    c_out[row * N + col + 1] = v1


def gpu_fp4_gemm_sm100(
    ctx: DeviceContext,
    c_out: UInt64, a_bytes: UInt64, b_bytes: UInt64,
    a_sc: UInt64, b_sc: UInt64,
    M: Int, N: Int, K: Int,
) raises:
    """sm_100 (datacenter Blackwell) NVFP4 W4A4 GEMM via tcgen05. Same device-address ABI as
    gpu_fp4_gemm (lib/fp4_gemm.mojo). a_sc/b_sc must ALREADY be in the SF-atom layout
    (gpu_nvfp4_sf_scatter). M→128, N→multiple of MMA_N, K→multiple of 64 (caller pads)."""
    comptime MMA_N = 128
    comptime KSB = 32                 # MMA_K//2 packed bytes per K-slice
    var KB = K // 2                   # packed FP4 bytes per row

    # Wrap raw device addrs as non-owning DeviceBuffers so TMA descriptors can be built.
    var a_buf = DeviceBuffer[DType.uint8](
        ctx, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_bytes)),
        M * KB, owning=False,
    )
    var b_buf = DeviceBuffer[DType.uint8](
        ctx, UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_bytes)),
        N * KB, owning=False,
    )
    # A: [M, KB] uint8, box [128, 32] (one MMA_K=64 slice), 32B swizzle.
    var a_tma = create_tma_descriptor[DType.uint8, 2, TensorMapSwizzle.SWIZZLE_32B](
        a_buf, Index(M, KB), Index(KB, 1), Index(128, KSB),
    )
    var b_tma = create_tma_descriptor[DType.uint8, 2, TensorMapSwizzle.SWIZZLE_32B](
        b_buf, Index(N, KB), Index(KB, 1), Index(MMA_N, KSB),
    )

    var kern = ctx.compile_function[fp4_gemm_sm100_kernel[MMA_N]]()
    ctx.enqueue_function(
        kern,
        UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c_out)),
        a_tma, b_tma,
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(a_sc)),
        UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(b_sc)),
        Int32(M), Int32(N), Int32(K),
        grid_dim=(N // MMA_N, M // 128), block_dim=128,
    )
