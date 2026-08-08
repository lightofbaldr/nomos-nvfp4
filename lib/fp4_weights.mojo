"""NVFP4 weight dequant (FP4 Step 2, W4-weights) — NVFP4 → bf16.

Parallels lib/q4_weights.mojo's gpu_dequant_q4: reconstruct a bf16 weight from the
NVFP4 blob (per-tensor fp32 global × per-16-element e4m3 block scale × E2M1 nibble) so
the existing bf16 cuBLAS GEMM runs unchanged — W4A16: NVFP4 weight, bf16 activation.
This isolates weight-quant from activation-quant (Step 3 / W4A4 routes both through the
NVFP4 MMA in lib/fp4_gemm.mojo).

Blob layout (single device buffer, matches tools/quantize_nvfp4.py):
    [ e4m3 block scales : nb bytes ][ E2M1 packed : n/2 bytes ]
with the fp32 global scale passed separately. n = N*K total elements, nb = n/16.
Flat group indexing (block = i//16) is valid because K%16==0 for every Gemma-4
projection, so a 16-element block never straddles two rows.

e4m3 decode here is the arithmetic twin of the exact LUT in quantize_nvfp4.py / the
hardware ue4m3 the NVFP4 MMA uses. E2M1 magnitudes {0,.5,1,1.5,2,3,4,6}.
Requires Mojo ≥1.0.0b3 + --target-accelerator sm_121a.
"""
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer
from std.collections import List
from lib.cuda import cuda_malloc, cuda_upload_u8
from lib.io import read_nvfp4_bytes
from lib.engine_init import _env_flag_cached, ENV_NVFP4_FUSED

alias FP4_GROUP = 16
alias NVFP4_WARPS_PER_BLOCK = 4   # warps packed per CUDA block in the decode GEMV (occupancy)
alias E2M1_MAG = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# ue4m3 normal magnitude = (8+mant) * 2^(exp-10); this LUT is 2^(exp-10) for exp 0..15,
# so the hot path is one multiply (no branch on the exponent sign, no divide).
alias E4M3_P2 = SIMD[DType.float32, 16](
    0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125,
    0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
)


def _as_u8_ptr(addr: UInt64) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_bf16_ptr(addr: UInt64) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def e4m3_decode(b: Int) -> Float32:
    """ue4m3 byte → positive fp32 (bias 7, no inf). Matches the LUT in quantize_nvfp4.py.
    Branchless-on-exponent + no divide: normal = (8+mant)*2^(exp-10) via the E4M3_P2 LUT;
    only exp==0 (subnormal, rare for block scales) keeps a cheap predicated path."""
    var exp = (b >> 3) & 0xF
    var mant = b & 0x7
    if exp == 0:
        return Float32(mant) * Float32(0.001953125)   # subnormal: mant * 2^-9
    return Float32(8 + mant) * E4M3_P2[exp]           # (8+mant) * 2^(exp-10) == (1.mant) * 2^(exp-7)


def e2m1_mag(v: Int) -> Float32:
    """E2M1 magnitude (3 low bits) → {0,.5,1,1.5,2,3,4,6}, arithmetic — no SIMD-LUT
    gather (which compiles to a slow select-chain in the GEMV hot loop). exp=(v>>1),
    mant=(v&1); normal = (2+mant)*2^(exp-2), subnormal (exp==0) = mant*0.5."""
    var exp = (v >> 1) & 3
    var mant = v & 1
    if exp == 0:
        return Float32(mant) * Float32(0.5)
    return Float32(2 + mant) * Float32(1 << exp) * Float32(0.25)   # (2+mant)*2^(exp-2)


def dequant_nvfp4_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [e4m3 scales : nb][E2M1 packed : n/2]
    out_bf16: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    global_scale: Float32,
    n_arg: Int32,                                       # total elements (N*K)
    nb_arg: Int32,                                      # scale blocks (n/16)
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var n = Int(n_arg)
    var nb = Int(nb_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        var block = i // FP4_GROUP
        var bs_byte = Int(blob[block])                       # e4m3 scale (scales at blob start)
        var byte = Int(blob[nb + (i // 2)])                  # packed region starts at byte nb
        var nib = (byte & 0x0F) if (i & 1) == 0 else ((byte >> 4) & 0x0F)
        var sign = Float32(-1.0) if (nib & 8) != 0 else Float32(1.0)
        var mag = E2M1_MAG[nib & 7]
        var w = SIMD[DType.float32, 1](global_scale * e4m3_decode(bs_byte) * sign * mag)
        out_bf16[i] = w.cast[DType.bfloat16]()[0]


def gpu_dequant_nvfp4(
    ctx: DeviceContext, blob_ptr: UInt64, out_bf16_ptr: UInt64,
    global_scale: Float32, n: Int, nb: Int,
) raises:
    """Dequant an NVFP4 weight blob into a BF16 device buffer (n elements, 2 bytes each).
    The bf16 scratch then feeds the existing cuBLAS bf16 GEMM (W4A16). Parallels
    gpu_dequant_q4."""
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[dequant_nvfp4_kernel]()
    ctx.enqueue_function(
        k, _as_u8_ptr(blob_ptr), _as_bf16_ptr(out_bf16_ptr), global_scale, Int32(n), Int32(nb),
        grid_dim=blocks, block_dim=threads,
    )


# ─────────────────────────────────────────────────────────────────────────
# Load + matmul (W4A16-NVFP4) — mirrors lib/q4_weights.mojo's load_to_gpu_q4 /
# gpu_matmul_q4_dev. The NVFP4 weight is dequant'd to a bf16 scratch, then the
# EXISTING cuBLAS bf16 GEMM runs unchanged (activation stays high precision).
# ─────────────────────────────────────────────────────────────────────────

def load_to_gpu_nvfp4(
    path: String, mut gscale: List[Float32], mut input_gscale: List[Float32]
) -> UInt64:
    """Load a .nvfp4 weight onto one GPU buffer [e4m3 scales][E2M1 packed]; appends the
    per-tensor fp32 global scale to `gscale` (kept parallel to the weight list). Returns
    the blob ptr (0 on missing/empty — caller skips that layer)."""
    var hdr = List[Int]()             # [n, nb]
    var bytes = read_nvfp4_bytes(path, hdr, gscale, input_gscale)
    if len(bytes) == 0 or len(hdr) < 2:
        print("[WARN] Empty or missing nvfp4:", path)
        gscale.append(Float32(1.0))   # keep gscale aligned with the weight slot
        input_gscale.append(Float32(0.0))
        return UInt64(0)
    var ptr = cuda_malloc(len(bytes))
    cuda_upload_u8(ptr, bytes)
    return ptr


def gpu_matmul_nvfp4_dev(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_nvfp4: UInt64,                # [e4m3 scales][E2M1 packed] for weight [N,K]
    global_scale: Float32,
    n_w: Int,                         # N*K (element count of the weight)
    d_weight_bf16_scratch: UInt64,    # >= N*K*2 bytes (reused across GEMMs)
    d_scratch_in_bf16: UInt64,
    K: Int,
    N: Int,
) raises:
    """Decode (M=1) W4A16-NVFP4: dequant the NVFP4 weight -> bf16 scratch, then the
    identical cuBLAS BF16 GEMM. out[N] = W[N,K] @ in[K]. Mirror of gpu_matmul_q4_dev."""
    from lib.cublas import cublas_gemm_ex_bf16_nt
    from lib.cast_gpu_mojo import gpu_fp32_to_bf16_mojo
    gpu_dequant_nvfp4(ctx, d_w_nvfp4, d_weight_bf16_scratch, global_scale, n_w, n_w // 16)
    gpu_fp32_to_bf16_mojo(ctx, d_in_fp32, d_scratch_in_bf16, K)
    cublas_gemm_ex_bf16_nt(handle, d_weight_bf16_scratch, d_scratch_in_bf16, d_out_fp32, 1, K, N)


def gpu_matmul_nvfp4_dev_batched(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_nvfp4: UInt64,
    global_scale: Float32,
    n_w: Int,
    d_weight_bf16_scratch: UInt64,
    d_scratch_in_bf16: UInt64,
    S: Int,
    K: Int,
    N: Int,
) raises:
    """Prefill (M=S) W4A16-NVFP4: dequant the weight ONCE -> bf16 scratch, then the
    batched cuBLAS GEMM reuses it across all S tokens. Mirror of gpu_matmul_q4_dev_batched."""
    from lib.cublas import gpu_matmul_bf16_dev_batched
    gpu_dequant_nvfp4(ctx, d_w_nvfp4, d_weight_bf16_scratch, global_scale, n_w, n_w // 16)
    gpu_matmul_bf16_dev_batched(
        ctx, handle, d_out_fp32, d_in_fp32, d_weight_bf16_scratch, d_scratch_in_bf16, S, K, N
    )


# ─────────────────────────────────────────────────────────────────────────
# M2 FUSED NVFP4 dequant-GEMV (decode, M=1) — the NVFP4 analog of
# q4_weights.q4_gemv_kernel and of llama.cpp's gguf GEMV. Reads the NVFP4 blob
# DIRECTLY on-GPU (E2M1 nibble + per-16 e4m3 block scale + per-tensor global),
# dequant inline, multiply-accumulate against the fp32 activation, warp-reduce.
# No bf16 materialization, no cuBLAS — kills the per-token whole-weight dequant
# that made W4A16 slow (~0.65 tok/s). Activation stays fp32 (W4A16), so REASONING
# is preserved (unlike W4A4). The dequant math is the exact twin of
# dequant_nvfp4_kernel above, so greedy output matches the M1 path.
# ─────────────────────────────────────────────────────────────────────────

def nvfp4_gemv_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [e4m3 scales : nb][E2M1 packed : n/2]
    inp: UnsafePointer[Float32, MutAnyOrigin],    # activation, K elems, fp32
    outp: UnsafePointer[Float32, MutAnyOrigin],   # output, N elems, fp32
    global_scale: Float32,
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,                                       # scale blocks = N*K/16
    row_base_arg: Int32,                                 # first output row in this launch
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var row_base = Int(row_base_arg)
    # NVFP4_WARPS_PER_BLOCK warps per block (not 1) so the SM's 32-resident-block
    # limit no longer caps occupancy at 50%; each warp still owns one output row n.
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var n = row_base + Int(block_idx.x) * NVFP4_WARPS_PER_BLOCK + warp_id
    if n >= N:
        return
    var lane = Int(thread_idx.x) % WARP_SIZE      # 0..WARP_SIZE-1 within this warp
    # Lane processes whole 16-weight scale-blocks, strided by warp. The e4m3 block
    # scale is decoded ONCE per block (not per element) and the per-element work is
    # arithmetic-only (no SIMD-LUT gathers). Weight bytes per lane are contiguous.
    var row_block0 = (n * K) // FP4_GROUP      # this row's first scale block
    var packed_base = nb + (n * K) // 2        # this row's first packed-nibble byte
    var nbk = K // FP4_GROUP                    # 16-weight blocks per row (K % 16 == 0)

    var local = Scalar[DType.float32](0.0)
    var b = lane
    while b < nbk:
        var eff = global_scale * e4m3_decode(Int(blob[row_block0 + b]))   # ONCE / 16 weights
        var pbase = packed_base + b * (FP4_GROUP // 2)                     # 16 nibbles = 8 bytes
        var ibase = b * FP4_GROUP
        # ONE aligned 8-byte load of this block's 16 packed nibbles. pbase is 8-aligned
        # for every matmul (nb and (n*K)//2 are both multiples of 8), so the UInt64 read
        # is valid + makes the warp's weight bytes contiguous (coalesced) instead of the
        # stride-8 byte reads the per-byte loop produced. Nibble t = bits [4t,4t+3] (LE).
        var pk = (blob + pbase).bitcast[UInt64]()[0]
        for t in range(FP4_GROUP):
            var nib = Int((pk >> UInt64(4 * t)) & 0xF)
            var sign = Float32(-1.0) if (nib & 8) != 0 else Float32(1.0)
            local += eff * sign * e2m1_mag(nib & 7) * inp[ibase + t]
        b += WARP_SIZE

    # warp-shuffle butterfly reduce — every lane ends with the row's dot product
    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


def _nvfp4_use_fused() -> Bool:
    """NOMOS_NVFP4_FUSED toggle (default M2 fused GEMV for NVFP4 decode). Set =0 to fall back
    to the M1 dequant->bf16->cuBLAS path (for the M1-vs-M2 A/B in one binary)."""
    # Cached in the C shim (read once/process, not per-GEMV). Default ON; only "0" disables.
    return _env_flag_cached(ENV_NVFP4_FUSED) != 0


def gpu_matmul_nvfp4_fused_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,    # output N fp32 elems
    d_in_fp32: UInt64,     # input K fp32 elems (on GPU)
    d_w_nvfp4: UInt64,     # NVFP4 blob [e4m3 scales][E2M1 packed] for weight [N,K]
    global_scale: Float32,
    K: Int,
    N: Int,
) raises:
    """M2 fused: out[N] = dequant_nvfp4(W[N,K]) @ in[K], reading the NVFP4 weight directly —
    no bf16 scratch, no cuBLAS. One warp per output row. Activation fp32 (W4A16) -> reasoning
    preserved; fast like the Q4_0 M2 GEMV. Greedy output matches the M1 path."""
    var nb = (K * N) // FP4_GROUP
    var kern = ctx.compile_function[nvfp4_gemv_kernel]()
    # N=VOCAB=262144 produced exactly 65536 blocks at WPB=4. That boundary is
    # illegal-addressing on the PRO 6000 sm_120a path while ordinary projection
    # sizes pass. Keep grid.x in the already-proven <=8192 range and cover the
    # vocabulary with row-offset launches. The row math/output is unchanged.
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
