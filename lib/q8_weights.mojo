"""q8_weights.mojo — Q8_0 (group-32 symmetric INT8) WEIGHT path for the MTP drafter.

The MTP assistant (gemma-4-31B-it-assistant) ships Q8_0; we keep it Q8 (the law-allowed
drafter concession — naive-Q4 on the 0.5B head costs ~5pt, Q8 is nearly free). The drafter
forward is M=1 per draft step (the K=4 drafts are sequential — each step consumes the prev
token), so its matmuls are GEMVs, exactly like the main decode. This file is the Q8 twin of
q4_weights.mojo's M2 fused GEMV: read the int8 weight once on-GPU, dequant in the tile load,
no bf16 materialization. (The BATCHED int8-MMA verify path is separate — that's the
MMQ weight-once tiled GEMM, built with the verify seam.)

GPU blob layout (host strips the 16-byte .q8 header — same header as .q4): contiguous
  [ nb * float16 block-scales ][ n * int8 values ]
Element i: block = i//32. The int8 value lives at byte offset nb*2 + i (1 byte/elem, NOT
packed — unlike Q4's nibbles). w = int8[i] * scale[block].
"""
from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer, bitcast
from lib.cuda import cuda_malloc, cuda_upload_u8
from lib.io import read_q4_bytes   # identical 16-byte [int64 n][int64 nb] header

alias GROUP = 32


def _as_u8_ptr(addr: UInt64) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def load_to_gpu_q8(path: String) -> UInt64:
    """Load a .q8 file (header int64 n + int64 nb, then fp16 scales, then int8 values)
    onto a single GPU buffer holding [scales][int8]. Returns the blob ptr (0 on
    missing/empty — caller skips). Header is identical to .q4 so the host reader is shared."""
    var hdr = List[Int]()          # [n, nb]
    var bytes = read_q4_bytes(path, hdr)
    if len(bytes) == 0 or len(hdr) < 2:
        print("[WARN] Empty or missing q8:", path)
        return UInt64(0)
    var ptr = cuda_malloc(len(bytes))
    cuda_upload_u8(ptr, bytes)
    return ptr


# ─────────────────────────────────────────────────────────────────────────
# Fused INT8xFP32 dequant-GEMV (the drafter decode path). One WARP per output
# row n: the 32 lanes stride over the row's K elements (coalesced — a row's int8
# bytes are contiguous, K % 32 == 0 so every row starts on a group boundary),
# each lane accumulates a partial dot in FP32, then a warp-shuffle butterfly
# reduces the 32 partials. Input is read at FP32 (we quantize the WEIGHT). Mirror
# of q4_gemv_kernel — the only delta is the int8 (1 byte) vs Q4 (packed nibble) load.
# ─────────────────────────────────────────────────────────────────────────
def q8_gemv_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [nb fp16 scales][n int8 values]
    inp: UnsafePointer[Float32, MutAnyOrigin],    # input activation, K elems, FP32
    outp: UnsafePointer[Float32, MutAnyOrigin],   # output, N elems, FP32
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,                                       # (N*K + 31)//32
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var n = Int(block_idx.x)          # one block == one warp == one output row
    if n >= N:
        return
    var lane = Int(thread_idx.x)      # 0..31
    var row_base = n * K              # flat element index of row start (divisible by 32)

    var local = Scalar[DType.float32](0.0)
    var kk = lane
    while kk < K:
        var i = row_base + kk
        var block = i // GROUP
        var sbits = blob.bitcast[UInt16]()[block]
        var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
        var sval = blob.bitcast[Int8]()[nb * 2 + i]   # signed int8 weight value
        var w = Float32(Int(sval)) * scale[0]
        local += w * inp[kk]
        kk += WARP_SIZE

    # warp-shuffle butterfly reduce (every lane ends with the row's dot product)
    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


def q8_gemv_rows_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [nb fp16 scales][n int8 values]
    inp: UnsafePointer[Float32, MutAnyOrigin],    # M activation rows, K elems each, FP32
    outp: UnsafePointer[Float32, MutAnyOrigin],   # M output rows, N elems each, FP32
    K_arg: Int32,
    N_arg: Int32,
    nb_arg: Int32,
    M_arg: Int32,
):
    """Batched twin of q8_gemv_kernel: grid (N, M), one warp per (output row n,
    activation row m). Weight is read once per warp from GPU memory; the M
    activation rows are small (K fp32) and stay L2-resident across the N warps.
    Same fp32 accumulate order per output as the M=1 kernel."""
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var K = Int(K_arg)
    var N = Int(N_arg)
    var nb = Int(nb_arg)
    var M = Int(M_arg)
    var n = Int(block_idx.x)
    var m = Int(block_idx.y)
    if n >= N or m >= M:
        return
    var lane = Int(thread_idx.x)
    var row_base = n * K
    var in_base = m * K

    var local = Scalar[DType.float32](0.0)
    var kk = lane
    while kk < K:
        var i = row_base + kk
        var block = i // GROUP
        var sbits = blob.bitcast[UInt16]()[block]
        var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
        var sval = blob.bitcast[Int8]()[nb * 2 + i]
        var w = Float32(Int(sval)) * scale[0]
        local += w * inp[in_base + kk]
        kk += WARP_SIZE

    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[m * N + n] = Float32(local)


def gpu_matmul_q8_rows_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,    # M*N FP32 elems
    d_in_fp32: UInt64,     # M*K FP32 elems (already on GPU)
    d_w_q8: UInt64,        # Q8 blob (scales + int8), header stripped
    K: Int,
    N: Int,
    M: Int,
) raises:
    """out[M,N] = in[M,K] @ dequant_q8(W[N,K])^T, one warp per output element row.
    The DFlash Q8-drafter path (the law-allowed drafter concession applied to
    DFlash — MTP/EAGLE3 precedent in this file's header)."""
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q8_gemv_rows_kernel]()
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q8), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb), Int32(M),
        grid_dim=(N, M), block_dim=WARP_SIZE,
    )


def gpu_matmul_q8_fused_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,    # output N FP32 elems
    d_in_fp32: UInt64,     # input K FP32 elems (already on GPU)
    d_w_q8: UInt64,        # Q8 blob (scales + int8), header stripped
    K: Int,
    N: Int,
) raises:
    """Fused: out[N] = dequant_q8(W[N,K]) @ in[K], reading the int8 weight directly
    on-GPU — no scratch, no cuBLAS. One warp per output row. out[N] = W[N,K] @ in[K]."""
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q8_gemv_kernel]()
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q8), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb),
        grid_dim=N, block_dim=WARP_SIZE,
    )
