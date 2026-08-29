"""q4_weights.mojo — Q4_0 (group-32 symmetric INT4) WEIGHT path for the kernel.

The Google QAT checkpoint (gemma-4-31B-it-qat-q4_0) was QAT-trained for this exact
format, so it leverages the QAT fidelity (unlike our host/on-disk codec, which is for
our own models). Quantizer: tools/quantize_q4_0.py.

M1 (this file): dequant a packed Q4_0 weight blob -> BF16 scratch, then the existing
cuBLAS BF16 GEMM runs unchanged. Wins: weights stored ~4.5 bits/wt (62GB -> ~17GB) and
the Q4 quality is measurable. The decode SPEED win is M2 (fused dequant in the GEMM
tile load — never materialize the bf16 weight).

GPU blob layout (host strips the 16-byte .q4 file header): contiguous
  [ nb * float16 block-scales ][ nb * 16 bytes packed nibbles ]
Element i: block = i//32, j = i%32. The (q+8) nibble lives in packed[block*16 + j//2]
(low nibble for even j, high for odd). w = (nibble - 8) * scale[block].
"""
from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer, bitcast
from lib.cuda import cuda_malloc, cuda_upload_u8
from lib.io import read_q4_bytes
from lib.engine_init import _env_flag_cached, ENV_Q4_FUSED

alias GROUP = 32


def _as_u8_ptr(addr: UInt64) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_u16_ptr(addr: UInt64) -> UnsafePointer[UInt16, MutAnyOrigin]:
    return UnsafePointer[UInt16, MutAnyOrigin](unsafe_from_address=Int(addr))


def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


def dequant_q4_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [scales fp16][packed u8]
    out_bf16: UnsafePointer[UInt16, MutAnyOrigin],
    n_arg: Int32,
    nb_arg: Int32,
):
    # DevicePassable (mojo dev2026080106): fixed-width GPU args, widened once here.
    var n = Int(n_arg)
    var nb = Int(nb_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        var block = i // GROUP
        var j = i % GROUP
        # scale: float16 at byte offset block*2 (scales are contiguous at blob start)
        var sbits = blob.bitcast[UInt16]()[block]
        var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
        # packed nibble: packed region starts at byte nb*2
        var byte = blob[nb * 2 + block * 16 + (j // 2)]
        var nib = (byte & 0x0F) if (j & 1) == 0 else ((byte >> 4) & 0x0F)
        var q = Int(nib) - 8
        var val = SIMD[DType.float32, 1](Float32(q) * scale[0])
        var bf = val.cast[DType.bfloat16]()
        out_bf16[i] = bitcast[DType.uint16, 1](bf)[0]


def gpu_dequant_q4(
    ctx: DeviceContext, blob_ptr: UInt64, out_bf16_ptr: UInt64, n: Int, nb: Int
) raises:
    """Dequant a Q4_0 weight blob into a BF16 device buffer (n elements, 2 bytes each)."""
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[dequant_q4_kernel]()
    ctx.enqueue_function(
        k, _as_u8_ptr(blob_ptr), _as_u16_ptr(out_bf16_ptr), Int32(n), Int32(nb),
        grid_dim=blocks, block_dim=threads,
    )


def load_to_gpu_q4(path: String) -> UInt64:
    """Load a .q4 file (header int64 n + int64 nb, then fp16 scales, then packed u8)
    onto a single GPU buffer holding [scales][packed]. Returns the blob ptr (0 on
    missing/empty — caller skips). n/nb are derived at the matmul from K,N, so the
    file header is only used here to size the body read."""
    var hdr = List[Int]()          # [n, nb]
    var bytes = read_q4_bytes(path, hdr)   # host: strips the 16-byte header, returns the rest
    if len(bytes) == 0 or len(hdr) < 2:
        print("[WARN] Empty or missing q4:", path)
        return UInt64(0)
    var ptr = cuda_malloc(len(bytes))
    cuda_upload_u8(ptr, bytes)
    return ptr


def _q4_body_ok(path: String, hdr: List[Int], body: List[UInt8]) -> Bool:
    if len(body) == 0 or len(hdr) < 2:
        print("[WARN] Empty or missing q4:", path)
        return False
    var n = hdr[0]
    var nb = hdr[1]
    if n % GROUP != 0:
        print("[WARN] q4 concat requires group-aligned body:", path, " n=", n)
        return False
    if len(body) != nb * 18:
        print(
            "[WARN] q4 concat refusing non-q4 body:",
            path,
            " bytes=",
            len(body),
            " expected=",
            nb * 18,
        )
        return False
    return True


def _append_q4_bytes(
    mut out: List[UInt8], src: List[UInt8], start: Int, count: Int
):
    for i in range(count):
        out.append(src[start + i])


def _upload_q4_concat2(
    body0: List[UInt8], nb0: Int, body1: List[UInt8], nb1: Int
) -> UInt64:
    var bytes = List[UInt8](capacity=(nb0 + nb1) * 18)
    _append_q4_bytes(bytes, body0, 0, nb0 * 2)
    _append_q4_bytes(bytes, body1, 0, nb1 * 2)
    _append_q4_bytes(bytes, body0, nb0 * 2, nb0 * 16)
    _append_q4_bytes(bytes, body1, nb1 * 2, nb1 * 16)
    var ptr = cuda_malloc(len(bytes))
    cuda_upload_u8(ptr, bytes)
    return ptr


def _upload_q4_concat3(
    body0: List[UInt8],
    nb0: Int,
    body1: List[UInt8],
    nb1: Int,
    body2: List[UInt8],
    nb2: Int,
) -> UInt64:
    var bytes = List[UInt8](capacity=(nb0 + nb1 + nb2) * 18)
    _append_q4_bytes(bytes, body0, 0, nb0 * 2)
    _append_q4_bytes(bytes, body1, 0, nb1 * 2)
    _append_q4_bytes(bytes, body2, 0, nb2 * 2)
    _append_q4_bytes(bytes, body0, nb0 * 2, nb0 * 16)
    _append_q4_bytes(bytes, body1, nb1 * 2, nb1 * 16)
    _append_q4_bytes(bytes, body2, nb2 * 2, nb2 * 16)
    var ptr = cuda_malloc(len(bytes))
    cuda_upload_u8(ptr, bytes)
    return ptr


def load_to_gpu_q4_concat2(path0: String, path1: String) -> UInt64:
    """Load two row-stacked .q4 weights as one Q4 blob.

    The source files store [scales][packed]. Row concat along N therefore needs
    [scales0][scales1][packed0][packed1], not raw body concatenation.
    """
    var hdr0 = List[Int]()
    var body0 = read_q4_bytes(path0, hdr0)
    if not _q4_body_ok(path0, hdr0, body0):
        return UInt64(0)
    var hdr1 = List[Int]()
    var body1 = read_q4_bytes(path1, hdr1)
    if not _q4_body_ok(path1, hdr1, body1):
        return UInt64(0)
    return _upload_q4_concat2(body0, hdr0[1], body1, hdr1[1])


def load_to_gpu_q4_concat3(
    path0: String, path1: String, path2: String
) -> UInt64:
    """Load three row-stacked .q4 weights as one Q4 blob."""
    var hdr0 = List[Int]()
    var body0 = read_q4_bytes(path0, hdr0)
    if not _q4_body_ok(path0, hdr0, body0):
        return UInt64(0)
    var hdr1 = List[Int]()
    var body1 = read_q4_bytes(path1, hdr1)
    if not _q4_body_ok(path1, hdr1, body1):
        return UInt64(0)
    var hdr2 = List[Int]()
    var body2 = read_q4_bytes(path2, hdr2)
    if not _q4_body_ok(path2, hdr2, body2):
        return UInt64(0)
    return _upload_q4_concat3(body0, hdr0[1], body1, hdr1[1], body2, hdr2[1])


def gpu_matmul_q4_dev(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,                # packed Q4 blob for weight [N, K]
    n_w: Int,                      # N*K (element count of the weight)
    nb_w: Int,                     # blocks = ceil(N*K/32)
    d_weight_bf16_scratch: UInt64, # scratch >= N*K*2 bytes (reused across GEMMs)
    d_scratch_in_bf16: UInt64,
    K: Int,
    N: Int,
) raises:
    """M1 q4 twin of gpu_matmul_bf16_dev: dequant the Q4 weight -> bf16 scratch, then
    the identical cuBLAS BF16 GEMM. out[N] = W[N,K] @ in[K]."""
    from lib.cublas import cublas_gemm_ex_bf16_nt
    from lib.cast_gpu_mojo import gpu_fp32_to_bf16_mojo
    gpu_dequant_q4(ctx, d_w_q4, d_weight_bf16_scratch, n_w, nb_w)
    gpu_fp32_to_bf16_mojo(ctx, d_in_fp32, d_scratch_in_bf16, K)
    cublas_gemm_ex_bf16_nt(handle, d_weight_bf16_scratch, d_scratch_in_bf16, d_out_fp32, 1, K, N)


# ─────────────────────────────────────────────────────────────────────────
# M2 — FUSED INT4xbf16 dequant-GEMV (the decode SPEED path).
#
# M1 (above) dequants the whole Q4 weight into a bf16 scratch, then runs cuBLAS
# — so the weight crosses HBM ~3x (read Q4, write bf16, read bf16), WORSE than
# plain bf16. M2 reads ONLY the Q4 blob (4.5 bits/elem) once and does the matmul
# on-chip: out[n] = sum_k dequant_q4(W[n,k]) * in[k]. On the bandwidth-starved
# GB10 that ~3.5x less weight traffic IS the decode win.
#
# Decode is M=1 (one token) => GEMV. ONE WARP per output row n: the 32 lanes
# stride over the row's K elements (coalesced — a row's Q4 bytes are contiguous
# because K % 32 == 0, so every row starts on a group boundary), each lane
# accumulates a partial dot in FP32, then a warp-shuffle butterfly reduces the
# 32 partials. No shared memory, no barrier. Input is read at full FP32
# precision (we quantize the WEIGHT, not the activation).
# ─────────────────────────────────────────────────────────────────────────
def q4_gemv_kernel(
    blob: UnsafePointer[UInt8, MutAnyOrigin],     # [nb fp16 scales][nb*16 packed nibbles]
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
        var j = i % GROUP
        var sbits = blob.bitcast[UInt16]()[block]
        var scale = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](sbits)).cast[DType.float32]()
        var byte = blob[nb * 2 + block * 16 + (j // 2)]
        var nib = (byte & 0x0F) if (j & 1) == 0 else ((byte >> 4) & 0x0F)
        var q = Int(nib) - 8
        var w = Float32(q) * scale[0]
        local += w * inp[kk]
        kk += WARP_SIZE

    # warp-shuffle butterfly reduce (every lane ends with the row's dot product)
    var off = WARP_SIZE // 2
    while off > 0:
        local += shuffle_xor(local, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(local)


def gpu_matmul_q4_fused_dev(
    ctx: DeviceContext,
    d_out_fp32: UInt64,    # output N FP32 elems
    d_in_fp32: UInt64,     # input K FP32 elems (already on GPU)
    d_w_q4: UInt64,        # Q4 blob (scales + packed), header stripped
    K: Int,
    N: Int,
) raises:
    """M2 fused: out[N] = dequant_q4(W[N,K]) @ in[K], reading the Q4 weight
    directly on-GPU — no bf16 materialization, no scratch, no cuBLAS. One warp
    per output row."""
    var nb = (K * N + GROUP - 1) // GROUP
    var kern = ctx.compile_function[q4_gemv_kernel]()
    ctx.enqueue_function(
        kern,
        _as_u8_ptr(d_w_q4), _as_f32_ptr(d_in_fp32), _as_f32_ptr(d_out_fp32),
        Int32(K), Int32(N), Int32(nb),
        grid_dim=N, block_dim=WARP_SIZE,
    )


def _q4_use_fused() -> Bool:
    """NOMOS_Q4_FUSED toggle (default M2 fused GEMV). Set NOMOS_Q4_FUSED=0 to use
    the M1 materialized path instead — for the M1-vs-M2 decode A/B in one binary."""
    # Cached in the C shim (read once/process, not per-GEMV). Default ON; only "0" disables.
    return _env_flag_cached(ENV_Q4_FUSED) != 0


def gpu_matmul_q4_auto_dev(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,
    d_in_fp32: UInt64,
    d_w_q4: UInt64,
    d_weight_bf16_scratch: UInt64,   # only used by the M1 path
    d_scratch_in_bf16: UInt64,       # only used by the M1 path
    K: Int,
    N: Int,
) raises:
    """Decode (M=1) Q4 dispatch: M2 fused GEMV by default, M1 materialized if
    NOMOS_Q4_FUSED=0. Both produce equivalent output (greedy-parity verified)."""
    if _q4_use_fused():
        gpu_matmul_q4_fused_dev(ctx, d_out_fp32, d_in_fp32, d_w_q4, K, N)
    else:
        gpu_matmul_q4_dev(
            ctx, handle, d_out_fp32, d_in_fp32, d_w_q4, K * N,
            (K * N + GROUP - 1) // GROUP, d_weight_bf16_scratch, d_scratch_in_bf16, K, N,
        )


def gpu_matmul_q4_dev_batched(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,              # [S, N] row-major
    d_in_fp32: UInt64,               # [S, K] row-major (on GPU)
    d_w_q4: UInt64,                  # Q4 blob [N, K]
    d_weight_bf16_scratch: UInt64,   # >= N*K*2 bytes (the dequant'd weight)
    d_scratch_in_bf16: UInt64,       # >= S*K*2 bytes (the bf16 input)
    S: Int,
    K: Int,
    N: Int,
) raises:
    """Prefill (M=S) Q4: dequant the weight ONCE -> bf16 scratch, then the batched
    cuBLAS GEMM reuses that weight across all S tokens — the dequant cost amortizes
    over S, so materialized (not fused-GEMV) is the right choice for prefill."""
    from lib.cublas import gpu_matmul_bf16_dev_batched
    var n_w = K * N
    gpu_dequant_q4(ctx, d_w_q4, d_weight_bf16_scratch, n_w, (n_w + GROUP - 1) // GROUP)
    gpu_matmul_bf16_dev_batched(
        ctx, handle, d_out_fp32, d_in_fp32, d_weight_bf16_scratch, d_scratch_in_bf16, S, K, N,
    )
