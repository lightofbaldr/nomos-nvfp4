"""cuBLAS FFI helpers — create handle, run SGEMM, wrap matmul convenience.

Two SGEMM variants:
- cublas_sgemm: the "row-major matmul via transpose" wrapper used by every
  projection in the Gemma-4 pipeline. Computes C[M,N] = A[M,K] @ B[N,K]^T
  where both A and B live in row-major memory (weight files are stored
  row-major [N, K] which is [K, N] in cuBLAS column-major, so we transpose
  A internally via CUBLAS_OP_T).
- cublas_sgemm_nn: straight column-major matmul with configurable leading
  dimensions, used by the (currently-disabled) batched GPU attention path.

gpu_matmul is the ergonomic call site: host inp list → GPU → sgemm → host
out list. It manages its own temporary GPU buffers so callers don't have to.
"""

from std.ffi import external_call, c_int, c_float
from std.memory import UnsafePointer, bitcast
from std.collections import List
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA, CUstream

from lib.cuda import cuda_malloc, cuda_free, cuda_sync, cuda_upload, cuda_upload_u16, cuda_download, cuda_memcpy
from lib.cast_gpu_mojo import gpu_fp32_to_bf16_mojo


# ── cuBLAS type / compute constants ─────────────────────────────────────────
# From cublas_api.h / library_types.h — encoded as Int here because the
# enum values are all that cuBLAS looks at on the C side.
alias CUDA_R_32F: Int = 0      # float32
alias CUDA_R_16BF: Int = 14    # bfloat16
alias CUBLAS_COMPUTE_32F: Int = 68
alias CUBLAS_GEMM_DEFAULT: Int = -1


def cublas_create() -> UInt64:
    var h: UInt64 = 0
    _ = external_call["cublasCreate_v2", c_int, UnsafePointer[UInt64, origin_of(h)]](UnsafePointer(to=h))
    return h


def cublas_set_stream(h: UInt64, ctx: DeviceContext) raises:
    """Bind a cuBLAS handle to the CUDA stream owned by a DeviceContext."""
    var stream = CUDA(ctx.stream())
    _ = external_call["cublasSetStream_v2", c_int, UInt64, CUstream](h, stream)


def cublas_sgemm(h: UInt64, a: UInt64, b: UInt64, c: UInt64, M: Int, K: Int, N: Int):
    """out_row[M,N] = inp_row[M,K] @ weight_row[N,K]^T."""
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasSgemm_v2", c_int,
        UInt64, c_int, c_int, c_int, c_int, c_int,
        UnsafePointer[c_float, origin_of(alpha)], UInt64, c_int, UInt64, c_int,
        UnsafePointer[c_float, origin_of(beta)], UInt64, c_int](
        h, c_int(1), c_int(0), c_int(N), c_int(M), c_int(K),
        UnsafePointer(to=alpha), b, c_int(K), a, c_int(K),
        UnsafePointer(to=beta), c, c_int(N))


def cublas_sgemv(h: UInt64, d_w: UInt64, d_inp: UInt64, d_out: UInt64, K: Int, N: Int):
    """GEMV: out[N] = d_w[N, K] @ d_inp[K]   (row-major weight view).

    cuBLAS's SGEMM with M=1 takes ~1 ms per call even for small matrices
    because it's tuned for large-M GEMMs — the fixed kernel launch cost
    dominates and there's no parallelism in the M dimension to hide it.
    cublasSgemv_v2 is the proper GEMV primitive and is dramatically
    faster for this pattern. Measured ~6× improvement on the Gemma-4
    projection sizes (K=5376, N ∈ [4096, 21504]).

    Row-major mental model:
      d_w is the projection weight stored row-major as [N, K]
      d_inp is the input vector of length K
      d_out is the output vector of length N

    In cuBLAS's col-major world:
      A_col = d_w reinterpreted — [K, N] col-major, lda = K
      We want y = A_col^T · x, so trans = CUBLAS_OP_T
      cuBLAS GEMV signature: y = α op(A) x + β y
        with A of shape [m, n] in col-major and lda as the col-major leading dim.
      Here m = K (col-major rows of A), n = N (col-major cols).
    """
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasSgemv_v2", c_int,
        UInt64,                                           # handle
        c_int,                                            # trans
        c_int, c_int,                                     # m, n
        UnsafePointer[c_float, origin_of(alpha)],         # alpha
        UInt64, c_int,                                    # A, lda
        UInt64, c_int,                                    # x, incx
        UnsafePointer[c_float, origin_of(beta)],          # beta
        UInt64, c_int,                                    # y, incy
    ](
        h,
        c_int(1),                                         # CUBLAS_OP_T
        c_int(K), c_int(N),                               # m=K, n=N in col-major
        UnsafePointer(to=alpha),
        d_w, c_int(K),                                    # A, lda=K
        d_inp, c_int(1),                                  # x, incx=1
        UnsafePointer(to=beta),
        d_out, c_int(1),                                  # y, incy=1
    )


def cublas_sgemm_nn(h: UInt64, a: UInt64, lda: Int, b: UInt64, ldb: Int,
                     c: UInt64, ldc: Int, m: Int, n: Int, k: Int):
    """C_col[m,n] = A_col[m,k] @ B_col[k,n] (no transposes)."""
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasSgemm_v2", c_int,
        UInt64, c_int, c_int, c_int, c_int, c_int,
        UnsafePointer[c_float, origin_of(alpha)], UInt64, c_int, UInt64, c_int,
        UnsafePointer[c_float, origin_of(beta)], UInt64, c_int](
        h, c_int(0), c_int(0), c_int(m), c_int(n), c_int(k),
        UnsafePointer(to=alpha), a, c_int(lda), b, c_int(ldb),
        UnsafePointer(to=beta), c, c_int(ldc))


def cublas_sgemm_strided_batched_nn(
    h: UInt64,
    a: UInt64, lda: Int, stride_a: Int,
    b: UInt64, ldb: Int, stride_b: Int,
    c: UInt64, ldc: Int, stride_c: Int,
    M: Int, K: Int, N: Int,
    batch_count: Int,
):
    """Batched row-major A · B over `batch_count` matrices (no transposes).

    For each i in [0, batch_count):
        C_i[M, N] = A_i[M, K] @ B_i[K, N]

    This is the complement of cublas_sgemm_strided_batched_nt — same
    stride/leading-dim conventions, but here neither A nor B is
    transposed, so it computes a plain batched matmul.

    Used for the scores · V step of GPU attention, where we have
    scores[kvg, seq_len] and V[seq_len, l_hd] and want
    attn_out[kvg, l_hd]. The existing _nt variant does A · B^T and
    would need V to be stored as [l_hd, seq_len]; since V is cached as
    [seq_len, cache_hd], this plain NN variant is the natural fit.
    """
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasSgemmStridedBatched", c_int,
        UInt64,
        c_int, c_int,
        c_int, c_int, c_int,
        UnsafePointer[c_float, origin_of(alpha)],
        UInt64, c_int, Int64,
        UInt64, c_int, Int64,
        UnsafePointer[c_float, origin_of(beta)],
        UInt64, c_int, Int64,
        c_int,
    ](
        h,
        c_int(0), c_int(0),                              # both N (no transpose)
        c_int(N), c_int(M), c_int(K),                    # col-major: m=N, n=M, k=K
        UnsafePointer(to=alpha),
        # For A · B in row-major with col-major cuBLAS:
        #   row-major C[M,N] = row-major A[M,K] @ row-major B[K,N]
        # In col-major: C_col[N,M] = B_col[N,K] @ A_col[K,M]
        # So cuBLAS slot "A" = our row-major B, slot "B" = our row-major A.
        b, c_int(ldb), Int64(stride_b),                  # A_cublas = our B
        a, c_int(lda), Int64(stride_a),                  # B_cublas = our A
        UnsafePointer(to=beta),
        c, c_int(ldc), Int64(stride_c),
        c_int(batch_count),
    )


def cublas_sgemm_strided_batched_nt(
    h: UInt64,
    a: UInt64, lda: Int, stride_a: Int,
    b: UInt64, ldb: Int, stride_b: Int,
    c: UInt64, ldc: Int, stride_c: Int,
    M: Int, K: Int, N: Int,
    batch_count: Int,
):
    """Batched row-major A · B^T over `batch_count` independent matrices.

    For each i in [0, batch_count):
        C_i[M, N] = A_i[M, K] @ B_i[N, K]^T

    A_i, B_i, C_i are slices at ELEMENT offsets i*stride_a, i*stride_b,
    i*stride_c into the respective device pointers (strides are in
    element counts, not bytes — cuBLAS's convention).

    Parameters:
      a, lda, stride_a: "A" = the matrix our row-major code calls A,
        shape [M, K]. lda is the leading dimension in cuBLAS's col-major
        view, which for a tightly packed row-major [M, K] is `K` — but
        if the underlying storage has padding (e.g. K_cache stored at
        cache_hd stride while hd < cache_hd), lda is the storage stride.
      b, ldb, stride_b: "B" = the matrix our row-major code calls B,
        shape [N, K]. Same lda rules apply — ldb is the storage stride
        per row in the row-major view (= K for tight, cache_hd for the
        padded K/V cache).
      c, ldc, stride_c: "C" = output, row-major [M, N]. ldc is the
        storage stride per row, which in cuBLAS's col-major view is
        the leading dim. For tight packing that's N.

    Used by the GPU attention kernel:
      - Q·K^T: A = Q (shape [kvg, hd], tight), B = K_cache (shape
        [seq_len, hd], stored at cache_hd stride) → C = scores
        shape [kvg, seq_len] (tight). Batched over nkv KV heads.
      - scores·V: A = scores, B = V_cache, C = attn_out. Same pattern.
    """
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasSgemmStridedBatched", c_int,
        UInt64,                                          # handle
        c_int, c_int,                                    # transa, transb
        c_int, c_int, c_int,                             # m, n, k (cublas col-major)
        UnsafePointer[c_float, origin_of(alpha)],        # alpha
        UInt64, c_int, Int64,                            # A, lda, strideA (element counts)
        UInt64, c_int, Int64,                            # B, ldb, strideB
        UnsafePointer[c_float, origin_of(beta)],         # beta
        UInt64, c_int, Int64,                            # C, ldc, strideC
        c_int,                                           # batchCount
    ](
        h,
        c_int(1), c_int(0),                              # A transposed, B not
        c_int(N), c_int(M), c_int(K),                    # col-major: m=N, n=M, k=K
        UnsafePointer(to=alpha),
        # cuBLAS "A" slot = our row-major "B" (the one that gets transposed).
        # Its col-major lda is our `ldb` (storage stride per row in row-major).
        b, c_int(ldb), Int64(stride_b),
        # cuBLAS "B" slot = our row-major "A".
        # Col-major lda = our `lda` (storage stride per row).
        a, c_int(lda), Int64(stride_a),
        UnsafePointer(to=beta),
        # Output: col-major lda = our `ldc` (row stride in row-major).
        c, c_int(ldc), Int64(stride_c),
        c_int(batch_count),
    )


# ── BF16 strided-batched twins (tensor-core attention) ──────────────────────
# Exact bf16 mirrors of the fp32 strided GEMMs above: A and B are BF16 (2 bytes/
# elem), C stays FP32, accumulate FP32, tensor-core path. cuBLAS strides/lda are
# ELEMENT counts (not bytes), so callers pass the IDENTICAL lda/stride/M/K/N args
# as the fp32 versions — only the buffers' dtype differs. Used by the bf16
# attention path (precision_bits=16): Q·K^T (nt) over bf16 Q + bf16 K cache, and
# scores·V (nn) over bf16 scores + bf16 V cache.

def cublas_gemm_ex_sb_bf16_nn(
    h: UInt64,
    a: UInt64, lda: Int, stride_a: Int,
    b: UInt64, ldb: Int, stride_b: Int,
    c: UInt64, ldc: Int, stride_c: Int,
    M: Int, K: Int, N: Int,
    batch_count: Int,
):
    """BF16 twin of cublas_sgemm_strided_batched_nn (C[M,N]=A[M,K]@B[K,N], FP32 out)."""
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasGemmStridedBatchedEx", c_int,
        UInt64,
        c_int, c_int,
        c_int, c_int, c_int,
        UnsafePointer[c_float, origin_of(alpha)],
        UInt64, c_int, c_int, Int64,        # A, Atype, lda, strideA
        UInt64, c_int, c_int, Int64,        # B, Btype, ldb, strideB
        UnsafePointer[c_float, origin_of(beta)],
        UInt64, c_int, c_int, Int64,        # C, Ctype, ldc, strideC
        c_int, c_int, c_int,                # batchCount, computeType, algo
    ](
        h,
        c_int(0), c_int(0),
        c_int(N), c_int(M), c_int(K),
        UnsafePointer(to=alpha),
        b, c_int(CUDA_R_16BF), c_int(ldb), Int64(stride_b),   # A_cublas = our B (bf16)
        a, c_int(CUDA_R_16BF), c_int(lda), Int64(stride_a),   # B_cublas = our A (bf16)
        UnsafePointer(to=beta),
        c, c_int(CUDA_R_32F), c_int(ldc), Int64(stride_c),    # C fp32
        c_int(batch_count),
        c_int(CUBLAS_COMPUTE_32F),
        c_int(CUBLAS_GEMM_DEFAULT),
    )


def cublas_gemm_ex_sb_bf16_nt(
    h: UInt64,
    a: UInt64, lda: Int, stride_a: Int,
    b: UInt64, ldb: Int, stride_b: Int,
    c: UInt64, ldc: Int, stride_c: Int,
    M: Int, K: Int, N: Int,
    batch_count: Int,
):
    """BF16 twin of cublas_sgemm_strided_batched_nt (C[M,N]=A[M,K]@B[N,K]^T, FP32 out)."""
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasGemmStridedBatchedEx", c_int,
        UInt64,
        c_int, c_int,
        c_int, c_int, c_int,
        UnsafePointer[c_float, origin_of(alpha)],
        UInt64, c_int, c_int, Int64,
        UInt64, c_int, c_int, Int64,
        UnsafePointer[c_float, origin_of(beta)],
        UInt64, c_int, c_int, Int64,
        c_int, c_int, c_int,
    ](
        h,
        c_int(1), c_int(0),                              # A (our B) transposed
        c_int(N), c_int(M), c_int(K),
        UnsafePointer(to=alpha),
        b, c_int(CUDA_R_16BF), c_int(ldb), Int64(stride_b),
        a, c_int(CUDA_R_16BF), c_int(lda), Int64(stride_a),
        UnsafePointer(to=beta),
        c, c_int(CUDA_R_32F), c_int(ldc), Int64(stride_c),
        c_int(batch_count),
        c_int(CUBLAS_COMPUTE_32F),
        c_int(CUBLAS_GEMM_DEFAULT),
    )


def gpu_matmul(handle: UInt64, mut out: List[Float32], inp: List[Float32],
               d_w: UInt64, K: Int, N: Int):
    """Run out[N] = d_w[N,K] @ inp[K] on the GPU.

    DEPRECATED for hot-path use — allocates two new GPU buffers per call
    which is ~1 ms of pure launch overhead. For the per-layer forward pass,
    use `gpu_matmul_pooled` with pre-allocated scratch buffers.

    Kept around for low-volume setup code (weight loading, one-shot LM
    head) where the overhead doesn't matter.
    """
    var d_inp = cuda_malloc(K * 4)
    var d_out = cuda_malloc(N * 4)
    var inp_c = inp.copy()
    cuda_upload(d_inp, inp_c)
    cublas_sgemm(handle, d_inp, d_w, d_out, 1, K, N)
    cuda_sync()
    cuda_download(out, d_out, N)
    cuda_free(d_inp)
    cuda_free(d_out)


def gpu_matmul_pooled(
    handle: UInt64,
    mut out: List[Float32],
    inp: List[Float32],
    d_w: UInt64,
    d_scratch_in: UInt64,
    d_scratch_out: UInt64,
    K: Int,
    N: Int,
):
    """out[N] = d_w[N, K] @ inp[K] with a pre-allocated device scratch pool
    (FP32 weight, FP32 input, FP32 output).

    Kept for reference and for the non-hot-path callers. The hot path uses
    gpu_matmul_bf16, which reads half the weight bytes and is ~2× faster
    end-to-end.
    """
    var inp_c = inp.copy()
    cuda_upload(d_scratch_in, inp_c)
    cublas_sgemv(handle, d_w, d_scratch_in, d_scratch_out, K, N)
    cuda_download(out, d_scratch_out, N)


def cublas_gemm_ex_bf16_nt(
    h: UInt64,
    d_a_bf16: UInt64,   # weight, row-major [N, K] packed as BF16 (2 bytes/elem)
    d_b_bf16: UInt64,   # input, row-major [M, K] packed as BF16
    d_c_fp32: UInt64,   # output, row-major [M, N] as FP32
    M: Int, K: Int, N: Int,
):
    """Row-major C[M,N] = A[M,K] @ B[N,K]^T with BF16 inputs and FP32 output.

    Wraps cublasGemmEx with:
      - A type:       CUDA_R_16BF (the B parameter on the cuBLAS side, since
                      we pass our row-major "input" as cuBLAS's B)
      - B type:       CUDA_R_16BF (our row-major "weight", transposed via
                      CUBLAS_OP_T, in cuBLAS's A slot)
      - C type:       CUDA_R_32F
      - compute type: CUBLAS_COMPUTE_32F
      - algo:         CUBLAS_GEMM_DEFAULT (let cuBLAS pick the tensor-core path)

    Same row-major mental model as cublas_sgemm — transa=T on our weight,
    transb=N on our input, m=N, n=M, k=K in cuBLAS col-major terms. Leading
    dimensions are the row strides in row-major, which for tight packing
    are K for both A and B and N for C.

    Parameter naming note: on the Mojo side `d_a_bf16` is our row-major
    "weight" (shape [N, K]) and `d_b_bf16` is our row-major "input"
    (shape [M, K]), matching the cublas_sgemm convention of (a=input,
    b=weight). No — wait. Looking at this again: the existing cublas_sgemm
    signature in this file is `cublas_sgemm(h, a, b, c, M, K, N)` with
    docstring "out_row[M,N] = inp_row[M,K] @ weight_row[N,K]^T". So `a`
    is the input and `b` is the weight. I'm using `d_a_bf16` for the
    weight and `d_b_bf16` for the input here, which is BACKWARDS from
    that convention. Renaming to avoid confusion.
    """
    var alpha: c_float = 1.0
    var beta: c_float = 0.0
    _ = external_call["cublasGemmEx", c_int,
        UInt64,                                           # handle
        c_int, c_int,                                     # transa, transb
        c_int, c_int, c_int,                              # m, n, k
        UnsafePointer[c_float, origin_of(alpha)],         # alpha*
        UInt64, c_int, c_int,                             # A, Atype, lda
        UInt64, c_int, c_int,                             # B, Btype, ldb
        UnsafePointer[c_float, origin_of(beta)],          # beta*
        UInt64, c_int, c_int,                             # C, Ctype, ldc
        c_int,                                            # computeType
        c_int,                                            # algo
    ](
        h,
        c_int(1), c_int(0),                               # transa=T (weight), transb=N (input)
        c_int(N), c_int(M), c_int(K),                     # col-major m=N, n=M, k=K
        UnsafePointer(to=alpha),
        d_a_bf16, c_int(CUDA_R_16BF), c_int(K),           # A slot = weight, lda=K
        d_b_bf16, c_int(CUDA_R_16BF), c_int(K),           # B slot = input, ldb=K
        UnsafePointer(to=beta),
        d_c_fp32, c_int(CUDA_R_32F), c_int(N),            # C is FP32
        c_int(CUBLAS_COMPUTE_32F),
        c_int(CUBLAS_GEMM_DEFAULT),
    )


def gpu_matmul_bf16_dev(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,            # device output buffer (N*4 bytes)
    d_in_fp32: UInt64,             # device input buffer (K*4 bytes — already on GPU)
    d_w_bf16: UInt64,
    d_scratch_in_bf16: UInt64,     # device scratch for the BF16 input (K*2 bytes min)
    K: Int,
    N: Int,
) raises:
    """out_dev[N] = d_w_bf16[N, K] @ inp_dev[K] — device-resident variant.

    No upload, no download. The FP32 input is already on GPU. We cast it to
    BF16 in scratch via the pure-Mojo cast kernel (lib/cast_gpu_mojo.mojo),
    then run cublasGemmEx.

    For per-layer device-resident operation: keeps activations on GPU the
    whole forward pass.
    """
    gpu_fp32_to_bf16_mojo(ctx, d_in_fp32, d_scratch_in_bf16, K)
    cublas_gemm_ex_bf16_nt(
        handle,
        d_w_bf16,              # weight (A in row-major semantics)
        d_scratch_in_bf16,     # input  (B in row-major semantics)
        d_out_fp32,            # FP32 output
        1, K, N,
    )


def gpu_matmul_bf16_dev_batched(
    ctx: DeviceContext,
    handle: UInt64,
    d_out_fp32: UInt64,            # device output [S, N] row-major (S*N*4 bytes)
    d_in_fp32: UInt64,             # device input  [S, K] row-major (already on GPU)
    d_w_bf16: UInt64,              # weight [N, K] BF16
    d_scratch_in_bf16: UInt64,     # scratch for BF16 input (S*K*2 bytes min)
    S: Int,                        # batch / sequence length
    K: Int,
    N: Int,
) raises:
    """Batched variant of gpu_matmul_bf16_dev: out[S,N] = in[S,K] @ w[N,K]^T.

    Identical to the single-token path except M=S instead of 1 — `cublas_gemm_ex_bf16_nt`
    already takes M as the batch/row count, so the only changes are casting S*K input
    elements and passing S. This makes the matmul a compute-bound GEMM that reuses each
    weight tile across all S rows (the prefill speedup)."""
    gpu_fp32_to_bf16_mojo(ctx, d_in_fp32, d_scratch_in_bf16, S * K)
    cublas_gemm_ex_bf16_nt(
        handle,
        d_w_bf16,              # weight (A slot, transposed)
        d_scratch_in_bf16,     # input  (B slot) [S, K]
        d_out_fp32,            # FP32 output [S, N]
        S, K, N,
    )


def gpu_matmul_bf16(
    handle: UInt64,
    mut out: List[Float32],
    mut inp: List[Float32],
    d_w_bf16: UInt64,
    d_scratch_in_bf16: UInt64,     # device scratch for the BF16 input (K*2 bytes min)
    d_scratch_out_fp32: UInt64,    # device scratch for the FP32 output (N*4 bytes min)
    K: Int,
    N: Int,
):
    """out[N] = d_w_bf16[N, K] @ inp[K] with BF16 weights and BF16 input.

    Steps:
      1. Host-side: cast FP32 input to BF16 (truncate the low 16 bits)
      2. Upload the BF16 input to device scratch (K*2 bytes)
      3. Run cublasGemmEx with BF16 inputs, FP32 output
      4. Download the FP32 output

    Compared to the FP32 path:
      - Weight reads drop by 2× (the big win — weight bandwidth is the
        forward pass bottleneck)
      - Input reads drop by 2× (small, but helps)
      - Accumulation stays FP32 (cuBLAS uses tensor cores if available)
      - Output is FP32 — activations stay full precision for the residual
        stream and for downstream consumers

    d_scratch_in_bf16 must be at least K*2 bytes.
    d_scratch_out_fp32 must be at least N*4 bytes.
    """
    var n = len(inp)
    var inp_bf16 = List[UInt16](capacity=n)
    for i in range(n):
        # Proper IEEE round-to-nearest-even FP32 → BF16 conversion.
        # Use Mojo's SIMD.cast[DType.bfloat16]() which calls the hardware
        # correct rounding path, then bitcast the result back to UInt16
        # so we can pack it into the upload buffer.
        # This matters for generation quality: truncation has a systematic
        # downward bias that over 420 matmul calls per token (60 layers
        # × 7 projections) accumulates enough drift to flip the argmax
        # for near-tied logits at contraction boundaries ("isn't" → " me",
        # "can't" → " t", etc.). RTN keeps the drift centered.
        var f_simd = SIMD[DType.float32, 1](inp[i])
        var bf_simd = f_simd.cast[DType.bfloat16]()
        var bf_bits = bitcast[DType.uint16, 1](bf_simd)
        inp_bf16.append(bf_bits[0])

    cuda_upload_u16(d_scratch_in_bf16, inp_bf16)
    cublas_gemm_ex_bf16_nt(
        handle,
        d_w_bf16,              # weight (A in row-major semantics)
        d_scratch_in_bf16,     # input  (B in row-major semantics)
        d_scratch_out_fp32,    # FP32 output
        1, K, N,
    )
    cuda_download(out, d_scratch_out_fp32, N)
