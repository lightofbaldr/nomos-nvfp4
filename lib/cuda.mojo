"""CUDA runtime FFI helpers — thin wrappers around cudaMalloc/Memcpy/Sync.

All functions use external_call to hit libcudart.so directly; link with
-lcudart -L/usr/local/cuda/targets/sbsa-linux/lib at build time.

Conventions:
- All sizes are in bytes unless explicitly named as counts.
- Pointers are UInt64 (device addresses). We don't use typed device pointers
  because Mojo's FFI doesn't need them for external_call and it keeps the
  call sites uniform with cuBLAS (which also takes UInt64).
"""

from std.ffi import external_call, c_int
from std.memory import UnsafePointer
from std.collections import List
from std.os import abort
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA, CUstream, _CUstream_st


def cuda_malloc(bytes: Int) -> UInt64:
    var ptr: UInt64 = 0
    var rc = external_call["cudaMalloc", c_int, UnsafePointer[UInt64, origin_of(ptr)], UInt64](
        UnsafePointer(to=ptr), UInt64(bytes))
    if rc != c_int(0) or ptr == 0:
        abort(t"cudaMalloc failed: bytes={bytes} rc={Int(rc)} ptr={ptr}")
    return ptr


def cuda_free(ptr: UInt64):
    _ = external_call["cudaFree", c_int, UInt64](ptr)


def cuda_mem_free_bytes() -> Int:
    """Free device memory in bytes (cudaMemGetInfo).

    On DISCRETE cards (sm_120 RTX PRO / 5090, sm_100 B200) this is exact and is the
    ground truth for "will the model fit". On GB10's unified pool, lazy commit makes
    it UNDER-report what is actually reserved — treat it as a floor there.
    """
    var free_b: UInt64 = 0
    var total_b: UInt64 = 0
    _ = external_call["cudaMemGetInfo", c_int,
        UnsafePointer[UInt64, origin_of(free_b)],
        UnsafePointer[UInt64, origin_of(total_b)]](
        UnsafePointer(to=free_b), UnsafePointer(to=total_b))
    return Int(free_b)


def cuda_mem_total_bytes() -> Int:
    """Total device memory in bytes (cudaMemGetInfo)."""
    var free_b: UInt64 = 0
    var total_b: UInt64 = 0
    _ = external_call["cudaMemGetInfo", c_int,
        UnsafePointer[UInt64, origin_of(free_b)],
        UnsafePointer[UInt64, origin_of(total_b)]](
        UnsafePointer(to=free_b), UnsafePointer(to=total_b))
    return Int(total_b)


def cuda_sync():
    _ = external_call["cudaDeviceSynchronize", c_int]()


def cuda_stream_sync(ctx: DeviceContext) raises:
    ctx.stream().synchronize()


def cuda_budget_active() -> Bool:
    return external_call["nomos_budget_active", c_int]() != c_int(0)


def cuda_budget_token_begin(ctx: DeviceContext) raises:
    var stream = CUDA(ctx.stream())
    _ = external_call["nomos_budget_token_begin", c_int, CUstream](stream)


def cuda_budget_mark(ctx: DeviceContext, category: Int) raises:
    var stream = CUDA(ctx.stream())
    _ = external_call["nomos_budget_mark", c_int, CUstream, c_int](stream, c_int(category))


def cuda_budget_token_end(ctx: DeviceContext) raises:
    var stream = CUDA(ctx.stream())
    _ = external_call["nomos_budget_token_end", c_int, CUstream](stream)


def cuda_last_error() -> Int:
    """cudaGetLastError() — returns + RESETS the last runtime error (0 = cudaSuccess).
    The malloc/sync wrappers discard their codes, so this surfaces an OOB / failed-
    resource state that would otherwise masquerade as deterministic GEMM garbage."""
    return Int(external_call["cudaGetLastError", c_int]())


def cuda_memcpy(dst: UInt64, src: UInt64, bytes: Int, kind: Int):
    """kind: 1 = H2D, 2 = D2H, 3 = D2D (per cudaMemcpyKind enum)."""
    _ = external_call["cudaMemcpy", c_int, UInt64, UInt64, UInt64, c_int](
        dst, src, UInt64(bytes), c_int(kind))


def cuda_memcpy_async(
    dst: UInt64, src: UInt64, bytes: Int, kind: Int, ctx: DeviceContext
) raises:
    """cudaMemcpyAsync on ctx's CUDA stream.

    kind: 1 = H2D, 2 = D2H, 3 = D2D (per cudaMemcpyKind enum).
    """
    var stream = CUDA(ctx.stream())
    _ = external_call[
        "cudaMemcpyAsync",
        c_int,
        UInt64,
        UInt64,
        UInt64,
        c_int,
        CUstream,
    ](dst, src, UInt64(bytes), c_int(kind), stream)


def cuda_memcpy_2d(
    dst: UInt64, dpitch: Int,
    src: UInt64, spitch: Int,
    width_bytes: Int, height: Int,
    kind: Int,
):
    """Strided 2D copy — one API call that fills `height` rows at `dpitch`
    bytes stride in the destination, each row holding `width_bytes` bytes,
    from a source at `spitch` bytes stride. Used to scatter new KV cache
    entries into the strided [max_nkv, max_seq, cache_hd] layout with a
    single cudaMemcpy2D instead of nkv separate cudaMemcpy calls.
    """
    _ = external_call["cudaMemcpy2D", c_int,
        UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, c_int](
        dst, UInt64(dpitch),
        src, UInt64(spitch),
        UInt64(width_bytes), UInt64(height),
        c_int(kind))


def cuda_memcpy_2d_async(
    dst: UInt64, dpitch: Int,
    src: UInt64, spitch: Int,
    width_bytes: Int, height: Int,
    kind: Int,
    ctx: DeviceContext,
) raises:
    """cudaMemcpy2DAsync on ctx's CUDA stream."""
    var stream = CUDA(ctx.stream())
    _ = external_call[
        "cudaMemcpy2DAsync",
        c_int,
        UInt64,
        UInt64,
        UInt64,
        UInt64,
        UInt64,
        UInt64,
        c_int,
        CUstream,
    ](
        dst, UInt64(dpitch),
        src, UInt64(spitch),
        UInt64(width_bytes), UInt64(height),
        c_int(kind),
        stream,
    )


def cuda_host_alloc(bytes: Int) -> UInt64:
    """Pinned (page-locked) host memory, so H2D/D2H copies run at full PCIe speed and truly async.
    Returns 0 on failure (no abort): the KV store's RAM tier degrades instead of killing the serve."""
    var ptr: UInt64 = 0
    var rc = external_call["cudaHostAlloc", c_int, UnsafePointer[UInt64, origin_of(ptr)], UInt64, UInt32](
        UnsafePointer(to=ptr), UInt64(bytes), UInt32(0))
    if rc != c_int(0):
        # Handled failure: consume the non-sticky error so a later cuda_last_error() diagnostic does not misread it.
        _ = external_call["cudaGetLastError", c_int]()
        return 0
    return ptr


def cuda_free_host(ptr: UInt64):
    _ = external_call["cudaFreeHost", c_int, UInt64](ptr)


def cuda_stream_create() -> UInt64:
    """A non-blocking stream (cudaStreamNonBlocking = 1): no implicit sync with the legacy default stream."""
    var s: UInt64 = 0
    var rc = external_call["cudaStreamCreateWithFlags", c_int, UnsafePointer[UInt64, origin_of(s)], UInt32](
        UnsafePointer(to=s), UInt32(1))
    if rc != c_int(0):
        abort(t"cudaStreamCreateWithFlags failed rc={Int(rc)}")
    return s


def cuda_stream_destroy(s: UInt64):
    _ = external_call["cudaStreamDestroy", c_int, UInt64](s)


def cuda_event_create() -> UInt64:
    """cudaEventDisableTiming (= 2): a pure ordering fence, the cheapest event to record and wait on."""
    var e: UInt64 = 0
    var rc = external_call["cudaEventCreateWithFlags", c_int, UnsafePointer[UInt64, origin_of(e)], UInt32](
        UnsafePointer(to=e), UInt32(2))
    if rc != c_int(0):
        abort(t"cudaEventCreateWithFlags failed rc={Int(rc)}")
    return e


def cuda_event_destroy(e: UInt64):
    _ = external_call["cudaEventDestroy", c_int, UInt64](e)


def cuda_event_record(e: UInt64, stream: UInt64) -> Int:
    return Int(external_call["cudaEventRecord", c_int, UInt64, UInt64](e, stream))


def cuda_event_sync(e: UInt64) -> Int:
    return Int(external_call["cudaEventSynchronize", c_int, UInt64](e))


def _as_custream(h: UInt64) -> CUstream:
    """A raw stream handle as the CUstream that cuda_memcpy_async / cuda_memcpy_2d_async declare.

    The kernel is one compilation unit, so a C symbol may be declared with only one set of operand types:
    cudaMemcpyAsync and cudaMemcpy2DAsync take a CUstream here, not a UInt64."""
    return CUstream(UnsafePointer[_CUstream_st, MutUntrackedOrigin](unsafe_from_address=Int(h)))


def cuda_memcpy_async_on(dst: UInt64, src: UInt64, bytes: Int, kind: Int, stream: UInt64) -> Int:
    """cudaMemcpyAsync on an explicit stream. kind: 1 = H2D, 2 = D2H, 3 = D2D."""
    return Int(external_call["cudaMemcpyAsync", c_int, UInt64, UInt64, UInt64, c_int, CUstream](
        dst, src, UInt64(bytes), c_int(kind), _as_custream(stream)))


def cuda_memcpy_2d_async_on(
    dst: UInt64, dpitch: Int, src: UInt64, spitch: Int, width_bytes: Int, height: Int, kind: Int, stream: UInt64,
) -> Int:
    """cudaMemcpy2DAsync on an explicit stream: `height` rows of `width_bytes`, at the given pitches."""
    return Int(external_call["cudaMemcpy2DAsync", c_int, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, c_int, CUstream](
        dst, UInt64(dpitch), src, UInt64(spitch), UInt64(width_bytes), UInt64(height), c_int(kind), _as_custream(stream)))


def cuda_memset_2d(dst: UInt64, pitch: Int, value: Int, width_bytes: Int, height: Int):
    """Strided 2D memset — set `height` rows of `width_bytes` bytes at `pitch`
    byte stride to `value`. One call zeros positions 0..P across all KV heads
    (heads are laid out at MAX_SEQ*FULL_HD*4 stride), instead of NKV memsets."""
    _ = external_call["cudaMemset2D", c_int,
        UInt64, UInt64, c_int, UInt64, UInt64](
        dst, UInt64(pitch), c_int(value), UInt64(width_bytes), UInt64(height))


def cuda_upload(dst: UInt64, mut src: List[Float32]):
    """Host Float32 list → device pointer."""
    cuda_memcpy(dst, UInt64(Int(src.unsafe_ptr())), len(src) * 4, 1)


def cuda_upload_u16(dst: UInt64, mut src: List[UInt16]):
    """Host UInt16 list → device pointer (BF16 values packed as u16)."""
    cuda_memcpy(dst, UInt64(Int(src.unsafe_ptr())), len(src) * 2, 1)


def cuda_upload_u8(dst: UInt64, mut src: List[UInt8]):
    """Host UInt8 list → device pointer (raw bytes, e.g. a packed Q4 weight blob)."""
    cuda_memcpy(dst, UInt64(Int(src.unsafe_ptr())), len(src), 1)


def cuda_download(mut dst: List[Float32], src: UInt64, count: Int):
    """Device pointer → host Float32 list (must be pre-sized to `count`)."""
    cuda_memcpy(UInt64(Int(dst.unsafe_ptr())), src, count * 4, 2)


def cuda_download_i32(mut dst: List[Int32], src: UInt64, count: Int):
    """Device pointer → host Int32 list (Perf Lever A: 4-byte token read)."""
    cuda_memcpy(UInt64(Int(dst.unsafe_ptr())), src, count * 4, 2)
