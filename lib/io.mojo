"""File I/O helpers — read raw FP32 binary weight files and upload to GPU.

The Gemma-4 weights are pre-converted to raw little-endian FP32 binary
files by tools/convert_weights.py, split into /tmp/nomos_gemma4_weights/
primary/ (layers 0-29 + embedding) and secondary/ (layers 30-59 + norms).

Each file contains just the float values in row-major order — no header,
no shape metadata — so the caller needs to know the expected dimensions.
"""

from std.ffi import external_call, c_int, c_size_t
from std.memory import UnsafePointer, bitcast
from std.collections import List

from lib.cuda import cuda_malloc, cuda_free, cuda_upload_u16, cuda_memcpy


comptime STREAM_CHUNK_BYTES = 64 * 1024 * 1024
comptime STAGING_GUARD_BYTES = 4096


def read_f32(path: String) -> List[Float32]:
    """Slurp a whole FP32 binary file into a host List. Returns empty on error.

    Uses a pread loop because Linux read() caps at ~2 GB (0x7ffff000) per
    syscall. The embed_tokens_weight.bin (5.6 GB) silently truncates if we
    use a single read() call — leaving the LM-head's upper rows zero and
    degenerating the logits to garbage. Same fingerprint as the 2026-05-06
    truncation bug; fix re-applied here 2026-05-20."""
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()): pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](
        pb.unsafe_ptr(), c_int(0), c_int(0))
    if Int(fd) < 0: return List[Float32]()
    var size = Int(external_call["nomos_lseek", c_size_t](
        fd, c_size_t(0), c_int(2)))
    _ = external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(0))
    if size <= 0 or (size % 4) != 0:
        _ = external_call["nomos_close", c_int](fd)
        return List[Float32]()
    var elems = size // 4
    if elems > 2147483647:
        print("[WARN] read_f32 refusing oversized host List:", path, " elems=", elems)
        _ = external_call["nomos_close", c_int](fd)
        return List[Float32]()
    # Reserve guard capacity behind the logical file body while we isolate the
    # two-engine load heap corruption. The returned length stays exactly elems.
    var r = List[Float32](capacity=elems + ((STAGING_GUARD_BYTES + 3) // 4))
    for _ in range(elems):
        r.append(0.0)
    # pread loop — keeps going until every byte is read or we hit EOF.
    # IMPORTANT: advance the buffer pointer by the file offset each round.
    # Forgetting this overwrites the start of the buffer with later file
    # bytes — corruption worse than the original truncation bug.
    var p_bytes = UnsafePointer(to=r[0]).bitcast[UInt8]()
    var offset: Int = 0
    var ok = True
    while offset < size:
        var want = size - offset
        if want > STREAM_CHUNK_BYTES:
            want = STREAM_CHUNK_BYTES
        var n_bytes = Int(external_call["nomos_pread", c_size_t](
            fd, (p_bytes + offset), c_size_t(want), c_size_t(offset)))
        if n_bytes <= 0 or n_bytes > want:
            ok = False
            break
        offset += n_bytes
    _ = external_call["nomos_close", c_int](fd)
    if (not ok) or offset != size:
        print("[WARN] short f32 read:", path, " bytes=", offset, " expected=", size)
        return List[Float32]()
    return r^


def load_f32_file_to_gpu(path: String) -> UInt64:
    """Stream a raw FP32 file directly to a device allocation.

    This avoids materializing very large files as `List[Float32]`.  E2B's
    per-layer input table is 2.35B floats (>2^31 elements), which is too large
    for the host List path and can corrupt allocator metadata before a later
    free detonates.
    """
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()):
        pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](
        pb.unsafe_ptr(), c_int(0), c_int(0)
    )
    if Int(fd) < 0:
        return UInt64(0)
    var size = Int(external_call["nomos_lseek", c_size_t](
        fd, c_size_t(0), c_int(2)
    ))
    _ = external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(0))
    if size <= 0 or (size % 4) != 0:
        _ = external_call["nomos_close", c_int](fd)
        return UInt64(0)

    var ptr = cuda_malloc(size)
    var chunk = STREAM_CHUNK_BYTES if size > STREAM_CHUNK_BYTES else size
    var buf = List[UInt8](capacity=chunk + STAGING_GUARD_BYTES)
    for _ in range(chunk):
        buf.append(0)
    var bp = buf.unsafe_ptr()

    var offset = 0
    var ok = True
    while offset < size:
        var want = size - offset
        if want > chunk:
            want = chunk
        var got = Int(external_call["nomos_pread", c_size_t](
            fd, bp, c_size_t(want), c_size_t(offset)
        ))
        if got <= 0 or got > want:
            ok = False
            break
        cuda_memcpy(ptr + UInt64(offset), UInt64(Int(bp)), got, 1)
        offset += got
    _ = external_call["nomos_close", c_int](fd)
    if (not ok) or offset != size:
        cuda_free(ptr)
        return UInt64(0)
    return ptr


def file_size_bytes(path: String) -> Int:
    """Return a file's byte size, or -1 if it cannot be opened/stat'd."""
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()):
        pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](
        pb.unsafe_ptr(), c_int(0), c_int(0)
    )
    if Int(fd) < 0:
        return -1
    var size = Int(external_call["nomos_lseek", c_size_t](
        fd, c_size_t(0), c_int(2)
    ))
    _ = external_call["nomos_close", c_int](fd)
    return size


def load_bf16_file_to_gpu(path: String) -> UInt64:
    """Stream a raw BF16 file directly to a device allocation.

    The E2B gather tables are large enough that bf16 must stay on the same
    chunked path as the fp32 form.  This loader expects already-packed
    little-endian BF16 values on disk and copies the bytes unchanged.
    """
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()):
        pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](
        pb.unsafe_ptr(), c_int(0), c_int(0)
    )
    if Int(fd) < 0:
        return UInt64(0)
    var size = Int(external_call["nomos_lseek", c_size_t](
        fd, c_size_t(0), c_int(2)
    ))
    _ = external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(0))
    if size <= 0 or (size % 2) != 0:
        _ = external_call["nomos_close", c_int](fd)
        return UInt64(0)

    var ptr = cuda_malloc(size)
    var chunk = STREAM_CHUNK_BYTES if size > STREAM_CHUNK_BYTES else size
    var buf = List[UInt8](capacity=chunk + STAGING_GUARD_BYTES)
    for _ in range(chunk):
        buf.append(0)
    var bp = buf.unsafe_ptr()

    var offset = 0
    var ok = True
    while offset < size:
        var want = size - offset
        if want > chunk:
            want = chunk
        var got = Int(external_call["nomos_pread", c_size_t](
            fd, bp, c_size_t(want), c_size_t(offset)
        ))
        if got <= 0 or got > want:
            ok = False
            break
        cuda_memcpy(ptr + UInt64(offset), UInt64(Int(bp)), got, 1)
        offset += got
    _ = external_call["nomos_close", c_int](fd)
    if (not ok) or offset != size:
        cuda_free(ptr)
        return UInt64(0)
    return ptr


def read_q4_bytes(path: String, mut hdr: List[Int]) -> List[UInt8]:
    """Read a .q4 file: 16-byte header (int64 n, int64 nb) then [fp16 scales][packed u8].
    Appends [n, nb] to `hdr`; returns the body bytes (scales+packed), header stripped.
    Empty list on error. Mirrors read_f32's pread loop (2 GB syscall cap safe)."""
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()): pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](pb.unsafe_ptr(), c_int(0), c_int(0))
    if Int(fd) < 0: return List[UInt8]()
    var size = Int(external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(2)))
    _ = external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(0))
    if size < 16:
        _ = external_call["nomos_close", c_int](fd)
        return List[UInt8]()
    # header (16 bytes = two int64, little-endian)
    var h = List[UInt8](capacity=16)
    for _ in range(16): h.append(0)
    var hread = Int(external_call["nomos_pread", c_size_t](
        fd, h.unsafe_ptr(), c_size_t(16), c_size_t(0)))
    if hread != 16:
        print("[WARN] short quant header read:", path, " bytes=", hread)
        _ = external_call["nomos_close", c_int](fd)
        return List[UInt8]()
    var hp = UnsafePointer(to=h[0]).bitcast[Int64]()
    hdr.append(Int(hp[0]))   # n
    hdr.append(Int(hp[1]))   # nb
    var n = Int(hp[0])
    var nb = Int(hp[1])
    if n < 0 or nb != ((n + 31) // 32):
        print("[WARN] bad quant header:", path, " n=", n, " nb=", nb)
        _ = external_call["nomos_close", c_int](fd)
        return List[UInt8]()
    # This reader is shared by q4 ([nb fp16 scales][nb*16 packed]) and
    # q8 ([nb fp16 scales][n int8 values]).
    var q4_body = nb * 18
    var q8_body = nb * 2 + n
    var body = size - 16
    if body != q4_body and body != q8_body:
        print("[WARN] bad quant body:", path, " bytes=", body,
              " q4_expected=", q4_body, " q8_expected=", q8_body)
        _ = external_call["nomos_close", c_int](fd)
        return List[UInt8]()
    # body (scales+packed/int8), read from file offset 16 — no copy
    var buf = List[UInt8](capacity=body + STAGING_GUARD_BYTES)
    for _ in range(body): buf.append(0)
    var p = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))
    var offset: Int = 0
    var ok = True
    while offset < body:
        var want = body - offset
        if want > STREAM_CHUNK_BYTES:
            want = STREAM_CHUNK_BYTES
        var nbytes = Int(external_call["nomos_pread", c_size_t](
            fd, (p + offset), c_size_t(want), c_size_t(16 + offset)))
        if nbytes <= 0 or nbytes > want:
            ok = False
            break
        offset += nbytes
    _ = external_call["nomos_close", c_int](fd)
    if (not ok) or offset != body:
        print("[WARN] short quant read:", path, " bytes=", offset, " expected=", body)
        return List[UInt8]()
    return buf^


def read_nvfp4_bytes(
    path: String, mut hdr: List[Int], mut gscale: List[Float32],
    mut input_gscale: List[Float32],
) -> List[UInt8]:
    """Read a .nvfp4 file: 24-byte header (int64 n, int64 nb, float32 global, float32 pad)
    then [e4m3 scales : nb][E2M1 packed : n/2]. Appends [n, nb] to `hdr` and [global] to
    `gscale` and the calibrated activation multiplier in the former pad word to
    `input_gscale`; returns the body bytes (scales+packed), header stripped. Empty on error."""
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()): pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](pb.unsafe_ptr(), c_int(0), c_int(0))
    if Int(fd) < 0: return List[UInt8]()
    var size = Int(external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(2)))
    _ = external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(0))
    if size < 24:
        _ = external_call["nomos_close", c_int](fd)
        return List[UInt8]()
    # header (24 bytes = two int64 + float32 global + float32 pad, little-endian)
    var h = List[UInt8](capacity=24)
    for _ in range(24): h.append(0)
    var hread = Int(external_call["nomos_pread", c_size_t](
        fd, h.unsafe_ptr(), c_size_t(24), c_size_t(0)))
    if hread != 24:
        print("[WARN] short nvfp4 header read:", path, " bytes=", hread)
        _ = external_call["nomos_close", c_int](fd)
        return List[UInt8]()
    var hp = UnsafePointer(to=h[0]).bitcast[Int64]()
    hdr.append(Int(hp[0]))   # n
    hdr.append(Int(hp[1]))   # nb
    var gp = UnsafePointer(to=h[16]).bitcast[Float32]()
    gscale.append(gp[0])     # per-tensor global scale
    input_gscale.append(gp[1])  # calibrated activation multiplier (1 / CT divisor)
    # body (scales+packed), from file offset 24
    var body = size - 24
    var buf = List[UInt8](capacity=body + STAGING_GUARD_BYTES)
    for _ in range(body): buf.append(0)
    var p = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))
    var offset: Int = 0
    var ok = True
    while offset < body:
        var want = body - offset
        if want > STREAM_CHUNK_BYTES:
            want = STREAM_CHUNK_BYTES
        var nbytes = Int(external_call["nomos_pread", c_size_t](
            fd, (p + offset), c_size_t(want), c_size_t(24 + offset)))
        if nbytes <= 0 or nbytes > want:
            ok = False
            break
        offset += nbytes
    _ = external_call["nomos_close", c_int](fd)
    if (not ok) or offset != body:
        print("[WARN] short nvfp4 read:", path, " bytes=", offset, " expected=", body)
        return List[UInt8]()
    return buf^


def load_to_gpu(path: String) -> UInt64:
    """Read a weight file off disk and upload it to a newly-allocated GPU buffer.
    Returns 0 on error (caller should treat as sentinel and skip the layer).
    """
    var ptr = load_f32_file_to_gpu(path)
    if ptr == 0:
        print("[WARN] Empty or missing:", path)
        return UInt64(0)
    return ptr


def load_to_gpu_bf16(path: String) -> UInt64:
    """Read an FP32 weight file, downcast to BF16, upload half the bytes.

    The Gemma-4 weights in the safetensors files were originally BF16 —
    the tools/gemma4_loader.go pipeline expanded them to FP32 binary files
    for simpler Mojo consumption. Since the memory bandwidth profile shows
    weight reads dominate the forward pass, we want to keep them as BF16
    on device. No precision is lost compared to the original model because
    the FP32 binary values all have zeros in the low 16 bits — we're just
    dropping the padding.

    Conversion: for each FP32 value, reinterpret the 4 bytes as a UInt32,
    shift right 16 bits, and take the top 16 bits as the BF16 value.
    This is "round toward zero" for the low bits, which matches the
    truncation the safetensors-to-FP32 pipeline implicitly did anyway
    (all dropped bits are zero, so the round direction doesn't matter).

    Returns a device pointer to the packed BF16 buffer (len * 2 bytes).
    """
    var data = read_f32(path)
    if len(data) == 0:
        print("[WARN] Empty or missing:", path)
        return UInt64(0)
    var n = len(data)
    var bf16 = List[UInt16](capacity=n)
    for i in range(n):
        # Use Mojo's SIMD cast for proper IEEE round-to-nearest-even.
        # (In practice the safetensors-to-FP32 pipeline already dropped
        # the low 16 bits as zeros, so this is equivalent to truncation
        # on our current weights — but it's the right behavior if we
        # ever load weights that actually have non-zero low bits.)
        var f_simd = SIMD[DType.float32, 1](data[i])
        var bf_simd = f_simd.cast[DType.bfloat16]()
        var bf_bits = bitcast[DType.uint16, 1](bf_simd)
        bf16.append(bf_bits[0])
    var ptr = cuda_malloc(n * 2)
    cuda_upload_u16(ptr, bf16)
    return ptr
