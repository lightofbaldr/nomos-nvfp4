"""I/O and copy primitives for the KV session store.

Run: bash tools/run_mojo_test.sh tools/kvstore_io_test.mojo
"""
from std.memory import UnsafePointer
from std.os import remove
from std.testing import assert_equal, assert_true, assert_false
from max.gpu.host import DeviceContext
from lib.io import read_f32, file_size_bytes
from lib.cuda import (
    cuda_malloc, cuda_free, cuda_host_alloc, cuda_free_host, cuda_stream_create, cuda_stream_destroy,
    cuda_event_create, cuda_event_destroy, cuda_event_record, cuda_event_sync,
    cuda_memcpy_async, cuda_memcpy_2d_async, cuda_memcpy_async_on, cuda_memcpy_2d_async_on,
)
from lib.kvstore.fileio import (
    O_RDWR, O_CREAT, open_file, close_file, pread_all, pwrite_all, fsync_file, file_size,
    truncate_file, try_lock_exclusive, random_u64,
)


def _bytes(addr: UInt64) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    return UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr))


def test_pinned_2d_round_trip() raises:
    """Pack tokens [2,5) of every head out of a head-first [3, 8, 4] device buffer, the way export will."""
    var heads = 3
    var cap = 8
    var row = 4
    var total = heads * cap * row
    var dev = cuda_malloc(total)
    var src = cuda_host_alloc(total)
    assert_true(src != 0)
    var sp = _bytes(src)
    for i in range(total):
        sp[i] = UInt8(i % 251)
    var stream = cuda_stream_create()
    var ev = cuda_event_create()
    assert_equal(cuda_memcpy_async_on(dev, src, total, 1, stream), 0)
    var packed = cuda_host_alloc(heads * 3 * row)
    assert_equal(cuda_memcpy_2d_async_on(packed, 3 * row, dev + UInt64(2 * row), cap * row, 3 * row, heads, 2, stream), 0)
    assert_equal(cuda_event_record(ev, stream), 0)
    assert_equal(cuda_event_sync(ev), 0)
    var pp = _bytes(packed)
    for h in range(heads):
        for t in range(3):
            for r in range(row):
                assert_equal(Int(pp[(h * 3 + t) * row + r]), Int(sp[(h * cap + 2 + t) * row + r]))
    cuda_free_host(packed)
    cuda_free_host(src)
    cuda_free(dev)
    cuda_event_destroy(ev)
    cuda_stream_destroy(stream)


def test_file_write_read_truncate() raises:
    var path = String("build/kvstore_io_test.bin")
    var fd = open_file(path, O_RDWR | O_CREAT, 420)  # 420 = 0o644
    assert_true(fd >= 0)
    var buf = List[UInt8](capacity=64)
    for i in range(64):
        buf.append(UInt8(i))
    pwrite_all(fd, UInt64(Int(buf.unsafe_ptr())), 64, 0)
    _ = buf^  # Int(ptr) drops the origin: with no later use Mojo frees buf BEFORE pwrite_all runs (zeroed first word)
    fsync_file(fd)
    assert_equal(file_size(fd), 64)
    var back = List[UInt8](length=64, fill=0)
    assert_true(pread_all(fd, UInt64(Int(back.unsafe_ptr())), 64, 0))
    for i in range(64):
        assert_equal(Int(back[i]), i)
    assert_false(pread_all(fd, UInt64(Int(back.unsafe_ptr())), 64, 32))  # short read past the end
    _ = back^  # same reason: keep back alive across the last call that writes through its address
    truncate_file(fd, 10)
    assert_equal(file_size(fd), 10)
    close_file(fd)
    remove(path)


def test_lock_is_exclusive() raises:
    var path = String("build/kvstore_io_test.lock")
    var a = open_file(path, O_RDWR | O_CREAT, 420)
    var b = open_file(path, O_RDWR | O_CREAT, 420)
    assert_true(try_lock_exclusive(a))
    assert_false(try_lock_exclusive(b))  # flock locks belong to the open file description
    close_file(a)
    assert_true(try_lock_exclusive(b))
    close_file(b)
    remove(path)


def test_random_ids_differ() raises:
    assert_true(random_u64() != random_u64())


def test_coexists_with_kernel_declarations() raises:
    """One program holds lib.io, the DeviceContext copy wrappers and the new primitives, as the kernel does.

    The kernel already declares nomos_pread (pointer buffer), cudaMemcpyAsync and cudaMemcpy2DAsync (CUstream). A second
    declaration of one of these symbols with other operand types does not fail one test: it fails the whole kernel build.
    """
    # Shim: lib.io's nomos_open/lseek/pread/close next to fileio's.
    assert_equal(len(read_f32(String("build/kvstore_io_missing.f32"))), 0)  # a missing file reads as empty
    var path = String("build/kvstore_io_coexist.f32")
    var fd = open_file(path, O_RDWR | O_CREAT, 420)
    assert_true(fd >= 0)
    var vals = List[Float32](capacity=4)
    vals.append(1.5)
    vals.append(2.5)
    vals.append(-3.0)
    vals.append(4.25)
    pwrite_all(fd, UInt64(Int(vals.unsafe_ptr())), 16, 0)
    _ = vals^
    assert_equal(file_size(fd), 16)
    assert_equal(file_size_bytes(path), file_size(fd))
    var floats = read_f32(path)  # lib.io's pread loop reads what fileio's pwrite wrote
    assert_equal(len(floats), 4)
    assert_equal(floats[0], 1.5)
    assert_equal(floats[1], 2.5)
    assert_equal(floats[2], -3.0)
    assert_equal(floats[3], 4.25)
    close_file(fd)
    remove(path)

    # CUDA: the DeviceContext-stream wrappers next to the explicit-stream ones.
    var ctx = DeviceContext()
    var host = cuda_host_alloc(64)
    var back = cuda_host_alloc(64)
    assert_true(host != 0 and back != 0)
    var hp = _bytes(host)
    var bp = _bytes(back)
    for i in range(64):
        hp[i] = UInt8(i + 1)
        bp[i] = 0
    var dev = cuda_malloc(64)
    var stream = cuda_stream_create()
    var ev = cuda_event_create()
    cuda_memcpy_async(dev, host, 64, 1, ctx)  # H2D on the DeviceContext's stream
    ctx.synchronize()
    assert_equal(cuda_memcpy_async_on(back, dev, 64, 2, stream), 0)  # D2H on our own stream
    assert_equal(cuda_event_record(ev, stream), 0)
    assert_equal(cuda_event_sync(ev), 0)
    for i in range(64):
        assert_equal(Int(bp[i]), i + 1)
    for i in range(64):
        bp[i] = 0
    cuda_memcpy_2d_async(dev, 16, host, 8, 8, 4, 1, ctx)  # H2D: 4 rows of 8 bytes, packed source -> pitch 16
    ctx.synchronize()
    assert_equal(cuda_memcpy_2d_async_on(back, 8, dev, 16, 8, 4, 2, stream), 0)  # D2H: pitch 16 -> packed
    assert_equal(cuda_event_record(ev, stream), 0)
    assert_equal(cuda_event_sync(ev), 0)
    for i in range(32):
        assert_equal(Int(bp[i]), i + 1)
    cuda_event_destroy(ev)
    cuda_stream_destroy(stream)
    cuda_free(dev)
    cuda_free_host(back)
    cuda_free_host(host)


def main() raises:
    test_pinned_2d_round_trip()
    test_file_write_read_truncate()
    test_lock_is_exclusive()
    test_random_ids_differ()
    test_coexists_with_kernel_declarations()
    print("kvstore_io_test: all passed")
