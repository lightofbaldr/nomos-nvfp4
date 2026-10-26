"""File I/O for the KV session store over the repo's C libc shim (nomos_libc_shim.c), as lib/io.mojo does."""
from std.ffi import external_call, c_int, c_size_t

comptime O_RDONLY = 0
comptime O_RDWR = 2
comptime O_CREAT = 64
comptime _SEEK_END = 2
comptime _LOCK_EX = 2
comptime _LOCK_NB = 4


def _cpath(path: String) -> List[UInt8]:
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()):
        pb.append(path.as_bytes()[i])
    pb.append(0)
    return pb^


def open_file(path: String, flags: Int, mode: Int) -> Int:
    var pb = _cpath(path)
    return Int(external_call["nomos_open", c_int](pb.unsafe_ptr(), c_int(flags), c_int(mode)))


def close_file(fd: Int):
    _ = external_call["nomos_close", c_int](c_int(fd))


def pread_all(fd: Int, addr: UInt64, n: Int, off: Int) -> Bool:
    """Read exactly n bytes at off into addr. False on a short read or an error.

    addr is a raw address, so Mojo cannot see it in use: keep its owner alive until this returns (Mojo frees a List
    right after its last use, which can be BEFORE the call runs, e.g. `_ = buf^` after it)."""
    var done = 0
    while done < n:
        var buf = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr) + done)
        var got = external_call["nomos_pread", Int](c_int(fd), buf, c_size_t(n - done), c_size_t(off + done))
        if got <= 0:
            return False
        done += got
    return True


def pwrite_all(fd: Int, addr: UInt64, n: Int, off: Int) raises:
    """Write exactly n bytes from addr at off. Raises on an error. addr must stay valid until this returns (see pread_all)."""
    var done = 0
    while done < n:
        var buf = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr) + done)
        var put = external_call["nomos_pwrite", Int](c_int(fd), buf, c_size_t(n - done), c_size_t(off + done))
        if put <= 0:
            raise Error("pwrite failed at offset " + String(off + done))
        done += put


def fsync_file(fd: Int) raises:
    if Int(external_call["nomos_fsync", c_int](c_int(fd))) != 0:
        raise Error("fsync failed")


def file_size(fd: Int) -> Int:
    return Int(external_call["nomos_lseek", Int](c_int(fd), c_size_t(0), c_int(_SEEK_END)))


def truncate_file(fd: Int, size: Int) raises:
    if Int(external_call["nomos_ftruncate", c_int](c_int(fd), c_size_t(size))) != 0:
        raise Error("ftruncate failed")


def try_lock_exclusive(fd: Int) -> Bool:
    return Int(external_call["nomos_flock", c_int](c_int(fd), c_int(_LOCK_EX | _LOCK_NB))) == 0


def random_u64() raises -> UInt64:
    var v: UInt64 = 0
    var got = external_call["nomos_getrandom", Int](UnsafePointer(to=v), c_size_t(8), c_int(0))
    if got != 8:
        raise Error("getrandom failed")
    return v
