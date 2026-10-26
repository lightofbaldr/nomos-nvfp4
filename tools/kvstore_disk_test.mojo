"""Unit tests for the KV session store NVMe tier (CPU only; heap buffers stand in for pinned ones).

Run: bash tools/run_mojo_test.sh tools/kvstore_disk_test.mojo
"""
from std.ffi import external_call, c_int
from std.memory import UnsafePointer
from std.os import listdir, makedirs, remove, rmdir, stat
from std.os.path import exists, isdir
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from lib.kvstore.disk_tier import DiskTier, ByteWriter
from lib.kvstore.fileio import (
    O_RDWR, O_CREAT, open_file, close_file, pread_all, pwrite_all, file_size, truncate_file, random_u64,
)
from lib.kvstore.hashing import hex16
from lib.kvstore.layout import KvLayout, ComponentDesc, KIND_K_DATA, KIND_DRAFT_K
from lib.kvstore.key import CacheKey
from lib.kvstore.index import ChunkInfo
from lib.kvstore.index import SessionRecord


def _layout() -> KvLayout:
    var lay = KvLayout()
    lay.comps.append(ComponentDesc(KIND_K_DATA, 0, 2, 3))
    lay.comps.append(ComponentDesc(KIND_DRAFT_K, 0, 1, 4))
    return lay^          # 10 bytes per token


def _key(mode: Int) -> CacheKey:
    return CacheKey(2, UInt64(0xABC), mode, _layout().hash(), UInt64(0), UInt64(0))


comptime _ROOT = "build/kvstore_disk_test"


def _dir() raises -> String:
    return _ROOT + "/" + hex16(random_u64())


def _rmtree(path: String):
    """Remove a directory tree with std.os only. Best effort: a directory that cannot be listed or emptied stays."""
    try:
        var names = listdir(path)
        for i in range(len(names)):
            var child = path + "/" + names[i]
            if isdir(child):
                _rmtree(child)
            else:
                remove(child)
        rmdir(path)
    except:
        pass


def _clear_root():
    """Every test makes its own directory under _ROOT. main() clears it before the first test, for what an earlier failed
    run left as evidence, and after the last, so a passing run leaves nothing. Run one disk-test process at a time."""
    if exists(_ROOT):
        _rmtree(_ROOT)


def _payload(ntok: Int, salt: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(ntok * 10):
        out.append(UInt8((i * 13 + salt) % 256))
    return out^


def _toks(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(a, b):
        out.append(1000 + i)
    return out^


def _addr(b: List[UInt8]) -> UInt64:
    return UInt64(Int(b.unsafe_ptr()))


def _write_three(mut d: DiskTier, id: UInt64) raises -> List[Int]:
    d.create(id)
    var offs = List[Int]()
    var p0 = _payload(5, 1)
    offs.append(d.append(id, 0, 5, 5, _toks(0, 5), _addr(p0), len(p0)))
    _ = p0^
    var p1 = _payload(4, 2)
    offs.append(d.append(id, 5, 9, 5, _toks(5, 9), _addr(p1), len(p1)))      # no drafter rows in this chunk
    _ = p1^
    var p2 = _payload(11, 3)
    offs.append(d.append(id, 9, 20, 20, _toks(9, 20), _addr(p2), len(p2)))
    _ = p2^
    return offs^


def test_scan_rebuilds_sessions() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(42))
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), 1)
    assert_equal(recs[0].id, UInt64(42))
    assert_equal(recs[0].kv_len(), 20)
    assert_equal(len(recs[0].chunks), 3)
    assert_equal(recs[0].chunks[2].file_offset, offs[2])
    assert_equal(recs[0].drafter_len, 5)          # the gap in chunk 2 stops the drafter prefix
    assert_true(recs[0].on_disk)
    d2.close()


def test_read_payload_round_trip_and_checksum() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(7))
    var back = List[UInt8](length=40, fill=0)
    d.read_payload(UInt64(7), ChunkInfo(5, 9, 5, offs[1], UInt64(0)), _toks(5, 9), _addr(back))
    var want = _payload(4, 2)
    for i in range(40):
        assert_equal(Int(back[i]), Int(want[i]))
    var fd = open_file(d.path(UInt64(7)), O_RDWR, 0)
    var bad = List[UInt8]()
    bad.append(UInt8(0xFF))
    pwrite_all(fd, _addr(bad), 1, offs[1] + 3)           # corrupt one payload byte
    _ = bad^
    close_file(fd)
    with assert_raises():
        d.read_payload(UInt64(7), ChunkInfo(5, 9, 5, offs[1], UInt64(0)), _toks(5, 9), _addr(back))
    _ = back^
    d.close()


def test_torn_tail_is_cut_back() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(9))
    var fd = open_file(d.path(UInt64(9)), O_RDWR, 0)
    truncate_file(fd, offs[2] + 50)                        # the last record is half written
    close_file(fd)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(recs[0].kv_len(), 9)
    var fd2 = open_file(d2.path(UInt64(9)), O_RDWR, 0)
    assert_equal(file_size(fd2), offs[1] + 40 + 4)        # cut at the end of chunk 1's record
    close_file(fd2)
    d2.close()


def test_other_keys_are_left_alone() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(11))
    d.close()
    var d2 = DiskTier(dir, _key(2), _layout())
    assert_equal(len(d2.scan()), 0)
    var names = listdir(dir)
    var found = False
    for i in range(len(names)):
        if names[i] == hex16(UInt64(11)) + ".nkvs":
            found = True
    assert_true(found)
    d2.close()


def test_other_format_versions_are_left_alone() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(17))
    var fd = open_file(d.path(UInt64(17)), O_RDWR, 0)
    var v = List[UInt8]()
    v.append(UInt8(2))
    pwrite_all(fd, _addr(v), 1, 4)                         # the u32 format version is header bytes 4-7
    _ = v^
    close_file(fd)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    assert_equal(len(d2.scan()), 0)                        # skipped, and the file is not deleted
    var names = listdir(dir)
    var found = False
    for i in range(len(names)):
        if names[i] == hex16(UInt64(17)) + ".nkvs":
            found = True
    assert_true(found)
    d2.close()


def test_lock_refuses_a_second_store() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    with assert_raises():
        _ = DiskTier(dir, _key(1), _layout())
    d.close()


def test_drop_and_total_bytes() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(13))
    assert_true(d.total_bytes() > 200)
    d.drop(UInt64(13))
    assert_equal(d.total_bytes(), 0)
    d.close()


def test_fail_writes_hook() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.create(UInt64(15))
    d.fail_writes = True
    var p = _payload(2, 9)
    with assert_raises():
        _ = d.append(UInt64(15), 0, 2, 2, _toks(0, 2), _addr(p), len(p))
    _ = p^
    d.close()


# ── Beyond the brief's eight tests: the recovery paths they do not reach ─────────────────────────────────────────────
# _write_three leaves three records: tokens [0,5) [5,9) [9,20), payloads of 50, 40 and 110 bytes. A record is
# 16 header bytes, 4 per token, 8 for the payload length, the payload, then 4 for its CRC.


def _other_layout() -> KvLayout:
    var lay = KvLayout()
    lay.comps.append(ComponentDesc(KIND_K_DATA, 0, 2, 3))      # DFlash off: the drafter's component is gone
    return lay^


def _rec_start(payload_off: Int, ntok: Int) -> Int:
    return payload_off - (16 + 4 * ntok + 8)


def _size_of(path: String) -> Int:
    var fd = open_file(path, O_RDWR, 0)
    var n = file_size(fd)
    close_file(fd)
    return n


def _has_file(dir: String, id: UInt64) raises -> Bool:
    var names = listdir(dir)
    for i in range(len(names)):
        if names[i] == hex16(id) + ".nkvs":
            return True
    return False


def _put(path: String, off: Int, v: Int) raises:
    """Overwrite one byte of a file."""
    var fd = open_file(path, O_RDWR, 0)
    var b = List[UInt8]()
    b.append(UInt8(v))
    pwrite_all(fd, _addr(b), 1, off)
    _ = b^
    close_file(fd)


def _flip(path: String, off: Int) raises:
    """Invert one byte of a file, so the byte is certain to change."""
    var fd = open_file(path, O_RDWR, 0)
    var b = List[UInt8](length=1, fill=0)
    assert_true(pread_all(fd, _addr(b), 1, off))
    b[0] = ~b[0]
    pwrite_all(fd, _addr(b), 1, off)
    _ = b^
    close_file(fd)


def _write_bytes(path: String, data: List[UInt8]) raises:
    var fd = open_file(path, O_RDWR | O_CREAT, 420)
    assert_true(fd >= 0)
    pwrite_all(fd, _addr(data), len(data), 0)      # data is borrowed: the caller keeps it alive for the whole call
    close_file(fd)


def _write_junk(path: String, n: Int) raises:
    var junk = List[UInt8]()
    for i in range(n):
        junk.append(UInt8((i * 7 + 1) % 256))
    _write_bytes(path, junk)


# Real failures, for the tests below. A file-size limit makes a write fail halfway (EFBIG, after writing up to the
# limit); an address-space limit makes an oversized allocation fail (which aborts the process). These three C symbols
# are declared here and nowhere else in the repo, so they cannot clash with the kernel's declarations.
comptime _RLIMIT_FSIZE = 1      # Linux, x86_64 and aarch64
comptime _RLIMIT_AS = 9
comptime _SIGXFSZ = 25
comptime _SIG_IGN = 1


def _rlimit(resource: Int) raises -> List[UInt64]:
    """[soft, hard] of one resource limit."""
    var lim = List[UInt64](length=2, fill=0)
    if Int(external_call["getrlimit", c_int](c_int(resource), lim.unsafe_ptr())) != 0:
        raise Error("getrlimit failed")
    return lim^


def _set_rlimit(resource: Int, soft: UInt64, hard: UInt64) raises:
    var lim = List[UInt64](length=2, fill=0)
    lim[0] = soft
    lim[1] = hard
    if Int(external_call["setrlimit", c_int](c_int(resource), lim.unsafe_ptr())) != 0:
        raise Error("setrlimit failed")


def _signal(sig: Int, handler: Int) -> Int:
    """signal(2). Returns the previous handler: an address, or 0 (SIG_DFL), or 1 (SIG_IGN)."""
    return Int(external_call["signal", Int](c_int(sig), handler))


def _cap_file_size(limit: Int) raises -> List[UInt64]:
    """From now on a write that would take a file past `limit` bytes writes up to the limit, then fails with EFBIG.
    SIGXFSZ, which ends the process by default, is ignored. Returns what _lift_file_size_cap puts back. Nothing may
    print while the cap is on: stdout can be a file."""
    var saved = _rlimit(_RLIMIT_FSIZE)
    saved.append(UInt64(_signal(_SIGXFSZ, _SIG_IGN)))
    _set_rlimit(_RLIMIT_FSIZE, UInt64(limit), saved[1])
    return saved^


def _lift_file_size_cap(saved: List[UInt64]) raises:
    _set_rlimit(_RLIMIT_FSIZE, saved[0], saved[1])
    _ = _signal(_SIGXFSZ, Int(saved[2]))


def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _vm_bytes() raises -> Int:
    """The address space this process holds now (VmSize in /proc/self/status). It is large: every binary that
    run_mojo_test.sh builds links nomos_cuda_budget.o, whose constructor creates CUDA events at load (about 15 GiB)."""
    var text = _read_text("/proc/self/status")
    var at = text.find("VmSize:")
    assert_true(at >= 0)
    var b = text.as_bytes()
    var i = at + 7
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    var kb = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        kb = kb * 10 + Int(b[i]) - 48
        i += 1
    return kb * 1024


def _scan_capped(mut d: DiskTier, headroom: Int) raises -> List[SessionRecord]:
    """scan() with the address space capped `headroom` bytes above what the process holds now, lifted straight after.
    A scan needs a few KB. A buffer sized from a corrupt record header needs gigabytes, and an allocation that fails
    aborts the process, so a scan that trusts such a header kills the test instead of passing quietly. The cap is
    relative because of the 15 GiB above, and no CUDA call is made while it is set."""
    var old = _rlimit(_RLIMIT_AS)
    var cap = UInt64(_vm_bytes() + headroom)
    if cap > old[1]:
        cap = old[1]
    _set_rlimit(_RLIMIT_AS, cap, old[1])
    try:
        return d.scan()
    finally:
        _set_rlimit(_RLIMIT_AS, old[0], old[1])


def test_bad_checksum_on_the_last_record_is_cut_back() raises:
    """A last record of full length whose bytes are wrong (not merely short) is cut like a torn one."""
    for variant in range(3):
        var dir = _dir()
        var d = DiskTier(dir, _key(1), _layout())
        var offs = _write_three(d, UInt64(21))
        var at = offs[2] + 55                                   # a payload byte
        if variant == 1:
            at = offs[2] + 110                                  # a byte of the stored CRC
        elif variant == 2:
            at = _rec_start(offs[2], 11) + 16 + 5               # a token byte: the CRC covers the tokens too
        _flip(d.path(UInt64(21)), at)
        d.close()
        var d2 = DiskTier(dir, _key(1), _layout())
        var recs = d2.scan()
        assert_equal(len(recs), 1)
        assert_equal(recs[0].kv_len(), 9)
        assert_equal(_size_of(d2.path(UInt64(21))), offs[1] + 40 + 4)
        d2.close()


def test_every_kind_of_torn_tail_is_cut_back() raises:
    """The file ends inside each part of the last record: header, tokens, payload length, payload, CRC."""
    var into: List[Int] = [1, 15, 16, 30, 60, 67, 68, 69, 177, 178, 181]   # bytes of the last record that survive
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var rec2 = 0
    for i in range(len(into)):
        var offs = _write_three(d, UInt64(100 + i))
        rec2 = _rec_start(offs[2], 11)
        var fd = open_file(d.path(UInt64(100 + i)), O_RDWR, 0)
        truncate_file(fd, rec2 + into[i])
        close_file(fd)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), len(into))
    for i in range(len(recs)):
        assert_equal(recs[i].kv_len(), 9)
        assert_equal(_size_of(d2.path(recs[i].id)), rec2)
    d2.close()


def test_corrupt_record_headers_are_cut_back_without_trusting_them() raises:
    """Bad bytes in the last record's fixed fields. A corrupt token count must not size a buffer: with b's top byte
    set the record claims 4.3 billion tokens, a 17 GB allocation. The scan runs under an address-space cap (1 GiB
    above the process's current size), so a scan that trusts b aborts the test instead of failing a count afterwards."""
    var what: List[String] = ["magic", "a (a gap)", "b == a", "b huge", "drafter_end past b", "drafter_end before a", "payload length"]
    var at: List[Int] = [0, 4, 8, 11, 12, 12, 60]        # byte of the record header to overwrite
    var to: List[Int] = [0, 10, 9, 0xFF, 21, 8, 0]       # the record is a=9, b=20, drafter_end=20, payload length 110
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var rec2 = 0
    for i in range(len(at)):
        var offs = _write_three(d, UInt64(200 + i))
        rec2 = _rec_start(offs[2], 11)
        _put(d.path(UInt64(200 + i)), rec2 + at[i], to[i])
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = _scan_capped(d2, 1024 * 1024 * 1024)       # a scan that sizes a buffer from "b huge" aborts here
    assert_equal(len(recs), len(at))
    for i in range(len(recs)):
        var which = Int(recs[i].id) - 200
        assert_equal(recs[i].kv_len(), 9, msg=what[which])
        assert_equal(_size_of(d2.path(recs[i].id)), rec2, msg=what[which])
    d2.close()


def test_nothing_after_a_bad_record_is_trusted() raises:
    """Records cannot be found without their predecessors: a record in the middle that does not continue the chain
    cuts the file there, even when it is well formed in itself (still four tokens, so only the chain check sees it)."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(33))
    var rec1 = _rec_start(offs[1], 4)
    _put(d.path(UInt64(33)), rec1 + 4, 6)                 # a: chunk 1 claims to start at token 6, not 5
    _put(d.path(UInt64(33)), rec1 + 8, 10)                # b: and to end at 10, so it still holds four tokens
    _put(d.path(UInt64(33)), rec1 + 12, 6)                # drafter_end: back inside [a, b]
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(recs[0].kv_len(), 5)
    assert_equal(len(recs[0].chunks), 1)
    assert_equal(_size_of(d2.path(UInt64(33))), rec1)     # the intact third record went with it
    d2.close()


def test_other_layouts_are_left_alone() raises:
    """DFlash on or off changes the layout and its hash in the key; the other layout's files are skipped and kept."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(29))
    d.close()
    var other_key = CacheKey(2, UInt64(0xABC), 1, _other_layout().hash(), UInt64(0), UInt64(0))
    var d2 = DiskTier(dir, other_key, _other_layout())
    assert_equal(len(d2.scan()), 0)
    d2.close()
    var d3 = DiskTier(dir, _key(1), _other_layout())      # an equal key, but a different layout table
    assert_equal(len(d3.scan()), 0)
    d3.close()
    assert_true(_has_file(dir, UInt64(29)))
    var d4 = DiskTier(dir, _key(1), _layout())            # and its owner still finds it whole
    var recs = d4.scan()
    assert_equal(len(recs), 1)
    assert_equal(recs[0].kv_len(), 20)
    d4.close()


def test_files_that_are_not_sessions_are_ignored() raises:
    """Junk shorter and longer than a header, and unrelated names, beside one real session."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(34))
    var real = _size_of(d.path(UInt64(34)))
    _write_junk(d.path(UInt64(35)), 10)
    _write_junk(d.path(UInt64(36)), 500)
    _write_junk(dir + "/notes.txt", 20)
    _write_junk(dir + "/not-a-session-id.nkvs", 20)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), 1)
    assert_equal(recs[0].id, UInt64(34))
    assert_equal(d2.total_bytes(), real)                  # skipped files count against no budget
    assert_true(_has_file(dir, UInt64(35)))
    assert_true(_has_file(dir, UInt64(36)))
    assert_equal(_size_of(d2.path(UInt64(35))), 10)
    assert_equal(_size_of(d2.path(UInt64(36))), 500)
    d2.close()


def test_a_session_without_records_is_removed() raises:
    """A crash between create and the first append leaves only a header: nothing to restore, so scan deletes it."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.create(UInt64(26))
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    assert_equal(len(d2.scan()), 0)
    assert_false(_has_file(dir, UInt64(26)))
    assert_equal(d2.total_bytes(), 0)
    d2.close()


def test_append_after_a_restart_extends_the_session() raises:
    """What the store does after startup: scan, then append to a session found on disk."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(28))
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    _ = d2.scan()
    var p = _payload(3, 4)
    var off = d2.append(UInt64(28), 20, 23, 20, _toks(20, 23), _addr(p), len(p))
    _ = p^
    assert_equal(d2.total_bytes(), _size_of(d2.path(UInt64(28))))
    d2.close()
    var d3 = DiskTier(dir, _key(1), _layout())
    var recs = d3.scan()
    assert_equal(recs[0].kv_len(), 23)
    assert_equal(len(recs[0].chunks), 4)
    assert_equal(recs[0].chunks[3].file_offset, off)
    var back = List[UInt8](length=30, fill=0)
    d3.read_payload(UInt64(28), recs[0].chunks[3], _toks(20, 23), _addr(back))   # read through scan's ChunkInfo
    var want = _payload(3, 4)
    for i in range(30):
        assert_equal(Int(back[i]), Int(want[i]))
    _ = back^
    d3.close()


def test_older_records_are_checked_when_read_not_at_scan() raises:
    """Startup CRC-checks only the last record (plan note 5); an older bad payload is caught when it is read."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(37))
    _flip(d.path(UInt64(37)), offs[0] + 2)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs[0].chunks), 3)
    var back = List[UInt8](length=50, fill=0)
    with assert_raises():
        d2.read_payload(UInt64(37), recs[0].chunks[0], _toks(0, 5), _addr(back))
    _ = back^
    d2.close()


def test_a_refused_second_store_does_not_release_the_lock() raises:
    """A flock lock belongs to the open file, so a refused attempt closing its own descriptor must not unlock the first."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    with assert_raises(contains="in use by another process"):     # the text the store logs when it disables itself
        _ = DiskTier(dir, _key(1), _layout())
    with assert_raises(contains="in use by another process"):
        _ = DiskTier(dir, _key(1), _layout())
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())            # close() does release it
    d2.close()


def _nibble(c: UInt8) -> Int:
    var v = Int(c)
    return v - 48 if v <= 57 else v - 87


def _unhex(h: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = h.as_bytes()
    for i in range(0, len(b), 2):
        out.append(UInt8(_nibble(b[i]) * 16 + _nibble(b[i + 1])))
    return out^


def _golden_session() -> List[UInt8]:
    """The 144-byte file for the key and session of test_file_bytes_are_the_documented_format: an 88-byte header and
    one 56-byte record. It was made by an independent implementation of the format (the module docstring of
    disk_tier.mojo), with a bitwise CRC-32C and the layout hash recomputed from KvLayout.hash()'s definition."""
    var h = String()
    h += "4e4b5653" + "01000000" + "02000000" + "03000000" + "0807060504030201"                # NKVS, v1, model 2, mode 3, build_id
    h += "87e9b98f30460930" + "1817161514131211" + "2827262524232221"                          # layout_hash, variant_hash, drafter_id
    h += "02000000"                                                                           # 2 components
    h += "00000000" + "00000000" + "02000000" + "03000000"                                    #   K data: layer 0, 2 heads, 3 B/row
    h += "06000000" + "00000000" + "01000000" + "04000000"                                    #   draft K: layer 0, 1 head, 4 B/row
    h += "a1efaecf"                                                                           # CRC-32C of the 84 bytes above
    h += "43484e4b" + "00000000" + "02000000" + "01000000"                                    # CHNK, a=0, b=2, drafter_end=1
    h += "04030201" + "efcdab00" + "1400000000000000"                                         # tokens 0x01020304 0x00ABCDEF, payload_len 20
    h += "0714212e3b4855626f7c8996a3b0bdcad7e4f1fe"                                           # payload: byte i = (13 i + 7) mod 256
    h += "b31667ad"                                                                           # CRC-32C of the token bytes then the payload
    return _unhex(h)


def _golden_key() -> CacheKey:
    return CacheKey(2, UInt64(0x0102030405060708), 3, _layout().hash(), UInt64(0x1112131415161718), UInt64(0x2122232425262728))


def test_file_bytes_are_the_documented_format() raises:
    """The format is permanent on disk, so pin every byte. A change that is consistent on both sides (field order,
    widths, byte order, a magic, the version, the layout hash, the CRC) passes every other test and orphans old files."""
    var want = _golden_session()
    assert_equal(len(want), 144)
    var dir = _dir()
    var d = DiskTier(dir, _golden_key(), _layout())
    d.create(UInt64(0xAB))
    var toks = List[Int]()
    toks.append(0x01020304)
    toks.append(0x00ABCDEF)
    var p = _payload(2, 7)
    var off = d.append(UInt64(0xAB), 0, 2, 1, toks, _addr(p), len(p))
    _ = p^
    assert_equal(off, 88 + 16 + 8 + 8)
    assert_equal(_size_of(d.path(UInt64(0xAB))), len(want))
    var fd = open_file(d.path(UInt64(0xAB)), O_RDWR, 0)
    var got = List[UInt8](length=len(want), fill=0)
    assert_true(pread_all(fd, _addr(got), len(want), 0))
    for i in range(len(want)):
        assert_equal(Int(got[i]), Int(want[i]), msg="byte " + String(i))
    _ = got^
    close_file(fd)
    d.close()
    # and the other direction: a file written by the independent implementation is read back whole
    var dir2 = _dir()
    var d2 = DiskTier(dir2, _golden_key(), _layout())
    _write_bytes(d2.path(UInt64(0xCD)), want)
    var recs = d2.scan()
    assert_equal(len(recs), 1)
    assert_equal(recs[0].id, UInt64(0xCD))
    assert_equal(len(recs[0].tokens), 2)
    assert_equal(recs[0].tokens[0], 0x01020304)
    assert_equal(recs[0].tokens[1], 0x00ABCDEF)
    assert_equal(recs[0].drafter_len, 1)
    assert_equal(recs[0].chunks[0].file_offset, 120)
    var back = List[UInt8](length=20, fill=0)
    d2.read_payload(UInt64(0xCD), recs[0].chunks[0], recs[0].tokens, _addr(back))
    var pay = _payload(2, 7)
    for i in range(20):
        assert_equal(Int(back[i]), Int(pay[i]))
    _ = back^
    d2.close()


def test_scan_returns_exactly_what_was_written() raises:
    """Every token, every chunk bound and every payload of a three-record session, after a restart."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(38))
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), 1)
    assert_equal(len(recs[0].tokens), 20)
    for i in range(20):
        assert_equal(recs[0].tokens[i], 1000 + i)
    var starts: List[Int] = [0, 5, 9]
    var ends: List[Int] = [5, 9, 20]
    var dends: List[Int] = [5, 5, 20]
    var salts: List[Int] = [1, 2, 3]
    for i in range(3):
        var ch = recs[0].chunks[i]
        assert_equal(ch.start, starts[i])
        assert_equal(ch.end, ends[i])
        assert_equal(ch.drafter_end, dends[i])
        assert_equal(ch.file_offset, offs[i])
        assert_equal(ch.ram_addr, UInt64(0))
        var back = List[UInt8](length=10 * (ends[i] - starts[i]), fill=0)
        d2.read_payload(UInt64(38), ch, _toks(starts[i], ends[i]), _addr(back))
        var want = _payload(ends[i] - starts[i], salts[i])
        for k in range(len(want)):
            assert_equal(Int(back[k]), Int(want[k]))
        _ = back^
    d2.close()


def test_fail_writes_blocks_create_too() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.fail_writes = True
    with assert_raises():
        d.create(UInt64(16))
    assert_false(_has_file(dir, UInt64(16)))
    assert_equal(d.total_bytes(), 0)
    d.fail_writes = False
    d.create(UInt64(16))                                  # the hook is only a switch
    assert_true(_has_file(dir, UInt64(16)))
    d.close()


def test_append_never_creates_a_session() raises:
    """A record in a file without a header could never be read back, so append must not make one."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var p = _payload(2, 9)
    with assert_raises():
        _ = d.append(UInt64(99), 0, 2, 2, _toks(0, 2), _addr(p), len(p))
    _ = p^
    assert_false(_has_file(dir, UInt64(99)))
    assert_equal(d.total_bytes(), 0)
    d.close()


def test_a_payload_over_the_crc_slice_is_verified_across_slices() raises:
    """Real chunks dwarf the 16 MiB slice scan checks the last record in (4096 tokens with DFlash is about 267 MB), so
    every real restart takes the multi-slice path. Here one record of 1.7 million tokens: 17 MB of payload, two slices."""
    var ntok = 1700000
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.create(UInt64(40))
    var p = _payload(ntok, 5)
    var off = d.append(UInt64(40), 0, ntok, ntok, _toks(0, ntok), _addr(p), len(p))
    _ = p^
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), 1)                                 # intact: the CRC over both slices matched
    assert_equal(recs[0].kv_len(), ntok)
    assert_equal(recs[0].chunks[0].file_offset, off)
    _flip(d2.path(UInt64(40)), off + 16 * 1024 * 1024 + 100)   # a payload byte in the second slice
    d2.close()
    var d3 = DiskTier(dir, _key(1), _layout())
    assert_equal(len(d3.scan()), 0)                            # the only record is bad, so the session goes
    assert_false(_has_file(dir, UInt64(40)))
    d3.close()


def test_an_empty_chunk_is_not_a_chunk() raises:
    """A record of no tokens would hand the store a chunk of no bytes to restore, so scan cuts the file before it."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(41))
    var end = offs[2] + 110 + 4
    var w = ByteWriter()                                  # built by hand, so this does not depend on what append accepts
    w.u32(0x4B4E4843)                                     # "CHNK"
    w.u32(20)
    w.u32(20)
    w.u32(20)
    w.u64(0)                                              # payload length
    w.u32(0)                                              # CRC-32C of nothing
    var fd = open_file(d.path(UInt64(41)), O_RDWR, 0)
    pwrite_all(fd, w.addr(), w.size(), end)
    _ = w^
    close_file(fd)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(recs[0].kv_len(), 20)
    assert_equal(len(recs[0].chunks), 3)
    assert_equal(_size_of(d2.path(UInt64(41))), end)
    d2.close()


def test_read_payload_says_what_went_wrong() raises:
    """The store logs this error when it drops an unreadable session, so truncation, corruption and a missing file
    must read differently."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(42))
    var ch = ChunkInfo(9, 20, 20, offs[2], UInt64(0))
    var back = List[UInt8](length=110, fill=0)
    d.read_payload(UInt64(42), ch, _toks(9, 20), _addr(back))                 # intact
    _flip(d.path(UInt64(42)), offs[2] + 7)
    with assert_raises(contains="checksum mismatch"):
        d.read_payload(UInt64(42), ch, _toks(9, 20), _addr(back))
    var fd = open_file(d.path(UInt64(42)), O_RDWR, 0)
    truncate_file(fd, offs[2] + 50)                                          # now the payload is cut short
    close_file(fd)
    with assert_raises(contains="short read"):
        d.read_payload(UInt64(42), ch, _toks(9, 20), _addr(back))
    with assert_raises(contains="cannot open"):
        d.read_payload(UInt64(43), ch, _toks(9, 20), _addr(back))            # no such session
    _ = back^
    d.close()


def test_a_failed_append_is_cut_back_so_later_appends_chain() raises:
    """A write that really fails halfway (the file-size limit: EFBIG) must not leave its half record in front of the
    next one. Scan accepts a half record that is not the last, then misreads the good record behind it as its payload
    and cuts that off too."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.create(UInt64(53))
    var p0 = _payload(5, 1)
    _ = d.append(UInt64(53), 0, 5, 5, _toks(0, 5), _addr(p0), len(p0))
    _ = p0^
    var path = d.path(UInt64(53))
    var before = _size_of(path)
    # The next record is 84 bytes: 40 of header, tokens and length, 40 of payload, 4 of CRC. Fail inside each part.
    var written: List[Int] = [20, 40, 50, 82]
    for i in range(len(written)):
        var p1 = _payload(4, 2)
        var saved = _cap_file_size(before + written[i])
        try:
            with assert_raises(contains="pwrite failed"):         # the original error comes through
                _ = d.append(UInt64(53), 5, 9, 5, _toks(5, 9), _addr(p1), len(p1))
        finally:
            _lift_file_size_cap(saved)
        _ = p1^
        assert_equal(_size_of(path), before, msg="after failing " + String(written[i]) + " bytes in")
        assert_equal(d.total_bytes(), before)
    # The disk is fine again: the same chunk and the one after it go in, right behind record 0, and survive a restart.
    var p1 = _payload(4, 2)
    var off1 = d.append(UInt64(53), 5, 9, 5, _toks(5, 9), _addr(p1), len(p1))
    _ = p1^
    assert_equal(off1, before + 40)
    var p2 = _payload(11, 3)
    var off2 = d.append(UInt64(53), 9, 20, 20, _toks(9, 20), _addr(p2), len(p2))
    _ = p2^
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), 1)
    assert_equal(recs[0].kv_len(), 20)
    assert_equal(len(recs[0].chunks), 3)
    assert_equal(recs[0].chunks[1].file_offset, off1)
    assert_equal(recs[0].chunks[2].file_offset, off2)
    var starts: List[Int] = [0, 5, 9]
    var ends: List[Int] = [5, 9, 20]
    var salts: List[Int] = [1, 2, 3]
    for i in range(3):
        var back = List[UInt8](length=10 * (ends[i] - starts[i]), fill=0)
        d2.read_payload(UInt64(53), recs[0].chunks[i], _toks(starts[i], ends[i]), _addr(back))
        var want = _payload(ends[i] - starts[i], salts[i])
        for k in range(len(want)):
            assert_equal(Int(back[k]), Int(want[k]))
        _ = back^
    d2.close()


def test_a_failed_create_leaves_no_file() raises:
    """A header that cannot be written must not leave a header-less file: scan skips it, and keeps it, forever."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var saved = _cap_file_size(20)                         # the header is 88 bytes
    try:
        with assert_raises(contains="pwrite failed"):
            d.create(UInt64(54))
    finally:
        _lift_file_size_cap(saved)
    assert_false(_has_file(dir, UInt64(54)))
    assert_equal(d.total_bytes(), 0)
    d.create(UInt64(54))                                   # and the id is usable once the disk is
    assert_true(_has_file(dir, UInt64(54)))
    # A failed create over a session that already exists must not delete it.
    var p = _payload(5, 1)
    _ = d.append(UInt64(54), 0, 5, 5, _toks(0, 5), _addr(p), len(p))
    _ = p^
    saved = _cap_file_size(20)
    try:
        with assert_raises(contains="pwrite failed"):
            d.create(UInt64(54))
    finally:
        _lift_file_size_cap(saved)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = d2.scan()
    assert_equal(len(recs), 1)
    assert_equal(recs[0].kv_len(), 5)
    d2.close()


def test_the_record_bound_is_the_layouts_size_not_the_files() raises:
    """A corrupt b sizes the token buffer before anything checks it. Bounded only by the file, it still asks for
    gigabytes in a big sparse file: here 2.8 GB, which fits a 3 GiB file. The layout says what that many tokens occupy
    (7 GB more), and the whole record has to fit. Scanned under an address-space cap, as above."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    var offs = _write_three(d, UInt64(55))
    var rec1 = _rec_start(offs[1], 4)
    var path = d.path(UInt64(55))
    _put(path, rec1 + 8, 0x05)                             # b = 5 + 700,000,000 = 0x29B92705, little-endian
    _put(path, rec1 + 9, 0x27)
    _put(path, rec1 + 10, 0xB9)
    _put(path, rec1 + 11, 0x29)
    var fd = open_file(path, O_RDWR, 0)
    truncate_file(fd, 3 * 1024 * 1024 * 1024)              # sparse: costs no disk
    close_file(fd)
    d.close()
    var d2 = DiskTier(dir, _key(1), _layout())
    var recs = _scan_capped(d2, 1024 * 1024 * 1024)
    assert_equal(len(recs), 1)
    assert_equal(recs[0].kv_len(), 5)
    assert_equal(_size_of(path), rec1)                     # cut at the bad record; the 3 GiB hole went with it
    d2.close()


# ── Fix round 1 (Task 8 review, Minor 1): a session file that is already gone ──────────────────────────────────────


def test_dropping_a_session_whose_file_is_already_gone_counts_as_dropped() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(61))
    assert_true(d.total_bytes() > 200)
    remove(d.path(UInt64(61)))                             # an operator deletes the file by hand
    d.drop(UInt64(61))                                     # nothing left to remove: dropped, and no longer counted
    assert_equal(d.total_bytes(), 0)
    d.drop(UInt64(61))                                     # and dropping it again is harmless
    d.drop(UInt64(62))                                     # as is dropping an id that never existed
    d.close()


def test_a_drop_that_fails_for_another_reason_still_raises() raises:
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.create(UInt64(63))
    var counted = d.total_bytes()
    assert_true(counted > 0)
    remove(d.path(UInt64(63)))
    makedirs(d.path(UInt64(63)))                           # a directory where the session file was: unlink fails
    try:
        with assert_raises(contains="Can not remove"):
            d.drop(UInt64(63))
        assert_true(_has_file(dir, UInt64(63)))            # it is still there
        assert_equal(d.total_bytes(), 0)                   # but the size goes first, whatever happens to the file
    finally:
        rmdir(d.path(UInt64(63)))
    d.close()


def test_fail_once_at_fails_exactly_one_write() raises:
    """The hook the store's tests use to fail one chunk's append and let every other write through."""
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    d.fail_once_at = 2                                     # writes are counted from now: create 0, append 1, append 2
    d.create(UInt64(64))
    var p = _payload(2, 9)
    var off = d.append(UInt64(64), 0, 2, 2, _toks(0, 2), _addr(p), len(p))
    assert_true(off > 0)
    with assert_raises(contains="simulated write failure"):
        _ = d.append(UInt64(64), 2, 4, 4, _toks(2, 4), _addr(p), len(p))
    assert_equal(d.fail_once_at, -1)                       # it fired once and switched itself off
    var off2 = d.append(UInt64(64), 2, 4, 4, _toks(2, 4), _addr(p), len(p))
    assert_true(off2 > off)                                # the next write goes through, and chains onto the first
    d.fail_once_at = 0                                     # 0 = the very next write
    with assert_raises(contains="simulated write failure"):
        d.create(UInt64(65))
    assert_false(_has_file(dir, UInt64(65)))
    _ = p^
    d.close()


def test_session_files_and_their_directory_are_the_owners_alone() raises:
    # A session file holds a conversation (its token ids in clear, and the KV that encodes them): the store's directory is
    # 0700, and the lock and every session file 0600. A umask can only take bits away, so these are exact.
    var dir = _dir()
    var d = DiskTier(dir, _key(1), _layout())
    _ = _write_three(d, UInt64(7))
    assert_equal(Int(stat(dir).st_mode) & 0o777, 0o700)
    assert_equal(Int(stat(dir + "/.lock").st_mode) & 0o777, 0o600)
    assert_equal(Int(stat(d.path(UInt64(7))).st_mode) & 0o777, 0o600)
    d.close()


def main() raises:
    _clear_root()
    test_session_files_and_their_directory_are_the_owners_alone()
    test_scan_rebuilds_sessions()
    test_read_payload_round_trip_and_checksum()
    test_torn_tail_is_cut_back()
    test_other_keys_are_left_alone()
    test_other_format_versions_are_left_alone()
    test_lock_refuses_a_second_store()
    test_drop_and_total_bytes()
    test_fail_writes_hook()
    test_bad_checksum_on_the_last_record_is_cut_back()
    test_every_kind_of_torn_tail_is_cut_back()
    test_corrupt_record_headers_are_cut_back_without_trusting_them()
    test_nothing_after_a_bad_record_is_trusted()
    test_other_layouts_are_left_alone()
    test_files_that_are_not_sessions_are_ignored()
    test_a_session_without_records_is_removed()
    test_append_after_a_restart_extends_the_session()
    test_older_records_are_checked_when_read_not_at_scan()
    test_a_refused_second_store_does_not_release_the_lock()
    test_file_bytes_are_the_documented_format()
    test_scan_returns_exactly_what_was_written()
    test_fail_writes_blocks_create_too()
    test_append_never_creates_a_session()
    test_a_payload_over_the_crc_slice_is_verified_across_slices()
    test_an_empty_chunk_is_not_a_chunk()
    test_read_payload_says_what_went_wrong()
    test_a_failed_append_is_cut_back_so_later_appends_chain()
    test_a_failed_create_leaves_no_file()
    test_the_record_bound_is_the_layouts_size_not_the_files()
    test_dropping_a_session_whose_file_is_already_gone_counts_as_dropped()
    test_a_drop_that_fails_for_another_reason_still_raises()
    test_fail_once_at_fails_exactly_one_write()
    _clear_root()
    print("kvstore_disk_test: all passed")
