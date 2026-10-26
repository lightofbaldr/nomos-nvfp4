"""NVMe tier of the KV session store (docs/KV_SESSION_STORE_DESIGN.md §3.1): one append-only file per session.

  header  u32 magic "NKVS" | u32 version | u32 model_id | u32 kv_mode | u64 build_id | u64 layout_hash
          | u64 variant_hash | u64 drafter_id | u32 n_components | n x (u32 kind, layer, heads, row_bytes)
          | u32 CRC-32C of everything before it
  record  u32 magic "CHNK" | u32 a | u32 b | u32 drafter_end | (b-a) x u32 token | u64 payload_len
          | payload | u32 CRC-32C of the token bytes followed by the payload

Every append is fsync'd. At startup every record's structure is checked, plus the CRC of the last record (a torn
write can only be the last one); older payloads are CRC-checked when read back."""
from std.memory import UnsafePointer
from std.os import listdir, makedirs, remove
from std.os.path import exists
from lib.kvstore.fileio import (
    O_RDONLY, O_RDWR, O_CREAT, open_file, close_file, pread_all, pwrite_all, fsync_file, file_size,
    truncate_file, try_lock_exclusive,
)
from lib.kvstore.hashing import Crc32c, hex16, parse_hex16
from lib.kvstore.layout import KvLayout
from lib.kvstore.key import CacheKey
from lib.kvstore.index import SessionRecord, ChunkInfo

comptime NKVS_MAGIC: UInt32 = 0x53564B4E
comptime CHNK_MAGIC: UInt32 = 0x4B4E4843
comptime FORMAT_VERSION: UInt32 = 1
# A session file holds a conversation: its token ids in clear and the KV that encodes them. So the files and the lock are
# the owner's alone (0600), and so is the store's directory (0700).
comptime _MODE_600 = 0o600
comptime _DIR_MODE_700 = 0o700
comptime _CRC_SLICE = 16 * 1024 * 1024


def _ptr(addr: UInt64) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    return UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr))


def _addr(b: List[UInt8]) -> UInt64:
    return UInt64(Int(b.unsafe_ptr()))


struct ByteWriter(Movable):
    var buf: List[UInt8]

    def __init__(out self):
        self.buf = List[UInt8]()

    def u32(mut self, v: UInt32):
        for i in range(4):
            self.buf.append(UInt8((v >> UInt32(8 * i)) & 0xFF))

    def u64(mut self, v: UInt64):
        for i in range(8):
            self.buf.append(UInt8((v >> UInt64(8 * i)) & 0xFF))

    def addr(self) -> UInt64:
        return _addr(self.buf)

    def size(self) -> Int:
        return len(self.buf)


def get_u32(b: List[UInt8], off: Int) -> UInt32:
    return UInt32(b[off]) | (UInt32(b[off + 1]) << 8) | (UInt32(b[off + 2]) << 16) | (UInt32(b[off + 3]) << 24)


def get_u64(b: List[UInt8], off: Int) -> UInt64:
    return UInt64(get_u32(b, off)) | (UInt64(get_u32(b, off + 4)) << 32)


struct DiskTier(Movable):
    var dir: String
    var key: CacheKey
    var layout: KvLayout
    var lock_fd: Int
    var crc: Crc32c
    var sizes: Dict[UInt64, Int]   # session id -> file size
    var fail_writes: Bool          # test hook: append() and create() raise as if the disk were full
    var fail_once_at: Int          # test hook: the Nth write from now (0 = the next) raises once, then writes work again; -1 = off

    def __init__(out self, dir: String, key: CacheKey, layout: KvLayout) raises:
        self.dir = dir
        self.key = key
        self.layout = layout.copy()
        self.crc = Crc32c()
        self.sizes = Dict[UInt64, Int]()
        self.fail_writes = False
        self.fail_once_at = -1
        makedirs(dir, mode=_DIR_MODE_700, exist_ok=True)   # the mode applies to a directory made here, not to one that exists
        self.lock_fd = open_file(dir + "/.lock", O_RDWR | O_CREAT, _MODE_600)
        if self.lock_fd < 0:
            raise Error("kv store: cannot open the lock file in " + dir)
        if not try_lock_exclusive(self.lock_fd):
            close_file(self.lock_fd)
            self.lock_fd = -1
            raise Error("kv store: " + dir + " is in use by another process")

    def close(mut self):
        if self.lock_fd >= 0:
            close_file(self.lock_fd)
            self.lock_fd = -1

    def path(self, id: UInt64) -> String:
        return self.dir + "/" + hex16(id) + ".nkvs"

    def total_bytes(self) -> Int:
        var t = 0
        for e in self.sizes.items():
            t += e.value
        return t

    def _header(self) -> ByteWriter:
        var w = ByteWriter()
        w.u32(NKVS_MAGIC)
        w.u32(FORMAT_VERSION)
        w.u32(UInt32(self.key.model_id))
        w.u32(UInt32(self.key.kv_mode))
        w.u64(self.key.build_id)
        w.u64(self.key.layout_hash)
        w.u64(self.key.variant_hash)
        w.u64(self.key.drafter_id)
        w.u32(UInt32(len(self.layout.comps)))
        for i in range(len(self.layout.comps)):
            var c = self.layout.comps[i]
            w.u32(UInt32(c.kind))
            w.u32(UInt32(c.layer))
            w.u32(UInt32(c.heads))
            w.u32(UInt32(c.row_bytes))
        w.u32(self.crc.update(0, _ptr(w.addr()), w.size()))
        return w^

    def _remove_quietly(self, id: UInt64):
        try:
            remove(self.path(id))
        except e:
            print("[kvstore] could not remove the unfinished", self.path(id), ":", e)

    def _cut_back(self, fd: Int, off: Int, id: UInt64):
        """After a failed append: drop whatever the failed writes left past off. Best effort, never raises."""
        if off < 0:
            return
        try:
            truncate_file(fd, off)
            fsync_file(fd)
        except e:
            print("[kvstore] could not cut a failed append back to byte", off, "in", self.path(id), ":", e)

    def _simulated_failure(mut self) raises:
        """The test hooks: fail_writes fails every write, fail_once_at fails one."""
        if self.fail_writes:
            raise Error("kv store: simulated write failure")
        if self.fail_once_at == 0:
            self.fail_once_at = -1
            raise Error("kv store: simulated write failure")
        if self.fail_once_at > 0:
            self.fail_once_at -= 1

    def create(mut self, id: UInt64) raises:
        """Write a session's header and fsync. A header that cannot be written leaves no file behind."""
        self._simulated_failure()
        var fd = open_file(self.path(id), O_RDWR | O_CREAT, _MODE_600)
        if fd < 0:
            raise Error("kv store: cannot create " + self.path(id))
        var fresh = file_size(fd) == 0   # made by this call, so it is this call's to remove; a session already there is not
        try:
            var h = self._header()
            var hlen = h.size()
            pwrite_all(fd, h.addr(), hlen, 0)
            _ = h^   # an address is not a use: keep the buffer alive until pwrite has read it
            fsync_file(fd)
            self.sizes[id] = hlen
        except e:
            if fresh:
                self._remove_quietly(id)
            raise e
        finally:
            close_file(fd)

    def append(mut self, id: UInt64, a: Int, b: Int, drafter_end: Int, tokens: List[Int], addr: UInt64, nbytes: Int) raises -> Int:
        """Append one record and fsync. Returns the payload's file offset. A failed append cuts the file back to where
        it started (best effort), so a half-written record never sits in front of the next one."""
        self._simulated_failure()
        var fd = open_file(self.path(id), O_RDWR, 0)
        if fd < 0:
            raise Error("kv store: cannot open " + self.path(id))
        var off = file_size(fd)
        try:
            var w = ByteWriter()
            w.u32(CHNK_MAGIC)
            w.u32(UInt32(a))
            w.u32(UInt32(b))
            w.u32(UInt32(drafter_end))
            var tok_start = w.size()
            for i in range(len(tokens)):
                w.u32(UInt32(tokens[i]))
            var crc = self.crc.update(0, _ptr(w.addr() + UInt64(tok_start)), w.size() - tok_start)
            crc = self.crc.update(crc, _ptr(addr), nbytes)
            w.u64(UInt64(nbytes))
            var head = w.size()
            pwrite_all(fd, w.addr(), head, off)
            _ = w^   # an address is not a use: keep the buffer alive until pwrite has read it
            var payload_off = off + head
            pwrite_all(fd, addr, nbytes, payload_off)
            var tail = ByteWriter()
            tail.u32(crc)
            pwrite_all(fd, tail.addr(), 4, payload_off + nbytes)
            _ = tail^   # an address is not a use: keep the buffer alive until pwrite has read it
            fsync_file(fd)
            self.sizes[id] = payload_off + nbytes + 4
            return payload_off
        except e:
            self._cut_back(fd, off, id)
            raise e
        finally:
            close_file(fd)

    def read_payload(mut self, id: UInt64, chunk: ChunkInfo, tokens: List[Int], dst: UInt64) raises:
        """Read a chunk's payload into dst and verify its CRC against the stored one."""
        var nbytes = self.layout.chunk_bytes(chunk.end - chunk.start)
        var fd = open_file(self.path(id), O_RDONLY, 0)
        if fd < 0:
            raise Error("kv store: cannot open " + self.path(id))
        try:
            if not pread_all(fd, dst, nbytes, chunk.file_offset):
                raise Error("kv store: short read in " + self.path(id))
            var tb = ByteWriter()
            for i in range(len(tokens)):
                tb.u32(UInt32(tokens[i]))
            var crc = self.crc.update(0, _ptr(tb.addr()), tb.size())
            _ = tb^   # keep the token bytes alive until the CRC has read them
            crc = self.crc.update(crc, _ptr(dst), nbytes)
            var stored = List[UInt8](length=4, fill=0)
            if not pread_all(fd, _addr(stored), 4, chunk.file_offset + nbytes) or get_u32(stored, 0) != crc:
                raise Error("kv store: checksum mismatch in " + self.path(id))
        finally:
            close_file(fd)

    def drop(mut self, id: UInt64) raises:
        """Forget a session and remove its file. The size goes first, so a file that has vanished (deleted by hand) never
        stays counted; a file that is already gone counts as dropped. Any other failure to remove it raises."""
        if id in self.sizes:
            _ = self.sizes.pop(id)
        try:
            remove(self.path(id))
        except e:
            if exists(self.path(id)):
                raise e

    def _crc_range(self, fd: Int, off: Int, n: Int, crc0: UInt32) -> Optional[UInt32]:
        var slice = max(min(n, _CRC_SLICE), 1)
        var buf = List[UInt8](length=slice, fill=0)
        var crc = crc0
        var done = 0
        while done < n:
            var take = min(slice, n - done)
            if not pread_all(fd, _addr(buf), take, off + done):
                return None
            crc = self.crc.update(crc, _ptr(_addr(buf)), take)
            done += take
        _ = buf^
        return crc

    def _load(mut self, id: UInt64) raises -> Optional[SessionRecord]:
        var fd = open_file(self.path(id), O_RDWR, 0)
        if fd < 0:
            return None
        try:
            var size = file_size(fd)
            var expect = self._header()
            var hlen = expect.size()
            if size < hlen:
                return None
            var got = List[UInt8](length=hlen, fill=0)
            if not pread_all(fd, _addr(got), hlen, 0):
                return None
            for i in range(hlen):
                if got[i] != expect.buf[i]:
                    return None     # another format version, key or layout: leave the file alone
            var tokens = List[Int]()
            var chunks = List[ChunkInfo]()
            var off = hlen
            while off + 16 <= size:
                var rh = List[UInt8](length=16, fill=0)
                if not pread_all(fd, _addr(rh), 16, off) or get_u32(rh, 0) != CHNK_MAGIC:
                    break
                var a = Int(get_u32(rh, 4))
                var b = Int(get_u32(rh, 8))
                var dend = Int(get_u32(rh, 12))
                if a != len(tokens) or b <= a or dend < a or dend > b:
                    break
                var ntok = b - a
                if off + 16 + 4 * ntok + 8 + self.layout.chunk_bytes(ntok) + 4 > size:
                    # b is unchecked here and is about to size the token buffer (4 * ntok + 8 bytes). A record of ntok
                    # tokens takes exactly the bytes tested here (header, tokens, length, the layout's payload, CRC); if
                    # they do not fit in the rest of the file, b is corrupt. That bounds the buffer by
                    # 4 / (4 + bytes per token) of the rest of the file, not all of it.
                    break
                var tb = List[UInt8](length=4 * ntok + 8, fill=0)
                if not pread_all(fd, _addr(tb), 4 * ntok + 8, off + 16):
                    break
                var plen = Int(get_u64(tb, 4 * ntok))
                if plen != self.layout.chunk_bytes(ntok):
                    break
                var payload_off = off + 16 + 4 * ntok + 8
                var end = payload_off + plen + 4
                if end > size:
                    break
                if end == size:   # the last record: verify its CRC, a torn write can only be here
                    var crc = self._crc_range(fd, payload_off, plen, self.crc.update(0, _ptr(_addr(tb)), 4 * ntok))
                    var stored = List[UInt8](length=4, fill=0)
                    if not crc or not pread_all(fd, _addr(stored), 4, payload_off + plen) or get_u32(stored, 0) != crc.value():
                        break
                for k in range(ntok):
                    tokens.append(Int(get_u32(tb, 4 * k)))
                chunks.append(ChunkInfo(a, b, dend, payload_off, UInt64(0)))
                off = end
            if off < size:
                truncate_file(fd, off)
                fsync_file(fd)
                print("[kvstore] cut a torn tail from", self.path(id), "at byte", off)
            if len(chunks) == 0:
                self.drop(id)
                return None
            self.sizes[id] = off
            var dlen = 0
            for i in range(len(chunks)):
                if dlen == chunks[i].start:
                    dlen = chunks[i].drafter_end
            return SessionRecord(id, tokens^, chunks^, dlen, 0, True)
        finally:
            close_file(fd)

    def scan(mut self) raises -> List[SessionRecord]:
        """Every readable session file under this key and layout; torn tails are cut back first."""
        var out = List[SessionRecord]()
        var skipped = 0
        var names = listdir(self.dir)
        for i in range(len(names)):
            if not names[i].endswith(".nkvs"):
                continue
            var id = parse_hex16(String(names[i].removesuffix(".nkvs")))
            if not id:
                continue
            var rec = self._load(id.value())
            if rec:
                out.append(rec.value().copy())
            else:
                skipped += 1
        if skipped > 0:
            # Files from another cache key (DFlash on/off, KV mode) are kept for that key but count against no
            # budget here: say so, so the space is never invisible.
            print("[kvstore]", skipped, "session files in", self.dir, "skipped (another cache key, or unreadable)")
        return out^
