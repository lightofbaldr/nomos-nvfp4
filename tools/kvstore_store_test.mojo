"""SessionStore unit tests against a CPU fake engine.

Run: bash tools/run_mojo_test.sh tools/kvstore_store_test.mojo
"""
from std.memory import Pointer
from std.testing import assert_equal, assert_true, assert_false
from lib.kvstore.engine_trait import KvSessionEngine
from lib.kvstore.fileio import random_u64
from lib.kvstore.hashing import FNV_OFFSET, fnv1a64_tokens, hex16
from lib.kvstore.key import CacheKey
from lib.kvstore.layout import KvLayout, ComponentDesc, KIND_K_DATA, KIND_K_SCALE, KIND_DRAFT_K
from lib.kvstore.store import (
    SessionStore, StoreConfig, STAT_REUSED_TOKENS, STAT_RESTORES_RAM, STAT_RESTORES_NVME, STAT_FORKS,
    STAT_RAM_EVICTIONS,
)
from std.os import listdir, makedirs, remove, rmdir
from std.os.path import exists, isdir
from lib.kvstore.disk_tier import DiskTier
from lib.kvstore.fileio import O_RDONLY, O_RDWR, open_file, close_file, pread_all, pwrite_all, file_size
from lib.kvstore.index import sub
from lib.kvstore.ram_tier import RamTier
from lib.kvstore.store import (
    STAT_PREPARES, STAT_PROMPT_TOKENS, STAT_RAM_BYTES, STAT_NVME_BYTES, STAT_NVME_EVICTIONS,
)

comptime CAP = 2048


def _layout(drafter: Bool) -> KvLayout:
    var lay = KvLayout()
    lay.comps.append(ComponentDesc(KIND_K_DATA, 0, 2, 3))
    lay.comps.append(ComponentDesc(KIND_K_SCALE, 0, 2, 4))
    if drafter:
        lay.comps.append(ComponentDesc(KIND_DRAFT_K, 0, 1, 5))
    return lay^


def _byte(tokens: List[Int], pos: Int, c: Int, h: Int, r: Int) -> UInt8:
    """The fake KV byte for one (component, head, position, row): a function of the prefix up to pos."""
    var hp = fnv1a64_tokens(FNV_OFFSET, tokens, 0, pos + 1)
    return UInt8((hp >> UInt64((c * 7 + h * 3 + r) % 56)) & 0xFF)


struct FakeEngine(KvSessionEngine):
    """A CPU stand-in for the GPU engine. Its KV for a position is derived from the tokens up to it, so a correct
    restore reproduces it exactly and a wrong one does not. Stored head-first with stride CAP, like the engine."""
    var tokens: List[Int]
    var kv: List[UInt8]
    var layout: KvLayout
    var drafter_rows: Int
    var has_drafter: Bool
    var imports: Int
    var desync: Bool

    def __init__(out self, drafter: Bool):
        self.layout = _layout(drafter)
        self.tokens = List[Int]()
        self.kv = List[UInt8](length=CAP * self.layout.bytes_per_token(), fill=0)
        self.drafter_rows = 0
        self.has_drafter = drafter
        self.imports = 0
        self.desync = False

    def _slot(self, c: Int, h: Int, pos: Int, r: Int) -> Int:
        var base = 0
        for i in range(c):
            base += CAP * self.layout.comps[i].bytes_per_token()
        var row = self.layout.comps[c].row_bytes
        return base + (h * CAP + pos) * row + r

    def _compute(mut self, pos: Int, drafter: Bool):
        for c in range(len(self.layout.comps)):
            var comp = self.layout.comps[c]
            if comp.is_drafter() and not drafter:
                continue
            for h in range(comp.heads):
                for r in range(comp.row_bytes):
                    self.kv[self._slot(c, h, pos, r)] = _byte(self.tokens, pos, c, h, r)

    def prefill(mut self, ids: List[Int]):
        self.tokens = ids.copy()
        for p in range(len(ids)):
            self._compute(p, True)
        self.drafter_rows = len(ids) if self.has_drafter else 0

    def extend(mut self, ids: List[Int], drafter: Bool):
        """The prefill_cont / decode path: the drafter follows only when it was in step (speculative path)."""
        var follow = drafter and self.has_drafter and self.drafter_rows == len(self.tokens)
        for i in range(len(ids)):
            self.tokens.append(ids[i])
            self._compute(len(self.tokens) - 1, follow)
        if follow:
            self.drafter_rows = len(self.tokens)

    def expected_ok(self) -> Bool:
        for c in range(len(self.layout.comps)):
            var comp = self.layout.comps[c]
            var n = self.drafter_rows if comp.is_drafter() else len(self.tokens)
            for p in range(n):
                for h in range(comp.heads):
                    for r in range(comp.row_bytes):
                        if self.kv[self._slot(c, h, p, r)] != _byte(self.tokens, p, c, h, r):
                            return False
        return True

    def kv_len(self) -> Int:
        return len(self.tokens) + (5 if self.desync else 0)

    def drafter_len(self) -> Int:
        return self.drafter_rows if self.has_drafter else 0

    def live_tokens_copy(self) -> List[Int]:
        return self.tokens.copy()

    def export_chunk(mut self, layout: KvLayout, a: Int, b: Int, drafter_end: Int, dst: UInt64) raises:
        var out = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(dst))
        var off = 0
        var ntok = b - a
        for c in range(len(layout.comps)):
            var comp = layout.comps[c]
            var last = min(drafter_end, b) if comp.is_drafter() else b
            for h in range(comp.heads):
                for p in range(a, last):
                    for r in range(comp.row_bytes):
                        out[unsafe_offset = off + (h * ntok + (p - a)) * comp.row_bytes + r] = self.kv[self._slot(c, h, p, r)]
            off += ntok * comp.bytes_per_token()

    def import_chunk(mut self, layout: KvLayout, a: Int, b: Int, drafter_end: Int, src: UInt64, src_tokens: Int, tokens: List[Int]) raises:
        if a != len(self.tokens):
            raise Error("fake import: gap")
        self.imports += 1
        var inp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(src))
        var do_drafter = self.has_drafter and self.drafter_rows == a and drafter_end > a
        var off = 0
        for c in range(len(layout.comps)):
            var comp = layout.comps[c]
            var last = drafter_end if comp.is_drafter() else b
            if not comp.is_drafter() or do_drafter:
                for h in range(comp.heads):
                    for p in range(a, last):
                        for r in range(comp.row_bytes):
                            self.kv[self._slot(c, h, p, r)] = inp[unsafe_offset = off + (h * src_tokens + (p - a)) * comp.row_bytes + r]
            off += src_tokens * comp.bytes_per_token()
        for i in range(len(tokens)):
            self.tokens.append(tokens[i])
        if do_drafter:
            self.drafter_rows = drafter_end

    def truncate_to(mut self, p: Int) raises:
        while len(self.tokens) > p:
            _ = self.tokens.pop()
        self.drafter_rows = min(self.drafter_rows, p)

    def reset_session(mut self) raises:
        self.tokens = List[Int]()
        self.drafter_rows = 0
        self.desync = False


def _toks(n: Int, salt: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append((i * 37 + salt * 7919) % 199000)
    return out^


def _cat(a: List[Int], b: List[Int]) -> List[Int]:
    var out = a.copy()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def _store(dir: String, drafter: Bool, ram_bytes: Int) raises -> SessionStore:
    var lay = _layout(drafter)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    return SessionStore(key, lay, StoreConfig(dir, ram_bytes, 1 << 30, 16, 256, False, 4096))


comptime _ROOT = "build/kvstore_store_test"       # every directory the tests make is under here; main() clears it


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
    """The tests leave their session directories under _ROOT (about 60 per run). main() clears it before the first test,
    for what an earlier failed run left as evidence, and after the last, so a passing run leaves nothing. Run one
    store-test process at a time: a second one would lose its directories."""
    if exists(_ROOT):
        _rmtree(_ROOT)


def _turn(mut s: SessionStore, mut e: FakeEngine, ids: List[Int], gen: Int, spec: Bool) raises -> Int:
    var c = s.prepare(e, ids, spec)
    assert_equal(e.kv_len(), c)
    if c == 0:
        e.prefill(ids)
    else:
        var rest = List[Int]()
        for i in range(c, len(ids)):
            rest.append(ids[i])
        e.extend(rest, spec)
    e.extend(_toks(gen, len(ids)), spec)   # generated tokens (all but the last have KV, like the engine)
    assert_true(s.commit(e) >= 0)
    return c


def test_next_turn_reuses_without_copies() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var ids1 = _toks(300, 1)
    assert_equal(_turn(s, e, ids1, 20, False), 0)
    var ids2 = _cat(e.tokens, _toks(30, 2))
    assert_equal(_turn(s, e, ids2, 10, False), 320)
    assert_equal(e.imports, 0)
    assert_true(e.expected_ok())
    s.close()


def test_full_prompt_returns_n_minus_1_without_fork() raises:      # Review Focus 5
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var ids = _toks(300, 3)
    _ = _turn(s, e, ids, 0, False)
    assert_equal(s.prepare(e, ids, False), 299)
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)
    s.close()


def test_switch_restores_parked_session() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 4)
    _ = _turn(s, e, a, 0, False)
    var b = _toks(400, 5)
    assert_equal(_turn(s, e, b, 0, False), 0)
    var a2 = _cat(a, _toks(10, 6))
    assert_equal(s.prepare(e, a2, False), 600)
    assert_true(e.expected_ok())
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM], 1)
    s.close()


def test_divergence_forks() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 7)
    _ = _turn(s, e, a, 0, False)
    var req = List[Int]()
    for i in range(500):
        req.append(a[i])
    var req2 = _cat(req, _toks(50, 8))
    assert_equal(s.prepare(e, req2, False), 500)
    assert_equal(s.stats_snapshot()[STAT_FORKS], 1)
    s.close()


def test_short_overlap_starts_fresh() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(300, 9)
    _ = _turn(s, e, a, 0, False)
    var req = List[Int]()
    for i in range(10):
        req.append(a[i])
    assert_equal(s.prepare(e, _cat(req, _toks(20, 10)), False), 0)
    s.close()


def test_spec_caps_at_drafter_len_and_forks() raises:
    var s = _store(_dir(), True, 1 << 30)
    var e = FakeEngine(True)
    var a = _toks(300, 11)
    _ = _turn(s, e, a, 0, True)
    e.extend(_toks(50, 12), False)          # a base-path turn: the drafter falls behind
    assert_true(s.commit(e) > 0)
    var req = _cat(e.tokens, _toks(20, 13))
    assert_equal(s.prepare(e, req, True), 300)
    assert_equal(s.stats_snapshot()[STAT_FORKS], 1)
    s.close()


def test_ram_eviction_restores_from_disk() raises:
    var s = _small_chunk_store(_dir(), False, 1000)      # far smaller than one session (300 tokens x 14 bytes), but one 64-token chunk (896 bytes) fits
    var e = FakeEngine(False)
    var a = _toks(300, 14)
    _ = _turn(s, e, a, 0, False)
    _ = _turn(s, e, _toks(300, 15), 0, False)
    assert_true(s.stats_snapshot()[STAT_RAM_EVICTIONS] >= 1)
    assert_equal(s.prepare(e, _cat(a, _toks(5, 16)), False), 300)
    assert_true(e.expected_ok())
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 1)
    s.close()


def test_restart_restores_from_disk() raises:
    var dir = _dir()
    var s = _store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(400, 17)
    _ = _turn(s, e, a, 0, False)
    s.close()
    var s2 = _store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    assert_equal(s2.prepare(e2, _cat(a, _toks(5, 18)), False), 400)
    assert_true(e2.expected_ok())
    s2.close()


def test_desynced_engine_is_not_trusted() raises:         # Review Focus 2
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 19)
    _ = _turn(s, e, a, 0, False)
    e.desync = True                                        # KV changed behind the store's back
    var c = s.prepare(e, _cat(a, _toks(5, 20)), False)
    assert_equal(c, 600)                                   # restored from the committed copy, not trusted live
    assert_true(e.expected_ok())
    s.close()


def test_disk_failure_keeps_session_in_ram() raises:     # Review Focus 4
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var a = _toks(300, 21)
    assert_equal(_turn(s, e, a, 0, False), 0)             # the turn still succeeds
    s.cfg.ram_bytes = 4300                                 # now force RAM pressure: no room for a second 4200-byte chunk
    # The disk is still failing and the budget is hard (ruling R27): A's only chunk has no disk copy, so it cannot be
    # evicted, and B can no longer be pinned past the budget. B is not saved; A must survive.
    var b = _toks(300, 22)
    assert_equal(s.prepare(e, b, False), 0)
    e.prefill(b)
    assert_equal(s.commit(e), -3)
    assert_equal(s.prepare(e, _cat(a, _toks(5, 23)), False), 300)   # still restorable: never evicted
    assert_true(e.expected_ok())
    s.close()


def test_disk_failure_catches_up_when_the_disk_recovers() raises:   # ledger R25
    var dir = _dir()
    var s = _store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var a = _toks(300, 31)
    _ = _turn(s, e, a, 0, False)                      # chunk [0, 300) stays in RAM only
    s.disk.fail_writes = False
    e.extend(_toks(20, 32), False)
    assert_equal(s.commit(e), 20)                     # writes [0, 300) first, then [300, 320)
    s.close()
    var s2 = _store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    var ids = _cat(_cat(a, _toks(20, 32)), _toks(5, 33))
    assert_equal(s2.prepare(e2, ids, False), 320)     # the whole session survived the restart
    assert_true(e2.expected_ok())
    s2.close()


def test_commit_after_engine_reset_saves_a_new_session() raises:   # plan note 7
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(300, 24)
    _ = _turn(s, e, a, 0, False)
    e.reset_session()                       # what nomos_session_prepare's error path does to the engine
    e.prefill(_toks(400, 25))               # then the caller's fresh prefill
    assert_equal(s.commit(e), 400)          # saved as a new session, not refused
    assert_equal(s.stats_snapshot()[STAT_FORKS], 1)
    assert_equal(s.prepare(e, _cat(a, _toks(5, 26)), False), 300)   # the old session is intact
    assert_true(e.expected_ok())
    s.close()


def test_abandoned_turn_leaves_no_record() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    _ = s.prepare(e, _toks(50, 27), False)  # prepared, but never prefilled or committed
    _ = s.prepare(e, _toks(60, 28), False)
    assert_equal(len(s.index.sessions), 1)
    s.close()


def test_long_commit_is_split_into_bounded_chunks() raises:      # 32K-128K contexts
    var dir = _dir()
    var lay = _layout(True)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var cfg = StoreConfig(dir, 1 << 30, 1 << 30, 16, 16, False, 64)
    var s = SessionStore(key, lay, cfg.copy())
    var e = FakeEngine(True)
    var a = _toks(64 * 2 + 10, 29)
    _ = _turn(s, e, a, 0, True)
    var pos = s.index.find(s.live_id)
    assert_equal(len(s.index.sessions[pos].chunks), 3)              # 64 + 64 + 10
    assert_equal(s.index.sessions[pos].drafter_len, len(a))
    s.close()
    var s2 = SessionStore(key, lay, cfg.copy())                     # the pieces read back as one session
    var e2 = FakeEngine(True)
    assert_equal(s2.prepare(e2, _cat(a, _toks(5, 30)), True), len(a))
    assert_true(e2.expected_ok())
    s2.close()


# ── Added after mutation-testing the tests above (task-8-report.md). Each pins something the first 14 leave open. ──


def _prep(mut s: SessionStore, mut e: FakeEngine, ids: List[Int], spec: Bool) raises -> Int:
    """prepare() and its whole contract: 0 <= c <= len(ids) - 1, and the engine then holds exactly ids[0:c]."""
    var c = s.prepare(e, ids, spec)
    assert_true(c >= 0 and c <= max(len(ids) - 1, 0))
    assert_equal(e.kv_len(), c)
    assert_equal(len(e.tokens), c)
    for i in range(c):
        assert_equal(e.tokens[i], ids[i])
    assert_true(e.expected_ok())
    return c


def _resident_bytes(s: SessionStore) -> Int:
    """The RAM the index says it holds: every chunk that has a buffer. The RAM tier's count must always equal it."""
    var total = 0
    for i in range(len(s.index.sessions)):
        for ci in range(len(s.index.sessions[i].chunks)):
            var ch = s.index.sessions[i].chunks[ci]
            if ch.ram_addr != 0:
                total += s.layout.chunk_bytes(ch.end - ch.start)
    return total


def _has_file(s: SessionStore, id: UInt64) raises -> Bool:
    var names = listdir(s.cfg.dir)
    for i in range(len(names)):
        if names[i] == hex16(id) + ".nkvs":
            return True
    return False


def _file_bytes(s: SessionStore, id: UInt64) -> Int:
    var fd = open_file(s.disk.path(id), O_RDONLY, 0)
    var n = file_size(fd)
    close_file(fd)
    return n


def _flip_byte(path: String, back: Int) raises:
    """Corrupt one byte, `back` bytes before the end of a file."""
    var fd = open_file(path, O_RDWR, 0)
    assert_true(fd >= 0)
    var at = file_size(fd) - back
    var b = List[UInt8](length=1, fill=0)
    assert_true(pread_all(fd, UInt64(Int(b.unsafe_ptr())), 1, at))
    b[0] = b[0] ^ 0xFF
    pwrite_all(fd, UInt64(Int(b.unsafe_ptr())), 1, at)
    _ = b^          # an address is not a use: keep the byte alive until pwrite has read it
    close_file(fd)


def _small_chunk_store(dir: String, drafter: Bool, ram_bytes: Int, chunk_tokens: Int = 64) raises -> SessionStore:
    """min_reuse 16, min_switch 16, chunks of at most 64 tokens (or chunk_tokens): sessions of a few hundred tokens have
    several. The RAM budget must hold one chunk."""
    var lay = _layout(drafter)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    return SessionStore(key, lay, StoreConfig(dir, ram_bytes, 1 << 30, 16, 16, False, chunk_tokens))


def test_ram_tier_counts_the_bytes_it_holds() raises:
    var r = RamTier(False)
    var a = r.alloc(100)
    var b = r.alloc(250)
    assert_true(a != 0 and b != 0 and a != b)
    assert_equal(r.used, 350)
    assert_equal(r.peak, 350)
    r.free(a, 100)
    assert_equal(r.used, 250)
    r.free(UInt64(0), 50)                   # freeing nothing counts nothing
    assert_equal(r.used, 250)
    r.free(b, 250)
    assert_equal(r.used, 0)
    assert_equal(r.peak, 350)               # the high-water mark stays


def test_close_frees_every_ram_chunk() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(300, 110), 0, False)
    s.disk.fail_writes = True
    _ = _turn(s, e, _toks(200, 111), 0, False)         # this session exists only in RAM
    assert_equal(s.ram.used, 500 * 14)
    assert_equal(s.stats_snapshot()[STAT_RAM_BYTES], 500 * 14)
    s.close()                                          # the disk is still down, so close() cannot save it: it is freed anyway
    assert_equal(s.ram.used, 0)                        # RAM-only chunks too: the store is going away
    assert_equal(_resident_bytes(s), 0)


def test_stats_count_a_scripted_run() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    assert_equal(_turn(s, e, _toks(300, 120), 20, False), 0)         # 300 prompt tokens, nothing to reuse
    var id_a = s.live_id
    var ids2 = _cat(e.tokens, _toks(30, 121))
    assert_equal(_turn(s, e, ids2, 10, False), 320)                   # 350 prompt tokens, 320 reused
    var a_full = e.tokens.copy()                                      # 360 committed
    assert_equal(_turn(s, e, _toks(400, 122), 0, False), 0)          # an unrelated conversation
    var id_b = s.live_id
    assert_equal(_prep(s, e, _cat(a_full, _toks(5, 123)), False), 360)   # back to the first one, parked in RAM
    var st = s.stats_snapshot()
    assert_equal(st[STAT_PREPARES], 4)
    assert_equal(st[STAT_PROMPT_TOKENS], 300 + 350 + 400 + 365)
    assert_equal(st[STAT_REUSED_TOKENS], 320 + 360)
    assert_equal(st[STAT_RESTORES_RAM], 1)
    assert_equal(st[STAT_RESTORES_NVME], 0)
    assert_equal(st[STAT_FORKS], 0)
    assert_equal(st[STAT_RAM_BYTES], (360 + 400) * 14)
    assert_equal(st[STAT_NVME_BYTES], _file_bytes(s, id_a) + _file_bytes(s, id_b))   # measured on the files themselves
    assert_true(st[STAT_NVME_BYTES] > (360 + 400) * 14)
    assert_equal(st[STAT_RAM_EVICTIONS], 0)
    assert_equal(st[STAT_NVME_EVICTIONS], 0)
    s.close()


def test_desynced_engine_is_restored_even_when_its_kv_is_wrong() raises:      # Review Focus 2, with KV that really differs
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 130)
    _ = _turn(s, e, a, 0, False)
    e.desync = True                          # its length no longer matches its tokens...
    for i in range(len(e.kv)):
        e.kv[i] = 0xEE                       # ...and its KV was overwritten by something else
    assert_equal(s.commit(e), -1)            # a desynced engine is never exported
    var pos = s.index.find(s.live_id)
    assert_equal(s.index.sessions[pos].kv_len(), 600)
    assert_equal(len(s.index.sessions[pos].chunks), 1)
    assert_equal(_prep(s, e, _cat(a, _toks(5, 131)), False), 600)           # rebuilt from the committed copy
    assert_true(e.imports >= 1)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM], 1)
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)
    s.close()


def test_reuse_threshold_and_degenerate_prompts() raises:
    for shared in range(15, 18):                                   # min_reuse is 16
        var s = _store(_dir(), False, 1 << 30)
        var e = FakeEngine(False)
        var a = _toks(300, 140)
        _ = _turn(s, e, a, 0, False)
        var c = _prep(s, e, _cat(sub(a, 0, shared), _toks(20, 141)), False)
        assert_equal(c, shared if shared >= 16 else 0)
        s.close()
    var s2 = _store(_dir(), False, 1 << 30)
    var e2 = FakeEngine(False)
    _ = _turn(s2, e2, _toks(300, 142), 0, False)
    assert_equal(_prep(s2, e2, List[Int](), False), 0)             # nothing to prefill, nothing to reuse
    assert_equal(_prep(s2, e2, _toks(1, 143), False), 0)           # one token is all the caller must prefill
    assert_equal(len(s2.index.sessions), 2)                        # the abandoned empty record did not pile up
    s2.close()


def _switch_case(over: Int) raises -> Int:
    """A (600 tokens) is parked. B shares A's first 100 tokens and is live. The request shares 356 + over tokens with A
    and 100 with B, so switching to A gains 256 + over tokens (min_switch is 256)."""
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 150)
    _ = _turn(s, e, a, 0, False)
    _ = _turn(s, e, _cat(sub(a, 0, 100), _toks(300, 151)), 0, False)
    var c = _prep(s, e, _cat(sub(a, 0, 356 + over), _toks(20, 152)), False)
    s.close()
    return c


def test_switching_needs_more_than_min_switch_tokens_of_gain() raises:
    assert_equal(_switch_case(0), 100)         # a gain of exactly 256 is not enough: the live session is reused
    assert_equal(_switch_case(1), 357)         # one more token and the parked session wins (a partial chunk import)


def test_leaving_a_session_persists_its_uncommitted_tail() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 160)
    _ = _turn(s, e, a, 0, False)
    _ = _turn(s, e, _toks(400, 161), 0, False)
    var id_b = s.live_id
    e.extend(_toks(40, 162), False)                                   # B grows 40 tokens no commit has seen
    var b_more = e.tokens.copy()
    assert_equal(_prep(s, e, _cat(a, _toks(5, 163)), False), 600)    # a switch parks B: its tail is committed first
    assert_equal(s.index.sessions[s.index.find(id_b)].kv_len(), 440)
    assert_equal(_prep(s, e, _cat(b_more, _toks(5, 164)), False), 440)          # and the tail restores
    e.extend(_toks(30, 165), False)                                   # B grows again, uncommitted
    var b_most = e.tokens.copy()
    assert_equal(_prep(s, e, _toks(200, 166), False), 0)             # a fresh start leaves it the same way
    assert_equal(s.index.sessions[s.index.find(id_b)].kv_len(), 470)
    assert_equal(_prep(s, e, _cat(b_most, _toks(5, 167)), False), 470)
    s.close()


def test_full_prompt_of_a_parked_session_returns_n_minus_1_without_fork() raises:      # Review Focus 5, parked
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(300, 270)
    _ = _turn(s, e, a, 0, False)
    var id_a = s.live_id
    _ = _turn(s, e, _toks(300, 271), 0, False)                       # A is parked
    assert_equal(_prep(s, e, a, False), 299)                         # the last token is left to prefill
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)
    assert_equal(s.live_id, id_a)                                    # A continues
    e.extend(sub(a, 299, 300), False)                                # the caller prefills that token
    assert_equal(s.commit(e), 0)                                     # and there is nothing new to save
    s.close()


def test_partial_restore_cuts_inside_a_chunk_and_forks() raises:
    var s = _small_chunk_store(_dir(), True, 1 << 30)
    var e = FakeEngine(True)
    var a = _toks(200, 170)
    _ = _turn(s, e, a, 0, True)
    var id_a = s.live_id
    var pa = s.index.find(id_a)
    assert_equal(len(s.index.sessions[pa].chunks), 4)                # chunks are exactly 64 tokens, then the rest
    for ci in range(4):
        assert_equal(s.index.sessions[pa].chunks[ci].start, ci * 64)
        assert_equal(s.index.sessions[pa].chunks[ci].end, min((ci + 1) * 64, 200))
    _ = _turn(s, e, _toks(100, 171), 0, True)                        # B is live, A parked
    var req = _cat(sub(a, 0, 100), _toks(30, 172))                   # shares 100 tokens: cuts chunk 1 ([64, 128)) in half
    assert_equal(_prep(s, e, req, True), 100)
    assert_equal(e.drafter_len(), 100)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM], 1)
    assert_equal(s.stats_snapshot()[STAT_FORKS], 1)                  # the request leaves A's history: a fork
    assert_true(s.live_id != id_a)
    e.extend(sub(req, 100, 130), True)
    assert_equal(s.commit(e), 130)                                   # the fork saves its whole history as its own session
    assert_equal(s.index.sessions[s.index.find(id_a)].kv_len(), 200)         # A is untouched
    assert_equal(_prep(s, e, _cat(a, _toks(5, 173)), True), 200)
    assert_equal(s.live_id, id_a)
    s.close()


def test_restart_restore_reads_only_the_chunks_it_needs() raises:
    var dir = _dir()
    var s = _small_chunk_store(dir, True, 1 << 30)
    var e = FakeEngine(True)
    var a = _toks(200, 280)
    _ = _turn(s, e, a, 0, True)
    s.close()
    var s2 = _small_chunk_store(dir, True, 1 << 30)
    var e2 = FakeEngine(True)
    assert_equal(_prep(s2, e2, _cat(sub(a, 0, 128), _toks(30, 281)), False), 128)     # ends exactly on a chunk boundary
    assert_equal(s2.stats_snapshot()[STAT_RESTORES_NVME], 1)
    assert_equal(_resident_bytes(s2), 128 * 19)                      # the two chunks it needed, not the third
    assert_equal(s2.ram.used, 128 * 19)
    s2.close()


def test_restore_that_extends_the_session_does_not_fork() raises:
    var s = _store(_dir(), True, 1 << 30)
    var e = FakeEngine(True)
    var a = _toks(300, 180)
    _ = _turn(s, e, a, 0, True)
    var id_a = s.live_id
    _ = _turn(s, e, _toks(300, 181), 0, True)                        # A is parked, its drafter rows complete
    assert_equal(_prep(s, e, _cat(a, _toks(5, 182)), True), 300)
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)                  # nothing diverged: A simply continues
    assert_equal(s.live_id, id_a)
    assert_equal(len(s.index.sessions), 2)
    s.close()


def test_chunks_without_drafter_rows_survive_a_restart() raises:
    var dir = _dir()
    var s = _small_chunk_store(dir, True, 1 << 30)
    var e = FakeEngine(True)
    _ = _turn(s, e, _toks(100, 190), 0, True)                        # a speculative turn: drafter rows for all 100
    e.extend(_toks(150, 191), False)                                 # a base-path turn: the drafter stays at 100
    assert_equal(s.commit(e), 150)                                   # chunks [100,164) [164,228) [228,250), none with drafter rows
    var pos = s.index.find(s.live_id)
    assert_equal(s.index.sessions[pos].drafter_len, 100)
    assert_equal(len(s.index.sessions[pos].chunks), 5)
    for ci in range(2, 5):
        var ch = s.index.sessions[pos].chunks[ci]
        assert_equal(ch.drafter_end, ch.start)                       # an empty drafter range inside [start, end]
    var all_toks = e.tokens.copy()
    s.close()
    var s2 = _small_chunk_store(dir, True, 1 << 30)
    var e2 = FakeEngine(True)
    assert_equal(_prep(s2, e2, _cat(all_toks, _toks(5, 192)), False), 250)    # the whole file reads back
    assert_equal(e2.drafter_len(), 100)
    s2.close()
    var s3 = _small_chunk_store(dir, True, 1 << 30)
    var e3 = FakeEngine(True)
    assert_equal(_prep(s3, e3, _cat(all_toks, _toks(5, 193)), True), 100)    # a speculative request stops at the gap
    assert_equal(s3.stats_snapshot()[STAT_FORKS], 1)
    s3.close()


def test_caught_up_chunks_are_not_written_twice() raises:                 # ledger R25
    var dir = _dir()
    var s = _store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    _ = _turn(s, e, _toks(300, 200), 0, False)         # [0, 300) stays in RAM only
    s.disk.fail_writes = False
    e.extend(_toks(20, 201), False)
    assert_equal(s.commit(e), 20)                      # catches [0, 300) up, then writes [300, 320)
    e.extend(_toks(25, 202), False)
    assert_equal(s.commit(e), 25)                      # nothing is left to catch up: [320, 345) is appended once
    var pos = s.index.find(s.live_id)
    for ci in range(len(s.index.sessions[pos].chunks)):
        assert_true(s.index.sessions[pos].chunks[ci].file_offset > 0)    # every chunk is on disk, and knows it
    var all_toks = e.tokens.copy()
    s.close()
    var s2 = _store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(all_toks, _toks(5, 203)), False), 345)   # a gap-free, duplicate-free file
    s2.close()


def test_ram_pressure_keeps_the_ram_only_tail_of_a_partly_written_session() raises:     # Review Focus 4, mixed
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(300, 260), 0, False)         # [0, 300) is on disk
    var id_a = s.live_id
    s.disk.fail_writes = True
    e.extend(_toks(40, 261), False)
    assert_equal(s.commit(e), 40)                      # [300, 340) stays in RAM only: the disk refuses it
    s.disk.fail_writes = False
    # The disk is back, so parking A below would write its tail out first (commit's catch-up runs even with nothing new).
    # Fail that one write, or the tail would no longer be RAM-only when the pressure comes.
    s.disk.fail_once_at = 0
    var a_full = e.tokens.copy()
    s.cfg.ram_bytes = 560 + 4200 + 100                 # room for A's 40-token tail and one 300-token chunk, not for both of A's
    var b = _toks(300, 262)
    assert_equal(s.prepare(e, b, False), 0)            # parks A (its tail's write fails once) and starts B
    var pa = s.index.find(id_a)
    assert_equal(s.index.sessions[pa].chunks[1].file_offset, -1)       # the premise: A's tail is still unsaved when B's commit starts
    assert_true(s.index.sessions[pa].chunks[1].ram_addr != 0)
    e.prefill(b)
    assert_true(s.commit(e) >= 0)                      # B needs room: A's on-disk chunk goes, its RAM-only tail stays
    assert_true(s.stats_snapshot()[STAT_RAM_EVICTIONS] >= 1)
    assert_equal(_prep(s, e, _cat(a_full, _toks(5, 263)), False), 340)      # the prefix from disk, the tail from RAM
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 1)
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


def test_ram_eviction_takes_the_least_recently_used_session() raises:
    var s = _small_chunk_store(_dir(), False, 2 * 4200, 512)         # room for exactly two 300-token sessions (one chunk each)
    var e = FakeEngine(False)
    var a = _toks(300, 220)
    var b = _toks(300, 221)
    _ = _turn(s, e, a, 0, False)                                      # clock 1
    _ = _turn(s, e, b, 0, False)                                      # clock 2: RAM is exactly full, not over
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 0)
    assert_equal(_prep(s, e, _cat(a, _toks(5, 222)), False), 300)    # clock 3: A is used again, so B is the oldest
    _ = _turn(s, e, _toks(300, 223), 0, False)                        # a third session does not fit
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 1)
    assert_equal(s.ram.used, _resident_bytes(s))
    assert_equal(_prep(s, e, _cat(a, _toks(5, 224)), False), 300)    # A is still in RAM...
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM], 2)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 0)
    assert_equal(_prep(s, e, _cat(b, _toks(5, 225)), False), 300)    # ...and B was the one that went to disk
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 1)
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


def test_restore_keeps_the_restored_session_resident_under_a_tight_budget() raises:
    var s = _small_chunk_store(_dir(), False, 2200)                  # a session is 2100 bytes in three chunks: one fits
    var e = FakeEngine(False)
    var a = _toks(150, 290)
    _ = _turn(s, e, a, 0, False)
    _ = _turn(s, e, _toks(150, 291), 0, False)                        # B's commit pushes A's RAM copy out
    assert_equal(_resident_bytes(s), 150 * 14)                        # only B is resident now
    assert_equal(_prep(s, e, _cat(a, _toks(5, 292)), False), 150)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 1)
    assert_equal(_resident_bytes(s), 150 * 14)                        # all of A: its chunks go last, so B's went first
    assert_equal(s.ram.used, 150 * 14)
    assert_true(s.ram.peak <= 2200)                                   # and the budget held all along
    s.close()


def test_unreadable_session_is_dropped_and_the_engine_reset() raises:
    var dir = _dir()
    var s = _small_chunk_store(dir, False, 1000)                     # RAM far smaller than a session: chunks come from disk
    var e = FakeEngine(False)
    var a = _toks(150, 210)                                           # chunks of 64, 64 and 22 tokens
    _ = _turn(s, e, a, 0, False)
    var id_a = s.live_id
    _ = _turn(s, e, _toks(150, 211), 0, False)                        # B's commit pushes A's RAM copy out
    var id_b = s.live_id
    _flip_byte(s.disk.path(id_a), 40)                                 # inside the last chunk's payload
    var req = _cat(a, _toks(5, 212))
    assert_equal(s.prepare(e, req, False), 0)                         # chunks 0 and 1 import, chunk 2 fails its CRC
    assert_equal(e.kv_len(), 0)                                       # the half-imported prefix is gone
    assert_equal(s.index.find(id_a), -1)                              # A is dropped...
    assert_false(_has_file(s, id_a))                                  # ...with its file
    assert_equal(s.ram.used, _resident_bytes(s))                      # and no read buffer was leaked
    assert_true(s.live_id != 0 and s.live_id != id_b)                 # a fresh session is live
    assert_equal(len(s.index.sessions), 2)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 0)
    e.prefill(req)                                                    # the caller prefills the whole prompt
    assert_equal(s.commit(e), 155)                                    # and it is saved as a session of its own
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)
    assert_equal(len(s.index.sessions), 2)
    s.close()


def test_session_dropped_to_make_room_is_not_restored() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(300, 250)
    _ = _turn(s, e, a, 0, False)
    var id_a = s.live_id
    _ = _turn(s, e, _toks(300, 251), 0, False)                        # B is live, A parked and older
    s.cfg.disk_bytes = s.disk.total_bytes()                           # the disk is exactly full
    e.extend(_toks(40, 252), False)                                   # B grows past its last commit
    # Switching to A commits B's tail first. That pushes the disk over budget and drops A, the oldest, before it restores.
    assert_equal(_prep(s, e, _cat(a, _toks(5, 253)), False), 0)
    assert_equal(s.index.find(id_a), -1)
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 1)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM] + s.stats_snapshot()[STAT_RESTORES_NVME], 0)
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


def test_disk_budget_drops_whole_least_recently_used_sessions() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(300, 230)
    _ = _turn(s, e, a, 0, False)                                      # clock 1
    var id_a = s.live_id
    var per = s.disk.total_bytes()                                    # one session file; the others are the same size
    assert_equal(per, _file_bytes(s, id_a))
    s.cfg.disk_bytes = 3 * per                                        # room for exactly three
    var b = _toks(300, 231)
    _ = _turn(s, e, b, 0, False)                                      # clock 2
    var id_b = s.live_id
    _ = _turn(s, e, _toks(300, 232), 0, False)                        # clock 3: the disk is exactly full, not over
    var id_c = s.live_id
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 0)
    assert_equal(s.disk.total_bytes(), 3 * per)
    assert_equal(_prep(s, e, _cat(a, _toks(5, 233)), False), 300)    # clock 4: A is used again, so B is the oldest
    _ = _turn(s, e, _toks(300, 234), 0, False)                        # clock 5: a fourth session, one file over budget
    var id_d = s.live_id
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 1)
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 0)           # dropping a session frees its RAM but is no eviction
    assert_equal(s.index.find(id_b), -1)                              # B went, whole...
    assert_false(_has_file(s, id_b))                                  # ...file and all
    for sid in [id_a, id_c, id_d]:
        assert_true(s.index.find(sid) >= 0)
        assert_true(_has_file(s, sid))
    assert_equal(s.disk.total_bytes(), 3 * per)
    assert_equal(s.stats_snapshot()[STAT_NVME_BYTES], 3 * per)
    assert_equal(s.ram.used, 3 * 300 * 14)                            # B's RAM copy went with it
    assert_equal(s.ram.used, _resident_bytes(s))
    assert_equal(_prep(s, e, _cat(b, _toks(5, 235)), False), 0)      # B is gone for good
    assert_equal(_prep(s, e, _cat(a, _toks(5, 236)), False), 300)    # A is still there
    s.close()


def test_disk_budget_never_drops_the_live_or_an_unwritten_session() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var x = _toks(300, 240)
    s.disk.fail_writes = True
    _ = _turn(s, e, x, 0, False)                                      # X exists only in RAM: its file was never written
    s.disk.fail_writes = False
    var id_x = s.live_id
    s.cfg.disk_bytes = 100                                            # smaller than any session file
    s.disk.fail_once_at = 0                                           # parking X tries to write it out (the next write): that fails once
    _ = _turn(s, e, _toks(300, 241), 0, False)
    var id_a = s.live_id
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 0)          # A is live, and X has no file to drop
    assert_true(s.index.find(id_a) >= 0 and _has_file(s, id_a))
    assert_true(s.index.find(id_x) >= 0)
    _ = _turn(s, e, _toks(250, 242), 0, False)                        # Y is live now; A is parked and over budget
    var id_y = s.live_id
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 1)
    assert_equal(s.index.find(id_a), -1)
    assert_true(s.index.find(id_y) >= 0 and _has_file(s, id_y))       # Y alone exceeds the budget and stays
    assert_true(s.index.find(id_x) >= 0)                              # X stays: dropping it would free no disk
    assert_equal(_prep(s, e, _cat(x, _toks(5, 243)), False), 300)    # and it still restores, from RAM
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


struct FlakyEngine(KvSessionEngine):
    """A FakeEngine whose Nth export or import raises, as a failed CUDA copy would (N counts from 0; -1 = never)."""
    var inner: FakeEngine
    var fail_export_at: Int
    var fail_import_at: Int
    var exports: Int
    var imports_seen: Int

    def __init__(out self, drafter: Bool):
        self.inner = FakeEngine(drafter)
        self.fail_export_at = -1
        self.fail_import_at = -1
        self.exports = 0
        self.imports_seen = 0

    def kv_len(self) -> Int:
        return self.inner.kv_len()

    def drafter_len(self) -> Int:
        return self.inner.drafter_len()

    def live_tokens_copy(self) -> List[Int]:
        return self.inner.live_tokens_copy()

    def export_chunk(mut self, layout: KvLayout, a: Int, b: Int, drafter_end: Int, dst: UInt64) raises:
        self.exports += 1
        if self.exports - 1 == self.fail_export_at:
            raise Error("simulated export failure")
        self.inner.export_chunk(layout, a, b, drafter_end, dst)

    def import_chunk(mut self, layout: KvLayout, a: Int, b: Int, drafter_end: Int, src: UInt64, src_tokens: Int, tokens: List[Int]) raises:
        self.imports_seen += 1
        if self.imports_seen - 1 == self.fail_import_at:
            raise Error("simulated import failure")
        self.inner.import_chunk(layout, a, b, drafter_end, src, src_tokens, tokens)

    def truncate_to(mut self, p: Int) raises:
        self.inner.truncate_to(p)

    def reset_session(mut self) raises:
        self.inner.reset_session()


def test_failed_export_leaks_no_ram() raises:
    var s = _small_chunk_store(_dir(), False, 1 << 30)
    var e = FlakyEngine(False)
    var a = _toks(150, 310)                                           # chunks of 64, 64 and 22 tokens
    assert_equal(s.prepare(e, a, False), 0)
    e.inner.prefill(a)
    e.fail_export_at = 1                                              # the second chunk's export fails
    var raised = False
    try:
        _ = s.commit(e)
    except:
        raised = True
    assert_true(raised)
    assert_equal(s.ram.used, 64 * 14)                                 # chunk 0 is saved; chunk 1's buffer was given back
    assert_equal(s.ram.used, _resident_bytes(s))
    assert_equal(s.index.sessions[s.index.find(s.live_id)].kv_len(), 64)
    e.fail_export_at = -1
    assert_equal(s.commit(e), 86)                                     # the next commit resumes where this one stopped
    assert_equal(s.ram.used, 150 * 14)
    s.close()
    assert_equal(s.ram.used, 0)


def test_failed_import_leaves_a_store_that_can_retry() raises:
    var s = _small_chunk_store(_dir(), False, 1000)                   # chunks come back from disk
    var e = FlakyEngine(False)
    var a = _toks(150, 320)
    assert_equal(s.prepare(e, a, False), 0)
    e.inner.prefill(a)
    assert_equal(s.commit(e), 150)
    var b = _toks(150, 321)
    assert_equal(s.prepare(e, b, False), 0)
    e.inner.prefill(b)
    assert_equal(s.commit(e), 150)                                    # A's RAM copy was pushed out for B
    e.fail_import_at = e.imports_seen + 1                             # the restore fails on its second chunk
    var req = _cat(a, _toks(5, 322))
    var raised = False
    try:
        _ = s.prepare(e, req, False)
    except:
        raised = True
    assert_true(raised)
    assert_equal(s.ram.used, _resident_bytes(s))                      # nothing was leaked on the way out
    e.fail_import_at = -1
    assert_equal(s.prepare(e, req, False), 150)                       # a retry rebuilds the whole prefix
    assert_equal(e.kv_len(), 150)
    for i in range(150):
        assert_equal(e.inner.tokens[i], req[i])
    assert_true(e.inner.expected_ok())
    assert_equal(s.ram.used, _resident_bytes(s))
    assert_equal(len(s.index.sessions), 2)                            # A and B: the half-restored prefix was not saved as a session
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)
    s.close()
    assert_equal(s.ram.used, 0)


# ── Fix round 1, ruling R27: the RAM budget is hard ──────────────────────────────────────────────────────────────


def _refuses(cfg: StoreConfig, variable: String) raises:
    """SessionStore must refuse this config before it touches the disk, and the message must name the variable."""
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var raised = False
    try:
        var s = SessionStore(key, lay, cfg.copy())
        s.close()
    except ex:
        raised = True
        assert_true(variable in String(ex))
    assert_true(raised)
    assert_false(exists(cfg.dir))              # validation comes first: no directory, no lock


def test_ram_budget_smaller_than_one_chunk_is_refused() raises:
    # 64 tokens x 14 bytes = 896 bytes per full chunk
    _refuses(StoreConfig(_dir(), 895, 1 << 30, 16, 16, False, 64), "NOMOS_KVSTORE_RAM_GB")
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var ok = SessionStore(key, lay, StoreConfig(_dir(), 896, 1 << 30, 16, 16, False, 64))   # exactly one chunk is enough
    ok.close()


def test_a_long_live_session_stays_inside_a_two_chunk_ram_budget() raises:
    var dir = _dir()
    var budget = 2 * 896 + 100                                        # two 64-token chunks
    var s = _small_chunk_store(dir, False, budget)
    var e = FakeEngine(False)
    var first = _toks(100, 330)
    assert_equal(_prep(s, e, first, False), 0)
    e.prefill(first)
    assert_equal(s.commit(e), 100)
    assert_true(s.ram.used <= budget)
    for turn in range(10):
        var ids = _cat(e.tokens, _toks(70, 331 + turn))
        var c = _prep(s, e, ids, False)                                 # the live session continues: nothing is copied
        e.extend(sub(ids, c, len(ids)), False)
        e.extend(_toks(30, 360 + turn), False)                          # 70 prompt + 30 generated tokens
        assert_equal(s.commit(e), 100)
        assert_true(s.ram.used <= budget)                               # after every commit
    var pos = s.index.find(s.live_id)
    assert_true(len(s.index.sessions[pos].chunks) >= 16)                # a long session, a small part of it resident
    assert_true(s.ram.peak <= budget)                                   # and never above it, not even mid-commit
    var all_toks = e.tokens.copy()
    s.close()
    var s2 = _small_chunk_store(dir, False, budget)                     # a restart restores it exactly
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(all_toks, _toks(5, 399)), False), len(all_toks))
    assert_true(s2.ram.peak <= budget)
    s2.close()


def test_a_session_larger_than_the_ram_budget_restores_exactly_from_disk() raises:
    var dir = _dir()
    var budget = 2 * 896 + 100
    var s = _small_chunk_store(dir, False, budget)
    var e = FakeEngine(False)
    var a = _toks(300, 400)                                             # five chunks, 4200 bytes: more than twice the budget
    _ = _turn(s, e, a, 0, False)
    var id_a = s.live_id
    _ = _turn(s, e, _toks(300, 401), 0, False)                          # B takes the RAM A had left
    assert_true(s.ram.peak <= budget)
    assert_equal(_prep(s, e, _cat(a, _toks(5, 402)), False), 300)       # A comes back from disk, a chunk at a time
    assert_equal(s.stats_snapshot()[STAT_RESTORES_NVME], 1)
    assert_equal(s.live_id, id_a)
    assert_true(s.ram.peak <= budget)
    s.close()
    var s2 = _small_chunk_store(dir, False, budget)                     # and again after a restart, with a fresh peak
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(a, _toks(5, 403)), False), 300)
    assert_true(s2.ram.peak <= budget)
    assert_true(s2.ram.used <= budget)
    s2.close()


def test_a_failing_disk_cannot_grow_pinned_memory_past_the_budget() raises:
    var dir = _dir()
    var budget = 3 * 896 + 100                                          # three 64-token chunks
    var s = _small_chunk_store(dir, False, budget)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var first = _toks(64, 410)
    assert_equal(_prep(s, e, first, False), 0)
    e.prefill(first)
    assert_equal(s.commit(e), 64)                                       # one chunk, in RAM only
    for turn in range(2):                                               # two more turns fill the budget with unsaved chunks
        var nxt = _cat(e.tokens, _toks(30, 411 + turn))
        var c = _prep(s, e, nxt, False)
        e.extend(sub(nxt, c, len(nxt)), False)
        e.extend(_toks(34, 420 + turn), False)
        assert_equal(s.commit(e), 64)
    assert_equal(s.ram.used, 3 * 896)
    var id_a = s.live_id
    for turn in range(3):                                               # the disk is still down: nothing more can be pinned
        var nxt = _cat(e.tokens, _toks(30, 430 + turn))
        var c = _prep(s, e, nxt, False)
        e.extend(sub(nxt, c, len(nxt)), False)
        e.extend(_toks(34, 440 + turn), False)
        assert_equal(s.commit(e), -3)                                   # "not saved", and memory did not grow
        assert_equal(s.ram.used, 3 * 896)
    assert_true(s.ram.peak <= budget)
    var a_all = e.tokens.copy()                                         # 192 saved, 384 in the engine
    assert_equal(s.index.sessions[s.index.find(id_a)].kv_len(), 192)
    # The saved part stays restorable: another conversation starts (it cannot be saved either), then A comes back.
    var x = _toks(100, 450)
    assert_equal(_prep(s, e, x, False), 0)
    e.prefill(x)
    assert_equal(s.commit(e), -3)
    assert_equal(_prep(s, e, _cat(sub(a_all, 0, 192), _toks(5, 451)), False), 192)
    assert_equal(s.live_id, id_a)
    # The disk recovers: the next commit writes the unsaved chunks out, then goes on saving.
    s.disk.fail_writes = False
    var ids3 = _cat(a_all, _toks(10, 452))                              # the client resends the whole conversation
    var c3 = _prep(s, e, ids3, False)
    assert_equal(c3, 192)
    e.extend(sub(ids3, c3, len(ids3)), False)
    e.extend(_toks(20, 453), False)
    var kv = len(e.tokens)
    assert_equal(s.commit(e), kv - 192)
    assert_true(s.ram.used <= budget)
    assert_true(s.ram.peak <= budget)
    var pa = s.index.find(id_a)
    assert_equal(s.index.sessions[pa].kv_len(), kv)
    for ci in range(len(s.index.sessions[pa].chunks)):
        assert_true(s.index.sessions[pa].chunks[ci].file_offset > 0)    # every chunk has a disk copy now
    var saved = e.tokens.copy()
    s.close()
    var s2 = _small_chunk_store(dir, False, budget)                     # a restart restores everything saved
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(saved, _toks(5, 454)), False), kv)
    s2.close()


def test_a_parked_sessions_unsaved_chunks_cannot_starve_the_live_session() raises:
    var dir = _dir()
    var budget = 3 * 896 + 100
    var s = _small_chunk_store(dir, False, budget)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var a = _toks(192, 460)                                             # A: three chunks, none saved: RAM is full
    assert_equal(_prep(s, e, a, False), 0)
    e.prefill(a)
    assert_equal(s.commit(e), 192)
    var id_a = s.live_id
    assert_equal(s.ram.used, 3 * 896)
    var b = _toks(150, 461)                                             # B starts, A is parked
    assert_equal(_prep(s, e, b, False), 0)
    e.prefill(b)
    assert_equal(s.commit(e), -3)                                       # the disk is down and A holds the whole budget
    s.disk.fail_writes = False                                          # the disk recovers
    assert_equal(s.commit(e), 150)                                      # B's next commit succeeds: A is written out, then evicted
    assert_true(s.ram.used <= budget)
    assert_true(s.ram.peak <= budget)
    var pa = s.index.find(id_a)
    for ci in range(len(s.index.sessions[pa].chunks)):
        assert_true(s.index.sessions[pa].chunks[ci].file_offset > 0)    # A, parked, was persisted along the way
    var b_all = e.tokens.copy()
    s.close()
    var s2 = _small_chunk_store(dir, False, budget)                     # both survive a restart
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(a, _toks(5, 462)), False), 192)
    assert_equal(_prep(s2, e2, _cat(b_all, _toks(5, 463)), False), 150)
    s2.close()


# ── Fix round 1, ruling R27 (continued): close() and the eviction count ──────────────────────────────────────────


def test_close_writes_out_unsaved_chunks_if_the_disk_has_recovered() raises:
    var dir = _dir()
    var s = _small_chunk_store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var a = _toks(150, 470)
    _ = _turn(s, e, a, 0, False)                      # three chunks, none saved
    s.disk.fail_writes = False                        # the disk recovers, but nobody commits again
    s.close()                                         # close() makes a last try before it frees them
    var s2 = _small_chunk_store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(a, _toks(5, 471)), False), 150)
    s2.close()


def test_ram_evictions_count_chunks_lowest_first_and_not_closes() raises:
    var s = _small_chunk_store(_dir(), False, 3 * 896 + 100)          # three 64-token chunks
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(192, 480), 0, False)                         # A: three chunks, the budget is full
    var id_a = s.live_id
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 0)
    _ = _turn(s, e, _toks(128, 481), 0, False)                         # B: two chunks, so two of A's chunks must go
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 2)           # one count per chunk freed, not per session
    var pa = s.index.find(id_a)
    assert_equal(s.index.sessions[pa].chunks[0].ram_addr, UInt64(0))  # the lowest chunks went first...
    assert_equal(s.index.sessions[pa].chunks[1].ram_addr, UInt64(0))
    assert_true(s.index.sessions[pa].chunks[2].ram_addr != 0)         # ...and the newest stays
    assert_true(s.ram.peak <= 3 * 896 + 100)
    s.close()
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 2)           # freeing at close is not an eviction


# ── Fix round 1, ruling R27 (continued): several evictions, a partial commit, a second close() ───────────────────


def test_one_big_chunk_can_evict_several_small_ones() raises:
    var s = _small_chunk_store(_dir(), False, 1600)                    # three 36-token chunks (504 bytes) fit; a 64-token one is 896
    var e = FakeEngine(False)
    var ids = _toks(36, 500)
    assert_equal(_prep(s, e, ids, False), 0)
    e.prefill(ids)
    assert_equal(s.commit(e), 36)
    for turn in range(2):
        var nxt = _cat(e.tokens, _toks(20, 501 + turn))
        var c = _prep(s, e, nxt, False)
        e.extend(sub(nxt, c, len(nxt)), False)
        e.extend(_toks(16, 510 + turn), False)
        assert_equal(s.commit(e), 36)
    assert_equal(s.ram.used, 3 * 504)
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 0)
    var nxt = _cat(e.tokens, _toks(40, 520))
    var c = _prep(s, e, nxt, False)
    e.extend(sub(nxt, c, len(nxt)), False)
    e.extend(_toks(24, 521), False)
    assert_equal(s.commit(e), 64)                                      # 896 bytes needed, 88 free: two 504-byte chunks must go
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 2)
    assert_equal(s.ram.used, 504 + 896)
    assert_true(s.ram.peak <= 1600)
    s.close()


def test_a_commit_that_runs_out_of_ram_reports_what_it_saved() raises:
    var budget = 3 * 896 + 100
    var s = _small_chunk_store(_dir(), False, budget)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var ids = _toks(320, 490)                                          # five chunks, room for three
    assert_equal(_prep(s, e, ids, False), 0)
    e.prefill(ids)
    assert_equal(s.commit(e), 192)                                     # three chunks saved, then RAM was full: 192, not -3
    assert_equal(s.index.sessions[s.index.find(s.live_id)].kv_len(), 192)
    assert_equal(s.commit(e), -3)                                      # nothing more fits
    s.disk.fail_writes = False
    assert_equal(s.commit(e), 128)                                     # the disk is back: the three go out, the rest is saved
    assert_true(s.ram.peak <= budget)
    s.close()


def test_closing_twice_is_harmless() raises:
    var s = _small_chunk_store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    _ = _turn(s, e, _toks(150, 530), 0, False)                         # unsaved chunks
    s.close()                                                          # the disk is down: they are lost, and freed
    s.disk.fail_writes = False
    s.close()                                                          # nothing left to save, nothing to free
    assert_equal(s.ram.used, 0)


def test_restore_degrades_to_a_fresh_start_when_ram_is_full_of_unsaved_chunks() raises:
    var budget = 3 * 896 + 100
    var s = _small_chunk_store(_dir(), False, budget)
    var e = FakeEngine(False)
    var b = _toks(150, 540)
    _ = _turn(s, e, b, 0, False)                                       # B is saved, on a healthy disk
    var id_b = s.live_id
    s.disk.fail_writes = True
    _ = _turn(s, e, _toks(192, 541), 0, False)                         # A: three unsaved chunks take the whole budget (B's RAM goes)
    assert_equal(s.ram.used, 3 * 896)
    var req = _cat(b, _toks(5, 542))
    assert_equal(s.prepare(e, req, False), 0)                          # B cannot come back: there is no room and none can be made
    assert_equal(e.kv_len(), 0)                                        # the engine was reset, not left half-restored
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM] + s.stats_snapshot()[STAT_RESTORES_NVME], 0)
    assert_true(s.index.find(id_b) >= 0)                               # and B is still there
    assert_equal(s.ram.used, _resident_bytes(s))
    s.disk.fail_writes = False                                         # the disk recovers
    assert_equal(_prep(s, e, req, False), 150)                         # now A's chunks are written out to make room
    assert_true(s.ram.peak <= budget)
    s.close()


# ── Fix round 1 (Task 8 review, Minor 1): a session file deleted under a running store ──────────────────────────


def test_a_session_file_deleted_by_hand_does_not_wedge_the_disk_budget() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(300, 600), 0, False)                         # A, the oldest
    var id_a = s.live_id
    var per = s.disk.total_bytes()
    _ = _turn(s, e, _toks(300, 601), 0, False)
    var id_b = s.live_id
    _ = _turn(s, e, _toks(300, 602), 0, False)
    var id_c = s.live_id
    remove(s.disk.path(id_a))                                          # an operator frees space by hand
    s.cfg.disk_bytes = 2 * per + per // 2                              # two files and a half; the store still counts A's
    _ = _turn(s, e, _toks(300, 603), 0, False)                         # the next commit must succeed
    var id_d = s.live_id
    assert_equal(s.index.find(id_a), -1)                               # the deleted session left the index...
    assert_equal(s.index.find(id_b), -1)                               # ...and the budget went on to the next oldest
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 2)
    assert_true(s.index.find(id_c) >= 0 and s.index.find(id_d) >= 0)
    assert_equal(s.disk.total_bytes(), 2 * per)                        # the books match the files again
    assert_equal(s.ram.used, _resident_bytes(s))
    _ = _turn(s, e, _toks(300, 604), 0, False)                         # later commits keep evicting normally
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 3)
    assert_equal(s.index.find(id_c), -1)
    assert_equal(s.disk.total_bytes(), 2 * per)
    s.close()


def test_a_failed_disk_drop_leaves_the_session_and_its_unsaved_chunks_alone() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(300, 610), 0, False)                         # M: [0, 300) on disk
    var id_m = s.live_id
    s.disk.fail_writes = True
    e.extend(_toks(40, 611), False)
    assert_equal(s.commit(e), 40)                                      # [300, 340) exists only in RAM
    var path = s.disk.path(id_m)
    remove(path)
    makedirs(path)                                                     # a directory where the file was: removing it fails
    s.cfg.disk_bytes = 1                                               # the budget wants M gone
    var x = _toks(100, 612)
    assert_equal(s.prepare(e, x, False), 0)                            # N starts, M is parked
    e.prefill(x)
    var raised = False
    try:
        _ = s.commit(e)                                                # N is saved, then dropping M fails
    except:
        raised = True
    assert_true(raised)
    var pm = s.index.find(id_m)
    assert_true(pm >= 0)                                               # M is still in the index...
    assert_true(s.index.sessions[pm].chunks[1].ram_addr != 0)          # ...with its unsaved tail still in RAM
    assert_equal(s.index.sessions[pm].chunks[1].file_offset, -1)
    assert_equal(s.ram.used, _resident_bytes(s))
    rmdir(path)
    s.close()


# ── Fix round 1 (Task 8 review, Minors 2 and 5): what SessionStore.__init__ does with a bad config and a failed scan ──────


def test_a_scan_that_raises_releases_the_directory_lock() raises:
    var dir = _dir()
    makedirs(dir, mode=0o300, exist_ok=True)               # -wx: the lock file can be made, the directory cannot be listed
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    try:
        var raised = False
        try:
            var s = SessionStore(key, lay, StoreConfig(dir, 1 << 30, 1 << 30, 16, 256, False, 4096))
            s.close()
        except ex:
            raised = True
            assert_true("Permission denied" in String(ex))     # scan()'s own error comes through unchanged
        assert_true(raised)
        var d = DiskTier(dir, key, lay)                    # the lock was let go: a retry in this process can take it
        d.close()
    finally:
        # The directory has no read permission, so a later `rm -rf` of build/ could not empty it: remove it here, even
        # when an assertion above failed.
        if exists(dir + "/.lock"):
            remove(dir + "/.lock")
        rmdir(dir)


def test_chunk_tokens_must_be_positive() raises:
    _refuses(StoreConfig(_dir(), 1 << 30, 1 << 30, 16, 16, False, 0), "NOMOS_KVSTORE_CHUNK_TOKENS")
    _refuses(StoreConfig(_dir(), 1 << 30, 1 << 30, 16, 16, False, -64), "NOMOS_KVSTORE_CHUNK_TOKENS")
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var ok = SessionStore(key, lay, StoreConfig(_dir(), 1 << 30, 1 << 30, 16, 16, False, 1))     # one token per chunk is allowed
    ok.close()


def test_min_reuse_must_be_at_least_one() raises:
    _refuses(StoreConfig(_dir(), 1 << 30, 1 << 30, 0, 16, False, 64), "NOMOS_KVSTORE_MIN_REUSE")
    _refuses(StoreConfig(_dir(), 1 << 30, 1 << 30, -3, 16, False, 64), "NOMOS_KVSTORE_MIN_REUSE")
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var ok = SessionStore(key, lay, StoreConfig(_dir(), 1 << 30, 1 << 30, 1, 16, False, 64))
    ok.close()


def test_min_switch_must_not_be_negative() raises:
    _refuses(StoreConfig(_dir(), 1 << 30, 1 << 30, 16, -1, False, 64), "NOMOS_KVSTORE_MIN_SWITCH")
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var ok = SessionStore(key, lay, StoreConfig(_dir(), 1 << 30, 1 << 30, 16, 0, False, 64))    # 0: switch for any gain
    ok.close()


def test_the_disk_budget_must_not_be_negative() raises:
    _refuses(StoreConfig(_dir(), 1 << 30, -1, 16, 16, False, 64), "NOMOS_KVSTORE_DISK_GB")
    var lay = _layout(False)
    var key = CacheKey(2, UInt64(1), 1, lay.hash(), UInt64(0), UInt64(0))
    var ok = SessionStore(key, lay, StoreConfig(_dir(), 1 << 30, 0, 16, 16, False, 64))         # 0: only the live session stays
    ok.close()


# ── Fix round 1 (Task 8 review, Minor 7): the live record is left out of the parked search only while the engine holds it ──


def test_an_engine_reset_behind_the_stores_back_costs_no_reuse() raises:
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(600, 700)
    _ = _turn(s, e, a, 0, False)
    var id_a = s.live_id
    e.reset_session()                                                  # an empty engine, still "in step" with its (empty) tokens
    assert_equal(_prep(s, e, _cat(a, _toks(5, 701)), False), 600)      # A is restored from its committed chunks
    assert_equal(s.stats_snapshot()[STAT_FORKS], 0)
    assert_equal(s.stats_snapshot()[STAT_RESTORES_RAM], 1)
    assert_equal(s.live_id, id_a)
    assert_equal(len(s.index.sessions), 1)
    s.close()


def _refilled_case(session_tokens: Int, refill_tokens: Int, salt: Int) raises:
    """A is committed; the engine is then reset and refilled with unrelated tokens through the tracked path."""
    var s = _store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    var a = _toks(session_tokens, salt)
    _ = _turn(s, e, a, 0, False)
    var id_a = s.live_id
    var other = _toks(refill_tokens, salt + 1)
    e.reset_session()
    e.prefill(other)
    assert_equal(_prep(s, e, _cat(a, _toks(5, salt + 2)), False), session_tokens)    # A comes back, exactly...
    assert_equal(s.stats_snapshot()[STAT_FORKS], 1)                    # ...and the refilled tokens were saved as a session of their own
    assert_equal(s.live_id, id_a)
    assert_equal(len(s.index.sessions), 2)
    var fork_pos = 1 - s.index.find(id_a)
    assert_equal(s.index.sessions[fork_pos].kv_len(), refill_tokens)
    assert_equal(_prep(s, e, _cat(other, _toks(5, salt + 3)), False), refill_tokens)    # and they are there to be restored
    s.close()


def test_an_engine_refilled_behind_the_stores_back_saves_the_new_tokens_and_restores_the_session() raises:
    _refilled_case(600, 400, 710)       # the engine now holds fewer tokens than the session
    _refilled_case(300, 400, 720)       # ...and more: long enough, but not the session's tokens


# ── Fix round 1 (Task 8 review, Minors 8 and 9): the two R25 clauses, and what commit does when there is nothing new ──────


def test_after_a_failed_append_the_rest_of_the_commit_stays_off_the_disk() raises:     # ledger R25
    var dir = _dir()
    var s = _small_chunk_store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    var ids = _toks(192, 800)                                          # three 64-token chunks
    assert_equal(_prep(s, e, ids, False), 0)
    e.prefill(ids)
    s.disk.fail_once_at = 2                                            # create is write 0, chunk 0's append write 1, chunk 1's write 2
    assert_equal(s.commit(e), 192)
    assert_equal(s.disk.fail_once_at, -1)                              # the hook fired once and the disk works again
    var pos = s.index.find(s.live_id)
    assert_true(s.index.sessions[pos].chunks[0].file_offset > 0)       # chunk 0 is on disk
    assert_equal(s.index.sessions[pos].chunks[1].file_offset, -1)      # chunk 1's append failed
    assert_equal(s.index.sessions[pos].chunks[2].file_offset, -1)      # chunk 2 was not even tried, though the disk is back
    s.disk.fail_writes = True                                          # so that close() cannot write them out either
    s.close()
    var s2 = _small_chunk_store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(ids, _toks(5, 801)), False), 64)   # the file is a gap-free prefix: chunk 0
    s2.close()


def test_a_catch_up_that_fails_stops_there() raises:                   # ledger R25
    var dir = _dir()
    var s = _small_chunk_store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    var ids = _toks(200, 810)                                          # chunks of 64, 64, 64 and 8 tokens, all on disk
    assert_equal(_prep(s, e, ids, False), 0)
    e.prefill(ids)
    assert_equal(s.commit(e), 200)
    s.disk.fail_writes = True
    e.extend(_toks(100, 811), False)
    assert_equal(s.commit(e), 100)                                     # chunks [200, 264) and [264, 300) are in RAM only
    s.disk.fail_writes = False
    s.disk.fail_once_at = 0                                            # the catch-up's first append fails, once
    e.extend(_toks(50, 812), False)
    assert_equal(s.commit(e), 50)
    var pos = s.index.find(s.live_id)
    assert_equal(len(s.index.sessions[pos].chunks), 7)
    for ci in range(4):
        assert_true(s.index.sessions[pos].chunks[ci].file_offset > 0)
    for ci in range(4, 7):                                             # the later unsaved chunk and the new one stayed off the disk too
        assert_equal(s.index.sessions[pos].chunks[ci].file_offset, -1)
    var all_toks = e.tokens.copy()
    s.disk.fail_writes = True
    s.close()
    var s2 = _small_chunk_store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(all_toks, _toks(5, 813)), False), 200)    # a gap-free prefix again
    s2.close()


def test_commit_with_nothing_new_still_writes_out_unsaved_chunks() raises:
    var dir = _dir()
    var s = _small_chunk_store(dir, False, 1 << 30)
    var e = FakeEngine(False)
    s.disk.fail_writes = True
    var ids = _toks(150, 820)
    _ = _turn(s, e, ids, 0, False)                                     # three chunks, none saved
    s.disk.fail_writes = False                                         # the disk is back
    assert_equal(s.commit(e), 0)                                       # nothing new to save, but the old chunks go out
    var pos = s.index.find(s.live_id)
    for ci in range(3):
        assert_true(s.index.sessions[pos].chunks[ci].file_offset > 0)
    s.disk.fail_writes = True                                          # close() must not be what saved them
    s.close()
    var s2 = _small_chunk_store(dir, False, 1 << 30)
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(ids, _toks(5, 821)), False), 150)
    s2.close()


def test_a_partial_commit_still_refreshes_lru_and_enforces_the_disk_budget() raises:
    var budget = 3 * 896 + 100
    var s = _small_chunk_store(_dir(), False, budget)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(100, 830), 0, False)                         # P: parked on disk, older
    var id_p = s.live_id
    s.disk.fail_writes = True                                          # nothing S saves can reach the disk
    s.cfg.disk_bytes = 1                                               # and the disk budget wants P gone
    var ids = _toks(320, 831)                                          # five chunks, room for three
    assert_equal(s.prepare(e, ids, False), 0)
    e.prefill(ids)
    s.clock += 10                                                      # time passes between prepare and commit
    assert_equal(s.commit(e), 192)                                     # three chunks saved, then RAM was full
    var pos = s.index.find(s.live_id)
    assert_equal(s.index.sessions[pos].last_used, s.clock)             # the partial commit refreshed S...
    assert_equal(s.index.find(id_p), -1)                               # ...and enforced the disk budget: P is gone
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 1)
    s.close()


# ── Fix round 2 (re-review, Minor 9): the export-raise exit of commit does the bookkeeping too ──────────────────────────


def _partly_saved_commit_setup(mut s: SessionStore, mut e: FlakyEngine, fail_chunk: Int) raises -> UInt64:
    """P is saved and parked, the disk budget wants it gone, and S (150 tokens: chunks of 64, 64 and 22) is prefilled with
    the export of its chunk number fail_chunk set to fail. Returns P's id."""
    var p = _toks(100, 900)
    assert_equal(s.prepare(e, p, False), 0)
    e.inner.prefill(p)
    assert_equal(s.commit(e), 100)
    var id_p = s.live_id
    s.cfg.disk_bytes = 1                                              # only _finish_commit enforces the disk budget
    var ids = _toks(150, 901)
    assert_equal(s.prepare(e, ids, False), 0)
    e.inner.prefill(ids)
    e.fail_export_at = e.exports + fail_chunk                         # that chunk's export fails
    s.clock += 10                                                     # time passes between prepare and commit
    return id_p


def test_a_failed_export_after_a_partial_save_still_does_the_bookkeeping() raises:
    var s = _small_chunk_store(_dir(), False, 1 << 30)
    var e = FlakyEngine(False)
    var id_p = _partly_saved_commit_setup(s, e, 1)                    # the second chunk's export fails
    var raised = False
    try:
        _ = s.commit(e)
    except ex:
        raised = True
        assert_true("simulated export failure" in String(ex))
    assert_true(raised)
    var pos = s.index.find(s.live_id)
    assert_equal(s.index.sessions[pos].kv_len(), 64)                  # chunk 0 is saved
    assert_equal(s.index.sessions[pos].last_used, s.clock)            # the session was stamped...
    assert_equal(s.index.find(id_p), -1)                              # ...and the disk budget was enforced: P is gone
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 1)
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


def test_a_failed_export_still_reports_the_export_error_when_the_bookkeeping_fails() raises:
    var s = _small_chunk_store(_dir(), False, 1 << 30)
    var e = FlakyEngine(False)
    var id_p = _partly_saved_commit_setup(s, e, 1)
    var path = s.disk.path(id_p)
    remove(path)
    makedirs(path)                                                    # a directory where P's file was: dropping P fails
    try:
        var msg = String("")
        try:
            _ = s.commit(e)
        except ex:
            msg = String(ex)
        assert_true("simulated export failure" in msg)                # the caller sees the export error...
        assert_false("Can not remove" in msg)                         # ...not the bookkeeping's
        assert_true(s.index.find(id_p) >= 0)                          # P could not be dropped
        assert_equal(s.ram.used, _resident_bytes(s))
    finally:
        rmdir(path)
    s.close()


def test_a_failed_export_with_nothing_saved_does_no_bookkeeping() raises:
    var s = _small_chunk_store(_dir(), False, 1 << 30)
    var e = FlakyEngine(False)
    var id_p = _partly_saved_commit_setup(s, e, 0)                    # the very first chunk's export fails
    var id_s = s.live_id
    var stamp = s.index.sessions[s.index.find(id_s)].last_used
    var raised = False
    try:
        _ = s.commit(e)
    except:
        raised = True
    assert_true(raised)
    var pos = s.index.find(id_s)                                      # (a dropped session would shift it: look it up again)
    assert_equal(s.index.sessions[pos].kv_len(), 0)                   # nothing was saved...
    assert_equal(s.index.sessions[pos].last_used, stamp)              # ...so the session is not stamped
    assert_true(s.index.find(id_p) >= 0)                              # and the disk budget did not run
    assert_equal(s.stats_snapshot()[STAT_NVME_EVICTIONS], 0)
    assert_equal(s.ram.used, 100 * 14)                                # only P's chunks: the failed chunk's buffer was freed
    s.close()


# ── Fix round 2 (re-review, new breakage 2): a chunk larger than the whole RAM budget ──────────────────────────────────


def test_a_chunk_bigger_than_the_ram_budget_drops_its_session_and_flushes_nothing() raises:
    var dir = _dir()
    var s1 = _small_chunk_store(dir, False, 1 << 30)                   # a run with 64-token chunks writes B
    var e1 = FakeEngine(False)
    var b = _toks(150, 950)                                            # chunks of 64, 64 and 22 tokens: 896, 896 and 308 bytes
    _ = _turn(s1, e1, b, 0, False)
    var id_b = s1.live_id
    s1.close()
    # The next run has 16-token chunks (224 bytes) and a 400-byte budget, which passes the check in __init__. The cache
    # key does not include chunk_tokens, so B's 896-byte chunks are still loaded, and none of them can ever fit.
    var s = _small_chunk_store(dir, False, 400, 16)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(16, 951), 0, False)                          # C: one 224-byte chunk, resident and on disk
    var id_c = s.live_id
    var pc = s.index.find(id_c)
    assert_true(s.index.sessions[pc].chunks[0].ram_addr != 0)
    assert_true(s.index.sessions[pc].chunks[0].file_offset > 0)
    assert_equal(s.prepare(e, _cat(sub(b, 0, 100), _toks(10, 952)), False), 0)    # B's prefix cannot be restored here
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 0)            # and nothing was flushed to try
    pc = s.index.find(id_c)
    assert_true(s.index.sessions[pc].chunks[0].ram_addr != 0)          # C's chunk is still resident
    assert_equal(s.index.find(id_b), -1)                               # B is gone: from the index...
    assert_false(_has_file(s, id_b))                                   # ...and from disk
    assert_equal(e.kv_len(), 0)
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


def test_a_chunk_exactly_as_big_as_the_budget_fits() raises:
    var dir = _dir()
    var s = _small_chunk_store(dir, False, 896)                        # one 64-token chunk, to the byte
    var e = FakeEngine(False)
    var a = _toks(64, 960)
    _ = _turn(s, e, a, 0, False)                                       # saved: 896 bytes is not more than the budget
    assert_equal(s.ram.used, 896)
    s.close()
    var s2 = _small_chunk_store(dir, False, 896)
    var e2 = FakeEngine(False)
    assert_equal(_prep(s2, e2, _cat(a, _toks(5, 961)), False), 64)     # and restored: it is not "bigger than the budget"
    s2.close()


def test_a_request_bigger_than_the_whole_budget_evicts_nothing() raises:
    var s = _small_chunk_store(_dir(), False, 1 << 30)
    var e = FakeEngine(False)
    _ = _turn(s, e, _toks(100, 970), 0, False)                         # P: 1400 bytes, resident and on disk
    s.cfg.ram_bytes = 500                                              # now smaller than one 896-byte chunk
    var ids = _toks(64, 971)
    assert_equal(s.prepare(e, ids, False), 0)
    e.prefill(ids)
    assert_equal(s.commit(e), -3)                                      # S's chunk can never fit...
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 0)            # ...and P's RAM was not flushed to find that out
    assert_equal(s.ram.used, 100 * 14)
    s.close()


# ── Fix round 2 (re-review, new breakage 3): the restoring session is strictly newer than the one the park-commit stamped ──


def test_a_restoring_session_outlives_the_session_the_park_commit_just_stamped() raises:
    var budget = 3 * 896 + 100                                         # three 64-token chunks
    var s = _small_chunk_store(_dir(), False, budget)
    var e = FakeEngine(False)
    var r = _toks(192, 980)
    _ = _turn(s, e, r, 0, False)                                       # R: three chunks, all resident (list position 0)
    var id_r = s.live_id
    _ = _turn(s, e, _toks(64, 981), 0, False)                          # P: one chunk; it pushes R's chunk 0 out
    var id_p = s.live_id
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 1)
    e.extend(_toks(5, 982), False)                                     # P grows 5 tokens that no commit has seen
    # Switching to R parks P first: that commit saves the tail and stamps P with this prepare's clock. R's restore then
    # needs room for chunk 0. R must be the newer of the two, or a tie in last_used evicts R's own chunks (lower position).
    assert_equal(_prep(s, e, _cat(r, _toks(5, 983)), False), 192)
    var pr = s.index.find(id_r)
    for ci in range(3):
        assert_true(s.index.sessions[pr].chunks[ci].ram_addr != 0)     # all of R is resident: its chunks went last
    var pp = s.index.find(id_p)
    assert_equal(s.index.sessions[pp].chunks[0].ram_addr, UInt64(0))   # P's first chunk paid for R's chunk 0
    assert_equal(s.stats_snapshot()[STAT_RAM_EVICTIONS], 2)
    assert_equal(s.ram.used, _resident_bytes(s))
    assert_true(s.ram.peak <= budget)
    s.close()


# ── Fix round 2 (re-review, item 7): the unreadable-chunk branch of _restore, without chmod ───────────────────────────


def test_an_unreadable_chunk_resets_the_engine_even_when_the_drop_raises() raises:
    var dir = _dir()
    var s1 = _small_chunk_store(dir, False, 1 << 30)
    var e1 = FakeEngine(False)
    var a = _toks(150, 990)                                            # chunks of 64, 64 and 22 tokens
    _ = _turn(s1, e1, a, 0, False)
    var id_a = s1.live_id
    s1.close()
    var s = _small_chunk_store(dir, False, 1 << 30)                    # a restart: nothing is resident
    var e = FakeEngine(False)
    assert_equal(_prep(s, e, _cat(sub(a, 0, 40), _toks(20, 991)), False), 40)    # a short restore reads only chunk 0
    var pa = s.index.find(id_a)
    assert_true(s.index.sessions[pa].chunks[0].ram_addr != 0)
    assert_equal(s.index.sessions[pa].chunks[1].ram_addr, UInt64(0))
    var path = s.disk.path(id_a)
    remove(path)
    makedirs(path)                    # a directory where the file was: reading chunk 1 fails, and so does removing it
    try:
        var raised = False
        try:
            _ = s.prepare(e, _cat(a, _toks(5, 992)), False)            # chunk 0 imports, chunk 1 cannot be read
        except:
            raised = True
        assert_true(raised)                                            # the drop of the unreadable session raised...
        assert_equal(e.kv_len(), 0)                                    # ...but the engine had been reset before it
        assert_true(s.index.find(id_a) >= 0)                           # the session is still there: its drop did not happen
        assert_equal(s.ram.used, _resident_bytes(s))                   # and the read buffer was given back
    finally:
        rmdir(path)
    # With the file simply gone, the same request drops the unreadable session cleanly and starts afresh.
    assert_equal(s.prepare(e, _cat(a, _toks(5, 993)), False), 0)
    assert_equal(s.index.find(id_a), -1)                               # the session is gone
    assert_equal(e.kv_len(), 0)
    assert_equal(s.ram.used, _resident_bytes(s))
    s.close()


def main() raises:
    _clear_root()
    test_next_turn_reuses_without_copies()
    test_full_prompt_returns_n_minus_1_without_fork()
    test_switch_restores_parked_session()
    test_divergence_forks()
    test_short_overlap_starts_fresh()
    test_spec_caps_at_drafter_len_and_forks()
    test_ram_eviction_restores_from_disk()
    test_restart_restores_from_disk()
    test_desynced_engine_is_not_trusted()
    test_disk_failure_keeps_session_in_ram()
    test_disk_failure_catches_up_when_the_disk_recovers()
    test_commit_after_engine_reset_saves_a_new_session()
    test_abandoned_turn_leaves_no_record()
    test_long_commit_is_split_into_bounded_chunks()
    test_ram_tier_counts_the_bytes_it_holds()
    test_close_frees_every_ram_chunk()
    test_stats_count_a_scripted_run()
    test_desynced_engine_is_restored_even_when_its_kv_is_wrong()
    test_reuse_threshold_and_degenerate_prompts()
    test_switching_needs_more_than_min_switch_tokens_of_gain()
    test_leaving_a_session_persists_its_uncommitted_tail()
    test_full_prompt_of_a_parked_session_returns_n_minus_1_without_fork()
    test_partial_restore_cuts_inside_a_chunk_and_forks()
    test_restart_restore_reads_only_the_chunks_it_needs()
    test_restore_that_extends_the_session_does_not_fork()
    test_chunks_without_drafter_rows_survive_a_restart()
    test_caught_up_chunks_are_not_written_twice()
    test_ram_pressure_keeps_the_ram_only_tail_of_a_partly_written_session()
    test_ram_eviction_takes_the_least_recently_used_session()
    test_restore_keeps_the_restored_session_resident_under_a_tight_budget()
    test_unreadable_session_is_dropped_and_the_engine_reset()
    test_session_dropped_to_make_room_is_not_restored()
    test_disk_budget_drops_whole_least_recently_used_sessions()
    test_disk_budget_never_drops_the_live_or_an_unwritten_session()
    test_failed_export_leaks_no_ram()
    test_failed_import_leaves_a_store_that_can_retry()
    test_ram_budget_smaller_than_one_chunk_is_refused()
    test_a_long_live_session_stays_inside_a_two_chunk_ram_budget()
    test_a_session_larger_than_the_ram_budget_restores_exactly_from_disk()
    test_a_failing_disk_cannot_grow_pinned_memory_past_the_budget()
    test_a_parked_sessions_unsaved_chunks_cannot_starve_the_live_session()
    test_close_writes_out_unsaved_chunks_if_the_disk_has_recovered()
    test_ram_evictions_count_chunks_lowest_first_and_not_closes()
    test_one_big_chunk_can_evict_several_small_ones()
    test_a_commit_that_runs_out_of_ram_reports_what_it_saved()
    test_closing_twice_is_harmless()
    test_restore_degrades_to_a_fresh_start_when_ram_is_full_of_unsaved_chunks()
    test_a_session_file_deleted_by_hand_does_not_wedge_the_disk_budget()
    test_a_failed_disk_drop_leaves_the_session_and_its_unsaved_chunks_alone()
    test_a_scan_that_raises_releases_the_directory_lock()
    test_chunk_tokens_must_be_positive()
    test_min_reuse_must_be_at_least_one()
    test_min_switch_must_not_be_negative()
    test_the_disk_budget_must_not_be_negative()
    test_an_engine_reset_behind_the_stores_back_costs_no_reuse()
    test_an_engine_refilled_behind_the_stores_back_saves_the_new_tokens_and_restores_the_session()
    test_after_a_failed_append_the_rest_of_the_commit_stays_off_the_disk()
    test_a_catch_up_that_fails_stops_there()
    test_commit_with_nothing_new_still_writes_out_unsaved_chunks()
    test_a_partial_commit_still_refreshes_lru_and_enforces_the_disk_budget()
    test_a_failed_export_after_a_partial_save_still_does_the_bookkeeping()
    test_a_failed_export_still_reports_the_export_error_when_the_bookkeeping_fails()
    test_a_failed_export_with_nothing_saved_does_no_bookkeeping()
    test_a_chunk_bigger_than_the_ram_budget_drops_its_session_and_flushes_nothing()
    test_a_chunk_exactly_as_big_as_the_budget_fits()
    test_a_request_bigger_than_the_whole_budget_evicts_nothing()
    test_a_restoring_session_outlives_the_session_the_park_commit_just_stamped()
    test_an_unreadable_chunk_resets_the_engine_even_when_the_drop_raises()
    _clear_root()
    print("kvstore_store_test: all passed")
