"""The KV session store (docs/KV_SESSION_STORE_DESIGN.md §3.2-3.3): decide how much of a request the engine can reuse,
park and restore sessions through a RAM tier and an append-only NVMe tier, and persist every turn. Model-agnostic:
it sees the engine only through KvSessionEngine and the chunk layout."""
from lib.kvstore.disk_tier import DiskTier
from lib.kvstore.engine_trait import KvSessionEngine
from lib.kvstore.fileio import random_u64
from lib.kvstore.hashing import hex16
from lib.kvstore.index import SessionIndex, SessionRecord, ChunkInfo, lcp, sub
from lib.kvstore.key import CacheKey
from lib.kvstore.layout import KvLayout
from lib.kvstore.ram_tier import RamTier

comptime STAT_PREPARES = 0
comptime STAT_PROMPT_TOKENS = 1
comptime STAT_REUSED_TOKENS = 2
comptime STAT_RESTORES_RAM = 3
comptime STAT_RESTORES_NVME = 4
comptime STAT_FORKS = 5
comptime STAT_RAM_BYTES = 6
comptime STAT_NVME_BYTES = 7
comptime STAT_RAM_EVICTIONS = 8
comptime STAT_NVME_EVICTIONS = 9
comptime N_STATS = 10


@fieldwise_init
struct StoreConfig(Copyable, Movable):
    var dir: String
    var ram_bytes: Int
    var disk_bytes: Int
    var min_reuse: Int
    var min_switch: Int
    var pinned: Bool
    var chunk_tokens: Int   # a commit is saved as chunks of at most this many tokens (32K-128K contexts)


struct SessionStore(Movable):
    var key: CacheKey
    var layout: KvLayout
    var cfg: StoreConfig
    var index: SessionIndex
    var disk: DiskTier
    var ram: RamTier
    var live_id: UInt64     # 0 = no live session
    var clock: Int
    var stats: List[Int]

    def __init__(out self, key: CacheKey, layout: KvLayout, cfg: StoreConfig) raises:
        # A bad config is refused before anything touches the disk. Each message names the variable to fix.
        if cfg.chunk_tokens < 1:            # commit would never advance
            raise Error("kv store: NOMOS_KVSTORE_CHUNK_TOKENS must be at least 1 (got " + String(cfg.chunk_tokens) + ")")
        if cfg.min_reuse < 1:               # a zero-length match would truncate the live session for nothing
            raise Error("kv store: NOMOS_KVSTORE_MIN_REUSE must be at least 1 (got " + String(cfg.min_reuse) + ")")
        if cfg.min_switch < 0:
            raise Error("kv store: NOMOS_KVSTORE_MIN_SWITCH must be at least 0 (got " + String(cfg.min_switch) + ")")
        if cfg.disk_bytes < 0:
            raise Error("kv store: NOMOS_KVSTORE_DISK_GB must be at least 0 (got " + String(cfg.disk_bytes) + " bytes)")
        var chunk_ram = layout.chunk_bytes(cfg.chunk_tokens)
        if cfg.ram_bytes < chunk_ram:       # the budget is hard (ruling R27): it must be able to hold the chunk being saved
            raise Error("kv store: NOMOS_KVSTORE_RAM_GB (" + String(cfg.ram_bytes) + " bytes) cannot hold one chunk of "
                        + String(cfg.chunk_tokens) + " tokens (" + String(chunk_ram) + " bytes)")
        self.key = key
        self.layout = layout.copy()
        self.cfg = cfg.copy()
        self.index = SessionIndex(key.seed())
        self.disk = DiskTier(cfg.dir, key, layout)
        self.ram = RamTier(cfg.pinned)
        self.live_id = UInt64(0)
        self.clock = 0
        self.stats = List[Int](length=N_STATS, fill=0)
        try:
            var found = self.disk.scan()
            for i in range(len(found)):
                self.index.add(found[i].copy())
        except e:
            self.disk.close()   # DiskTier has no destructor: without this the directory stays locked until the process exits
            raise e

    def close(mut self):
        """Best effort: a chunk that exists only in RAM is lost with the store unless the disk has recovered, so try to
        write those out first (a failure is logged). Then free every RAM chunk and release the directory lock."""
        for i in range(len(self.index.sessions)):
            if self._has_ram_only(i):
                _ = self._catch_up_disk(i)
        for i in range(len(self.index.sessions)):
            self._free_session_ram(i)
        self.disk.close()

    def stats_snapshot(mut self) -> List[Int]:
        var s = self.stats.copy()
        s[STAT_RAM_BYTES] = self.ram.used
        s[STAT_NVME_BYTES] = self.disk.total_bytes()
        return s^

    # ── prepare ────────────────────────────────────────────────────────────────────────────
    def prepare[E: KvSessionEngine](mut self, mut eng: E, ids: List[Int], spec: Bool) raises -> Int:
        """Leave the engine holding KV for exactly ids[0:c] and return c, 0 <= c <= len(ids) - 1."""
        var n = len(ids)
        self.clock += 1
        self.stats[STAT_PREPARES] += 1
        self.stats[STAT_PROMPT_TOKENS] += n
        var live_toks = eng.live_tokens_copy()
        var kv = eng.kv_len()
        var live_pos = self.index.find(self.live_id) if self.live_id != 0 else -1
        var live_ok = live_pos >= 0 and len(live_toks) == kv
        # The live record stays out of the parked search only while the engine holds all of its committed tokens. An
        # engine reset or refilled behind the store's back no longer does; then the record competes like any parked
        # session (restoring it still needs p_park > p_live + min_switch).
        var live_holds = False
        if live_ok:
            var rec_len = self.index.sessions[live_pos].kv_len()
            live_holds = kv >= rec_len and lcp(live_toks, self.index.sessions[live_pos].tokens, rec_len) == rec_len
        var p_live = 0
        if live_ok and n > 1:
            p_live = lcp(ids, live_toks, min(n - 1, kv))
            if spec:
                p_live = min(p_live, eng.drafter_len())
        var park_id = UInt64(0)
        var p_park = 0
        if n > 1:
            # An untrustworthy live engine still leaves its committed history restorable (Review Focus 2).
            var best = self.index.longest_match(ids, n - 1, self.live_id if live_holds else UInt64(0), spec)
            if best[0] >= 0:
                park_id = self.index.sessions[best[0]].id
                p_park = best[1]
        if park_id != 0 and p_park > p_live + self.cfg.min_switch:
            if live_ok:
                _ = self.commit(eng)   # persist the live session's uncommitted tail before parking it
            if self._restore(eng, park_id, ids, p_park, spec):
                self.stats[STAT_REUSED_TOKENS] += p_park
                return p_park
            self._start_new(eng)
            return 0
        if live_ok and p_live >= self.cfg.min_reuse:
            if p_live < kv:
                eng.truncate_to(p_live)
                if not self._continues(self.index.find(self.live_id), ids, spec):
                    self._start_fork()
            var pos = self.index.find(self.live_id)
            self.index.sessions[pos].last_used = self.clock
            self.stats[STAT_REUSED_TOKENS] += p_live
            return p_live
        if live_ok:
            _ = self.commit(eng)
        self._start_new(eng)
        return 0

    def _continues(self, pos: Int, ids: List[Int], spec: Bool) -> Bool:
        """The request extends this session's whole committed history (so no fork is needed)."""
        var a = self.index.sessions[pos].kv_len()
        if len(ids) < a or lcp(ids, self.index.sessions[pos].tokens, a) != a:
            return False
        return not spec or self.index.sessions[pos].drafter_len >= a

    def _set_live(mut self, id: UInt64) raises:
        """Make id the live session, dropping the previous live record if it never stored anything (a turn that
        was prepared but never committed), so abandoned records do not pile up in the index."""
        if self.live_id != 0 and self.live_id != id:
            var old = self.index.find(self.live_id)
            if old >= 0 and self.index.sessions[old].kv_len() == 0:
                self._drop(self.live_id)
        self.live_id = id

    def _new_live_record(mut self) raises:
        var new_id = random_u64()
        while new_id == 0:          # 0 means "no session" (live_id, longest_match's exclude)
            new_id = random_u64()
        var rec = SessionRecord(new_id, List[Int](), List[ChunkInfo](), 0, self.clock, False)
        var id = rec.id
        self.index.add(rec^)
        self._set_live(id)

    def _start_new[E: KvSessionEngine](mut self, mut eng: E) raises:
        eng.reset_session()
        self._new_live_record()

    def _start_fork(mut self) raises:
        self.stats[STAT_FORKS] += 1
        self._new_live_record()

    def _restore[E: KvSessionEngine](mut self, mut eng: E, id: UInt64, ids: List[Int], p: Int, spec: Bool) raises -> Bool:
        """Import the session's KV for [0, p). False (engine reset) if it could not be read, if RAM is full of chunks that
        cannot be evicted, or if one of its chunks is bigger than the whole RAM budget (the session is dropped then)."""
        var pos = self.index.find(id)
        if pos < 0:
            return False
        eng.reset_session()
        var tokens = self.index.sessions[pos].tokens.copy()
        # Its own chunks are evicted last when RAM runs short (R27). prepare's park-commit may just have stamped the old
        # live session with this prepare's clock, so tick first: on a tie the lower list position would be evicted first.
        self.clock += 1
        self.index.sessions[pos].last_used = self.clock
        var from_disk = False
        for ci in range(len(self.index.sessions[pos].chunks)):
            # INVARIANT (see _ram_alloc): from this lookup to this chunk's import below, nothing may allocate RAM except
            # for this very chunk. An allocation for any other chunk could evict this chunk's buffer before it is imported.
            var ch = self.index.sessions[pos].chunks[ci]
            if ch.start >= p:
                break
            var nbytes = self.layout.chunk_bytes(ch.end - ch.start)
            if ch.ram_addr == 0:
                if nbytes > self.cfg.ram_bytes:
                    # The cache key does not include chunk_tokens, so a chunk written under a larger
                    # NOMOS_KVSTORE_CHUNK_TOKENS can exceed this run's RAM budget. It can never be loaded, and trying
                    # would only flush RAM each time: the session cannot be restored under this config.
                    print("[kvstore] session", hex16(id), "has a chunk of", nbytes, "bytes, more than NOMOS_KVSTORE_RAM_GB allows (",
                          self.cfg.ram_bytes, "bytes): it cannot be restored under this config and is dropped")
                    eng.reset_session()
                    self._drop(id)
                    return False
                var addr = self._ram_alloc(nbytes)
                if addr == 0:
                    eng.reset_session()
                    return False
                try:
                    self.disk.read_payload(id, ch, sub(tokens, ch.start, ch.end), addr)
                except e:
                    print("[kvstore] dropping unreadable session", hex16(id), ":", e)
                    self.ram.free(addr, nbytes)
                    eng.reset_session()     # the prefix imported so far is not a session; clean even if the drop raises
                    self._drop(id)
                    return False
                pos = self.index.find(id)
                self.index.sessions[pos].chunks[ci].ram_addr = addr
                ch.ram_addr = addr
                from_disk = True
            var end = min(ch.end, p)
            try:
                eng.import_chunk(self.layout, ch.start, end, min(ch.drafter_end, end), ch.ram_addr, ch.end - ch.start,
                                 sub(tokens, ch.start, end))
            except ex:
                # A half-restored engine is not a session. Leave it clean, so the next request starts from a known state
                # instead of mistaking the partial prefix for the live session (it would be saved as a junk fork).
                try:
                    eng.reset_session()
                except rex:
                    print("[kvstore] could not reset the engine after a failed import:", rex)
                raise ex
        pos = self.index.find(id)
        if self._continues(pos, ids, spec):
            self._set_live(id)
        else:
            self._start_fork()
        self.stats[STAT_RESTORES_NVME if from_disk else STAT_RESTORES_RAM] += 1
        return True

    # ── commit ─────────────────────────────────────────────────────────────────────────────
    def commit[E: KvSessionEngine](mut self, mut eng: E) raises -> Int:
        """Persist the live session's tokens since its last commit. Returns how many (>= 0), -1 when the
        engine's KV no longer matches its tokens, or -3 when RAM was full and nothing could be saved (the budget is
        hard); if it fills midway, the number saved so far. If the engine no longer extends the session's history, what
        it holds is saved as a new session (plan note 7)."""
        if self.live_id == 0:
            return 0
        var pos = self.index.find(self.live_id)
        if pos < 0:
            return 0
        var toks = eng.live_tokens_copy()
        var kv = eng.kv_len()
        if len(toks) != kv:
            return -1
        var a = self.index.sessions[pos].kv_len()
        var shared = min(kv, a)
        if lcp(toks, self.index.sessions[pos].tokens, shared) != shared:
            # The engine no longer holds this session's history (an error path reset it, or its KV was replaced
            # with tokens kept in step): save what it holds as a new session instead of failing every later commit.
            self._start_fork()
            pos = self.index.find(self.live_id)
            a = 0
        # The session file must stay a gap-free prefix of the session (a chunk after a gap cannot be chained at
        # restart): first persist any chunks an earlier disk failure left in RAM only, and write nothing new to disk
        # unless that succeeded (ledger R25). This also runs when there is nothing new to save: the disk may be back.
        var disk_ok = self._catch_up_disk(pos)
        if kv <= a:
            return 0   # the engine holds a prefix of the session, e.g. after prepare cut it to n - 1
        var dend = min(max(eng.drafter_len(), a), kv)
        # Saved as chunks of at most cfg.chunk_tokens, so a long turn (a 30K-token paste) never needs one
        # multi-GB pinned buffer, and RAM eviction works at a useful granularity.
        var s = a
        while s < kv:
            var e = min(s + self.cfg.chunk_tokens, kv)
            var cd = min(max(dend, s), e)          # this chunk's drafter rows are [s, cd)
            var nbytes = self.layout.chunk_bytes(e - s)
            var addr = self._ram_alloc(nbytes)
            if addr == 0:
                if s > a:                           # it saved something: the same bookkeeping as a commit that finished
                    self._finish_commit(pos)
                return -3 if s == a else s - a      # what was saved stays saved; the next commit resumes at s
            try:
                eng.export_chunk(self.layout, s, e, cd, addr)
            except ex:
                self.ram.free(addr, nbytes)   # not indexed yet: nothing else would ever free it (or its byte count)
                if s > a:
                    # Earlier chunks of this commit are saved: do the bookkeeping a finished commit does. Best effort, so
                    # that the error the caller sees stays the export's.
                    try:
                        self._finish_commit(pos)
                    except fex:
                        print("[kvstore] could not finish the bookkeeping of a partly saved commit:", fex)
                raise ex
            var new_toks = sub(toks, s, e)
            var off = -1
            if disk_ok:
                try:
                    if not self.index.sessions[pos].on_disk:
                        self.disk.create(self.live_id)
                        self.index.sessions[pos].on_disk = True
                    off = self.disk.append(self.live_id, s, e, cd, new_toks, addr, nbytes)
                except ex:
                    disk_ok = False
                    print("[kvstore] ERROR: session", hex16(self.live_id), "could not be written to disk; it stays in RAM:", ex)
            self.index.append_chunk(pos, ChunkInfo(s, e, cd, off, addr), new_toks)
            s = e
        self._finish_commit(pos)
        return kv - a

    def _finish_commit(mut self, pos: Int) raises:
        """After a commit that saved at least one chunk: refresh the session's LRU stamp and bring the disk back inside
        its budget (the live session is never the one dropped)."""
        self.index.sessions[pos].last_used = self.clock
        self._enforce_disk_budget()

    # ── tiers ───────────────────────────────────────────────────────────────────────────────
    def _catch_up_disk(mut self, pos: Int) -> Bool:
        """Append, in order, the chunks of this session that exist only in RAM (an earlier disk failure). Those
        always form a suffix of the chunk list, because a failure stops all later disk writes for the commit.
        False if the disk still refuses: then nothing new may go to disk either. DiskTier.append cuts a failed
        record back, so the file stays a gap-free prefix of the session (ledger R25)."""
        for ci in range(len(self.index.sessions[pos].chunks)):
            var ch = self.index.sessions[pos].chunks[ci]
            if ch.file_offset >= 0:
                continue
            try:
                if not self.index.sessions[pos].on_disk:
                    self.disk.create(self.index.sessions[pos].id)
                    self.index.sessions[pos].on_disk = True
                var toks = sub(self.index.sessions[pos].tokens, ch.start, ch.end)
                var off = self.disk.append(self.index.sessions[pos].id, ch.start, ch.end, ch.drafter_end, toks,
                                           ch.ram_addr, self.layout.chunk_bytes(ch.end - ch.start))
                self.index.sessions[pos].chunks[ci].file_offset = off
            except ex:
                print("[kvstore] session", hex16(self.index.sessions[pos].id), "still cannot be written to disk:", ex)
                return False
        return True

    def _free_session_ram(mut self, pos: Int):
        """Free every RAM chunk of a session that is being dropped, or of every session when the store closes. Unlike an
        eviction this also frees chunks with no disk copy, because the session itself is going away."""
        for ci in range(len(self.index.sessions[pos].chunks)):
            var ch = self.index.sessions[pos].chunks[ci]
            if ch.ram_addr != 0:
                self.ram.free(ch.ram_addr, self.layout.chunk_bytes(ch.end - ch.start))
                self.index.sessions[pos].chunks[ci].ram_addr = UInt64(0)

    def _evictable_chunk(self, pos: Int) -> Int:
        """This session's lowest chunk that is in RAM and also on disk, or -1. A chunk with no disk copy is never
        evicted (Review Focus 4)."""
        for ci in range(len(self.index.sessions[pos].chunks)):
            var ch = self.index.sessions[pos].chunks[ci]
            if ch.ram_addr != 0 and ch.file_offset >= 0:
                return ci
        return -1

    def _has_ram_only(self, pos: Int) -> Bool:
        """The session holds a chunk that exists only in RAM (a disk write failed and was not yet caught up)."""
        for ci in range(len(self.index.sessions[pos].chunks)):
            var ch = self.index.sessions[pos].chunks[ci]
            if ch.ram_addr != 0 and ch.file_offset < 0:
                return True
        return False

    def _evict_chunk(mut self, pos: Int, ci: Int):
        """Give back one chunk's RAM copy; its disk copy stays."""
        var ch = self.index.sessions[pos].chunks[ci]
        self.ram.free(ch.ram_addr, self.layout.chunk_bytes(ch.end - ch.start))
        self.index.sessions[pos].chunks[ci].ram_addr = UInt64(0)
        self.stats[STAT_RAM_EVICTIONS] += 1

    def _ram_alloc(mut self, nbytes: Int) -> UInt64:
        """A chunk buffer that never takes RAM past cfg.ram_bytes (ruling R27), or 0.

        Room is made one chunk at a time. The least recently used session that holds a chunk with a disk copy gives up
        its lowest such chunk; any session may, the live one too. A chunk being exported is not in the index yet, so it
        cannot be picked. A chunk being imported IS in the index, and on disk, so it could be: what keeps its buffer safe
        is that nothing allocates between the chunk's lookup in _restore and its import. Keep it so; reading the next
        chunk before the current one is imported (a prefetch) would let this loop free the buffer about to be imported.
        When nothing is evictable the disk may have recovered since the write that failed,
        so every session holding unsaved chunks is caught up, once per call; that makes those chunks evictable. If that
        frees nothing the answer is 0: commit then stops saving (-3 or a partial count) and restore falls back to a full
        prefill, rather than pinned memory growing without limit. A request bigger than the whole budget is refused at
        once, before anything is evicted: no amount of eviction could make it fit."""
        if nbytes > self.cfg.ram_bytes:
            return UInt64(0)
        var caught_up = False
        while self.ram.used + nbytes > self.cfg.ram_bytes:
            var victim = -1
            var victim_chunk = -1
            for i in range(len(self.index.sessions)):
                var ci = self._evictable_chunk(i)
                if ci < 0:
                    continue
                if victim < 0 or self.index.sessions[i].last_used < self.index.sessions[victim].last_used:
                    victim = i
                    victim_chunk = ci
            if victim >= 0:
                self._evict_chunk(victim, victim_chunk)
                continue
            if caught_up:
                return UInt64(0)
            caught_up = True
            for i in range(len(self.index.sessions)):
                if self._has_ram_only(i):
                    _ = self._catch_up_disk(i)
        return self.ram.alloc(nbytes)

    def _drop(mut self, id: UInt64) raises:
        var pos = self.index.find(id)
        if pos < 0:
            return
        if self.index.sessions[pos].on_disk:
            self.disk.drop(id)          # first: if it raises, the record and its unsaved chunks are left as they were
        self._free_session_ram(pos)
        self.index.drop(id)
        if self.live_id == id:
            self.live_id = UInt64(0)

    def _enforce_disk_budget(mut self) raises:
        while self.disk.total_bytes() > self.cfg.disk_bytes:
            var victim = -1
            for i in range(len(self.index.sessions)):
                if self.index.sessions[i].id == self.live_id or not self.index.sessions[i].on_disk:
                    continue
                if victim < 0 or self.index.sessions[i].last_used < self.index.sessions[victim].last_used:
                    victim = i
            if victim < 0:
                break
            self._drop(self.index.sessions[victim].id)
            self.stats[STAT_NVME_EVICTIONS] += 1
