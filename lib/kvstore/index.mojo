"""Session index of the KV session store (docs/KV_SESSION_STORE_DESIGN.md §3.1): sessions, their chunks, and a map
from chained 256-token block hashes to the session holding that prefix."""
from lib.kvstore.hashing import fnv1a64_tokens

comptime BLOCK = 256


def block_hashes(seed: UInt64, tokens: List[Int], n: Int) -> List[UInt64]:
    """Chained hashes of the full blocks of tokens[0:n]: h[i] = H(h[i-1], block i), h[-1] = seed. Equal hashes
    mean equal prefixes up to collisions, which callers rule out by comparing tokens."""
    var out = List[UInt64]()
    var h = seed
    for i in range(n // BLOCK):
        h = fnv1a64_tokens(h, tokens, i * BLOCK, (i + 1) * BLOCK)
        out.append(h)
    return out^


def lcp(a: List[Int], b: List[Int], limit: Int) -> Int:
    var n = min(limit, min(len(a), len(b)))
    var i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def sub(tokens: List[Int], a: Int, b: Int) -> List[Int]:
    var out = List[Int](capacity=max(b - a, 0))
    for i in range(a, b):
        out.append(tokens[i])
    return out^


@fieldwise_init
struct ChunkInfo(Copyable, Movable, ImplicitlyCopyable):
    var start: Int
    var end: Int
    var drafter_end: Int     # drafter context rows are stored for [start, drafter_end)
    var file_offset: Int     # payload offset in the session file; -1 when not on disk
    var ram_addr: UInt64     # host copy of the payload; 0 when not in RAM


@fieldwise_init
struct SessionRecord(Copyable, Movable):
    var id: UInt64
    var tokens: List[Int]          # the tokens with stored KV
    var chunks: List[ChunkInfo]
    var drafter_len: Int           # contiguous prefix with stored drafter context
    var last_used: Int
    var on_disk: Bool              # its file has a header

    def kv_len(self) -> Int:
        return len(self.tokens)


struct SessionIndex(Movable):
    var seed: UInt64
    var sessions: List[SessionRecord]
    var owner: Dict[UInt64, UInt64]   # block-chain hash -> id of a session holding that prefix

    def __init__(out self, seed: UInt64):
        self.seed = seed
        self.sessions = List[SessionRecord]()
        self.owner = Dict[UInt64, UInt64]()

    def find(self, id: UInt64) -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].id == id:
                return i
        return -1

    def _index_blocks(mut self, pos: Int, from_token: Int):
        var hs = block_hashes(self.seed, self.sessions[pos].tokens, self.sessions[pos].kv_len())
        for i in range(from_token // BLOCK, len(hs)):
            self.owner[hs[i]] = self.sessions[pos].id

    def add(mut self, var rec: SessionRecord):
        self.sessions.append(rec^)
        self._index_blocks(len(self.sessions) - 1, 0)

    def drop(mut self, id: UInt64):
        var p = self.find(id)
        if p < 0:
            return
        _ = self.sessions.pop(p)
        self.owner = Dict[UInt64, UInt64]()
        for i in range(len(self.sessions)):
            self._index_blocks(i, 0)

    def append_chunk(mut self, pos: Int, chunk: ChunkInfo, new_tokens: List[Int]):
        var old_len = self.sessions[pos].kv_len()
        for i in range(len(new_tokens)):
            self.sessions[pos].tokens.append(new_tokens[i])
        if self.sessions[pos].drafter_len == chunk.start:
            self.sessions[pos].drafter_len = chunk.drafter_end
        self.sessions[pos].chunks.append(chunk)
        self._index_blocks(pos, old_len)

    def longest_match(self, ids: List[Int], cap: Int, exclude: UInt64, spec: Bool) -> Tuple[Int, Int]:
        """(position, length) of the session other than `exclude` that shares the longest prefix with ids: at most
        cap tokens and, for spec, at most that session's drafter_len; (-1, 0) when none shares a token. The length is
        always confirmed by comparing tokens. The deepest block hash names one holder of the shared blocks, which is
        enough for an ordinary request: another holder could match further only inside the block that follows the
        shared ones. Every session is compared instead when no hash leads to a holder, when the excluded session owns
        a deeper block (it may be hiding the holder that matches furthest), and for spec (the named holder may have
        less drafter context than another holder of the same blocks)."""
        var cands = List[Int]()
        var shadowed = False   # the excluded session owns a block deeper than the holder found
        var hs = block_hashes(self.seed, ids, min(cap, len(ids)))
        for i in range(len(hs) - 1, -1, -1):
            var holder = self.owner.get(hs[i])   # Dict's subscript raises; get() does not
            if holder:
                var p = self.find(holder.value())
                if p >= 0 and self.sessions[p].id == exclude:
                    shadowed = True
                elif p >= 0:
                    cands.append(p)
                    break
        if len(cands) == 0 or shadowed or spec:
            cands = List[Int]()
            for p in range(len(self.sessions)):
                if self.sessions[p].id != exclude:
                    cands.append(p)
        var best_pos = -1
        var best = 0
        for j in range(len(cands)):
            var p = cands[j]
            var limit = min(cap, self.sessions[p].kv_len())
            if spec:
                limit = min(limit, self.sessions[p].drafter_len)
            var m = lcp(ids, self.sessions[p].tokens, limit)
            if m > best:
                best = m
                best_pos = p
        return (best_pos, best)
