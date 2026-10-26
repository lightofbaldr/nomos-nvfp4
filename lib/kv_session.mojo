"""Engine-side primitives for the KV session store (docs/KV_SESSION_STORE_DESIGN.md §3.1): describe the live KV as
byte components and copy token ranges of it between the GPU and pinned host memory. With a DFlash drafter loaded the
components also cover the drafter's context rows (§3.7). Every copy runs on the engine's own copy stream after a
device-wide sync, so a copy can never race a kernel that is still writing the KV. Also the engine's end of the store
itself: its configuration from the environment, whether it may run, and creating, re-keying and releasing the
SessionStore the engine points to (GemmaEngine.kvstore_ptr)."""
from std.memory import Pointer, UnsafePointer
from std.memory.alloc import unsafe_alloc
from std.os import getenv, listdir
from std.os.path import getsize, isdir
from lib.gemma4_engine import GemmaEngine, _env_float
from lib.build_stamp import BUILD_ID
from lib.model_config import MODEL_ID
from lib.cuda import cuda_sync, cuda_stream_create, cuda_event_create, cuda_event_record, cuda_event_sync, cuda_memcpy_2d_async_on, cuda_memcpy_async_on
from lib.dflash_drafter import DFlashDrafter, DFLASH_LAYERS, DFLASH_KV_DIM
from lib.kvstore.support import KV_MODE_UNSUPPORTED, KV_MODE_INT8, KV_MODE_INT4_B1, KV_MODE_INT4_B32, store_supported
from lib.kvstore.layout import (
    KvLayout, ComponentDesc, target_layout, KIND_K_DATA, KIND_V_DATA, KIND_K_SCALE, KIND_V_SCALE, KIND_K_TAIL,
    KIND_DRAFT_K, KIND_DRAFT_V,
)
from lib.kvstore.hashing import fnv1a64_u64, hex16, parse_hex16
from lib.kvstore.key import CacheKey, weights_id_of
from lib.kvstore.store import SessionStore, StoreConfig

comptime _H2D = 1
comptime _D2H = 2


def kv_mode_of(eng: GemmaEngine) -> Int:
    if not eng.kv_quant:
        return KV_MODE_UNSUPPORTED
    if not eng.kv_int4:
        return KV_MODE_INT8
    return KV_MODE_INT4_B32 if eng.kv_int4_block32 else KV_MODE_INT4_B1


def _drafter(eng: GemmaEngine) -> Pointer[DFlashDrafter, MutUntrackedOrigin]:
    """The loaded DFlash drafter. Only valid while eng.dflash_ptr != 0."""
    return Pointer[DFlashDrafter, MutUntrackedOrigin](unsafe_from_address=Int(eng.dflash_ptr))


def build_layout(eng: GemmaEngine) -> KvLayout:
    """The live KV as byte components: the target's, then, with a DFlash drafter loaded, K and V of each of its layers."""
    var lay = target_layout(eng.layer_nkv, eng.layer_hd, kv_mode_of(eng))
    if eng.dflash_ptr != 0:
        for dl in range(DFLASH_LAYERS):
            lay.comps.append(ComponentDesc(KIND_DRAFT_K, dl, 1, DFLASH_KV_DIM * 4))
            lay.comps.append(ComponentDesc(KIND_DRAFT_V, dl, 1, DFLASH_KV_DIM * 4))
    return lay^


def kv_is_linear_quantized(eng: GemmaEngine) -> Bool:
    """True for the only KV the chunk layout describes: quantized (INT8 or INT4) in a linear cache. An unquantized
    engine has no d_k_cache_i8 / d_k_scales to address. Under NOMOS_KV_SWA=1 the sliding layers are rings (slot is
    pos % cache_cap), so a copy addressed by position hits the wrong rows and can run past the allocation."""
    return kv_mode_of(eng) != KV_MODE_UNSUPPORTED and not eng.kv_swa


def _require_linear_quantized(eng: GemmaEngine, what: String) raises:
    if not kv_is_linear_quantized(eng):
        raise Error(what + ": the KV session store copies a linear quantized KV, so it needs NOMOS_KV_QUANT=1 and NOMOS_KV_SWA=0")


def _target_base(eng: GemmaEngine, c: ComponentDesc) -> UInt64:
    """Device address of head 0, position 0 of target component c (rows are h * cache_cap + pos)."""
    var l = c.layer
    if c.kind == KIND_K_DATA:
        return eng.d_k_cache_i8[l]
    if c.kind == KIND_V_DATA:
        return eng.d_v_cache_i8[l]
    if c.kind == KIND_K_SCALE:
        return eng.d_k_scales[l]
    if c.kind == KIND_V_SCALE:
        return eng.d_v_scales[l]
    # block-32 fp32 tail: after nkv * cap * ceil(hd/32) fp16 scales (kv_cache_quant.mojo)
    var tail = UInt64(eng.layer_nkv[l] * eng.layer_cache_cap[l] * ((eng.layer_hd[l] + 31) // 32) * 2)
    if c.kind == KIND_K_TAIL:
        return eng.d_k_scales[l] + tail
    return eng.d_v_scales[l] + tail


def _drafter_base(eng: GemmaEngine, c: ComponentDesc) -> UInt64:
    """Device address of row 0 of drafter component c: the drafter's context cache, position-major fp32
    [max_ctx, DFLASH_KV_DIM] (dflash_drafter.mojo:490), so a token range is one contiguous run."""
    if c.kind == KIND_DRAFT_K:
        return _drafter(eng)[unsafe_offset=0].d_ctx_k_cache[c.layer]
    return _drafter(eng)[unsafe_offset=0].d_ctx_v_cache[c.layer]


def _copy_stream(mut eng: GemmaEngine) -> UInt64:
    if eng.kv_copy_stream == 0:
        eng.kv_copy_stream = cuda_stream_create()
        eng.kv_copy_event = cuda_event_create()
    return eng.kv_copy_stream


def _wait_copies(eng: GemmaEngine) raises:
    if cuda_event_record(eng.kv_copy_event, eng.kv_copy_stream) != 0 or cuda_event_sync(eng.kv_copy_event) != 0:
        cuda_sync()   # never leave a copy in flight behind a raise
        raise Error("kv session: copy stream failed")


def export_range(mut eng: GemmaEngine, layout: KvLayout, a: Int, b: Int, drafter_end: Int, dst: UInt64) raises:
    """Copy tokens [a, b) of every component into host memory at dst, packed per component as [head][token][row].
    drafter_end: the drafter's context rows are valid for [a, drafter_end). That is all the drafter components carry;
    the rest of each is left as the caller had it (the drafter trails the target after a base-path turn)."""
    _require_linear_quantized(eng, "kv export")
    if a < 0 or b <= a or b > eng.cache_len():
        raise Error("kv export: bad range [" + String(a) + ", " + String(b) + ") with cache_len " + String(eng.cache_len()))
    var stream = _copy_stream(eng)
    cuda_sync()   # every kernel that wrote this KV, on any stream, has finished
    var ntok = b - a
    var off = 0
    for i in range(len(layout.comps)):
        var c = layout.comps[i]
        if c.is_drafter():
            var rows = min(drafter_end, b) - a
            if rows > 0:
                var rc_d = cuda_memcpy_async_on(dst + UInt64(off), _drafter_base(eng, c) + UInt64(a * c.row_bytes),
                                                rows * c.row_bytes, _D2H, stream)
                if rc_d != 0:
                    cuda_sync()   # never leave a copy in flight behind a raise (ledger R29)
                    raise Error("kv export: drafter cudaMemcpyAsync rc=" + String(rc_d))
            off += ntok * c.bytes_per_token()
            continue
        var cap = eng.layer_cache_cap[c.layer]
        var src = _target_base(eng, c) + UInt64(a * c.row_bytes)
        var rc = cuda_memcpy_2d_async_on(
            dst + UInt64(off), ntok * c.row_bytes, src, cap * c.row_bytes, ntok * c.row_bytes, c.heads, _D2H, stream)
        if rc != 0:
            cuda_sync()   # earlier components' copies may still be in flight into the caller's buffer
            raise Error("kv export: cudaMemcpy2DAsync rc=" + String(rc))
        off += ntok * c.bytes_per_token()
    _wait_copies(eng)


def import_range(
    mut eng: GemmaEngine, layout: KvLayout, a: Int, b: Int, drafter_end: Int,
    src: UInt64, src_tokens: Int, tokens: List[Int],
) raises:
    """Inverse of export_range for tokens [a, b): the first b - a tokens of a chunk exported with src_tokens tokens.
    Requires a == cache_len() and live_tokens in step with it; appends `tokens` to live_tokens. The drafter's rows
    [a, drafter_end) are imported only onto a drafter that holds exactly [0, a), whose length then becomes drafter_end;
    after a gap in its context the drafter is left as it is (rows after a gap would be read as contiguous). The caller
    keeps drafter_end <= b."""
    _require_linear_quantized(eng, "kv import")
    if a != eng.cache_len() or len(eng.live_tokens) != a or eng.live_desync:
        # live_desync (Task 4, ledger R15): untracked KV is below us; importing on top would mislabel it.
        raise Error("kv import: must continue at cache_len " + String(eng.cache_len()) + ", got " + String(a))
    if b <= a or b - a > src_tokens or len(tokens) != b - a or b > eng.max_seq:
        raise Error("kv import: bad range [" + String(a) + ", " + String(b) + ")")
    var do_drafter = eng.dflash_ptr != 0 and drafter_end > a and _drafter(eng)[unsafe_offset=0].cache_len() == a
    var stream = _copy_stream(eng)
    cuda_sync()
    var ntok = b - a
    var off = 0
    for i in range(len(layout.comps)):
        var c = layout.comps[i]
        if c.is_drafter():
            if do_drafter:
                var rc_d = cuda_memcpy_async_on(_drafter_base(eng, c) + UInt64(a * c.row_bytes), src + UInt64(off),
                                                (drafter_end - a) * c.row_bytes, _H2D, stream)
                if rc_d != 0:
                    cuda_sync()   # never leave a copy in flight behind a raise (ledger R29)
                    raise Error("kv import: drafter cudaMemcpyAsync rc=" + String(rc_d))
            off += src_tokens * c.bytes_per_token()
            continue
        var cap = eng.layer_cache_cap[c.layer]
        var dst = _target_base(eng, c) + UInt64(a * c.row_bytes)
        var rc = cuda_memcpy_2d_async_on(
            dst, cap * c.row_bytes, src + UInt64(off), src_tokens * c.row_bytes, ntok * c.row_bytes, c.heads, _H2D, stream)
        if rc != 0:
            cuda_sync()   # earlier components' copies may still be in flight out of the caller's buffer
            raise Error("kv import: cudaMemcpy2DAsync rc=" + String(rc))
        off += src_tokens * c.bytes_per_token()
    _wait_copies(eng)
    eng.set_cache_len(b)
    for i in range(len(tokens)):
        eng.live_tokens.append(tokens[i])
    if do_drafter:
        _drafter(eng)[unsafe_offset=0].set_cache_len(drafter_end)


def kv_truncate(mut eng: GemmaEngine, p: Int) raises:
    if p < 0 or p > eng.cache_len():
        raise Error("kv truncate: " + String(p) + " beyond cache_len " + String(eng.cache_len()))
    eng.set_cache_len(p)   # also trims live_tokens
    if eng.dflash_ptr != 0:
        # The drafter's context follows the target's cut but never grows to it: it may trail the target.
        var dp = _drafter(eng)
        dp[unsafe_offset=0].set_cache_len(min(dp[unsafe_offset=0].cache_len(), p))


def kv_reset_session(mut eng: GemmaEngine) raises:
    eng.reset_kv_cache()
    if eng.dflash_ptr != 0:
        _drafter(eng)[unsafe_offset=0].reset()


def kv_drafter_len(eng: GemmaEngine) -> Int:
    """How many leading tokens have drafter context rows: the drafter's length, 0 when none is loaded."""
    return _drafter(eng)[unsafe_offset=0].cache_len() if eng.dflash_ptr != 0 else 0


def kv_drafter_cap(eng: GemmaEngine) -> Int:
    """How many context rows the loaded drafter can hold (its max_ctx), 0 when none is loaded."""
    return _drafter(eng)[unsafe_offset=0].max_ctx if eng.dflash_ptr != 0 else 0


def drafter_id_of(dir: String) raises -> UInt64:
    """The drafter's weights identity (docs/KV_SESSION_STORE_DESIGN.md §3.7): context produced by different drafter
    weights must never be reused. Same recipe as the target's weights (`weights_id_of`, lib/kvstore/key.mojo, ledger
    R35): convert_manifest.json plus every file's name and size, order-independent over files."""
    return weights_id_of(dir)


comptime _GIB = 1073741824.0


def kvstore_root() -> String:
    """NOMOS_KVSTORE_DIR, or its default: where every muse-<id> store directory lives."""
    return getenv("NOMOS_KVSTORE_DIR", getenv("HOME", "/root") + "/nomos_data/kvstore")


def store_config_from_env(build_id: UInt64) -> StoreConfig:
    """NOMOS_KVSTORE_* (docs/KV_SESSION_STORE_DESIGN.md §3.5). Files live under <dir>/muse-<build_id as 16 hex digits>,
    where build_id is the cache key's: the kernel's BUILD_ID mixed with the engine's numerics id (ledger R35). Engines
    with different numerics therefore never share files, locks or budgets."""
    var root = kvstore_root()
    return StoreConfig(
        root + "/muse-" + hex16(build_id),
        Int(Float64(_env_float("NOMOS_KVSTORE_RAM_GB", 8.0)) * _GIB),
        Int(Float64(_env_float("NOMOS_KVSTORE_DISK_GB", 100.0)) * _GIB),
        Int(_env_float("NOMOS_KVSTORE_MIN_REUSE", 16.0)),
        Int(_env_float("NOMOS_KVSTORE_MIN_SWITCH", 256.0)),
        True,
        Int(_env_float("NOMOS_KVSTORE_CHUNK_TOKENS", 4096.0)),
    )


def store_status(eng: GemmaEngine) -> Int:
    """1 = enabled and supported (the store may still have been disabled for this process: nomos_session_prepare), 0 = off
    (NOMOS_KVSTORE unset), -10 profile not supported, -11 KV_SWA=1, -12 unquantized KV."""
    if _env_float("NOMOS_KVSTORE", 0.0) <= 0.5:
        return 0
    if not store_supported(MODEL_ID):
        return -10
    if eng.kv_swa:
        return -11
    if kv_mode_of(eng) == KV_MODE_UNSUPPORTED:
        return -12
    return 1


def release_kvstore_handle(addr: UInt64):
    """Close the store (best-effort disk catch-up, free its RAM, release the directory lock) and free its heap cell.
    The caller sets GemmaEngine.kvstore_ptr to 0 afterwards. Frees pinned memory, so the CUDA context must be alive."""
    if addr == 0:
        return
    var sp = Pointer[SessionStore, MutUntrackedOrigin](unsafe_from_address=Int(addr))
    sp[unsafe_offset=0].close()
    sp.unsafe_deinit_pointee()
    sp.unsafe_free()


def _bytes_in(dir: String, skip: Dict[UInt64, Int]) -> Tuple[Int, Int]:
    """(count, bytes) of the session files directly under dir whose id is not in skip. Best effort: what cannot be listed
    or sized counts as nothing."""
    var n = 0
    var total = 0
    try:
        var names = listdir(dir)
        for i in range(len(names)):
            var id = parse_hex16(String(names[i].removesuffix(".nkvs")))
            if not names[i].endswith(".nkvs") or not id or id.value() in skip:
                continue
            n += 1
            total += getsize(dir + "/" + names[i])
    except:
        pass
    return (n, total)


def _print_other_files(root: String, store: SessionStore):
    """One line on what lies under NOMOS_KVSTORE_DIR besides this store's sessions (final review M5): the other muse-*
    directories (other builds, weights or numerics) and the session files this store skipped (another cache key, or
    unreadable). Neither counts against NOMOS_KVSTORE_DISK_GB or in /v1/kvstore/stats; tools/kvstore_dirs.sh lists them."""
    var ndirs = 0
    var dir_bytes = 0
    try:
        var names = listdir(root)
        for i in range(len(names)):
            var p = root + "/" + names[i]
            if names[i].startswith("muse-") and p != store.cfg.dir and isdir(p):
                ndirs += 1
                dir_bytes += _bytes_in(p, Dict[UInt64, Int]())[1]
    except:
        pass
    var skipped = _bytes_in(store.cfg.dir, store.disk.sizes)
    print("[kvstore] also under " + root + ": " + String(ndirs) + " other muse-* directories (" + String(dir_bytes)
          + " bytes of sessions) and " + String(skipped[0]) + " skipped session files in this one (" + String(skipped[1])
          + " bytes); none of them counts against NOMOS_KVSTORE_DISK_GB")


def session_store(mut eng: GemmaEngine) raises -> UnsafePointer[SessionStore, MutUntrackedOrigin]:
    """The engine's store, created on first use and re-created when the cache key changes, e.g. when a DFlash
    drafter is loaded after the store exists (Review Focus 3): old-layout sessions are then never imported.

    The key's build component is the kernel's BUILD_ID mixed with the engine's numerics id (its weights and precision
    flags, ledger R35), so KV computed by other code, weights or precision is never restored.

    The returned pointer is valid only until the next session_store() call, which may re-key (free this store and make
    another), or until nomos_shutdown. A re-key starts the new store's counters from zero. Each store it creates prints
    `[kvstore] dir=<path>`, naming its directory, and one line on the other files under NOMOS_KVSTORE_DIR."""
    var layout = build_layout(eng)
    var build_id = fnv1a64_u64(BUILD_ID, eng.numerics_id)
    var key = CacheKey(MODEL_ID, build_id, kv_mode_of(eng), layout.hash(), UInt64(0), eng.dflash_id)
    if eng.kvstore_ptr != 0:
        var sp = UnsafePointer[SessionStore, MutUntrackedOrigin](unsafe_from_address=Int(eng.kvstore_ptr))
        if sp[unsafe_offset=0].key == key:
            return sp
        release_kvstore_handle(eng.kvstore_ptr)
        eng.kvstore_ptr = UInt64(0)
    var store = SessionStore(key, layout, store_config_from_env(build_id))   # raises on a bad config or a held lock
    # The line that names the live directory as NOMOS_KVSTORE_DIR spells it (docs/RUNBOOK.md §7; the serve's /health gives
    # its canonical path). Directories of other builds, weights and numerics sit beside it: the next line counts them.
    print("[kvstore] dir=" + store.cfg.dir)
    _print_other_files(kvstore_root(), store)
    var fresh = unsafe_alloc[SessionStore](1)
    fresh.unsafe_write(store^)
    eng.kvstore_ptr = UInt64(Int(fresh))
    return UnsafePointer[SessionStore, MutUntrackedOrigin](unsafe_from_address=Int(fresh))
