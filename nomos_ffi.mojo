"""Nomos kernel FFI — C ABI exports for the Go cgo bridge.

This file is the public C-ABI surface of the Nomos kernel as a shared
library. It mirrors what `gemma4_unified` exposes over TCP today, but
via direct function calls. Pointers cross the boundary as Int64 addresses
(matching the C-ABI FFI convention).

Build:
    pixi run mojo build --emit shared-lib src/nomos_ffi.mojo \\
        -o libnomos_kernel.so \\
        -Xlinker -L/usr/local/cuda/lib64 \\
        -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm

Stage 1: skeleton. Functions return placeholder values; just enough to
prove the cgo build chain is wired correctly. Stage 2 will replace stubs
with real init/generate logic.

ABI conventions:
- All pointer args are Int64 addresses (cgo passes uintptr_t as int64_t)
- All status returns use Int32: 0=ok, negative=error
- Token IDs are Int32 (room for the full 262144 vocab + sentinels)
- Opaque context handles are Int64 (heap pointer)
"""

from std.memory import UnsafePointer, alloc
from std.ffi import external_call, c_size_t
from std.math import sqrt
from lib.gemma4_engine import GemmaEngine, init_engine_handle, release_engine_handle, D, VOCAB, TOTAL_LAYERS, SLIDING_WINDOW, _env_float
from lib.model_config import (
    MODEL_ID, EAGLE3_TAPS, DRAFTER_EMBED_SQRT_SCALE, RMS_EPS_FINAL,
    HAS_LINEAR_ATTENTION, GDN_NUM_V_HEADS, GDN_KEY_HEAD_DIM,
    GDN_VALUE_HEAD_DIM, GDN_CONV_DIM, GDN_CONV_KERNEL,
    MAX_PROBE_TOKENS,
)
from lib.gdn_scan import gpu_gdn_bf16_to_f32
from lib.mtp_drafter import MTPDrafter, mtp_draft_k, release_mtp_drafter_handle
from lib.engine_prefill import prefill_batch_impl
from lib.eagle3_drafter import (
    Eagle3Drafter,
    eagle3_draft_k,
    eagle3_commit_prefix,
    release_eagle3_drafter_handle,
    MAX_VERIFY_ROWS,
    EAGLE_MODE_Q8,
    EAGLE_MODE_NVFP4,
    EAGLE_MODE_GOLD,
    DRAFT_VOCAB,
)
from lib.e2b_draft_engine import (
    E2BDraftEngine,
    init_e2b_draft_handle,
    init_e2b_draft_handle_with_context,
    release_e2b_draft_handle,
)
from lib.dflash_drafter import (
    DFlashDrafter,
    DFLASH_BLOCK,
    DFLASH_CANDIDATES,
    DFLASH_CTX_BATCH,
    DFLASH_FC_IN,
    DFLASH_HIDDEN,
    DFLASH_LAYERS,
    DFLASH_MAX_CTX,
    DFLASH_MAX_VERIFY_ROWS,
    dflash_append_context_rows,
    dflash_draft_block,
    dflash_draft_block_nvfp4,
    dflash_project_context,
    dflash_forward_block_embeddings,
    dflash_markov_bias,
    dflash_markov_sample_logits,
    release_dflash_drafter_handle,
)
from lib.dflash_profile import (
    dflash_profile_tap_layers,
    eagle3_profile_tap_layers,
    validate_dflash_profile_metadata,
)
from lib.io import file_size_bytes
from lib.cuda import cuda_malloc, cuda_free, cuda_memcpy, cuda_sync, cuda_upload, cuda_download
from lib.gemma4_ops import rmsnorm
from lib.engine_init import _read_env_bytes
from lib.kv_cache_quant import (
    gpu_append_quant_kv_i4,
    gpu_append_quant_kv_i4_with_q8,
    gpu_dequant_kv_i4_to_q8_layer,
)


# ── C-ABI helpers (read/write across the cgo boundary) ──────────────────────
def _read_cstr(ptr: Int64) raises -> String:
    """Read a NUL-terminated C string from address `ptr` into a Mojo String."""
    if ptr == 0:
        return String("")
    var p = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ptr))
    var s = String("")
    var i = 0
    while True:
        var c = p[i]
        if c == 0:
            break
        s += chr(Int(c))
        i += 1
        if i > 4096:
            break
    return s

def _read_int32_array_into(ptr: Int64, n: Int, mut out: List[Int]) raises:
    """Read `n` Int32 values from address `ptr` into `out`."""
    var p = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ptr))
    for i in range(n):
        out.append(Int(p[i]))

def _write_int32_array(ptr: Int64, ref values: List[Int], capacity: Int) raises -> Int:
    """Write min(len(values), capacity) values to `ptr` as Int32. Returns count written."""
    var p = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ptr))
    var n = len(values) if len(values) < capacity else capacity
    for i in range(n):
        p[i] = Int32(values[i])
    return n


def _env_is_one(name: String) -> Bool:
    var b = _read_env_bytes(name)
    return len(b) == 1 and b[0] == UInt8(ord("1"))


# ─────────────────────────────────────────────────────────────────────────────
# Library version
# ─────────────────────────────────────────────────────────────────────────────


@export
def nomos_model_id() -> Int32:
    """Which MODEL PROFILE this .so was compiled for (lib/model_profiles/*.mojo).

    Exists so the host can ask the KERNEL what it is instead of trusting a filename or
    a build log. One .so serves exactly one model geometry, so loading the wrong weights
    cannot work and should fail at the front door -- see tools/check_model_identity.py.
    """
    return Int32(MODEL_ID)


@export
def nomos_version() -> Int32:
    """Returns ABI version. Bump when breaking changes land."""
    return Int32(1)


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 stub: prove cgo round-trip
# ─────────────────────────────────────────────────────────────────────────────

@export
def nomos_strict_violation_count() -> Int64:
    """Precision-law guard: # forbidden-precision dispatches reached since reset (only counts
    under NOMOS_STRICT_Q4=1). 0 after a Q4-prod prefill+decode+verify == law-compliant."""
    return Int64(external_call["nomos_violation_count", Int64]())

@export
def nomos_strict_violation_mask() -> Int64:
    """Bitmask of which VIOL_* precision-law violations fired (see lib/engine_init.mojo)."""
    return Int64(external_call["nomos_violation_mask", Int64]())

@export
def nomos_strict_reset() -> Int32:
    """Reset the precision-law violation counter+mask (call before a guarded run)."""
    return Int32(external_call["nomos_reset_violations", Int32]())

@export
def nomos_init(weights_dir_ptr: Int64) -> Int64:
    """Load weights, allocate KV caches, return context handle.

    Returns 0 on failure, otherwise opaque heap pointer (GemmaEngine address).
    """
    try:
        var weights_dir = _read_cstr(weights_dir_ptr)
        var handle = init_engine_handle(weights_dir)
        return Int64(Int(handle))
    except:
        return Int64(0)

@export
def nomos_shutdown(handle: Int64) -> Int32:
    """Free a context allocated by nomos_init. Idempotent."""
    if handle == 0:
        return Int32(0)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )
    if engine_ptr[0].mtp_ptr != 0:
        release_mtp_drafter_handle(engine_ptr[0].mtp_ptr)
        engine_ptr[0].mtp_ptr = UInt64(0)
    if engine_ptr[0].eagle3_ptr != 0:
        release_eagle3_drafter_handle(engine_ptr[0].eagle3_ptr)
        engine_ptr[0].eagle3_ptr = UInt64(0)
        engine_ptr[0].clear_drafter_taps()
    if engine_ptr[0].dflash_ptr != 0:
        release_dflash_drafter_handle(engine_ptr[0].dflash_ptr)
        engine_ptr[0].dflash_ptr = UInt64(0)
        engine_ptr[0].clear_drafter_taps()
    release_engine_handle(UInt64(handle))
    return Int32(0)


@export
def nomos_lm_draft_load(dir_ptr: Int64, max_seq: Int32) -> Int64:
    """Load a standalone LM draft model, currently Gemma-4-E2B.

    Returns an opaque draft-engine handle. This handle is independent of the
    31B target handle; host code owns the two KV lifecycles separately.
    """
    if dir_ptr == 0:
        return Int64(0)
    try:
        var dir = _read_cstr(dir_ptr)
        var ms = Int(max_seq)
        if ms <= 0:
            ms = 1024
        var handle = init_e2b_draft_handle(dir, ms)
        return Int64(Int(handle))
    except e:
        print("[nomos_lm_draft_load EXC]", e)
        return Int64(0)


@export
def nomos_lm_draft_load_with_target(
    target_handle: Int64, dir_ptr: Int64, max_seq: Int32
) -> Int64:
    """Load an LM draft handle for target-paired speculative decoding.

    Target and draft keep disjoint weights/KV/scratch. By default the draft
    engine also gets its own Mojo DeviceContext/stream so host-side overlap can
    draft ahead while target verify runs. Set NOMOS_LM_DRAFT_SHARE_TARGET_CTX=1
    to restore the old single-stream behavior for A/B or rollback.
    """
    if target_handle == 0 or dir_ptr == 0:
        return Int64(0)
    try:
        var dir = _read_cstr(dir_ptr)
        var ms = Int(max_seq)
        if ms <= 0:
            ms = 1024
        var ep = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
            unsafe_from_address=Int(target_handle)
        )
        var handle = UInt64(0)
        if _env_is_one("NOMOS_LM_DRAFT_SHARE_TARGET_CTX"):
            handle = init_e2b_draft_handle_with_context(dir, ms, ep[0].ctx)
        else:
            handle = init_e2b_draft_handle(dir, ms)
        return Int64(Int(handle))
    except e:
        print("[nomos_lm_draft_load_with_target EXC]", e)
        return Int64(0)


@export
def nomos_lm_draft_shutdown(draft_handle: Int64) -> Int32:
    if draft_handle == 0:
        return Int32(0)
    release_e2b_draft_handle(UInt64(draft_handle))
    return Int32(0)


@export
def nomos_lm_draft_prefill(
    draft_handle: Int64,
    ids_ptr: Int64,
    n: Int32,
    out_token_ptr: Int64,
) -> Int32:
    """Reset + prefill the draft model with ids[0..n-1].

    Writes the greedy next-token prediction after the final prompt token.
    Leaves draft KV length == n. 0=ok, -1=bad arg, -99=exception.
    """
    if draft_handle == 0 or ids_ptr == 0 or out_token_ptr == 0 or n <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var ids = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), ids)
        var tok = dp[0].prefill(ids)
        UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_token_ptr))[0] = Int32(tok)
        return Int32(0)
    except e:
        print("[nomos_lm_draft_prefill EXC]", e, " n=", Int(n))
        return Int32(-99)


@export
def nomos_lm_draft_debug_prefill(
    draft_handle: Int64,
    ids_ptr: Int64,
    n: Int32,
    out_hs_ptr: Int64,
    out_logits_ptr: Int64,
    out_token_ptr: Int64,
) -> Int32:
    """Debug prefill for E2B parity.

    Runs reset+prefill over ids[0..n-1], writes:
    - out_hs_ptr: optional host float[36*1536], rows [embed, layer0..layer34]
      for the last prompt token;
    - out_logits_ptr: optional host float[262144], final post-softcap logits;
    - out_token_ptr: required Int32 greedy next token.
    Leaves draft KV length == n. 0=ok, -1=bad arg, -99=exception.
    """
    if draft_handle == 0 or ids_ptr == 0 or out_token_ptr == 0 or n <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var ids = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), ids)
        var tok = dp[0].prefill_debug_last(ids, UInt64(out_hs_ptr), UInt64(out_logits_ptr))
        UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_token_ptr))[0] = Int32(tok)
        return Int32(0)
    except e:
        print("[nomos_lm_draft_debug_prefill EXC]", e, " n=", Int(n))
        return Int32(-99)


@export
def nomos_lm_draft_debug_prefill_layer(
    draft_handle: Int64,
    ids_ptr: Int64,
    n: Int32,
    trace_layer: Int32,
    out_phase_ptr: Int64,
    out_meta_ptr: Int64,
    out_token_ptr: Int64,
) -> Int32:
    """Debug one E2B layer on the last prompt token.

    Runs reset+prefill over ids[0..n-1], writes:
    - out_phase_ptr: optional host float[4*1536]:
      [layer input, post-attn residual, post-MLP residual, final layer output];
    - out_meta_ptr: optional host int32[4]:
      [read_layer, q_head_dim, read_head_dim, is_shared];
    - out_token_ptr: required Int32 greedy next token.
    Leaves draft KV length == n. 0=ok, -1=bad arg, -99=exception.
    """
    if draft_handle == 0 or ids_ptr == 0 or out_token_ptr == 0 or n <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var ids = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), ids)
        var tok = dp[0].prefill_debug_layer(
            ids, Int(trace_layer), UInt64(out_phase_ptr), UInt64(out_meta_ptr)
        )
        UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_token_ptr))[0] = Int32(tok)
        return Int32(0)
    except e:
        print("[nomos_lm_draft_debug_prefill_layer EXC]", e, " n=", Int(n), " layer=", Int(trace_layer))
        return Int32(-99)


@export
def nomos_lm_draft_debug_prefill_layer_attn(
    draft_handle: Int64,
    ids_ptr: Int64,
    n: Int32,
    trace_layer: Int32,
    out_phase_ptr: Int64,
    out_meta_ptr: Int64,
    out_attn_ptr: Int64,
    out_attn_meta_ptr: Int64,
    out_token_ptr: Int64,
) -> Int32:
    """Debug one E2B layer plus attention inputs on the last prompt token.

    Includes all outputs from nomos_lm_draft_debug_prefill_layer and adds:
    - out_attn_ptr: optional host float[80], five rows of 16 floats:
      [q0, k_first, v_first, k_last, v_last];
    - out_attn_meta_ptr: optional host int32[8]:
      [read_layer, pos_plus_one, attn_len, win_start,
       q_head_dim, read_head_dim, kvg, max_seq].
    """
    if draft_handle == 0 or ids_ptr == 0 or out_token_ptr == 0 or n <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var ids = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), ids)
        var tok = dp[0].prefill_debug_layer(
            ids,
            Int(trace_layer),
            UInt64(out_phase_ptr),
            UInt64(out_meta_ptr),
            UInt64(out_attn_ptr),
            UInt64(out_attn_meta_ptr),
        )
        UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_token_ptr))[0] = Int32(tok)
        return Int32(0)
    except e:
        print("[nomos_lm_draft_debug_prefill_layer_attn EXC]", e, " n=", Int(n), " layer=", Int(trace_layer))
        return Int32(-99)


@export
def nomos_lm_draft_step_token(draft_handle: Int64, token: Int32, out_token_ptr: Int64) -> Int32:
    """Append one token to the draft model and write its greedy next-token prediction."""
    if draft_handle == 0 or out_token_ptr == 0:
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var tok = dp[0].step_token(Int(token))
        UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_token_ptr))[0] = Int32(tok)
        return Int32(0)
    except e:
        print("[nomos_lm_draft_step_token EXC]", e, " token=", Int(token))
        return Int32(-99)


@export
def nomos_lm_draft_draft(
    draft_handle: Int64,
    seed_token: Int32,
    k: Int32,
    out_ptr: Int64,
) -> Int32:
    """Draft up to k tokens by sequential LM decode.

    Contract for the two-call spec loop:
    - caller positions draft KV at the committed prefix before seed_token;
    - this call consumes seed_token, then each produced draft token;
    - caller rolls back with nomos_lm_draft_set_len(base+1+num_acc).
    """
    if draft_handle == 0 or out_ptr == 0 or k <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var drafts = List[Int32]()
        var n = dp[0].draft_k(Int(seed_token), Int(k), drafts)
        var op = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_ptr))
        for i in range(n):
            op[i] = drafts[i]
        return Int32(n)
    except e:
        print("[nomos_lm_draft_draft EXC]", e, " seed=", Int(seed_token), " k=", Int(k))
        return Int32(-99)


@export
def nomos_lm_draft_draft_conf(
    draft_handle: Int64,
    seed_token: Int32,
    k: Int32,
    out_ptr: Int64,
    out_conf_ptr: Int64,
    out_gap_ptr: Int64,
) -> Int32:
    """Draft up to k tokens and write per-slot drafter confidence.

    out_ptr: Int32[k] draft token ids.
    out_conf_ptr: Float32[k] top-1 softmax probabilities.
    out_gap_ptr: optional Float32[k] top-1 prob minus top-2 prob; pass 0 to skip.
    """
    if draft_handle == 0 or out_ptr == 0 or out_conf_ptr == 0 or k <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var drafts = List[Int32]()
        var confs = List[Float32]()
        var gaps = List[Float32]()
        var n = dp[0].draft_k_conf(
            Int(seed_token), Int(k), drafts, confs, gaps
        )
        var op = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_ptr))
        var cp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_conf_ptr))
        for i in range(n):
            op[i] = drafts[i]
            cp[i] = confs[i]
        if out_gap_ptr != 0:
            var gp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_gap_ptr))
            for i in range(n):
                gp[i] = gaps[i]
        return Int32(n)
    except e:
        print("[nomos_lm_draft_draft_conf EXC]", e, " seed=", Int(seed_token), " k=", Int(k))
        return Int32(-99)


@export
def nomos_lm_draft_draft_until(
    draft_handle: Int64,
    seed_token: Int32,
    k_max: Int32,
    p_min: Float32,
    out_ptr: Int64,
    out_conf_ptr: Int64,
) -> Int32:
    """Draft until confidence drops below p_min or k_max is reached.

    Writes only accepted-for-verification slots:
    - out_ptr: Int32[count] draft token ids;
    - out_conf_ptr: Float32[count] top-1 softmax probabilities.

    A below-threshold probe advances draft KV just enough to decide to stop,
    but it is not returned. The resulting KV length matches the existing
    rollback contract: base + 1 + returned_count.
    """
    if draft_handle == 0 or out_ptr == 0 or out_conf_ptr == 0 or k_max <= Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    try:
        var drafts = List[Int32]()
        var confs = List[Float32]()
        var n = dp[0].draft_until_conf(
            Int(seed_token), Int(k_max), p_min, drafts, confs
        )
        var op = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_ptr))
        var cp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_conf_ptr))
        for i in range(n):
            op[i] = drafts[i]
            cp[i] = confs[i]
        return Int32(n)
    except e:
        print(
            "[nomos_lm_draft_draft_until EXC]",
            e,
            " seed=",
            Int(seed_token),
            " k=",
            Int(k_max),
            " p_min=",
            p_min,
        )
        return Int32(-99)


@export
def nomos_lm_draft_cache_len(draft_handle: Int64) -> Int32:
    if draft_handle == 0:
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    return Int32(dp[0].cache_len())


@export
def nomos_lm_draft_set_len(draft_handle: Int64, n: Int32) -> Int32:
    if draft_handle == 0 or n < Int32(0):
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    dp[0].set_cache_len(Int(n))
    return Int32(0)


@export
def nomos_lm_draft_reset(draft_handle: Int64) -> Int32:
    if draft_handle == 0:
        return Int32(-1)
    var dp = UnsafePointer[E2BDraftEngine, MutUntrackedOrigin](unsafe_from_address=Int(draft_handle))
    dp[0].reset()
    return Int32(0)


# (Mode-2 admin surface was removed in the perf-clean fork; ABI kept stable.)

@export
def nomos_generate(
    handle: Int64,
    prompt_ptr: Int64,
    n_prompt: Int32,
    max_new_tokens: Int32,
    out_ptr: Int64,
    out_capacity: Int32,
    temperature: Float32,
    top_p: Float32,
    rep_penalty: Float32,
) -> Int32:
    """Run prefill + decode. Returns the number of tokens written, or negative on error.

    Sampling params override the engine's per-request defaults: temperature < 0
    keeps the current default (temperature == 0 means greedy); top_p <= 0 keeps
    its default. Repetition penalty uses HF-style multiplicative semantics:
    rep_penalty == 1.0 is OFF and, like <= 0, preserves the raw legacy path."""
    if handle == 0:
        return Int32(-1)
    if n_prompt <= 0 or max_new_tokens <= 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    # Per-call sampling: snapshot the engine defaults so a request's overrides
    # do NOT persist into later requests (2026-06-14: overrides were leaking via
    # engine state). When no override is passed, saved==current → restore is a
    # no-op → default behavior is byte-identical.
    var saved_temp = engine_ptr[0].temp
    var saved_top_p = engine_ptr[0].top_p
    var saved_rep = engine_ptr[0].rep_penalty
    try:
        if temperature >= Float32(0.0):
            engine_ptr[0].temp = temperature
        if top_p > Float32(0.0):
            engine_ptr[0].top_p = top_p
        if rep_penalty > Float32(0.0):
            engine_ptr[0].rep_penalty = rep_penalty
        var prompt = List[Int]()
        _read_int32_array_into(prompt_ptr, Int(n_prompt), prompt)
        var out = List[Int]()
        # This export owns a complete independent request.  Reset both the
        # append-only KV lengths and any recurrent GDN pools before prefill;
        # otherwise a reused handle leaks the previous request into this one.
        engine_ptr[0].reset_kv_cache()
        engine_ptr[0].run_inference(
            prompt, Int(max_new_tokens), False, out,
            force_rep_penalty=(
                rep_penalty > Float32(0.0) and rep_penalty != Float32(1.0)
            ),
        )
        engine_ptr[0].temp = saved_temp
        engine_ptr[0].top_p = saved_top_p
        engine_ptr[0].rep_penalty = saved_rep
        var written = _write_int32_array(out_ptr, out, Int(out_capacity))
        return Int32(written)
    except e:
        print("[nomos_generate EXC]", e, "  n_prompt=", Int(n_prompt), " max_new=", Int(max_new_tokens))
        engine_ptr[0].temp = saved_temp
        engine_ptr[0].top_p = saved_top_p
        engine_ptr[0].rep_penalty = saved_rep
        return Int32(-99)

@export
def nomos_prefill(handle: Int64, ids_ptr: Int64, n: Int32, logits_out_ptr: Int64) -> Int32:
    """LOGITS-ENGINE prefill (Python-host path). Run the prompt through the model and
    write the last-position logits (VOCAB fp32, post-softcap) to logits_out_ptr. The KV
    cache is left populated for nomos_decode_step. If DFlash is loaded, this fresh
    prefill also resets/builds the DFlash context KV from captured prompt taps.
    The host does sampling + grammar. 0 = ok, -1 = bad arg, -3 = DFlash ctx cap,
    -99 = exception."""
    if handle == 0 or n <= Int32(0) or ids_ptr == 0 or logits_out_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var prompt = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), prompt)
        # nomos_prefill is the authoritative FRESH-request boundary.  Clearing
        # only cache lengths was sufficient for attention-only Gemma/Muse, but
        # Qwen's recurrent GDN pools otherwise carried request N into request
        # N+1.  Continuations use nomos_prefill_cont and deliberately bypass
        # this reset.
        engine_ptr[0].reset_kv_cache()
        var dflash_loaded = engine_ptr[0].dflash_ptr != 0
        if dflash_loaded:
            var dp_reset = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
                unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
            dp_reset[0].reset()
        # #74: a success return MUST mean the buffer was written. The chunked-prefill defect
        # returned rc=0 with logits_out untouched, so every caller downstream decoded whatever
        # was already in its own buffer -- zeros on a fresh one, the PREVIOUS REQUEST'S LOGITS
        # otherwise. That is the failure that reads as a clean result. Poison the first and last
        # element, then verify the callee overwrote them.
        var _lp0 = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(logits_out_ptr))
        _lp0[0] = Float32(-1.0e30)
        _lp0[VOCAB - 1] = Float32(-1.0e30)
        engine_ptr[0].prefill_logits(prompt, logits_out_ptr)
        if _lp0[0] == Float32(-1.0e30) and _lp0[VOCAB - 1] == Float32(-1.0e30):
            print("[nomos_prefill] ERROR: engine returned without writing logits.",
                  "Refusing to report success (#74).")
            return Int32(-4)
        if dflash_loaded:
            var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
                unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
            if dp[0].cache_len() + 1 > DFLASH_MAX_CTX:
                return Int32(-3)
            dflash_append_context_rows(
                engine_ptr[0].ctx,
                dp[0],
                dp[0].d_taps_buf,
                1,
            )
            engine_ptr[0].ctx.synchronize()
        return Int32(0)
    except e:
        print("[nomos_prefill EXC]", e, "  n=", Int(n))
        return Int32(-99)

@export
def nomos_decode_step(handle: Int64, token: Int32, logits_out_ptr: Int64) -> Int32:
    """LOGITS-ENGINE decode step (Python-host path). Append `token` at the current cache
    position (continuing the held KV) and write the next logits (VOCAB fp32, post-softcap)
    to logits_out_ptr. The host does sampling + grammar. 0 = ok, -1 = bad arg, -99 = exc."""
    if handle == 0 or logits_out_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        engine_ptr[0].step_logits(Int(token), logits_out_ptr)
        return Int32(0)
    except e:
        print("[nomos_decode_step EXC]", e, "  token=", Int(token))
        return Int32(-99)

@export
def nomos_mtp_load(handle: Int64, dir_ptr: Int64) -> Int32:
    """Load the MTP spec-decode drafter from the assistant weights dir (C-string at
    `dir_ptr`) into the engine, behind engine.mtp_ptr. Call once after nomos_init.
    0 = ok, -1 = bad arg, -99 = exception."""
    if handle == 0 or dir_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var dir = _read_cstr(dir_ptr)
        var mp = alloc[MTPDrafter](1)
        mp.init_pointee_move(MTPDrafter(dir))
        engine_ptr[0].mtp_ptr = UInt64(Int(mp))
        return Int32(0)
    except e:
        print("[nomos_mtp_load EXC]", e)
        return Int32(-99)

@export
def nomos_mtp_draft(handle: Int64, seed_token: Int32, k: Int32, out_ptr: Int64) -> Int32:
    """Draft up to `k` tokens off the 31B's CURRENT post-final-norm hidden (the tap =
    engine.d_lmhead_in, populated by the last decode step), seeded by `seed_token`.
    Writes the draft token ids (Int32) to out_ptr. Returns the count drafted, or
    negative on error (-2 = MTP not loaded; call nomos_mtp_load first)."""
    if handle == 0 or out_ptr == 0 or k <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].mtp_ptr == 0:
        return Int32(-2)
    try:
        var dp = UnsafePointer[MTPDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].mtp_ptr))
        dp[0].engine_ptr = UInt64(handle)        # GemmaEngine address — the shared-KV read
        var drafts = List[Int32]()
        mtp_draft_k(engine_ptr[0].ctx, dp[0], engine_ptr[0].d_lmhead_in, Int(seed_token),
                    engine_ptr[0].d_embed_lmhead, Int(k), drafts)
        var op = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_ptr))
        var n = len(drafts) if len(drafts) < Int(k) else Int(k)
        for i in range(n):
            op[i] = drafts[i]
        return Int32(n)
    except e:
        print("[nomos_mtp_draft EXC]", e)
        return Int32(-99)

@export
def nomos_mtp_verify(handle: Int64, drafts_ptr: Int64, k: Int32, start_pos: Int32, logits_out: Int64) -> Int32:
    """Batched-verify K draft tokens: process drafts[0..k-1] at positions start_pos..start_pos+k-1
    (KV-reuse, attending the cached prefix [0,start_pos)) through the 31B as an M=k batch, writing
    per-position logits [k, VOCAB] (post-final-norm dp4a Q4 lm-head) to logits_out. Populates the KV
    at those positions; the caller rolls back rejected positions via nomos_kv_set_len after accept.
    0=ok, -1=bad arg, -99=exc."""
    if handle == 0 or drafts_ptr == 0 or logits_out == 0 or k <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var drafts = List[Int]()
        _read_int32_array_into(drafts_ptr, Int(k), drafts)
        prefill_batch_impl(engine_ptr[0], drafts, Int(start_pos), logits_out)
        return Int32(0)
    except e:
        print("[nomos_mtp_verify EXC]", e)
        return Int32(-99)


@export
def nomos_verify_fused(
    handle: Int64,
    tokens_ptr: Int64,
    n_rows: Int32,
    start_pos: Int32,
    logits_out: Int64,
) -> Int32:
    """Generic target fused verify/decode rows for LM-draft spec loops.

    tokens_ptr: Int32[n_rows]. For the E2B fused loop this is
    [c_prev, draft_0, ..., draft_{K-1}], where c_prev is the pending
    correction/bonus token from the previous cycle.
    logits_out: fp32[n_rows, VOCAB], row i = target logits after consuming
    tokens[0..i]. Host acceptance checks draft_j against argmax(row j), then
    chooses c_next from row num_acc and rolls target/draft KV to
    start_pos + 1 + num_acc. Leaves target KV at start_pos+n_rows until that
    host rollback. 0=ok, -1=bad arg, -3=too many rows, -99=exception.
    """
    if handle == 0 or tokens_ptr == 0 or logits_out == 0 or n_rows <= Int32(0):
        return Int32(-1)
    if Int(n_rows) > MAX_VERIFY_ROWS:
        return Int32(-3)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var tokens = List[Int]()
        _read_int32_array_into(tokens_ptr, Int(n_rows), tokens)
        prefill_batch_impl(engine_ptr[0], tokens, Int(start_pos), logits_out)
        return Int32(0)
    except e:
        print("[nomos_verify_fused EXC]", e)
        return Int32(-99)


@export
def nomos_verify_exact_next_token(
    handle: Int64,
    tokens_ptr: Int64,
    n_rows: Int32,
    start_pos: Int32,
    num_acc: Int32,
    out_token_ptr: Int64,
) -> Int32:
    """Lossless correction seam for fused verify loops.

    After nomos_verify_fused(tokens=[c_prev, draft_0, ...]) and host accept,
    row num_acc is the verifier row whose argmax would become c_next. Because
    fused verify is batched and not bit-identical to decode, this helper rewinds
    to just before that accepted-final token, replays exactly that one token
    through the normal decode-logits path, and writes decode-exact c_next to
    out_token_ptr. Leaves target KV at start_pos+num_acc+1.

    Valid num_acc range is [0, n_rows). 0=ok, -1=bad arg, -99=exception.
    """
    if handle == 0 or tokens_ptr == 0 or out_token_ptr == 0 or n_rows <= Int32(0):
        return Int32(-1)
    if num_acc < Int32(0) or num_acc >= n_rows:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var tokens = List[Int]()
        _read_int32_array_into(tokens_ptr, Int(n_rows), tokens)
        var final_row = Int(num_acc)
        engine_ptr[0].set_cache_len(Int(start_pos) + final_row)
        var logits = List[Float32](capacity=VOCAB)
        for _ in range(VOCAB):
            logits.append(0.0)
        engine_ptr[0].step_logits(tokens[final_row], Int64(Int(logits.unsafe_ptr())))
        var best = logits[0]
        var best_i = 0
        for i in range(1, VOCAB):
            if logits[i] > best:
                best = logits[i]
                best_i = i
        UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_token_ptr))[0] = Int32(best_i)
        return Int32(0)
    except e:
        print("[nomos_verify_exact_next_token EXC]", e,
              " start=", Int(start_pos), " num_acc=", Int(num_acc),
              " n_rows=", Int(n_rows))
        return Int32(-99)


@export
def nomos_debug_prefill_all_layers(
    handle: Int64,
    ids_ptr: Int64,
    n: Int32,
    out_ptr: Int64,
) -> Int32:
    """Capture raw residual output for every prompt row and all 60 layers.

    out_ptr is Float32[n, TOTAL_LAYERS, D], row-major. This is a correctness
    oracle surface only; it does not alter the normal prefill ABI.
    """
    if handle == 0 or ids_ptr == 0 or out_ptr == 0 or n <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    var d_rows = UInt64(0)
    try:
        var prompt = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), prompt)
        engine_ptr[0].reset_kv_cache()
        d_rows = cuda_malloc(Int(n) * TOTAL_LAYERS * D * 4)
        var layers = List[Int]()
        for layer in range(TOTAL_LAYERS):
            layers.append(layer)
        engine_ptr[0].configure_drafter_taps(UInt64(0), d_rows, layers)
        prefill_batch_impl(
            engine_ptr[0], prompt, 0,
            eagle_tap_rows=Int(n),
        )
        cuda_memcpy(UInt64(out_ptr), d_rows, Int(n) * TOTAL_LAYERS * D * 4, 2)
        engine_ptr[0].clear_drafter_taps()
        cuda_free(d_rows)
        return Int32(0)
    except e:
        engine_ptr[0].clear_drafter_taps()
        if d_rows != 0:
            cuda_free(d_rows)
        print("[nomos_debug_prefill_all_layers EXC]", e)
        return Int32(-99)


@export
def nomos_debug_final_norm(
    handle: Int64,
    ids_ptr: Int64,
    n: Int32,
    out_ptr: Int64,
) -> Int32:
    """Capture the last prompt row after the model's final RMSNorm.

    out_ptr is host Float32[D]. This deliberately runs the ordinary production
    prefill/logits path, then applies the same host RMSNorm implementation used
    by decode to the retained final residual. It is a separate fresh-state debug
    arm; callers must not use it as the source for a subsequent state comparison.
    """
    if handle == 0 or ids_ptr == 0 or out_ptr == 0 or n <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var prompt = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), prompt)
        engine_ptr[0].reset_kv_cache()

        # prefill_logits requires a full host logits sink. The values are not
        # consumed here; the call is used so this debug surface follows exactly
        # the same prefill route as production.
        var logits = List[Float32](capacity=VOCAB)
        for _ in range(VOCAB):
            logits.append(0.0)
        engine_ptr[0].prefill_logits(prompt, Int64(Int(logits.unsafe_ptr())))

        var residual = List[Float32](capacity=D)
        var normed = List[Float32](capacity=D)
        for _ in range(D):
            residual.append(0.0)
            normed.append(0.0)
        cuda_download(residual, engine_ptr[0].d_x_buf, D)
        rmsnorm(normed, residual, engine_ptr[0].final_norm, D, RMS_EPS_FINAL)

        var out = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_ptr))
        for i in range(D):
            out[i] = normed[i]
        return Int32(0)
    except e:
        print("[nomos_debug_final_norm EXC]", e)
        return Int32(-99)


@export
def nomos_debug_gdn_state(
    handle: Int64,
    recurrent_out_ptr: Int64,
    conv_out_ptr: Int64,
) -> Int32:
    """Capture Qwen3.5 GDN state as host fp32 arrays after production prefill.

    Common ABI: non-GDN engines return -2 loudly. The Qwen second engine wires
    this export to its native-bf16 recurrent/conv pools and must CAST bf16 to
    fp32; a raw byte copy into these host buffers is never valid.
    """
    if handle == 0 or recurrent_out_ptr == 0 or conv_out_ptr == 0:
        return Int32(-1)
    if not HAS_LINEAR_ATTENTION:
        return Int32(-2)
    try:
        var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
            unsafe_from_address=Int(handle)
        )
        if engine_ptr[0].gdn_state.n_gdn != 48:
            print("[nomos_debug_gdn_state] expected 48 ordered GDN slots, got",
                  engine_ptr[0].gdn_state.n_gdn)
            return Int32(-3)
        var rec_n = (
            engine_ptr[0].gdn_state.n_gdn * GDN_NUM_V_HEADS
            * GDN_KEY_HEAD_DIM * GDN_VALUE_HEAD_DIM
        )
        var conv_n = (
            engine_ptr[0].gdn_state.n_gdn * GDN_CONV_DIM
            * GDN_CONV_KERNEL
        )
        var d_rec_f32 = cuda_malloc(rec_n * 4)
        var d_conv_f32 = cuda_malloc(conv_n * 4)
        gpu_gdn_bf16_to_f32(
            engine_ptr[0].ctx, d_rec_f32,
            engine_ptr[0].gdn_state.rec_base, rec_n,
        )
        gpu_gdn_bf16_to_f32(
            engine_ptr[0].ctx, d_conv_f32,
            engine_ptr[0].gdn_state.conv_base, conv_n,
        )
        engine_ptr[0].ctx.synchronize()
        cuda_memcpy(UInt64(recurrent_out_ptr), d_rec_f32, rec_n * 4, 2)
        cuda_memcpy(UInt64(conv_out_ptr), d_conv_f32, conv_n * 4, 2)
        cuda_free(d_rec_f32); cuda_free(d_conv_f32)
        return Int32(0)
    except e:
        print("[nomos_debug_gdn_state EXC]", e)
        return Int32(-99)


@export
def nomos_debug_kv_block32_q8_ab(
    handle: Int64,
    layer_i: Int32,
    out_i32: Int64,
    out_f32: Int64,
) -> Int32:
    """Compare verify's append-with-Q8 view with decode's cache-dequant Q8 view.

    Uses one deterministic synthetic KV row at position zero.  Required outputs:
      out_i32[0:8] = hd, nkv, K byte diffs, V byte diffs,
                     first K diff, first V diff, scale blocks, compared bytes.
      out_f32[0:4] = verify/decode K scale, verify/decode V scale for head zero.
    The live engine cache is not modified.
    """
    if handle == 0 or out_i32 == 0 or out_f32 == 0:
        return Int32(-1)
    if layer_i < Int32(0) or layer_i >= Int32(TOTAL_LAYERS):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )
    if not engine_ptr[0].kv_int4_block32:
        return Int32(-2)
    try:
        var layer = Int(layer_i)
        var nkv = engine_ptr[0].layer_nkv[layer]
        var hd = engine_ptr[0].layer_hd[layer]
        var kvd = engine_ptr[0].layer_kvd[layer]
        var nblocks = (hd + 31) // 32
        var packed_bytes = nkv * (hd // 2)
        var q8_bytes = nkv * hd

        var h_k = List[Float32](capacity=kvd)
        var h_v = List[Float32](capacity=kvd)
        for i in range(kvd):
            var raw = ((i * 37 + layer * 17) % 257) - 128
            var x = Float32(raw) / Float32(17.0)
            h_k.append(x)
            h_v.append(x * Float32(0.73) + Float32(0.19))

        var d_src_k = cuda_malloc(kvd * 4)
        var d_src_v = cuda_malloc(kvd * 4)
        var d_k = cuda_malloc(packed_bytes)
        var d_v = cuda_malloc(packed_bytes)
        var d_ks = cuda_malloc(nkv * nblocks * 2 + nkv * 4)
        var d_vs = cuda_malloc(nkv * nblocks * 2 + nkv * 4)
        var d_kq8 = cuda_malloc(q8_bytes)
        var d_vq8 = cuda_malloc(q8_bytes)
        var d_kqs = cuda_malloc(nkv * 4)
        var d_vqs = cuda_malloc(nkv * 4)
        cuda_upload(d_src_k, h_k)
        cuda_upload(d_src_v, h_v)
        cuda_sync()

        gpu_append_quant_kv_i4_with_q8(
            engine_ptr[0].ctx,
            d_src_k, d_src_v, d_k, d_v, d_ks, d_vs,
            d_kq8, d_vq8, d_kqs, d_vqs,
            1, nkv, hd, kvd, 1, hd, 0, 1, nblocks,
        )
        engine_ptr[0].ctx.synchronize()

        var verify_k = List[Int8](capacity=q8_bytes)
        var verify_v = List[Int8](capacity=q8_bytes)
        for _ in range(q8_bytes):
            verify_k.append(0)
            verify_v.append(0)
        var scales = List[Float32](capacity=4)
        for _ in range(4):
            scales.append(0.0)
        cuda_memcpy(UInt64(Int(verify_k.unsafe_ptr())), d_kq8, q8_bytes, 2)
        cuda_memcpy(UInt64(Int(verify_v.unsafe_ptr())), d_vq8, q8_bytes, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())), d_kqs, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(8), d_vqs, 4, 2)

        gpu_dequant_kv_i4_to_q8_layer(
            engine_ptr[0].ctx, d_k, d_ks, d_kq8, d_kqs,
            nkv, 1, hd, 1, 1, 0, nblocks,
        )
        gpu_dequant_kv_i4_to_q8_layer(
            engine_ptr[0].ctx, d_v, d_vs, d_vq8, d_vqs,
            nkv, 1, hd, 1, 1, 0, nblocks,
        )
        engine_ptr[0].ctx.synchronize()

        var decode_k = List[Int8](capacity=q8_bytes)
        var decode_v = List[Int8](capacity=q8_bytes)
        for _ in range(q8_bytes):
            decode_k.append(0)
            decode_v.append(0)
        cuda_memcpy(UInt64(Int(decode_k.unsafe_ptr())), d_kq8, q8_bytes, 2)
        cuda_memcpy(UInt64(Int(decode_v.unsafe_ptr())), d_vq8, q8_bytes, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(4), d_kqs, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(12), d_vqs, 4, 2)

        var kd = 0
        var vd = 0
        var fk = -1
        var fv = -1
        for i in range(q8_bytes):
            if verify_k[i] != decode_k[i]:
                if fk < 0:
                    fk = i
                kd += 1
            if verify_v[i] != decode_v[i]:
                if fv < 0:
                    fv = i
                vd += 1

        var oi = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_i32))
        oi[0] = Int32(hd)
        oi[1] = Int32(nkv)
        oi[2] = Int32(kd)
        oi[3] = Int32(vd)
        oi[4] = Int32(fk)
        oi[5] = Int32(fv)
        oi[6] = Int32(nblocks)
        oi[7] = Int32(q8_bytes)
        var of = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_f32))
        for i in range(4):
            of[i] = scales[i]

        cuda_free(d_src_k)
        cuda_free(d_src_v)
        cuda_free(d_k)
        cuda_free(d_v)
        cuda_free(d_ks)
        cuda_free(d_vs)
        cuda_free(d_kq8)
        cuda_free(d_vq8)
        cuda_free(d_kqs)
        cuda_free(d_vqs)
        return Int32(0)
    except e:
        print("[nomos_debug_kv_block32_q8_ab EXC]", e)
        return Int32(-99)


@export
def nomos_debug_kv_append_ab(
    handle: Int64,
    layer_i: Int32,
    base_pos: Int32,
    n_rows: Int32,
    row_i: Int32,
    head_i: Int32,
    out_i32: Int64,
    out_f32: Int64,
    out_batch_k: Int64,
    out_decode_k: Int64,
    out_batch_v: Int64,
    out_decode_v: Int64,
) -> Int32:
    """Append-codec A/B probe for INT4 KV.

    Runs the live INT4 append kernel twice into temporary caches:
    - batched path: n_s=n_rows, base_pos=base_pos, then inspect row_i/head_i;
    - decode path: n_s=1 using the exact same source row, base_pos+row_i.

    This isolates the append quantize/scatter codec from the full verify forward.
    It does not modify the engine's live KV cache.

    Required:
      out_i32: Int32[16]
        [0]=layer, [1]=base_pos, [2]=n_rows, [3]=row_i, [4]=head_i,
        [5]=nkv, [6]=hd, [7]=cache_cap, [8]=packed_bytes_per_row,
        [9]=first_k_diff_or_-1, [10]=k_diff_count,
        [11]=first_v_diff_or_-1, [12]=v_diff_count,
        [13]=k_scale_equal, [14]=v_scale_equal, [15]=selected_ring_slot.
    Optional:
      out_f32: Float32[4] = batch_k_scale, decode_k_scale, batch_v_scale, decode_v_scale.
      out_* byte buffers: UInt8[packed_bytes_per_row].
    """
    if handle == 0 or out_i32 == 0:
        return Int32(-1)
    if layer_i < Int32(0) or layer_i >= Int32(TOTAL_LAYERS):
        return Int32(-1)
    if n_rows <= Int32(1) or n_rows > Int32(MAX_VERIFY_ROWS):
        return Int32(-1)
    if row_i < Int32(0) or row_i >= n_rows or head_i < Int32(0):
        return Int32(-1)
    if base_pos < Int32(0):
        return Int32(-1)

    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )
    if not engine_ptr[0].kv_quant or not engine_ptr[0].kv_int4:
        return Int32(-2)

    try:
        var layer = Int(layer_i)
        var rows = Int(n_rows)
        var row = Int(row_i)
        var head = Int(head_i)
        var nkv = engine_ptr[0].layer_nkv[layer]
        var hd = engine_ptr[0].layer_hd[layer]
        var kvd = engine_ptr[0].layer_kvd[layer]
        var ccap = engine_ptr[0].layer_cache_cap[layer]
        if head >= nkv:
            return Int32(-1)
        if hd <= 0 or ccap <= 0:
            return Int32(-1)

        var hcache = hd // 2
        var total_src = rows * kvd
        var h_k = List[Float32](capacity=total_src)
        var h_v = List[Float32](capacity=total_src)

        # Deliberately vary row magnitudes. If any append path accidentally derives
        # quant stats across S rows, row_i will differ from the n_s=1 append.
        for idx in range(total_src):
            var s = idx // kvd
            var j = idx - s * kvd
            var raw = ((j * 37 + s * 101 + layer * 17 + Int(base_pos) * 13) % 257) - 128
            var row_gain = Float32((s + 1) * (s + 1))
            var x = (Float32(raw) / Float32(17.0)) * row_gain
            h_k.append(x)
            h_v.append((x * Float32(0.73)) + Float32(s + 1) * Float32(0.19))

        var src_bytes = total_src * 4
        var d_src_k = cuda_malloc(src_bytes)
        var d_src_v = cuda_malloc(src_bytes)
        cuda_upload(d_src_k, h_k)
        cuda_upload(d_src_v, h_v)
        cuda_sync()

        var cache_bytes = nkv * ccap * hcache
        var scale_bytes = nkv * ccap * 4
        var d_bk = cuda_malloc(cache_bytes)
        var d_bv = cuda_malloc(cache_bytes)
        var d_bs_k = cuda_malloc(scale_bytes)
        var d_bs_v = cuda_malloc(scale_bytes)
        var d_dk = cuda_malloc(cache_bytes)
        var d_dv = cuda_malloc(cache_bytes)
        var d_ds_k = cuda_malloc(scale_bytes)
        var d_ds_v = cuda_malloc(scale_bytes)

        gpu_append_quant_kv_i4(
            engine_ptr[0].ctx, d_src_k, d_src_v,
            d_bk, d_bv, d_bs_k, d_bs_v,
            rows, nkv, hd, kvd, ccap, hd, Int(base_pos),
        )
        var row_byte_off = UInt64(row * kvd * 4)
        gpu_append_quant_kv_i4(
            engine_ptr[0].ctx, d_src_k + row_byte_off, d_src_v + row_byte_off,
            d_dk, d_dv, d_ds_k, d_ds_v,
            1, nkv, hd, kvd, ccap, hd, Int(base_pos) + row,
        )
        cuda_sync()

        var ring = (Int(base_pos) + row) % ccap
        var cache_row = head * ccap + ring
        var byte_off = UInt64(cache_row * hcache)
        var scale_off = UInt64(cache_row * 4)

        var batch_k = List[UInt8](capacity=hcache)
        var decode_k = List[UInt8](capacity=hcache)
        var batch_v = List[UInt8](capacity=hcache)
        var decode_v = List[UInt8](capacity=hcache)
        for _ in range(hcache):
            batch_k.append(0)
            decode_k.append(0)
            batch_v.append(0)
            decode_v.append(0)
        cuda_memcpy(UInt64(Int(batch_k.unsafe_ptr())), d_bk + byte_off, hcache, 2)
        cuda_memcpy(UInt64(Int(decode_k.unsafe_ptr())), d_dk + byte_off, hcache, 2)
        cuda_memcpy(UInt64(Int(batch_v.unsafe_ptr())), d_bv + byte_off, hcache, 2)
        cuda_memcpy(UInt64(Int(decode_v.unsafe_ptr())), d_dv + byte_off, hcache, 2)

        var scales = List[Float32](capacity=4)
        for _ in range(4):
            scales.append(0.0)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())), d_bs_k + scale_off, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(4), d_ds_k + scale_off, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(8), d_bs_v + scale_off, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(12), d_ds_v + scale_off, 4, 2)

        if out_batch_k != 0:
            cuda_memcpy(UInt64(out_batch_k), d_bk + byte_off, hcache, 2)
        if out_decode_k != 0:
            cuda_memcpy(UInt64(out_decode_k), d_dk + byte_off, hcache, 2)
        if out_batch_v != 0:
            cuda_memcpy(UInt64(out_batch_v), d_bv + byte_off, hcache, 2)
        if out_decode_v != 0:
            cuda_memcpy(UInt64(out_decode_v), d_dv + byte_off, hcache, 2)
        if out_f32 != 0:
            var fp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_f32))
            for i in range(4):
                fp[i] = scales[i]

        var first_k = -1
        var kdiff = 0
        var first_v = -1
        var vdiff = 0
        for i in range(hcache):
            if batch_k[i] != decode_k[i]:
                if first_k < 0:
                    first_k = i
                kdiff += 1
            if batch_v[i] != decode_v[i]:
                if first_v < 0:
                    first_v = i
                vdiff += 1
        var k_scale_equal = 1 if scales[0] == scales[1] else 0
        var v_scale_equal = 1 if scales[2] == scales[3] else 0

        var outp = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_i32))
        outp[0] = Int32(layer)
        outp[1] = Int32(Int(base_pos))
        outp[2] = Int32(rows)
        outp[3] = Int32(row)
        outp[4] = Int32(head)
        outp[5] = Int32(nkv)
        outp[6] = Int32(hd)
        outp[7] = Int32(ccap)
        outp[8] = Int32(hcache)
        outp[9] = Int32(first_k)
        outp[10] = Int32(kdiff)
        outp[11] = Int32(first_v)
        outp[12] = Int32(vdiff)
        outp[13] = Int32(k_scale_equal)
        outp[14] = Int32(v_scale_equal)
        outp[15] = Int32(ring)

        cuda_free(d_src_k)
        cuda_free(d_src_v)
        cuda_free(d_bk)
        cuda_free(d_bv)
        cuda_free(d_bs_k)
        cuda_free(d_bs_v)
        cuda_free(d_dk)
        cuda_free(d_dv)
        cuda_free(d_ds_k)
        cuda_free(d_ds_v)
        return Int32(0)
    except e:
        print("[nomos_debug_kv_append_ab EXC]", e)
        return Int32(-99)


@export
def nomos_debug_kv_append_overwrite_ab(
    handle: Int64,
    layer_i: Int32,
    base_pos: Int32,
    n_rows: Int32,
    row_i: Int32,
    head_i: Int32,
    out_i32: Int64,
    out_f32: Int64,
    out_batch_k: Int64,
    out_decode_k: Int64,
    out_batch_v: Int64,
    out_decode_v: Int64,
) -> Int32:
    """Overwrite-sensitive INT4 append A/B probe.

    Same final comparison as nomos_debug_kv_append_ab, but first populates the
    temporary caches at the same positions with different garbage values:
    - batch temp cache: one batched garbage append over n_rows;
    - decode temp cache: n_rows single-row garbage appends.
    Then it overwrites with the final A/B rows and compares the selected row.

    Required out_i32 is Int32[20]:
      [0..15] match nomos_debug_kv_append_ab.
      [16]=prepopulated(1), [17]=decode_prepop_was_per_row(1),
      [18]=n_rows, [19]=reserved.
    Optional buffers match nomos_debug_kv_append_ab.
    """
    if handle == 0 or out_i32 == 0:
        return Int32(-1)
    if layer_i < Int32(0) or layer_i >= Int32(TOTAL_LAYERS):
        return Int32(-1)
    if n_rows <= Int32(1) or n_rows > Int32(MAX_VERIFY_ROWS):
        return Int32(-1)
    if row_i < Int32(0) or row_i >= n_rows or head_i < Int32(0):
        return Int32(-1)
    if base_pos < Int32(0):
        return Int32(-1)

    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )
    if not engine_ptr[0].kv_quant or not engine_ptr[0].kv_int4:
        return Int32(-2)

    try:
        var layer = Int(layer_i)
        var rows = Int(n_rows)
        var row = Int(row_i)
        var head = Int(head_i)
        var nkv = engine_ptr[0].layer_nkv[layer]
        var hd = engine_ptr[0].layer_hd[layer]
        var kvd = engine_ptr[0].layer_kvd[layer]
        var ccap = engine_ptr[0].layer_cache_cap[layer]
        if head >= nkv:
            return Int32(-1)
        if hd <= 0 or ccap <= 0:
            return Int32(-1)

        var hcache = hd // 2
        var total_src = rows * kvd
        var final_k = List[Float32](capacity=total_src)
        var final_v = List[Float32](capacity=total_src)
        var pre_k = List[Float32](capacity=total_src)
        var pre_v = List[Float32](capacity=total_src)

        # Final rows match the fresh-cache probe. Prepopulate rows use a
        # different high-magnitude pattern to expose any resident-state read.
        for idx in range(total_src):
            var s = idx // kvd
            var j = idx - s * kvd
            var raw = ((j * 37 + s * 101 + layer * 17 + Int(base_pos) * 13) % 257) - 128
            var row_gain = Float32((s + 1) * (s + 1))
            var x = (Float32(raw) / Float32(17.0)) * row_gain
            final_k.append(x)
            final_v.append((x * Float32(0.73)) + Float32(s + 1) * Float32(0.19))

            var praw = ((j * 53 + s * 197 + layer * 29 + Int(base_pos) * 31 + 911) % 509) - 254
            var pgain = Float32((rows - s + 2) * 5)
            var px = (Float32(praw) / Float32(9.0)) * pgain
            pre_k.append(px)
            pre_v.append((px * Float32(-0.41)) + Float32(s + 3) * Float32(1.37))

        var src_bytes = total_src * 4
        var d_final_k = cuda_malloc(src_bytes)
        var d_final_v = cuda_malloc(src_bytes)
        var d_pre_k = cuda_malloc(src_bytes)
        var d_pre_v = cuda_malloc(src_bytes)
        cuda_upload(d_final_k, final_k)
        cuda_upload(d_final_v, final_v)
        cuda_upload(d_pre_k, pre_k)
        cuda_upload(d_pre_v, pre_v)
        cuda_sync()

        var cache_bytes = nkv * ccap * hcache
        var scale_bytes = nkv * ccap * 4
        var d_bk = cuda_malloc(cache_bytes)
        var d_bv = cuda_malloc(cache_bytes)
        var d_bs_k = cuda_malloc(scale_bytes)
        var d_bs_v = cuda_malloc(scale_bytes)
        var d_dk = cuda_malloc(cache_bytes)
        var d_dv = cuda_malloc(cache_bytes)
        var d_ds_k = cuda_malloc(scale_bytes)
        var d_ds_v = cuda_malloc(scale_bytes)

        # Dirty both temp caches at the exact target positions before the final
        # overwrite. The decode temp is dirtied through n_s=1 launches so the
        # previous writer shape differs from the batch temp, matching the live
        # batch-vs-decode contrast.
        gpu_append_quant_kv_i4(
            engine_ptr[0].ctx, d_pre_k, d_pre_v,
            d_bk, d_bv, d_bs_k, d_bs_v,
            rows, nkv, hd, kvd, ccap, hd, Int(base_pos),
        )
        for s in range(rows):
            var s_off = UInt64(s * kvd * 4)
            gpu_append_quant_kv_i4(
                engine_ptr[0].ctx, d_pre_k + s_off, d_pre_v + s_off,
                d_dk, d_dv, d_ds_k, d_ds_v,
                1, nkv, hd, kvd, ccap, hd, Int(base_pos) + s,
            )
        cuda_sync()

        gpu_append_quant_kv_i4(
            engine_ptr[0].ctx, d_final_k, d_final_v,
            d_bk, d_bv, d_bs_k, d_bs_v,
            rows, nkv, hd, kvd, ccap, hd, Int(base_pos),
        )
        var row_byte_off = UInt64(row * kvd * 4)
        gpu_append_quant_kv_i4(
            engine_ptr[0].ctx, d_final_k + row_byte_off, d_final_v + row_byte_off,
            d_dk, d_dv, d_ds_k, d_ds_v,
            1, nkv, hd, kvd, ccap, hd, Int(base_pos) + row,
        )
        cuda_sync()

        var ring = (Int(base_pos) + row) % ccap
        var cache_row = head * ccap + ring
        var byte_off = UInt64(cache_row * hcache)
        var scale_off = UInt64(cache_row * 4)

        var batch_k = List[UInt8](capacity=hcache)
        var decode_k = List[UInt8](capacity=hcache)
        var batch_v = List[UInt8](capacity=hcache)
        var decode_v = List[UInt8](capacity=hcache)
        for _ in range(hcache):
            batch_k.append(0)
            decode_k.append(0)
            batch_v.append(0)
            decode_v.append(0)
        cuda_memcpy(UInt64(Int(batch_k.unsafe_ptr())), d_bk + byte_off, hcache, 2)
        cuda_memcpy(UInt64(Int(decode_k.unsafe_ptr())), d_dk + byte_off, hcache, 2)
        cuda_memcpy(UInt64(Int(batch_v.unsafe_ptr())), d_bv + byte_off, hcache, 2)
        cuda_memcpy(UInt64(Int(decode_v.unsafe_ptr())), d_dv + byte_off, hcache, 2)

        var scales = List[Float32](capacity=4)
        for _ in range(4):
            scales.append(0.0)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())), d_bs_k + scale_off, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(4), d_ds_k + scale_off, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(8), d_bs_v + scale_off, 4, 2)
        cuda_memcpy(UInt64(Int(scales.unsafe_ptr())) + UInt64(12), d_ds_v + scale_off, 4, 2)

        if out_batch_k != 0:
            cuda_memcpy(UInt64(out_batch_k), d_bk + byte_off, hcache, 2)
        if out_decode_k != 0:
            cuda_memcpy(UInt64(out_decode_k), d_dk + byte_off, hcache, 2)
        if out_batch_v != 0:
            cuda_memcpy(UInt64(out_batch_v), d_bv + byte_off, hcache, 2)
        if out_decode_v != 0:
            cuda_memcpy(UInt64(out_decode_v), d_dv + byte_off, hcache, 2)
        if out_f32 != 0:
            var fp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_f32))
            for i in range(4):
                fp[i] = scales[i]

        var first_k = -1
        var kdiff = 0
        var first_v = -1
        var vdiff = 0
        for i in range(hcache):
            if batch_k[i] != decode_k[i]:
                if first_k < 0:
                    first_k = i
                kdiff += 1
            if batch_v[i] != decode_v[i]:
                if first_v < 0:
                    first_v = i
                vdiff += 1
        var k_scale_equal = 1 if scales[0] == scales[1] else 0
        var v_scale_equal = 1 if scales[2] == scales[3] else 0

        var outp = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_i32))
        outp[0] = Int32(layer)
        outp[1] = Int32(Int(base_pos))
        outp[2] = Int32(rows)
        outp[3] = Int32(row)
        outp[4] = Int32(head)
        outp[5] = Int32(nkv)
        outp[6] = Int32(hd)
        outp[7] = Int32(ccap)
        outp[8] = Int32(hcache)
        outp[9] = Int32(first_k)
        outp[10] = Int32(kdiff)
        outp[11] = Int32(first_v)
        outp[12] = Int32(vdiff)
        outp[13] = Int32(k_scale_equal)
        outp[14] = Int32(v_scale_equal)
        outp[15] = Int32(ring)
        outp[16] = Int32(1)
        outp[17] = Int32(1)
        outp[18] = Int32(rows)
        outp[19] = Int32(0)

        cuda_free(d_final_k)
        cuda_free(d_final_v)
        cuda_free(d_pre_k)
        cuda_free(d_pre_v)
        cuda_free(d_bk)
        cuda_free(d_bv)
        cuda_free(d_bs_k)
        cuda_free(d_bs_v)
        cuda_free(d_dk)
        cuda_free(d_dv)
        cuda_free(d_ds_k)
        cuda_free(d_ds_v)
        return Int32(0)
    except e:
        print("[nomos_debug_kv_append_overwrite_ab EXC]", e)
        return Int32(-99)


@export
def nomos_debug_qkv_source_ab(
    handle: Int64,
    tokens_ptr: Int64,
    n_rows: Int32,
    start_pos: Int32,
    row_i: Int32,
    layer_i: Int32,
    head_i: Int32,
    out_i32: Int64,
    out_f32: Int64,
    out_decode_k: Int64,
    out_batch_k: Int64,
    out_decode_v: Int64,
    out_batch_v: Int64,
    out_decode_normed: Int64,
    out_batch_normed: Int64,
) -> Int32:
    """Decode-vs-batched QKV source probe.

    Compares the Float32 rows produced immediately after QKV prep and before KV
    append:
      batched path: run prefill_batch_impl(tokens, start_pos) to layer_i and
        dump prepare_qkv_batched row_i.
      decode path: consume tokens[0..row_i-1] sequentially, then dump
        prepare_qkv_dev(tokens[row_i]) at layer_i.

    The batched leg runs FIRST so aged inter-call scratch is sampled before any
    decode replay can overwrite it. This localizes strict fused-loop drift
    upstream of the append codec. It rewinds cache_lens to start_pos after both
    sub-runs.

    Required:
      out_i32: Int32[20]
        [0]=layer [1]=start_pos [2]=n_rows [3]=row_i [4]=head_i [5]=token
        [6]=nkv [7]=hd [8]=kvd
        [9]=first_k_elem_diff_or_-1 [10]=k_elem_diff_count
        [11]=first_v_elem_diff_or_-1 [12]=v_elem_diff_count
        [13]=first_norm_elem_diff_or_-1 [14]=norm_elem_diff_count
        [15]=orig_cache_len [16]=final_cache_len [17..19]=reserved.
    Optional:
      out_f32: Float32[12]
        [0]=k_max_abs_delta [1]=v_max_abs_delta [2]=norm_max_abs_delta
        [3]=k_cos [4]=v_cos [5]=norm_cos
        [6]=decode_k_l2 [7]=batch_k_l2 [8]=decode_v_l2 [9]=batch_v_l2
        [10]=decode_norm_l2 [11]=batch_norm_l2.
      out_* K/V buffers are Float32[hd] for the selected KV head.
      out_*_normed buffers are Float32[D].
    """
    if handle == 0 or tokens_ptr == 0 or out_i32 == 0:
        return Int32(-1)
    if n_rows <= Int32(0) or n_rows > Int32(MAX_VERIFY_ROWS):
        return Int32(-1)
    if row_i < Int32(0) or row_i >= n_rows:
        return Int32(-1)
    if layer_i < Int32(0) or layer_i >= Int32(TOTAL_LAYERS):
        return Int32(-1)
    if head_i < Int32(0) or start_pos < Int32(0):
        return Int32(-1)

    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )

    try:
        var rows = Int(n_rows)
        var row = Int(row_i)
        var layer = Int(layer_i)
        var head = Int(head_i)
        var tokens = List[Int]()
        _read_int32_array_into(tokens_ptr, rows, tokens)

        var nkv = engine_ptr[0].layer_nkv[layer]
        var hd = engine_ptr[0].layer_hd[layer]
        var kvd = engine_ptr[0].layer_kvd[layer]
        if head >= nkv or hd <= 0 or kvd <= 0:
            return Int32(-1)

        var dec_norm = List[Float32](capacity=D)
        var bat_norm = List[Float32](capacity=D)
        var dec_k = List[Float32](capacity=kvd)
        var bat_k = List[Float32](capacity=kvd)
        var dec_v = List[Float32](capacity=kvd)
        var bat_v = List[Float32](capacity=kvd)
        for _ in range(D):
            dec_norm.append(0.0)
            bat_norm.append(0.0)
        for _ in range(kvd):
            dec_k.append(0.0)
            bat_k.append(0.0)
            dec_v.append(0.0)
            bat_v.append(0.0)

        var orig_len = engine_ptr[0].cache_len()
        # The batched leg mutates Qwen's recurrent/conv pools.  Rewinding only
        # cache_len makes the subsequent decode leg start from the wrong state,
        # so snapshot the full non-appendable state at the shared prefix.
        @parameter
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.snapshot_for_verify()
        engine_ptr[0].set_cache_len(Int(start_pos))
        prefill_batch_impl(
            engine_ptr[0],
            tokens,
            Int(start_pos),
            Int64(0),
            -1,
            0,
            layer,
            row,
            Int64(Int(bat_norm.unsafe_ptr())),
            Int64(Int(bat_k.unsafe_ptr())),
            Int64(Int(bat_v.unsafe_ptr())),
        )
        engine_ptr[0].set_cache_len(Int(start_pos))
        @parameter
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.restore_verify_snapshot()

        if row > 0:
            var tmp_logits = List[Float32](capacity=VOCAB)
            for _ in range(VOCAB):
                tmp_logits.append(0.0)
            var replay_start = 0
            if Int(start_pos) == 0:
                # step_logits is continuation-only and deliberately drops a leading
                # BOS. Seed a fresh sequential replay through prefill_logits instead;
                # otherwise the debug leg silently omits token 0 on real prompts.
                var seed = List[Int]()
                seed.append(tokens[0])
                engine_ptr[0].prefill_logits(seed, Int64(Int(tmp_logits.unsafe_ptr())))
                replay_start = 1
            for i in range(replay_start, row):
                engine_ptr[0].step_logits(tokens[i], Int64(Int(tmp_logits.unsafe_ptr())))

        engine_ptr[0].debug_decode_qkv_source(
            tokens[row],
            layer,
            Int64(Int(dec_norm.unsafe_ptr())),
            Int64(Int(dec_k.unsafe_ptr())),
            Int64(Int(dec_v.unsafe_ptr())),
        )
        engine_ptr[0].set_cache_len(Int(start_pos))
        @parameter
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.restore_verify_snapshot()
            engine_ptr[0].gdn_state.finish_verify_transaction()

        var k_first = -1
        var k_count = 0
        var v_first = -1
        var v_count = 0
        var n_first = -1
        var n_count = 0
        var k_max = Float64(0.0)
        var v_max = Float64(0.0)
        var n_max = Float64(0.0)
        var k_dot = Float64(0.0)
        var k_da = Float64(0.0)
        var k_ba = Float64(0.0)
        var v_dot = Float64(0.0)
        var v_da = Float64(0.0)
        var v_ba = Float64(0.0)
        var n_dot = Float64(0.0)
        var n_da = Float64(0.0)
        var n_ba = Float64(0.0)

        var ko = head * hd
        var dec_k_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(dec_k.unsafe_ptr())
        )
        var bat_k_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(bat_k.unsafe_ptr())
        )
        var dec_v_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(dec_v.unsafe_ptr())
        )
        var bat_v_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(bat_v.unsafe_ptr())
        )
        for i in range(hd):
            var idx = ko + i
            var kd = Float64(dec_k[idx])
            var kb = Float64(bat_k[idx])
            var vd = Float64(dec_v[idx])
            var vb = Float64(bat_v[idx])
            var dk = abs(kd - kb)
            var dv = abs(vd - vb)
            if dk > k_max:
                k_max = dk
            if dv > v_max:
                v_max = dv
            k_dot += kd * kb
            k_da += kd * kd
            k_ba += kb * kb
            v_dot += vd * vb
            v_da += vd * vd
            v_ba += vb * vb

            var k_diff = False
            var v_diff = False
            for b in range(4):
                if dec_k_bytes[(idx * 4) + b] != bat_k_bytes[(idx * 4) + b]:
                    k_diff = True
                if dec_v_bytes[(idx * 4) + b] != bat_v_bytes[(idx * 4) + b]:
                    v_diff = True
            if k_diff:
                if k_first < 0:
                    k_first = i
                k_count += 1
            if v_diff:
                if v_first < 0:
                    v_first = i
                v_count += 1

        var dec_n_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(dec_norm.unsafe_ptr())
        )
        var bat_n_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(bat_norm.unsafe_ptr())
        )
        for i in range(D):
            var nd = Float64(dec_norm[i])
            var nb = Float64(bat_norm[i])
            var dn = abs(nd - nb)
            if dn > n_max:
                n_max = dn
            n_dot += nd * nb
            n_da += nd * nd
            n_ba += nb * nb
            var n_diff = False
            for b in range(4):
                if dec_n_bytes[(i * 4) + b] != bat_n_bytes[(i * 4) + b]:
                    n_diff = True
            if n_diff:
                if n_first < 0:
                    n_first = i
                n_count += 1

        if out_decode_k != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_decode_k))
            for i in range(hd):
                p[i] = dec_k[ko + i]
        if out_batch_k != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_batch_k))
            for i in range(hd):
                p[i] = bat_k[ko + i]
        if out_decode_v != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_decode_v))
            for i in range(hd):
                p[i] = dec_v[ko + i]
        if out_batch_v != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_batch_v))
            for i in range(hd):
                p[i] = bat_v[ko + i]
        if out_decode_normed != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_decode_normed))
            for i in range(D):
                p[i] = dec_norm[i]
        if out_batch_normed != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_batch_normed))
            for i in range(D):
                p[i] = bat_norm[i]

        var outp = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_i32))
        outp[0] = Int32(layer)
        outp[1] = Int32(Int(start_pos))
        outp[2] = Int32(rows)
        outp[3] = Int32(row)
        outp[4] = Int32(head)
        outp[5] = Int32(tokens[row])
        outp[6] = Int32(nkv)
        outp[7] = Int32(hd)
        outp[8] = Int32(kvd)
        outp[9] = Int32(k_first)
        outp[10] = Int32(k_count)
        outp[11] = Int32(v_first)
        outp[12] = Int32(v_count)
        outp[13] = Int32(n_first)
        outp[14] = Int32(n_count)
        outp[15] = Int32(orig_len)
        outp[16] = Int32(engine_ptr[0].cache_len())
        outp[17] = Int32(0)
        outp[18] = Int32(0)
        outp[19] = Int32(0)

        if out_f32 != 0:
            var fp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_f32))
            fp[0] = Float32(k_max)
            fp[1] = Float32(v_max)
            fp[2] = Float32(n_max)
            fp[3] = Float32(k_dot / sqrt(k_da * k_ba)) if k_da > 0.0 and k_ba > 0.0 else Float32(0.0)
            fp[4] = Float32(v_dot / sqrt(v_da * v_ba)) if v_da > 0.0 and v_ba > 0.0 else Float32(0.0)
            fp[5] = Float32(n_dot / sqrt(n_da * n_ba)) if n_da > 0.0 and n_ba > 0.0 else Float32(0.0)
            fp[6] = Float32(sqrt(k_da))
            fp[7] = Float32(sqrt(k_ba))
            fp[8] = Float32(sqrt(v_da))
            fp[9] = Float32(sqrt(v_ba))
            fp[10] = Float32(sqrt(n_da))
            fp[11] = Float32(sqrt(n_ba))
        return Int32(0)
    except e:
        engine_ptr[0].set_cache_len(Int(start_pos))
        @parameter
        if HAS_LINEAR_ATTENTION:
            if engine_ptr[0].gdn_state.snapshot_valid:
                try:
                    engine_ptr[0].gdn_state.restore_verify_snapshot()
                except restore_e:
                    print("[nomos_debug_qkv_source_ab restore EXC]", restore_e)
                engine_ptr[0].gdn_state.finish_verify_transaction()
        print("[nomos_debug_qkv_source_ab EXC]", e,
              " start=", Int(start_pos), " row=", Int(row_i),
              " n_rows=", Int(n_rows), " layer=", Int(layer_i))
        return Int32(-99)


@export
def nomos_debug_omlp_stage_ab(
    handle: Int64,
    tokens_ptr: Int64,
    n_rows: Int32,
    start_pos: Int32,
    row_i: Int32,
    layer_i: Int32,
    stage_i: Int32,
    out_i32: Int64,
    out_f32: Int64,
    out_decode: Int64,
    out_batch: Int64,
) -> Int32:
    """Decode-vs-batched O/MLP stage probe for one layer and row.

    stage_i:
      0 attn_out, 1 o_proj, 2 post_attn_norm, 3 x_post_attn,
      4 pre_ff_norm, 5 down_proj, 6 post_ff_norm, 7 layer_out,
      8 post-RoPE Q, 9 q8-Q-as-f32 plus q-scale tail,
      10 K/V q8 views plus K/V scale tails as consumed by attention,
      11 raw packed INT4 K/V source bytes plus source scales,
      12 pre-softmax QK scores [nh,klen],
      13 GDN recurrent core [6144], 14 GDN gated-norm output [6144],
      15 GDN out-projection [D], 16 GDN post-attention residual [D],
      17 GDN post-MLP residual [D], 18 GDN input norm [D],
      19 GDN raw QKV projection [10240], 20 GDN raw Z projection [6144],
      21/22 GDN raw A/B projections [48], 23/24 prepared decay/beta [48],
      25 GDN post-convolution QKV [10240]. Stages 13..25 compare the
      batched GDN formulation directly with the production S=1 decode route.
      In particular, a nonzero stage 25 characterizes the raw batched-prefill
      formulation; it is not a verdict on NOMOS_VERIFY_GDN_FAST_EXACT.  Gate
      that production verify route with GDN state parity and token identity.
      26 post-Q-norm Q before RoPE [layer_qd].

    out_i32: Int32[16]
      [0]=layer [1]=start_pos [2]=n_rows [3]=row [4]=stage [5]=token
      [6]=dim [7]=first_diff_or_-1 [8]=diff_count [9]=orig_cache_len
      [10]=final_cache_len. For stage 9: [11]=l_hd [12]=nqb [13]=lo
      [14]=klen [15]=scale_off. For stage 10: [11]=qk_grid [12]=qk_block
      [13]=pv_grid [14]=pv_block [15]=dynamic_smem_bytes. For stage 11:
      [11]=l_hd [12]=packed_bytes_per_key [13]=source_start [14]=klen
      [15]=cache_cap. For stage 12: [11]=nh [12]=klen [13]=qk_grid
      [14]=qk_block [15]=dynamic_smem_bytes.
    out_f32: Float32[4] = [max_abs_delta, cos, decode_l2, batch_l2].
    Optional out_decode/out_batch are Float32[dim].
    """
    if handle == 0 or tokens_ptr == 0 or out_i32 == 0:
        return Int32(-1)
    # This debug A/B also serves as the real-prompt GDN stage dumper. Production
    # verify remains capped at MAX_VERIFY_ROWS; the probe may inspect a longer
    # prompt because it does not write the fixed verify-logits workspace.
    if n_rows <= Int32(0) or n_rows > Int32(MAX_PROBE_TOKENS):
        return Int32(-1)
    if row_i < Int32(0) or row_i >= n_rows:
        return Int32(-1)
    if layer_i < Int32(0) or layer_i >= Int32(TOTAL_LAYERS):
        return Int32(-1)
    if stage_i < Int32(0) or stage_i > Int32(26) or start_pos < Int32(0):
        return Int32(-1)

    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
        unsafe_from_address=Int(handle)
    )
    try:
        var rows = Int(n_rows)
        var row = Int(row_i)
        var layer = Int(layer_i)
        var stage = Int(stage_i)
        var tokens = List[Int]()
        _read_int32_array_into(tokens_ptr, rows, tokens)

        var dim = D
        if stage == 0 or stage == 8 or stage == 26:
            dim = engine_ptr[0].layer_qd[layer]
        elif stage == 13 or stage == 14 or stage == 20:
            dim = 6144
        elif stage == 19 or stage == 25:
            dim = 10240
        elif stage == 21 or stage == 22 or stage == 23 or stage == 24:
            dim = 48
        elif stage == 9:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_qd = engine_ptr[0].layer_qd[layer]
            var l_nh = l_qd // l_hd
            dim = l_qd + l_nh * ((l_hd + 31) // 32)
        elif stage == 10:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_kvd = engine_ptr[0].layer_kvd[layer]
            var l_nkv = l_kvd // l_hd
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            dim = 2 * l_nkv * klen * l_hd + 2 * l_nkv * klen
        elif stage == 11:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_kvd = engine_ptr[0].layer_kvd[layer]
            var l_nkv = l_kvd // l_hd
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            dim = 2 * l_nkv * klen * ((l_hd + 1) // 2) + 2 * l_nkv * klen
        elif stage == 12:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_qd = engine_ptr[0].layer_qd[layer]
            var l_nh = l_qd // l_hd
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            dim = l_nh * klen
        if dim <= 0:
            return Int32(-1)

        var dec = List[Float32](capacity=dim)
        var bat = List[Float32](capacity=dim)
        for _ in range(dim):
            dec.append(0.0)
            bat.append(0.0)

        var orig_len = engine_ptr[0].cache_len()
        @parameter
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.snapshot_for_verify()
        engine_ptr[0].set_cache_len(Int(start_pos))
        prefill_batch_impl(
            engine_ptr[0],
            tokens,
            Int(start_pos),
            Int64(0),
            -1,
            0,
            -1,
            0,
            Int64(0),
            Int64(0),
            Int64(0),
            layer,
            row,
            stage,
            Int64(Int(bat.unsafe_ptr())),
        )
        engine_ptr[0].set_cache_len(Int(start_pos))
        @parameter
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.restore_verify_snapshot()

        if row > 0:
            var tmp_logits = List[Float32](capacity=VOCAB)
            for _ in range(VOCAB):
                tmp_logits.append(0.0)
            var replay_start = 0
            if Int(start_pos) == 0:
                # Match the real fresh-request route for token 0. Feeding BOS through
                # step_logits makes continue-mode drop it and creates a false layer-0
                # decode-vs-batched cliff in this diagnostic.
                var seed = List[Int]()
                seed.append(tokens[0])
                engine_ptr[0].prefill_logits(seed, Int64(Int(tmp_logits.unsafe_ptr())))
                replay_start = 1
            for i in range(replay_start, row):
                engine_ptr[0].step_logits(tokens[i], Int64(Int(tmp_logits.unsafe_ptr())))

        engine_ptr[0].debug_decode_omlp_stage(
            tokens[row],
            layer,
            stage,
            Int64(Int(dec.unsafe_ptr())),
        )
        engine_ptr[0].set_cache_len(Int(start_pos))
        @parameter
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.restore_verify_snapshot()
            engine_ptr[0].gdn_state.finish_verify_transaction()

        var first = -1
        var count = 0
        var max_delta = Float64(0.0)
        var dot = Float64(0.0)
        var da = Float64(0.0)
        var ba = Float64(0.0)
        var dec_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(dec.unsafe_ptr())
        )
        var bat_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(bat.unsafe_ptr())
        )
        for i in range(dim):
            var dv = Float64(dec[i])
            var bv = Float64(bat[i])
            var delta = abs(dv - bv)
            if delta > max_delta:
                max_delta = delta
            dot += dv * bv
            da += dv * dv
            ba += bv * bv
            var differs = False
            for b in range(4):
                if dec_bytes[(i * 4) + b] != bat_bytes[(i * 4) + b]:
                    differs = True
            if differs:
                if first < 0:
                    first = i
                count += 1

        if out_decode != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_decode))
            for i in range(dim):
                p[i] = dec[i]
        if out_batch != 0:
            var p = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_batch))
            for i in range(dim):
                p[i] = bat[i]

        var outp = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_i32))
        outp[0] = Int32(layer)
        outp[1] = Int32(Int(start_pos))
        outp[2] = Int32(rows)
        outp[3] = Int32(row)
        outp[4] = Int32(stage)
        outp[5] = Int32(tokens[row])
        outp[6] = Int32(dim)
        outp[7] = Int32(first)
        outp[8] = Int32(count)
        outp[9] = Int32(orig_len)
        outp[10] = Int32(engine_ptr[0].cache_len())
        for i in range(11, 16):
            outp[i] = Int32(0)
        if stage == 9:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_qd = engine_ptr[0].layer_qd[layer]
            var l_nh = l_qd // l_hd
            var nqb = (l_hd + 31) // 32
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            outp[11] = Int32(l_hd)
            outp[12] = Int32(nqb)
            outp[13] = Int32(lo)
            outp[14] = Int32(klen)
            outp[15] = Int32(lo * 4)
        elif stage == 10:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_qd = engine_ptr[0].layer_qd[layer]
            var l_nh = l_qd // l_hd
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            var qk_block = 256
            if klen < 256:
                qk_block = ((klen + 31) // 32) * 32
                if qk_block < 32:
                    qk_block = 32
            if qk_block > 1024:
                qk_block = 1024
            var pv_block = 256
            if l_hd < 256:
                pv_block = ((l_hd + 31) // 32) * 32
                if pv_block < 32:
                    pv_block = 32
            if pv_block > 1024:
                pv_block = 1024
            outp[11] = Int32(l_nh)
            outp[12] = Int32(qk_block)
            outp[13] = Int32(l_nh)
            outp[14] = Int32(pv_block)
            outp[15] = Int32(0)
        elif stage == 11:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            outp[11] = Int32(l_hd)
            outp[12] = Int32((l_hd + 1) // 2)
            outp[13] = Int32(lo)
            outp[14] = Int32(klen)
            outp[15] = Int32(engine_ptr[0].layer_cache_cap[layer])
        elif stage == 12:
            var l_hd = engine_ptr[0].layer_hd[layer]
            var l_qd = engine_ptr[0].layer_qd[layer]
            var l_nh = l_qd // l_hd
            var apos = Int(start_pos) + row
            var lo = 0
            if not engine_ptr[0].layer_is_full[layer] and apos - SLIDING_WINDOW + 1 > 0:
                lo = apos - SLIDING_WINDOW + 1
            var klen = apos + 1 - lo
            var qk_block = 256
            if klen < 256:
                qk_block = ((klen + 31) // 32) * 32
                if qk_block < 32:
                    qk_block = 32
            if qk_block > 1024:
                qk_block = 1024
            outp[11] = Int32(l_nh)
            outp[12] = Int32(klen)
            outp[13] = Int32(l_nh)
            outp[14] = Int32(qk_block)
            outp[15] = Int32(0)

        if out_f32 != 0:
            var fp = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_f32))
            fp[0] = Float32(max_delta)
            fp[1] = Float32(dot / sqrt(da * ba)) if da > 0.0 and ba > 0.0 else Float32(0.0)
            fp[2] = Float32(sqrt(da))
            fp[3] = Float32(sqrt(ba))
        return Int32(0)
    except e:
        engine_ptr[0].set_cache_len(Int(start_pos))
        @parameter
        if HAS_LINEAR_ATTENTION:
            if engine_ptr[0].gdn_state.snapshot_valid:
                try:
                    engine_ptr[0].gdn_state.restore_verify_snapshot()
                except restore_e:
                    print("[nomos_debug_omlp_stage_ab restore EXC]", restore_e)
                engine_ptr[0].gdn_state.finish_verify_transaction()
        print("[nomos_debug_omlp_stage_ab EXC]", e,
              " start=", Int(start_pos), " row=", Int(row_i),
              " n_rows=", Int(n_rows), " layer=", Int(layer_i),
              " stage=", Int(stage_i))
        return Int32(-99)


def _eagle_mode_env() -> Int:
    """NOMOS_EAGLE_MODE: 1 = NVFP4 (default; drafter-study eagle3-flat blobs on the
    R2 W4A4 path), 2 = GOLD (bf16 exact-checkpoint weights, fp32 GEMV — the G2
    wiring-parity mode), 0 = Q8 legacy (GB10 stems)."""
    var b = _read_env_bytes(String("NOMOS_EAGLE_MODE"))
    if len(b) == 1:
        if b[0] == UInt8(ord("0")):
            return EAGLE_MODE_Q8
        if b[0] == UInt8(ord("2")):
            return EAGLE_MODE_GOLD
    return EAGLE_MODE_NVFP4


@export
def nomos_eagle3_load(handle: Int64, dir_ptr: Int64) -> Int32:
    """Load the EAGLE-3 drafter (weights per NOMOS_EAGLE_MODE; embedding SHARED with the
    target's mmap'd table) from the dir (C-string at dir_ptr, trailing slash) into the
    engine, behind engine.eagle3_ptr, and arm the L[1,29,56] aux-tap stash.
    ENV-GATED: requires NOMOS_EAGLE=1 (default OFF — base decode pays nothing).
    0=ok, -1=bad arg, -3=NOMOS_EAGLE off, -99=exc."""
    if handle == 0 or dir_ptr == 0:
        return Int32(-1)
    if not _env_is_one(String("NOMOS_EAGLE")):
        return Int32(-3)
    if EAGLE3_TAPS == 0:
        print("[nomos_eagle3_load] ERROR: this model profile has no EAGLE-3 assistant")
        return Int32(-4)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var dir = _read_cstr(dir_ptr)
        var ep = alloc[Eagle3Drafter](1)
        ep.init_pointee_move(Eagle3Drafter(dir, _eagle_mode_env()))
        engine_ptr[0].eagle3_ptr = UInt64(Int(ep))
        var tap_layers = eagle3_profile_tap_layers()
        engine_ptr[0].configure_drafter_taps(
            ep[0].d_taps_buf,   # arm the L[1,29,56] taps (decode_step fills them)
            ep[0].d_verify_taps_buf,
            tap_layers,
        )
        return Int32(0)
    except e:
        print("[nomos_eagle3_load EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_load(handle: Int64, dir_ptr: Int64) -> Int32:
    """Load converted Muse DFlash weights and arm five target taps.

    This establishes the device-resident weights/scratch and target tap capture
    layout [1,13,25,37,49]. Use nomos_dflash_forward_synth for the staged
    synthetic-golden forward over host-provided target_hidden/noise_embedding.
    0=ok, -1=bad arg, -99=exception.
    """
    if handle == 0 or dir_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        if engine_ptr[0].dflash_ptr != 0:
            release_dflash_drafter_handle(engine_ptr[0].dflash_ptr)
            engine_ptr[0].dflash_ptr = UInt64(0)
            engine_ptr[0].clear_drafter_taps()

        var dir = _read_cstr(dir_ptr)
        if (file_size_bytes(dir + "fc_weight.q4") <= 0 and
            file_size_bytes(dir + "fc_weight.q8") <= 0 and
            file_size_bytes(dir + "fc.weight.bf16") <= 0):
            print("[nomos_dflash_load] ERROR: no drafter weights under '", dir,
                  "' -- expected fc_weight.q4, fc_weight.q8, or fc.weight.bf16. Refusing to build a",
                  "drafter from null pointers.")
            return Int32(-2)
        validate_dflash_profile_metadata(dir)
        var dp = alloc[DFlashDrafter](1)
        dp.init_pointee_move(DFlashDrafter(dir, engine_ptr[0].ctx, engine_ptr[0].handle))
        engine_ptr[0].dflash_ptr = UInt64(Int(dp))

        var tap_layers = dflash_profile_tap_layers()
        engine_ptr[0].configure_drafter_taps(
            dp[0].d_taps_buf,
            dp[0].d_verify_taps_buf,
            tap_layers,
        )
        return Int32(0)
    except e:
        print("[nomos_dflash_load EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_get_taps(handle: Int64, out_ptr: Int64) -> Int32:
    """D2H-copy the live n-tap target residuals as fp32 [tap_count, D]."""
    if handle == 0 or out_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    try:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].dflash_ptr)
        )
        cuda_memcpy(UInt64(out_ptr), dp[0].d_taps_buf, DFLASH_FC_IN * 4, 2)
        return Int32(0)
    except e:
        print("[nomos_dflash_get_taps EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_project_context(handle: Int64, taps_ptr: Int64, out_context: Int64) -> Int32:
    """gold-diff helper: hidden_norm(fc([tap_count,D] taps)) -> [D] context.

    If taps_ptr is non-zero, it is a host fp32[tap_count*D] input and is uploaded into
    the DFlash live tap buffer first. If taps_ptr is zero, the already armed live
    tap buffer is used. out_context is host fp32[D].
    0=ok, -1=bad arg, -2=DFlash not loaded, -99=exception.
    """
    if handle == 0 or out_context == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    try:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
        if taps_ptr != 0:
            cuda_memcpy(dp[0].d_taps_buf, UInt64(taps_ptr), DFLASH_FC_IN * 4, 1)
            cuda_sync()
        dflash_project_context(engine_ptr[0].ctx, dp[0], dp[0].d_taps_buf)
        cuda_sync()
        cuda_memcpy(UInt64(out_context), dp[0].d_context, DFLASH_HIDDEN * 4, 2)
        cuda_sync()
        return Int32(0)
    except e:
        print("[nomos_dflash_project_context EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_forward_synth(
    handle: Int64,
    target_hidden_ptr: Int64,
    ctx_len: Int32,
    noise_embedding_ptr: Int64,
    out_trace: Int64,
) -> Int32:
    """Synthetic-golden DFlash forward.

    Inputs are host fp32 target_hidden[ctx_len,tap_count*D] and
    noise_embedding[16,D]. out_trace is host fp32 laid out as:
      fc_out[ctx_len,D], ctx_fused[ctx_len,D],
      layer0_out[16,D]..layer4_out[16,D], final_norm[16,D].
    This path intentionally stops before target lm_head; it gates drafter math
    against dflash/dflash_synth_goldens.npz.

    Debug sentinel: ctx_len=-1 copies the most recent production draft trace as
    noise[16,D], layer_out[5,16,D], final[16,D], logits[15,VOCAB]. This requires
    NOMOS_DFLASH_TRACE=1 before drafter load.
    DSpark sentinels reuse this ABI without adding public symbols:
      ctx_len=-2: target_hidden_ptr=int32[block] token ids; out_trace=f32[block,V]
                  receives the raw rank-256 Markov bias.
      ctx_len=-3: target_hidden_ptr=int32[1] anchor; noise_embedding_ptr=f32[candidates,V]
                  base logits; out_trace=int32[candidates] receives sequential
                  Markov-biased greedy ids (each row conditions on the prior pick).
    0=ok, -1=bad arg, -2=DFlash not loaded, -3=trace disabled, -99=exception.
    """
    if handle == 0 or out_trace == 0:
        return Int32(-1)
    if ctx_len < Int32(-3) or Int(ctx_len) > DFLASH_CTX_BATCH:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    try:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
        if ctx_len == Int32(-3):
            if target_hidden_ptr == 0 or noise_embedding_ptr == 0:
                return Int32(-1)
            var anchor_values = List[Int]()
            _read_int32_array_into(target_hidden_ptr, 1, anchor_values)
            cuda_memcpy(
                dp[0].d_logits, UInt64(noise_embedding_ptr),
                DFLASH_CANDIDATES * VOCAB * 4, 1,
            )
            dflash_markov_sample_logits(
                engine_ptr[0].ctx, dp[0], anchor_values[0]
            )
            engine_ptr[0].ctx.synchronize()
            cuda_memcpy(
                UInt64(out_trace), dp[0].d_decode_ids,
                DFLASH_CANDIDATES * 4, 2,
            )
            cuda_sync()
            return Int32(0)
        if ctx_len == Int32(-2):
            if target_hidden_ptr == 0:
                return Int32(-1)
            var markov_ids = List[Int]()
            _read_int32_array_into(target_hidden_ptr, DFLASH_BLOCK, markov_ids)
            dflash_markov_bias(engine_ptr[0].ctx, dp[0], markov_ids)
            engine_ptr[0].ctx.synchronize()
            cuda_memcpy(
                UInt64(out_trace), dp[0].d_markov_bias,
                DFLASH_BLOCK * VOCAB * 4, 2,
            )
            cuda_sync()
            return Int32(0)
        if ctx_len == Int32(-1):
            if not dp[0].trace_enabled:
                return Int32(-3)
            var out_debug = UInt64(out_trace)
            var block_bytes_debug = UInt64(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
            cuda_memcpy(out_debug, dp[0].d_target_hidden, Int(block_bytes_debug), 2)
            out_debug += block_bytes_debug
            cuda_memcpy(
                out_debug,
                dp[0].d_layer_out,
                DFLASH_LAYERS * DFLASH_BLOCK * DFLASH_HIDDEN * 4,
                2,
            )
            out_debug += UInt64(DFLASH_LAYERS) * block_bytes_debug
            cuda_memcpy(out_debug, dp[0].d_final_norm, Int(block_bytes_debug), 2)
            out_debug += block_bytes_debug
            cuda_memcpy(
                out_debug,
                dp[0].d_logits,
                DFLASH_CANDIDATES * VOCAB * 4,
                2,
            )
            cuda_sync()
            return Int32(0)
        if target_hidden_ptr == 0 or noise_embedding_ptr == 0:
            return Int32(-1)
        var clen = Int(ctx_len)
        if clen > 0:
            cuda_memcpy(
                dp[0].d_target_hidden,
                UInt64(target_hidden_ptr),
                clen * DFLASH_FC_IN * 4,
                1,
            )
        cuda_memcpy(
            dp[0].d_block_h,
            UInt64(noise_embedding_ptr),
            DFLASH_BLOCK * DFLASH_HIDDEN * 4,
            1,
        )
        cuda_sync()
        dflash_forward_block_embeddings(engine_ptr[0].ctx, dp[0], clen)
        engine_ptr[0].ctx.synchronize()

        var out = UInt64(out_trace)
        var off = UInt64(0)
        var ctx_bytes = UInt64(clen * DFLASH_HIDDEN * 4)
        if clen > 0:
            cuda_memcpy(out + off, dp[0].d_fc_out, Int(ctx_bytes), 2)
            off += ctx_bytes
            cuda_memcpy(out + off, dp[0].d_ctx_fused, Int(ctx_bytes), 2)
            off += ctx_bytes
        var block_bytes = UInt64(DFLASH_BLOCK * DFLASH_HIDDEN * 4)
        cuda_memcpy(
            out + off,
            dp[0].d_layer_out,
            DFLASH_LAYERS * DFLASH_BLOCK * DFLASH_HIDDEN * 4,
            2,
        )
        off += UInt64(DFLASH_LAYERS) * block_bytes
        cuda_memcpy(out + off, dp[0].d_final_norm, Int(block_bytes), 2)
        cuda_sync()
        return Int32(0)
    except e:
        print("[nomos_dflash_forward_synth EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_verify_fused(
    handle: Int64,
    tokens_ptr: Int64,
    n_rows: Int32,
    start_pos: Int32,
    logits_out: Int64,
) -> Int32:
    """Fused DFlash target verify/context-capture rows.

    tokens_ptr: Int32[n_rows]. logits_out may be zero when the host only wants
    tap capture for DFlash context prefill. Captures n-tap rows
    [1,13,25,37,49] into the DFlash verify-tap cache; host then calls
    nomos_dflash_append_verify_context(start_row,n_rows) for the accepted prefix.
    0=ok, -1=bad arg, -2=DFlash not loaded, -3=too many rows, -99=exception.
    """
    if handle == 0 or tokens_ptr == 0 or n_rows <= Int32(0):
        return Int32(-1)
    if Int(n_rows) > DFLASH_MAX_VERIFY_ROWS:
        return Int32(-3)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    try:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
        if HAS_LINEAR_ATTENTION:
            # Transaction boundary: both target and drafter context must describe
            # exactly the committed prefix before the first speculative row mutates
            # recurrent state. No call between this snapshot and prefill touches GDN.
            if engine_ptr[0].dflash_verify_pending:
                raise Error("DFlash verify started with an uncommitted GDN transaction")
            if engine_ptr[0].cache_len() != Int(start_pos):
                raise Error("DFlash verify start_pos does not match target cache length")
            if dp[0].cache_len() != Int(start_pos):
                raise Error("DFlash verify start_pos does not match drafter context length")
        dp[0].last_verify_rows = Int(n_rows)
        var tokens = List[Int]()
        _read_int32_array_into(tokens_ptr, Int(n_rows), tokens)
        if HAS_LINEAR_ATTENTION:
            engine_ptr[0].gdn_state.snapshot_for_verify()
            engine_ptr[0].dflash_verify_pending = True
            engine_ptr[0].dflash_verify_start = Int(start_pos)
            engine_ptr[0].dflash_verify_rows = Int(n_rows)
            engine_ptr[0].dflash_verify_tokens = tokens.copy()
        var gdn_decode_order = (
            HAS_LINEAR_ATTENTION and
            _env_float("NOMOS_VERIFY_GDN_DECODE_ORDER", 0.0) > Float32(0.5)
        )
        var gdn_fast_exact = (
            gdn_decode_order and
            _env_float("NOMOS_VERIFY_GDN_FAST_EXACT", 0.0) > Float32(0.5)
        )
        if gdn_decode_order and not gdn_fast_exact:
            # Correctness reference: execute the exact production S=1 target
            # route for every verify input. This reproduces not only the GDN
            # BF16 state round-trip cadence but also M=1 projections, softmax,
            # residuals and MLPs that feed the next GDN layer. Copy each live
            # tap row into the existing fused-verify cache; no ABI change.
            for row in range(Int(n_rows)):
                if engine_ptr[0].cache_len() != Int(start_pos) + row:
                    raise Error("decode-order verify target length drift")
                var row_logits = (
                    logits_out + Int64(row * VOCAB * 4)
                    if logits_out != 0 else Int64(0)
                )
                engine_ptr[0].step_logits(tokens[row], row_logits)
                engine_ptr[0].ctx.synchronize()
                cuda_memcpy(
                    dp[0].d_verify_taps_buf + UInt64(row * DFLASH_FC_IN * 4),
                    dp[0].d_taps_buf,
                    DFLASH_FC_IN * 4,
                    3,
                )
            engine_ptr[0].ctx.synchronize()
        else:
            prefill_batch_impl(
                engine_ptr[0],
                tokens,
                Int(start_pos),
                logits_out,
                Int(n_rows) - 1,
                Int(n_rows),
            )
        return Int32(0)
    except e:
        if HAS_LINEAR_ATTENTION and engine_ptr[0].dflash_verify_pending:
            try:
                engine_ptr[0].gdn_state.restore_verify_snapshot()
                engine_ptr[0].set_cache_len(engine_ptr[0].dflash_verify_start)
                engine_ptr[0].gdn_state.finish_verify_transaction()
                engine_ptr[0].dflash_verify_pending = False
            except:
                pass
        print("[nomos_dflash_verify_fused EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_append_verify_context(
    handle: Int64, start_row: Int32, n_rows: Int32
) -> Int32:
    """Append captured target tap rows into the DFlash context KV cache.

    start_row/n_rows address rows from the most recent nomos_dflash_verify_fused
    call. Each row is projected as hidden_norm(fc(raw n-tap concat)) and its
    per-layer K/V context is appended at the current DFlash cache length.
    Returns the new DFlash cache length, or a negative status.
    """
    if handle == 0 or start_row < Int32(0) or n_rows < Int32(0):
        return Int32(-1)
    if Int(n_rows) == 0:
        var engine_zero = UnsafePointer[GemmaEngine, MutUntrackedOrigin](
            unsafe_from_address=Int(handle))
        if engine_zero[0].dflash_ptr == 0:
            return Int32(-2)
        var dp_zero = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_zero[0].dflash_ptr))
        return Int32(dp_zero[0].cache_len())
    if Int(n_rows) > DFLASH_CTX_BATCH:
        return Int32(-3)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    try:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
        if Int(start_row) + Int(n_rows) > dp[0].last_verify_rows:
            return Int32(-1)
        if dp[0].cache_len() + Int(n_rows) > DFLASH_MAX_CTX:
            return Int32(-3)
        if HAS_LINEAR_ATTENTION:
            # Target verify inputs are [seed, draft_0, ...]. If a drafts are
            # accepted, exactly 1+a INPUT rows become committed. The correction
            # token is a verifier logit and is not processed until the next cycle.
            if not engine_ptr[0].dflash_verify_pending:
                raise Error("DFlash GDN commit requested without a verify snapshot")
            if Int(start_row) != 0:
                raise Error("DFlash GDN rollback only supports a prefix of verify rows")
            var committed = Int(n_rows)
            var verify_start = engine_ptr[0].dflash_verify_start
            var verify_rows = engine_ptr[0].dflash_verify_rows
            if committed <= 0 or committed > verify_rows:
                raise Error("DFlash GDN committed row count is outside verify transaction")

            # Rejected verify rows were never appended by today's call order, but
            # truncate explicitly: correctness must not depend on that incidental
            # ordering surviving a refactor.
            dp[0].set_cache_len(verify_start)

            if committed < verify_rows:
                # Append-only softmax KV rolls back by logical length. Recurrent
                # state cannot: restore the pre-verify pools, then replay the exact
                # committed target-input prefix through the FULL model so every GDN
                # layer sees qkv/a/b derived from the correct preceding layers.
                engine_ptr[0].gdn_state.restore_verify_snapshot()
                engine_ptr[0].set_cache_len(verify_start)
                var replay = List[Int]()
                for i in range(committed):
                    replay.append(engine_ptr[0].dflash_verify_tokens[i])
                var gdn_decode_order = (
                    _env_float("NOMOS_VERIFY_GDN_DECODE_ORDER", 0.0) > Float32(0.5)
                )
                var gdn_fast_exact = (
                    gdn_decode_order and
                    _env_float("NOMOS_VERIFY_GDN_FAST_EXACT", 0.0) > Float32(0.5)
                )
                if gdn_decode_order and not gdn_fast_exact:
                    for row in range(committed):
                        if engine_ptr[0].cache_len() != verify_start + row:
                            raise Error("decode-order rollback replay target length drift")
                        engine_ptr[0].step_logits(replay[row], Int64(0))
                        engine_ptr[0].ctx.synchronize()
                        cuda_memcpy(
                            dp[0].d_verify_taps_buf + UInt64(row * DFLASH_FC_IN * 4),
                            dp[0].d_taps_buf,
                            DFLASH_FC_IN * 4,
                            3,
                        )
                    engine_ptr[0].ctx.synchronize()
                else:
                    prefill_batch_impl(
                        engine_ptr[0], replay, verify_start,
                        Int64(0), committed - 1, committed,
                    )
            else:
                # Full accept: the first verify already produced the exact state.
                engine_ptr[0].set_cache_len(verify_start + committed)

            if engine_ptr[0].cache_len() != verify_start + committed:
                raise Error("DFlash GDN rollback left target cache at wrong length")
            if dp[0].cache_len() != verify_start:
                raise Error("DFlash context was not truncated before committed append")
        dflash_append_context_rows(
            engine_ptr[0].ctx,
            dp[0],
            dp[0].d_verify_taps_buf + UInt64(Int(start_row) * DFLASH_FC_IN * 4),
            Int(n_rows),
        )
        engine_ptr[0].ctx.synchronize()
        if HAS_LINEAR_ATTENTION:
            if dp[0].cache_len() != engine_ptr[0].cache_len():
                raise Error("DFlash target/drafter cache lengths diverged after commit")
            engine_ptr[0].gdn_state.finish_verify_transaction()
            engine_ptr[0].dflash_verify_pending = False
            engine_ptr[0].dflash_verify_tokens = List[Int]()
        return Int32(dp[0].cache_len())
    except e:
        print("[nomos_dflash_append_verify_context EXC]", e)
        return Int32(-99)


@export
def nomos_dflash_cache_len(handle: Int64) -> Int32:
    if handle == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
        unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
    return Int32(dp[0].cache_len())


@export
def nomos_dflash_set_len(handle: Int64, n: Int32) -> Int32:
    if handle == 0 or n < Int32(0) or Int(n) > DFLASH_MAX_CTX:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
        unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
    dp[0].set_cache_len(Int(n))
    return Int32(0)


@export
def nomos_dflash_reset(handle: Int64) -> Int32:
    if handle == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
        unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
    dp[0].reset()
    return Int32(0)


@export
def nomos_dflash_draft_block(
    handle: Int64,
    committed_token: Int32,
    abs_start_pos: Int32,
    out_ids: Int64,
) -> Int32:
    """Draft one DFlash block from [committed_token, mask*15].

    The block embeddings are gathered from the target embedding table at
    absolute positions [abs_start_pos, abs_start_pos+16). Returns 15 and writes
    greedy draft ids to out_ids[15]. Requires the target Q4 lm-head.
    """
    if handle == 0 or out_ids == 0 or committed_token < Int32(0) or committed_token >= Int32(VOCAB):
        return Int32(-1)
    if abs_start_pos < Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].dflash_ptr == 0:
        return Int32(-2)
    # NVFP4 (discrete-card) build: read out through d_embed_nvfp4; GB10 build: d_embed_q4.
    var use_nvfp4 = engine_ptr[0].d_embed_nvfp4 != 0
    if not use_nvfp4 and engine_ptr[0].d_embed_q4 == 0:
        return Int32(-3)
    try:
        var dp = UnsafePointer[DFlashDrafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].dflash_ptr))
        var block = List[Float32](capacity=DFLASH_BLOCK * DFLASH_HIDDEN)
        for _ in range(DFLASH_BLOCK * DFLASH_HIDDEN):
            block.append(0.0)
        for row in range(DFLASH_BLOCK):
            var tok = Int(committed_token)
            if row > 0:
                tok = dp[0].mask_token_id
            _ = external_call["nomos_pread", c_size_t](
                engine_ptr[0].embed_fd,
                UnsafePointer(to=block[row * DFLASH_HIDDEN]),
                c_size_t(DFLASH_HIDDEN * 4),
                c_size_t(tok * DFLASH_HIDDEN * 4),
            )
        @parameter
        if DRAFTER_EMBED_SQRT_SCALE:
            var embed_scale = sqrt(Float32(DFLASH_HIDDEN))
            for i in range(DFLASH_BLOCK * DFLASH_HIDDEN):
                block[i] *= embed_scale
        cuda_upload(dp[0].d_block_h, block)
        if dp[0].trace_enabled:
            cuda_memcpy(
                dp[0].d_target_hidden,
                dp[0].d_block_h,
                DFLASH_BLOCK * DFLASH_HIDDEN * 4,
                3,
            )
        cuda_sync()
        if use_nvfp4:
            dflash_draft_block_nvfp4(
                engine_ptr[0].ctx,
                dp[0],
                engine_ptr[0].handle,
                engine_ptr[0].d_embed_nvfp4,
                engine_ptr[0].embed_global,
                engine_ptr[0].d_w4a4_packed,
                engine_ptr[0].d_w4a4_bs,
                engine_ptr[0].d_w4a4_global,
                engine_ptr[0].d_w4a4_bs_sf,
                engine_ptr[0].d_w4a4_wbs_sf,
                engine_ptr[0].d_lmhead_cpad,
                Int(committed_token),
                Int(abs_start_pos),
            )
        else:
            dflash_draft_block(
                engine_ptr[0].ctx,
                dp[0],
                engine_ptr[0].d_embed_q4,
                Int(committed_token),
                Int(abs_start_pos),
            )
        engine_ptr[0].ctx.synchronize()
        cuda_memcpy(UInt64(out_ids), dp[0].d_decode_ids, DFLASH_CANDIDATES * 4, 2)
        cuda_sync()
        return Int32(DFLASH_CANDIDATES)
    except e:
        print("[nomos_dflash_draft_block EXC]", e)
        return Int32(-99)


@export
def nomos_eagle3_verify_fused(
    handle: Int64,
    tokens_ptr: Int64,
    n_rows: Int32,
    start_pos: Int32,
    logits_out: Int64,
) -> Int32:
    """Fused EAGLE-3 target verify/decode rows.

    tokens_ptr: Int32[n_rows], usually K drafts plus one host-chosen bonus/correction row.
    logits_out: fp32[n_rows, VOCAB], row i = target logits after consuming tokens[0..i].
    Leaves target KV at start_pos+n_rows; caller rolls back with nomos_kv_set_len after accept.
    Captures L[1,29,56] taps for every row into the drafter verify-tap cache, and arms the
    last row by default. After host computes num_acc, call nomos_eagle3_select_verify_tap(row)
    to arm the accepted-final row for the next nomos_eagle3_draft.
    0=ok, -1=bad arg, -2=EAGLE not loaded, -3=too many rows, -99=exception.
    """
    if handle == 0 or tokens_ptr == 0 or logits_out == 0 or n_rows <= Int32(0):
        return Int32(-1)
    if Int(n_rows) > MAX_VERIFY_ROWS:
        return Int32(-3)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        ep[0].last_verify_rows = Int(n_rows)
        var tokens = List[Int]()
        _read_int32_array_into(tokens_ptr, Int(n_rows), tokens)
        prefill_batch_impl(
            engine_ptr[0],
            tokens,
            Int(start_pos),
            logits_out,
            Int(n_rows) - 1,
            Int(n_rows),
        )
        return Int32(0)
    except e:
        print("[nomos_eagle3_verify_fused EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_select_verify_tap(handle: Int64, row: Int32) -> Int32:
    """Arm the live EAGLE-3 [3,D] tap from a previously fused-verified row.

    row is zero-based within the last nomos_eagle3_verify_fused tokens array.
    Call after host accept/rollback so the next nomos_eagle3_draft consumes the
    tap for the accepted-final token. 0=ok, -1=bad arg/stale row, -2=EAGLE not loaded.
    """
    if handle == 0 or row < Int32(0) or Int(row) >= MAX_VERIFY_ROWS:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
        unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
    if Int(row) >= ep[0].last_verify_rows:
        return Int32(-1)
    cuda_memcpy(
        ep[0].d_taps_buf,
        ep[0].d_verify_taps_buf + UInt64(Int(row) * ep[0].n_taps * ep[0].dB * 4),
        ep[0].n_taps * ep[0].dB * 4,
        3,
    )
    cuda_sync()
    return Int32(0)

@export
def nomos_eagle3_draft_from_taps(handle: Int64, taps_ptr: Int64, seed_token: Int32, k: Int32,
        out_draft_ids: Int64, out_target_ids: Int64, out_inter: Int64) -> Int32:
    """gold-diff entry. Upload FIXED taps [3*dB] (host fp32 at taps_ptr), draft k steps from
    seed_token at position_start=1, write per-step draft_ids + target_ids (Int32) to
    out_draft_ids / out_target_ids, and (if out_inter!=0) the captured intermediates as fp32:
    fc_out[dB], then per step attn_out[nh*hd], o_out[dB], hidden_out[dB]. Returns the count
    drafted, or -1 bad arg / -2 not loaded / -99 exc."""
    if handle == 0 or taps_ptr == 0 or k <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        var dB = ep[0].dB
        var nh_hd = ep[0].nh * ep[0].hd
        # upload the fixed taps host -> device [3*dB]
        var d_taps = cuda_malloc(3 * dB * 4)
        cuda_memcpy(d_taps, UInt64(taps_ptr), 3 * dB * 4, 1)   # H2D
        cuda_sync()
        var drafts = List[Int32]()
        # seed_pos = 1 (the gold's position_start); embed = the SHARED target table (mmap fd)
        eagle3_draft_k(engine_ptr[0].ctx, ep[0], d_taps, 1, Int(seed_token), Int(engine_ptr[0].embed_fd), Int(k), drafts)
        cuda_free(d_taps)
        var n = len(drafts) if len(drafts) < Int(k) else Int(k)
        var od = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_draft_ids))
        var ot = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_target_ids))
        for i in range(n):
            od[i] = ep[0].cap_draft_ids[i]
            ot[i] = drafts[i]                  # drafts holds the target ids
        if out_inter != 0:
            var oi = UInt64(out_inter)
            cuda_memcpy(oi, ep[0].cap_fc_out, dB * 4, 2)       # D2H fc_out[dB]
            var off = UInt64(dB * 4)
            for i in range(n):
                cuda_memcpy(oi + off, ep[0].cap_attn_out + UInt64(i * nh_hd * 4), nh_hd * 4, 2)
                off += UInt64(nh_hd * 4)
                cuda_memcpy(oi + off, ep[0].cap_o_out + UInt64(i * dB * 4), dB * 4, 2)
                off += UInt64(dB * 4)
                cuda_memcpy(oi + off, ep[0].cap_hidden_out + UInt64(i * dB * 4), dB * 4, 2)
                off += UInt64(dB * 4)
        return Int32(n)
    except e:
        print("[nomos_eagle3_draft_from_taps EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_head_logits(handle: Int64, taps_ptr: Int64, seed_token: Int32, k: Int32,
        out_draft_ids: Int64, out_target_ids: Int64, out_logits: Int64) -> Int32:
    """D2 parity-gate entry (G2/G3): clean single-shot head forward from HOST taps.
    Resets the drafter's persistent KV (seq_len=0), uploads taps [3*dB] fp32, drafts
    k steps (KV_REUSE=0 semantics: no committed prefix), and writes per-step
    draft_ids/target_ids (Int32[k]) plus (if out_logits!=0) the FULL pre-d2t draft
    logits fp32[k, 32000]. Returns count drafted, -1 bad arg / -2 not loaded / -99 exc."""
    if handle == 0 or taps_ptr == 0 or k <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        var dB = ep[0].dB
        ep[0].seq_len = 0                    # clean-slate drafter KV (single-shot gate run)
        ep[0].last_draft_base_len = 0
        var d_taps = cuda_malloc(3 * dB * 4)
        cuda_memcpy(d_taps, UInt64(taps_ptr), 3 * dB * 4, 1)   # H2D
        cuda_sync()
        var drafts = List[Int32]()
        eagle3_draft_k(engine_ptr[0].ctx, ep[0], d_taps, 1, Int(seed_token),
                       Int(engine_ptr[0].embed_fd), Int(k), drafts)
        cuda_sync()
        cuda_free(d_taps)
        var n = len(drafts) if len(drafts) < Int(k) else Int(k)
        if out_draft_ids != 0:
            var od = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_draft_ids))
            for i in range(n):
                od[i] = ep[0].cap_draft_ids[i]
        if out_target_ids != 0:
            var ot = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_target_ids))
            for i in range(n):
                ot[i] = drafts[i]
        if out_logits != 0:
            var ol = UInt64(out_logits)
            for i in range(n):
                cuda_memcpy(
                    ol + UInt64(i * DRAFT_VOCAB * 4),
                    ep[0].cap_logits_out + UInt64(i * DRAFT_VOCAB * 4),
                    DRAFT_VOCAB * 4,
                    2,
                )
            cuda_sync()
        return Int32(n)
    except e:
        print("[nomos_eagle3_head_logits EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_draft(handle: Int64, seed_token: Int32, k: Int32, out_target_ids: Int64) -> Int32:
    """Live spec path: draft k tokens off the engine's ARMED L[1,29,56] tap — already in the drafter's
    d_taps_buf, filled by the last nomos_decode_step (the tap is armed at nomos_eagle3_load). Writes the
    k target ids (Int32) to out_target_ids. The drafter KV writes from this call remain speculative until
    nomos_eagle3_commit rolls them back/replays the accepted prefix. Returns count drafted, -1 bad arg /
    -2 not loaded / -99 exc."""
    if handle == 0 or out_target_ids == 0 or k <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        var drafts = List[Int32]()
        # taps = the engine's L[1,29,56] tap in ep[0].d_taps_buf (last decode); seed_pos=1, shared embed
        ep[0].next_seed_token = Int(seed_token)
        eagle3_draft_k(engine_ptr[0].ctx, ep[0], ep[0].d_taps_buf, 1, Int(seed_token), Int(engine_ptr[0].embed_fd), Int(k), drafts)
        var ot = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_target_ids))
        var n = len(drafts) if len(drafts) < Int(k) else Int(k)
        for i in range(n):
            ot[i] = drafts[i]
        return Int32(n)
    except e:
        print("[nomos_eagle3_draft EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_draft_trace(
    handle: Int64,
    seed_token: Int32,
    k: Int32,
    out_draft_ids: Int64,
    out_target_ids: Int64,
    out_inter: Int64,
    out_logits: Int64,
) -> Int32:
    """Debug-only live EAGLE-3 draft trace using the exact armed-tap path from
    nomos_eagle3_draft.

    out_draft_ids: Int32[n] draft-vocab argmax ids.
    out_target_ids: Int32[n] post-d2t target ids.
    out_inter: optional fp32[n, dB + nh*hd + dB + dB], per step:
      fused_in[dB], attn_out[nh*hd], o_out[dB], hidden_out[dB].
      For step 0, fused_in is fc(taps); for later steps it is the prior drafter
      hidden fed forward.
    out_logits: optional fp32[n, 32000] pre-d2t draft logits for host top-k.
    Returns count drafted, -1 bad arg, -2 not loaded, -99 exception.
    """
    if handle == 0 or k <= Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        var dB = ep[0].dB
        var nh_hd = ep[0].nh * ep[0].hd
        var dv = 32000
        var drafts = List[Int32]()
        ep[0].next_seed_token = Int(seed_token)
        eagle3_draft_k(engine_ptr[0].ctx, ep[0], ep[0].d_taps_buf, 1, Int(seed_token), Int(engine_ptr[0].embed_fd), Int(k), drafts)
        cuda_sync()
        var n = len(drafts) if len(drafts) < Int(k) else Int(k)
        if out_draft_ids != 0:
            var od = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_draft_ids))
            for i in range(n):
                od[i] = ep[0].cap_draft_ids[i]
        if out_target_ids != 0:
            var ot = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out_target_ids))
            for i in range(n):
                ot[i] = drafts[i]
        if out_inter != 0:
            var oi = UInt64(out_inter)
            var off = UInt64(0)
            for i in range(n):
                cuda_memcpy(oi + off, ep[0].cap_fc_out + UInt64(i * dB * 4), dB * 4, 2)
                off += UInt64(dB * 4)
                cuda_memcpy(oi + off, ep[0].cap_attn_out + UInt64(i * nh_hd * 4), nh_hd * 4, 2)
                off += UInt64(nh_hd * 4)
                cuda_memcpy(oi + off, ep[0].cap_o_out + UInt64(i * dB * 4), dB * 4, 2)
                off += UInt64(dB * 4)
                cuda_memcpy(oi + off, ep[0].cap_hidden_out + UInt64(i * dB * 4), dB * 4, 2)
                off += UInt64(dB * 4)
        if out_logits != 0:
            var ol = UInt64(out_logits)
            for i in range(n):
                cuda_memcpy(
                    ol + UInt64(i * dv * 4),
                    ep[0].cap_logits_out + UInt64(i * dv * 4),
                    dv * 4,
                    2,
                )
        return Int32(n)
    except e:
        print("[nomos_eagle3_draft_trace EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_commit(
    handle: Int64,
    prefix_tokens_ptr: Int64,
    prefix_taps_ptr: Int64,
    prefix_count: Int32,
    next_token: Int32,
    next_taps_ptr: Int64,
) -> Int32:
    """Accept-aware drafter KV update for the live EAGLE-3 loop.

    Roll back the speculative drafter KV to the committed length captured before the
    last nomos_eagle3_draft, replay prefix_tokens[0:prefix_count] using raw target
    taps [prefix_count, 3*dB], then install next_taps [3*dB] for the next draft seed.

    The replay prefix normally is [previous_seed, accepted_draft_0..accepted_draft_{a-1}].
    next_token is the correction/bonus token that should be passed as the next
    nomos_eagle3_draft seed; it is recorded but not appended by commit.

    Returns 0 ok, -1 bad arg, -2 EAGLE not loaded, -3 prefix exceeds drafter KV cap,
    -99 exception.
    """
    if handle == 0 or next_taps_ptr == 0 or prefix_count < Int32(0):
        return Int32(-1)
    if prefix_count > Int32(0) and (prefix_tokens_ptr == 0 or prefix_taps_ptr == 0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        var pc = Int(prefix_count)
        if ep[0].last_draft_base_len + pc > 1024:
            return Int32(-3)
        var d_next_taps = cuda_malloc(3 * ep[0].dB * 4)
        cuda_memcpy(d_next_taps, UInt64(next_taps_ptr), 3 * ep[0].dB * 4, 1)   # H2D
        var d_prefix_taps = UInt64(0)
        if pc > 0:
            d_prefix_taps = cuda_malloc(pc * 3 * ep[0].dB * 4)
            cuda_memcpy(d_prefix_taps, UInt64(prefix_taps_ptr), pc * 3 * ep[0].dB * 4, 1)  # H2D
        cuda_sync()
        eagle3_commit_prefix(
            engine_ptr[0].ctx,
            ep[0],
            UInt64(prefix_tokens_ptr),
            d_prefix_taps,
            pc,
            d_next_taps,
            Int(next_token),
            Int(engine_ptr[0].embed_fd),
        )
        cuda_sync()
        if d_prefix_taps != UInt64(0):
            cuda_free(d_prefix_taps)
        cuda_free(d_next_taps)
        return Int32(0)
    except e:
        print("[nomos_eagle3_commit EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_get_taps(handle: Int64, out_ptr: Int64) -> Int32:
    """D2H-copy the drafter's current tap buffer (d_taps_buf, [3*dB] = the last decode's L[1,29,56]
    aux-hidden tap) to host out_ptr fp32. For the bf16-vs-Q8 accept diagnostic. 0=ok, -1/-2 err, -99 exc."""
    if handle == 0 or out_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        cuda_memcpy(UInt64(out_ptr), ep[0].d_taps_buf, 3 * ep[0].dB * 4, 2)   # D2H
        return Int32(0)
    except e:
        print("[nomos_eagle3_get_taps EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_get_verify_taps(handle: Int64, row: Int32, out_ptr: Int64) -> Int32:
    """D2H-copy ONE fused-verify tap row ([3*dB] fp32, the L[1,29,56] hidden captured for
    verify row `row` by the last nomos_eagle3_verify_fused) to host out_ptr. The host
    spec loop feeds these back as the RAW target taps for nomos_eagle3_commit's accepted-
    prefix replay (row i pairs with accepted draft i; row num_acc = next_taps).
    0=ok, -1=bad arg/stale row, -2=EAGLE not loaded, -99=exc."""
    if handle == 0 or out_ptr == 0 or row < Int32(0) or Int(row) >= MAX_VERIFY_ROWS:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    try:
        var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
            unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
        if Int(row) >= ep[0].last_verify_rows:
            return Int32(-1)
        cuda_memcpy(
            UInt64(out_ptr),
            ep[0].d_verify_taps_buf + UInt64(Int(row) * ep[0].n_taps * ep[0].dB * 4),
            ep[0].n_taps * ep[0].dB * 4,
            2,
        )
        cuda_sync()
        return Int32(0)
    except e:
        print("[nomos_eagle3_get_verify_taps EXC]", e)
        return Int32(-99)

@export
def nomos_eagle3_reset(handle: Int64) -> Int32:
    """Reset the drafter's persistent KV (seq_len=0) for a new sequence. 0=ok, -1/-2 err."""
    if handle == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    if engine_ptr[0].eagle3_ptr == 0:
        return Int32(-2)
    var ep = UnsafePointer[Eagle3Drafter, MutUntrackedOrigin](
        unsafe_from_address=Int(engine_ptr[0].eagle3_ptr))
    ep[0].seq_len = 0
    ep[0].last_draft_base_len = 0
    ep[0].next_seed_token = 0
    return Int32(0)

@export
def nomos_decode_step_token(handle: Int64, token: Int32, out_token_ptr: Int64) -> Int32:
    """Perf Lever A (additive; nomos_decode_step untouched). Append `token`, run the
    decode forward, then on-device softcap+argmax -> write the chosen greedy token
    (Int32, 4 bytes) to out_token_ptr. NO VOCAB logits copy. Greedy fast-path only;
    sampling/grammar-constrained steps stay on nomos_decode_step. 0 = ok, -1 = bad arg,
    -99 = exception."""
    if handle == 0 or out_token_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        engine_ptr[0].step_token(Int(token), out_token_ptr)
        return Int32(0)
    except e:
        print("[nomos_decode_step_token EXC]", e, "  token=", Int(token))
        return Int32(-99)

@export
def nomos_kv_cache_len(handle: Int64) -> Int32:
    """KV reuse: current cache valid length from the profile's authoritative
    softmax-KV layer (layer 0 for Gemma/Muse, first full layer for GDN hybrids).
    -1 = bad arg."""
    if handle == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    return Int32(engine_ptr[0].cache_len())

@export
def nomos_kv_set_len(handle: Int64, n: Int32) -> Int32:
    """KV reuse: position the cache to length n (truncate to the common-prefix length) before a
    continue-prefill of the new suffix. The KV memory [0,n) stays valid. 0 = ok, -1 = bad arg."""
    if handle == 0 or n < Int32(0):
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    engine_ptr[0].set_cache_len(Int(n))
    return Int32(0)

@export
def nomos_prefill_cont(handle: Int64, ids_ptr: Int64, n: Int32, logits_out_ptr: Int64) -> Int32:
    """LOGITS-ENGINE continue-prefill (KV reuse): prefill the `n` suffix tokens at the CURRENT
    cache position (set via nomos_kv_set_len), attending the held prefix, and write the last-
    position logits to logits_out_ptr. 0 = ok, -1 = bad arg, -99 = exception."""
    if handle == 0 or n <= Int32(0) or ids_ptr == 0 or logits_out_ptr == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    try:
        var suffix = List[Int]()
        _read_int32_array_into(ids_ptr, Int(n), suffix)
        engine_ptr[0].prefill_cont_logits(suffix, logits_out_ptr)
        return Int32(0)
    except e:
        print("[nomos_prefill_cont EXC]", e, "  n=", Int(n))
        return Int32(-99)

@export
def nomos_generate_stream(
    handle: Int64,
    prompt_ptr: Int64,
    n_prompt: Int32,
    max_new_tokens: Int32,
    cb_id: Int64,
    temperature: Float32,
    top_p: Float32,
    rep_penalty: Float32,
) -> Int32:
    """Streaming generate. After each generated token, calls
    `nomos_token_cb(cb_id, token_id) -> Int32` (resolved at runtime via the
    dynamic linker — the Go cgo binary exports this symbol). Callback
    returns 0 to continue, non-zero to stop.

    Returns the total token count emitted, or negative on error.
    """
    if handle == 0:
        return Int32(-1)
    if n_prompt <= 0 or max_new_tokens <= 0:
        return Int32(-1)
    if cb_id == 0:
        return Int32(-1)
    var engine_ptr = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    # Per-call sampling (see nomos_generate): snapshot defaults, restore after.
    var saved_temp = engine_ptr[0].temp
    var saved_top_p = engine_ptr[0].top_p
    var saved_rep = engine_ptr[0].rep_penalty
    try:
        if temperature >= Float32(0.0):
            engine_ptr[0].temp = temperature
        if top_p > Float32(0.0):
            engine_ptr[0].top_p = top_p
        if rep_penalty > Float32(0.0):
            engine_ptr[0].rep_penalty = rep_penalty
        var prompt = List[Int]()
        _read_int32_array_into(prompt_ptr, Int(n_prompt), prompt)
        var out = List[Int]()
        # Streaming generation is also a complete independent request; it must
        # share nomos_generate/nomos_prefill's fresh-state contract.
        engine_ptr[0].reset_kv_cache()
        engine_ptr[0].run_inference(
            prompt, Int(max_new_tokens), False, out, cb_id,
            force_rep_penalty=(
                rep_penalty > Float32(0.0) and rep_penalty != Float32(1.0)
            ),
        )
        engine_ptr[0].temp = saved_temp
        engine_ptr[0].top_p = saved_top_p
        engine_ptr[0].rep_penalty = saved_rep
        return Int32(len(out))
    except:
        engine_ptr[0].temp = saved_temp
        engine_ptr[0].top_p = saved_top_p
        engine_ptr[0].rep_penalty = saved_rep
        return Int32(-99)

@export
def nomos_reset_kv(handle: Int64) -> Int32:
    """Reset the KV cache to fresh-init state for a NEW stateless request.
    CALLER-GATED — this is the `fresh` boundary. Stateless serving / each
    benchmark query calls this before inject+generate; multi-turn conversation
    callers call it only at turn 1 (NOT between turns — it wipes context).
    Returns 0 ok, -1 null handle, -99 exception."""
    if handle == 0:
        return Int32(-1)
    try:
        var engine = UnsafePointer[GemmaEngine, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        engine[0].reset_kv_cache()
        return Int32(0)
    except:
        return Int32(-99)
