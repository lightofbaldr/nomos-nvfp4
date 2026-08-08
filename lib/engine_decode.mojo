"""run_inference — the decode loop, extracted from GemmaEngine so the struct
stays a thin orchestrator (file-size rule: keep modules in the 500-800 band).
Free def taking `mut self: GemmaEngine`; the engine ↔ this-module mutual import
compiles in this Mojo (verified 2026-06-12). Verbatim move, behavior-identical.
"""
from lib.gemma4_engine import (
    GemmaEngine, _env_float,
    D, FF, FULL_HD, MAX_PROBE_TOKENS, NH, SLIDING_WINDOW, TOTAL_LAYERS, VOCAB,
)
from lib.cuda import cuda_download, cuda_memcpy, cuda_memset_2d, cuda_sync, cuda_upload, cuda_last_error, cuda_malloc, cuda_free
from lib.cuda import cuda_budget_active, cuda_budget_token_begin, cuda_budget_mark, cuda_budget_token_end
from lib.cublas import gpu_matmul
from lib.fp4_act import gpu_matmul_nvfp4_w4a4_dev, _w4a4_sync_checkpoint
from lib.fp4_weights import gpu_matmul_nvfp4_fused_dev
from lib.fp4_gemv_v2 import gpu_matmul_nvfp4_fused_v3_dev
from lib.decode_argmax import device_decode_token   # Perf Lever A: on-device softcap+argmax
from lib.q4_gemv_dp4a import gpu_matmul_q4_dp4a_dev, act_precision   # dp4a Q4 lm-head
from lib.gemma4_ops import rmsnorm
from lib.gemma4_layer import prepare_qkv_dev, apply_output_and_mlp_dev, W4A4Scratch
from lib.kv_cache_quant import gpu_append_quant_kv_i8, gpu_dequant_kv_i8_layer
from lib.kv_cache_quant import gpu_append_quant_kv_i4, gpu_dequant_kv_i4_layer, gpu_dequant_kv_i4_to_q8_layer, debug_kv_i4_source_to_f32_dev
from std.time import perf_counter_ns
from lib.attention_gpu import append_kv_gpu_dev, attention_gpu_fp32_dev
from lib.attention_gpu import append_kv_gpu_bf16_dev, attention_gpu_bf16_dev
from lib.attention_gpu_int8 import attention_gpu_int8_dev, debug_q_i8_to_f32_dev, debug_kv_i8_view_to_f32_dev, debug_qk_scores_int8_dev
from lib.engine_init import _strict_q4, _flag_violation, VIOL_BF16_ATTN_DECODE, VIOL_FP32_ATTN_DECODE
from lib.batched_attn_gpu import gpu_append_kv_batched_bf16
from lib.sampling import (
    final_logit_softcap, apply_rep_penalty, mask_eos_if_too_early,
    fix_contraction_me_bug, fix_contraction_t_bug, sample_top_p, argmax,
    mask_tool_grammar, mask_special_tokens,
)
from lib.spec_draft_e2b import e2b_draft_contract_ok
from std.ffi import external_call, c_size_t
from std.math import sqrt
from std.memory import UnsafePointer
from std.collections import List


def run_inference_impl(
    mut self: GemmaEngine,
    prompt: List[Int],
    max_new_tokens: Int,
    store_mode: Bool,
    mut out_tokens: List[Int],
    cb_id: Int64,
    logits_out: Int64 = 0,
    fast_token_out: Int64 = 0,   # Perf Lever A: if !=0, device argmax -> 4-byte token here; skip the VOCAB copy
    debug_qkv_layer: Int = -1,
    debug_normed_out: Int64 = 0,
    debug_k_out: Int64 = 0,
    debug_v_out: Int64 = 0,
    debug_omlp_layer: Int = -1,
    debug_omlp_stage: Int = -1,
    debug_omlp_out: Int64 = 0,
) raises:
    """Run prefill + decode (or store mode). Appends generated token IDs
    to out_tokens.

    If `cb_id` is non-zero, after each generated token (decode phase only)
    we call back into the Go side via external_call["nomos_token_cb", ...]
    with (cb_id, token_id). The callback returns Int32: 0 = continue,
    1 = stop generation early. This is how the cgo bridge does true
    per-token streaming — no buffer-then-replay.

    Caller manages the start_pos via cache truncation (we always truncate
    to 0 here for simple stateless requests)."""

    if _env_float("NOMOS_SPEC_DECODE", 0.0) > Float32(0.5):
        var spec_k = Int(_env_float("NOMOS_SPEC_K", 6.0))
        if not e2b_draft_contract_ok(spec_k, VOCAB):
            raise Error("NOMOS_SPEC_DECODE invalid draft contract")
        raise Error(
            "NOMOS_SPEC_DECODE=1 reached the Unit F scaffold: E2B draft weights "
            + "and target [K+1,V] verify logits are not executor-bound in this jail"
        )

    # KV-prefix continue mode (one-shot, one-shot continue): keep the injected
    # cache and prefill the live prompt at positions cache_lens.. instead of
    # resetting to 0, so the prompt attends to the injected prefix.
    var continue_mode = self.kv_continue
    self.kv_continue = False
    var start_pos: Int = self.cache_lens_fp32[0] if continue_mode else 0
    var num_new = len(prompt)
    var max_tokens = 0 if store_mode else max_new_tokens
    var prompt_end = start_pos + num_new
    if _env_float("NOMOS_DEBUG_KV", 0.0) > 0.5:
        print("[kvdbg] run_inference ENTER: continue_mode=", continue_mode,
              " inherited cache_lens[0]=", self.cache_lens_fp32[0],
              " start_pos=", start_pos, " num_new=", num_new,
              " store_mode=", store_mode, " temp=", self.temp,
              " rep_pen=", self.rep_penalty)

    # FIX-TEST (GenerateWithAtom echo bug, 2026-06-11): the attention scratch
    # buffers (d_scores_scratch NH*self.max_seq, d_attn_out_scratch NH*FULL_HD) are
    # allocated once and reused across calls, never cleared. A cold/reused
    # engine reads stale/uninitialized values that perturb the followup's
    # attention to the injected prefix → echo instead of recall (warmth-
    # dependent: cold echoes, warm recalls). Zeroing them at entry should make
    # every call deterministic. Gated for A/B confirmation; make unconditional
    # once proven.
    if _env_float("NOMOS_ZERO_SCRATCH", 0.0) > 0.5:
        cuda_memset_2d(self.d_scores_scratch, NH * self.max_seq * 4, 0, NH * self.max_seq * 4, 1)
        cuda_memset_2d(self.d_attn_out_scratch, NH * FULL_HD * 4, 0, NH * FULL_HD * 4, 1)
        cuda_sync()

    if not continue_mode:
        # Reset KV caches for a fresh request
        for li in range(TOTAL_LAYERS):
            self.cache_lens_fp32[li] = 0

    # In continue mode, drop a leading <bos> (id 2) from the live prompt: the
    # injected prefix already opened the sequence, and a mid-stream BOS makes
    # the model wander after the recalled content. We prefill from prompt[skip].
    var prompt_skip = 0
    if continue_mode and num_new > 0 and prompt[0] == 2:
        prompt_skip = 1
        num_new = num_new - 1
        prompt_end = start_pos + num_new

    # Batched prefill: process prompt[0:num_new-1] as one [S,d] batch to
    # populate the KV cache, then let the per-token loop below handle the
    # LAST prompt token + decode unchanged. Gated by NOMOS_BATCHED_PREFILL
    # (default on); skipped for single-token prompts.
    #
    # #284: store_mode is normally per-token (slow: 128 sequential fwd passes),
    # but the batched store path is SAFE through batched — prefill_batch writes the KV cache identically.
    var prefill_start = 0
    # #2 KV reuse: batched prefill now also serves continue mode (suffix appended at start_pos,
    # attending the cached prefix). Gated to the kv_quant product path — non-quant continue keeps the proven per-token path (prefill_batch offset
    # raises without kv_quant by design).
    var use_batched = _env_float("NOMOS_BATCHED_PREFILL", 1.0) > Float32(0.5)
    if continue_mode and not self.kv_quant:
        use_batched = False
    if use_batched and num_new > 1 and (not store_mode):
        var dflash_prefill_context = self.dflash_ptr != 0 and not continue_mode and max_new_tokens == 0
        var pref = List[Int]()
        for i in range(num_new - 1):
            pref.append(prompt[i + prompt_skip])
        # ── CHUNKED PREFILL (NOMOS_PREFILL_CHUNK=C, default 0=off) ────────────────────────────
        # THE 24GB CONTEXT CEILING: prefill_batch allocates O(S²) attention scores (NH·S²·4 =
        # 2.8GB at S=4.7K) and O(S·FF) transients, so a 24GB card cannot prefill past ~4K in any
        # config (measured 2026-08-03). Chunking caps every S-scaled
        # allocation at C: each chunk prefills at offset start_pos+off, attending the cached
        # prefix — the continue-mode machinery that already exists and is exact.
        # Requirements/notes:
        #   - kv_quant only (prefill_batch offset raises without it, by design) — else ignored.
        #   - RoPE positions are absolute via the start_pos argument; cache_lens advances
        #     per chunk inside prefill_batch exactly as continue-mode always has.
        #   - dflash context taps accumulate per chunk (TRUE as of the dflash_capture fix that
        #     removed its start_pos==0 predicate — before that, chunks 1+ captured NOTHING and
        #     the drafter saw only the first C rows: E=1.00 at depth, caught by the dead-drafter
        #     guard on the first ceiling proof). Piecewise capture cannot break losslessness —
        #     taps feed drafts only, never verify.
        #   - Chunk-boundary attention today rides the offset route (per-token fallback unless a
        #     faster exact prefix route lands — Codex design in flight). SLOW but EXACT: this
        #     driver removes the memory ceiling; the speed upgrade slots in behind it.
        # v1 SCOPE: chunking is DISABLED under the ring (kv_swa). Codex's audit found direct
        # chunk-append-then-attend on the ring is TEMPORALLY ALIASED — with W=1024 and C near W,
        # later writes in the chunk overwrite ring slots that earlier rows still need as keys.
        # Not an append-arithmetic bug (rows are byte-independent); a scheduling one. The fix is
        # the packed-transient pattern (snapshot logical tail -> append linearly -> attend
        # row-relative -> commit to ring AFTER attention) — v2 scope. Until then: chunk + kv_swa
        # would corrupt silently, so it is refused here. SWA-off is where the O(S²) ceiling
        # bites anyway; chunking there is the immediate win.
        # ── CAPACITY GUARD (#74, 2026-08-05) ─────────────────────────────────────────────
        # THE BUG THIS FIXES: the decode loop below breaks silently at `pos >= max_seq`
        # (see the loud check there). Unchunked, an over-long prompt died first in the O(S^2)
        # cudaMalloc, so the capacity bound was never reachable. NOMOS_PREFILL_CHUNK removed the
        # MEMORY ceiling but not the CAPACITY bound, so a 6032-token prompt at max_seq=4096
        # (tools/gold_env.sh:32) reached a path that assumes it cannot:
        #   - chunks wrote KV for positions up to 6031 into a cache allocated at
        #     layer_cache_cap = max_seq = 4096 (gemma4_engine.mojo:380/:586) -> ~1935 positions
        #     PAST THE END of the allocation, i.e. a device-side overrun into adjacent buffers.
        #     That is why every LATER request returned all-NaN and only a daemon restart cleared
        #     it: the corruption was never IN the KV cache, it was in what the KV cache ran into.
        #   - then the loop broke before the lm-head, so logits_out came back UNTOUCHED with
        #     rc=0 after 12.7s of real compute. Silent, HTTP 200, no log line.
        # Refuse here, before any of that happens. Loud beats corrupt.
        # (nexus, Gold; four instrumented arms to localize. Do not soften this to a warning.)
        if start_pos + num_new > self.max_seq:
            raise Error(
                String("prompt does not fit: start_pos+tokens=") + String(start_pos + num_new)
                + String(" exceeds NOMOS_MAX_SEQ=") + String(self.max_seq)
                + String(". Raise NOMOS_MAX_SEQ. NOTE: NOMOS_PREFILL_CHUNK caps prefill MEMORY,")
                + String(" it does NOT raise the cache capacity -- chunking an over-long prompt")
                + String(" overruns the KV allocation and poisons the engine (#74).")
            )
        var pf_chunk = Int(_env_float("NOMOS_PREFILL_CHUNK", 0.0))
        if pf_chunk > 0 and self.kv_swa:
            print("[prefill] NOMOS_PREFILL_CHUNK ignored under NOMOS_KV_SWA=1: chunked ring",
                  "append is temporally aliased (later writes clobber keys earlier rows need).",
                  "Packed-transient ring chunking is v2. Running unchunked.")
            pf_chunk = 0
        if pf_chunk > 0 and len(pref) > pf_chunk and self.kv_quant:
            var off = 0
            while off < len(pref):
                var end = off + pf_chunk
                if end > len(pref):
                    end = len(pref)
                var sub = List[Int]()
                for i in range(off, end):
                    sub.append(pref[i])
                # chunk_prefix=True routes this chunk's OFFSET attention to the batched prefix
                # path (route A) instead of the per-token fallback — bf16/fp32 only; int8 chunks
                # stay per-token inside the impl until route C. Explicit param: verify's routing
                # through the same impl is untouched.
                # NOMOS_CHUNK_NO_PREFIX=1 -> chunk, but route offset attention through the
                # PER-TOKEN fallback instead of the batched prefix path. This is the diagnostic
                # arm for #74: int8 already gets exactly this (it is excluded from the chunk
                # trigger at engine_prefill.mojo:132 because of a known PV-anchor defect), and
                # GB10/int8 is CLEAN while Gold/bf16 returns an untouched logits buffer and then
                # poisons the engine. This flag gives bf16 the same treatment, so "chunking is
                # broken" and "the bf16 prefix-block path is broken" become separable.
                # Measured 2026-08-05: GB10 chunked is clean in both arms (writeback ordinary),
                # so the defect tracks use_int8_attn==False, not chunking itself.
                var _no_prefix = _env_float("NOMOS_CHUNK_NO_PREFIX", 0.0) > Float32(0.5)
                self.prefill_batch(sub, start_pos + off, dflash_prefill_context,
                                   chunk_prefix=not _no_prefix)
                off = end
        else:
            self.prefill_batch(pref, start_pos, dflash_prefill_context)
        prefill_start = num_new - 1

    var last_output: Int = 0
    var saw_channel_end: Bool = False
    var response_gen_count: Int = 0
    var gen_tokens = List[Int]()
    self.rng_state = self.rng_state + 1

    var prof = _env_float("NOMOS_PROFILE", 0.0) > Float32(0.5)
    var fine = _env_float("NOMOS_PROFILE_FINE", 0.0) > Float32(0.5)   # extra per-phase syncs
    var budget_prof = cuda_budget_active()
    var t_layers: Int = 0
    var t_other: Int = 0
    var t_qkv: Int = 0      # qkv proj + kv append   (fine)
    var t_attn: Int = 0     # kv dequant + attention (fine)
    var t_omlp: Int = 0     # o-proj + MLP           (fine)
    var n_steps: Int = 0
    var _dt0: Int = 0
    var _dtL: Int = 0

    # NOMOS_CHUNK_TRACE=1 -> dump every variable that bounds this loop. #74: chunked prefill
    # returns with logits_out UNTOUCHED (nexus, Gold, sentinel-verified), and the position
    # arithmetic is identical to the unchunked branch ON PAPER -- pref = prompt[0:num_new-1],
    # prefill_start = num_new-1, so step 0 has pos == prompt_end-1 and the `continue` below
    # must NOT fire. Something breaks that equality on Gold and it is not visible from outside.
    # Print the actuals rather than reason about them.
    var _ctrace = _env_float("NOMOS_CHUNK_TRACE", 0.0) > Float32(0.5)
    if _ctrace:
        print("[chunk-trace] ENTRY prefill_start=", prefill_start, " num_new=", num_new,
              " max_tokens=", max_tokens, " start_pos=", start_pos, " prompt_end=", prompt_end,
              " prompt_skip=", prompt_skip, " continue_mode=", continue_mode,
              " => loop runs", (num_new + max_tokens) - prefill_start, "step(s)")
    for step in range(prefill_start, num_new + max_tokens):
        # BODY-TOP. The other chunk-trace print lives at :559, i.e. AFTER the full
        # 60-layer forward -- so "no step= line" localizes the exit to somewhere in
        # [body-top, :559], NOT to the loop failing to iterate. This bracket says which.
        if _ctrace:
            print("[chunk-trace] BODY-TOP step=", step, " entering layer loop")
        if prof: _dt0 = Int(perf_counter_ns())
        var token_id: Int
        if step < num_new:
            token_id = prompt[step + prompt_skip]
        else:
            token_id = last_output
        var pos = start_pos + step
        if pos >= self.max_seq:
            # UNREACHABLE for prefill after the capacity guard above; still reachable for
            # generation that runs past max_seq. It must never be silent again: this break
            # skips the lm-head AND the logits_out writeback, so a caller gets rc=0 with an
            # unwritten buffer (#74 defect 1, cost four instrumented arms to find).
            print("[engine] STOP: pos", pos, ">= max_seq", self.max_seq,
                  "-- generation truncated here. No logits produced for this step.",
                  "If you are seeing this during PREFILL, the capacity guard was bypassed.")
            break

        # ── Embed (mmap'd file) ────────────────────────────────
        var x = List[Float32](capacity=D)
        var eo = token_id * D
        for _ in range(D): x.append(0.0)
        _ = external_call["nomos_pread", c_size_t](
            self.embed_fd, x.unsafe_ptr(), c_size_t(D * 4), c_size_t(eo * 4))
        var embed_scale = sqrt(Float32(D))
        for i in range(D): x[i] *= embed_scale
        cuda_upload(self.d_x_buf, x)
        if budget_prof: cuda_budget_token_begin(self.ctx)

        # ── 60-layer forward pass ──────────────────────────────
        for layer in range(TOTAL_LAYERS):
            if _ctrace and (layer % 16 == 0 or layer == TOTAL_LAYERS - 1):
                print("[chunk-trace]   layer", layer, "ok")
            var is_full = self.layer_is_full[layer]
            var l_nh = NH
            var l_nkv = self.layer_nkv[layer]
            var l_hd = self.layer_hd[layer]
            var i4_scale_blocks = (l_hd + 31) // 32 if self.kv_int4_block32 else 1
            var l_qd = self.layer_qd[layer]
            var l_kvd = self.layer_kvd[layer]
            var l_kvg = self.layer_kvg[layer]
            var d_vw_or_zero: UInt64 = UInt64(0) if is_full else self.d_vw[layer]

            if fine:
                cuda_sync()
                _dtL = Int(perf_counter_ns())
            prepare_qkv_dev(
                self.ctx,
                self.d_x_buf, self.d_normed_buf, self.d_q_gpu, self.d_k_new_buf, self.d_v_new_buf,
                self.handle,
                self.d_in_norms[layer], self.d_q_norms[layer], self.d_k_norms[layer],
                self.d_qw[layer], self.d_kw[layer], d_vw_or_zero,
                self.d_matmul_in_bf16,
                is_full, D, l_nh, l_nkv, l_hd, l_qd, l_kvd, pos, self.d_weight_bf16_scratch,
                self.d_qw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_kw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_vw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_qw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_kw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_vw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                W4A4Scratch(self.d_w4a4_packed, self.d_w4a4_bs, self.d_w4a4_global, self.d_w4a4_cpad, self.w4a4_chunk_mpad, self.w4a4, self.d_w4a4_bs_sf, self.d_w4a4_wbs_sf),
            )
            if budget_prof: cuda_budget_mark(self.ctx, 0)
            if layer == debug_omlp_layer and debug_omlp_stage == 8:
                self.ctx.synchronize()
                if debug_omlp_out != 0:
                    cuda_memcpy(UInt64(debug_omlp_out), self.d_q_gpu, l_qd * 4, 2)
                return
            if layer == debug_omlp_layer and debug_omlp_stage == 9:
                if debug_omlp_out != 0:
                    var q_dbg_len = l_qd + l_nh * ((l_hd + 31) // 32)
                    var d_q_dbg = cuda_malloc(q_dbg_len * 4)
                    debug_q_i8_to_f32_dev(
                        self.ctx,
                        self.d_q_gpu,
                        d_q_dbg,
                        l_nh,
                        l_hd,
                    )
                    self.ctx.synchronize()
                    cuda_memcpy(UInt64(debug_omlp_out), d_q_dbg, q_dbg_len * 4, 2)
                    cuda_free(d_q_dbg)
                return
            if layer == debug_qkv_layer:
                self.ctx.synchronize()
                if debug_normed_out != 0:
                    cuda_memcpy(UInt64(debug_normed_out), self.d_normed_buf, D * 4, 2)
                if debug_k_out != 0:
                    cuda_memcpy(UInt64(debug_k_out), self.d_k_new_buf, l_kvd * 4, 2)
                if debug_v_out != 0:
                    cuda_memcpy(UInt64(debug_v_out), self.d_v_new_buf, l_kvd * 4, 2)
                return

            var cl = self.cache_lens_fp32[layer]
            if cl >= self.max_seq: break
            var ccap = self.layer_cache_cap[layer]   # #430: ring capacity (==max_seq unless SWA sliding layer)
            if self.kv_quant:
                # INT8 KV cache: quantize+scatter the new token at position cl
                # (per-(head,pos) absmax). Dequant-on-read happens before attention.
                if self.kv_int4:
                    gpu_append_quant_kv_i4(
                        self.ctx, self.d_k_new_buf, self.d_v_new_buf,
                        self.d_k_cache_i8[layer], self.d_v_cache_i8[layer],
                        self.d_k_scales[layer], self.d_v_scales[layer],
                        1, l_nkv, l_hd, l_nkv * l_hd, ccap, l_hd, cl, i4_scale_blocks,
                    )
                else:
                    gpu_append_quant_kv_i8(
                        self.ctx, self.d_k_new_buf, self.d_v_new_buf,
                        self.d_k_cache_i8[layer], self.d_v_cache_i8[layer],
                        self.d_k_scales[layer], self.d_v_scales[layer],
                        1, l_nkv, l_hd, l_nkv * l_hd, ccap, l_hd, cl,
                    )
            elif self.precision_bits <= 16:
                # Fused 1-launch append (cast+scatter) — byte-identical to the
                # 4-op append_kv_gpu_bf16_dev (same .cast[bfloat16]()), n_s=1 at
                # position cl. Kills 3 of every 4 append launches per layer.
                gpu_append_kv_batched_bf16(
                    self.ctx,
                    self.d_k_new_buf, self.d_v_new_buf,
                    self.d_k_cache[layer], self.d_v_cache[layer],
                    1, l_nkv, l_hd, l_nkv * l_hd, self.max_seq, l_hd, cl,
                )
            else:
                append_kv_gpu_dev(
                    self.d_k_cache[layer], self.d_v_cache[layer],
                    self.d_k_new_buf, self.d_v_new_buf,
                    l_nkv, l_hd, l_hd, self.max_seq, cl,
                )
            self.cache_lens_fp32[layer] = cl + 1
            if budget_prof: cuda_budget_mark(self.ctx, 1)
            var seq_len = cl + 1
            if fine:
                cuda_sync()
                t_qkv += Int(perf_counter_ns()) - _dtL
                _dtL = Int(perf_counter_ns())

            if _env_float("NOMOS_DEBUG_KV", 0.0) > 0.5 and pos == prompt_end - 1:
                # K-cache READ region (head 0, pos 0..cl) that this layer's
                # attention is about to consume. If identical call0-vs-call1
                # but residual diverges here ⇒ compute-order chaos, not data.
                var kn = seq_len * l_hd
                var kr = List[Float32](capacity=kn)
                for _ in range(kn): kr.append(0.0)
                cuda_download(kr, self.d_k_cache[layer], kn)
                var krs: Float64 = 0.0
                for kri in range(kn): krs += Float64(kr[kri]) * Float64(kr[kri])
                print("[kread] L", layer, " Ksumsq=", krs)

            # (#4) prepare_qkv_dev now writes Q straight into d_q_gpu — the old
            # synchronous D2D cuda_memcpy(d_q_gpu, d_q_buf) drained the async pipeline
            # 60x/token (one serialization point per layer). Q is read-only in both
            # attention paths, so writing the final buffer directly is bit-identical.
            # Gemma-4 sliding window: the current token (position cl) attends to
            # keys [lo, cl] for local layers; full layers see all. Two routings:
            #  - non-SWA: dequant ALL [0,seq_len) absolute into the bf16 scratch, then
            #    offset the attention base by `lo` positions (koff) + shrink to klen.
            #  - SWA (#430, sliding layers): the cache is a ring of SLIDING_WINDOW slots;
            #    dequant the window [win_start, seq_len) LINEARLY into scratch [0,deq_klen)
            #    and attention reads it from offset 0 (windowing folded into the dequant).
            var l_window = 0 if is_full else SLIDING_WINDOW
            var use_int8_attn = self.kv_quant and self.kv_int4 and self.precision_bits <= 16 and act_precision() == 8
            var kvb = 1 if use_int8_attn else (2 if self.precision_bits <= 16 else 4)
            var use_swa = self.kv_swa and not is_full
            var lo = 0
            var win_start = 0
            var deq_klen = seq_len
            if use_swa:
                win_start = seq_len - SLIDING_WINDOW
                if win_start < 0: win_start = 0
                deq_klen = seq_len - win_start          # = min(seq_len, SLIDING_WINDOW)
            else:
                if l_window > 0 and cl - l_window + 1 > 0:
                    lo = cl - l_window + 1
            var klen = deq_klen if use_swa else (seq_len - lo)
            var koff = UInt64(0) if use_swa else UInt64(lo * l_hd * kvb)
            # Deq-trim: sliding layers under SWA pack the deq scratch at deq_slide_pitch
            # (=SLIDING_WINDOW), full layers at max_seq. attention reads the deq (kv_quant)
            # at that pitch, or the raw bf16/fp32 cache (non-quant) at max_seq.
            var deq_pitch = self.max_seq if is_full else self.deq_slide_pitch
            var attn_pitch = deq_pitch if self.kv_quant else self.max_seq
            var d_k_read = self.d_k_cache[layer] if not self.kv_quant else self.d_k_deq
            var d_v_read = self.d_v_cache[layer] if not self.kv_quant else self.d_v_deq
            var d_k_scale_read = self.d_q_bf16
            var d_v_scale_read = self.d_scores_bf16
            if self.kv_quant:
                if use_int8_attn:
                    gpu_dequant_kv_i4_to_q8_layer(self.ctx, self.d_k_cache_i8[layer],
                        self.d_k_scales[layer], self.d_k_deq, d_k_scale_read, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start, i4_scale_blocks)
                    gpu_dequant_kv_i4_to_q8_layer(self.ctx, self.d_v_cache_i8[layer],
                        self.d_v_scales[layer], self.d_v_deq, d_v_scale_read, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start, i4_scale_blocks)
                elif self.kv_int4:
                    gpu_dequant_kv_i4_layer(self.ctx, self.d_k_cache_i8[layer],
                        self.d_k_scales[layer], self.d_k_deq, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start, i4_scale_blocks)
                    gpu_dequant_kv_i4_layer(self.ctx, self.d_v_cache_i8[layer],
                        self.d_v_scales[layer], self.d_v_deq, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start, i4_scale_blocks)
                else:
                    gpu_dequant_kv_i8_layer(self.ctx, self.d_k_cache_i8[layer],
                        self.d_k_scales[layer], self.d_k_deq, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start)
                    gpu_dequant_kv_i8_layer(self.ctx, self.d_v_cache_i8[layer],
                        self.d_v_scales[layer], self.d_v_deq, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start)
            if layer == debug_omlp_layer and debug_omlp_stage == 10:
                if debug_omlp_out != 0 and use_int8_attn:
                    var scale_off_dbg = UInt64(0) if use_swa else UInt64(lo * 4)
                    var kv_dbg_len = 2 * l_nkv * klen * l_hd + 2 * l_nkv * klen
                    var d_kv_dbg = cuda_malloc(kv_dbg_len * 4)
                    debug_kv_i8_view_to_f32_dev(
                        self.ctx,
                        d_k_read + koff,
                        d_v_read + koff,
                        d_k_scale_read + scale_off_dbg,
                        d_v_scale_read + scale_off_dbg,
                        d_kv_dbg,
                        l_nkv,
                        klen,
                        l_hd,
                        attn_pitch,
                    )
                    self.ctx.synchronize()
                    cuda_memcpy(UInt64(debug_omlp_out), d_kv_dbg, kv_dbg_len * 4, 2)
                    cuda_free(d_kv_dbg)
                return
            if layer == debug_omlp_layer and debug_omlp_stage == 11:
                if debug_omlp_out != 0 and self.kv_quant and self.kv_int4:
                    var source_start = win_start if use_swa else lo
                    var raw_len = 2 * l_nkv * klen * ((l_hd + 1) // 2) + 2 * l_nkv * klen
                    var d_raw_dbg = cuda_malloc(raw_len * 4)
                    debug_kv_i4_source_to_f32_dev(
                        self.ctx,
                        self.d_k_cache_i8[layer],
                        self.d_v_cache_i8[layer],
                        self.d_k_scales[layer],
                        self.d_v_scales[layer],
                        d_raw_dbg,
                        l_nkv,
                        klen,
                        l_hd,
                        ccap,
                        source_start,
                    )
                    self.ctx.synchronize()
                    cuda_memcpy(UInt64(debug_omlp_out), d_raw_dbg, raw_len * 4, 2)
                    cuda_free(d_raw_dbg)
                return
            if layer == debug_omlp_layer and debug_omlp_stage == 12:
                if debug_omlp_out != 0 and use_int8_attn:
                    var scale_off_dbg = UInt64(0) if use_swa else UInt64(lo * 4)
                    var score_len = l_nh * klen
                    var d_score_dbg = cuda_malloc(score_len * 4)
                    debug_qk_scores_int8_dev(
                        self.ctx,
                        self.d_q_gpu,
                        d_k_read + koff,
                        d_k_scale_read + scale_off_dbg,
                        d_score_dbg,
                        l_nh,
                        l_nkv,
                        l_hd,
                        l_hd,
                        attn_pitch,
                        klen,
                        l_kvg,
                    )
                    self.ctx.synchronize()
                    cuda_memcpy(UInt64(debug_omlp_out), d_score_dbg, score_len * 4, 2)
                    cuda_free(d_score_dbg)
                return
            if use_int8_attn:
                var scale_off = UInt64(0) if use_swa else UInt64(lo * 4)
                attention_gpu_int8_dev(
                    self.ctx, self.d_q_gpu,
                    d_k_read + koff, d_v_read + koff,
                    d_k_scale_read + scale_off, d_v_scale_read + scale_off,
                    self.d_scores_scratch, self.d_attn_out_scratch,
                    l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                )
            elif self.precision_bits <= 16:
                if _strict_q4(): _flag_violation(VIOL_BF16_ATTN_DECODE)
                attention_gpu_bf16_dev(
                    self.ctx, self.handle, self.d_q_gpu,
                    d_k_read + koff, d_v_read + koff,
                    self.d_scores_scratch, self.d_q_bf16, self.d_scores_bf16,
                    self.d_attn_out_scratch,
                    l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                )
            else:
                if _strict_q4(): _flag_violation(VIOL_FP32_ATTN_DECODE)
                attention_gpu_fp32_dev(
                    self.ctx,
                    self.handle,
                    self.d_q_gpu,
                    self.d_k_cache[layer] + koff, self.d_v_cache[layer] + koff,
                    self.d_scores_scratch, self.d_attn_out_scratch,
                    l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg,
                )
            if _env_float("NOMOS_DEQ_BOUNDS_CHECK", 0.0) > 0.5:
                self.ctx.synchronize()
                var deq_cuda_error = cuda_last_error()
                if deq_cuda_error != 0:
                    raise Error(
                        "deq/attention CUDA bounds check failed: layer="
                        + String(layer) + " seq_len=" + String(seq_len)
                        + " cuda_error=" + String(deq_cuda_error)
                    )
            if budget_prof: cuda_budget_mark(self.ctx, 2)

            if fine:
                cuda_sync()
                t_attn += Int(perf_counter_ns()) - _dtL
                _dtL = Int(perf_counter_ns())
            apply_output_and_mlp_dev(
                self.ctx,
                self.d_x_buf, self.d_attn_out_scratch, self.d_o_out_buf, self.d_pn_buf,
                self.d_gate_buf, self.d_up_buf, self.d_mlp_v_buf, self.d_down_buf,
                self.handle,
                self.d_ow[layer],
                self.d_post_attn_norms[layer], self.d_pre_ff_norms[layer], self.d_post_ff_norms[layer],
                self.d_gw[layer], self.d_uw[layer], self.d_dw[layer],
                self.d_matmul_in_bf16,
                self.layer_scalars[layer], D, FF, l_qd, self.d_weight_bf16_scratch,
                self.d_ow_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_gw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_uw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_dw_gs[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_ow_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_gw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_uw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                self.d_dw_ags[layer] if self.weight_nvfp4 else Float32(0.0),
                W4A4Scratch(self.d_w4a4_packed, self.d_w4a4_bs, self.d_w4a4_global, self.d_w4a4_cpad, self.w4a4_chunk_mpad, self.w4a4, self.d_w4a4_bs_sf, self.d_w4a4_wbs_sf),
                debug_omlp_stage if layer == debug_omlp_layer else -1,
                debug_omlp_out if layer == debug_omlp_layer else Int64(0),
                budget_prof,
            )
            if layer == debug_omlp_layer:
                return
            if fine:
                cuda_sync()
                t_omlp += Int(perf_counter_ns()) - _dtL

            # Drafter mid-layer tap seam: the layer-`layer` residual-after-MLP-add lives
            # in self.d_x_buf here (= the input to layer+1). Dormant in the base product tree
            # (one pointer compare/layer); armed only when a drafter sets d_eagle3_tap_out.
            # NOTE: the MTP drafter does NOT tap here — it needs the 31B's POST-final-norm
            # hidden (= normed_final / d_lmhead_in, below), per llama.cpp gemma4.cpp:405-414.
            # EAGLE config installs target layers [1,29,56] (HF hidden-state ids [2,30,57]).
            # DFlash installs [1,12,23,35,46,57]. Slots are the configured low->high order.
            # Raw residual, no norm. Read-only D2D copy -> parity-safe when dormant.
            if self.d_eagle3_tap_out != 0 and self.drafter_tap_count > 0:
                var tap_slot = self.drafter_tap_slot(layer)
                if tap_slot >= 0:
                    cuda_memcpy(self.d_eagle3_tap_out + UInt64(tap_slot * D * 4), self.d_x_buf, D * 4, 3)

        # Prefill fast-path: only the last prompt token onward needs logits.
        # The KV cache for this position is already populated by the layer
        # loop above, and a sampled token at an earlier prompt position is
        # never used — so skip the LM-head matmul (D × 262144) + sampling
        # for every non-generating prefill position. This is the dominant
        # prefill cost: a 2K-token injected prompt drops from ~60s of wasted
        # LM-head matmuls to a few seconds.
        if _ctrace:
            print("[chunk-trace] step=", step, " pos=", pos, " prompt_end-1=", prompt_end - 1,
                  " skip_lmhead=", pos < prompt_end - 1)
        if pos < prompt_end - 1:
            continue

        if prof:
            cuda_sync()
            t_layers += Int(perf_counter_ns()) - _dt0
            _dt0 = Int(perf_counter_ns())

        # ── Final norm + LM head ───────────────────────────────
        cuda_download(x, self.d_x_buf, D)
        var normed_final = List[Float32](capacity=D)
        for _ in range(D): normed_final.append(0.0)
        rmsnorm(normed_final, x, self.final_norm, D)

        var logits = List[Float32](capacity=VOCAB)
        for _ in range(VOCAB): logits.append(0.0)
        if self.w4a4:
            # The decode lm-head is M=1 and HBM-bandwidth-bound over VOCAB. Use the
            # direct NVFP4-weight/fp32-activation GEMV on every Blackwell target:
            # the vocab-sized W4A4 fused-postscale GEMM faults on sm_120a at
            # N=VOCAB, and A16 preserves the most argmax-sensitive activation.
            cuda_upload(self.d_lmhead_in, normed_final)
            if budget_prof: cuda_budget_mark(self.ctx, 10)
            # NOMOS_LMHEAD_V3=1 routes the lm-head through the v3 GEMV instead of v1. v3 is
            # documented bit-identical to v1 (same per-lane block set, same sequential
            # t=0..15 accumulation) and only swaps the branchy per-element E2M1 decode for
            # branchless SIMD — so this must stay BYTE-EXACT; divergence is a bug, not a knob.
            # Why: the CUDA-event budget measured the lm-head at 9.18 ms/token (18.3%, the
            # largest single category) achieving ~86 GB/s while our own layer GEMMs hit ~459
            # GB/s on the same card. The vocab launch is ALREADY chunked at 32768 rows, so
            # grid geometry is largely exonerated and the scalar decode loop is the suspect.
            # MEASURED on Gold GPU1, one binary / both arms / one session, serve path:
            #   v1  20.339 tok/s, lm-head 9.285 ms/token (85 GB/s)
            #   v3  21.870 tok/s, lm-head 5.745 ms/token (138 GB/s)  -38% on the kernel, +7.5% e2e
            #   12/12 byte-exact vs the NVFP4 base in BOTH arms; token-identity PASS.
            # v3 is therefore the DEFAULT. NOMOS_LMHEAD_V3=0 restores v1 as an escape hatch.
            # Not the end of it: at the layer GEMMs' own 459 GB/s the lm-head would be 1.73 ms,
            # so ~4 ms remains — but that needs a different kernel, not a decode tweak.
            if _env_float("NOMOS_LMHEAD_V3", 1.0) > 0.5:
                gpu_matmul_nvfp4_fused_v3_dev(self.ctx, self.d_lmhead_logits, self.d_lmhead_in,
                                              self.d_embed_nvfp4, self.embed_global, D, VOCAB)
            else:
                gpu_matmul_nvfp4_fused_dev(self.ctx, self.d_lmhead_logits, self.d_lmhead_in,
                                           self.d_embed_nvfp4, self.embed_global, D, VOCAB)
            _w4a4_sync_checkpoint(
                self.ctx, "lmhead direct NVFP4 GEMV", 1, VOCAB, D
            )
            if _env_float("NOMOS_DBG", 0.0) > 0.5:
                self.ctx.synchronize()
                print("HYGIENE lmhead: cudaLastError=", cuda_last_error(),
                      " | cpad=", self.d_lmhead_cpad, " embed_nvfp4=", self.d_embed_nvfp4,
                      " w4a4_packed=", self.d_w4a4_packed, " w4a4_bs=", self.d_w4a4_bs,
                      " w4a4_global=", self.d_w4a4_global, " logits=", self.d_lmhead_logits,
                      " in=", self.d_lmhead_in, " | cpad_alloc=", 128 * VOCAB * 4,
                      " cpad_need=", 16 * VOCAB * 4)
            # Perf Lever A (2026-06-27): on-device softcap->argmax over
            # d_lmhead_logits -> return the 4-byte token, skipping the ~1MB VOCAB
            # host copy + the host argmax. Greedy fast-path only (serve sets
            # fast_token_out for unconstrained temp<=0 steps; sampling/grammar keep
            # the full-logits path). pen_ids empty => no rep-penalty (serve uses none).
            if fast_token_out != 0:
                var _ft = device_decode_token(self.ctx, self.d_lmhead_logits, VOCAB,
                                              List[Int32](), Float32(1.0))
                if budget_prof:
                    cuda_budget_mark(self.ctx, 11)
                    cuda_budget_token_end(self.ctx)
                UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(fast_token_out))[0] = Int32(_ft)
                return
            # lm-head GEMM is queued on the ctx stream; the raw cuda_download is NOT a
            # DeviceContext sync, so the host read can race a slow large-N (VOCAB) GEMM
            # -> garbage logits. Sync the ctx queue before the host read.
            # 2026-06-29; race window = GEMM duration, so it bites the big lm-head, not small layers.
            self.ctx.synchronize()
            cuda_download(logits, self.d_lmhead_logits, VOCAB)
        elif act_precision() == 8 and self.d_embed_q4 != 0:
            # dp4a Q4 lm-head (2026-06-27): q8_1-quantize normed_final[D] +
            # int8 dp4a over the Q4 embed [VOCAB,D] -> ~4ms vs ~30ms fp32 (and kills
            # the 5.63GB fp32 read = quant-4-SOP compliance). The most output-sensitive
            # GEMV: gated by greedy parity-vs-reference. d_weight_bf16_scratch (231MB,
            # Q4-mode) holds the q8_1 scratch.
            cuda_upload(self.d_lmhead_in, normed_final)
            gpu_matmul_q4_dp4a_dev[4](self.ctx, self.d_lmhead_logits, self.d_lmhead_in,
                                      self.d_embed_q4, self.d_weight_bf16_scratch, D, VOCAB)
            if fast_token_out != 0:
                var _ft = device_decode_token(self.ctx, self.d_lmhead_logits, VOCAB,
                                              List[Int32](), Float32(1.0))
                UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(fast_token_out))[0] = Int32(_ft)
                return
            cuda_download(logits, self.d_lmhead_logits, VOCAB)
        else:
            gpu_matmul(self.handle, logits, normed_final, self.d_embed_lmhead, D, VOCAB)

        # ── Sampling ───────────────────────────────────────────
        final_logit_softcap(logits, VOCAB, 30.0)
        # PYTHON-HOST path (logits engine): if a caller wants the raw post-softcap
        # logits, write them out and return — sampling + grammar live in the host now.
        # No band-aids, no token append, no callback. (Python sampler owns policy.)
        if logits_out != 0:
            var _lp = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(logits_out))
            for _li in range(VOCAB):
                _lp[_li] = logits[_li]
            return
        # Sampling band-aids: DEFAULT OFF as of 2026-06-15. The forward pass is
        # HF-correct (verified: Fibonacci/Hash-map/JSON byte-identical to stock HF
        # Gemma-4 over 120-token greedy; zero loops). The legacy band-aids (forced
        # rep_penalty floor on softcapped logits, contraction fixes, same-token
        # backstop, tool-grammar mask, early-eos mask) were CORRUPTING correct
        # logits — dropping spaces/flags, flipping near-ties to Cyrillic homoglyphs,
        # breaking JSON/tool structure (the agentic-reliability bug). They are now
        # OPT-IN via NOMOS_RAW_SAMPLING=0 (escape hatch). The structural/unused-token
        # masking below stays (near-no-op on real-word tokens). See
        # docs/FORWARD_PASS_BUG_INVESTIGATION.md + the 2026-06-15 bug-hunt memory.
        var raw_sampling = _env_float("NOMOS_RAW_SAMPLING", 1.0) > Float32(0.5)
        # Force genuinely-unused vocab IDs to -inf. INLINED here (not a call
        # to mask_special_tokens) because Mojo's incremental build was
        # caching imports across edits during the 2026-05-29 debug session.
        # Gemma-4-31B vocab probed via tokenizer-svc (md5 72b1044…):
        #   0=<pad>, 1=<eos>(skip — mask_eos handles), 2=<bos>, 3=<unk>,
        #   4=<mask>, 5=[multimodal], 6..45=<unused...>, 46=<|tool>,
        #   47=<tool|>, 48=<|tool_call>, 49=<tool_call|>, 50=<|tool_response>,
        #   51=<tool_response|>, 52=<|"|>, 53..97=<unused...>, 98=<|think|>,
        #   99=<unused>, 100=<|channel>, 101=<channel|>, 102..104=<unused>,
        #   105=<|turn>, 106=<turn|>(skip), 255999/256000=sentinels,
        #   256001..262143=upper <unused...>.
        # 2026-05-30 fix: the prior `for spec_i in range(6, 106)` blanket
        # masked the 11 structural specials inside that range, blocking the
        # base model from ever emitting <|tool_call>=48 et al. Now skip those.
        logits[0] = Float32(-1e9)
        logits[2] = Float32(-1e9)
        logits[3] = Float32(-1e9)
        logits[4] = Float32(-1e9)
        logits[5] = Float32(-1e9)
        for spec_i in range(6, 106):
            if (spec_i == 46 or spec_i == 47 or spec_i == 48
                or spec_i == 49 or spec_i == 50 or spec_i == 51
                or spec_i == 52 or spec_i == 98
                or spec_i == 100 or spec_i == 101 or spec_i == 105):
                continue
            logits[spec_i] = Float32(-1e9)
        logits[255999] = Float32(-1e9)
        logits[256000] = Float32(-1e9)
        for spec_i in range(256001, VOCAB):
            logits[spec_i] = Float32(-1e9)
        if pos >= prompt_end and not raw_sampling:
            # Greedy decoding is more vulnerable to repetition loops than
            # sampling (no diversity to break out). Floor the effective
            # rep_penalty at 1.5 when we're in the greedy regime, regardless
            # of what the caller passed — observed 2026-05-29 that the serve
            # layer was passing a rep_penalty too weak to break "To write write
            # write..." style loops on certain Mojo prompts.
            var effective_rep_penalty = self.rep_penalty
            # Floor at 1.15 for greedy (down from 1.5). With apply_rep_penalty
            # now applying once-per-unique-token (HF semantics), 1.5 was
            # over-suppressing common tokens via the softcapped logit range
            # and producing tail garbage. Window also reduced from 128 to 64.
            if self.temp <= Float32(0.05) and effective_rep_penalty < Float32(1.15):
                effective_rep_penalty = Float32(1.15)
            apply_rep_penalty(logits, gen_tokens, VOCAB, effective_rep_penalty, 64)
        if not raw_sampling:
            if not saw_channel_end:
                mask_eos_if_too_early(logits, len(gen_tokens), 4)
            else:
                mask_eos_if_too_early(logits, response_gen_count, 4)
        if len(gen_tokens) > 0 and not raw_sampling:
            fix_contraction_me_bug(logits, gen_tokens[len(gen_tokens) - 1])
        if len(gen_tokens) > 1 and not raw_sampling:
            fix_contraction_t_bug(logits,
                gen_tokens[len(gen_tokens) - 1],
                gen_tokens[len(gen_tokens) - 2])
        # Same-token loop safety net: if the last 4 generated tokens are
        # all identical, we're in a degenerate loop the rep_penalty failed
        # to break. Force that token's logit to -inf so argmax/sample_top_p
        # picks something else. Hard backstop, not a substitute for proper
        # rep_penalty tuning.
        if len(gen_tokens) >= 4 and not raw_sampling:
            var t_last = gen_tokens[len(gen_tokens) - 1]
            if t_last >= 0 and t_last < VOCAB \
               and gen_tokens[len(gen_tokens) - 2] == t_last \
               and gen_tokens[len(gen_tokens) - 3] == t_last \
               and gen_tokens[len(gen_tokens) - 4] == t_last:
                logits[t_last] = Float32(-1e9)
        # Grammar-constrained tool-call decoding (Stage A): only fires while
        # inside an open <|tool_call> … <tool_call|> block; prevents malformed
        # tool-call syntax that the OpenAI-compat parser can't recover.
        if pos >= prompt_end and not raw_sampling:
            mask_tool_grammar(logits, gen_tokens, VOCAB)

        if _env_float("NOMOS_DEBUG_KV", 0.0) > 0.5 and pos == prompt_end - 1:
            # first-generated-token logit margin probe: is recall a hard switch
            # or a knife-edge decision? recall='Before'(13286) vs echo='Repeat'(55107).
            var t1i = 0
            var t1v = Float32(-1e30)
            var t2v = Float32(-1e30)
            for vi in range(VOCAB):
                var lv = logits[vi]
                if lv > t1v:
                    t2v = t1v; t1v = lv; t1i = vi
                elif lv > t2v:
                    t2v = lv
            print("[kvdbg] FIRST-TOK: recall[13286]=", logits[13286],
                  " echo[55107]=", logits[55107],
                  " gap(recall-echo)=", logits[13286] - logits[55107],
                  " | argmax=", t1i, "(", t1v, ") runnerup(", t2v, ") top1-2gap=", t1v - t2v)
            # prefix-K integrity: hash L0 head-0 prefix keys (pos 0..100) AT
            # attention time. Identical across calls ⇒ prefix clean (decay
            # downstream); drifting ⇒ prefix corrupted in place.
            var kpf = List[Float32](capacity=101 * FULL_HD)
            for _ in range(101 * FULL_HD): kpf.append(0.0)
            cuda_download(kpf, self.d_k_cache[0], 101 * FULL_HD)
            var ksum: Float64 = 0.0
            var kssq: Float64 = 0.0
            for ki in range(101 * FULL_HD):
                var kvv = Float64(kpf[ki]); ksum += kvv; kssq += kvv * kvv
            print("[kvdbg]   prefix-K[L0,h0,pos0..100] sum=", ksum, " sumsq=", kssq)

        # ── decode-loop seam ──
        # A future optional token-override hook can
        # set blend_tok>=0 to override the sampled token. -1 = no override (default).
        var blend_tok: Int = -1

        var bi: Int = 0
        if blend_tok >= 0 and blend_tok < VOCAB:
            # Overridden token from the optional hook.
            bi = blend_tok
        elif pos < prompt_end - 1 or self.temp <= Float32(0.05):
            # Prompt-echo phase, or greedy decoding when temp≈0 (avoids the
            # divide-by-temperature blowup in sample_top_p).
            bi = argmax(logits, VOCAB)
        else:
            var sr = sample_top_p(logits, VOCAB, self.temp, self.top_p, 64, self.rng_state)
            bi = sr.token_id
            self.rng_state = sr.rng_state

        last_output = bi
        if prof:
            t_other += Int(perf_counter_ns()) - _dt0
            n_steps += 1
            if n_steps % 20 == 0:
                print("[prof] decode n=", n_steps,
                      " layers_us/tok=", t_layers // n_steps // 1000,
                      " other_us/tok=", t_other // n_steps // 1000)
                if fine:
                    print("[prof-fine] us/tok  qkv+append=", t_qkv // n_steps // 1000,
                          "  kvdeq+attn=", t_attn // n_steps // 1000,
                          "  o+mlp=", t_omlp // n_steps // 1000)
        # gen_tokens / out_tokens gates use prompt_end - 1 / num_new - 1
        # because sampling first fires at pos = prompt_end - 1 (the model
        # predicting the position right after the last prompt token —
        # that's the first response token). Earlier these were off-by-one
        # high, which silently ate the first generated token from both
        # the rep-penalty/EOS history and the stream callback.
        if pos >= prompt_end - 1:
            gen_tokens.append(bi)
            if bi == 101 and not saw_channel_end:
                saw_channel_end = True
                response_gen_count = 0
            elif saw_channel_end:
                response_gen_count += 1

        if step >= num_new - 1:
            out_tokens.append(bi)

            # Per-token streaming hook — fires only during decode phase
            # (step >= num_new). The Go side registers a callback against
            # cb_id; nomos_token_cb is resolved at runtime by the dynamic
            # linker once Go's cgo glue has loaded the .so into the same
            # address space.
            if cb_id != 0:
                var stop = external_call[
                    "nomos_token_cb", Int32, Int64, Int32
                ](cb_id, Int32(bi))
                if stop != 0:
                    break

            if bi == 1 or bi == 106:
                break


def engine_attend_shared(mut self: GemmaEngine, l31b: Int, d_q: UInt64, l_nh: Int, d_out: UInt64) raises:
    """MTP shared-KV read: run the engine's EXACT int8 decode-attend for 31B layer `l31b`
    at the current cache length, writing the attended values to d_out. `d_q` must ALREADY be
    q_normed + roped by the caller (the assistant's own q_norm + split-θ rope). This helper
    does ONLY {int4->q8 dequant + window + attention_gpu_int8_dev} — it never touches d_q
    (no q_norm/rope/q_proj). Faithful replica of the decode int8-attend path (engine_decode
    250-313); the hot decode loop is left untouched for parity safety. Correctness is gated
    by the MTP accept-rate measurement (a wrong replica => garbage drafts)."""
    var is_full = self.layer_is_full[l31b]
    var l_nkv = self.layer_nkv[l31b]
    var l_hd = self.layer_hd[l31b]
    var i4_scale_blocks = (l_hd + 31) // 32 if self.kv_int4_block32 else 1
    var l_kvg = self.layer_kvg[l31b]
    var ccap = self.layer_cache_cap[l31b]
    var seq_len = self.cache_lens_fp32[l31b]          # List[Int]; current cache length
    if seq_len <= 0:
        return
    var cl = seq_len - 1
    var kvb = 1                                        # int8 path: 1 byte/elem
    var use_swa = self.kv_swa and not is_full
    var lo = 0
    var win_start = 0
    var deq_klen = seq_len
    if use_swa:
        win_start = seq_len - SLIDING_WINDOW
        if win_start < 0:
            win_start = 0
        deq_klen = seq_len - win_start
    else:
        var l_window = 0 if is_full else SLIDING_WINDOW
        if l_window > 0 and cl - l_window + 1 > 0:
            lo = cl - l_window + 1
    var klen = deq_klen if use_swa else (seq_len - lo)
    var koff = UInt64(0) if use_swa else UInt64(lo * l_hd * kvb)
    var deq_pitch = self.max_seq if is_full else self.deq_slide_pitch
    var attn_pitch = deq_pitch
    var d_k_scale_read = self.d_q_bf16
    var d_v_scale_read = self.d_scores_bf16
    gpu_dequant_kv_i4_to_q8_layer(self.ctx, self.d_k_cache_i8[l31b], self.d_k_scales[l31b],
        self.d_k_deq, d_k_scale_read, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start, i4_scale_blocks)
    gpu_dequant_kv_i4_to_q8_layer(self.ctx, self.d_v_cache_i8[l31b], self.d_v_scales[l31b],
        self.d_v_deq, d_v_scale_read, l_nkv, deq_klen, l_hd, deq_pitch, ccap, win_start, i4_scale_blocks)
    var scale_off = UInt64(0) if use_swa else UInt64(lo * 4)
    attention_gpu_int8_dev(self.ctx, d_q, self.d_k_deq + koff, self.d_v_deq + koff,
        d_k_scale_read + scale_off, d_v_scale_read + scale_off, self.d_scores_scratch, d_out,
        l_nh, l_nkv, l_hd, l_hd, attn_pitch, klen, l_kvg)
