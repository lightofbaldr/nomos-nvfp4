"""Model registry — makes the Nomos serve model-agnostic.

The COMPILED PROFILE (``nomos_model_id()``) is the authority for vocab / logits-buffer size:
the .so emits exactly ``PROFILE_VOCAB[model_id]`` floats per step, so the host sizes its buffer from
that, never from a tokenizer length or an HTTP model string. The launcher picks a model by NAME and
maps it to its ``.so`` / weights / tokenizer; per-box path overrides still come from ~/.nomos_host_env.

Adding a model = one entry here (+ its compiled profile in lib/model_profiles/). No serve code changes.
"""
import os


def _p(path: str) -> str:
    return os.path.expanduser(path)


# Compiled profile id -> authoritative vocab (the size the .so's logits vector actually is).
PROFILE_VOCAB = {1: 262144, 2: 202048, 3: 248320, 4: 100278}   # gemma4, muse, qwen3_5, olmo3
# Expected tokenizer bos_token_id per profile — the DEPLOY-time coherence gate (serve.py) asserts
# the loaded tokenizer's BOS matches this. Verified against each tokenizer 2026-09-17. A profile
# with no entry is not armed (serve warns). This exists because olmo3 shipped TARGET_BOS_ID=1 while
# id 1 is the literal '"' in olmo's tokenizer (real BOS 100257) — the quote-runaway. None = the model
# has no BOS (qwen). muse omitted: its tok_dir is not on the bar host, so its BOS is unverified.
PROFILE_BOS = {1: 2, 3: None, 4: 100257}       # gemma4 <bos>, qwen3_5 none, olmo3 <|endoftext|>
PROFILE_NAME = {1: "gemma4", 2: "muse", 3: "qwen3_5", 4: "olmo3"}

# name -> serve config. Relative `.so` is resolved against the repo; paths overridable per-box.
MODELS = {
    "gemma4": {
        "model_id": 1, "so": "libnomos_kernel-gemma4.so",
        "weights": _p("~/nomos_data/gemma-4-31b") + "/",
        "tok_dir": _p("~/models/gemma-4-31b-it"),
    },
    "muse": {
        "model_id": 2, "so": "libnomos_kernel-muse.so",
        "weights": _p("~/nomos_data/muse-glimmer-flat-nvfp4") + "/",  # -30b was stale (Prime); this is the real dir
        "tok_dir": _p("~/nomos_data/muse-glimmer-tokenizer"),         # NOTE: absent on GB10 — unverified, muse not served
    },
    "qwen3_5": {
        "model_id": 3, "so": "libnomos_kernel-qwen3_5.so",
        "weights": _p("~/nomos_data/qwen3_8_27b-deploy-nvfp4") + "/",
        "tok_dir": _p("~/nomos_data/qwen3_8_27b-base-bf16"),           # tokenizer+chat_template live here
        "drafter": _p("~/nomos_data/qwen3_8_27b-dspark-blobs") + "/",  # DSpark spec-decode (optional)
        # Deploy precision env (NVFP4 all the way down incl. GDN — the deploy default, Adam 2026-08-25).
        # Overrides the launcher's arch-detected Gemma-Q4_0 defaults; user pre-set env still wins.
        "env": {
            "NOMOS_WEIGHT_NVFP4": "1", "NOMOS_W4A4": "1", "NOMOS_PRECISION_BITS": "16",
            "NOMOS_GDN_NVFP4": "1", "NOMOS_Q4_ATTN": "0", "NOMOS_Q4_DP4A": "0",
            "NOMOS_KV_QUANT": "1", "NOMOS_KV_INT4": "1", "NOMOS_KV_I4_BLOCK": "32",
        },
    },
    "olmo3": {
        "model_id": 4, "so": "libnomos_kernel-olmo3.so",
        "weights": _p("~/nomos_data/olmo-3.1-32b-instruct-nvfp4") + "/",
        "tok_dir": _p("~/nomos_data/olmo-3.1-32b-instruct-tok"),       # fleet-standard tokenizer dir name
        # (resolves on deploy targets AND GB10; GB10's -tok symlinks the local -bf16 dir). Was -bf16,
        # which existed only on GB10 — a path validated against an artifact the deploy serve never loads,
        # the same class as the BOS bug, one level up (Prime caught it 2026-09-17).
        # NVFP4 weight path (mirrors the qwen deploy pattern; no GDN — olmo is standard attention).
        # The Q4 arm is not runnable until the olmo q4 dir gets its missing lm_head_weight.q4.
        # Serve-default repetition penalty 1.15 (Adam, 2026-09-03): at vendor sampling
        # (temp .6 / top_p .95, no penalty) olmo loops on ~60% of title-bearing prompts
        # (89/141 sampled draws); 1.15 measured loop-free on the direct path at the same
        # sampling. Per-request repetition_penalty overrides it.
        # BOS-BUG HISTORY (Kvasir+Prime+Codex, 2026-09-17 — FIXED, see end): this penalty MASKED a
        # KERNEL bug, not model behavior. ROOT CAUSE: lib/model_profiles/olmo3.mojo TARGET_BOS_ID=1, but id 1
        # is the literal '"' in olmo's tokenizer (verified; real BOS = <|endoftext|> = 100257).
        # engine_decode.mojo's continue-mode filter drops prompt[0]==TARGET_BOS_ID, so the serve's
        # host decode path (nomos_prefill + N*nomos_decode_step) eats standalone quote tokens ->
        # position shift; when a GENERATED token is '"', num_new->0, the decode loop never runs,
        # decode_step returns 0, stale logits are re-emitted to the length cap = the "quote-runaway".
        # (nomos_generate is safe — it never re-enters the filter.) rep 1.0 = 3/6 runaway, 1.15 +
        # same-token backstop = 0/24 was the workaround shape. FIXED 2026-09-17: TARGET_BOS_ID -> 100257
        # (Adam-approved; re-gate PASSED — the quote prompt now yields '"the sea remembers"', no runaway).
        # Prime's engine_decode num_new>1 strip guard + nomos_decode_step cache-advance rc-2 (kernel
        # 539f330) close the residual step(EOS) hole. The masked bug is RESOLVED, so the 1.15 default now
        # stands as a NORMAL repetition default (independently addresses ~20% non-quote loops, 09-03) —
        # re-litigable as a tuning choice, not a bug mask.
        #
        # There was never a RoPE or attention arithmetic fault. The "~20 logit units of divergence"
        # reported earlier was STALE LOGITS, which is also why it was quant-independent. Measured:
        # position drift of exactly -1 per stripped token, dose-response 0 / -1 / -2 against a
        # prediction made IN ADVANCE, |(c,s)| = 1.000000 throughout (a pure rotation, so both RoPE
        # kernels were innocent). See the BOS-strip fix record.
        #
        # LOADED-BINARY ATTESTED (kernel 0606837, .so sha256 3c7bd468...a51a68), via the new
        # no-handle exports rather than inferred from source:
        #     nomos_target_bos_id -> 100257   eos_id_0 -> 100257   eos_id_1 -> -1 (no second EOS)
        # That distinction matters: the generated config and the build validator are SOURCE
        # evidence, and a correct profile beside a stale .so passes both while still serving the bug.
        #
        # RE-LITIGATING THE 1.15 DEFAULT REQUIRES A QUOTE-BEARING TEST. Quote-free prompts cannot
        # fail on this bug and will pass happily on a broken build — which is precisely how it
        # survived for weeks in a harness whose every generic prompt was quote-free.
        "env": {
            "NOMOS_WEIGHT_NVFP4": "1", "NOMOS_W4A4": "1", "NOMOS_PRECISION_BITS": "16",
            "NOMOS_Q4_ATTN": "0", "NOMOS_Q4_DP4A": "0",
            "NOMOS_KV_QUANT": "1", "NOMOS_KV_INT4": "1", "NOMOS_KV_I4_BLOCK": "32",
            "NOMOS_LOGITS_ROWS": "1", "NOMOS_MAX_SEQ": "2048",   # 24GB 5090: reclaim the 6.8GB spec-verify buffer
            "NOMOS_SERVE_REP_PENALTY": "1.15",
        },
    },
}


def resolve(name: str) -> dict:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; known: {sorted(MODELS)}")
    return dict(MODELS[name])
