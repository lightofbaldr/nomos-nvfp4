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
        "weights": _p("~/nomos_data/muse-glimmer-30b") + "/",
        "tok_dir": _p("~/nomos_data/muse-glimmer-tokenizer"),
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
        "tok_dir": _p("~/nomos_data/olmo-3.1-32b-instruct-bf16"),      # tokenizer+chat_template
        # NVFP4 weight path (mirrors the qwen deploy pattern; no GDN — olmo is standard attention).
        # Q4 attn/dp4a disabled by default; NVFP4 is the supported olmo arm here.
        "env": {
            "NOMOS_WEIGHT_NVFP4": "1", "NOMOS_W4A4": "1", "NOMOS_PRECISION_BITS": "16",
            "NOMOS_Q4_ATTN": "0", "NOMOS_Q4_DP4A": "0",
            "NOMOS_KV_QUANT": "1", "NOMOS_KV_INT4": "1", "NOMOS_KV_I4_BLOCK": "32",
        },
    },
}


def resolve(name: str) -> dict:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; known: {sorted(MODELS)}")
    return dict(MODELS[name])
