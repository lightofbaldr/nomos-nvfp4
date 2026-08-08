#!/usr/bin/env python3
"""Exercise the Q4 prefill, decode, and verify paths under NOMOS_STRICT_Q4."""

import ctypes
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SO = os.environ.get("SO_PATH", str(ROOT / "libnomos_kernel.so"))
WEIGHTS = os.environ.get("WEIGHTS", "")
TOK_DIR = os.environ.get("TOK_DIR", "")
VOCAB = 262144
VIOLATIONS = {
    0: "BF16_PROJ",
    1: "BF16_PROJ_BATCHED(prefill Q4->bf16)",
    2: "BF16_ATTN_DECODE",
    3: "BF16_ATTN_BATCHED(prefill/verify)",
    4: "FP32_ATTN_DECODE",
    5: "FP32_ATTN_BATCHED",
    6: "FP32_PROJ(occ4)",
}


def bind(lib, name, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def main() -> int:
    if os.environ.get("NOMOS_STRICT_Q4") != "1":
        raise SystemExit("set NOMOS_STRICT_Q4=1 so forbidden dispatches are recorded")
    if not WEIGHTS:
        raise SystemExit("set WEIGHTS to the converted Gemma-4-31B directory")
    if not TOK_DIR:
        raise SystemExit("set TOK_DIR to the Gemma-4 tokenizer directory")

    lib = ctypes.CDLL(SO)
    init = bind(lib, "nomos_init", ctypes.c_int64, [ctypes.c_char_p])
    shutdown = bind(lib, "nomos_shutdown", ctypes.c_int32, [ctypes.c_int64])
    prefill = bind(
        lib,
        "nomos_prefill",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64],
    )
    decode = bind(
        lib,
        "nomos_decode_step",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int32, ctypes.c_int64],
    )
    verify = bind(
        lib,
        "nomos_verify_fused",
        ctypes.c_int32,
        [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int64,
        ],
    )
    cache_len = bind(lib, "nomos_kv_cache_len", ctypes.c_int32, [ctypes.c_int64])
    set_len = bind(
        lib, "nomos_kv_set_len", ctypes.c_int32, [ctypes.c_int64, ctypes.c_int32]
    )
    strict_count = bind(lib, "nomos_strict_violation_count", ctypes.c_int64, [])
    strict_mask = bind(lib, "nomos_strict_violation_mask", ctypes.c_int64, [])
    strict_reset = bind(lib, "nomos_strict_reset", ctypes.c_int32, [])

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count to twenty."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens = np.ascontiguousarray(ids, dtype=np.int32)
    logits = np.empty(VOCAB, dtype=np.float32)

    weights = WEIGHTS if WEIGHTS.endswith(os.sep) else WEIGHTS + os.sep
    handle = init(weights.encode())
    if not handle:
        raise SystemExit("nomos_init failed")
    try:
        if strict_reset() != 0:
            raise RuntimeError("nomos_strict_reset failed")
        if set_len(handle, 0) != 0:
            raise RuntimeError("nomos_kv_set_len(0) failed")
        if prefill(handle, tokens.ctypes.data, len(tokens), logits.ctypes.data) != 0:
            raise RuntimeError("nomos_prefill failed")

        token = int(np.argmax(logits))
        generated = []
        for _ in range(12):
            generated.append(token)
            if decode(handle, token, logits.ctypes.data) != 0:
                raise RuntimeError("nomos_decode_step failed")
            token = int(np.argmax(logits))

        start_pos = cache_len(handle)
        verify_tokens = np.ascontiguousarray(generated[-8:], dtype=np.int32)
        verify_logits = np.empty((len(verify_tokens), VOCAB), dtype=np.float32)
        if (
            verify(
                handle,
                verify_tokens.ctypes.data,
                len(verify_tokens),
                start_pos,
                verify_logits.ctypes.data,
            )
            != 0
        ):
            raise RuntimeError("nomos_verify_fused failed")
        if set_len(handle, start_pos) != 0:
            raise RuntimeError("nomos_kv_set_len rollback failed")

        count = strict_count()
        mask = strict_mask()
        hits = [name for bit, name in VIOLATIONS.items() if mask & (1 << bit)]
        status = count == 0
        print(
            "PRECISION-LAW GUARD:",
            "GREEN (0 violations)" if status else f"RED — {count} forbidden dispatches",
        )
        for violation in hits:
            print("   VIOLATION:", violation)
        print("STATUS:", "PASS" if status else "FAIL", flush=True)
        return 0 if status else 1
    finally:
        shutdown(handle)


if __name__ == "__main__":
    raise SystemExit(main())
