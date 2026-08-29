#!/usr/bin/env python3
"""Golden capture: HF Muse-Glimmer-30B text-tower reference for kernel parity.

Run with the ISOLATED venv (transformers 5.16.0.dev0), never the system python:
    ~/nomos-ref/muse-ref/.venv/bin/python tools/capture_muse_goldens.py \
        ~/nomos_data/muse-glimmer/base ~/nomos-ref/muse-ref/goldens

This is THE adjudicating instrument for #79. Eleven arithmetic deltas, one OOB
write and two converter bugs were all found BEFORE any end-to-end run — every one
silent. Nothing about the port is "working" until the kernel matches these tensors
layer by layer. (A smoke test cannot rank them: fluent output is compatible with
several of the deltas being wrong.)

Design decisions, deliberate:
  - attn_implementation="eager": SDPA dispatch varies by backend/shape; the golden
    must not depend on which fused kernel torch picked on the day it was captured.
  - bf16 on GPU: fp32 would not co-fit with anything else in 119 GB unified memory,
    and our comparison bar is per-layer cosine/max-|d| against a Q4 kernel — quant
    error dominates ULP-level reference noise.
  - The kernel side replays SAVED TOKEN IDS, never re-tokenizes text: tokenizer
    disagreement is thereby excluded from the parity surface entirely.
  - Manual greedy loop (not .generate) so every step's last-position hidden states
    and logits are captured — the DECODE path is a different code path from prefill
    in our kernel and needs its own goldens.
  - RECORDED, NOT ASSUMED: whether hidden_states[0] includes Muse's embedding
    RMSNorm depends on whether the norm lives inside the embedding module. The
    manifest records the open question; the comparison tool must establish the
    mapping from source before diffing layer 0. Do not silently assume either way.
"""
import json
import os
import sys

import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    import transformers
    from transformers import AutoTokenizer

    print(f"transformers {transformers.__version__}  torch {torch.__version__}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)

    # Prefer the specific class; fall back through the Auto registry.
    model = None
    last_err = None
    for loader in ("MuseGlimmerForConditionalGeneration",
                   "AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        try:
            cls = getattr(transformers, loader)
            model = cls.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16,
                attn_implementation="eager", device_map="cuda:0")
            print(f"loaded via {loader}", flush=True)
            break
        except Exception as e:  # noqa: BLE001 — try the next loader, report at the end
            last_err = e
    if model is None:
        sys.exit(f"FATAL: no loader worked; last error: {last_err}")
    model.eval()

    PROMPTS = {
        "en_short": "The capital of France is",
        "code_short": "def fibonacci(n):\n    if n <= 1:\n        return",
    }
    N_STEPS = 32

    manifest = {
        "model_dir": model_dir,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "n_greedy_steps": N_STEPS,
        "hidden_states_semantics": (
            "UNRESOLVED BY THIS SCRIPT: whether hidden_states[0] includes the "
            "embedding RMSNorm depends on the modeling code's module boundary. "
            "Establish from source before diffing layer 0; layers 1..52 map to "
            "decoder layer outputs regardless."),
        "logits_semantics": "as returned by HF forward (post output_multiplier + softcap)",
        "prompts": {},
    }

    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids.to("cuda:0")
        print(f"[{name}] prompt tokens: {ids.shape[1]}", flush=True)

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, output_hidden_states=True)
        # (n_layers+1) x [1, S, D] -> [n_layers+1, S, D]
        prompt_hidden = torch.stack([h[0] for h in out.hidden_states]).float().cpu().numpy()
        prompt_logits_last = out.logits[0, -1].float().cpu().numpy()
        past = out.past_key_values

        step_tokens, step_logits, step_hidden = [], [], []
        cur = int(torch.argmax(out.logits[0, -1]))
        for _ in range(N_STEPS):
            step_tokens.append(cur)
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[cur]], device="cuda:0"),
                            past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            past = out.past_key_values
            step_logits.append(out.logits[0, -1].float().cpu().numpy())
            step_hidden.append(
                torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
            cur = int(torch.argmax(out.logits[0, -1]))

        np.savez_compressed(
            os.path.join(out_dir, f"golden_{name}.npz"),
            prompt_ids=ids[0].cpu().numpy().astype(np.int64),
            prompt_hidden=prompt_hidden.astype(np.float32),
            prompt_logits_last=prompt_logits_last.astype(np.float32),
            step_tokens=np.array(step_tokens, dtype=np.int64),
            step_logits=np.stack(step_logits).astype(np.float32),
            step_hidden=np.stack(step_hidden).astype(np.float32),
        )
        manifest["prompts"][name] = {
            "text": text,
            "prompt_ids": ids[0].cpu().tolist(),
            "greedy_tokens": step_tokens,
            "greedy_decoded": tok.decode(step_tokens),
        }
        print(f"[{name}] greedy: {tok.decode(step_tokens)!r}", flush=True)

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
