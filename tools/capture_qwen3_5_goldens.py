#!/usr/bin/env python3
"""Golden capture: HF Qwen3.8-27B (qwen3_5) text-tower reference for kernel parity.

Run with the DGX SYSTEM python — /usr/bin/python3 has a CUDA torch (2.11.0+cu130) and qwen3_5
(transformers 5.12.1). The pixi env torch is CPU-only, so it CANNOT run this on GPU. Both
transformers builds carry qwen3_5 and preserve the hybrid layer_types (the harness asserts 48/16):
    /usr/bin/python3 tools/capture_qwen3_5_goldens.py \
        ~/nomos_data/qwen3_8_27b-base-bf16 ~/nomos-ref/qwen-ref/goldens
PROVENANCE: the manifest records the ACTUAL run's transformers/torch versions — those win over any
version named in this docstring (the 2026-08-16 goldens were captured on transformers 5.12.1).

THE adjudicating instrument for #81, exactly as capture_muse_goldens.py was for #79 —
the Muse port's 11 arithmetic deltas + 1 OOB + 2 converter bugs were all silent and all
found ONLY here, layer by layer, before any e2e run. A smoke test cannot rank them.

Goldens come from the bf16 BASE, NOT the NVFP4 checkpoint — using the served quant as
both kernel input and reference makes quant error and kernel bugs indistinguishable
(that ambiguity is why Muse's `step28 +0.2926` residual is still open). bf16 SEPARATES
a port bug from acceptable PTQ error.

=== TWO THINGS THIS HARNESS DOES THAT THE MUSE ONE DID NOT (Prime, cheap-now-impossible-later) ===

(1) PIN THE GDN PATH. `attn_implementation="eager"` governs only the 16 SOFTMAX layers. The
    48 GDN layers run HF's own implementation, which has a fused (fla / causal_conv1d) vs a
    pure-pytorch reference path, chosen by whether those libs import — AND chunked-scan for
    prefill vs recurrent-step for decode. For a PORTABLE reference we want the pytorch path
    (`torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`), and the manifest
    records fla/causal_conv1d availability so the golden's provenance has no hole in the 75%
    of the model we've never written.

(2) CAPTURE THE RECURRENT STATE, not just hidden states. For softmax layers hidden states
    suffice (KV derivable). For GDN they do NOT: if the state update is wrong, hidden states
    say WHICH layer diverged but not WHETHER the recurrent state or the readout is at fault —
    different bugs, different fixes, unseparable post-hoc without the reference. We paid this
    once already (Muse bf16 gone -> step28 unresolvable). HF exposes it cleanly at
    cache.layers[i].recurrent_states[0] / .conv_states[0]; captured here while the 55GB bf16
    is on disk. Saved fp16 (state is bf16 in-model; below our NVFP4 comparison noise floor).

Design decisions carried from the Muse instrument (deliberate):
  - attn_implementation="eager": SDPA dispatch varies; golden must not depend on it (16 softmax layers).
  - bf16 on GPU: fp32 won't co-fit; bar is per-layer cosine / max-|d| vs an NVFP4 kernel.
  - Kernel replays SAVED TOKEN IDS, never re-tokenizes — tokenizer disagreement excluded.
  - Manual greedy loop (not .generate): capture every decode step's last-position hidden + logits.
  - Text-only (no pixel_values -> language_model path). mRoPE STILL applies on text (mrope_section
    [11,11,10]); it does NOT collapse without an image. This capture exercises it.

The GDN scan's silent-fail deltas, quoted from modeling_qwen3_5.py GatedDeltaNet.forward (for the
implementer/verifier, NOT applied here — this harness only captures the reference):
  - use_qk_l2norm_in_kernel=True  (q,k L2-normalized INSIDE the scan)
  - g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)   (decay, fp32)
  - beta = b.sigmoid();  GQA: query/key repeat_interleave(num_v_heads//num_k_heads = 48//16 = 3)
  - final readout: self.norm(core_attn_out, z)  (gated RMSNorm, z the gate)

RECORDED, NOT ASSUMED (establish from source before diffing — do not guess):
  - hidden_states[0]: whether it includes an embedding-side norm. Qwen typically has none.
  - logits: whether qwen3_5 applies output multiplier / softcap. Qwen3 typically does NOT.
"""
import json
import os
import sys

import numpy as np
import torch


def _lib_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    import transformers
    from transformers import AutoConfig, AutoTokenizer

    print(f"transformers {transformers.__version__}  torch {torch.__version__}", flush=True)

    # Verify the artifact before loading 50GB of weights: hybrid layer_types must be present.
    cfg = AutoConfig.from_pretrained(model_dir)
    tc = getattr(cfg, "text_config", cfg)
    lt = list(getattr(tc, "layer_types", []) or [])
    n_lin = lt.count("linear_attention")
    n_full = lt.count("full_attention")
    gdn_layers = [i for i, t in enumerate(lt) if t == "linear_attention"]
    print(f"config: {type(cfg).__name__}/{cfg.model_type}  text={type(tc).__name__}  "
          f"layers={len(lt)} linear={n_lin} full={n_full}", flush=True)
    assert cfg.model_type == "qwen3_5", f"expected qwen3_5, got {cfg.model_type}"
    assert n_lin == 48 and n_full == 16, f"hybrid flattened: linear={n_lin} full={n_full} (want 48/16)"

    # (1) GDN path provenance. Pure-pytorch reference is used iff these libs are absent.
    fla_avail = _lib_available("fla")
    conv_avail = _lib_available("causal_conv1d")
    gdn_path = "pytorch-reference" if not (fla_avail or conv_avail) else "FUSED (fla/causal_conv1d present!)"
    print(f"GDN path: {gdn_path}  (fla={fla_avail}, causal_conv1d={conv_avail})", flush=True)
    if fla_avail or conv_avail:
        print("  WARNING: a fused GDN path is importable; golden will NOT be the portable pytorch "
              "reference. Uninstall fla/causal_conv1d for a reference golden, or record this "
              "deliberately.", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)

    model = None
    last_err = None
    load_kwargs = dict(dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
    for loader in ("Qwen3_5ForConditionalGeneration", "AutoModelForCausalLM",
                   "AutoModelForImageTextToText", "AutoModel"):
        try:
            cls = getattr(transformers, loader)
            try:
                model = cls.from_pretrained(model_dir, language_model_only=True, **load_kwargs)
            except TypeError:
                model = cls.from_pretrained(model_dir, **load_kwargs)
            print(f"loaded via {loader}", flush=True)
            break
        except Exception as e:  # noqa: BLE001 — try the next loader, report at the end
            last_err = e
            print(f"  loader {loader} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
    if model is None:
        sys.exit(f"FATAL: no loader worked; last error: {last_err}")
    model.eval()

    _probed = {"done": False}

    def snapshot_gdn_state(past):
        """Read per-GDN-layer recurrent + conv state out of the HF cache, fp16.
        Returns (rec[len(gdn),*rec_shape], conv[len(gdn),*conv_shape]) or (None,None) if unavailable."""
        try:
            layers = past.layers
        except AttributeError:
            return None, None
        rec_list, conv_list = [], []
        for i in gdn_layers:
            lyr = layers[i]
            rs = getattr(lyr, "recurrent_states", None)
            cs = getattr(lyr, "conv_states", None)
            if rs is None or cs is None or rs[0] is None or cs[0] is None:
                return None, None
            rec_list.append(rs[0].detach().float().cpu().numpy().astype(np.float16))
            conv_list.append(cs[0].detach().float().cpu().numpy().astype(np.float16))
        if not _probed["done"]:
            print(f"  [state probe] GDN layer {gdn_layers[0]}: recurrent {rec_list[0].shape} "
                  f"conv {conv_list[0].shape}  (fp16, {len(gdn_layers)} GDN layers/snapshot)", flush=True)
            _probed["done"] = True
        return np.stack(rec_list), np.stack(conv_list)

    # LONG-CONTEXT prompts are load-bearing (Adam): short prompts miss the paths where GDN bugs
    # live — the chunked prefill scan crosses chunk_size=64 boundaries, the recurrent state is
    # exercised at depth, mRoPE runs at high positions, and softmax full-attention spans a long
    # context. Built from real repo files (varied token distribution; reproducibility is via the
    # SAVED IDS in the manifest, not the file, so source drift can't corrupt a captured golden).
    def _read_trunc(rel, max_chars):
        p = os.path.join(REPO, rel)
        try:
            with open(p) as f:
                return f.read()[:max_chars]
        except OSError:
            return None

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROMPTS = {
        "en_short": "The capital of France is",
        "code_short": "def fibonacci(n):\n    if n <= 1:\n        return",
        "reason_short": "If a train travels 60 miles in 1.5 hours, its average speed is",
    }
    long_code = _read_trunc("tools/dflash_spec_loop.py", 8000)     # ~2K tokens
    long_prose = _read_trunc("docs/ARCHITECTURE.md", 8000)         # ~2K tokens
    _a = _read_trunc("docs/ARCHITECTURE.md", 14000)
    _b = _read_trunc("lib/model_profiles/muse.mojo", 8000)
    xlong = (_a + "\n\n" + _b) if (_a and _b) else None             # ~4-5K tokens: many GDN chunks + high mRoPE
    if long_code:  PROMPTS["long_code"] = long_code
    if long_prose: PROMPTS["long_prose"] = long_prose
    if xlong:      PROMPTS["xlong_ctx"] = xlong

    N_STEPS = 24          # decode steps captured (hidden + logits) — exercises decode AT high position
    STATE_STEPS = 8       # GDN state snapshots per prompt beyond prefill (bounds ~150MB/snapshot fp16)
    FULL_HIDDEN_MAX_S = 160  # save every position's hidden below this; sample above it

    manifest = {
        "model_dir": model_dir,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "gdn_path": gdn_path,
        "fla_available": fla_avail,
        "causal_conv1d_available": conv_avail,
        "gdn_prefill_fn": "torch_chunk_gated_delta_rule (chunk_size 64)",
        "gdn_decode_fn": "torch_recurrent_gated_delta_rule (seq_len==1)",
        "gdn_scan_notes": "use_qk_l2norm_in_kernel=True; g=-A_log.exp()*softplus(a+dt_bias) fp32; "
                          "beta=b.sigmoid(); GQA repeat_interleave 3; readout norm(out, z) gated RMSNorm",
        "gdn_state_captured": True,
        "gdn_state_dtype": "float16",
        "gdn_state_precision_note": (
            "The in-model state is bf16; this reference is bf16->float32->float16, which is NOT a "
            "lossless bf16 serialization (fp16 has more mantissa but less range). Calibrate state "
            "thresholds as fp16-reference; NEVER describe as byte-exact bf16. (Codex flag.)"),
        "gdn_layers": gdn_layers,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "n_greedy_steps": N_STEPS,
        "n_layers": len(lt),
        "layer_types": lt,
        "hidden_states_semantics": (
            "RESOLVED (Codex audit, source-derived): [0] = RAW embed_tokens output, NO embedding norm "
            "(Qwen3_5TextModel: hidden_states = inputs_embeds). [1..63] = raw outputs of decoder layers "
            "0..62. [64] = POST-FINAL-NORM model output (tie_last_hidden_states=True replaces the final "
            "collected hidden with outputs.last_hidden_state, which has self.norm applied) — NOT the raw "
            "output of decoder layer 63. So the raw layer-63 output is ABSENT here; the comparator must "
            "treat [64] as post-final-norm (kernel applies its final RMSNorm before diffing it). To "
            "isolate a layer-63-only bug from a final-norm bug, add a pre-final-norm hook (v2)."),
        "logits_semantics": (
            "RESOLVED (Codex audit): bare untied lm_head, NO output multiplier / softcap / logit_scale "
            "(none in config). tie_word_embeddings=False -> comparator MUST use the distinct lm_head "
            "weights, not embed_tokens."),
        "prompts": {},
    }

    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids.to("cuda:0")
        S = ids.shape[1]
        # Size-adaptive: full per-position hidden for short prompts; for long ones sample the
        # chunk boundaries (every 64) + first/last so the golden localizes a chunk-crossing bug
        # without saving [64layers x S x 5120] for a 4K-token prompt.
        if S <= FULL_HIDDEN_MAX_S:
            hid_pos = list(range(S))
        else:
            hid_pos = sorted(set(list(range(0, S, 64)) + [S - 1]))
        print(f"[{name}] prompt tokens: {S}  hidden positions saved: {len(hid_pos)}", flush=True)

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, output_hidden_states=True)
        n_hidden = len(out.hidden_states)
        assert n_hidden == len(lt) + 1, f"expected {len(lt)+1} hidden_states, got {n_hidden}"
        prompt_hidden = torch.stack([h[0, hid_pos] for h in out.hidden_states]).float().cpu().numpy()
        prompt_logits_last = out.logits[0, -1].float().cpu().numpy()
        past = out.past_key_values
        rec0, conv0 = snapshot_gdn_state(past)  # GDN state AFTER prefill — THE long-context signal

        step_tokens, step_logits, step_hidden = [], [], []
        step_rec, step_conv = [], []
        cur = int(torch.argmax(out.logits[0, -1]))
        for si in range(N_STEPS):
            step_tokens.append(cur)
            with torch.no_grad():
                out = model(input_ids=torch.tensor([[cur]], device="cuda:0"),
                            past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            past = out.past_key_values
            step_logits.append(out.logits[0, -1].float().cpu().numpy())
            step_hidden.append(
                torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
            if si < STATE_STEPS:  # bound state volume: first STATE_STEPS decode updates suffice to validate recurrence
                r, c = snapshot_gdn_state(past)
                if r is not None:
                    step_rec.append(r); step_conv.append(c)
            cur = int(torch.argmax(out.logits[0, -1]))

        np.savez_compressed(
            os.path.join(out_dir, f"golden_{name}.npz"),
            prompt_ids=ids[0].cpu().numpy().astype(np.int64),
            prompt_hidden=prompt_hidden.astype(np.float32),
            prompt_hidden_positions=np.array(hid_pos, dtype=np.int64),
            prompt_logits_last=prompt_logits_last.astype(np.float32),
            step_tokens=np.array(step_tokens, dtype=np.int64),
            step_logits=np.stack(step_logits).astype(np.float32),
            step_hidden=np.stack(step_hidden).astype(np.float32),
        )
        # GDN state in a separate file (bulky) so the hidden/logits golden stays light.
        state_saved = rec0 is not None
        if state_saved:
            np.savez_compressed(
                os.path.join(out_dir, f"state_{name}.npz"),
                prefill_recurrent=rec0, prefill_conv=conv0,
                step_recurrent=np.stack(step_rec) if step_rec else np.zeros(0),
                step_conv=np.stack(step_conv) if step_conv else np.zeros(0),
                gdn_layers=np.array(gdn_layers, dtype=np.int64),
            )
        manifest["prompts"][name] = {
            "text": text if S <= FULL_HIDDEN_MAX_S else (text[:200] + f"... [{len(text)} chars total]"),
            "n_prompt_tokens": S,
            "hidden_positions_saved": len(hid_pos),
            "prompt_ids": ids[0].cpu().tolist(),
            "greedy_tokens": step_tokens,
            "greedy_decoded": tok.decode(step_tokens),
            "gdn_state_file": f"state_{name}.npz" if state_saved else None,
            "gdn_state_steps_saved": len(step_rec),
        }
        print(f"[{name}] greedy: {tok.decode(step_tokens)!r}  state_saved={state_saved}", flush=True)

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
