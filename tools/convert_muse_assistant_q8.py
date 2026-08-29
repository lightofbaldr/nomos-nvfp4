#!/usr/bin/env python3
"""Muse-Glimmer-30B-assistant (the DFlash drafter) bf16 safetensors -> flat .q8 dir.

    python3 tools/convert_muse_assistant_q8.py \
        ~/nomos_data/muse-glimmer/assistant ~/nomos_data/muse-assistant-flat

Third member of the DFlash converter family (convert_dflash_q8.py,
convert_dspark_q8.py). SEPARATE from convert_muse_glimmer.py on purpose: the
drafter and its target have DISJOINT weight contracts, and a shared code path is
the entry-point class of bug dflash_spec_loop.py warns about in its own comment.

CONTRACT — every line below verified against modeling_muse_glimmer_assistant.py
source or the downloaded tensors, not inferred from the target:
  - Norms are STANDARD RMSNorm ("equivalent to T5LayerNorm"): direct weight
    multiply, `self.weight * normed`. NO centering. The target's (1+w) contract
    does NOT apply here — a +1 on these files would be silently wrong in the
    opposite direction of the target bug we guarded against.
  - Drafter HAS per-head q_norm/k_norm weights (128,) — the target has none.
    Mirror-image of the target's contract; both directions are asserted.
  - NO embed_tokens and NO lm_head in the checkpoint: forward() consumes
    `noise_embeds` (tokens already embedded by the TARGET's embed_tokens) and
    returns hiddens for the TARGET's lm_head to score. The drafter cannot run
    standalone; first parity check must be through the target's head.
  - Encoder = output_norm_enc(fc(tap_concat)), fc: [6656, 33280] where
    33280 = 5 taps x 6656 — verified exact against the tensor.
  - Fusion order matches the family convention: qkv = concat(q,k,v) axis=0
    (4096+1024+1024 = 6144 rows); gate_up = concat(gate,up) (39936 rows).

.q8 format shared with the family via import — ONE writer, so the format cannot
drift between converters.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dspark_q8 import write_f32, write_q8  # noqa: E402 — the family's single .q8/.bin writer

LAYERS = 5
TAPS = [1, 13, 25, 37, 49]
HIDDEN = 6656


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    from safetensors import safe_open
    import torch

    shards = [f for f in os.listdir(in_dir) if f.endswith(".safetensors")]
    assert len(shards) == 1, f"expected one shard, got {shards}"
    st = safe_open(os.path.join(in_dir, shards[0]), framework="pt")
    have = set(st.keys())
    consumed = set()

    def t(name: str) -> np.ndarray:
        consumed.add(name)
        return st.get_tensor(name).to(torch.float32).numpy()

    # ── contract guards, both directions ─────────────────────────────────────
    for absent in ("lm_head.weight", "embed_tokens.weight", "model.embed_tokens.weight"):
        if absent in have:
            sys.exit(f"FATAL: {absent} present — the head/embed-sharing contract with "
                     f"the target has changed; this converter's manifest would lie. Stop.")

    # ── encoder (tap fusion) + final norm ────────────────────────────────────
    fc = t("encoder.fc.weight")
    assert fc.shape == (HIDDEN, len(TAPS) * HIDDEN), f"encoder.fc {fc.shape}"
    write_q8(os.path.join(out_dir, "fc_weight.q8"), fc)
    write_f32(os.path.join(out_dir, "hidden_norm_weight.bin"), t("encoder.output_norm_enc.weight"))
    write_f32(os.path.join(out_dir, "norm_weight.bin"), t("norm.weight"))

    # ── per-layer ────────────────────────────────────────────────────────────
    for i in range(LAYERS):
        p, o = f"layers.{i}.", os.path.join(out_dir, f"layers_{i}_")
        write_q8(o + "self_attn_qkv_weight.q8", np.concatenate(
            [t(p + "self_attn.q_proj.weight"), t(p + "self_attn.k_proj.weight"),
             t(p + "self_attn.v_proj.weight")], axis=0))
        write_q8(o + "self_attn_o_proj_weight.q8", t(p + "self_attn.o_proj.weight"))
        write_q8(o + "mlp_gate_up_weight.q8", np.concatenate(
            [t(p + "mlp.gate_proj.weight"), t(p + "mlp.up_proj.weight")], axis=0))
        write_q8(o + "mlp_down_proj_weight.q8", t(p + "mlp.down_proj.weight"))
        # RAW norms — see contract note above. And the drafter DOES carry q/k norms.
        for src, dst in (("input_layernorm.weight", "input_layernorm_weight"),
                         ("post_attention_layernorm.weight", "post_attention_layernorm_weight"),
                         ("self_attn.q_norm.weight", "self_attn_q_norm_weight"),
                         ("self_attn.k_norm.weight", "self_attn_k_norm_weight")):
            if p + src not in have:
                sys.exit(f"FATAL: missing {p}{src} — drafter contract changed")
            write_f32(o + dst + ".bin", t(p + src))

    # ── total accounting: every checkpoint tensor consumed exactly once ─────
    leftover = have - consumed
    if leftover:
        sys.exit(f"FATAL: {len(leftover)} unconsumed tensors — the contract above missed "
                 f"something real: {sorted(leftover)[:6]}")

    json.dump({
        "source": in_dir, "layers": LAYERS, "taps": TAPS, "block_size": 16,
        "hidden": HIDDEN, "kv_heads": 8, "mask_token_id": 201818,
        "norms": "RAW (standard RMSNorm, direct multiply) — NOT centered; the "
                 "target's (1+w) contract does not apply to the drafter",
        "head_and_embed": "NONE IN CHECKPOINT by design: anchor+mask tokens are "
                          "embedded with the TARGET's embed_tokens; draft logits "
                          "= TARGET's lm_head over drafter hiddens. Cannot run "
                          "standalone; parity must go through the target head.",
        "fusion": "qkv=concat(q,k,v) axis0; gate_up=concat(gate,up) — family convention",
    }, open(os.path.join(out_dir, "convert_manifest.json"), "w"), indent=2)
    print(f"\n  done -> {out_dir}  ({len(consumed)}/{len(have)} tensors consumed)")


if __name__ == "__main__":
    main()
