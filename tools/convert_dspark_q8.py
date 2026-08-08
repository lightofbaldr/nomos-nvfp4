#!/usr/bin/env python3
"""Convert the DSpark speculator (RedHatAI, bf16 safetensors) -> our flat .q8 layout.

DSpark = DFlash + a Markov logit-bias head + a confidence head (their config.py:
`class DSparkSpeculatorConfig(DFlashSpeculatorConfig)`, "All DFlash fields are inherited
unchanged"), warm-started from a DFlash checkpoint. So this converter is deliberately a
sibling of convert_dflash_q8.py, same .q8 writer and same fusion order.

WHAT DIFFERS FROM OUR DFLASH — this is why DSpark needs its own drafter struct, not a flag:
                     ours (DFlash)              DSpark
    taps             6  [1,12,23,35,46,57]      5  [1,17,29,47,58]
    fc               [5376, 32256]              [5376, 26880]
    Q / KV heads     64x128 / 8x128             32x256 / 16x256
    qkv fused        10240 rows                 16384 rows   (8192 q + 4096 k + 4096 v)
    MLP              10752                      21504
    draft vocab      262144                     32000  (+ d2t/t2d remap)
    extra heads      --                         markov_w1/w2, confidence_head

THE HEADS ARE NOT TRANSPLANTABLE ONTO OUR BACKBONE. markov_w2 maps into THEIR 32k draft
vocab and the bias is calibrated to THEIR lm_head's logit scale; the head was trained to
correct THEIR backbone's errors. Hence a full port, not a graft. (I originally scoped #73
as "add the head to our DFlash" — the tensor manifest says otherwise; corrected.)

.q8 format (q8_weights.mojo reader): [int64 n][int64 nb][nb fp16 scales][n int8], group=32,
symmetric scale=amax/127. Row-major [N,K]; every K here is divisible by 32.

Norms, d2t/t2d and the tiny heads stay f32 (.bin) — they are lookups/scalars, not GEMV
weights, and quantizing them buys nothing while risking the remap.
"""
import json, os, sys
import numpy as np

GROUP = 32
SRC = os.environ.get("DSPARK_SRC", os.path.expanduser("~/nomos_data/dspark-gemma-4-31b-src/model.safetensors"))
OUT = os.environ.get("DSPARK_OUT", os.path.expanduser("~/nomos_data/dspark-gemma-4-31b-flat"))
LAYERS = 5


def write_q8(path, w):
    w = np.ascontiguousarray(w, dtype=np.float32).ravel()
    n = w.size
    assert n % GROUP == 0, f"{path}: {n} % {GROUP} != 0"
    wb = w.reshape(-1, GROUP)
    amax = np.max(np.abs(wb), axis=1)
    scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(wb / scale[:, None]), -127, 127).astype(np.int8)
    with open(path, "wb") as f:
        f.write(np.int64(n).tobytes()); f.write(np.int64(scale.size).tobytes())
        f.write(scale.astype(np.float16).tobytes()); f.write(q.tobytes())
    print(f"  {os.path.basename(path):>44}  n={n:>10}  nb={scale.size:>8}")


def write_f32(path, w):
    a = np.ascontiguousarray(w, dtype=np.float32)
    a.tofile(path)
    print(f"  {os.path.basename(path):>44}  {a.shape} f32")


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC} — download it first")
    os.makedirs(OUT, exist_ok=True)
    from safetensors import safe_open
    import torch
    st = safe_open(SRC, framework="pt")
    have = set(st.keys())

    def t(name):
        return st.get_tensor(name).to(torch.float32).numpy()

    # ── backbone (same fusion order as convert_dflash_q8.py) ──────────────────
    write_q8(os.path.join(OUT, "fc_weight.q8"), t("fc.weight"))
    for i in range(LAYERS):
        p, o = f"layers.{i}.", os.path.join(OUT, f"layers_{i}_")
        write_q8(o + "self_attn_qkv_weight.q8", np.concatenate(
            [t(p + "self_attn.q_proj.weight"), t(p + "self_attn.k_proj.weight"),
             t(p + "self_attn.v_proj.weight")], axis=0))
        write_q8(o + "self_attn_o_proj_weight.q8", t(p + "self_attn.o_proj.weight"))
        write_q8(o + "mlp_gate_up_weight.q8", np.concatenate(
            [t(p + "mlp.gate_proj.weight"), t(p + "mlp.up_proj.weight")], axis=0))
        write_q8(o + "mlp_down_proj_weight.q8", t(p + "mlp.down_proj.weight"))
        for src, dst in (("input_layernorm.weight", "input_layernorm_weight"),
                         ("post_attention_layernorm.weight", "post_attention_layernorm_weight"),
                         ("self_attn.q_norm.weight", "self_attn_q_norm_weight"),
                         ("self_attn.k_norm.weight", "self_attn_k_norm_weight")):
            if p + src in have:
                write_f32(o + dst + ".bin", t(p + src))

    for src, dst in (("hidden_norm.weight", "hidden_norm_weight"), ("norm.weight", "norm_weight")):
        if src in have:
            write_f32(os.path.join(OUT, dst + ".bin"), t(src))

    # ── draft head (32000 x 5376) ────────────────────────────────────────────
    write_q8(os.path.join(OUT, "lm_head_weight.q8"), t("lm_head.weight"))

    # ── DSpark-specific: the whole reason for this port ──────────────────────
    # Markov head: bias = w1[prev_tok] @ w2.T. w1 is an INDEXED LOOKUP (67.1M, never a
    # GEMV); only w2 (8.2M) is compute per draft position. Both stay f32: w1 because it
    # is a lookup table and quantizing it would inject error into every draft position's
    # bias, w2 because 8.2M is negligible next to a ~129 ms verify.
    if "markov_head.markov_w1.weight" in have:
        write_f32(os.path.join(OUT, "markov_w1.bin"), t("markov_head.markov_w1.weight"))
        write_f32(os.path.join(OUT, "markov_w2.bin"), t("markov_head.markov_w2.weight"))
    if "confidence_head.proj.weight" in have:
        write_f32(os.path.join(OUT, "confidence_proj_weight.bin"), t("confidence_head.proj.weight"))
        write_f32(os.path.join(OUT, "confidence_proj_bias.bin"), t("confidence_head.proj.bias"))

    # ── vocab remap. d2t is INDEX+OFFSET encoded (verified: as_offset yields 32,000
    # unique target ids in [107,255970]; as-direct yields only 21,639 unique => wrong).
    # Written as resolved absolute target ids so the Mojo side needs no decode rule.
    if "d2t" in have:
        d2t = st.get_tensor("d2t").numpy().astype(np.int64)
        resolved = (np.arange(d2t.size, dtype=np.int64) + d2t)
        assert resolved.min() >= 0 and resolved.max() < 262144, "d2t decode out of range"
        assert len(np.unique(resolved)) == d2t.size, "d2t decode not injective"
        resolved.astype(np.int32).tofile(os.path.join(OUT, "d2t_i32.bin"))
        print(f"  {'d2t_i32.bin':>44}  {d2t.size} draft slots -> target ids [{resolved.min()},{resolved.max()}]")
    if "t2d" in have:
        st.get_tensor("t2d").numpy().astype(np.uint8).tofile(os.path.join(OUT, "t2d_u8.bin"))
        print(f"  {'t2d_u8.bin':>44}  262144-entry target->draft membership")

    json.dump({"source": SRC, "layers": LAYERS, "taps": [1, 17, 29, 47, 58],
               "draft_vocab": 32000, "markov_rank": 256, "note": "DSpark = DFlash + markov + confidence"},
              open(os.path.join(OUT, "convert_manifest.json"), "w"), indent=2)
    print(f"\n  done -> {OUT}")


if __name__ == "__main__":
    main()
