#!/usr/bin/env python3
"""Muse-Glimmer-30B (text tower) bf16 safetensors -> our flat fp32 .bin serve dir.

Emits the flat layout the kernel expects; quantize afterwards with the existing
tools (batch_quantize_q4.sh for the Linears, quantize_embed_nvfp4.py for embed).
Vision tower is IGNORED — this is the text-tower port only.

WHY THIS IS NOT convert_bf16_to_q4.py WITH A FLAG
-------------------------------------------------
Muse and Gemma-4 have DISJOINT weight contracts, and every difference below is
silent if you get it wrong — the model still runs and still emits fluent text:

  1. UNTIED lm_head.  Gemma-4 ties; Muse does not.  MEASURED on the real
     checkpoint: lm_head vs embed_tokens over the first 4096 rows gives
     max|diff| 3.09, cos -0.0009 -- essentially ORTHOGONAL.  Feeding embed to
     the output projection yields noise logits.  We emit them as two files.
  2. CENTERED per-layer norms.  Upstream applies `out *= (1 + w)`; our kernel
     primitive is `out[i] = inp[i] * ri * w[i]` (direct multiply, no centering).
     So the +1 lives HERE, in the data.  Applied to EXACTLY the four per-layer
     norms.  The FINAL norm is ordinary RMSNorm and must NOT be centered.
  3. self_attn.gate_proj -- an extra per-layer Linear Gemma has no analogue for
     (6656 -> 4096).  Distinct from mlp.gate_proj (6656 -> 19968); conflating
     them is silent and catastrophic.
  4. NO q_norm/k_norm tensors.  Muse's qk_norm is parameterless
     (MuseGlimmerRMSNorm(with_scale=False)), so there is nothing to convert --
     but a converter that "helpfully" emits Gemma-shaped q_norm/k_norm files
     would arm a code path Muse must not run.

The centering in (2) is a SILENT DATA TRANSFORM: the written file no longer
equals the checkpoint tensor.  That is a deliberate contract, so it is asserted
below and recorded in the manifest rather than left to a comment.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_q4_0 import quantize as q4_quantize, write_q4  # noqa: E402

# FORMAT IS BOX-SCOPED, exactly like the rest of the kernel's precision law:
#   --format q4     GB10 / unified memory (the GB10 dev box). The proven-coherent decode path,
#                   and the apples-to-apples quant against the llama.cpp bar.
#   --format nvfp4  x86 discrete Blackwell (5090 / discrete Blackwell). W4A4 shipping path,
#                   proven at 53.34 tok/s lossless on the 5090.
# We encode from META'S OWN bf16 in BOTH cases rather than repacking a third-party NVFP4
# (Inferact/RadixArk). Not dogma -- a parity argument: the golden reference is HF bf16, so
# our-kernel-on-our-quant isolates two things we control. Our-kernel-on-their-quant vs HF
# conflates three, and a divergence at layer 30 becomes unattributable. Their layouts are
# also modelopt-style (split dense file + 41 KB per-tensor config), so repacking them would
# be MORE work than encoding from bf16, not less.
# CAVEAT, unresolved: tools/convert_redhat_nvfp4.py still carries a 2026-06-29 note that our
# NVFP4 decode was incoherent ON GB10. Discrete is proven; GB10+NVFP4 is NOT, and this
# converter does not settle it. Check before trusting an nvfp4 build on the GB10 dev box.

HIDDEN = 6656
N_LAYERS = 52
VOCAB = 202048

# Per-layer Linears -> flat names.  NOTE self_attn.gate_proj: attention gate,
# NOT the MLP gate.  Both exist per layer and they are different shapes.
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "self_attn.gate_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

# THE FOUR CENTERED NORMS.  out *= (1 + w) upstream -> we pre-add 1 here.
CENTERED_NORMS = ("input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm")


def _stem(name: str) -> str:
    """Flat name for a per-layer tensor, for BOTH arities.

        layers.I.self_attn.q_proj.weight -> layers_I_self_attn_q_proj_weight   (3-part)
        layers.I.input_layernorm.weight  -> layers_I_input_layernorm_weight    (2-part)

    The naive `f"...{p[0]}_{p[1]}_{p[2]}_weight"` is correct for the Linears and
    produces `..._weight_weight` for the norms, because p[2] IS "weight" there.
    That shipped 208 misnamed norm files the loader would never find — caught by
    reading the output inventory, not by the converter's own rc=0.
    """
    p = name.replace("model.language_model.layers.", "").split(".")
    assert p[-1] == "weight", f"unexpected tensor suffix in {name!r}"
    return "layers_" + "_".join(p[:-1]) + "_weight"


def write_f32(path: str, a: np.ndarray) -> None:
    """Atomic + resumable, matching convert_bf16_to_q4.py's kill-safety."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    a.tofile(path + ".tmp")
    os.replace(path + ".tmp", path)


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) < 2:
        print(__doc__)
        sys.exit(1)
    in_dir, out_dir = pos[0], pos[1]
    fmt = "q4"
    for a in sys.argv[1:]:
        if a.startswith("--format"):
            fmt = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    if fmt not in ("q4", "nvfp4"):
        sys.exit(f"FATAL: --format must be q4 or nvfp4, got {fmt!r}")
    write_nvfp4 = None
    if fmt == "nvfp4":
        # NB: the module exports nvfp4_quantize/write_nvfp4 — there is no `quantize`.
        # write_nvfp4(path, W) encodes AND writes, so it takes the raw weight directly.
        from quantize_nvfp4 import write_nvfp4  # noqa: F811
    os.makedirs(out_dir, exist_ok=True)
    print(f"  format={fmt}  ({'GB10/Q4_0 parity path' if fmt=='q4' else 'discrete W4A4 shipping path'})")

    from safetensors import safe_open
    import torch

    idx = json.load(open(os.path.join(in_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def T(name: str) -> np.ndarray:
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(in_dir, shard), framework="pt")
        return handles[shard].get_tensor(name).to(torch.float32).numpy()

    report = {"centered_norms": [], "final_norm_centered": False,
              "untied_lm_head": None, "layers": N_LAYERS, "vocab": VOCAB,
              "hidden": HIDDEN, "note": "text tower only; vision ignored"}

    # ── per-layer ────────────────────────────────────────────────────────────
    for i in range(N_LAYERS):
        pre = f"model.language_model.layers.{i}."
        for suf in LINEARS:
            src = pre + suf + ".weight"
            if src not in idx:
                sys.exit(f"FATAL: missing {src} -- refusing to emit a partial layer")
            stem = os.path.join(out_dir, _stem(src))
            if os.path.exists(stem + "." + fmt):
                continue                                   # resumable
            # Quantize INLINE. Writing these as fp32 first would spill ~103 GB of
            # intermediate purely to delete it; the existing contract keeps only
            # norms/embed/lm_head unquantized.
            W = T(src)
            if fmt == "q4":
                s16, p4, n = q4_quantize(W)
                write_q4(stem + ".tmp", s16, p4, n)
                os.replace(stem + ".tmp.q4", stem + ".q4")     # atomic
            else:
                write_nvfp4(stem + ".tmp.nvfp4", W)
                os.replace(stem + ".tmp.nvfp4", stem + ".nvfp4")

        for nm in CENTERED_NORMS:
            src = pre + nm + ".weight"
            if src not in idx:
                sys.exit(f"FATAL: missing {src}")
            raw = T(src)
            cen = raw + 1.0
            # ASSERT the contract rather than trusting the line above. Catches a
            # skipped add, a double add, and a dtype surprise -- all of which are
            # invisible in the output file.
            d = float(np.mean(cen) - np.mean(raw))
            if abs(d - 1.0) > 1e-5:
                sys.exit(f"FATAL: centering delta {d} != 1.0 for {src}")
            write_f32(os.path.join(out_dir, _stem(src) + ".bin"), cen)
            if i == 0:
                report["centered_norms"].append(_stem(src))

        # Guard the absent-by-design tensors: if a future checkpoint DOES carry
        # q_norm/k_norm, Muse's parameterless qk_norm contract has changed and
        # silently converting them would arm the wrong path.
        for absent in ("self_attn.q_norm", "self_attn.k_norm"):
            if pre + absent + ".weight" in idx:
                sys.exit(f"FATAL: {pre}{absent} exists -- Muse should have NO q/k norm "
                         f"weights (qk_norm is parameterless). Contract changed; stop.")
        if i % 13 == 0:
            print(f"  layer {i}/{N_LAYERS}", flush=True)

    # ── embeddings + UNTIED head ─────────────────────────────────────────────
    emb = T("model.language_model.embed_tokens.weight")
    write_f32(os.path.join(out_dir, "embed_tokens_weight.bin"), emb)

    lm_name = "lm_head.weight"
    if lm_name not in idx:
        sys.exit("FATAL: no lm_head.weight -- config says tie_word_embeddings=False, "
                 "so a tied checkpoint means the contract changed. Stop.")
    lm = T(lm_name)
    # Prove they are genuinely distinct before writing two 5.4 GB files that
    # only differ if the model is really untied.
    n = min(4096, lm.shape[0])
    same = np.array_equal(lm[:n], emb[:n])
    if same:
        sys.exit("FATAL: lm_head == embed_tokens on the first rows, but config says "
                 "untied. Refusing to emit a redundant head under a false contract.")
    a, b = lm[:n].ravel(), emb[:n].ravel()
    cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
    report["untied_lm_head"] = {"cos_vs_embed_first_rows": round(cos, 6)}
    write_f32(os.path.join(out_dir, "lm_head_weight.bin"), lm)

    # ── final norm: RAW, deliberately NOT centered ───────────────────────────
    fn = T("model.language_model.norm.weight")
    write_f32(os.path.join(out_dir, "norm_weight.bin"), fn)

    report["source"] = in_dir
    json.dump(report, open(os.path.join(out_dir, "convert_manifest.json"), "w"), indent=2)
    print(f"\n  done -> {out_dir}")
    print(f"  centered (+1): {report['centered_norms']}  x{N_LAYERS} layers")
    print(f"  final norm centered: {report['final_norm_centered']}  (must be False)")
    print(f"  lm_head cos vs embed: {report['untied_lm_head']['cos_vs_embed_first_rows']}"
          f"  (near 0 => genuinely untied)")


if __name__ == "__main__":
    main()
