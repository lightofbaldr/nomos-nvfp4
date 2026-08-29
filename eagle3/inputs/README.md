# EAGLE-3 drafter port — shared ground truth (`eagle3/inputs/`)

Ground truth for porting the EAGLE-3 drafter forward (`RedHatAI/gemma-4-31B-it-speculator.eagle3`,
Apache-2.0) to the Nomos kernel. Independent implementations can be diffed against
the gold below. **No weights here** — the self-check code runs against a separately
obtained model.

## Files
- `config.json` — the checkpoint config. `transformer_layer_config` is the architecture:
  `model_type` llama, head_dim 256, num_attention_heads 32, num_key_value_heads 16 (GQA),
  intermediate_size 21504, rope_theta 1e4, rms_norm_eps 1e-6, draft_vocab_size 32000,
  `norm_before_fc`=false, `norm_before_residual`=true, `eagle_aux_hidden_state_layer_ids` [2,30,57].
- `shapes.json` — every tensor name + shape + dtype (the weight layout; no weights). Note the fused
  widths: qkv in=10752 (2×hidden), fc in=16128 (3×hidden), lm_head [32000,5376],
  embed_tokens [262144,5376], d2t [32000].
- `eagle3_oracle_ref.npz` — **THE GOLD.** Byte-verified against the real `Eagle3DraftModel`
  (speculators 0.6.0) on FIXED synthetic taps (`torch.manual_seed(0)`). The arbiter for every port.

## gold-diff protocol
Feed the drafter `taps_flat` [16128] (= cat of the 3 aux hiddens, low→high) + `seed_token` (=1),
run `k_steps` (=3) draft steps. Diff:
- `draft_ids` [3], `target_ids` [3] — token-level (exact, modulo quant flips deep in the chain).
- per step `fc_out`[5376], `attn_out`[8192], `o_out`[5376], `hidden_out`[5376] — tensor rel-error;
  a wrong op localizes to the first diverging tensor.

The gold is bf16; a quantized kernel diffs at the quant tolerance (Q8 ~sub-1%, NVFP4 ~few %), NOT
bit-exact — a real bug blows past that.

**Derive the forward yourselves; the gold tells you where you're wrong.** It has several
wrong-but-plausible traps and is built to catch them — trust the per-tensor diff over your intuition.
