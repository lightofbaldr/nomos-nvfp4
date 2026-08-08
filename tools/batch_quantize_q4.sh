#!/bin/bash
# Quantize the 7 per-layer weight matrices (q/k/v/o proj + mlp gate/up/down) in a
# flat weights dir to Q4_0 (.q4, written in place next to the .bin). Embed / lm_head /
# norms / layer_scalar stay fp32 — the kernel's precision_bits==4 path only loads .q4
# for those 7 per-layer weights. Usage: batch_quantize_q4.sh <weights_dir>
set -e
WD="${1:?usage: batch_quantize_q4.sh <weights_dir>}"
Q="${Q4_QUANTIZER:-$(cd "$(dirname "$0")" && pwd)/quantize_q4_0.py}"
cd "$WD"
echo "quantizing per-layer weights in $WD -> .q4 (Q4_0) ..."
n=0; skipped=0
for f in layers_*_self_attn_q_proj_weight.bin layers_*_self_attn_k_proj_weight.bin \
         layers_*_self_attn_v_proj_weight.bin layers_*_self_attn_o_proj_weight.bin \
         layers_*_mlp_gate_proj_weight.bin layers_*_mlp_up_proj_weight.bin \
         layers_*_mlp_down_proj_weight.bin; do
  [ -f "$f" ] || continue
  out="${f%.bin}"
  if [ -f "$out.q4" ]; then skipped=$((skipped+1)); continue; fi
  python3 "$Q" "$f" "$out" >/dev/null 2>&1 || echo "  FAIL $f"
  n=$((n+1)); [ $((n % 60)) -eq 0 ] && echo "  $n quantized..."
done
echo "DONE: $n quantized, $skipped already present"
echo "total .q4:"; du -ch *.q4 2>/dev/null | tail -1
