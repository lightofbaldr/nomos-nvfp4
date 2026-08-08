#!/usr/bin/env bash
# Fair same-card llama.cpp bench on Gold GPU0 over the 12-prompt gold bar.
# Pinned to physical GPU0 by UUID so it can NEVER land on GPU1 (llama-server)
# or GPU2 (ollama). Raw completion mode (-no-cnv): the gold-bar prompts are
# already gemma-4 chat-formatted, so we must NOT re-apply a chat template.
# Greedy (--temp 0), n=96 (== PERF_NTOK), fa on, full offload (-ngl 99).
#
# NOTE: the only ~4-bit gemma GGUF on Gold is the 26B-A4B *MoE* (25.23B total,
# ~4B active) -- NOT the gemma-4-31B *dense* model our 21.7/50 numbers use.
# So this measures llama.cpp on a structurally lighter model; see report caveats.
set -euo pipefail

BIN="${LLAMA_CLI_BIN:-llama-cli}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
GPU0_UUID="${LLAMA_GPU_UUID:-}"
SP="$(cd "$(dirname "$0")" && pwd)"
PROMPTS="$SP/gold_prompts_text.json"
NTOK="${NTOK:-96}"
OUT="$SP/llamacli_goldbar_results.jsonl"
: > "$OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"

n=$(python3 -c "import json;print(len(json.load(open('$PROMPTS'))))")
echo "running $n prompts, n_gen=$NTOK, model=$(basename "$MODEL")"
for i in $(seq 0 $((n-1))); do
  pid=$(python3 -c "import json;print(json.load(open('$PROMPTS'))[$i]['id'])")
  pf="$SP/_p_$pid.txt"
  python3 -c "import json;open('$pf','w').write(json.load(open('$PROMPTS'))[$i]['text'])"
  log="$SP/_cli_$pid.log"
  "$BIN" -m "$MODEL" -ngl 99 -fa 1 --temp 0 -n "$NTOK" -no-cnv -f "$pf" \
     > "$log" 2>&1 || { echo "FAIL $pid"; continue; }
  # parse: "eval time = ... ( X tok per token, Y tokens per second)"  -> decode tok/s
  dec=$(grep -E "eval time" "$log" | grep -vi "prompt" | grep -oE "[0-9]+\.[0-9]+ tokens per second" | grep -oE "^[0-9]+\.[0-9]+" | head -1)
  pp=$(grep -E "prompt eval time" "$log" | grep -oE "[0-9]+\.[0-9]+ tokens per second" | grep -oE "^[0-9]+\.[0-9]+" | head -1)
  echo "{\"id\":\"$pid\",\"decode_tok_s\":${dec:-null},\"prefill_tok_s\":${pp:-null}}" | tee -a "$OUT"
  rm -f "$pf"
done
echo "=== mean decode tok/s ==="
python3 -c "import json;xs=[json.loads(l)['decode_tok_s'] for l in open('$OUT') if json.loads(l)['decode_tok_s']];print(round(sum(xs)/len(xs),2),'over',len(xs),'prompts')"
