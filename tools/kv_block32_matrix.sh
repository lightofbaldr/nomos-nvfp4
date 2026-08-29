#!/bin/bash
# THE BLOCK-32 x READ-PATH MATRIX — one harness, every box.
#
# WHY THIS EXISTS. On 2026-08-02 NOMOS_KV_I4_BLOCK=32 measured +15.7% on the RTX PRO 4000 (PRO 4000) and
# -10% on the demo laptop (5090) at the SAME COMMIT. Two different causes, both invisible from
# one box:
#   1. A silent-corruption bug. gpu_append_quant_kv_i4 writes the block-32 scale layout
#      UNCONDITIONALLY as float16 at [nkv, cache_cap, nblocks]. Only the int8-attention reader
#      (gpu_dequant_kv_i4_to_q8_layer, selected by act_precision()==8 i.e. NOMOS_Q4_DP4A=1) could
#      read it; the bf16 reader loaded Float32 at [nkv, cache_cap], reinterpreting TWO fp16 scales
#      as ONE fp32. Garbage KV, no crash, acceptance -> EXACTLY 0.000 at every depth.
#   2. A hardware-dependent cost. Routing around (1) via DP4A costs +37% verify on the 5090 but
#      is the champion config on the RTX PRO 4000. So the correct config is NOT the same on every card.
#
# Therefore: never quote a block-32 number without naming the box AND the read path.
# This script runs the same three arms everywhere so those numbers are comparable.
#
# IT IS A 2x2, NOT THREE ARMS. Two binary factors — read path x scale granularity:
#
#                    block=1              block=32
#   bf16 (DP4A=0)      A                    B        <- B needs the 488deda dequant fix
#   int8 (DP4A=1)      D                    C        <- C is the config the discrete box measured +15.7% on
#
#   A  bf16 / block-1    baseline. On the laptop this is the current best (46.58).
#   B  bf16 / block-32   THE FIX. On a build without 488deda this arm is garbage BY CONSTRUCTION
#                        (E ~= 1.00 / survival .000) — that is the bug, not a result.
#   C  int8 / block-32   the path that always worked; where the laptop's +37% verify appeared.
#   D  int8 / block-1    the control that makes C interpretable.
#
# EVERY comparison needs its own control, and each edge answers exactly one question:
#   A->B   does block-32 pay on the CHEAP verify path?   (the value of the fix)
#   D->C   does block-32 pay on the int8 path?           (reproduces the discrete box +15.7%)
#   A->D   what does the int8 read path itself COST?     (Prime's +37% on the 5090)
#   B vs D the actual ship decision: fixed-block-32-on-bf16 vs today's best config
#
# D was missing from the first version of this script and its absence bit immediately: arm C on
# GB10 returned base decode 10.70 against a ~21.7 reference, and with no int8/block-1 control
# there was NO WAY to tell whether that was block-32's cost on this silicon or a build
# regression. Three arms could not answer it. Do not drop D to save a run.
#
# Usage:  tools/kv_block32_matrix.sh            (all three arms)
#         ARMS="A B" tools/kv_block32_matrix.sh (subset)
# Env:    SPEC_VB PERF_NTOK override the pinned defaults — keep them IDENTICAL across boxes or
#         the matrix is not comparable.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO=$PWD

# ── per-host paths. ~/.nomos_host_env is the canonical source, but it sets WEIGHTS/DFLASH_DIR
#    UNCONDITIONALLY, so sourcing it CLOBBERS anything the caller exported. That forced hand-editing
#    of host_env to run this matrix on two different boxes (both the discrete box's and the laptop's point at a
#    TUNED model), and a hand-edit you have to remember to revert is a trap of its own.
#    Precedence is now: caller env > host_env > default.
_CALLER_WEIGHTS="${WEIGHTS:-}"
_CALLER_DFLASH="${DFLASH_DIR:-}"
_CALLER_TOK="${TOK_DIR:-}"
[ -f "$HOME/.nomos_host_env" ] && . "$HOME/.nomos_host_env"
[ -n "$_CALLER_WEIGHTS" ] && WEIGHTS="$_CALLER_WEIGHTS"
[ -n "$_CALLER_DFLASH" ]  && DFLASH_DIR="$_CALLER_DFLASH"
[ -n "$_CALLER_TOK" ]     && TOK_DIR="$_CALLER_TOK"
export WEIGHTS="${WEIGHTS:-$HOME/nomos_data/gemma-4-31b/}"
# NORMALISE the trailing slash instead of relying on a default that can never fire. host_env is
# sourced FIRST, so it always sets WEIGHTS and the `${WEIGHTS:-...}` default above is unreachable —
# which left the engine's `WEIGHTS must end with '/'` assert load-bearing, and it killed all three
# arms on any box whose host_env lacks the slash. Now the assert is unreachable by construction.
# (The engine concats WEIGHTS+filename; no slash => null weight paths => Xid 31 MMU fault at 0x0.)
export WEIGHTS="${WEIGHTS%/}/"
export SO_PATH="${SO_PATH:-$REPO/libnomos_kernel.so}"
# the drafter dir moved between boxes; accept either layout rather than fail late
if [ -z "${DFLASH_DIR:-}" ]; then
  for d in "$HOME/nomos_data/dflash-gemma-4-31b-flat" "$HOME/nomos_data/dflash/dflash-gemma-4-31b-flat"; do
    [ -d "$d" ] && export DFLASH_DIR="$d/" && break
  done
fi

# ── quant per arch, per the locked rule: GB10/unified -> Q4_0, x86 discrete Blackwell -> NVFP4.
#    (Q4_0 MMU-faults on a discrete card — a Q4_0 host pointer is unmapped on separate VRAM.)
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  export NOMOS_WEIGHT_NVFP4=0 NOMOS_W4A4=0 NOMOS_PRECISION_BITS=4
  DEFCUDA=$(echo /usr/local/cuda*/targets/sbsa-linux/lib | awk '{print $1}')
  BOXQUANT="Q4_0 (GB10 unified)"
else
  export NOMOS_WEIGHT_NVFP4=1 NOMOS_W4A4=1 NOMOS_PRECISION_BITS=16
  DEFCUDA=/usr/local/cuda/lib64
  BOXQUANT="NVFP4 W4A4 (x86 discrete Blackwell)"
fi
# CUDA_LIB from ~/.nomos_host_env WINS — it is already set and correct on every box. Hardcoding
# /usr/local/cuda/lib64 broke the laptop, where CUDA lives inside the pixi env; and because this
# line OVERWRITES LD_LIBRARY_PATH rather than prepending, it could not be worked around from
# outside either. Fall back to the arch default, then to the pixi env, and say which was used.
CUDALIB="${CUDA_LIB:-$DEFCUDA}"
[ -d "$CUDALIB" ] || CUDALIB="$REPO/.pixi/envs/default/lib"
export LD_LIBRARY_PATH="$REPO:$CUDALIB:$REPO/.pixi/envs/default/lib"
export NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA=0 NOMOS_KV_REUSE=0
export NOMOS_MAX_SEQ=4096 NOMOS_BATCHED_PREFILL=1 CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NOMOS_VERIFY_MMQ_SMALL="${NOMOS_VERIFY_MMQ_SMALL:-1}"
# PIN THESE. Different VB or token count between boxes makes the matrix meaningless.
export SPEC_VB="${SPEC_VB:-5}" PERF_NTOK="${PERF_NTOK:-64}" PERF_REPEAT="${PERF_REPEAT:-1}"

PY="$REPO/.pixi/envs/default/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)

echo "════ KV BLOCK-32 x READ-PATH MATRIX ════"
echo "  box     : $(hostname) / $ARCH / $BOXQUANT"
echo "  gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  HEAD    : $(env -u LD_LIBRARY_PATH git rev-parse --short HEAD 2>/dev/null)"
# PRINT THE WEIGHTS. A row measured on a different model is the same class of invisible error as
# arm B on a build without the fix — same script, same flags, different weights, and nothing in
# the output would say so. Prime's host_env points at the TUNED model; an unmodified run there
# would silently have measured a different model than every other row.
echo "  weights : $WEIGHTS"
echo "  drafter : ${DFLASH_DIR:-<UNSET>}"
echo "  cudalib : $CUDALIB"
case "$WEIGHTS" in
  *train*|*tuned*|*sft*|*lora*|*ft-*)
    echo "  *** WARNING: WEIGHTS path looks like a TUNED model, not the base. ***"
    echo "      Cross-box rows must all use the SAME weights or the matrix is meaningless."
    ;;
esac
echo "  so      : $SO_PATH ($(stat -c%s "$SO_PATH" 2>/dev/null) bytes, $(nm -D "$SO_PATH" 2>/dev/null | grep -c ' T nomos_') syms)"
# the single most useful provenance line: is the bf16 block-32 reader even in this build?
if grep -q 'dequant_kv_i4_block32_layer_kernel' "$REPO/lib/kv_cache_quant.mojo" 2>/dev/null; then
  echo "  bf16 blk32 reader: PRESENT — arm B is a real measurement"
else
  echo "  bf16 blk32 reader: ABSENT  — arm B WILL BE GARBAGE (expect E~1.00 / survival .000)."
  echo "                              That is the bug, not a result. Do not quote arm B."
fi
echo "  pinned  : SPEC_VB=$SPEC_VB PERF_NTOK=$PERF_NTOK PERF_REPEAT=$PERF_REPEAT"
# ── GB10 CANNOT HOST ARMS A/B. Measured on the GB10 dev box, 2026-08-02, arm A (DP4A=0, BLOCK=1 — so
#    write and read agree and the block-32 bug CANNOT be present): base decode 4.17 tok/s against
#    a ~21.7 reference, and acceptance EXACTLY 0.000 at every depth, E=1.00. Something about
#    dropping dp4a on the Q4_0 path kills acceptance on its own — cause not yet isolated (q8
#    drafter GEMV wants act_precision()==8, and/or base decode itself is wrong there).
#    THE TRAP: run only "block=32, DP4A=0" on this box, see 0.000, and conclude the block-32 fix
#    failed. It would be a false negative — the BASELINE is already 0.000. This is why arm A
#    exists. On aarch64 only arm C is interpretable; the A->B isolation needs an NVFP4 box.
if [ "$ARCH" = "aarch64" ]; then
  echo ""
  echo "  *** WARNING: on GB10/Q4_0, arms A and B (NOMOS_Q4_DP4A=0) are DEGENERATE. ***"
  echo "      Measured 2026-08-02: arm A gives E=1.00 / survival .000 at BLOCK=1, where the"
  echo "      block-32 bug cannot exist. A 0.000 in arm B here is NOT evidence about block-32."
  echo "      Only arm C is interpretable on this box. Run A->B on an NVFP4 card."
fi
echo ""

run () {  # $1=label $2=DP4A $3=BLOCK $4=what
  echo "═══ ARM $1 — NOMOS_Q4_DP4A=$2 NOMOS_KV_I4_BLOCK=$3 — $4 ═══"
  # Keep the FULL output, then surface the verdicts explicitly before any truncation.
  # A `tail -N` on the filtered stream used to drop the "L1 LOSSLESS" line for higher VB, because
  # VB emits K=VB-1 depth lines and they pushed the verdict out of the window. That is not noise,
  # it is BIASED: the larger the VB, the more likely its correctness verdict silently vanished —
  # so the arms most worth trusting were the ones most likely to lose their gate, and an ABSENT
  # gate reads like a passed one to anyone scanning for the word FAIL.
  local raw="$REPO/.kv_matrix_arm_$1.out"
  NOMOS_Q4_DP4A=$2 NOMOS_KV_I4_BLOCK=$3 timeout 3600 "$PY" tools/dflash_spec_loop.py > "$raw" 2>&1
  # verdicts first, never truncated
  grep -E 'L1 LOSSLESS|L1 MISMATCH|DEAD DRAFTER|WARNING: E=|E\[tokens/cycle\]|WEIGHTED:|BASE DECODE' "$raw" | sed 's/^/   /'
  # then the rest, for context
  grep -iE 'accept|survival|draft |verify |median|IGNORED|Traceback|Error|illegal|out of memory' "$raw" \
    | tail -12 | sed 's/^/     · /'
  if ! grep -q 'L1 LOSSLESS' "$raw"; then
    echo "   *** NO LOSSLESS VERDICT IN OUTPUT — treat this arm as UNVERIFIED, not as a pass ***"
  fi
  rm -f "$raw"
  echo ""
}
for a in ${ARMS:-A B D C}; do
  case $a in
    A) run A 0 1  "bf16 read path, block-1  (baseline; the laptop's current best)" ;;
    B) run B 0 32 "bf16 read path, block-32 (THE FIX — needs 488deda)" ;;
    D) run D 1 1  "int8 read path, block-1  (control that makes C interpretable)" ;;
    C) run C 1 32 "int8 read path, block-32 (the discrete box's +15.7% config; the +37% verify suspect)" ;;
  esac
done
echo "MATRIX COMPLETE — report per-arm WEIGHTED / E[tok/cycle] / draft ms / verify ms."
echo "Compare arms WITHIN a box first; only then across boxes. A->B answers whether verify"
echo "cost depends on scale granularity ON THIS SILICON. It is not portable evidence."
