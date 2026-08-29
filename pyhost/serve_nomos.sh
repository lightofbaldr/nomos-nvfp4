#!/usr/bin/env bash
# Portable Nomos launcher — ONE script for every box. Arch-detects quant + CUDA path;
# per-host weights/tokenizer/port from ~/.nomos_host_env. See internal notes.
# Usage: serve_nomos.sh {up|down|engine|serve}
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

# --model <name> selects .so/weights/tokenizer from the registry (pyhost/models.py); the action
# (up|down|engine|serve) is the remaining positional. NOMOS_MODEL env is the non-flag equivalent.
NOMOS_MODEL="${NOMOS_MODEL:-}"
_pos=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) NOMOS_MODEL="$2"; shift 2 ;;
    --model=*) NOMOS_MODEL="${1#*=}"; shift ;;
    *) _pos+=("$1"); shift ;;
  esac
done
set -- "${_pos[@]+"${_pos[@]}"}"

# Resolve the model registry ONCE (paths + precision env). Env is applied respecting user pre-set
# (export K="${K:-V}") so it beats the arch-detected defaults below, but a user override still wins.
SO_NAME="libnomos_kernel.so"
if [ -n "$NOMOS_MODEL" ]; then
  _reg="$(python3 -c "
import sys; sys.path.insert(0,'$REPO/pyhost'); from models import resolve
m=resolve('$NOMOS_MODEL')
print('SO_NAME='+m['so']); print('MWEIGHTS='+m['weights']); print('MTOK='+m['tok_dir'])
print('MDRAFTER='+m.get('drafter',''))
for k,v in (m.get('env') or {}).items(): print('ENV '+k+' '+v)
")" || { echo "FATAL: unknown --model '$NOMOS_MODEL'"; exit 1; }
  eval "$(printf '%s\n' "$_reg" | grep -E '^(SO_NAME|MWEIGHTS|MTOK|MDRAFTER)=')"
  while read -r _k _v; do [ -n "$_k" ] && export "$_k"="${!_k:-$_v}"; done \
    < <(printf '%s\n' "$_reg" | awk '$1=="ENV"{print $2, $3}')
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ]; then
  # DGX Spark / GB10 — UNIFIED memory. Q4_0 product quant (byte-clean; runs on unified).
  CUDA_LIB="$(ls -d /usr/local/cuda*/targets/sbsa-linux/lib 2>/dev/null | sort -V | tail -1)"
  : "${NOMOS_WEIGHT_NVFP4:=0}" ; : "${NOMOS_W4A4:=0}" ; : "${NOMOS_PRECISION_BITS:=4}"
  # int8 attention path (precision-law compliant). MUST be set — unset = bf16 attention fallback
  # (slower AND a precision-law violation). The product ships int8-attn on by its OWN config.
  : "${NOMOS_Q4_ATTN:=1}" ; : "${NOMOS_Q4_DP4A:=1}"
else
  # x86_64 DISCRETE Blackwell (PRO 4000/6000, B200). NVFP4 W4A4 — Q4_0 MMU-faults on discrete.
  CUDA_LIB="/usr/local/cuda/lib64"
  : "${NOMOS_WEIGHT_NVFP4:=1}" ; : "${NOMOS_W4A4:=1}" ; : "${NOMOS_PRECISION_BITS:=16}"
  : "${CUDA_VISIBLE_DEVICES:=0}" ; export CUDA_VISIBLE_DEVICES
fi

# Per-host overrides (weights dir / tokenizer / port differ per box)
: "${WEIGHTS:=$HOME/nomos_data/gemma-4-31b}"
: "${TOK_DIR:=$HOME/models/gemma-4-31b-it}"
: "${NOMOS_PORT:=8091}"
[ -f "$HOME/.nomos_host_env" ] && source "$HOME/.nomos_host_env"

# --model paths are authoritative over the gemma defaults + host_env (registry is the per-model
# source of truth). Applied after host_env so a named model always resolves to its own weights.
if [ -n "$NOMOS_MODEL" ]; then
  WEIGHTS="$MWEIGHTS"; TOK_DIR="$MTOK"
  [ -n "$MDRAFTER" ] && DFLASH_DIR="${DFLASH_DIR:-$MDRAFTER}"
  echo "[serve] model=$NOMOS_MODEL so=$SO_NAME weights=$WEIGHTS tok=$TOK_DIR"
fi

export SO_PATH="$REPO/$SO_NAME" WEIGHTS TOK_DIR
export LD_LIBRARY_PATH="$REPO:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
export NOMOS_WEIGHT_NVFP4 NOMOS_W4A4 NOMOS_PRECISION_BITS NOMOS_Q4_ATTN NOMOS_Q4_DP4A
export NOMOS_KV_QUANT="${NOMOS_KV_QUANT:-1}" NOMOS_KV_INT4="${NOMOS_KV_INT4:-1}"
# BLOCK-32 IS A CORRECTNESS DEFAULT, NOT A TUNING KNOB (2026-08-04). This line was missing, so
# NOMOS_KV_I4_BLOCK defaulted to 0 -> block-1 -> under NVFP4/W4A4 that arm emits `//` mid-
# identifier and collapses on high-ignition prompts. Found on three boxes in one day, and it was
# reached WITHOUT ANYONE OPTING IN: KV_INT4 defaulted on above, the block size defaulted broken
# here, and the engine banner never printed granularity, so it took a code read to see.
# The engine now auto-upgrades (aeaecb2) — this makes the intent explicit at the launch layer
# too, so the two never disagree. GB10/Q4_0 is unaffected either way (int8 activations carry the
# headroom); overriding to 1 there is a measured -3.5% throughput choice, not a correctness one.
export NOMOS_KV_I4_BLOCK="${NOMOS_KV_I4_BLOCK:-32}"
export NOMOS_KV_SWA="${NOMOS_KV_SWA:-0}" NOMOS_KV_REUSE="${NOMOS_KV_REUSE:-0}"
export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-4096}" NOMOS_BATCHED_PREFILL="${NOMOS_BATCHED_PREFILL:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NOMOS_ENGINE_SOCKET="${NOMOS_ENGINE_SOCKET:-/tmp/nomos_engine.sock}"
export NOMOS_HOST="${NOMOS_HOST:-0.0.0.0}" NOMOS_PORT
export NOMOS_SERVE_SPEC="${NOMOS_SERVE_SPEC:-0}" DFLASH_DIR="${DFLASH_DIR:-}" SPEC_VB="${SPEC_VB:-7}"

case "${1:-up}" in
  engine) exec pixi run python3 pyhost/engine_daemon.py ;;
  serve)  exec pixi run python3 -m uvicorn pyhost.serve:app --host "$NOMOS_HOST" --port "$NOMOS_PORT" ;;
  down)
    tmux kill-session -t nomos-engine 2>/dev/null; tmux kill-session -t nomos-serve 2>/dev/null
    pkill -TERM -f '[p]yhost/engine_daemon.py' 2>/dev/null || true
    pkill -TERM -f '[u]vicorn pyhost.serve' 2>/dev/null || true
    for i in $(seq 1 15); do pgrep -f '[e]ngine_daemon.py' >/dev/null || break; sleep 1; done
    rm -f "$NOMOS_ENGINE_SOCKET"; echo "nomos down" ;;
  up)
    "$REPO/pyhost/serve_nomos.sh" down; sleep 1
    echo "[$(hostname) $ARCH] NVFP4=$NOMOS_WEIGHT_NVFP4 W4A4=$NOMOS_W4A4 BITS=$NOMOS_PRECISION_BITS port=$NOMOS_PORT weights=$WEIGHTS spec=$NOMOS_SERVE_SPEC vb=$SPEC_VB"
    # Forward the full run env AND the model selection to the tmux children (they re-invoke this
    # script fresh, so --model must ride along or they default back to gemma). tmux inherits the
    # server's (possibly stale) env, so pass everything the child needs explicitly.
    printf -v SPEC_ENV 'NOMOS_SERVE_SPEC=%q DFLASH_DIR=%q SPEC_VB=%q NOMOS_ENGINE_SOCKET=%q NOMOS_PORT=%q NOMOS_HOST=%q' \
      "$NOMOS_SERVE_SPEC" "$DFLASH_DIR" "$SPEC_VB" "$NOMOS_ENGINE_SOCKET" "$NOMOS_PORT" "$NOMOS_HOST"
    MODELFLAG=""; [ -n "$NOMOS_MODEL" ] && printf -v MODELFLAG -- '--model %q' "$NOMOS_MODEL"
    tmux new-session -d -s nomos-engine "$SPEC_ENV exec '$REPO/pyhost/serve_nomos.sh' $MODELFLAG engine 2>&1 | tee '$REPO/daemon.log'"
    for i in $(seq 1 180); do [ -S "$NOMOS_ENGINE_SOCKET" ] && break; sleep 1; done; sleep 6
    tmux new-session -d -s nomos-serve "$SPEC_ENV exec '$REPO/pyhost/serve_nomos.sh' $MODELFLAG serve 2>&1 | tee '$REPO/serve.log'"
    for i in $(seq 1 45); do curl -sf --max-time 3 "localhost:$NOMOS_PORT/health" 2>/dev/null | grep -q '"ready":true' && { echo "READY on :$NOMOS_PORT"; exit 0; }; sleep 2; done
    echo "NOT ready — tail:"; tail -n 8 "$REPO/serve.log" ; exit 1 ;;
  *) echo "usage: serve_nomos.sh {up|down|engine|serve}"; exit 2 ;;
esac
