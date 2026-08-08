#!/usr/bin/env bash
# Portable Nomos launcher — ONE script for every box. Arch-detects quant + CUDA path;
# per-host weights/tokenizer/port from ~/.nomos_host_env. See docs/PLATFORM_NOTES.md.
# Usage: serve_nomos.sh {up|down|engine|serve}
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

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

export SO_PATH="$REPO/libnomos_kernel.so" WEIGHTS TOK_DIR
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
    printf -v SPEC_ENV 'NOMOS_SERVE_SPEC=%q DFLASH_DIR=%q SPEC_VB=%q' \
      "$NOMOS_SERVE_SPEC" "$DFLASH_DIR" "$SPEC_VB"
    tmux new-session -d -s nomos-engine "$SPEC_ENV exec '$REPO/pyhost/serve_nomos.sh' engine 2>&1 | tee '$REPO/daemon.log'"
    for i in $(seq 1 180); do [ -S "$NOMOS_ENGINE_SOCKET" ] && break; sleep 1; done; sleep 6
    tmux new-session -d -s nomos-serve "$SPEC_ENV exec '$REPO/pyhost/serve_nomos.sh' serve 2>&1 | tee '$REPO/serve.log'"
    for i in $(seq 1 45); do curl -sf --max-time 3 "localhost:$NOMOS_PORT/health" 2>/dev/null | grep -q '"ready":true' && { echo "READY on :$NOMOS_PORT"; exit 0; }; sleep 2; done
    echo "NOT ready — tail:"; tail -n 8 "$REPO/serve.log" ; exit 1 ;;
  *) echo "usage: serve_nomos.sh {up|down|engine|serve}"; exit 2 ;;
esac
