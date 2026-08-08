#!/usr/bin/env bash
# Launch the nomos pyhost serve on the box that holds the .so.
# Env defaults to the Q4_0 product path (SWA off). Override any NOMOS_* before calling.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.pixi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/sbsa-linux/lib:$PWD:${LD_LIBRARY_PATH:-}"
export NOMOS_WEIGHT_NVFP4="${NOMOS_WEIGHT_NVFP4:-0}"
export NOMOS_PRECISION_BITS="${NOMOS_PRECISION_BITS:-4}"
export NOMOS_KV_QUANT="${NOMOS_KV_QUANT:-1}"
export NOMOS_KV_INT4="${NOMOS_KV_INT4:-1}"
export NOMOS_KV_SWA="${NOMOS_KV_SWA:-0}"
export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-16384}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NOMOS_HOST="${NOMOS_HOST:-127.0.0.1}"
export NOMOS_PORT="${NOMOS_PORT:-8090}"
# Daemon mode by default: connect to the persistent engine daemon (no in-process model load,
# no GPU touch on serve restart). Set NOMOS_ENGINE_SOCKET="" to force in-process KernelLM.
export NOMOS_ENGINE_SOCKET="${NOMOS_ENGINE_SOCKET-/tmp/nomos_engine.sock}"

exec python3 pyhost/serve.py
