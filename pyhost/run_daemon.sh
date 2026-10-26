#!/usr/bin/env bash
# Persistent engine daemon — loads the kernel ONCE, holds the GPU, serves the socket.
# Run in tmux; leave it up across serve restarts. Q4_0 product env by default.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.pixi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/sbsa-linux/lib:$PWD:${LD_LIBRARY_PATH:-}"
export NOMOS_WEIGHT_NVFP4="${NOMOS_WEIGHT_NVFP4:-0}"
export NOMOS_PRECISION_BITS="${NOMOS_PRECISION_BITS:-4}"
export NOMOS_KV_QUANT="${NOMOS_KV_QUANT:-1}"
export NOMOS_KV_INT4="${NOMOS_KV_INT4:-1}"
export NOMOS_KV_SWA="${NOMOS_KV_SWA:-0}"
# KV session store (docs/KV_SESSION_STORE_DESIGN.md): OFF unless the caller sets NOMOS_KVSTORE=1. It replaces the
# retired one-slot prefix reuse (NOMOS_KV_REUSE; #429: reused INT4 KV was not byte-identical to a fresh prefill).
export NOMOS_KVSTORE="${NOMOS_KVSTORE:-0}"
# GB10 unified memory: a pinned RAM tier frees no GPU memory there, so its budget defaults to 1 GB (the kernel's own
# default is 8 on every architecture). serve_nomos.sh sets the same.
[ "$(uname -m)" = "aarch64" ] && export NOMOS_KVSTORE_RAM_GB="${NOMOS_KVSTORE_RAM_GB:-1}"
export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-16384}"
export NOMOS_BATCHED_PREFILL="${NOMOS_BATCHED_PREFILL:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NOMOS_ENGINE_SOCKET="${NOMOS_ENGINE_SOCKET:-/tmp/nomos_engine.sock}"

exec python3 pyhost/engine_daemon.py
