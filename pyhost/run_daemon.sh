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
# Cross-turn KV-reuse OFF by default: reused INT4-quantized KV is NOT byte-identical to a fresh
# prefill (#429 — cache-reset/reuse not kv-quant-aware), which corrupts multi-turn agentic context
# (the model re-calls tools / degenerates — proven on the Pro 6000 e2e battery 2026-06-17). Reuse is
# a speed optimization, not correctness — full prefill each turn is correct + cheap for short agentic
# prompts. Re-enable (=1) once #429 makes reuse quant-aware.
export NOMOS_KV_REUSE="${NOMOS_KV_REUSE:-0}"
export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-16384}"
export NOMOS_BATCHED_PREFILL="${NOMOS_BATCHED_PREFILL:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NOMOS_ENGINE_SOCKET="${NOMOS_ENGINE_SOCKET:-/tmp/nomos_engine.sock}"

exec python3 pyhost/engine_daemon.py
