#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PATH="$HOME/.pixi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/sbsa-linux/lib:$PWD:${LD_LIBRARY_PATH:-}"
export NOMOS_WEIGHT_NVFP4=0 NOMOS_PRECISION_BITS=4 NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA=0 NOMOS_MAX_SEQ=16384 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NOMOS_BATCHED_PREFILL="${NOMOS_BATCHED_PREFILL:-1}"
exec python3 pyhost/parity_reuse.py
