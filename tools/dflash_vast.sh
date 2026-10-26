#!/bin/bash
# Vast RTX PRO 6000 (dedicated single-GPU box, sm_120) DFlash spec-decode runner.
# Vast variant of dflash_gold.sh:
#   - DROPS the Gold GPU0 compute-app guard (dedicated box, use the whole GPU).
#   - cd's to the repo relative to this script (no hardcoded Gold path).
#   - runs via `pixi run python` (no Gold venv); harness only needs ctypes+numpy.
#   tools/dflash_vast.sh <script.py> [args...]
# Env overrides: SO_PATH, PERF_NTOK, PERF_REPEAT, SPEC_VB, BASE_IDS, IDS_OUT, DFLASH_DIR.
set -e
# WEIGHTS GOTCHA (silent-corruption trap): the NVFP4 W4A4 deploy set must include the
# fp32 embed_tokens_weight.bin (5.6GB) — gemma4_engine.mojo mmaps it for INPUT token
# embedding (embed_fd); only the LM-HEAD readout uses embed_tokens_weight.nvfp4. If the
# .bin is absent the input embeds to ZERO -> all-zero logits -> every token id = 0, which
# masquerades as 100% draft acceptance + L1-self-lossless PASS (both base+spec degenerate)
# while 0/12 vs a real reference. Pulling only *.nvfp4 + small *.bin is NOT enough.
export PATH="$HOME/.pixi/bin:$PATH"
cd "$(cd "$(dirname "$0")/.." && pwd)"

export CUDA_VISIBLE_DEVICES=0   # single-GPU box -> device 0 is the only device
# Same NVFP4/W4A4 fast-verify base-decode env as dflash_gold.sh, WITHOUT NOMOS_EAGLE.
export NOMOS_WEIGHT_NVFP4=1 NOMOS_W4A4=1 NOMOS_PRECISION_BITS=16 \
       NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_SWA="${NOMOS_KV_SWA:-0}" NOMOS_KV_REUSE=0 \
       NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-4096}" NOMOS_BATCHED_PREFILL=1 CUBLAS_WORKSPACE_CONFIG=":4096:8"
export LD_LIBRARY_PATH="$PWD:/usr/local/cuda/lib64:$PWD/.pixi/envs/default/lib"
export WEIGHTS="${WEIGHTS:-$HOME/nomos_data/gemma-4-31b/}"
export DFLASH_DIR="${DFLASH_DIR:-$HOME/nomos_data/dflash/dflash-gemma-4-31b-flat/}"
export SO_PATH="${SO_PATH:-$PWD/libnomos_kernel.so}"
# See tools/dflash_gold.sh for the full rationale: block-32 INT4 KV scales are an ACCEPTANCE
# lever (deep-layer hidden cosine 0.60 -> 0.85), measured +15.7% spec on Gold, 12/12 lossless.
# Both read paths take scale_blocks as of 89da404 (before that, block=32 without NOMOS_Q4_DP4A=1
# silently corrupted KV -> acceptance exactly 0.000). Defaulted ON here because this runner is
# x86 discrete NVFP4 and every measured cell on that hardware is positive: 5090 laptop bf16
# +9.5% / int8 +8.7% (full 2x2, 12/12 lossless), Gold int8 +15.7%, verify flat on both paths.
# DO NOT copy this default to a GB10 runner — the same edge is -3.5% there. See dflash_gold.sh.
export NOMOS_KV_I4_BLOCK="${NOMOS_KV_I4_BLOCK:-32}"
export NOMOS_VERIFY_MMQ_SMALL="${NOMOS_VERIFY_MMQ_SMALL:-1}"
# SPEC_VB=7 — measured, not inherited. Gold PRO 4000, champion config (bf16 + block-32),
# base weights, all arms 12/12 LOSSLESS: VB5 43.23 / VB6 44.57 / VB7 45.03 / VB8 44.18,
# then a CLIFF: VB9 30.40, VB12 25.99. E rises monotonically (2.76->3.12) across the whole
# range, so acceptance is NOT the limiter past 8 — verify cost is, and it turns on DISCRETELY.
# That cliff is NOMOS_VERIFY_MMQ_SMALL's S=5..8 tensor-core window ending; VB=8 is the last
# value inside it. 7 and 8 are statistically tied (the anchor's own run-to-run spread is 2.3%,
# two orders of magnitude above their 0.02-1.9% gap), so 7 is chosen for MARGIN: it keeps a
# full step from a cliff whose far side is -32%, and falling off it is silent — the throughput
# halves and every gate still passes. Re-tune only if that MMQ window changes.
export SPEC_VB="${SPEC_VB:-7}"
export PERF_NTOK="${PERF_NTOK:-96}"
SCRIPT="$1"; shift || true
echo "== dflash_vast: SO=$SO_PATH DFLASH_DIR=$DFLASH_DIR script=$SCRIPT VB=$SPEC_VB NTOK=$PERF_NTOK =="
nvidia-smi -i 0 --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
exec pixi run python "$SCRIPT" "$@"
