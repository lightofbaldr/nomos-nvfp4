# tools/gold_env.sh — DISCRETE BLACKWELL (Gold PRO 4000 / PRO 6000 / 5090 laptop) kernel env.
# SOURCE this, do not execute it:    source tools/gold_env.sh
#
# VERIFIED 2026-08-04 @ HEAD 4eac19d on Gold: 12/12 LOSSLESS, 45.22 tok/s, E 3.01, verify ~60 ms.
# See the README for the quickstart.
#
# WHY THIS FILE EXISTS: `./nomos env` is WRONG for discrete cards. Its arch-common BASE_ENV sets
# NOMOS_VERIFY_BLOCK_ATTN=1, which is NOT byte-exact on NVFP4 (0/12 lossless), and it never sets
# NOMOS_KV_I4_BLOCK. The broken config runs ~1.3% FASTER, so a throughput check will not catch it.
# Task #65. On GB10 `./nomos env` is correct and this file must NOT be used.

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "ERROR: source this file, do not execute it:  source tools/gold_env.sh" >&2
  exit 1
fi

_gold_arch="$(uname -m)"
if [ "$_gold_arch" != "x86_64" ]; then
  echo "REFUSING: this is the DISCRETE (NVFP4) env and you are on $_gold_arch." >&2
  echo "  On GB10/Spark use:  eval \"\$(./nomos env | grep -oE '^[[:space:]]+NOMOS_[A-Z0-9_]+=.*' | sed 's/^[[:space:]]*/export /')\"" >&2
  return 1
fi

# Start from a clean slate so a previously-sourced launcher env cannot leak the lossy flags in.
unset NOMOS_VERIFY_BLOCK_ATTN NOMOS_VERIFY_DECODE_ORDER_ATTN NOMOS_VERIFY_DECODE_ORDER_MULTIROW_ATTN
unset NOMOS_Q4_DP4A          # off IS the champion on discrete — it is a bad trade here

export NOMOS_WEIGHT_NVFP4=1 NOMOS_W4A4=1 NOMOS_PRECISION_BITS=16
export NOMOS_KV_QUANT=1 NOMOS_KV_INT4=1 NOMOS_KV_I4_BLOCK=32
export NOMOS_KV_SWA=0 NOMOS_KV_REUSE=0
export NOMOS_VERIFY_MMQ_SMALL=1        # the ONLY NOMOS_VERIFY_* flag that belongs on discrete
export NOMOS_MAX_SEQ=4096 NOMOS_BATCHED_PREFILL=1 CUBLAS_WORKSPACE_CONFIG=":4096:8"
export SPEC_VB=7                       # Gold's optimum. NOT 9 (that is GB10's) — VB=9 costs -33% here.

# Repo root. BASH_SOURCE is NOT reliable here -- under a non-bash /bin/sh (e.g. a plain
# `ssh host 'source tools/gold_env.sh'`) it is unset, dirname "" -> "." and we would resolve one
# level too high. So: try BASH_SOURCE, then $PWD, and VALIDATE by requiring refresh_build.sh.
_gold_repo=""
if [ -n "${BASH_SOURCE:-}" ]; then
  _gold_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" || _gold_repo=""
fi
[ -f "$_gold_repo/refresh_build.sh" ] || _gold_repo="$PWD"
[ -f "$_gold_repo/refresh_build.sh" ] || _gold_repo="$(cd "$PWD/.." 2>/dev/null && pwd)"
if [ ! -f "$_gold_repo/refresh_build.sh" ]; then
  echo "FAIL: cannot locate the repo root (no refresh_build.sh found)." >&2
  echo "      cd into the kernel repo first, then: source tools/gold_env.sh" >&2
  unset _gold_arch _gold_repo
  return 1
fi
export LD_LIBRARY_PATH="$_gold_repo:/usr/local/cuda/lib64:$_gold_repo/.pixi/envs/default/lib"
export WEIGHTS="$HOME/nomos_data/gemma-4-31b/"                 # TRAILING SLASH REQUIRED
export DFLASH_DIR="$HOME/nomos_data/dflash-gemma-4-31b-flat/"  # TRAILING SLASH REQUIRED
export SO_PATH="$_gold_repo/libnomos_kernel.so"
export PERF_NTOK=64 PERF_REPEAT=1

# ---- self-check: fail loudly rather than produce a plausible, lossy number ----
_gold_ok=1
_gold_nv="$(env | grep -c '^NOMOS_VERIFY')"
if [ "$_gold_nv" -ne 1 ]; then
  echo "FAIL: expected exactly 1 NOMOS_VERIFY_* (MMQ_SMALL), found $_gold_nv." >&2
  env | grep '^NOMOS_VERIFY' | sed 's/^/       /' >&2
  echo "      Extra verify flags make this box LOSSY (0/12). Start a clean shell." >&2
  _gold_ok=0
fi
case "$WEIGHTS" in */) ;; *) echo "FAIL: WEIGHTS must end in '/' (else Xid 31 MMU fault at 0x0)." >&2; _gold_ok=0;; esac
case "$DFLASH_DIR" in */) ;; *) echo "FAIL: DFLASH_DIR must end in '/'." >&2; _gold_ok=0;; esac
[ -d "$WEIGHTS" ]    || { echo "FAIL: WEIGHTS not found: $WEIGHTS" >&2; _gold_ok=0; }
[ -d "$DFLASH_DIR" ] || { echo "FAIL: DFLASH_DIR not found: $DFLASH_DIR" >&2; _gold_ok=0; }
[ -f "$SO_PATH" ]    || { echo "FAIL: .so not built here: $SO_PATH  (run ./refresh_build.sh ON THIS BOX)" >&2; _gold_ok=0; }
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "NOTE: CUDA_VISIBLE_DEVICES is unset. Pick a GPU with >=19 GB free FIRST:" >&2
  echo "      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" >&2
  echo "      Gold's topology is FLUID and Nano is a co-resident owner -- never kill an" >&2
  echo "      unidentified process; ask Adam." >&2
fi

if [ "$_gold_ok" = "1" ]; then
  echo "gold env OK  |  SPEC_VB=$SPEC_VB  KV_I4_BLOCK=$NOMOS_KV_I4_BLOCK  GPU=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "  expect: 12/12 LOSSLESS, ~45 tok/s, verify ~60 ms/cycle, E ~3.01"
  echo "  run:    \"\$PWD/.pixi/envs/default/bin/python\" tools/dflash_spec_loop.py"
else
  echo "gold env NOT SAFE TO RUN -- fix the FAILs above." >&2
fi
unset _gold_arch _gold_repo _gold_nv _gold_ok
