#!/bin/bash
# restart_serve.sh — restart ONLY the pyhost serve (engine daemon untouched → ~instant
# reconnect, no GPU reload). For the fast edit→deploy→test loop on the Pro 6000 box.
# Honors NOMOS_DEBUG_RAW from the caller's env (so responses carry debug_raw/debug_prompt_tail).
set -uo pipefail
SRC=/workspace/nomos_kernel_src
pkill -TERM -f 'pyhost/serve.py' 2>/dev/null || true
sleep 1
cd "$SRC"
export PATH=~/.pixi/bin:$PATH
export TOK_DIR=/workspace/models/gemma-4-31b-it
export LD_LIBRARY_PATH=/workspace/lib:$SRC:/usr/local/cuda/lib64
export NOMOS_ENGINE_SOCKET=/tmp/nomos_engine.sock
export NOMOS_HOST=127.0.0.1 NOMOS_PORT=8090
export NOMOS_DEBUG_RAW="${NOMOS_DEBUG_RAW:-}"
setsid python3 pyhost/serve.py > /workspace/pyhost_serve.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
  curl -sf --max-time 3 localhost:8090/health 2>/dev/null | grep -q '"ready":true' && { echo "serve ready after ${i}s (debug_raw=${NOMOS_DEBUG_RAW:-off})"; exit 0; }
  sleep 1
done
echo "serve did NOT come ready"; tail -15 /workspace/pyhost_serve.log; exit 1
