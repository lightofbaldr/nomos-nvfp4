#!/bin/bash
set -eo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$(cd "$(dirname "$0")" && pwd)"
case "$(uname -m)" in aarch64) CUDA_ARCH_DIR=sbsa-linux;; x86_64) CUDA_ARCH_DIR=x86_64-linux;; *) exit 2;; esac
PIXI_ENV_DIR="$PWD/.pixi/envs/default"; CUDA_ROOT="${CUDA_HOME:-$PIXI_ENV_DIR}"
if ! ls "$CUDA_ROOT/targets/$CUDA_ARCH_DIR/lib"/libcudart.so* >/dev/null 2>&1; then CUDA_ROOT=/usr/local/cuda; fi
CUDA_LIB="$CUDA_ROOT/targets/$CUDA_ARCH_DIR/lib"
gcc -c -fPIC nomos_libc_shim.c -o nomos_libc_shim.o
gcc -c -fPIC nomos_token_cb_stub.c -o nomos_token_cb_stub.o
pixi run mojo build --emit shared-lib breeze_model_ffi.mojo -o libnomos_model-breeze.so -Xlinker nomos_libc_shim.o -Xlinker nomos_token_cb_stub.o -Xlinker "-L$CUDA_LIB" -Xlinker -rpath -Xlinker "$CUDA_LIB" -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
nm -D libnomos_model-breeze.so | grep nomos_breeze_model_prefill_lane
nm -D libnomos_model-breeze.so | grep nomos_breeze_model_step
