#!/usr/bin/env bash
set -euo pipefail
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
case "$(uname -m)" in
  aarch64) CUDA_ARCH_DIR=sbsa-linux ;;
  x86_64) CUDA_ARCH_DIR=x86_64-linux ;;
  *) exit 2 ;;
esac
CUDA_LIB="$CUDA_ROOT/targets/$CUDA_ARCH_DIR/lib"
gcc -c -fPIC nomos_libc_shim.c -o nomos_libc_shim.o
gcc -c -fPIC nomos_token_cb_stub.c -o nomos_token_cb_stub.o
pixi run mojo build --emit shared-lib qwen_vision_ffi.mojo -o libnomos_vision-qwen3_5.so \
  -Xlinker nomos_libc_shim.o -Xlinker nomos_token_cb_stub.o \
  -Xlinker "-L$CUDA_LIB" -Xlinker -rpath -Xlinker "$CUDA_LIB" \
  -Xlinker -lcudart -Xlinker -lcublas -Xlinker -lm
nm -D libnomos_vision-qwen3_5.so | grep nomos_qwen_vision
echo "Qwen vision build green: libnomos_vision-qwen3_5.so"
