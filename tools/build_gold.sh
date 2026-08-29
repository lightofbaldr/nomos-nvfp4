#!/bin/bash
# DEPRECATED SHIM — delegates to ../refresh_build.sh. Do not add build logic here.
#
# This script used to be a SECOND, divergent build path, and that divergence is the reason
# "the CUDA issue" kept coming back (root-caused 2026-08-15). Two concrete ways it differed
# from refresh_build.sh, either of which silently changes what you are testing:
#
#   1. It hardcoded `-L/usr/local/cuda/lib64`, a SYSTEM CUDA. Since 2026-08-10 pixi.toml
#      declares CUDA itself and the project vendors it in .pixi/envs/default; this laptop has
#      no system CUDA at all, so the link simply failed with "cannot find -lcudart".
#   2. It never compiled nomos_cuda_budget.o, so even where it DID link it produced a
#      different .so than the one we measure and ship.
#
# It also reported LINK_OK on a FAILED build, because its readelf check ran unconditionally and
# inspected the STALE .so still sitting on disk. A build script that reports green on failure is
# the same defect family as the BLOCK_ATTN trap — see docs/the measurement notes.
#
# One build path, arch-dispatched inside it. If you need a discrete-specific tweak, add it to
# refresh_build.sh behind an arch/host check so every box keeps getting the same treatment.
set -eo pipefail
echo "NOTE: tools/build_gold.sh is deprecated; running refresh_build.sh instead." >&2
exec "$(cd "$(dirname "$0")/.." && pwd)/refresh_build.sh" "$@"
