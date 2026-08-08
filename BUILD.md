# Building Nomos Kernel

Nomos is a Mojo GPU kernel. Mojo **bakes GPU code for one architecture into each
`libnomos_kernel.so`**, so the cardinal rule is:

> **Build on the machine you will run on.** Never copy a `.so` between GPUs of
> different compute capability, even when both are Blackwell.

## Prerequisites

- An NVIDIA Blackwell GPU and a recent driver + CUDA runtime (13.x).
  - Unified-memory GB10 (DGX Spark, `sm_121a`) → the Q4_0 path.
  - Discrete Blackwell (`sm_120`/`sm_100`) → the NVFP4 W4A4 path.
- [`pixi`](https://pixi.sh) for environment management.
- Gemma-4-31B weights you are licensed to use (see `THIRD_PARTY_NOTICES.md`),
  converted to the kernel's format with the tools in `tools/` and `dflash/`.

## Build

```bash
pixi install          # resolves mojo + max-core (GPU codegen) + Python deps
bash refresh_build.sh # compiles libnomos_kernel.so FOR THIS GPU (arch auto-detected)
```

`refresh_build.sh` syncs the pixi env, compiles the C shims, builds
`nomos_ffi.mojo` as a shared library, links the CUDA runtime + cuBLAS, and
prints the exported-symbol count. Verify the build produced a fresh library
with the full ABI:

```bash
nm -D libnomos_kernel.so | grep -c ' T nomos_'   # expect 85
```

> A ~10-second "build" is an incremental cache hit — confirm the `.so`
> timestamp and symbol count reflect your change, not just a zero exit code.

**`max-core` is required.** Today's `mojo` package no longer bundles GPU
accelerator codegen; without `max-core` the build fails with
`please install MAX for accelerator support`. `pixi install` pulls the pinned
`max-core` automatically — do not remove it.

## Run

```bash
./nomos env            # show the resolved precision/paths for this machine
./nomos bench          # base greedy decode on the 12-prompt bar (numpy only)
./nomos bench --spec   # + speculative decode with lossless verification
./nomos smoke --prompt "hello"   # one completion (needs transformers for the tokenizer)
```

Serving (OpenAI-compatible HTTP) is covered in the README.

## Per-architecture notes

| GPU | Compute cap | Precision path |
|---|---|---|
| DGX Spark (GB10) | `sm_121a` | Q4_0 (DP4A / MMQ) |
| Workstation/consumer Blackwell | `sm_120(a)` | NVFP4 W4A4 (`mma.sync`) |
| Datacenter Blackwell (B200/GB200) | `sm_100(a)` | NVFP4 W4A4 (`tcgen05`) — base decode |

Build once per target on that target. See `docs/ARCHITECTURE.md` for why the two
4-bit paths exist and how they differ.
