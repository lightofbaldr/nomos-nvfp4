# Third-Party Notices

Nomos Kernel is released under the MIT License (see `LICENSE`). It builds on,
derives from, or interoperates with the following third-party work. We gratefully
acknowledge these projects.

## Build & runtime toolchain

- **Mojo and MAX** (Modular Inc.) — the language, standard library, and GPU
  accelerator support (`max-core`) this kernel is written in and compiled with.
  These toolchain packages are obtained separately from Modular and are subject
  to Modular's applicable license terms; the pinned Conda packages identify
  their license as `LicenseRef-Modular-Proprietary`.
  https://github.com/modular/modular

## Derived designs & references

- **DFlash block drafter** — the speculative block-drafter architecture used in
  `lib/dflash_drafter.mojo` originates with **z-lab / Anbeeld**. The
  implementation here is our own, but the design and the credit are theirs.

- **llama.cpp** (ggml-org and contributors) — the Gemma-4 assistant compute
  graph followed by `lib/mtp_drafter.mojo` mirrors llama.cpp's reference graph,
  and llama.cpp is the reference decoder our benchmarks compare against.
  MIT License. https://github.com/ggml-org/llama.cpp

- **EAGLE-3 speculator** — the EAGLE-3 drafter path (`lib/eagle3_drafter.mojo`)
  follows the EAGLE-3 design; reference weights/format from RedHatAI.
  Apache License 2.0.

## Model weights (not distributed)

- **Gemma 4** (Google) — this repository does **not** distribute Gemma model
  weights. Users obtain weights directly from Google and are bound by the
  **Gemma Terms of Use** (https://ai.google.dev/gemma/terms). The conversion
  tools here produce quantized derivatives of Gemma weights the user already
  holds a license to; that license governs the derivatives.
