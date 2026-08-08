# Nomos Kernel

A pure-[Mojo](https://www.modular.com/mojo) inference kernel for **Gemma-4-31B** with
**lossless speculative decoding** — 4-bit weights and activations all the way down,
tensor-core batch verification, and streams that are **bit-identical to greedy
decoding by construction**.

> **Status: pre-release.** Built and measured in the open on NVIDIA DGX Spark
> (GB10, sm_121a). Discrete Blackwell (sm_120+, NVFP4 path) runs base decode
> today; its optimization pass and the speculative-stack port are in progress.

## Measured performance

12-prompt weighted benchmark (agentic / code / long-prose / structured buckets),
Gemma-4-31B QAT Q4_0, 96 tokens per prompt, greedy — NVIDIA GB10 (DGX Spark),
2026-07-14:

| Configuration | tok/s (weighted) | Losslessness |
|---|---|---|
| Base greedy decode | 10.91 | — (it *is* the reference) |
| **DFlash speculative, VB=9 (champion)** | **21.88** | **12/12 prompts token-exact vs greedy** |
| llama.cpp draft-assisted (same box/model/prompts, for context) | 26.54 | not bit-gated |

"Lossless" is the strong claim: the speculative pipeline's output is compared
token-for-token against the same engine's non-speculative greedy stream, every
run. It passes not by luck but by arithmetic — see *Correctness by construction*.

> **A number is only real for the box, the code path, and the entry point it was
> measured on.** The same KV-scale flag measures −3.5% on one card and +15.7% on
> another; the same greedy gate is sound for one change and blind to the next; and
> these figures come from the benchmark harness, not the served endpoint. Every
> number in this repo is reproducible with the commands below, on your own box —
> that is the point.

## Quickstart

```bash
git clone https://github.com/lightofbaldr/nomos-nvfp4.git && cd nomos-nvfp4
pixi install
bash refresh_build.sh            # builds libnomos_kernel.so FOR THIS GPU (arch is baked in)
# download the weights and export WEIGHTS + DFLASH_DIR — see "Weights" below
./nomos env                      # resolved precision/paths for this machine
./nomos bench                    # base decode on the 12-prompt bar (deps: numpy only)
./nomos bench --spec             # + DFlash speculative decode with lossless verification
./nomos smoke --prompt "hello"   # one completion (needs `transformers` for the tokenizer)
```

The CLI selects precision per architecture (unified-memory GB10 → Q4_0; discrete
Blackwell → NVFP4 W4A4) and handles the library-path and weights-path ceremony.
Any `NOMOS_*` variable you export beforehand wins — what runs is what was requested.

**Weights (Hugging Face).** The 4-bit weights in the kernel's flat format are at
[`Adam1010/nomos-gemma-4-31b-nvfp4`](https://huggingface.co/Adam1010/nomos-gemma-4-31b-nvfp4).
**One download gets both** the base model (repo root) and the DFlash drafter (`drafter/`):

```bash
pip install -U huggingface_hub
hf download Adam1010/nomos-gemma-4-31b-nvfp4 --local-dir ~/nomos_data/gemma-4-31b-nvfp4

export WEIGHTS=~/nomos_data/gemma-4-31b-nvfp4/              # base model — always required
export DFLASH_DIR=~/nomos_data/gemma-4-31b-nvfp4/drafter/   # drafter — required for --spec
```

- **Base decode** (`nomos bench`, `nomos verify`) needs only `WEIGHTS`.
- **Speculative decode** (`nomos bench --spec`, spec serve) needs `WEIGHTS` **and** `DFLASH_DIR`.
- **`smoke` / `serve`** additionally need the Gemma-4-31B **tokenizer** — obtain it from
  Google's official Gemma-4 release under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms),
  then `pip install transformers` and `export TOK_DIR=~/models/gemma-4-31b-it/`.

> These are **NVFP4** weights for discrete Blackwell (`sm_120` / `sm_100`). On a
> unified-memory **GB10 / DGX Spark**, build the **Q4_0** flat weights with the
> converters in `tools/` — see [BUILD.md](BUILD.md). The DFlash drafter design is
> z-lab / Anbeeld's; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Serving (OpenAI-compatible)

```bash
export NOMOS_SERVE_SPEC=1                                     # DEFAULTS TO 0 — see below
export DFLASH_DIR=$HOME/nomos_data/gemma-4-31b-nvfp4/drafter/   # TRAILING SLASH REQUIRED
pyhost/serve_nomos.sh up         # arch-detects quant, sets the precision env, serves
curl -s localhost:$NOMOS_PORT/health
```

> **The serve does NOT speculate by default.** `NOMOS_SERVE_SPEC` defaults to `0`
> and `DFLASH_DIR` to empty, so a fresh clone serves **base decode** — roughly half
> the throughput of the benchmark numbers above. On startup, a serve that will
> speculate logs `DFlash speculative decode enabled (vb=…)`; **if you don't see that
> line, you are not measuring the speculative path.** Speculation also requires
> `temperature <= 0` and no grammar/tool constraint on the request — sampling and
> tool-constrained requests transparently fall back to base decode.
>
> **Benchmark numbers come from `tools/dflash_spec_loop.py`, not from the serve.**
> Do not quote a harness number for a served endpoint without re-measuring it there.

Per-host `WEIGHTS` / `TOK_DIR` / `NOMOS_PORT` come from `~/.nomos_host_env`. The
serve script sets the precision env per architecture. (`pyhost/engine_daemon.py`
holds the GPU on a UNIX socket; the API layer restarts without reloading the model.)

Spec-serve setup and the flags that are off *for cause*:
[`docs/SPECULATIVE_SERVING.md`](docs/SPECULATIVE_SERVING.md).

## How it's fast

- **Everything 4-bit.** Q4_0 (or NVFP4) weights, int8 dp4a activations, int4 KV
  cache. The perf path materializes no bf16.
- **Block speculative decoding** ([DFlash](https://arxiv.org/abs/2602.06036)-style):
  a 5-layer drafter conditioned on the target's own hidden states proposes a
  16-token block in one forward pass; the target verifies rows in one batched pass.
  Drafting reads ~0.25 GB where a token-by-token drafter reads ~1.4 GB per token.
- **Tensor-core batch verify:** multi-row verify GEMMs run `mma.sync` int8 tiles
  fed by cooperative `cp.async` shared-memory staging, with a measured route
  crossover (small batches stay on the tuned dp4a path).
- **Verify-budget policy:** verifying only the first 4 drafted tokens (VB=4)
  maximizes throughput at current acceptance; full-block wins on structured output.

## Correctness by construction

Speculative decoding is only lossless if verify and decode agree on every
logit's argmax — and floating-point accumulation order breaks that in the last
bits, exactly where near-ties live. This kernel removes the problem instead of
managing it: every Q4 matmul accumulates its per-block integer dot products into
a **Q14 fixed-point integer sum** — order-independent by the arithmetic itself —
and rounds to float once. Verify equals decode bit-for-bit no matter how the
work is scheduled — on CUDA cores or tensor cores alike, on the same hardware.

The battery ships in-repo: a deterministic 12-prompt greedy-id dump
(`tools/gold_parity_ids.py`), the runtime precision guard
(`NOMOS_STRICT_Q4=1` + `tools/precision_law_guard.py` — an unrequested precision
escalation is a test failure), and the speculative bench's per-prompt
token-exact comparison against greedy.

## Layout

- `nomos` — the launch CLI (env / smoke / bench).
- `nomos_ffi.mojo` — C-ABI exports (init / prefill / decode_step / verify /
  dflash_* / kv_*).
- `lib/` — the engine (`gemma4_engine`, `engine_prefill`, `engine_decode`), the
  Q4 GEMV/MMA kernels (`q4_gemv_dp4a.mojo` — decode, staged tensor-core verify,
  fixed-point epilogue), int8 attention, INT4 KV quant, the DFlash drafter, sampling.
- `pyhost/` — Python serving host over the Mojo logits engine.
- `tools/` — converters, probes, the bar token set, the precision guard, build gates.

## Hardware

| Target | Arch | Precision | State |
|---|---|---|---|
| NVIDIA GB10 (DGX Spark) | sm_121a | Q4_0 | **fully optimized — the record config** |
| RTX PRO 4000/6000 Blackwell | sm_120(a) | NVFP4 W4A4 | base decode runs; optimization in progress |
| B200 | sm_100 | NVFP4 | builds; untuned |

Build on the machine you'll run on — the GPU architecture is baked into the `.so`.

## Verification gates

Every change passes: **build** (target arch) · **nm-gate** (`tools/nm_gate.sh`,
clean exports) · **greedy-id parity** (`tools/gold_parity_ids.py`) · **the bar**
(per-prompt speculative token-exact parity vs greedy, reported in every table).

## Roadmap

- Close the remaining gap to (and past) the llama.cpp draft-assisted reference:
  verify pipeline overlap, route-crossover tuning, drafter-quantization polish.
- NVFP4/sm_120 base-decode optimization and the speculative-stack port to
  discrete Blackwell (in progress).
- Additional drafter families (EAGLE-3, MTP) on the same verify seam.

## Documentation

- [BUILD.md](BUILD.md) — building on your GPU (per-architecture; `max-core` required).
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, the two 4-bit precision paths, the C ABI.
- [CONTRIBUTING.md](CONTRIBUTING.md) — the correctness gates and benchmark hygiene.

## License

MIT — see [LICENSE](LICENSE). Third-party components and design credits (DFlash /
z-lab·Anbeeld, llama.cpp, EAGLE-3, Mojo/MAX) are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Nomos does not distribute model
weights; Gemma-4 weights are obtained from Google under the Gemma Terms of Use.
