# Fair same-card llama.cpp comparison (Gold, RTX PRO 4000 Blackwell sm_120, GPU0)

Vanilla llama.cpp base-decode vs our NVFP4 kernel on the SAME card / model / quant /
prompts, to close the "vs llama.cpp on this silicon" comparison for gemma-4-31B.

## Setup (fairness)
- **Model:** gemma-4-31B **dense** (30.70B params). llama.cpp = official Google QAT
  GGUF `google/gemma-4-31B-it-qat-q4_0-gguf` (`gemma-4-31B_q4_0-it.gguf`, 16.42 GiB,
  Q4_0 ~4.5 bpw) — matches our QAT-Q4 / NVFP4 (~4.5 bpw) lineage and bit-width.
- **Prompts:** the 12-prompt gold bar. The EXACT token ids from `../gold_bar_tokens.json`
  are POSTed to `/completion` (bypasses tokenization → identical prompt tokens).
- **Sampling:** greedy (temperature 0), n_predict = 96 (== PERF_NTOK). Both sides.
- **GPU:** GPU0 only, pinned by UUID (`CUDA_VISIBLE_DEVICES=$LLAMA_GPU_UUID`). GPU1's
  llama-server + GPU2's ollama verified untouched before/during/after. `-ngl 99 -fa on`.
- llama.cpp build 06938ac.

## Results (llama.cpp, 2 runs x 12 prompts)
| method | decode tok/s | prefill tok/s |
|---|---|---|
| llama-bench tg96 (r=5) | 29.58 ± 0.09 | 1027 (pp96) |
| server /completion, exact gold ids, 2x12 | 30.29 mean (29.9–30.5) | ~500–940 |

llama.cpp base decode on Gold = **~30 tok/s**.

## Base-decode reference (Gold, RTX PRO 4000)

| comparison | ours | llama.cpp | result |
|---|---|---|---|
| base-vs-base (pure kernel) | 21.7 | ~30 | **llama.cpp base ~1.36–1.40x FASTER** |

- Vanilla llama.cpp base decode BEATS our NVFP4 base kernel on identical silicon/model/quant.
  Our base kernel has headroom to close. (An honest Gold-box base-decode measurement; the
  ~30 here is **Gold's** number and should not be fused with the 5090's 30.39 — different card.)

## Retired verdict — and why (read this if you're diffing the repo)

An earlier version of this file also carried a **"our stack ~1.65–1.69× faster"** row that
compared *our speculative decode* against llama.cpp's **un-drafted base**. **We are removing it,
and the honest reason is that it flattered us:** it was a stack-vs-base comparison that withheld
llama.cpp's own drafter, and it was only defensible while llama.cpp had no working gemma-4 drafter.

That was true against llama.cpp build **`06938ac`** (the assistant/DFlash head was a coupled
next-n head with no own K/V, not usable as `--model-draft`). It **expired** upstream: build
**`7bd8282`** added native DFlash *and* DSpark support, so llama.cpp now speculates on gemma-4.
The fair, spec-vs-spec comparison — where llama.cpp runs its own drafter — lives in the main
[README](../../README.md#head-to-head-vs-llamacpp--rtx-5090-laptop-discrete-blackwell) and is the
authoritative one. A dated claim is falsifiable; that's why we're dating it rather than deleting it.

## Files
- `bench_llamacpp_server.py` — the harness (launches GPU0-pinned llama-server, feeds exact ids).
- `gold_prompts_text.json` — gold ids decoded to text (reference; harness uses ids directly).
- `llamacpp_server_results.jsonl` — per-prompt per-run timings.
- `llama_bench_31b_q4_0.txt` — clean llama-bench cross-check.
- `bench_llamacpp_goldbar.sh` — earlier llama-cli attempt; this build's llama-cli gets stuck
  in an interactive `>` loop (`-no-cnv`/`--simple-io`/`</dev/null` all ineffective), so the
  server harness is authoritative. Kept for provenance.
- `moe_reference_discarded.txt` — early red-herring run on the 26B-A4B MoE (a DIFFERENT,
  lighter model); DISCARDED once the fair 31B-dense GGUF was pulled. Not part of the verdict.
