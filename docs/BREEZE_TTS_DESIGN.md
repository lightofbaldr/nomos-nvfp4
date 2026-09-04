# Breeze-TTS-2 on the Nomos kernel — design (approved 2026-09-04)

Adam-approved scope: **all model capabilities** (voice clone, voice design, voice direction,
vocal events, bilingual en/zh, streaming), landing as a **LAN fleet TTS service**. Support code
is public-repo eligible (usual Freyja rounds); **weights are research/non-commercial**
(breezeblue-research-and-non-commercial-license) — the service is local-only, nothing sold.
Public docs must label the model's license class.

## Model (BreezeBlue/Breeze-TTS-2, HF, 2026-08-25)
CSM-lineage composite, four sub-models (~3.5B total, bf16 ≈ 7GB):
1. **Text encoder** — T5Gemma2, 26L, hidden 1152, gemma sliding/full mix (5:1, window 512),
   vocab 262158 (gemma vocab + 12 speaker/instruction tokens). Output projected (linear) into
   backbone embedding space.
2. **Backbone** — Qwen3 flavor "llama-1B": 28L, hidden 2048, GQA 16/8, head_dim 128, per-head
   QK-norm, llama3-type RoPE scaling (factor 32). AR over Mimi frames @ 12.5Hz: predicts
   codebook-0 + hidden for the depth decoder. Text conditioning = encoder output spliced as
   prefix features (the `nomos_prefill_mm` pattern).
3. **Depth decoder** — "llama-100M": 12L, hidden 1024, max_pos 33; per-frame inner AR across
   16 codebooks (audio_vocab 2051/codebook). Same .so as backbone (drafter-scale).
4. **Audio codec decoder — Qwen3TTSTokenizerV2** (bundled `audio_tokenizer/`, 651MB; loaded via
   the `qwen-tts` package). CORRECTED 2026-09-04, found by Codex pre-build: the top-level
   `codec_config` (kyutai/mimi) is a legacy/training fallback the shipped inference path never
   uses — `breeze_infer/runtime.py` REQUIRES the bundled artifact (raises without it) and all
   generated codes decode through `audio_tokenizer.decode`. Real geometry: latent 1024,
   decoder_dim 1536, 16 heads, 16 quantizers, pre-conv 512→1024, 8L pre-transformer (hidden
   512), upsample_rates [8,5,4,3] + upsampling_ratios [2,2], transposed-conv/residual stack.
   **Standalone .so** (vision-encoder pattern, zero blast radius); streaming decode per frame
   for TTFA. M1 authority = bundled `audio_tokenizer/{config.json,model.safetensors}` + the
   `audio_tokenizer.decode` path as oracle.

CFG (voice design/direction): conditional + unconditional decode per step with logit mixing
(`--cfg-scale`), clone path runs without CFG. Vocal events are plain text tokens.

## Architecture decision
**In nomos-kernel-product, vision pattern** — engine profile(s) + standalone mimi .so + serve
endpoint. Rejected: separate repo (duplicates/bridges the engine that already runs 3 of the 4
sub-models, and every converter/parity/gate/serve asset); running upstream PyTorch as the
service (H100/flash-attn targeted, heavy on aarch64, and it's the dependency this kernel
replaces).

## Milestones (implementer ≠ verifier; HF reference is the external anchor at every gate)
- **M0** — weights + converters + HF reference running on Spark 2 (bf16), per-stage goldens
  captured (encoder hiddens, backbone logits/hidden per frame, depth codebook tokens, mimi
  waveform). Owner: Kvasir.
- **M1** — Mimi decoder .so: tokens→waveform parity vs HF goldens (relL2 gate + audible spot
  check). Owner: Codex, gated by Kvasir.
- **M2** — text-encoder tower parity (per-layer hiddens vs HF).
- **M3** — backbone + depth: teacher-forced frame parity, greedy token-for-token vs HF
  (bf16 first — parity before precision).
- **M4** — e2e voice clone: text+reference → WAV on-kernel; HF A/B listen + token-level match.
- **M5** — CFG: voice design + direction; gate vs HF at matched cfg-scale.
- **M6** — serve: `/v1/audio/speech` (OpenAI TTS shape) + streaming PCM + named-voice registry
  (voice = reference clip + transcript, stored server-side); fleet rollout on the LAN.
- **M7** — precision (LAST, measure-first per the precision law): quantize backbone
  (NVFP4/Q4 arms), keep depth/mimi bf16 unless measured safe; perceptual + token-metric A/B
  vs the bf16 kernel arm. No pre-emptive escalation, no unintentional precision.

## Serve
`/v1/audio/speech`: {model, input, voice, response_format(wav|pcm), stream, instruction?,
cfg_scale?}. Voice registry maps names → {ref_audio, ref_text} (fleet identities, e.g. each
agent's voice). Streaming = chunked PCM as mimi frames decode. Registry entry `breeze_tts`
in pyhost/models.py, same launcher flow as every model.

## References
- HF: BreezeBlue/Breeze-TTS-2 (config.json is the geometry authority)
- Inference code: github.com/breezeblue-ai/breeze-tts (Apache 2.0)
- H100 claims to beat locally: TTFA <40ms fast path, RTF 0.32; eager ~7.7GiB.
