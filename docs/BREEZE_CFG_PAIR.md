# Breeze paired CFG decode

The two conditioning lanes share projection weights and advance from the same
sampled audio codes. The paired APIs batch their projection/MLP rows together.
Each lane retains its own RoPE position, causal attention domain, KV buffers,
and backbone hidden state. Prefill stays separate with the actual prefix
lengths (23 and 10 in the golden fixture).

The original single-lane exports remain available. Depth weights, heads,
embeddings, norms, and sampling rules retain their existing precision.
This change does not add depth NVFP4 loading.

## C ABI

```c
// Existing: call independently for lane 0 and lane 1.
int32_t nomos_breeze_model_prefill_lane(
    int64_t handle, int32_t lane, const float *embeds, int32_t S,
    float *optional_layers, float *optional_final, float *logits2052);

// New: lane-major outputs (row 0 conditional, row 1 unconditional).
int32_t nomos_breeze_model_depth_begin2(
    int64_t handle, const int64_t *codes16, float *logits2x2051);
int32_t nomos_breeze_model_depth_advance2(
    int64_t handle, int32_t input_codebook,
    const int64_t *codes16, float *logits2x2051);
int32_t nomos_breeze_model_step_backbone2(
    int64_t handle, const int64_t *codes16, float *logits2x2052);
```

All arrays are contiguous host arrays; calls synchronize before returning.
Both lanes must have a live prefix. At each frame:

1. Sample codebook 0 from the CFG mixture of the two backbone logits.
2. Set `codes16[0]`, then call `depth_begin2` to obtain codebook-1 logits.
   Internally this is a four-row GEMM: [cond hidden, code0, uncond hidden, code0].
3. For input_codebook 1 through 14, set that sampled code and call
   `depth_advance2` to predict the next codebook. Each is a two-row GEMM with
   separate per-lane attention. Both cache cursors must match the requested step.
4. After sampling codebook 15, call `step_backbone2` with the completed frame.
   Each backbone cursor advances by one from its own current position.

Only codes already sampled are read by depth. The replay test fills later
codebooks with -1 to exercise this contract. A null required pointer returns
-1. Invalid paired cadence, missing prefill, or exhausted backbone capacity
returns -99 through the existing exception convention.

## Validation and timing

```sh
./refresh_breeze_model_build.sh
python3 tools/breeze_cfg_pair_gate.py --bench <dir with the two golden .npz> --repeats 3 --output /tmp/cfg-pair.json
```

The goldens (`breeze_model_goldens.npz`, `breeze_codec_codes_golden.npz`) are captured
from the HF reference per the design doc's M0 and are not distributed here (they derive
from the model weights). The sequential arm of this replay is the original M3 gate.

The paired replay uses the two actual prefixes and all 35 captured frames.
It requires 36/36 backbone top1 matches for each lane, zero depth misses at
golden margin >=0.1, and mean depth relL2 <0.05. Sequential and paired runs
alternate order on one handle with a fresh prefill for every arm. Timings
exclude initialization, prefill, sampling, and codec; each arm discards its
first frame from timing but includes it in the correctness gate.

The first replay on GB10 passed with paired depth 513/525 conditional and
512/525 unconditional (all misses below margin 0.1), mean relL2 0.01243/0.01224,
and backbone 36/36 both lanes. The existing ABI also passed the original M3
gate. Batching changes GEMM reduction order, so this is the M3 tolerance
contract, not a byte-identity guarantee between sampling trajectories.

Three alternating same-session replays on GB10 give median backbone/depth
60.60/172.16 ms per frame sequential and 28.16/96.49 paired:
232.76 -> 124.64 ms/frame for the model (1.87x, 46.5% lower latency).
These are model-call measurements. Adding the previously measured codec cost
(~8.76 ms/frame) estimates ~133.4 ms/frame; that sum is not an end-to-end serve
measurement. A separate set of repeated timings collected while a :8095
request was active is excluded from performance claims.

## Generation driver

Add `--paired-cfg` to `tools/breeze_generate_kernel.py` with both
`--cond-prefix` and `--uncond-prefix`. CFG mixing and sampling remain in the
host driver, producing one shared code at each step. The option is explicit;
single-lane cloning continues to use the original API.

A sampled driver smoke with the assembled cond/uncond prefixes, CFG 4,
seed 42, and a 40-frame limit emitted natural EOS after 38 frames.
The codec produced 72,960 finite samples (3.04 seconds at 24 kHz); all
codes were in range [0,2048). Listening and independent parity review
remain the verifier's checks before serve promotion.
