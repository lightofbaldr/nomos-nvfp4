#!/usr/bin/env python3
"""Repack RedHat `compressed-tensors` NVFP4 -> our kernel weights (FAITHFUL dequant, not a re-quant of the base).

RedHat target (RedHatAI/gemma-4-31B-it-NVFP4) is `nvfp4-pack-quantized`: group_size 16, E2M1
weights, e4m3 block scales, symmetric. We dequant RedHat's EXACT NVFP4 values to fp32, then encode
into our kernel format. Vision tower is in config.quantization_config.ignore -> skipped (text-only).

  --format q4   (DEFAULT): -> our Q4_0 (.q4). The PROVEN-COHERENT GB10 decode path. Lossy re-quant
                NVFP4->Q4_0, but both the matched-pair arms (our-QAT-Q4 vs RedHat-Q4) are Q4_0, so it
                cleanly isolates the MODEL (our-QAT vs RedHat) at fixed precision. RUNS TONIGHT.
  --format nvfp4: -> our .nvfp4 (faithful, idempotent re-encode). BLOCKED until NVFP4 decode coherence
                on GB10 is fixed (2026-06-29: NVFP4 decode emits degenerate 240017 garbage, upstream of
                the GEMV — see nvfp4-decode-incoherent-gb10). Use once that's resolved.

WHY (matched-pair test): the AEON eagle3 drafter was trained against RedHat's model hiddens. Our QAT
target gave 1.83x (off-distribution). RedHat-Q4 holds Q4_0 precision constant while swapping the MODEL
to the drafter's native one -> if accept JUMPS, the cause is the model; if ~1.83x, it's the bf16->Q4
precision gap (then we need coherent higher-precision decode).

compressed-tensors NVFP4 dequant:  W = e2m1(packed) * e4m3(weight_scale) / weight_global_scale
  weight_packed [N,K/2] uint8 (E2M1, 2/byte, low nibble=even) ; weight_scale [N,K/16] e4m3 ; global fp32
Range-checked (flipped global -> explodes/vanishes -> fails LOUD). Final authority = GPU coherence + accept.

Usage: python3 tools/convert_redhat_nvfp4.py <redhat_dir> <out_dir> [--format q4|nvfp4] [--verify-only]
"""
import sys, os, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_nvfp4 import E2M1_MAG, E4M3_POS, write_nvfp4, nvfp4_quantize, nvfp4_dequantize
from quantize_q4_0 import quantize as q4_quantize, dequant as q4_dequant, write_q4


def _e4m3_decode(byte_u8):
    b = np.clip(byte_u8.astype(np.uint8) & 0x7F, 0, 126)     # strip sign (scales positive)
    return E4M3_POS[b]


NIBBLE_SWAP = False   # set by --nibble-swap: compressed-tensors may pack low-nibble=ODD (vs our low=even)


def _e2m1_decode(packed_u8, K):
    N = packed_u8.shape[0]
    nib = np.empty((N, K), dtype=np.uint8)
    if NIBBLE_SWAP:                          # low nibble = ODD element
        nib[:, 1::2] = packed_u8 & 0x0F
        nib[:, 0::2] = packed_u8 >> 4
    else:                                    # low nibble = EVEN element (our convention)
        nib[:, 0::2] = packed_u8 & 0x0F
        nib[:, 1::2] = packed_u8 >> 4
    sign = np.where((nib & 8) != 0, np.float32(-1.0), np.float32(1.0))
    return (sign * E2M1_MAG[nib & 7]).astype(np.float32)


def dequant_compressed_tensors(packed, scale_e4m3, global_scale, K):
    N = packed.shape[0]; NB = K // 16
    e2 = _e2m1_decode(packed, K).reshape(N, NB, 16)
    bs = _e4m3_decode(scale_e4m3).reshape(N, NB)[:, :, None]
    return (e2 * bs / np.float32(global_scale)).reshape(N, K).astype(np.float32)


def _our_stem(base):
    p = base.replace("model.language_model.layers.", "").split(".")
    return f"layers_{p[0]}_{p[1]}_{p[2]}_weight" if len(p) >= 3 else base.replace(".", "_")


def main():
    args = sys.argv[1:]
    fmt = "nvfp4" if "--format" in args and args[args.index("--format") + 1] == "nvfp4" else "q4"
    verify_only = "--verify-only" in args
    global NIBBLE_SWAP
    NIBBLE_SWAP = "--nibble-swap" in args
    print(f"nibble_swap={NIBBLE_SWAP}", flush=True)
    pos = [a for a in args if not a.startswith("--") and a not in ("q4", "nvfp4")]
    if len(pos) < 2:
        print(__doc__); sys.exit(1)
    in_dir, out_dir = pos[0], pos[1]
    os.makedirs(out_dir, exist_ok=True)
    print(f"format={fmt}  in={in_dir}  out={out_dir}", flush=True)

    from safetensors import safe_open
    import torch  # weight_scale is float8_e4m3fn -> numpy has no float8 dtype; read via torch + view(uint8)
    shards = sorted(glob.glob(os.path.join(in_dir, "*.safetensors")))
    assert shards, f"no .safetensors in {in_dir}"
    key_to_shard = {}
    for s in shards:
        with safe_open(s, framework="numpy") as f:
            for k in f.keys():
                key_to_shard[k] = s
    packed_keys = [k for k in key_to_shard if k.endswith(".weight_packed")
                   and "language_model" in k and "vision_tower" not in k]
    print(f"{len(packed_keys)} quantized language-model Linears across {len(shards)} shards", flush=True)
    assert packed_keys, "no language_model *.weight_packed found"

    def _get(kk):  # torch tensor (handles float8_e4m3fn, which numpy can't represent)
        sh = key_to_shard.get(kk); assert sh, f"missing {kk}"
        with safe_open(sh, framework="pt") as f:
            return f.get_tensor(kk)

    done, first = 0, True
    for pk in sorted(packed_keys):
        base = pk[:-len(".weight_packed")]
        packed = _get(pk).numpy()                                   # uint8
        scale = _get(base + ".weight_scale").view(torch.uint8).numpy()   # float8_e4m3fn -> raw e4m3 bytes
        gscale = float(_get(base + ".weight_global_scale").reshape(-1)[0].item())
        N, Khalf = packed.shape; K = Khalf * 2
        W = dequant_compressed_tensors(packed, scale, gscale, K)
        amax = float(np.abs(W).max())
        if first:
            assert 1e-4 < amax < 1e4, f"DEQUANT SANITY FAIL |W|max={amax} (global convention flipped?)"
            if fmt == "nvfp4":
                gs2, bs2, pk2 = nvfp4_quantize(W); W2 = nvfp4_dequantize(gs2, bs2, pk2, N, K)
            else:
                s16, p4, n = q4_quantize(W.ravel()); W2 = q4_dequant(s16, p4, n).reshape(N, K)
            rel = float(np.abs(W2 - W).max() / (amax + 1e-9))
            print(f"  [verify] {base}: N={N} K={K} |W|max={amax:.4f} {fmt}-roundtrip rel={rel:.3f} "
                  f"({'idempotent' if fmt=='nvfp4' and rel<0.05 else 'lossy-q4 expected ~0.05-0.15' if fmt=='q4' else 'WARN'})",
                  flush=True)
            first = False
            if verify_only:
                return
        stem = os.path.join(out_dir, _our_stem(base))
        if fmt == "nvfp4":
            write_nvfp4(stem + ".nvfp4", W)
        else:
            s16, p4, n = q4_quantize(W.ravel()); write_q4(stem + ".tmp", s16, p4, n)  # write_q4 appends .q4
            os.replace(stem + ".tmp.q4", stem + ".q4")   # atomic -> a kill mid-write leaves .tmp, not a partial .q4
        done += 1
        if done % 60 == 0:
            print(f"  {done}/{len(packed_keys)} ...", flush=True)
    print(f"DONE: wrote {done} {fmt} files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
