#!/usr/bin/env python3
"""NVFP4 weight quantizer (FP4 Step 2, W4-weights).

Encodes a [N, K] weight to the NVFP4 format that lib/fp4_gemm.mojo consumes:
  - one fp32 PER-TENSOR global scale
  - one e4m3 scale per 16-element block      → bs_bytes [N, K/16] uint8
  - E2M1 (FP4) values packed 2/byte          → packed   [N, K/2]  uint8
    (byte[j] low nibble = element 2j, high nibble = 2j+1 — matches fp4_gemm/the spike)

Two-level scaling (standard NVFP4):
  global   = amax(W) / (6 * 448)                         # 6 = max |E2M1|, 448 = max e4m3
  bs[b]    = quant_e4m3( amax(W_block_b) / (6 * global) )
  q[i]     = round_E2M1( W[i] / (global * bs[block(i)]) )
  dequant: W_hat[i] = global * bs[block(i)] * E2M1_decode(q[i])

E2M1 magnitudes {0,.5,1,1.5,2,3,4,6}; nibble = (sign<<3)|mag_index (sign bit = 0x8).
e4m3 is done via an exact 127-entry lookup (no ml_dtypes dependency); the byte values
match the hardware ue4m3 the NVFP4 MMA decodes (validated in fp4_nvfp4_stock_test.mojo:
0x38→1.0, 0x40→2.0, 0x30→0.5, 0x48→4.0). Run: `python3 tools/quantize_nvfp4.py [w.bin K]`.
"""
import sys
import numpy as np

# ── E2M1 (FP4): 8 magnitudes + round-to-nearest midpoints ──────────────────────
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
E2M1_MAX = np.float32(6.0)


# ── e4m3 (block scales): exact lookup, byte 0..126 → positive value (monotonic) ──
def _build_e4m3_pos() -> np.ndarray:
    v = np.zeros(127, dtype=np.float32)   # byte 0x7F (exp15,mant7) = NaN, excluded
    for b in range(127):
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0:
            v[b] = mant * (2.0 ** -9)                  # subnormal
        else:
            v[b] = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return v                                            # v[0]=0 … v[126]=448, increasing


E4M3_POS = _build_e4m3_pos()
E4M3_MID = (E4M3_POS[:-1] + E4M3_POS[1:]) * 0.5
E4M3_MAX = np.float32(E4M3_POS[-1])                     # 448.0


def quant_e4m3_pos(x: np.ndarray):
    """Positive float array → (e4m3 byte uint8 0..126, dequantized fp32)."""
    xc = np.clip(x, 0.0, E4M3_MAX)
    byte = np.searchsorted(E4M3_MID, xc).astype(np.uint8)
    return byte, E4M3_POS[byte]


def _to_e2m1_nibble(x: np.ndarray) -> np.ndarray:
    sign = (x < 0).astype(np.uint8)
    idx = np.searchsorted(E2M1_MID, np.abs(x)).astype(np.uint8)   # nearest of the 8 mags
    return ((sign << 3) | idx).astype(np.uint8)


def nvfp4_quantize(W: np.ndarray):
    """W: [N, K] float32 (K % 16 == 0). Returns (global_scale, bs_bytes[N,K/16],
    packed[N,K/2])."""
    N, K = W.shape
    assert K % 16 == 0, "K must be a multiple of 16"
    NB = K // 16
    amax = np.float32(np.abs(W).max())
    global_scale = np.float32(amax / (E2M1_MAX * E4M3_MAX)) if amax > 0 else np.float32(1.0)

    Wb = W.reshape(N, NB, 16)
    block_amax = np.abs(Wb).max(axis=2)                      # [N, NB]
    bs_unq = block_amax / (E2M1_MAX * global_scale)          # unquantized block scale
    bs_bytes, bs_deq = quant_e4m3_pos(bs_unq)                # → e4m3 byte + dequant value

    denom = (global_scale * bs_deq)[:, :, None]
    denom = np.where(denom == 0.0, np.float32(1.0), denom)
    scaled = np.clip(Wb / denom, -E2M1_MAX, E2M1_MAX)        # [N, NB, 16]

    nib = _to_e2m1_nibble(scaled).reshape(N, K)
    packed = ((nib[:, 1::2] << 4) | nib[:, 0::2]).astype(np.uint8)   # [N, K/2]
    return global_scale, bs_bytes.reshape(N, NB).astype(np.uint8), packed


def nvfp4_dequantize(global_scale, bs_bytes, packed, N: int, K: int) -> np.ndarray:
    NB = K // 16
    bs_deq = E4M3_POS[bs_bytes.reshape(N, NB)]
    nib = np.empty((N, K), dtype=np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = packed >> 4
    sign = np.where((nib & 8) != 0, np.float32(-1.0), np.float32(1.0))
    mag = E2M1_MAG[nib & 7]
    fp4 = (sign * mag).reshape(N, NB, 16)
    return (global_scale * bs_deq[:, :, None] * fp4).reshape(N, K).astype(np.float32)


def write_nvfp4(path: str, W: np.ndarray):
    """Encode W[N,K] and write a .nvfp4 file:
        header 24B = [int64 n][int64 nb][float32 global][float32 pad]
        body      = [e4m3 scales : nb][E2M1 packed : n/2]
    (matches lib/io.read_nvfp4_bytes + lib/fp4_weights.load_to_gpu_nvfp4)."""
    N, K = W.shape
    gs, bs, packed = nvfp4_quantize(W)
    n = N * K
    nb = n // 16
    with open(path, "wb") as f:
        f.write(np.array([n, nb], dtype=np.int64).tobytes())   # 16B
        f.write(np.array([gs], dtype=np.float32).tobytes())    # 4B  global
        f.write(np.array([0.0], dtype=np.float32).tobytes())   # 4B  pad → 24B header
        f.write(bs.tobytes())                                  # nb bytes (e4m3)
        f.write(packed.tobytes())                              # n/2 bytes (E2M1)


# K (input dim) per Gemma-4-31B projection — constant across all 60 layers.
NVFP4_K = {
    "self_attn_q_proj_weight": 5376, "self_attn_k_proj_weight": 5376,
    "self_attn_v_proj_weight": 5376, "self_attn_o_proj_weight": 8192,
    "mlp_gate_proj_weight": 5376, "mlp_up_proj_weight": 5376,
    "mlp_down_proj_weight": 21504,
}


def requant_all(in_dir: str, out_dir: str):
    """Requant every layers_*_<proj>_weight.bin in in_dir → .nvfp4 in out_dir."""
    import os, glob
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    tot_in = 0
    tot_out = 0
    for layer in range(60):
        for proj, K in NVFP4_K.items():
            src = f"{in_dir}/layers_{layer}_{proj}.bin"
            if not os.path.exists(src):
                continue
            raw = np.fromfile(src, dtype=np.float32)
            N = raw.size // K
            if N * K != raw.size:
                print(f"  SKIP {src}: size {raw.size} not divisible by K={K}")
                continue
            dst = f"{out_dir}/layers_{layer}_{proj}.nvfp4"
            write_nvfp4(dst, raw.reshape(N, K))
            done += 1
            tot_in += raw.size * 4
            tot_out += os.path.getsize(dst)
    print(f"requant_all: {done} weights  {tot_in/1e9:.1f}GB fp32 → {tot_out/1e9:.1f}GB nvfp4 "
          f"({tot_in/max(tot_out,1):.2f}x)")


def _roundtrip(name: str, W: np.ndarray):
    N, K = W.shape
    gs, bs, packed = nvfp4_quantize(W)
    W_hat = nvfp4_dequantize(gs, bs, packed, N, K)
    err = float(np.linalg.norm(W_hat - W) / (np.linalg.norm(W) + 1e-12))
    bytes_fp4 = packed.size + bs.size + 4
    bytes_bf16 = W.size * 2
    print(f"  [{name:24}] {N}x{K}  global={gs:.3e}  RMSE/||W||={err:.4%}  "
          f"{bytes_fp4/1e6:.1f}MB vs bf16 {bytes_bf16/1e6:.1f}MB ({bytes_bf16/bytes_fp4:.2f}x)")
    return err


def main():
    print("=== NVFP4 weight round-trip (encode→decode error) ===")
    rng = np.random.default_rng(0)
    _roundtrip("gaussian 512x5376", rng.standard_normal((512, 5376)).astype(np.float32))
    t = rng.standard_normal((512, 5376)).astype(np.float32)
    t[rng.random(t.shape) < 0.01] *= 8.0                     # heavy-tailed (outliers)
    _roundtrip("heavy-tail 512x5376", t)
    if len(sys.argv) >= 3:
        path, K = sys.argv[1], int(sys.argv[2])
        raw = np.fromfile(path, dtype=np.float32)
        N = raw.size // K
        _roundtrip(path.split("/")[-1], raw[: N * K].reshape(N, K))


if __name__ == "__main__":
    main()
