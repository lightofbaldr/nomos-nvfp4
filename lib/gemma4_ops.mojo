"""Per-layer math ops shared across the Gemma-4 precision configs.

These are the CPU-side primitives that wrap the per-layer forward pass.
Everything here is scalar Mojo — no cuBLAS, no SIMD. Correctness-first
reference implementations. The GPU attention kernel work will replace
most of this with fused device kernels, so keep this module easy to
read and easy to compare against a PyTorch reference.

Functions:
  rmsnorm              — RMS normalization with learnable weight, out-of-place
  rmsnorm_inplace      — same, in-place (used for post-attn/post-ff norms
                         that are fused with residual)
  rmsnorm_no_weight    — RMSNorm without scale, per-head (used on V)
  qk_norm              — per-head RMSNorm applied to Q or K before RoPE
  rope                 — rotary position embedding (supports partial RoPE
                         for the full-attention layers that only rotate a
                         25% slice of the head dim)
  gelu                 — tanh-approximate GeLU (matches PyTorch
                         gelu_pytorch_tanh, Gemma-4's hidden_activation)

Values are Float32 to match the FP32 reference we've been debugging against
(see verify_layer0.py for known-good outputs at layer 0 step 4).
"""

from std.math import exp, sqrt, cos, sin
from std.collections import List


def rmsnorm(mut out: List[Float32], inp: List[Float32], w: List[Float32],
            dim: Int, eps: Float32 = 1e-6):
    """out[i] = inp[i] * rsqrt(mean(inp**2) + eps) * w[i]."""
    var ss: Float32 = 0.0
    for i in range(dim): ss += inp[i] * inp[i]
    var ri: Float32 = 1.0 / sqrt(ss / Float32(dim) + eps)
    for i in range(dim): out[i] = inp[i] * ri * w[i]


def rmsnorm_inplace(mut x: List[Float32], w: List[Float32], dim: Int,
                    eps: Float32 = 1e-6):
    """In-place variant — used after attn/MLP output before the residual add."""
    var ss: Float32 = 0.0
    for i in range(dim): ss += x[i] * x[i]
    var ri: Float32 = 1.0 / sqrt(ss / Float32(dim) + eps)
    for i in range(dim): x[i] = x[i] * ri * w[i]


def rmsnorm_no_weight(mut x: List[Float32], offset: Int, dim: Int,
                      eps: Float32 = 1e-6):
    """RMSNorm without learnable scale (Gemma-4's V-norm, per-head).

    Normalizes `dim` values starting at `offset` inside `x`. Used to apply
    per-head V normalization without allocating a separate output buffer —
    the caller passes the whole V tensor and the head offset.
    """
    var ss: Float32 = 0.0
    for i in range(dim): ss += x[offset + i] * x[offset + i]
    var ri: Float32 = 1.0 / sqrt(ss / Float32(dim) + eps)
    for i in range(dim): x[offset + i] = x[offset + i] * ri


def qk_norm(mut data: List[Float32], w: List[Float32], num_heads: Int,
            head_dim: Int):
    """Apply per-head RMSNorm to Q or K before RoPE.

    Gemma-4 uses a learnable per-dimension scale that's shared across all
    heads; we cycle through `w` with `i % len(w)` to handle both head_dim
    matching and head_dim > len(w).
    """
    for h in range(num_heads):
        var ho = h * head_dim
        var ss: Float32 = 0.0
        for i in range(head_dim): ss += data[ho + i] * data[ho + i]
        var ri: Float32 = 1.0 / sqrt(ss / Float32(head_dim) + 1e-6)
        for i in range(head_dim):
            data[ho + i] = data[ho + i] * ri * w[i % len(w)]


def rope(mut q: List[Float32], pos: Int, nh: Int, hd: Int,
         theta: Float64 = 10000.0, rope_dim: Int = 0):
    """Rotary position embeddings using the split-half convention.

    If rope_dim == 0, rotates the full head_dim. Otherwise only rotates the
    first `rope_dim` dimensions (partial RoPE for Gemma-4 full-attention
    layers, which use rope_dim = head_dim / 4).

    The rotation pairs (q[i], q[i + half]) — matching the rotate_half
    formulation used by HuggingFace Gemma — not (q[2i], q[2i+1]). This
    is not the interleaved format.
    """
    var rdim = hd if rope_dim == 0 else rope_dim
    var half = rdim // 2
    for h in range(nh):
        var ho = h * hd
        for i in range(half):
            var freq = 1.0 / (theta ** (Float64(2 * i) / Float64(rdim)))
            var angle = Float64(pos) * freq
            var co = Float32(cos(angle))
            var si = Float32(sin(angle))
            var x0 = q[ho + i]
            var x1 = q[ho + half + i]
            q[ho + i] = x0 * co - x1 * si
            q[ho + half + i] = x0 * si + x1 * co
        # Dims beyond rope_dim pass through unchanged.


def gelu(mut x: List[Float32], dim: Int):
    """Approximate GeLU (tanh-based), matching PyTorch's gelu_pytorch_tanh.

    GeLU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))

    The tanh is computed via `(e^2t - 1) / (e^2t + 1)` with `t` clamped to
    [-10, 10] before the exp to prevent overflow on large activations (the
    unclamped form NaNs on ~x > 4 in FP32).
    """
    for i in range(dim):
        var v = x[i]
        var t = Float32(0.7978845608) * (v + Float32(0.044715) * v * v * v)
        if t > 10.0: t = 10.0
        if t < -10.0: t = -10.0
        var e2t = exp(2.0 * t)
        var tanh_val = (e2t - 1.0) / (e2t + 1.0)
        x[i] = v * 0.5 * (1.0 + tanh_val)
