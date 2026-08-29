"""Pure-Mojo equivalents of the libops.so kernels (Phase 2, CUDA → Mojo migration).

Phase 2A: trivial elementwise kernels — no shared memory, no reductions.
Phase 2B: reduction kernels (rmsnorm family, qk_norm) — separate file with smem.

Mirrors ops_kernel.cu kernel for kernel. Each function takes a DeviceContext
plus raw UInt64 device pointers (matching the existing call sites in
gemma4_unified.mojo). The pointers are reinterpreted as UnsafePointer at
the kernel-launch boundary.

Pattern for trivial elementwise kernels:
  - One thread per element
  - 256 threads/block, ceil(n/256) blocks

This module is a drop-in replacement for the elementwise wrappers in
lib/gpu_ops.mojo (the FFI shim). The wrapper signatures are deliberately
similar so call sites can flip with minimal change.

After Phase 5 lands DeviceContext throughout gemma4_unified.mojo, the
UInt64 → UnsafePointer reinterpret can disappear; signatures will accept
DeviceBuffer directly.
"""

from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim, grid_dim
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer
from std.math import exp, log, cos, sin, tanh, floor, ceil


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — pointer reinterpret
# ─────────────────────────────────────────────────────────────────────────────

@always_inline
def _as_f32_ptr(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


@always_inline
def _as_bf16_ptr(addr: UInt64) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](unsafe_from_address=Int(addr))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Trivial elementwise kernels
# ═════════════════════════════════════════════════════════════════════════════

# ── residual_add: dst[i] += src[i] ───────────────────────────────────────────
def residual_add_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = dst[i] + src[i]


def gpu_residual_add_mojo(ctx: DeviceContext, d_dst: UInt64, d_src: UInt64, n: Int) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[residual_add_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_dst), _as_f32_ptr(d_src), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# ── elementwise_mul: dst[i] = a[i] * b[i] ────────────────────────────────────
def elementwise_mul_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[i] = a[i] * b[i]


def gpu_elementwise_mul_mojo(ctx: DeviceContext, d_dst: UInt64, d_a: UInt64,
                              d_b: UInt64, n: Int) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[elementwise_mul_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_dst), _as_f32_ptr(d_a), _as_f32_ptr(d_b), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


def residual_add_scaled_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
    scale: Float32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var added = Float32(dst[i] + src[i])
        dst[i] = added * scale


def gpu_residual_add_scaled_mojo(
    ctx: DeviceContext, d_dst: UInt64, d_src: UInt64, n: Int, scale: Float32
) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[residual_add_scaled_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_dst), _as_f32_ptr(d_src), Int32(n), scale,
        grid_dim=blocks, block_dim=threads,
    )


# ── scalar_mul: x[i] *= scalar ───────────────────────────────────────────────
def scalar_mul_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    scalar: Float32,
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        x[i] = x[i] * scalar


def gpu_scalar_mul_mojo(ctx: DeviceContext, d_x: UInt64, scalar: Float32, n: Int) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[scalar_mul_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), scalar, Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# ── gelu: tanh-approximate Gemma-4 hidden_activation ─────────────────────────
def gelu_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var v = x[i]
        # gelu(v) = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
        var c0: Float32 = 0.79788456     # sqrt(2/pi)
        var c1: Float32 = 0.044715
        var inner = c0 * (v + c1 * v * v * v)
        x[i] = 0.5 * v * (1.0 + tanh(inner))


def gpu_gelu_mojo(ctx: DeviceContext, d_x: UInt64, n: Int) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[gelu_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


def gelu_mul_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    other: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var v = gate[i]
        var c0: Float32 = 0.79788456
        var c1: Float32 = 0.044715
        var inner = c0 * (v + c1 * v * v * v)
        var gelu = Float32(0.5 * v * (1.0 + tanh(inner)))
        dst[i] = gelu * other[i]


def gpu_gelu_mul_mojo(
    ctx: DeviceContext, d_dst: UInt64, d_gate: UInt64, d_other: UInt64, n: Int
) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[gelu_mul_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_dst), _as_f32_ptr(d_gate), _as_f32_ptr(d_other), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# Muse-Glimmer MLP: dst = silu(gate) * other.
def silu_mul_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    other: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var v = gate[i]
        dst[i] = (v / (1.0 + exp(-v))) * other[i]


def gpu_silu_mul_mojo(
    ctx: DeviceContext, d_dst: UInt64, d_gate: UInt64, d_other: UInt64, n: Int
) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[silu_mul_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_dst), _as_f32_ptr(d_gate), _as_f32_ptr(d_other), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# Muse-Glimmer attention gate: attn *= sigmoid(gate_proj(layer_input)).
def sigmoid_mul_inplace_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        x[i] = x[i] / (1.0 + exp(-gate[i]))


def gpu_sigmoid_mul_inplace_mojo(
    ctx: DeviceContext, d_x: UInt64, d_gate: UInt64, n: Int
) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[sigmoid_mul_inplace_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_gate), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# ── silu: x[i] = x[i] * sigmoid(x[i])  (SwiGLU gate; Llama-style MLPs) ────────
# Ported from M3's eagle3 commit (6c09c79) for the EAGLE-3 drafter SwiGLU.
def silu_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var v = x[i]
        # silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
        x[i] = v / (1.0 + exp(-v))


def gpu_silu_mojo(ctx: DeviceContext, d_x: UInt64, n: Int) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[silu_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Indexed elementwise kernels (embed loaders)
# ═════════════════════════════════════════════════════════════════════════════

# ── embed_load: x[i] = table[token_id*d + i] * scale ─────────────────────────
def embed_load_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    table: UnsafePointer[Float32, MutAnyOrigin],
    token_id_arg: Int32,
    d_arg: Int32,
    scale: Float32,
):
    var token_id = Int(token_id_arg)
    var d = Int(d_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < d:
        x[i] = table[token_id * d + i] * scale


def gpu_embed_load_mojo(ctx: DeviceContext, d_x: UInt64, d_table: UInt64,
                        token_id: Int, d: Int, scale: Float32) raises:
    var threads = 256
    var blocks = (d + threads - 1) // threads
    var k = ctx.compile_function[embed_load_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_table), Int32(token_id), Int32(d), scale,
        grid_dim=blocks, block_dim=threads,
    )


def embed_load_bf16_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    table: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    token_id_arg: Int32,
    d_arg: Int32,
    scale: Float32,
):
    var token_id = Int(token_id_arg)
    var d = Int(d_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < d:
        x[i] = Float32(table[token_id * d + i]) * scale


def gpu_embed_load_bf16_mojo(
    ctx: DeviceContext,
    d_x: UInt64,
    d_table: UInt64,
    token_id: Int,
    d: Int,
    scale: Float32,
) raises:
    var threads = 256
    var blocks = (d + threads - 1) // threads
    var k = ctx.compile_function[embed_load_bf16_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_bf16_ptr(d_table), Int32(token_id), Int32(d), scale,
        grid_dim=blocks, block_dim=threads,
    )


# ── embed_copy_fp32: x[i] = src[i] * scale (source already on device) ────────
def embed_copy_fp32_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    d_arg: Int32,
    scale: Float32,
):
    var d = Int(d_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < d:
        x[i] = src[i] * scale


def gpu_embed_copy_fp32_mojo(ctx: DeviceContext, d_x: UInt64, d_src: UInt64,
                              d: Int, scale: Float32) raises:
    var threads = 256
    var blocks = (d + threads - 1) // threads
    var k = ctx.compile_function[embed_copy_fp32_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), _as_f32_ptr(d_src), Int32(d), scale,
        grid_dim=blocks, block_dim=threads,
    )


# Qwen3.5 stores each attention head's projection as [query, gate].  The
# projection therefore cannot be split with one contiguous memcpy: rows are
# [h0.q, h0.g, h1.q, h1.g, ...].
def split_q_gate_interleaved_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    raw: UnsafePointer[Float32, MutAnyOrigin],
    qd_arg: Int32,
    hd_arg: Int32,
    n_arg: Int32,
):
    var qd = Int(qd_arg)
    var hd = Int(hd_arg)
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var row = i // qd
        var col = i - row * qd
        var head = col // hd
        var lane = col - head * hd
        var src = row * (2 * qd) + head * (2 * hd) + lane
        q[i] = raw[src]
        gate[i] = raw[src + hd]


def gpu_split_q_gate_interleaved_mojo(
    ctx: DeviceContext, d_q: UInt64, d_gate: UInt64, d_raw: UInt64,
    rows: Int, qd: Int, hd: Int,
) raises:
    var n = rows * qd
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var k = ctx.compile_function[split_q_gate_interleaved_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_q), _as_f32_ptr(d_gate), _as_f32_ptr(d_raw),
        Int32(qd), Int32(hd), Int32(n),
        grid_dim=blocks, block_dim=threads,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. RoPE — per-head position-dependent rotation
# ═════════════════════════════════════════════════════════════════════════════

def rope_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    pos_arg: Int32,
    n_heads_arg: Int32,
    hd_arg: Int32,
    theta: Float32,
    rope_dim_arg: Int32,
):
    var pos = Int(pos_arg)
    var n_heads = Int(n_heads_arg)
    var hd = Int(hd_arg)
    var rope_dim = Int(rope_dim_arg)
    """Interleaved-pair RoPE (parity with ops_kernel.cu rope_kernel).

    Note: this uses (q[2i], q[2i+1]) pairs, matching the CUDA kernel.
    The CPU reference in lib/gemma4_ops.mojo:rope uses rotate_half
    (q[i], q[i+half]). The CUDA path has been producing coherent text
    so we keep parity here.
    """
    var head = block_idx.x
    var idx = thread_idx.x + block_idx.y * block_dim.x
    var rope_half = rope_dim // 2  # number of rotation pairs (= rope_angles)
    var hd_half = hd // 2          # pair offset (matches JAX rotate_half split at head_dim/2)
    if head >= n_heads or idx >= rope_half:
        return

    var row_off = head * hd
    # Rotate-half convention to match HF/JAX Gemma-4 apply_rotary_pos_emb.
    # Pair offset is hd/2 (the rotate_half split point), NOT rope_dim/2.
    # Frequency denominator is hd (head_dim), NOT rope_dim. For sliding
    # layers rope_dim == hd so these collapse to the same; for full-attn
    # layers (proportional RoPE, rope_dim < hd) they differ.
    var exponent = (2.0 * Float32(idx)) / Float32(hd)
    var ln_theta = log(theta)
    var inv_freq = exp(-exponent * ln_theta)
    var angle = Float32(pos) * inv_freq
    var c = cos(angle)
    var s = sin(angle)
    var a = x[row_off + idx]
    var b = x[row_off + idx + hd_half]
    x[row_off + idx]            = a * c - b * s
    x[row_off + idx + hd_half]  = a * s + b * c


def gpu_rope_mojo(ctx: DeviceContext, d_x: UInt64, pos: Int, n_heads: Int,
                  hd: Int, theta: Float32, rope_dim: Int) raises:
    var n_pairs = rope_dim // 2
    var threads = 128 if n_pairs >= 128 else n_pairs
    if threads <= 0:
        threads = 1
    var blocks_y = (n_pairs + threads - 1) // threads
    var k = ctx.compile_function[rope_kernel]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), Int32(pos), Int32(n_heads), Int32(hd), theta, Int32(rope_dim),
        grid_dim=(n_heads, blocks_y), block_dim=threads,
    )


# ── Batched RoPE: x is [S, n_heads, hd]; token t at position base_pos + t ─────
def rope_kernel_batched(
    x: UnsafePointer[Float32, MutAnyOrigin],
    base_pos_arg: Int32,
    S_arg: Int32,
    n_heads_arg: Int32,
    hd_arg: Int32,
    theta: Float32,
    rope_dim_arg: Int32,
):
    var base_pos = Int(base_pos_arg)
    var S = Int(S_arg)
    var n_heads = Int(n_heads_arg)
    var hd = Int(hd_arg)
    var rope_dim = Int(rope_dim_arg)
    var token = block_idx.z
    var head = block_idx.x
    var idx = thread_idx.x + block_idx.y * block_dim.x
    var rope_half = rope_dim // 2
    var hd_half = hd // 2
    if token >= S or head >= n_heads or idx >= rope_half:
        return
    var pos = base_pos + token
    var row_off = (token * n_heads + head) * hd
    var exponent = (2.0 * Float32(idx)) / Float32(hd)
    var ln_theta = log(theta)
    var inv_freq = exp(-exponent * ln_theta)
    var angle = Float32(pos) * inv_freq
    var c = cos(angle)
    var s = sin(angle)
    var a = x[row_off + idx]
    var b = x[row_off + idx + hd_half]
    x[row_off + idx]           = a * c - b * s
    x[row_off + idx + hd_half] = a * s + b * c


def gpu_rope_batched_mojo(ctx: DeviceContext, d_x: UInt64, base_pos: Int, S: Int,
                          n_heads: Int, hd: Int, theta: Float32, rope_dim: Int) raises:
    var n_pairs = rope_dim // 2
    var threads = 128 if n_pairs >= 128 else n_pairs
    if threads <= 0:
        threads = 1
    var blocks_y = (n_pairs + threads - 1) // threads
    var k = ctx.compile_function[rope_kernel_batched]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), Int32(base_pos), Int32(S), Int32(n_heads), Int32(hd), theta, Int32(rope_dim),
        grid_dim=(n_heads, blocks_y, S), block_dim=threads,
    )


# ── YaRN RoPE for target models ─────────────────────────────────────────────────────
# Mirrors transformers.modeling_rope_utils._compute_yarn_parameters. The
# correction bounds are derived on device from beta/theta/original_max rather
# than copied from a particular model (the DSpark helper's [14,29] is not OLMo).
def yarn_rope_kernel_batched(
    x: UnsafePointer[Float32, MutAnyOrigin],
    base_pos_arg: Int32, S_arg: Int32, n_heads_arg: Int32, hd_arg: Int32,
    theta: Float32, factor: Float32, original_max_arg: Int32,
    beta_fast: Float32, beta_slow: Float32, attention_factor: Float32,
):
    var S = Int(S_arg)
    var n_heads = Int(n_heads_arg)
    var hd = Int(hd_arg)
    var token = block_idx.z
    var head = block_idx.x
    var idx = thread_idx.x + block_idx.y * block_dim.x
    var half = hd // 2
    if token >= S or head >= n_heads or idx >= half:
        return

    # HF truncate=True: floor(low), ceil(high), clamped to [0, dim-1].
    var dim = Float32(hd)
    var original_max = Float32(Int(original_max_arg))
    var denom = Float32(2.0) * log(theta)
    var low_f = dim * log(original_max / (beta_fast * Float32(2.0 * 3.141592653589793))) / denom
    var high_f = dim * log(original_max / (beta_slow * Float32(2.0 * 3.141592653589793))) / denom
    var low = floor(low_f)
    var high = ceil(high_f)
    if low < 0.0: low = 0.0
    if high > dim - 1.0: high = dim - 1.0
    if high == low: high += 0.001

    var ramp = (Float32(idx) - low) / (high - low)
    if ramp < 0.0: ramp = 0.0
    if ramp > 1.0: ramp = 1.0
    var exponent = Float32(2 * idx) / dim
    var inv_extrap = exp(-exponent * log(theta))
    var inv_interp = inv_extrap / factor
    # HF: interpolation*ramp + extrapolation*(1-ramp).
    var inv_freq = inv_interp * ramp + inv_extrap * (1.0 - ramp)
    var angle = Float32(Int(base_pos_arg) + token) * inv_freq
    var c = cos(angle) * attention_factor
    var s = sin(angle) * attention_factor
    var off = (token * n_heads + head) * hd
    var a = x[off + idx]
    var b = x[off + idx + half]
    x[off + idx] = a * c - b * s
    x[off + idx + half] = a * s + b * c


def gpu_yarn_rope_batched_mojo(
    ctx: DeviceContext, d_x: UInt64, base_pos: Int, S: Int, n_heads: Int,
    hd: Int, theta: Float32, factor: Float32, original_max: Int,
    beta_fast: Float32, beta_slow: Float32, attention_factor: Float32,
) raises:
    var threads = 128 if hd // 2 >= 128 else hd // 2
    if threads <= 0: threads = 1
    var blocks_y = (hd // 2 + threads - 1) // threads
    var k = ctx.compile_function[yarn_rope_kernel_batched]()
    ctx.enqueue_function(
        k, _as_f32_ptr(d_x), Int32(base_pos), Int32(S), Int32(n_heads),
        Int32(hd), theta, factor, Int32(original_max), beta_fast, beta_slow,
        attention_factor, grid_dim=(n_heads, blocks_y, S), block_dim=threads,
    )


def gpu_yarn_rope_mojo(
    ctx: DeviceContext, d_x: UInt64, pos: Int, n_heads: Int, hd: Int,
    theta: Float32, factor: Float32, original_max: Int,
    beta_fast: Float32, beta_slow: Float32, attention_factor: Float32,
) raises:
    gpu_yarn_rope_batched_mojo(
        ctx, d_x, pos, 1, n_heads, hd, theta, factor, original_max,
        beta_fast, beta_slow, attention_factor,
    )


# ── bf16-weight GEMV (EAGLE-3 gold/wiring mode, M=1) ─────────────────────────
# out[N] = W_bf16[N,K] @ in_f32[K], fp32 accumulate. Weights are the EXACT
# checkpoint bytes (bf16 safetensors widened->fp32 blob->load_to_gpu_bf16 is a
# lossless round trip), activation stays fp32 -> this is the quant-BYPASS path
# for the G2 wiring-parity gate vs the HF Eagle3DraftModel reference. One warp
# per output row (the nvfp4_gemv_kernel dispatch shape); gate path, not perf.
def bf16w_gemv_kernel(
    w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],   # [N, K] bf16
    inp: UnsafePointer[Float32, MutAnyOrigin],                # [K] fp32
    outp: UnsafePointer[Float32, MutAnyOrigin],               # [N] fp32
    K_arg: Int32,
    N_arg: Int32,
):
    var K = Int(K_arg)
    var N = Int(N_arg)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var n = Int(block_idx.x) * 4 + warp_id        # 4 warps per block, one row each
    if n >= N:
        return
    var lane = Int(thread_idx.x) % WARP_SIZE
    var acc = Scalar[DType.float32](0.0)
    var base = n * K
    var i = lane
    while i < K:
        acc += w[base + i].cast[DType.float32]() * inp[i]
        i += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        acc += shuffle_xor(acc, UInt32(off))
        off //= 2
    if lane == 0:
        outp[n] = Float32(acc)


def gpu_matmul_bf16w_gemv_dev(
    ctx: DeviceContext, d_out: UInt64, d_in: UInt64, d_w_bf16: UInt64,
    K: Int, N: Int,
) raises:
    var k = ctx.compile_function[bf16w_gemv_kernel]()
    var blocks = (N + 3) // 4
    ctx.enqueue_function(
        k, _as_bf16_ptr(d_w_bf16), _as_f32_ptr(d_in), _as_f32_ptr(d_out),
        Int32(K), Int32(N),
        grid_dim=blocks, block_dim=4 * WARP_SIZE,
    )
