"""Qwen3.5 Gated-DeltaNet causal-conv and recurrent scan kernels.

The kernels deliberately operate on already-projected fp32 work tensors.  Weight
projection/readout routing belongs to the Qwen engine; this module owns the two
state-producing operations and the native-bf16 -> fp32 debug cast.

State contracts:
  conv_state [10240, 4]             bf16, oldest -> newest, FULL K
  recurrent  [48, 128, 128]        bf16, [value_head, key_dim, value_dim]

All recurrence arithmetic is fp32.  State is rounded to bf16 only on the pool
read/write boundary, matching the model's storage/compute dtype split.
"""

from max.gpu.host import DeviceContext
from std.gpu.primitives import thread_idx, block_idx, block_dim
from std.gpu.primitives.warp import shuffle_xor, WARP_SIZE
from std.memory import UnsafePointer
from std.math import exp, sqrt, log

# This module is Qwen-only. Import its compile profile directly so the common
# Gemma/Muse build can compile the debug ABI without inventing non-Qwen scan
# dimensions. Engine routing remains gated by model_config.HAS_LINEAR_ATTENTION.
from lib.model_profiles.qwen3_5 import (
    GDN_NUM_K_HEADS,
    GDN_NUM_V_HEADS,
    GDN_KEY_HEAD_DIM,
    GDN_VALUE_HEAD_DIM,
    GDN_KEY_DIM,
    GDN_VALUE_DIM,
    GDN_CONV_DIM,
    GDN_CONV_KERNEL,
    GDN_GQA_EXPAND,
)


@always_inline
def _gdn_f32(addr: UInt64) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(addr))


@always_inline
def _gdn_bf16(
    addr: UInt64,
) -> UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    return UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
        unsafe_from_address=Int(addr)
    )


def gdn_conv_scan_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    projected_qkv: UnsafePointer[Float32, MutAnyOrigin],
    conv_weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    conv_state: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    seq_len_arg: Int32,
):
    """Depthwise causal conv + SiLU, one thread per channel."""
    var seq_len = Int(seq_len_arg)
    var channel = block_idx.x * block_dim.x + thread_idx.x
    if channel >= GDN_CONV_DIM:
        return

    # For token t and kernel offset k, relative input position is
    # t-(K-1-k).  Negative positions read the previous FULL-K state at
    # K+relative_pos: -3 -> 1, -1 -> 3.  State[0] is intentionally not read
    # by a new token; it exists to match HF's post-forward cache surface.
    for token in range(seq_len):
        var acc = Float32(0.0)
        comptime for k in range(GDN_CONV_KERNEL):
            var rel = token - (GDN_CONV_KERNEL - 1 - k)
            var x = Float32(0.0)
            if rel >= 0:
                x = projected_qkv[rel * GDN_CONV_DIM + channel]
            else:
                var state_age = GDN_CONV_KERNEL + rel
                if state_age >= 0:
                    x = Float32(
                        conv_state[channel * GDN_CONV_KERNEL + state_age]
                    )
            var w = Float32(
                conv_weight[channel * GDN_CONV_KERNEL + k]
            )
            acc += x * w
        # HF applies SiLU inside both causal_conv1d_fn and the reference
        # conv1d fallback.  Keeping it here makes the scan surface complete;
        # a caller cannot silently feed the raw convolution into recurrence.
        output[token * GDN_CONV_DIM + channel] = acc / (
            Float32(1.0) + exp(-acc)
        )

    # Commit the final full-K raw-input window.  For a short continuation,
    # carry the surviving suffix of the old state; otherwise take the last
    # four projected rows.  Compute every source before overwriting its slot.
    var next_state = SIMD[DType.bfloat16, GDN_CONV_KERNEL](0)
    comptime for age in range(GDN_CONV_KERNEL):
        var source = seq_len - GDN_CONV_KERNEL + age
        if source >= 0:
            next_state[age] = Scalar[DType.bfloat16](
                projected_qkv[source * GDN_CONV_DIM + channel]
            )
        else:
            var old_age = GDN_CONV_KERNEL + source
            if old_age >= 0:
                next_state[age] = conv_state[
                    channel * GDN_CONV_KERNEL + old_age
                ]
    comptime for age in range(GDN_CONV_KERNEL):
        conv_state[channel * GDN_CONV_KERNEL + age] = next_state[age]


def gpu_gdn_conv_scan(
    ctx: DeviceContext,
    d_out: UInt64,
    d_projected_qkv: UInt64,
    d_conv_weight: UInt64,
    d_conv_state: UInt64,
    seq_len: Int,
) raises:
    var threads = 256
    var blocks = (GDN_CONV_DIM + threads - 1) // threads
    var kernel = ctx.compile_function[gdn_conv_scan_kernel]()
    ctx.enqueue_function(
        kernel,
        _gdn_f32(d_out),
        _gdn_f32(d_projected_qkv),
        _gdn_bf16(d_conv_weight),
        _gdn_bf16(d_conv_state),
        Int32(seq_len),
        grid_dim=blocks,
        block_dim=threads,
    )


def gdn_recurrent_scan_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    conv_qkv: UnsafePointer[Float32, MutAnyOrigin],
    g: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    recurrent_state: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    seq_len_arg: Int32,
):
    """FP32 gated-delta recurrence, one thread per (V-head, V-column)."""
    var seq_len = Int(seq_len_arg)
    var flat = block_idx.x * block_dim.x + thread_idx.x
    if flat >= GDN_VALUE_DIM:
        return
    var value_head = flat // GDN_VALUE_HEAD_DIM
    var value_col = flat % GDN_VALUE_HEAD_DIM
    var key_head = value_head // GDN_GQA_EXPAND

    var state_col = SIMD[DType.float32, GDN_KEY_HEAD_DIM](0.0)
    comptime for key_col in range(GDN_KEY_HEAD_DIM):
        var state_idx = (
            (value_head * GDN_KEY_HEAD_DIM + key_col)
            * GDN_VALUE_HEAD_DIM
            + value_col
        )
        state_col[key_col] = Float32(recurrent_state[state_idx])

    var q_scale = Float32(1.0) / sqrt(Float32(GDN_KEY_HEAD_DIM))
    for token in range(seq_len):
        var q_raw = SIMD[DType.float32, GDN_KEY_HEAD_DIM](0.0)
        var k_raw = SIMD[DType.float32, GDN_KEY_HEAD_DIM](0.0)
        var q_ss = Float32(0.0)
        var k_ss = Float32(0.0)
        var row = token * GDN_CONV_DIM
        var q_base = key_head * GDN_KEY_HEAD_DIM
        var k_base = GDN_KEY_DIM + key_head * GDN_KEY_HEAD_DIM
        comptime for key_col in range(GDN_KEY_HEAD_DIM):
            var qv = conv_qkv[row + q_base + key_col]
            var kv = conv_qkv[row + k_base + key_col]
            q_raw[key_col] = qv
            k_raw[key_col] = kv
            q_ss += qv * qv
            k_ss += kv * kv

        var q_inv = Float32(1.0) / sqrt(q_ss + Float32(1e-6))
        var k_inv = Float32(1.0) / sqrt(k_ss + Float32(1e-6))
        var decay = exp(g[token * GDN_NUM_V_HEADS + value_head])
        var beta_v = beta[token * GDN_NUM_V_HEADS + value_head]
        var memory = Float32(0.0)
        comptime for key_col in range(GDN_KEY_HEAD_DIM):
            state_col[key_col] *= decay
            memory += state_col[key_col] * (k_raw[key_col] * k_inv)

        var v = conv_qkv[
            row + 2 * GDN_KEY_DIM
            + value_head * GDN_VALUE_HEAD_DIM + value_col
        ]
        var delta = (v - memory) * beta_v
        var readout = Float32(0.0)
        comptime for key_col in range(GDN_KEY_HEAD_DIM):
            state_col[key_col] += (k_raw[key_col] * k_inv) * delta
            readout += state_col[key_col] * (
                q_raw[key_col] * q_inv * q_scale
            )
        output[token * GDN_VALUE_DIM + flat] = readout

    comptime for key_col in range(GDN_KEY_HEAD_DIM):
        var state_idx = (
            (value_head * GDN_KEY_HEAD_DIM + key_col)
            * GDN_VALUE_HEAD_DIM
            + value_col
        )
        recurrent_state[state_idx] = Scalar[DType.bfloat16](
            state_col[key_col]
        )


def gpu_gdn_recurrent_scan(
    ctx: DeviceContext,
    d_out: UInt64,
    d_conv_qkv: UInt64,
    d_g: UInt64,
    d_beta: UInt64,
    d_recurrent_state: UInt64,
    seq_len: Int,
) raises:
    var threads = 256
    var blocks = (GDN_VALUE_DIM + threads - 1) // threads
    var kernel = ctx.compile_function[gdn_recurrent_scan_kernel]()
    ctx.enqueue_function(
        kernel,
        _gdn_f32(d_out),
        _gdn_f32(d_conv_qkv),
        _gdn_f32(d_g),
        _gdn_f32(d_beta),
        _gdn_bf16(d_recurrent_state),
        Int32(seq_len),
        grid_dim=blocks,
        block_dim=threads,
    )


@always_inline
def _gdn_softplus(x: Float32) -> Float32:
    # Stable softplus: avoid exp overflow while preserving the reference's
    # fp32 transform boundary.
    if x > Float32(20.0):
        return x
    if x < Float32(-20.0):
        return exp(x)
    return log(Float32(1.0) + exp(x))


def gdn_prepare_decay_beta_kernel(
    g_out: UnsafePointer[Float32, MutAnyOrigin],
    beta_out: UnsafePointer[Float32, MutAnyOrigin],
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    a_log: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dt_bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    seq_len_arg: Int32,
):
    """HF-exact fp32 gate transforms from projected a/b rows."""
    var seq_len = Int(seq_len_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < seq_len * GDN_NUM_V_HEADS:
        var head = i % GDN_NUM_V_HEADS
        var av = a[i] + Float32(dt_bias[head])
        g_out[i] = -exp(Float32(a_log[head])) * _gdn_softplus(av)
        var bv = b[i]
        beta_out[i] = Float32(1.0) / (Float32(1.0) + exp(-bv))


def gpu_gdn_prepare_decay_beta(
    ctx: DeviceContext,
    d_g_out: UInt64,
    d_beta_out: UInt64,
    d_a: UInt64,
    d_b: UInt64,
    d_a_log: UInt64,
    d_dt_bias: UInt64,
    seq_len: Int,
) raises:
    var n = seq_len * GDN_NUM_V_HEADS
    var threads = 256
    var kernel = ctx.compile_function[gdn_prepare_decay_beta_kernel]()
    ctx.enqueue_function(
        kernel,
        _gdn_f32(d_g_out),
        _gdn_f32(d_beta_out),
        _gdn_f32(d_a),
        _gdn_f32(d_b),
        _gdn_bf16(d_a_log),
        _gdn_bf16(d_dt_bias),
        Int32(seq_len),
        grid_dim=(n + threads - 1) // threads,
        block_dim=threads,
    )


def gdn_gated_rmsnorm_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    core: UnsafePointer[Float32, MutAnyOrigin],
    z: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_rows_arg: Int32,
    eps: Float32,
):
    """Per-head RMSNorm(core) * raw weight * SiLU(z), one warp/row."""
    var n_rows = Int(n_rows_arg)
    var row = block_idx.x
    var lane = thread_idx.x
    if row >= n_rows:
        return
    var base = row * GDN_VALUE_HEAD_DIM
    var ss = Float32(0.0)
    var d = lane
    while d < GDN_VALUE_HEAD_DIM:
        var v = core[base + d]
        ss += v * v
        d += WARP_SIZE
    var off = WARP_SIZE // 2
    while off > 0:
        ss += shuffle_xor(ss, UInt32(off))
        off //= 2
    var inv = Float32(1.0) / sqrt(
        ss / Float32(GDN_VALUE_HEAD_DIM) + eps
    )
    d = lane
    while d < GDN_VALUE_HEAD_DIM:
        var zv = z[base + d]
        var silu_z = zv / (Float32(1.0) + exp(-zv))
        output[base + d] = (
            core[base + d] * inv * Float32(weight[d]) * silu_z
        )
        d += WARP_SIZE


def gpu_gdn_gated_rmsnorm(
    ctx: DeviceContext,
    d_out: UInt64,
    d_core: UInt64,
    d_z: UInt64,
    d_weight: UInt64,
    seq_len: Int,
    eps: Float32,
) raises:
    var n_rows = seq_len * GDN_NUM_V_HEADS
    var kernel = ctx.compile_function[gdn_gated_rmsnorm_kernel]()
    ctx.enqueue_function(
        kernel,
        _gdn_f32(d_out),
        _gdn_f32(d_core),
        _gdn_f32(d_z),
        _gdn_bf16(d_weight),
        Int32(n_rows),
        eps,
        grid_dim=n_rows,
        block_dim=WARP_SIZE,
    )


def gdn_bf16_to_f32_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    inp: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n_arg: Int32,
):
    var n = Int(n_arg)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        output[i] = Float32(inp[i])


def gpu_gdn_bf16_to_f32(
    ctx: DeviceContext, d_out: UInt64, d_in: UInt64, n: Int
) raises:
    var threads = 256
    var blocks = (n + threads - 1) // threads
    var kernel = ctx.compile_function[gdn_bf16_to_f32_kernel]()
    ctx.enqueue_function(
        kernel,
        _gdn_f32(d_out),
        _gdn_bf16(d_in),
        Int32(n),
        grid_dim=blocks,
        block_dim=threads,
    )
