"""Qwen3.5 GDN projection/scan/readout orchestration.

This is the one shared implementation used by prefill and decode. Inputs and
all work surfaces are fp32. The three large projections may be BF16 (shipped
fallback) or NVFP4 (Phase B); the six recurrence-critical small weights remain
BF16. State ownership remains in GdnStatePools.
"""

from max.gpu.host import DeviceContext
from lib.cublas import gpu_matmul_bf16_dev_batched
from lib.gemma4_layer import _mm_dev, _mm_dev_batched, W4A4Scratch
from lib.fp4_act import (
    gpu_quant_act_nvfp4_dec,
    gpu_matmul_nvfp4_w4a4_prequant_dev,
    _w4a4_prequant_off,
)
from lib.engine_init import _read_env_bytes
from lib.gdn_scan import (
    gpu_gdn_conv_scan,
    gpu_gdn_prepare_decay_beta,
    gpu_gdn_recurrent_scan,
    gpu_gdn_gated_rmsnorm,
)
from lib.model_profiles.qwen3_5 import (
    D, GDN_CONV_DIM, GDN_VALUE_DIM, GDN_NUM_V_HEADS, GDN_NORM_EPS,
)
from lib.cuda import cuda_memcpy


def _gdn_inproj_dedup_off() -> Bool:
    """NOMOS_GDN_INPROJ_DEDUP=0 restores the two independent decode calls."""
    var v = _read_env_bytes("NOMOS_GDN_INPROJ_DEDUP")
    return len(v) > 0 and v[0] == UInt8(48)


def gdn_forward_batched(
    ctx: DeviceContext,
    handle: UInt64,
    d_out: UInt64,             # [S,D] fp32
    d_normed: UInt64,          # [S,D] fp32, input-layernorm output
    d_qkv_raw: UInt64,         # [S,10240] fp32
    d_qkv_conv: UInt64,        # [S,10240] fp32
    d_z: UInt64,               # [S,6144] fp32; becomes gated-norm output
    d_a_g: UInt64,             # [S,48] fp32; a projection then g
    d_b_beta: UInt64,          # [S,48] fp32; b projection then beta
    d_core: UInt64,            # [S,6144] fp32 recurrent readout
    d_scratch_in_bf16: UInt64, # >= S*max(D,6144) bf16
    d_in_proj_qkv: UInt64,
    d_in_proj_z: UInt64,
    d_in_proj_a: UInt64,
    d_in_proj_b: UInt64,
    d_out_proj: UInt64,
    d_conv_weight: UInt64,
    d_a_log: UInt64,
    d_dt_bias: UInt64,
    d_norm_weight: UInt64,
    d_conv_state: UInt64,
    d_recurrent_state: UInt64,
    seq_len: Int,
    proj_nvfp4: Bool = False,
    qkv_gs: Float32 = 0.0,
    z_gs: Float32 = 0.0,
    out_gs: Float32 = 0.0,
    qkv_ags: Float32 = 0.0,
    z_ags: Float32 = 0.0,
    out_ags: Float32 = 0.0,
    d_weight_bf16_scratch: UInt64 = 0,
    w4a4: W4A4Scratch = W4A4Scratch(0, 0, 0, 0, 0, False, 0, 0),
    debug_row: Int = -1,
    debug_stage: Int = -1,
    debug_out: Int64 = 0,
    decode_order_scan: Bool = False,
) raises:
    """Run one complete Qwen GatedDeltaNet token-mixer for S rows."""
    if proj_nvfp4:
        if seq_len == 1:
            # QKV and Z are adjacent projections of the SAME immutable d_normed row.
            # When their activation-global calibration is identical, quantize that row
            # once and reuse the exact packed/scales/global bytes for both GEMMs. Keep
            # these calls adjacent: inserting any d_normed mutation between quantize and
            # the second projection invalidates this byte-equivalence contract.
            var dedup_inproj = (
                w4a4.on and Float64(ctx.default_device_info.compute) >= 12.0
                and qkv_gs != Float32(0.0) and z_gs != Float32(0.0)
                and qkv_ags == z_ags and not _w4a4_prequant_off()
                and not _gdn_inproj_dedup_off()
            )
            if dedup_inproj:
                gpu_quant_act_nvfp4_dec(
                    ctx, d_normed, w4a4.packed, w4a4.bs, w4a4.glob,
                    D, qkv_ags,
                )
                gpu_matmul_nvfp4_w4a4_prequant_dev(
                    ctx, d_qkv_raw, d_in_proj_qkv, qkv_gs,
                    w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad,
                    1, D, GDN_CONV_DIM,
                )
                gpu_matmul_nvfp4_w4a4_prequant_dev(
                    ctx, d_z, d_in_proj_z, z_gs,
                    w4a4.packed, w4a4.bs, w4a4.glob, w4a4.cpad,
                    1, D, GDN_VALUE_DIM,
                )
            else:
                _mm_dev(
                    ctx, handle, d_qkv_raw, d_normed, d_in_proj_qkv,
                    d_scratch_in_bf16, D, GDN_CONV_DIM,
                    d_weight_bf16_scratch, qkv_gs, qkv_ags, w4a4,
                    projection="gdn_in_proj_qkv",
                )
                _mm_dev(
                    ctx, handle, d_z, d_normed, d_in_proj_z,
                    d_scratch_in_bf16, D, GDN_VALUE_DIM,
                    d_weight_bf16_scratch, z_gs, z_ags, w4a4,
                    projection="gdn_in_proj_z",
                )
        else:
            _mm_dev_batched(
                ctx, handle, d_qkv_raw, d_normed, d_in_proj_qkv,
                d_scratch_in_bf16, seq_len, D, GDN_CONV_DIM,
                d_weight_bf16_scratch, qkv_gs, qkv_ags, w4a4,
                projection="gdn_in_proj_qkv",
            )
            _mm_dev_batched(
                ctx, handle, d_z, d_normed, d_in_proj_z,
                d_scratch_in_bf16, seq_len, D, GDN_VALUE_DIM,
                d_weight_bf16_scratch, z_gs, z_ags, w4a4,
                projection="gdn_in_proj_z",
            )
    else:
        gpu_matmul_bf16_dev_batched(
            ctx, handle, d_qkv_raw, d_normed, d_in_proj_qkv,
            d_scratch_in_bf16, seq_len, D, GDN_CONV_DIM,
        )
        gpu_matmul_bf16_dev_batched(
            ctx, handle, d_z, d_normed, d_in_proj_z,
            d_scratch_in_bf16, seq_len, D, GDN_VALUE_DIM,
        )
    # A/B are tiny [D -> 48] projections whose cuBLAS M>1 reduction differs
    # from the production M=1 decode tree by ~1e-6 relative.  That ULP-scale
    # seed enters the recurrent scan and amplifies into verify-vs-decode token
    # flips.  Preserve the exact M=1 arithmetic per row here; their ~1 MiB of
    # weights per GDN layer is negligible beside the large batched QKV/Z/out
    # projections, which remain byte-identical and weight-reusing above/below.
    for row in range(seq_len):
        gpu_matmul_bf16_dev_batched(
            ctx, handle,
            d_a_g + UInt64(row * GDN_NUM_V_HEADS * 4),
            d_normed + UInt64(row * D * 4),
            d_in_proj_a, d_scratch_in_bf16,
            1, D, GDN_NUM_V_HEADS,
        )
        gpu_matmul_bf16_dev_batched(
            ctx, handle,
            d_b_beta + UInt64(row * GDN_NUM_V_HEADS * 4),
            d_normed + UInt64(row * D * 4),
            d_in_proj_b, d_scratch_in_bf16,
            1, D, GDN_NUM_V_HEADS,
        )
    if debug_out != 0 and debug_row >= 0 and debug_row < seq_len:
        ctx.synchronize()
        if debug_stage == 19:
            cuda_memcpy(
                UInt64(debug_out),
                d_qkv_raw + UInt64(debug_row * GDN_CONV_DIM * 4),
                GDN_CONV_DIM * 4, 2,
            )
        elif debug_stage == 20:
            cuda_memcpy(
                UInt64(debug_out), d_z + UInt64(debug_row * GDN_VALUE_DIM * 4),
                GDN_VALUE_DIM * 4, 2,
            )
        elif debug_stage == 21:
            cuda_memcpy(
                UInt64(debug_out),
                d_a_g + UInt64(debug_row * GDN_NUM_V_HEADS * 4),
                GDN_NUM_V_HEADS * 4, 2,
            )
        elif debug_stage == 22:
            cuda_memcpy(
                UInt64(debug_out),
                d_b_beta + UInt64(debug_row * GDN_NUM_V_HEADS * 4),
                GDN_NUM_V_HEADS * 4, 2,
            )
    gpu_gdn_prepare_decay_beta(
        ctx, d_a_g, d_b_beta, d_a_g, d_b_beta, d_a_log, d_dt_bias,
        seq_len,
    )
    if debug_out != 0 and debug_row >= 0 and debug_row < seq_len:
        ctx.synchronize()
        if debug_stage == 23:
            cuda_memcpy(
                UInt64(debug_out),
                d_a_g + UInt64(debug_row * GDN_NUM_V_HEADS * 4),
                GDN_NUM_V_HEADS * 4, 2,
            )
        elif debug_stage == 24:
            cuda_memcpy(
                UInt64(debug_out),
                d_b_beta + UInt64(debug_row * GDN_NUM_V_HEADS * 4),
                GDN_NUM_V_HEADS * 4, 2,
            )
    if decode_order_scan:
        for row in range(seq_len):
            gpu_gdn_conv_scan(
                ctx,
                d_qkv_conv + UInt64(row * GDN_CONV_DIM * 4),
                d_qkv_raw + UInt64(row * GDN_CONV_DIM * 4),
                d_conv_weight, d_conv_state, 1,
            )
    else:
        gpu_gdn_conv_scan(
            ctx, d_qkv_conv, d_qkv_raw, d_conv_weight, d_conv_state,
            seq_len,
        )
    if debug_out != 0 and debug_row >= 0 and debug_row < seq_len and debug_stage == 25:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out),
            d_qkv_conv + UInt64(debug_row * GDN_CONV_DIM * 4),
            GDN_CONV_DIM * 4, 2,
        )
    if decode_order_scan:
        for row in range(seq_len):
            gpu_gdn_recurrent_scan(
                ctx,
                d_core + UInt64(row * GDN_VALUE_DIM * 4),
                d_qkv_conv + UInt64(row * GDN_CONV_DIM * 4),
                d_a_g + UInt64(row * GDN_NUM_V_HEADS * 4),
                d_b_beta + UInt64(row * GDN_NUM_V_HEADS * 4),
                d_recurrent_state, 1,
            )
    else:
        gpu_gdn_recurrent_scan(
            ctx, d_core, d_qkv_conv, d_a_g, d_b_beta,
            d_recurrent_state, seq_len,
        )
    if debug_out != 0 and debug_row >= 0 and debug_row < seq_len and debug_stage == 13:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_core + UInt64(debug_row * GDN_VALUE_DIM * 4),
            GDN_VALUE_DIM * 4, 2,
        )
    gpu_gdn_gated_rmsnorm(
        ctx, d_z, d_core, d_z, d_norm_weight, seq_len,
        Float32(GDN_NORM_EPS),
    )
    if debug_out != 0 and debug_row >= 0 and debug_row < seq_len and debug_stage == 14:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_z + UInt64(debug_row * GDN_VALUE_DIM * 4),
            GDN_VALUE_DIM * 4, 2,
        )
    if proj_nvfp4:
        if seq_len == 1:
            _mm_dev(
                ctx, handle, d_out, d_z, d_out_proj,
                d_scratch_in_bf16, GDN_VALUE_DIM, D,
                d_weight_bf16_scratch, out_gs, out_ags, w4a4,
                projection="gdn_out_proj",
            )
        else:
            _mm_dev_batched(
                ctx, handle, d_out, d_z, d_out_proj,
                d_scratch_in_bf16, seq_len, GDN_VALUE_DIM, D,
                d_weight_bf16_scratch, out_gs, out_ags, w4a4,
                projection="gdn_out_proj",
            )
    else:
        gpu_matmul_bf16_dev_batched(
            ctx, handle, d_out, d_z, d_out_proj,
            d_scratch_in_bf16, seq_len, GDN_VALUE_DIM, D,
        )
    if debug_out != 0 and debug_row >= 0 and debug_row < seq_len and debug_stage == 15:
        ctx.synchronize()
        cuda_memcpy(
            UInt64(debug_out), d_out + UInt64(debug_row * D * 4), D * 4, 2
        )
