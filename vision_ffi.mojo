"""Standalone Tier-0 vision M1 shared-library ABI."""

from std.memory import UnsafePointer, alloc
from lib.vision_attention import VisionM1


def _read_cstr(ptr: Int64) raises -> String:
    if ptr == 0: return String("")
    var p = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ptr))
    var out = String("")
    var i = 0
    while p[i] != 0 and i < 4096:
        out += chr(Int(p[i])); i += 1
    return out


@export
def nomos_vision_m1_init(weights_dir_ptr: Int64) -> Int64:
    try:
        var ptr = alloc[VisionM1](1)
        ptr.init_pointee_move(VisionM1(_read_cstr(weights_dir_ptr)))
        return Int64(Int(ptr))
    except e:
        print("[nomos_vision_m1_init EXC]", e)
        return Int64(0)


@export
def nomos_vision_m1_run_layer(
    handle: Int64, layer: Int32, hidden_ptr: Int64, position_ids_ptr: Int64,
    cu_seqlens_ptr: Int64, n_segments: Int32, S: Int32, out_ptr: Int64,
) -> Int32:
    """Run one Muse vision encoder layer.

    hidden and position_ids must already be reordered by window_index.  Pass
    cu_window_seqlens for window layers and cu_seqlens for full layers.
    """
    if handle == 0 or hidden_ptr == 0 or position_ids_ptr == 0 or cu_seqlens_ptr == 0 or out_ptr == 0:
        return Int32(-1)
    try:
        var ptr = UnsafePointer[VisionM1, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_layer(
            Int(layer), UInt64(hidden_ptr), UInt64(position_ids_ptr),
            UInt64(cu_seqlens_ptr), Int(n_segments), Int(S), UInt64(out_ptr),
            0, UInt64(0),
        )
        return Int32(0)
    except e:
        print("[nomos_vision_m1_run_layer EXC]", e)
        return Int32(-99)


@export
def nomos_vision_m1_run_layer_debug(
    handle: Int64, layer: Int32, hidden_ptr: Int64, position_ids_ptr: Int64,
    cu_seqlens_ptr: Int64, n_segments: Int32, S: Int32, out_ptr: Int64,
    debug_stage: Int32, debug_ptr: Int64,
) -> Int32:
    if handle == 0 or debug_ptr == 0:
        return Int32(-1)
    try:
        var ptr = UnsafePointer[VisionM1, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_layer(
            Int(layer), UInt64(hidden_ptr), UInt64(position_ids_ptr),
            UInt64(cu_seqlens_ptr), Int(n_segments), Int(S), UInt64(out_ptr),
            Int(debug_stage), UInt64(debug_ptr),
        )
        return Int32(0)
    except e:
        print("[nomos_vision_m1_run_layer_debug EXC]", e)
        return Int32(-99)


@export
def nomos_vision_frontend(
    handle: Int64, pixel_values_ptr: Int64, pos_rows_ptr: Int64,
    window_index_ptr: Int64, S: Int32, patch_out_ptr: Int64,
    ln_pre_out_ptr: Int64, ln_pre_reordered_ptr: Int64,
) -> Int32:
    """Patch projection + supplied positional rows + ln_pre + window reorder.

    Learned positional interpolation deliberately remains host-side. All three
    output surfaces are host fp32; patch_out is the HF patch_embedder surface
    after positional addition, ln_pre values reflect the BF16 model boundary,
    and reordered output feeds nomos_vision_m2_run directly.
    """
    if (
        handle == 0 or pixel_values_ptr == 0 or pos_rows_ptr == 0
        or window_index_ptr == 0 or S <= Int32(0) or patch_out_ptr == 0
        or ln_pre_out_ptr == 0 or ln_pre_reordered_ptr == 0
    ):
        return Int32(-1)
    try:
        var ptr = UnsafePointer[VisionM1, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_frontend(
            UInt64(pixel_values_ptr), UInt64(pos_rows_ptr), UInt64(window_index_ptr),
            Int(S), UInt64(patch_out_ptr), UInt64(ln_pre_out_ptr),
            UInt64(ln_pre_reordered_ptr),
        )
        return Int32(0)
    except e:
        print("[nomos_vision_frontend EXC]", e)
        return Int32(-99)


@export
def nomos_vision_m2_run(
    handle: Int64, hidden_reordered_ptr: Int64, position_ids_ptr: Int64,
    cu_window_ptr: Int64, n_window_segments: Int32, cu_full_ptr: Int64,
    window_index_ptr: Int64, S: Int32, t: Int32, h: Int32, w: Int32,
    layer49_ptr: Int64, ln_post_ptr: Int64, encoder_hidden_ptr: Int64,
    image_features_ptr: Int64,
) -> Int32:
    """Run the complete Tier-0 Muse vision tower from reordered ln_pre.

    The four outputs are deliberately exposed at the first surfaces that can
    distinguish encoder, un-reorder/ln_post, channel-major pixel shuffle, and
    projector/scaleless-RMSNorm failures.
    """
    if (
        handle == 0 or hidden_reordered_ptr == 0 or position_ids_ptr == 0
        or cu_window_ptr == 0 or cu_full_ptr == 0 or window_index_ptr == 0
        or layer49_ptr == 0 or ln_post_ptr == 0 or encoder_hidden_ptr == 0
        or image_features_ptr == 0
    ):
        return Int32(-1)
    try:
        var ptr = UnsafePointer[VisionM1, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_m2(
            UInt64(hidden_reordered_ptr), UInt64(position_ids_ptr),
            UInt64(cu_window_ptr), Int(n_window_segments), UInt64(cu_full_ptr),
            UInt64(window_index_ptr), Int(S), Int(t), Int(h), Int(w),
            UInt64(layer49_ptr), UInt64(ln_post_ptr),
            UInt64(encoder_hidden_ptr), UInt64(image_features_ptr),
        )
        return Int32(0)
    except e:
        print("[nomos_vision_m2_run EXC]", e)
        return Int32(-99)


@export
def nomos_vision_m1_free(handle: Int64) -> Int32:
    if handle == 0: return Int32(0)
    var ptr = UnsafePointer[VisionM1, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    ptr[0].free()
    ptr.free()
    return Int32(0)
