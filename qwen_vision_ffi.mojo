"""Standalone Qwen3.5 vision Tier-0 shared-library ABI."""

from std.memory import UnsafePointer, alloc
from lib.qwen_vision import QwenVision


def _read_cstr(ptr: Int64) raises -> String:
    if ptr == 0: return String("")
    var p = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ptr))
    var out = String(""); var i = 0
    while p[i] != 0 and i < 4096:
        out += chr(Int(p[i])); i += 1
    return out


@export
def nomos_qwen_vision_init(weights_dir: Int64) -> Int64:
    try:
        var p = alloc[QwenVision](1)
        p.init_pointee_move(QwenVision(_read_cstr(weights_dir)))
        return Int64(Int(p))
    except e:
        print("[nomos_qwen_vision_init EXC]", e); return Int64(0)


@export
def nomos_qwen_vision_frontend(
    handle: Int64, pixels: Int64, pos_rows: Int64, S: Int32,
    patch_out: Int64, block_in: Int64,
) -> Int32:
    if handle == 0 or pixels == 0 or pos_rows == 0 or S <= 0 or patch_out == 0 or block_in == 0:
        return Int32(-1)
    try:
        var p = UnsafePointer[QwenVision, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        p[0].run_frontend(UInt64(pixels),UInt64(pos_rows),Int(S),UInt64(patch_out),UInt64(block_in))
        return Int32(0)
    except e:
        print("[nomos_qwen_vision_frontend EXC]", e); return Int32(-99)


@export
def nomos_qwen_vision_run(
    handle: Int64, hidden: Int64, position_ids: Int64, cu_seqlens: Int64,
    n_segments: Int32, S: Int32, block0: Int64, block13: Int64,
    block26: Int64, image_features: Int64,
) -> Int32:
    if (
        handle == 0 or hidden == 0 or position_ids == 0 or cu_seqlens == 0
        or n_segments <= 0 or S <= 0 or block0 == 0 or block13 == 0
        or block26 == 0 or image_features == 0
    ): return Int32(-1)
    try:
        var p = UnsafePointer[QwenVision, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        p[0].run_encoder(
            UInt64(hidden),UInt64(position_ids),UInt64(cu_seqlens),Int(n_segments),Int(S),
            UInt64(block0),UInt64(block13),UInt64(block26),UInt64(image_features),
        )
        return Int32(0)
    except e:
        print("[nomos_qwen_vision_run EXC]", e); return Int32(-99)


@export
def nomos_qwen_vision_free(handle: Int64) -> Int32:
    if handle == 0: return Int32(-1)
    try:
        var p = UnsafePointer[QwenVision, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        p[0].free(); p.destroy_pointee(); p.free(); return Int32(0)
    except e:
        print("[nomos_qwen_vision_free EXC]", e); return Int32(-99)
