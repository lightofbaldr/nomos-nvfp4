"""Standalone Breeze Qwen3TTSTokenizerV2 codec ABI."""

from std.memory import UnsafePointer, alloc
from lib.breeze_codec import BreezeCodec


def _read_cstr(ptr: Int64) raises -> String:
    if ptr == 0:
        return String("")
    var data = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ptr))
    var result = String("")
    var i = 0
    while data[i] != 0 and i < 4096:
        result += chr(Int(data[i]))
        i += 1
    return result


@export
def nomos_breeze_codec_init(weights_dir_ptr: Int64) -> Int64:
    try:
        var ptr = alloc[BreezeCodec](1)
        ptr.init_pointee_move(BreezeCodec(_read_cstr(weights_dir_ptr)))
        return Int64(Int(ptr))
    except e:
        print("[nomos_breeze_codec_init EXC]", e)
        return Int64(0)


@export
def nomos_breeze_codec_frontend(
    handle: Int64,
    codes_ptr: Int64,
    frames: Int32,
    quantized_ptr: Int64,
    preconv_ptr: Int64,
) -> Int32:
    """Decode [1,16,T] int64 codes through RVQ + causal pre-conv.

    Outputs are host FP32 channel-major [1,512,T] and [1,1024,T].
    This debug/parity boundary becomes an internal stage of the final waveform
    entrypoint once the remainder of M1 is gated.
    """
    if handle == 0 or codes_ptr == 0 or quantized_ptr == 0 or preconv_ptr == 0:
        return Int32(-1)
    try:
        var ptr = UnsafePointer[BreezeCodec, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_frontend(
            UInt64(codes_ptr), Int(frames), UInt64(quantized_ptr), UInt64(preconv_ptr)
        )
        return Int32(0)
    except e:
        print("[nomos_breeze_codec_frontend EXC]", e)
        return Int32(-99)


@export
def nomos_breeze_codec_transformer_input(
    handle: Int64,
    preconv_ptr: Int64,
    frames: Int32,
    output_ptr: Int64,
) -> Int32:
    """Debug/parity seam: [1,1024,T] channel-major -> [1,T,512]."""
    if handle == 0 or preconv_ptr == 0 or output_ptr == 0:
        return Int32(-1)
    try:
        var ptr = UnsafePointer[BreezeCodec, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_transformer_input(
            UInt64(preconv_ptr), Int(frames), UInt64(output_ptr)
        )
        return Int32(0)
    except e:
        print("[nomos_breeze_codec_transformer_input EXC]", e)
        return Int32(-99)


@export
def nomos_breeze_codec_transformer(
    handle: Int64,
    input_ptr: Int64,
    frames: Int32,
    layer0_ptr: Int64,
    layer7_ptr: Int64,
    output_ptr: Int64,
) -> Int32:
    """Debug/parity seam for the complete 8-layer pre-transformer."""
    if handle == 0 or input_ptr == 0 or layer0_ptr == 0 or layer7_ptr == 0 or output_ptr == 0:
        return Int32(-1)
    try:
        var ptr = UnsafePointer[BreezeCodec, MutUntrackedOrigin](unsafe_from_address=Int(handle))
        ptr[0].run_transformer(
            UInt64(input_ptr), Int(frames), UInt64(layer0_ptr),
            UInt64(layer7_ptr), UInt64(output_ptr),
        )
        return Int32(0)
    except e:
        print("[nomos_breeze_codec_transformer EXC]", e)
        return Int32(-99)


@export
def nomos_breeze_codec_free(handle: Int64) -> Int32:
    if handle == 0:
        return Int32(0)
    var ptr = UnsafePointer[BreezeCodec, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    ptr[0].free()
    ptr.free()
    return Int32(0)
