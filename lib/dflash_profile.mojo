"""Compiled DFlash assistant profile and front-door metadata validation.

The assistant is a second model. Its hidden size and vocabulary are target-facing
interfaces, but its MLP, heads, taps, and scalars come from the assistant profile.
Disk metadata is validation-only: absent fields are skipped, present fields must match.
"""

from std.collections import List
from std.ffi import external_call, c_int, c_size_t

from lib.model_config import (
    D, VOCAB, TOTAL_LAYERS,
    DRAFTER_MLP, DRAFTER_TAPS, DRAFTER_Q_HEADS, DRAFTER_KV_HEADS, DRAFTER_BLOCK,
    DRAFTER_WEIGHTS_BF16,
    DRAFTER_MASK_TOKEN_ID, DRAFTER_ROPE_THETA, DRAFTER_RMS_EPS,
    DRAFTER_SOFTCAP, DRAFTER_SLIDING_WINDOW,
    DRAFTER_TAP_0, DRAFTER_TAP_1, DRAFTER_TAP_2,
    DRAFTER_TAP_3, DRAFTER_TAP_4, DRAFTER_TAP_5,
    EAGLE3_TAPS, EAGLE3_TAP_0, EAGLE3_TAP_1, EAGLE3_TAP_2,
)


comptime DFLASH_PROFILE_LAYERS = 5
comptime DFLASH_PROFILE_KV_HEADS = DRAFTER_KV_HEADS
comptime DFLASH_PROFILE_HEAD_DIM = 128
comptime DFLASH_PROFILE_BLOCK = DRAFTER_BLOCK
comptime DFLASH_METADATA_MAX_BYTES = 1024 * 1024


def dflash_profile_tap_layers() -> List[Int]:
    """Return exactly DRAFTER_TAPS entries; unused -1 slots are never scanned."""
    var taps = List[Int]()
    if DRAFTER_TAPS > 0: taps.append(DRAFTER_TAP_0)
    if DRAFTER_TAPS > 1: taps.append(DRAFTER_TAP_1)
    if DRAFTER_TAPS > 2: taps.append(DRAFTER_TAP_2)
    if DRAFTER_TAPS > 3: taps.append(DRAFTER_TAP_3)
    if DRAFTER_TAPS > 4: taps.append(DRAFTER_TAP_4)
    if DRAFTER_TAPS > 5: taps.append(DRAFTER_TAP_5)
    return taps^


def eagle3_profile_tap_layers() -> List[Int]:
    """Return exactly EAGLE3_TAPS entries; unused -1 slots are never scanned."""
    var taps = List[Int]()
    if EAGLE3_TAPS > 0: taps.append(EAGLE3_TAP_0)
    if EAGLE3_TAPS > 1: taps.append(EAGLE3_TAP_1)
    if EAGLE3_TAPS > 2: taps.append(EAGLE3_TAP_2)
    return taps^


def _read_small_text(path: String) raises -> List[UInt8]:
    """Read a small metadata file. Missing file returns empty; malformed I/O raises."""
    var pb = List[UInt8](capacity=path.byte_length() + 1)
    for i in range(path.byte_length()):
        pb.append(path.as_bytes()[i])
    pb.append(0)
    var fd = external_call["nomos_open", c_int](pb.unsafe_ptr(), c_int(0), c_int(0))
    if Int(fd) < 0:
        return List[UInt8]()
    var size = Int(external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(2)))
    _ = external_call["nomos_lseek", c_size_t](fd, c_size_t(0), c_int(0))
    if size <= 0 or size > DFLASH_METADATA_MAX_BYTES:
        _ = external_call["nomos_close", c_int](fd)
        raise Error("DFlash metadata unreadable/oversized: " + path + " bytes=" + String(size))
    var data = List[UInt8](capacity=size + 1)
    for _ in range(size):
        data.append(0)
    var got = Int(external_call["nomos_pread", c_size_t](
        fd, data.unsafe_ptr(), c_size_t(size), c_size_t(0)
    ))
    _ = external_call["nomos_close", c_int](fd)
    if got != size:
        raise Error("DFlash metadata short read: " + path + " got=" + String(got) + " expected=" + String(size))
    data.append(0)
    return data^


def _is_json_space(c: UInt8) -> Bool:
    return c == UInt8(32) or c == UInt8(9) or c == UInt8(10) or c == UInt8(13)


def _json_value_start(ref data: List[UInt8], key: String) -> Int:
    """Find the value after an exact quoted JSON key. Returns -1 when absent."""
    var kb = key.as_bytes()
    var kl = key.byte_length()
    var n = len(data)
    if kl == 0 or n < kl + 3:
        return -1
    var i = 0
    while i + kl + 2 < n:
        if data[i] == UInt8(34):
            var same = True
            var k = 0
            while k < kl:
                if data[i + 1 + k] != kb[k]:
                    same = False
                    break
                k += 1
            if same and data[i + 1 + kl] == UInt8(34):
                var j = i + kl + 2
                while j < n and _is_json_space(data[j]):
                    j += 1
                if j < n and data[j] == UInt8(58):
                    j += 1
                    while j < n and _is_json_space(data[j]):
                        j += 1
                    return j if j < n else -1
        i += 1
    return -1


def _json_int_at(ref data: List[UInt8], start: Int, field: String) raises -> Int:
    var i = start
    var sign = 1
    if data[i] == UInt8(45):
        sign = -1
        i += 1
    var value = 0
    var digits = 0
    while i < len(data) and data[i] >= UInt8(48) and data[i] <= UInt8(57):
        value = value * 10 + Int(data[i] - UInt8(48))
        digits += 1
        i += 1
    if digits == 0:
        raise Error("DFlash metadata field is not an integer: " + field)
    return sign * value


def _json_float_at(ref data: List[UInt8], start: Int, field: String) raises -> Float64:
    var i = start
    var sign = Float64(1.0)
    if data[i] == UInt8(45):
        sign = Float64(-1.0)
        i += 1
    var value = Float64(0.0)
    var digits = 0
    while i < len(data) and data[i] >= UInt8(48) and data[i] <= UInt8(57):
        value = value * 10.0 + Float64(Int(data[i] - UInt8(48)))
        digits += 1
        i += 1
    if i < len(data) and data[i] == UInt8(46):
        i += 1
        var place = Float64(0.1)
        while i < len(data) and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            value += Float64(Int(data[i] - UInt8(48))) * place
            place *= 0.1
            digits += 1
            i += 1
    if digits == 0:
        raise Error("DFlash metadata field is not numeric: " + field)
    if i < len(data) and (data[i] == UInt8(101) or data[i] == UInt8(69)):
        i += 1
        var exp_sign = 1
        if data[i] == UInt8(45):
            exp_sign = -1
            i += 1
        elif data[i] == UInt8(43):
            i += 1
        var exponent = 0
        var exp_digits = 0
        while i < len(data) and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            exponent = exponent * 10 + Int(data[i] - UInt8(48))
            exp_digits += 1
            i += 1
        if exp_digits == 0:
            raise Error("DFlash metadata field has malformed exponent: " + field)
        while exponent > 0:
            value = value * 10.0 if exp_sign > 0 else value / 10.0
            exponent -= 1
    return sign * value


def _json_int_array_at(ref data: List[UInt8], start: Int, field: String) raises -> List[Int]:
    if data[start] != UInt8(91):
        raise Error("DFlash metadata field is not an array: " + field)
    var out = List[Int]()
    var i = start + 1
    while i < len(data):
        while i < len(data) and _is_json_space(data[i]):
            i += 1
        if i < len(data) and data[i] == UInt8(93):
            return out^
        var value_start = i
        if i < len(data) and data[i] == UInt8(45):
            i += 1
        var digits = 0
        while i < len(data) and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            digits += 1
            i += 1
        if digits == 0:
            raise Error("DFlash metadata array contains a non-integer: " + field)
        out.append(_json_int_at(data, value_start, field))
        while i < len(data) and _is_json_space(data[i]):
            i += 1
        if i < len(data) and data[i] == UInt8(44):
            i += 1
            continue
        if i < len(data) and data[i] == UInt8(93):
            return out^
        raise Error("DFlash metadata array is malformed: " + field)
    raise Error("DFlash metadata array is unterminated: " + field)


def _check_int_if_present(
    ref data: List[UInt8], source: String, key: String, expected: Int
) raises -> Int:
    var start = _json_value_start(data, key)
    if start < 0:
        return 0
    var got = _json_int_at(data, start, key)
    if got != expected:
        raise Error("DFlash profile mismatch: " + source + " field=" + key + " disk=" + String(got) + " compiled=" + String(expected))
    return 1


def _check_float_if_present(
    ref data: List[UInt8], source: String, key: String, expected: Float64
) raises -> Int:
    var start = _json_value_start(data, key)
    if start < 0:
        return 0
    var got = _json_float_at(data, start, key)
    var delta = got - expected
    if delta < 0.0:
        delta = -delta
    var magnitude = expected if expected >= 0.0 else -expected
    var tolerance = Float64(1.0e-9) * (magnitude if magnitude > 1.0 else Float64(1.0))
    if delta > tolerance:
        raise Error("DFlash profile mismatch: " + source + " field=" + key + " disk=" + String(got) + " compiled=" + String(expected))
    return 1


def _check_taps_if_present(
    ref data: List[UInt8], source: String, key: String
) raises -> Int:
    var start = _json_value_start(data, key)
    if start < 0:
        return 0
    var got = _json_int_array_at(data, start, key)
    var expected = dflash_profile_tap_layers()
    if len(got) != len(expected):
        raise Error("DFlash profile mismatch: " + source + " field=" + key + " disk_count=" + String(len(got)) + " compiled_count=" + String(len(expected)))
    for i in range(len(expected)):
        if got[i] != expected[i]:
            raise Error("DFlash profile mismatch: " + source + " field=" + key + " index=" + String(i) + " disk=" + String(got[i]) + " compiled=" + String(expected[i]))
    return 1


def _validate_metadata(ref data: List[UInt8], source: String) raises -> Int:
    """Validate the union of known config.json and convert_manifest.json fields."""
    var checked = 0
    checked += _check_int_if_present(data, source, "intermediate_size", DRAFTER_MLP)
    checked += _check_int_if_present(data, source, "hidden_size", D)
    checked += _check_int_if_present(data, source, "num_attention_heads", DRAFTER_Q_HEADS)
    checked += _check_int_if_present(data, source, "num_hidden_layers", DFLASH_PROFILE_LAYERS)
    checked += _check_int_if_present(data, source, "num_key_value_heads", DFLASH_PROFILE_KV_HEADS)
    checked += _check_int_if_present(data, source, "vocab_size", VOCAB)
    checked += _check_int_if_present(data, source, "num_target_layers", TOTAL_LAYERS)
    checked += _check_int_if_present(data, source, "block_size", DFLASH_PROFILE_BLOCK)
    checked += _check_int_if_present(data, source, "mask_token_id", DRAFTER_MASK_TOKEN_ID)
    checked += _check_float_if_present(data, source, "rope_theta", Float64(DRAFTER_ROPE_THETA))
    checked += _check_float_if_present(data, source, "rms_norm_eps", Float64(DRAFTER_RMS_EPS))
    checked += _check_int_if_present(data, source, "sliding_window", DRAFTER_SLIDING_WINDOW)
    checked += _check_float_if_present(data, source, "final_logit_softcapping", Float64(DRAFTER_SOFTCAP))
    checked += _check_taps_if_present(data, source, "target_layer_ids")

    checked += _check_int_if_present(data, source, "hidden", D)
    checked += _check_int_if_present(data, source, "layers", DFLASH_PROFILE_LAYERS)
    checked += _check_int_if_present(data, source, "n_layers", DFLASH_PROFILE_LAYERS)
    checked += _check_int_if_present(data, source, "heads", DRAFTER_Q_HEADS)
    checked += _check_int_if_present(data, source, "kv_heads", DFLASH_PROFILE_KV_HEADS)
    checked += _check_int_if_present(data, source, "vocab", VOCAB)
    checked += _check_taps_if_present(data, source, "taps")
    return checked


def validate_dflash_profile_metadata(dir: String) raises:
    """Validate every metadata file present; missing files/fields narrow the check."""
    var total = 0
    var config_path = dir + "config.json"
    var config = _read_small_text(config_path)
    if len(config) > 0:
        total += _validate_metadata(config, config_path)
    var manifest_path = dir + "convert_manifest.json"
    var manifest = _read_small_text(manifest_path)
    if len(manifest) > 0:
        total += _validate_metadata(manifest, manifest_path)
    var dspark_manifest_path = dir + "manifest.json"
    var dspark_manifest = _read_small_text(dspark_manifest_path)
    if len(dspark_manifest) > 0:
        total += _validate_metadata(dspark_manifest, dspark_manifest_path)
    @parameter
    if DRAFTER_WEIGHTS_BF16:
        if total == 0:
            raise Error("DFlash BF16 assistant metadata missing/unrecognized under " + dir)
    print("[dflash] profile metadata OK: checked", total, "present field(s)")
