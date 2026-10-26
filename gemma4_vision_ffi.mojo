"""Standalone Gemma-4 vision Tier-0 ABI."""

from std.memory import UnsafePointer, alloc
from lib.gemma4_vision import Gemma4Vision

def _read_cstr(ptr: Int64) raises -> String:
    if ptr==0: return String("")
    var p=UnsafePointer[UInt8,MutUntrackedOrigin](unsafe_from_address=Int(ptr)); var out=String(""); var i=0
    while p[i]!=0 and i<4096: out+=chr(Int(p[i])); i+=1
    return out

@export
def nomos_gemma4_vision_init(weights_dir: Int64) -> Int64:
    try:
        var p=alloc[Gemma4Vision](1); p.init_pointee_move(Gemma4Vision(_read_cstr(weights_dir))); return Int64(Int(p))
    except e: print("[nomos_gemma4_vision_init EXC]",e); return Int64(0)

@export
def nomos_gemma4_vision_frontend(handle:Int64,pixels:Int64,positions:Int64,S:Int32,patch_out:Int64)->Int32:
    if handle==0 or pixels==0 or positions==0 or S<=0 or patch_out==0: return Int32(-1)
    try:
        var p=UnsafePointer[Gemma4Vision,MutUntrackedOrigin](unsafe_from_address=Int(handle)); p[0].frontend(UInt64(pixels),UInt64(positions),Int(S),UInt64(patch_out)); return Int32(0)
    except e: print("[nomos_gemma4_vision_frontend EXC]",e); return Int32(-99)

@export
def nomos_gemma4_vision_run(handle:Int64,hidden:Int64,positions:Int64,S:Int32,out0:Int64,out13:Int64,out26:Int64,features:Int64,nvis_out:Int64)->Int32:
    if handle==0 or hidden==0 or positions==0 or S<=0 or out0==0 or out13==0 or out26==0 or features==0 or nvis_out==0: return Int32(-1)
    try:
        var p=UnsafePointer[Gemma4Vision,MutUntrackedOrigin](unsafe_from_address=Int(handle)); p[0].run(UInt64(hidden),UInt64(positions),Int(S),UInt64(out0),UInt64(out13),UInt64(out26),UInt64(features),UInt64(nvis_out)); return Int32(0)
    except e: print("[nomos_gemma4_vision_run EXC]",e); return Int32(-99)

@export
def nomos_gemma4_vision_free(handle:Int64)->Int32:
    if handle==0: return Int32(-1)
    try:
        var p=UnsafePointer[Gemma4Vision,MutUntrackedOrigin](unsafe_from_address=Int(handle)); p[0].free(); p.destroy_pointee(); p.free(); return Int32(0)
    except e: print("[nomos_gemma4_vision_free EXC]",e); return Int32(-99)
