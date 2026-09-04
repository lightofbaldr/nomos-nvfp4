from std.memory import UnsafePointer, alloc
from lib.breeze_encoder import BreezeEncoder

def _read_cstr_encoder(ptr:Int64) raises -> String:
    if ptr==0:return String("")
    var p=UnsafePointer[UInt8,MutUntrackedOrigin](unsafe_from_address=Int(ptr));var s=String("");var i=0
    while i<4096 and p[i]!=0:s+=chr(Int(p[i]));i+=1
    return s

@export
def nomos_breeze_encoder_init(path:Int64)->Int64:
    try:
        var p=alloc[BreezeEncoder](1);p.init_pointee_move(BreezeEncoder(_read_cstr_encoder(path)));return Int64(Int(p))
    except e:print("[nomos_breeze_encoder_init EXC]",e);return Int64(0)

@export
def nomos_breeze_encoder_run(handle:Int64,ids:Int64,S:Int32,layers:Int64,final:Int64)->Int32:
    if handle==0 or ids==0 or layers==0 or final==0:return Int32(-1)
    try:
        var p=UnsafePointer[BreezeEncoder,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].run(UInt64(ids),Int(S),UInt64(layers),UInt64(final));return Int32(0)
    except e:print("[nomos_breeze_encoder_run EXC]",e);return Int32(-99)

@export
def nomos_breeze_encoder_free(handle:Int64)->Int32:
    if handle==0:return Int32(0)
    var p=UnsafePointer[BreezeEncoder,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].free();p.free();return Int32(0)
