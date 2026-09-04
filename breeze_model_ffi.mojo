from std.memory import UnsafePointer, alloc
from lib.breeze_model import BreezeModel

def _bm_cstr(ptr:Int64) raises->String:
    if ptr==0:return String("")
    var p=UnsafePointer[UInt8,MutUntrackedOrigin](unsafe_from_address=Int(ptr));var s=String("");var i=0
    while i<4096 and p[i]!=0:s+=chr(Int(p[i]));i+=1
    return s

@export
def nomos_breeze_model_init(path:Int64)->Int64:
    try:var p=alloc[BreezeModel](1);p.init_pointee_move(BreezeModel(_bm_cstr(path)));return Int64(Int(p))
    except e:print("[nomos_breeze_model_init EXC]",e);return Int64(0)

@export
def nomos_breeze_model_prefill_lane(handle:Int64,lane:Int32,embeds:Int64,S:Int32,layers:Int64,final:Int64,logits:Int64)->Int32:
    if handle==0 or embeds==0 or logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].prefill_lane(Int(lane),UInt64(embeds),Int(S),UInt64(layers),UInt64(final),UInt64(logits));return 0
    except e:print("[nomos_breeze_model_prefill_lane EXC]",e);return -99

@export
def nomos_breeze_model_step_backbone(handle:Int64,lane:Int32,codes:Int64,logits:Int64)->Int32:
    if handle==0 or codes==0 or logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].step_backbone(Int(lane),UInt64(codes),UInt64(logits));return 0
    except e:print("[nomos_breeze_model_step_backbone EXC]",e);return -99

@export
def nomos_breeze_model_depth(handle:Int64,codes:Int64,backbone_hidden:Int64,depth_logits:Int64)->Int32:
    if handle==0 or codes==0 or backbone_hidden==0 or depth_logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].depth(UInt64(codes),UInt64(backbone_hidden),UInt64(depth_logits));return 0
    except e:print("[nomos_breeze_model_depth EXC]",e);return -99

@export
def nomos_breeze_model_step(handle:Int64,lane:Int32,frame_codes:Int64,lm_logits:Int64,depth_logits:Int64)->Int32:
    if handle==0 or frame_codes==0 or lm_logits==0 or depth_logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].step(Int(lane),UInt64(frame_codes),UInt64(lm_logits),UInt64(depth_logits));return 0
    except e:print("[nomos_breeze_model_step EXC]",e);return -99

@export
def nomos_breeze_model_depth_lane(handle:Int64,lane:Int32,frame_codes:Int64,depth_logits:Int64)->Int32:
    if handle==0 or frame_codes==0 or depth_logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].depth_lane(Int(lane),UInt64(frame_codes),UInt64(depth_logits));return 0
    except e:print("[nomos_breeze_model_depth_lane EXC]",e);return -99

@export
def nomos_breeze_model_depth_begin(handle:Int64,lane:Int32,frame_codes:Int64,logits:Int64)->Int32:
    if handle==0 or logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].depth_begin(Int(lane),UInt64(frame_codes),UInt64(logits));return 0
    except e:print("[nomos_breeze_model_depth_begin EXC]",e);return -99

@export
def nomos_breeze_model_depth_advance(handle:Int64,lane:Int32,input_codebook:Int32,frame_codes:Int64,logits:Int64)->Int32:
    if handle==0 or logits==0:return -1
    try:var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].depth_advance(Int(lane),Int(input_codebook),UInt64(frame_codes),UInt64(logits));return 0
    except e:print("[nomos_breeze_model_depth_advance EXC]",e);return -99

@export
def nomos_breeze_model_free(handle:Int64)->Int32:
    if handle==0:return 0
    var p=UnsafePointer[BreezeModel,MutUntrackedOrigin](unsafe_from_address=Int(handle));p[0].free();p.free();return 0
