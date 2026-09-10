"""Standalone depth loader check; no new shipping FFI export.

argv: stem K N enabled expected_nvfp4. Malformed/missing weights must raise.
"""
from std.sys import argv
from lib.breeze_model import _dd_load_projection
from lib.cuda import cuda_free

def main() raises:
    var args=argv()
    if len(args)!=6:raise Error("usage: probe stem K N enabled expected_nvfp4")
    var gs=List[Float32]();var ags=List[Float32]()
    var p=_dd_load_projection(args[1],Int(args[2]),Int(args[3]),Int(args[4])==1,gs,ags)
    if p==0:raise Error("null loaded pointer")
    cuda_free(p)
    if len(gs)!=1 or len(ags)!=1:raise Error("unaligned scale lists")
    if (gs[0]!=0)!=(Int(args[5])==1):raise Error("wrong selected format")
    print("PASS depth loader: nvfp4 =",gs[0]!=0,"global_scale =",gs[0])
