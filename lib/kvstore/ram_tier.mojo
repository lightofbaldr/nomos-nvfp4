"""RAM tier of the KV session store: chunk payloads in pinned host memory (heap memory in CPU tests)."""
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from lib.cuda import cuda_host_alloc, cuda_free_host


struct RamTier(Movable):
    var used: Int
    var peak: Int       # the most bytes ever held at once: what the store's hard budget is checked against
    var pinned: Bool

    def __init__(out self, pinned: Bool):
        self.used = 0
        self.peak = 0
        self.pinned = pinned

    def alloc(mut self, nbytes: Int) -> UInt64:
        """A chunk-sized buffer, or 0 on failure. The budget is the store's (cfg.ram_bytes): it evicts before it asks."""
        var addr: UInt64
        if self.pinned:
            addr = cuda_host_alloc(nbytes)
        else:
            addr = UInt64(Int(unsafe_alloc[UInt8](nbytes)))
        if addr != 0:
            self.used += nbytes
            if self.used > self.peak:
                self.peak = self.used
        return addr

    def free(mut self, addr: UInt64, nbytes: Int):
        if addr == 0:
            return
        if self.pinned:
            cuda_free_host(addr)
        else:
            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr)).unsafe_free()
        self.used -= nbytes
