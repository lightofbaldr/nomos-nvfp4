"""Wire protocol for the nomos engine daemon <-> serve client (local UNIX socket).

The kernel logits-engine runs in ONE persistent daemon process (holds the GPU); the serve
is a thin client that crosses this socket for prefill/step/etc. and does sampling+grammar on
its side. Restarting the serve never touches the GPU → no model reload, no GB10 leak window.

Frame: [4-byte little-endian length][body]. Bodies are op-tagged (1 byte). Logits responses
carry the compiled profile's vocab-sized float32 vector (~1MB) — negligible over a localhost
UNIX socket vs the per-token compute.
"""
from __future__ import annotations

import socket
import struct

VOCAB = 262144
DEFAULT_SOCK = "/tmp/nomos_engine.sock"

# op codes (request)
OP_PREFILL = 1       # i32 n, i32[n] ids        -> i32 rc, f32[VOCAB] logits   (full, resets cache)
OP_STEP = 2          # i32 token                -> i32 rc, f32[VOCAB] logits
OP_PREFILL_CONT = 3  # i32 n, i32[n] ids        -> i32 rc, f32[VOCAB] logits
OP_SET_LEN = 4       # i32 n                    -> i32 rc
OP_CACHE_LEN = 5     # (none)                   -> i32 cache_len
OP_PING = 6          # (none)                   -> i32 0
OP_PREFILL_REUSE = 7 # i32 n, i32[n] ids        -> i32 rc, f32[VOCAB] logits   (reuse cached prefix)
OP_SPEC_STREAM = 8   # ids + max/vb + stop ids -> accepted-token frames; n=0 end
OP_DFLASH_LOAD = 9   # char[] dir               -> i32 rc
OP_MODEL_ID = 10     # (none)                   -> i32 compiled model profile id


def send_msg(sock: socket.socket, body: bytes) -> None:
    sock.sendall(struct.pack("<I", len(body)) + body)


def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> bytes | None:
    head = recv_exact(sock, 4)
    if head is None:
        return None
    (length,) = struct.unpack("<I", head)
    return recv_exact(sock, length)
