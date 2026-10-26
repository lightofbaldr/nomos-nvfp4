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
OP_PREFILL_REUSE = 7 # i32 n, i32[n] ids        -> i32 rc, f32[VOCAB] logits   (legacy: a plain prefill; see OP_SESSION_PREPARE)
# 8 was OP_SPEC_STREAM, the push stream R56 retired. Its number is never reused: a serve and a daemon on either side of R56 each
# answer the other's speculative request with -1 (an opcode they do not know) and the client raises, instead of misreading frames
# (ledger R42: restart them together).
OP_DFLASH_LOAD = 9   # char[] dir               -> i32 rc
OP_MODEL_ID = 10     # (none)                   -> i32 compiled model profile id
OP_SESSION_PREPARE = 11  # i32 spec, i32 n, i32[n] ids  -> i32 cached prefix length c
OP_SESSION_COMMIT = 12   # (none)                        -> i32 tokens committed (or a negative code)
OP_SESSION_STATS = 13    # (none)                        -> i32 rc, i64[10] counters
OP_SPEC_PULL = 14    # i32 n, i32[n] ids, i32 max, i32 vb, i32 ns, i32[ns] stop ids -> a PULL stream, see below
OP_SPEC_NEXT = 15    # (no body) the one message a daemon accepts inside a spec stream: the next token, please
OP_SESSION_OPEN = 16     # (none) -> i32 rc 0, i32 status, utf-8[] the store's canonical directory (empty unless status 1)
# A daemon older than an opcode answers it with i32 -1 alone (ledger R42: restart the serve and the daemon together).

# The speculative stream (OP_SPEC_PULL) is pull-driven, like the base path (ledger R56). The daemon runs the whole draft/verify/
# accept loop in its own process, but it advances that loop only when the client asks:
#   client -> daemon   OP_SPEC_PULL request
#   daemon -> client   <ii> (1, token)         one token per frame; the loop is now parked at that token's yield
#   client -> daemon   OP_SPEC_NEXT            sent only when the client's own consumer comes back for another token
#   ...                                         (repeat)
#   daemon -> client   <i> 0                   the loop ended on its own; its finally (the KV settle and the commit) has run
#                      <i> n < 0               an error (-99: the loop raised; -1: the daemon does not know the opcode)
# If the client's next message is EOF, a connection error, or anything but OP_SPEC_NEXT, the daemon closes the loop where it is
# parked: at the yield of the last token the client took. A loop that ran ahead would hand out tokens the client never took,
# and the turn's KV (settled to what was handed out) would hold rows the client never saw.


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
