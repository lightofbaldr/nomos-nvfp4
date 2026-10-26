"""KernelClient — thin socket client to the engine daemon. Same interface as KernelLM (the
decode/sampling/grammar logic comes from _GenMixin), so the serve is backend-agnostic. The
serve uses this in daemon mode → restarting the serve never touches the GPU.
"""
from __future__ import annotations

import os
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire                       # noqa: E402
from host import _GenMixin, VOCAB  # noqa: E402


_NEXT = struct.pack("<B", wire.OP_SPEC_NEXT)


def _rc_error(what: str, rc: int) -> str:
    """The error for a request the daemon refused with rc. A daemon answers -1 to an opcode it does not know: one from
    before this serve's protocol (ledger R42), for the requests where a current daemon never answers -1."""
    if rc == -1:
        return (f"{what} rc=-1: the engine daemon predates this serve's protocol: restart the daemon and the serve "
                "together (RUNBOOK §7)")
    return f"{what} rc={rc}"


class KernelClient(_GenMixin):
    """One long-lived socket carrying a strict request/response conversation.

    That conversation has no resync marker, so ANY failure mid-exchange desynchronises it
    permanently: leftover frames get read as the NEXT request's response. Observed symptoms
    were garbage logits decoding to <pad>, and
        ValueError: buffer is smaller than requested size
    reaching the client as a truncated SSE body / TransferEncodingError. Every failure path
    therefore marks the connection dirty, and the next call reconnects rather than reading
    misaligned bytes.
    """

    def __init__(self, sock_path: str | None = None) -> None:
        self.sock_path = sock_path or os.environ.get("NOMOS_ENGINE_SOCKET", wire.DEFAULT_SOCK)
        self._dflash_dir = ""
        self._dirty = False
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.sock_path)

    def _reconnect(self) -> None:
        """Drop the desynced socket and open a clean one.

        Safe because all engine state (KV cache, drafter) lives in the daemon and survives
        reconnects — the same property that lets the serve restart without touching the GPU.
        Re-loading the drafter is a no-op now that dflash_load is idempotent.
        """
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.sock_path)
        self._dirty = False
        if self._dflash_dir:
            self.dflash_load(self._dflash_dir)

    def _ensure_conn(self) -> None:
        if self._dirty:
            self._reconnect()

    def _logits_call(self, body: bytes) -> np.ndarray:
        self._ensure_conn()
        try:
            wire.send_msg(self.sock, body)
            resp = wire.recv_msg(self.sock)
            if resp is None:
                raise RuntimeError("engine daemon closed the connection")
            (rc,) = struct.unpack_from("<i", resp, 0)
            if rc != 0:
                raise RuntimeError(f"engine daemon rc={rc}")
            # The payload is profile-sized (Gemma 262144, Muse 202048). Validate its
            # structure rather than imposing Gemma's size on every daemon response.
            payload_bytes = len(resp) - 4
            if payload_bytes <= 0 or payload_bytes % 4 != 0:
                raise RuntimeError(
                    f"malformed logits frame: got {len(resp)} bytes "
                    "(engine daemon connection desynced)")
            count = payload_bytes // 4
            if count > VOCAB:
                raise RuntimeError(f"oversized logits frame: {count} values (max {VOCAB})")
            return np.frombuffer(resp, dtype=np.float32, count=count, offset=4)
        except Exception:
            self._dirty = True
            raise

    def _rc_call(self, body: bytes) -> int:
        self._ensure_conn()
        try:
            wire.send_msg(self.sock, body)
            resp = wire.recv_msg(self.sock)
            if resp is None:
                raise RuntimeError("engine daemon closed the connection")
            (rc,) = struct.unpack_from("<i", resp, 0)
            return rc
        except Exception:
            self._dirty = True
            raise

    def prefill(self, ids: list[int]) -> np.ndarray:
        a = np.asarray(ids, dtype=np.int32)
        return self._logits_call(struct.pack("<Bi", wire.OP_PREFILL, len(a)) + a.tobytes())

    def step(self, token: int) -> np.ndarray:
        return self._logits_call(struct.pack("<Bi", wire.OP_STEP, int(token)))

    def prefill_cont(self, suffix_ids: list[int]) -> np.ndarray:
        a = np.asarray(suffix_ids, dtype=np.int32)
        return self._logits_call(struct.pack("<Bi", wire.OP_PREFILL_CONT, len(a)) + a.tobytes())

    def set_cache_len(self, n: int) -> None:
        rc = self._rc_call(struct.pack("<Bi", wire.OP_SET_LEN, int(n)))
        if rc != 0:
            raise RuntimeError(f"set_cache_len rc={rc}")

    def cache_len(self) -> int:
        return self._rc_call(struct.pack("<B", wire.OP_CACHE_LEN))

    def model_id(self) -> int:
        """Return the model profile compiled into the daemon's kernel."""
        return self._rc_call(struct.pack("<B", wire.OP_MODEL_ID))

    def session_open(self) -> tuple[int, str]:
        """host.KernelLM.session_open, run by the daemon: (status, the store's canonical directory)."""
        self._ensure_conn()
        try:
            wire.send_msg(self.sock, struct.pack("<B", wire.OP_SESSION_OPEN))
            resp = wire.recv_msg(self.sock)
            if resp is None:
                raise RuntimeError("engine daemon closed the connection")
            (rc,) = struct.unpack_from("<i", resp, 0)
            if rc != 0 or len(resp) < 8:
                raise RuntimeError(_rc_error("session_open", rc))
            (status,) = struct.unpack_from("<i", resp, 4)
            return status, resp[8:].decode("utf-8", "surrogateescape")
        except Exception:
            self._dirty = True
            raise

    def session_prepare(self, ids: list[int], spec: bool = False) -> int:
        a = np.asarray(ids, dtype=np.int32)
        c = self._rc_call(struct.pack("<Bii", wire.OP_SESSION_PREPARE, 1 if spec else 0, len(a)) + a.tobytes())
        if c < 0:
            raise RuntimeError(_rc_error("session_prepare", c))     # the daemon answers its kernel's refusals with -99
        return c

    def session_commit(self) -> int:
        return self._rc_call(struct.pack("<B", wire.OP_SESSION_COMMIT))

    def session_stats(self) -> list[int]:
        self._ensure_conn()
        try:
            wire.send_msg(self.sock, struct.pack("<B", wire.OP_SESSION_STATS))
            resp = wire.recv_msg(self.sock)
            if resp is None:
                raise RuntimeError("engine daemon closed the connection")
            (rc,) = struct.unpack_from("<i", resp, 0)
            if rc != 0:
                raise RuntimeError(_rc_error("session_stats", rc))
            return list(struct.unpack_from("<10q", resp, 4))
        except Exception:
            self._dirty = True
            raise

    def dflash_load(self, dflash_dir: str) -> None:
        if not dflash_dir.endswith("/"):
            dflash_dir += "/"
        rc = self._rc_call(struct.pack("<B", wire.OP_DFLASH_LOAD) + dflash_dir.encode())
        if rc != 0:
            raise RuntimeError(f"dflash_load rc={rc} (dir={dflash_dir})")
        # Remembered so _reconnect() can restore spec decode on a fresh socket.
        self._dflash_dir = dflash_dir

    def spec_generate_stream(self, prompt_ids, max_new_tokens=1024, stop_ids=(), vb=8):
        """Run the whole DFlash loop daemon-side and yield its accepted tokens, PULLED one at a time (ledger R56; wire.py describes
        the exchange). The daemon parks its loop at each token and advances it only when this generator asks, and it asks only
        after its own consumer has come back for another token. A consumer that takes k tokens and closes leaves the daemon's
        loop at exactly the kth, so the KV session store settles the turn to what the client took."""
        self._ensure_conn()
        ids = np.asarray(prompt_ids, dtype=np.int32)
        stops = np.asarray(list(stop_ids), dtype=np.int32)
        body = (struct.pack("<Bi", wire.OP_SPEC_PULL, len(ids)) + ids.tobytes() +
                struct.pack("<iii", int(max_new_tokens), int(vb), len(stops)) +
                stops.tobytes())
        try:
            wire.send_msg(self.sock, body)
            while True:
                resp = wire.recv_msg(self.sock)
                if resp is None:
                    raise RuntimeError("engine daemon closed the spec stream")
                (n,) = struct.unpack_from("<i", resp, 0)
                if n < 0:
                    raise RuntimeError(_rc_error("engine daemon spec stream", n))
                if n == 0:
                    return
                if n != 1 or len(resp) != 8:
                    raise RuntimeError(f"malformed spec frame: n={n}, bytes={len(resp)}")
                (tok,) = struct.unpack_from("<i", resp, 4)
                yield int(tok)
                # Back from the consumer, which wants another token. Only now may the daemon advance its loop.
                wire.send_msg(self.sock, _NEXT)
        except BaseException:
            # BaseException deliberately: GeneratorExit is the common case here. Closing this generator mid-stream (the client
            # went away, a stop in OpenWebUI) leaves the daemon parked at its yield, waiting for the next-token message. That
            # stream has no other way to end: a next call on this socket would be read by the daemon as "not next", and it
            # would close the loop and drop the connection, failing that call (ledger R56). So poison the socket: the next
            # call reconnects instead (dflash_load, which _reconnect re-sends, is idempotent). A turn that ends on its stop
            # token does not come here: the serve drains the end frame after the stop token, and keeps the connection.
            self._dirty = True
            # Close NOW rather than lazily at the next call. The daemon is blocked reading our next message; the close is
            # what ends its wait. It then closes its loop where it is parked, at the last token we took, so the loop's finally
            # (the KV settle and the commit) runs, and only then does it accept our next connection. (When the daemon ran
            # the loop ahead of us, a stopped generation held the engine until it drained: 24.4 s measured.)
            try:
                self.sock.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    def shutdown(self) -> None:
        """Close the client socket. Does NOT free the daemon's engine — the daemon persists
        across serve restarts (that's the whole point: the GPU is never re-allocated)."""
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass
