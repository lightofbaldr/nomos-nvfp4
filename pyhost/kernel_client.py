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
            # Check the frame length explicitly: np.frombuffer's "buffer is smaller than
            # requested size" IS the desync signature, and naming it beats that bare ValueError.
            if len(resp) < 4 + 4 * VOCAB:
                raise RuntimeError(
                    f"short logits frame: got {len(resp)} bytes, want {4 + 4 * VOCAB} "
                    "(engine daemon connection desynced)")
            return np.frombuffer(resp, dtype=np.float32, count=VOCAB, offset=4)
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
        # Reuse-aware: the daemon prefix-matches against its cached_ids and only prefills the
        # new suffix (KV reuse). Cold/first request falls back to a full prefill inside the daemon.
        a = np.asarray(ids, dtype=np.int32)
        return self._logits_call(struct.pack("<Bi", wire.OP_PREFILL_REUSE, len(a)) + a.tobytes())

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

    def dflash_load(self, dflash_dir: str) -> None:
        if not dflash_dir.endswith("/"):
            dflash_dir += "/"
        rc = self._rc_call(struct.pack("<B", wire.OP_DFLASH_LOAD) + dflash_dir.encode())
        if rc != 0:
            raise RuntimeError(f"dflash_load rc={rc} (dir={dflash_dir})")
        # Remembered so _reconnect() can restore spec decode on a fresh socket.
        self._dflash_dir = dflash_dir

    def spec_generate_stream(self, prompt_ids, max_new_tokens=1024, stop_ids=(), vb=8):
        """Run the whole DFlash loop daemon-side and yield its accepted token frames."""
        self._ensure_conn()
        ids = np.asarray(prompt_ids, dtype=np.int32)
        stops = np.asarray(list(stop_ids), dtype=np.int32)
        body = (struct.pack("<Bi", wire.OP_SPEC_STREAM, len(ids)) + ids.tobytes() +
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
                    raise RuntimeError(f"engine daemon spec stream rc={n}")
                if n == 0:
                    return
                expected = 4 + 4 * n
                if len(resp) != expected:
                    raise RuntimeError(f"malformed spec frame: n={n}, bytes={len(resp)}")
                tokens = np.frombuffer(resp, dtype=np.int32, count=n, offset=4)
                for tok in tokens:
                    yield int(tok)
        except BaseException:
            # BaseException deliberately: GeneratorExit is the common case here. Stopping a
            # generation in OpenWebUI (or any client disconnect) closes this generator while
            # the daemon is still writing token frames, and every unread frame would be
            # misread as the next request's response. Poison the socket so the next call
            # reconnects; a partially-consumed stream is exactly the desync we cannot detect.
            self._dirty = True
            # Close NOW rather than lazily at the next call. Until this socket closes, the
            # daemon happily keeps generating the abandoned stream into the socket buffer and
            # holds the engine: a follow-up request measured 24.4s waiting for a stopped
            # generation to drain. Closing makes the daemon's next frame write fail with
            # EPIPE, which ends its loop immediately.
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
