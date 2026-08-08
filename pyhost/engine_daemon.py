"""Persistent engine daemon — loads the Mojo logits-engine ONCE and serves prefill/step/
prefill_cont/set_len/cache_len over a UNIX socket. The ONLY process that touches the GPU.

It survives serve restarts, so the dev loop (restart the Python serve on every change) never
reloads the 16GB model and never re-allocates GPU memory → no GB10 unified-pool leak window,
and instant serve iteration. Run in tmux on the box that holds the .so:
  NOMOS_WEIGHT_NVFP4=0 NOMOS_PRECISION_BITS=4 NOMOS_KV_SWA=0 python3 pyhost/engine_daemon.py
"""
from __future__ import annotations

import os
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire                       # noqa: E402
from host import KernelLM         # noqa: E402


def _serve_conn(conn: socket.socket, lm: KernelLM, state: dict) -> None:
    while True:
        body = wire.recv_msg(conn)
        if body is None:
            return
        op = body[0]
        try:
            if op == wire.OP_PREFILL_REUSE:
                (n,) = struct.unpack_from("<i", body, 1)
                ids = np.frombuffer(body, dtype=np.int32, count=n, offset=5).tolist()
                if state["reuse"]:
                    lg, reused = lm.prefill_reuse(ids, state["cached"])
                    if reused:
                        print(f"[daemon] reuse P={reused}/{len(ids)} (suffix {len(ids) - reused})", flush=True)
                else:
                    lg = lm.prefill(ids)
                state["cached"] = list(ids)
                resp = struct.pack("<i", 0) + lg.tobytes()
            elif op == wire.OP_PREFILL or op == wire.OP_PREFILL_CONT:
                (n,) = struct.unpack_from("<i", body, 1)
                ids = np.frombuffer(body, dtype=np.int32, count=n, offset=5).tolist()
                if op == wire.OP_PREFILL:
                    lg = lm.prefill(ids)
                    state["cached"] = list(ids)
                else:
                    lg = lm.prefill_cont(ids)
                    state["cached"].extend(ids)
                resp = struct.pack("<i", 0) + lg.tobytes()
            elif op == wire.OP_STEP:
                (tok,) = struct.unpack_from("<i", body, 1)
                resp = struct.pack("<i", 0) + lm.step(tok).tobytes()
                state["cached"].append(int(tok))
            elif op == wire.OP_SET_LEN:
                (n,) = struct.unpack_from("<i", body, 1)
                lm.set_cache_len(n)
                state["cached"] = state["cached"][:n]
                resp = struct.pack("<i", 0)
            elif op == wire.OP_CACHE_LEN:
                resp = struct.pack("<i", lm.cache_len())
            elif op == wire.OP_PING:
                resp = struct.pack("<i", 0)
            elif op == wire.OP_DFLASH_LOAD:
                dflash_dir = body[1:].decode()
                lm.dflash_load(dflash_dir)
                resp = struct.pack("<i", 0)
            elif op == wire.OP_SPEC_STREAM:
                (n,) = struct.unpack_from("<i", body, 1)
                ids_end = 5 + 4 * n
                ids = np.frombuffer(body, dtype=np.int32, count=n, offset=5).tolist()
                max_tokens, vb, ns = struct.unpack_from("<iii", body, ids_end)
                stop_ids = np.frombuffer(body, dtype=np.int32, count=ns,
                                         offset=ids_end + 12).tolist()
                emitted: list[int] = []
                # The complete draft/verify/accept loop stays in this GPU-owning process.
                # Send accepted bursts as frames; never cross the socket per FFI call.
                block: list[int] = []
                for tok in lm.spec_generate_stream(ids, max_new_tokens=max_tokens,
                                                   stop_ids=stop_ids, vb=vb):
                    emitted.append(int(tok))
                    block.append(int(tok))
                    if len(block) >= vb:
                        wire.send_msg(conn, struct.pack("<i", len(block)) +
                                      np.asarray(block, dtype=np.int32).tobytes())
                        block.clear()
                if block:
                    wire.send_msg(conn, struct.pack("<i", len(block)) +
                                  np.asarray(block, dtype=np.int32).tobytes())
                # The target KV contains the prompt and all generated inputs except the
                # final predicted token, matching the bookkeeping of base generation.
                state["cached"] = list(ids) + emitted[:-1]
                wire.send_msg(conn, struct.pack("<i", 0))
                continue
            else:
                resp = struct.pack("<i", -1)
        except Exception as e:  # noqa: BLE001
            print(f"[daemon] op={op} EXC: {e}", flush=True)
            resp = struct.pack("<i", -99)
        wire.send_msg(conn, resp)


def main() -> None:
    sock_path = os.environ.get("NOMOS_ENGINE_SOCKET", wire.DEFAULT_SOCK)
    lm = KernelLM()  # loads the model (the one GPU allocation for the box's lifetime)
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    # cache-state tracking lives WITH the cache (in the daemon) so it survives serve restarts.
    state = {"cached": [], "reuse": os.environ.get("NOMOS_KV_REUSE", "1") != "0"}
    print(f"[daemon] READY on {sock_path}  (kv_reuse={state['reuse']})", flush=True)
    while True:
        conn, _ = srv.accept()
        print("[daemon] client connected", flush=True)
        try:
            _serve_conn(conn, lm, state)
        except Exception as e:  # noqa: BLE001
            print(f"[daemon] conn error: {e}", flush=True)
        finally:
            conn.close()
            print("[daemon] client disconnected", flush=True)


if __name__ == "__main__":
    main()
