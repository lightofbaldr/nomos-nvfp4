#!/usr/bin/env python3
"""Compare base-greedy and DFlash output through /v1/chat/completions.

Requires a running engine daemon and DFLASH_DIR. The gate starts two short-lived HTTP
serve processes against that same daemon: base first, then spec. It compares the exact
UTF-8 bytes assembled from the streamed OpenAI content deltas.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("SPEC_GATE_PORT", "18091"))
URL = f"http://127.0.0.1:{PORT}"
PROMPT = os.environ.get("SPEC_GATE_PROMPT", "Explain why the sky is blue in one sentence.")


def _wait_ready(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"serve exited early with rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=1) as response:
                if json.load(response).get("ready") is True:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise RuntimeError("timed out waiting for serve readiness")


def _completion(spec: bool) -> tuple[bytes, str]:
    env = os.environ.copy()
    env.update({"NOMOS_PORT": str(PORT), "NOMOS_SERVE_SPEC": "1" if spec else "0"})
    proc = subprocess.Popen(
        ["pixi", "run", "python3", "-m", "uvicorn", "pyhost.serve:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        _wait_ready(proc)
        payload = json.dumps({
            "model": "nomos-gemma4-31b",
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0,
            "max_tokens": int(os.environ.get("SPEC_GATE_TOKENS", "64")),
            "enable_thinking": False,
            "stream": True,
        }).encode()
        request = urllib.request.Request(
            f"{URL}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        chunks: list[str] = []
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"]
                chunks.append(delta.get("content", ""))
        return "".join(chunks).encode(), ""
    finally:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()
        if spec and "DFlash speculative decode enabled" not in output:
            print(output, file=sys.stderr)
            raise RuntimeError("spec serve did not report DFlash enabled")


def main() -> int:
    if not os.environ.get("NOMOS_ENGINE_SOCKET"):
        raise RuntimeError("NOMOS_ENGINE_SOCKET must point to a running daemon")
    if not os.environ.get("DFLASH_DIR"):
        raise RuntimeError("DFLASH_DIR is required")
    base, _ = _completion(False)
    spec, _ = _completion(True)
    if base != spec:
        print(f"FAIL: base={base!r}\nspec={spec!r}")
        return 1
    print(f"SERVE LOSSLESS: {len(base)} bytes byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
