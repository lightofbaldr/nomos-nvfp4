#!/usr/bin/env python3
"""A conversation longer than NOMOS_MAX_PROMPT_TOKENS goes through the real serve when chunked prefill is on, and its
next turn reuses the cached KV. NOMOS_MAX_SEQ comes from the host env (8192 on the laptop today), so the test stays
under it. GPU; starts and stops the serve itself.

ThirtyTwoKThroughTheServe (opt-in, KVSTORE_LONG32K=1) runs Step 6's proposed host env end to end: NOMOS_MAX_SEQ=32768,
NOMOS_PREFILL_CHUNK=256, NOMOS_KVSTORE=1, DFlash at temperature 0."""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from kvsession_testlib import ROOT

PORT = int(os.environ.get("NOMOS_PORT", "8091"))
DIR = ROOT / "build/kvstore_long_serve_test"


def post(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


class LongConversationServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Registered before anything can fail: if the serve fails to come up below, a daemon that did come up still holds the
        # GPU, and tearDownClass would not run after a failed setUpClass. Cleanups run last-registered first: `down`, and then
        # the store's files are removed (final review M11), once no daemon is using them.
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)
        cls.addClassCleanup(subprocess.run, ["bash", "pyhost/serve_nomos.sh", "down"], cwd=ROOT)
        shutil.rmtree(DIR, ignore_errors=True)
        env = {**os.environ, "NOMOS_KVSTORE": "1", "NOMOS_KVSTORE_DIR": str(DIR), "NOMOS_PREFILL_CHUNK": "256"}
        r = subprocess.run(["bash", "pyhost/serve_nomos.sh", "--model", "muse", "up"], cwd=ROOT, env=env)
        assert r.returncode == 0, "serve did not come up"

    def test_a_7k_conversation_is_served_and_reused(self):
        text = (ROOT / "docs/MEASUREMENT_TRAPS.md").read_text()[:28000]     # ~7K tokens: past the 6,200 cap
        msgs = [{"role": "user", "content": "Summarise this document in three sentences:\n\n" + text}]
        before = post("/v1/kvstore/stats")
        reply = post("/v1/chat/completions", {"model": "muse", "messages": msgs, "max_tokens": 64, "temperature": 0.7})
        msg = reply["choices"][0]["message"]
        self.assertTrue((msg.get("content") or msg.get("reasoning_content") or "").strip())
        mid = post("/v1/kvstore/stats")
        self.assertGreater(mid["prompt_tokens"] - before["prompt_tokens"], 6200)
        msgs += [{"role": "assistant", "content": msg.get("content") or "",
                  **({"reasoning_content": msg["reasoning_content"]} if msg.get("reasoning_content") else {})},
                 {"role": "user", "content": "Now in one sentence."}]
        post("/v1/chat/completions", {"model": "muse", "messages": msgs, "max_tokens": 32, "temperature": 0.7})
        after = post("/v1/kvstore/stats")
        prefilled = (after["prompt_tokens"] - mid["prompt_tokens"]) - (after["reused_tokens"] - mid["reused_tokens"])
        print(f"turn 2 prefilled {prefilled} tokens of a {after['prompt_tokens'] - mid['prompt_tokens']}-token prompt",
              flush=True)
        self.assertLess(prefilled, 300)


# Tuned once (Task 16): 109,000 characters of the plan render a 32,041-token prompt. 31,500-32,200 is wanted, so that
# three turns fit in 32,768.
N_CHARS = 109_000


def candidate_host_env(text):
    """Adam's host env as Step 6 proposes it: the three settings removed wherever they are, then appended in the
    override-friendly ${VAR:-value} form (ledger R45, R67). NOMOS_KVSTORE_DIR and the rest are left as they are."""
    keep = [ln for ln in text.splitlines()
            if not re.match(r"\s*(export\s+)?NOMOS_(MAX_SEQ|PREFILL_CHUNK|KVSTORE)=", ln)]
    return "\n".join(keep + ['export NOMOS_MAX_SEQ="${NOMOS_MAX_SEQ:-32768}"',
                              'export NOMOS_PREFILL_CHUNK="${NOMOS_PREFILL_CHUNK:-256}"',
                              'export NOMOS_KVSTORE="${NOMOS_KVSTORE:-1}"']) + "\n"


class GpuMemory:
    """Peak GPU memory in use while a block runs, polled from nvidia-smi. Reported, not asserted: other processes
    share this GPU."""
    def __enter__(self):
        self.peak, self.total, self._stop = 0, 0, threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            used, total = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.split(",")
            self.peak, self.total = max(self.peak, int(used)), int(total)
            self._stop.wait(0.5)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


@unittest.skipUnless(os.environ.get("KVSTORE_LONG32K") == "1", "opt-in: about 2 minutes on the whole GPU (ledger R67)")
class ThirtyTwoKThroughTheServe(unittest.TestCase):
    """The configuration Step 6 asks Adam for, run through the real launcher first: a 32K check, always with its own 32K
    candidate host env (candidate_host_env). A 128K check on the Pro 6000 needs that machine's own candidate and a prompt
    of about 127K tokens (RUNBOOK §3d); this test does not do it."""

    @classmethod
    def setUpClass(cls):
        # As in LongConversationServeTest: a failed bring-up must still stop a daemon that came up, and the store's files
        # are removed after `down`.
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)
        cls.addClassCleanup(subprocess.run, ["bash", "pyhost/serve_nomos.sh", "down"], cwd=ROOT)
        shutil.rmtree(DIR, ignore_errors=True)
        cand = ROOT / "build/task16/host_env.candidate"
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_text(candidate_host_env((Path.home() / ".nomos_host_env").read_text()))
        env = {**os.environ, "NOMOS_HOST_ENV": str(cand), "NOMOS_KVSTORE_DIR": str(DIR)}
        for k in ("NOMOS_MAX_SEQ", "NOMOS_PREFILL_CHUNK", "NOMOS_KVSTORE"):
            env.pop(k, None)                              # the candidate file decides, as it will for Adam
        r = subprocess.run(["bash", "pyhost/serve_nomos.sh", "--model", "muse", "up"], cwd=ROOT, env=env)
        assert r.returncode == 0, "serve did not come up"

    def test_a_32k_conversation_at_temperature_0(self):
        # /health's "spec" says whether the DFlash drafter is loaded. Without it this test passes on the base decoder
        # and never exercises the drafter's context rows past 8,192.
        self.assertTrue(post("/health")["spec"],
                        "DFlash is not loaded: the host env needs NOMOS_SERVE_SPEC=1 and a DFLASH_DIR that loads")
        text = (ROOT / "docs/KV_SESSION_STORE_PLAN.md").read_text()[:N_CHARS]
        msgs = [{"role": "system", "content": "You are Muse, a helpful assistant."},   # fixed: KERNEL_CONTRACT rule 3
                {"role": "user", "content": "Which task adds chunked prefill to the serve? One sentence.\n\n" + text}]
        before = post("/v1/kvstore/stats")
        with GpuMemory() as mem:
            t0 = time.perf_counter()
            reply = post("/v1/chat/completions",
                         {"model": "muse", "messages": msgs, "max_tokens": 64, "temperature": 0})
            turn1_s = time.perf_counter() - t0
        msg = reply["choices"][0]["message"]
        self.assertTrue((msg.get("content") or msg.get("reasoning_content") or "").strip())
        mid = post("/v1/kvstore/stats")
        n_prompt = mid["prompt_tokens"] - before["prompt_tokens"]
        self.assertGreaterEqual(n_prompt, 31500, "the prompt must stand for a full 32K context: if the plan's first "
                                                 f"{N_CHARS:,} characters have changed, re-tune N_CHARS")
        msgs += [{"role": "assistant", "content": msg.get("content") or "",
                  **({"reasoning_content": msg["reasoning_content"]} if msg.get("reasoning_content") else {})},
                 {"role": "user", "content": "Now in five words."}]
        reply2 = post("/v1/chat/completions", {"model": "muse", "messages": msgs, "max_tokens": 32, "temperature": 0})
        after = post("/v1/kvstore/stats")
        prefilled = (after["prompt_tokens"] - mid["prompt_tokens"]) - (after["reused_tokens"] - mid["reused_tokens"])
        self.assertLess(prefilled, 300)
        # The demo's switch: another conversation parks this one (its drafter rows run past 8,192), then come back.
        m2 = reply2["choices"][0]["message"]
        msgs += [{"role": "assistant", "content": m2.get("content") or "",
                  **({"reasoning_content": m2["reasoning_content"]} if m2.get("reasoning_content") else {})},
                 {"role": "user", "content": "And in three?"}]
        post("/v1/chat/completions", {"model": "muse", "max_tokens": 16, "temperature": 0,
                                      "messages": [msgs[0], {"role": "user", "content": "Say hello."}]})
        parked = post("/v1/kvstore/stats")
        t0 = time.perf_counter()
        post("/v1/chat/completions", {"model": "muse", "messages": msgs, "max_tokens": 16, "temperature": 0})
        turn3_s = time.perf_counter() - t0
        back = post("/v1/kvstore/stats")
        restores = (back["restores_ram"] + back["restores_nvme"]) - (parked["restores_ram"] + parked["restores_nvme"])
        prefilled3 = (back["prompt_tokens"] - parked["prompt_tokens"]) - (back["reused_tokens"] - parked["reused_tokens"])
        print(f"32K through the serve: {n_prompt}-token prompt, turn 1 {turn1_s:.1f} s, GPU peak {mem.peak} of "
              f"{mem.total} MiB ({mem.total - mem.peak} MiB free at the peak); turn 2 prefilled {prefilled} tokens; "
              f"after a switch, turn 3 restored ({'RAM' if back['restores_ram'] > parked['restores_ram'] else 'NVMe'}) "
              f"and prefilled {prefilled3} tokens in {turn3_s:.1f} s", flush=True)
        self.assertEqual(restores, 1)
        self.assertLess(prefilled3, 300)


if __name__ == "__main__":
    unittest.main()
