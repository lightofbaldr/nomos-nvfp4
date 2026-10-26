#!/usr/bin/env python3
"""Serve path (spec §4): through the real OpenAI-compatible serve, turn k of a conversation prefills only its new
tokens, and a parked conversation is switched back in. Every reply is streamed, so time to first token is measured.
GPU; starts and stops the serve itself.
  KVSTORE_SERVE_TEST_TEMPERATURE  0.7 (default) takes the base path; 0 takes DFlash when the serve has it loaded.
  KVSTORE_SERVE_TEST_BASELINE=1   runs the serve with NOMOS_KVSTORE=0 and prints only the times: the "before" column.
                                  It asserts the store really is off (no prepare, no session file): serve_nomos.sh
                                  sources ~/.nomos_host_env before its defaults, so a host env that turns the store on
                                  would otherwise measure the "before" column with the store on."""
import json
import os
import pathlib
import shutil
import subprocess
import time
import unittest
import urllib.request

from kvsession_testlib import ROOT

PORT = int(os.environ.get("NOMOS_PORT", "8091"))
DIR = ROOT / "build/kvstore_serve_test"
TEMPERATURE = float(os.environ.get("KVSTORE_SERVE_TEST_TEMPERATURE", "0.7"))
BASELINE = os.environ.get("KVSTORE_SERVE_TEST_BASELINE") == "1"
QUESTIONS = ["Name three uses of a lighthouse.", "Which of those still matter today?",
             "Explain the second one in more detail.", "Summarise our conversation so far.",
             "Give it a title.", "Now translate the title into French."]
# Tokens a later turn may prefill: its new question and the template around it. 18-21 observed. A turn that prefilled its
# previous reply again would cost about 95, and a whole conversation more.
MAX_PREFILL = 64


def stats():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/kvstore/stats", timeout=120) as r:
        return json.loads(r.read())


def session_files():
    """The conversations the store has on disk: NOMOS_KVSTORE_DIR/muse-<id>/<session id>.nkvs."""
    return sorted(DIR.glob("muse-*/*.nkvs"))


def chat(messages, drop_after=None, max_tokens=64):
    """One streamed chat turn. Returns (seconds to the first streamed text or None, the assistant message to send back).
    The reasoning goes back too, as a well-behaved client sends it: Muse's template re-renders it, so the next
    prompt extends the cached KV instead of diverging inside the previous reply (plan note 12).
    drop_after=k: the client goes away after k chunks of text, with the connection closed mid-stream; the message is then
    the partial reply it had read."""
    body = {"model": "muse", "messages": messages, "max_tokens": max_tokens, "temperature": TEMPERATURE, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, ttft, parts, chunks = time.perf_counter(), None, {"content": "", "reasoning_content": ""}, 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"]
            for field in parts:
                if delta.get(field):
                    ttft = time.perf_counter() - t0 if ttft is None else ttft
                    parts[field] += delta[field]
                    chunks += 1
            if drop_after is not None and chunks >= drop_after:
                break                                       # leaving the `with` closes the connection: the client is gone
    msg = {"role": "assistant", "content": parts["content"]}
    if parts["reasoning_content"]:
        msg["reasoning_content"] = parts["reasoning_content"]
    return ttft, msg


class ServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Registered before anything can fail: if the serve fails to come up below, a daemon that did come up still holds the
        # GPU, and tearDownClass would not run after a failed setUpClass. Cleanups run last-registered first: `down`, and then
        # the store's files are removed (final review M11), once no daemon is using them.
        cls.addClassCleanup(shutil.rmtree, DIR, ignore_errors=True)
        cls.addClassCleanup(subprocess.run, ["bash", "pyhost/serve_nomos.sh", "down"], cwd=ROOT)
        shutil.rmtree(DIR, ignore_errors=True)
        env = {**os.environ, "NOMOS_KVSTORE": "0" if BASELINE else "1", "NOMOS_KVSTORE_DIR": str(DIR)}
        r = subprocess.run(["bash", "pyhost/serve_nomos.sh", "--model", "muse", "up"], cwd=ROOT, env=env)
        assert r.returncode == 0, "serve did not come up"
        # Before any test sends a chat request: what /health says about the store, and what the store has counted.
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=120) as r:
            cls.health_at_start = json.loads(r.read())
        cls.stats_at_start = stats()

    def test_health_reports_the_store_open_before_the_first_chat(self):     # ruling R79, final review I3
        kv = self.health_at_start["kvstore"]
        self.assertEqual(self.stats_at_start["prepares"], 0, "no chat request had reached the store yet")
        if BASELINE:
            self.assertEqual(kv, {"status": "off", "dir": None})
            return
        self.assertEqual(kv["status"], "open")
        self.assertEqual(pathlib.Path(kv["dir"]).parent, DIR.resolve())
        self.assertTrue((pathlib.Path(kv["dir"]) / ".lock").exists(), "the store took its directory's lock at launch")

    def _turn(self, msgs, label):
        """One turn; appends the reply to msgs and returns the store counters' change across it."""
        prev = stats()
        ttft, reply = chat(msgs)
        msgs.append(reply)
        cur = stats()
        d = {k: cur[k] - prev[k] for k in cur}
        t = f"{ttft:.3f}s" if ttft is not None else "no text"        # an empty reply has no first token
        if BASELINE:
            print(f"{label}: ttft {t}", flush=True)
            self.assertEqual(d["prepares"], 0, "the baseline must run with the store off")
        else:
            print(f"{label}: ttft {t}  prompt {d['prompt_tokens']}  reused {d['reused_tokens']}  "
                  f"prefilled {d['prompt_tokens'] - d['reused_tokens']}  forks {d['forks']}  "
                  f"restores ram/nvme {d['restores_ram']}/{d['restores_nvme']}", flush=True)
            self.assertEqual(d["prepares"], 1, "the request must go through the store")
        return d

    def _assert_extends(self, d, what):
        """A later turn of a conversation extends the cached history: no fork, and only its new tokens are prefilled."""
        if BASELINE:
            return
        self.assertEqual(d["forks"], 0, f"{what}: the prompt left the cached history (the previous reply, sent back, "
                                        "did not re-render to the tokens the KV holds)")
        self.assertLess(d["prompt_tokens"] - d["reused_tokens"], MAX_PREFILL,
                        f"{what}: a later turn must not re-prefill the conversation")

    def _assert_on_disk(self, files_before):
        """Store mode: the conversation reached the NVMe tier. Baseline: the store is off, so nothing was written."""
        if BASELINE:
            self.assertEqual(session_files(), [], "the baseline wrote session files: the store was on")
        else:
            self.assertGreater(len(session_files()), len(files_before), "no new session file in NOMOS_KVSTORE_DIR")

    def test_six_turns_prefill_only_new_tokens(self):      # success criterion 1
        files_before, msgs = session_files(), []
        for i, q in enumerate(QUESTIONS):
            msgs.append({"role": "user", "content": q})
            d = self._turn(msgs, f"turn {i + 1}")
            if i > 0:
                self._assert_extends(d, f"turn {i + 1}")
        self._assert_on_disk(files_before)

    def test_switch_between_two_conversations(self):
        # A is long enough (~600 tokens) that switching back beats a fresh prefill (NOMOS_KVSTORE_MIN_SWITCH=256).
        text = (ROOT / "docs/MEASUREMENT_TRAPS.md").read_text()[:2400]
        a = [{"role": "user", "content": "Summarise this in two sentences:\n\n" + text}]
        b = [{"role": "user", "content": "What is the capital of France?"}]
        self._turn(a, "A1")
        self._turn(b, "B1")                                 # parks A
        a.append({"role": "user", "content": "Now in one sentence."})
        d = self._turn(a, "A2")                             # A is switched back in
        if not BASELINE:
            self.assertEqual(d["restores_ram"] + d["restores_nvme"], 1)
            self._assert_extends(d, "A2")

    def test_dropped_stream_is_committed_and_the_next_turn_reuses_it(self):    # Review Focus 1, through the real serve
        # A first prompt of its own, so that nothing else in this class starts like it.
        msgs = [{"role": "user", "content": "What makes the tides rise and fall?"}]
        before = stats()
        t0 = time.perf_counter()
        # Two chunks of text out of a reply that would run for about 30 s, then the client goes away.
        _, partial = chat(msgs, drop_after=2, max_tokens=1000)
        after = stats()          # waits for the engine lock: the serve is done with the dropped turn, its commit included
        waited = time.perf_counter() - t0
        d1 = {k: after[k] - before[k] for k in after}
        print(f"dropped after 2 chunks: the serve was done {waited:.2f}s later; prompt {d1['prompt_tokens']}  "
              f"disk +{after['nvme_bytes'] - before['nvme_bytes']} bytes", flush=True)
        self.assertLess(waited, 10, "the serve must stop generating when the client goes away, not run on to max_tokens")
        if not BASELINE:
            self.assertEqual(d1["prepares"], 1)
            self.assertGreater(after["nvme_bytes"], before["nvme_bytes"], "the dropped turn must have been committed to disk")
        msgs += [partial, {"role": "user", "content": "And why are there two high tides a day?"}]
        d2 = self._turn(msgs, "next turn, with the partial reply")
        if not BASELINE:
            self.assertGreaterEqual(d2["reused_tokens"], d1["prompt_tokens"],
                                    "the next turn must reuse at least the dropped turn's whole prompt")
            self.assertLess(d2["prompt_tokens"] - d2["reused_tokens"], MAX_PREFILL,
                            "the next turn must prefill only what follows the dropped turn's prompt")
            self.assertEqual(d2["forks"], 0,
                             "the partial reply the client sent back diverged from the KV the dropped turn left: the engine "
                             "handed out (and kept rows for) tokens the client never took (ledger R55, R56). On the "
                             "speculative path this check has a one-token margin: it holds if the serve had pulled at most "
                             "one token past the last one the client read when the client's close reached it, and inside a "
                             "burst that is under a millisecond (the base path has a decode step, ~20 ms). A loaded machine "
                             "can lose that race with no code change; R56 itself is pinned by pyhost/test_spec_pull.py, so "
                             "relax this check rather than chase such a flake")

    def test_a_turn_that_ends_on_its_stop_token_keeps_the_daemon_connection(self):     # final review M3
        """The serve drains the stream after the stop token, so a turn that ends normally keeps its daemon connection.
        Before, every streamed speculative turn that ended on its stop token closed the socket mid-stream: daemon.log showed
        a disconnect, and the next request a reconnect. At temperature > 0 the turn takes the base path, which never
        streams from the daemon, so this holds there trivially."""
        log = ROOT / "daemon.log"
        count = lambda: {k: log.read_text().count(f"[daemon] client {k}") for k in ("connected", "disconnected")}  # noqa: E731
        stats()                                             # the serve's daemon connection is up and idle
        before = count()
        body = {"model": "muse", "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "max_tokens": 1500, "temperature": TEMPERATURE, "stream": True}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        finish = None
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    finish = json.loads(line[5:])["choices"][0].get("finish_reason") or finish
        self.assertEqual(finish, "stop", "the turn must end on its stop token for this test to mean anything")
        stats()                                             # the next request, on the same connection if it was kept
        time.sleep(0.5)                                     # daemon.log is written through tee
        self.assertEqual(count(), before, "the turn dropped its daemon connection and the next request reconnected")


if __name__ == "__main__":
    unittest.main()
