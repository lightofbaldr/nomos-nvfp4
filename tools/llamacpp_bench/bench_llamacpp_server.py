#!/usr/bin/env python3
"""Fair same-card llama.cpp decode bench on Gold GPU0 over the 12-prompt gold bar.

Drives our OWN llama-server (this build's llama-cli is stuck in an interactive
loop) pinned to physical GPU0 by UUID so it can NEVER touch GPU1 (the other
llama-server) or GPU2 (ollama). Feeds the EXACT gold_bar_tokens.json token ids
(bypasses tokenization -> identical prompt to our kernel), greedy (temp 0),
n_predict = PERF_NTOK (96). Reports llama.cpp timings.predicted_per_second
(decode tok/s) per prompt + mean, to compare vs our kernel's 21.7 base / ~50 spec.

Base:  python bench_llamacpp_server.py --model <q4_0.gguf>
Spec:  python bench_llamacpp_server.py --model <q4_0.gguf> --draft <assistant_q8_0.gguf> --nmax 4
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error, signal

GPU0_UUID = os.environ.get("LLAMA_GPU_UUID", "")
BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gold_bar_tokens.json")

def post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def wait_health(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            time.sleep(2)
    raise TimeoutError("health timeout")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ntok", type=int, default=int(os.environ.get("PERF_NTOK", "96")))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "llamacpp_server_results.jsonl"))
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()

    prompts = json.load(open(GOLD))["prompts"]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU0_UUID)
    cmd = [BIN, "-m", a.model, "-ngl", "99", "-fa", "on", "-c", "4096", "-np", "1",
           "--host", "127.0.0.1", "--port", str(a.port), "--no-webui"]
    if a.draft:
        cmd += ["-md", a.draft, "-ngld", "99", "--spec-draft-n-max", str(a.nmax),
                "--spec-draft-p-min", "0"]
    log = open(os.path.join(HERE, f"server_{a.tag}.log"), "w")
    print(f"[launch] {a.tag}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    results = []
    try:
        t_ready = wait_health(a.port, proc)
        print(f"[ready] {t_ready:.0f}s", flush=True)
        url = f"http://127.0.0.1:{a.port}/completion"
        # warmup (not recorded)
        post(url, {"prompt": prompts[0]["ids"], "n_predict": a.ntok, "temperature": 0.0,
                   "top_k": 1, "seed": 0, "cache_prompt": False})
        for run in range(1, a.runs + 1):
            for p in prompts:
                r = post(url, {"prompt": p["ids"], "n_predict": a.ntok, "temperature": 0.0,
                               "top_k": 1, "seed": 0, "cache_prompt": False, "n_probs": 0})
                tm = r.get("timings", {})
                row = {"tag": a.tag, "run": run, "id": p["id"], "bucket": p["bucket"],
                       "decode_tok_s": tm.get("predicted_per_second"),
                       "prefill_tok_s": tm.get("prompt_per_second"),
                       "n_pred": tm.get("predicted_n"),
                       "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted")}
                results.append(row)
                da = ""
                if row["draft_n"]:
                    da = f" draft_acc={row['draft_acc']}/{row['draft_n']}"
                print(f"  run{run} {p['id']:>8} decode={row['decode_tok_s']:.2f} t/s "
                      f"prefill={row['prefill_tok_s']:.0f}{da}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        log.close()

    with open(a.out, "a") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    dec = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    if dec:
        dec_sorted = sorted(dec)
        mean = sum(dec) / len(dec)
        med = dec_sorted[len(dec_sorted)//2]
        print(f"\n[{a.tag}] mean decode = {mean:.2f} t/s  median = {med:.2f}  "
              f"min={min(dec):.2f} max={max(dec):.2f}  over {len(dec)} (runs*prompts)", flush=True)
        # per-run means
        for run in range(1, a.runs + 1):
            rd = [r["decode_tok_s"] for r in results if r["run"] == run and r["decode_tok_s"]]
            if rd: print(f"    run{run} mean = {sum(rd)/len(rd):.2f} t/s", flush=True)

if __name__ == "__main__":
    main()
