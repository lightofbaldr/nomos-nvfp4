"""Spike agentic loop over the Python grammar host — drive the is_prime task to a
GREEN pytest, proving: kernel logits engine + Python sampling + grammar-forced tool
calls → reliable multi-turn tool use on the 4-bit model. Self-contained.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import KernelLM, H            # noqa: E402
from grammar import ToolGrammar          # noqa: E402
from transformers import AutoTokenizer   # noqa: E402

SANDBOX = os.path.expanduser("~/nomos_pyhost_sandbox")
os.makedirs(SANDBOX, exist_ok=True)

TOOLS = ["write_file", "read_file", "list_files", "run_pytest"]
TOOL_DEFS = [
    {"type": "function", "function": {"name": "write_file", "description": "Write text to a file in the sandbox",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_files", "description": "List files",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "run_pytest", "description": "Run pytest in the sandbox",
     "parameters": {"type": "object", "properties": {}}}},
]


def _safe(path: str) -> str:
    p = os.path.normpath(os.path.join(SANDBOX, path))
    if not p.startswith(SANDBOX):
        raise ValueError("path escapes sandbox")
    return p


def write_file(path: str, content: str) -> str:
    with open(_safe(path), "w") as f:
        f.write(content)
    return f"WROTE {path} ({len(content)} bytes)"


def read_file(path: str) -> str:
    try:
        return open(_safe(path)).read()[:4000]
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def list_files() -> str:
    fs = [f for f in os.listdir(SANDBOX) if not f.startswith(".")]
    return "\n".join(fs) if fs else "(empty)"


def run_pytest() -> str:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=SANDBOX,
                       capture_output=True, text=True, timeout=60)
    return f"exit_code={p.returncode}\n{(p.stdout + p.stderr)[-2500:]}"


# Gemma arg format: {key:<|"|>value<|"|>,key2:<|"|>v2<|"|>} or {} . Values are <|"|>-delimited.
_CALL_RE = re.compile(r"<\|tool_call>call:([a-zA-Z_][a-zA-Z0-9_]*)\{(.*?)\}<tool_call\|>", re.DOTALL)
_KV_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*):<\|"\|>(.*?)<\|"\|>', re.DOTALL)
# Canonical Gemma arg: STRING values are <|"|>-delimited, NUMBER/BOOL values are bare (no quotes),
# e.g. {name:<|"|>Acme<|"|>,count:3,total:18.5}. Match BOTH so numeric fields aren't dropped.
# group(2)=quoted string content; group(3)=bare value (number/bool/json-quoted-string).
_ARG_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', re.DOTALL)


def _parse_args(body: str) -> dict:
    """Extract tool-call args from the grammar body, robustly.

    The grammar's canonical form is `key:<|"|>value<|"|>` (special-token delimited),
    produced reliably on the NON-streaming path. The streaming engine path can emit
    free-form JSON-ish args instead (`"k": "v"`, `k: "v"`), which the canonical regex
    misses -> dropped args / `{}` over the wire (breaks OpenAI tool-calling clients like
    pydantic-ai and Open WebUI, which stream). So: try the canonical form first (zero
    behaviour change for the non-streaming path), then JSON, then a lenient quoted-value
    scan, so a streaming tool call never loses its arguments."""
    body = (body or "").strip()
    if not body:
        return {}
    # 1. canonical Gemma form — handles BOTH <|"|>-quoted strings AND bare numbers/bools (and
    # stray json-quoted values via the strip below), so numeric tool args are not dropped.
    args: dict = {}
    for m in _ARG_RE.finditer(body):
        k = m.group(1)
        if m.group(2) is not None:
            v = m.group(2)                                   # <|"|>-quoted string
        else:
            v = (m.group(3) or "").strip().strip('"').strip()  # bare number/bool / json-quoted
        if k not in args or v != "":
            args[k] = v
    if args:
        return args
    # 2. JSON object (wrap a bare `k: v, ...` body in braces)
    for cand in (body, "{" + body + "}"):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in obj.items()}
        except Exception:
            pass
    # 3. lenient: optionally-quoted key : "quoted value" (handles unquoted keys / stray chars)
    out: dict = {}
    for k, v in re.findall(r'"?([A-Za-z_][A-Za-z0-9_]*)"?\s*:\s*"([^"]*)"', body):
        out.setdefault(k, v)
    return out


def parse_call(text: str):
    m = None
    for m in _CALL_RE.finditer(text):
        pass  # take the LAST well-formed call
    if not m:
        return None
    name, body = m.group(1), m.group(2)
    return name, _parse_args(body)


def dispatch(name: str, args: dict) -> str:
    try:
        if name == "write_file":
            return write_file(args.get("path", ""), args.get("content", ""))
        if name == "read_file":
            return read_file(args.get("path", ""))
        if name == "list_files":
            return list_files()
        if name == "run_pytest":
            return run_pytest()
        return f"ERROR: unknown tool {name}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def main() -> None:
    for f in ("primes.py", "test_primes.py"):
        try:
            os.remove(os.path.join(SANDBOX, f))
        except OSError:
            pass
    tok = AutoTokenizer.from_pretrained(H.TOK_DIR)
    lm = KernelLM()
    gram = ToolGrammar(tok, TOOLS)
    TOOL_CLOSE, EOT = 49, tok.convert_tokens_to_ids("<end_of_turn>")
    stop = [tok.eos_token_id, EOT, TOOL_CLOSE]

    system = ("You are a Python engineer with tools. Work in small steps: write primes.py with "
              "is_prime(n); write test_primes.py with pytest asserting is_prime(2),is_prime(3),is_prime(17) "
              "True and is_prime(1),is_prime(4) False; call run_pytest; if it fails, fix the file and "
              "run_pytest again; stop only when exit_code=0.")
    messages = [{"role": "user", "content": "[Instructions: " + system + "]\n\nBegin."}]

    for turn in range(12):
        s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True, tools=TOOL_DEFS)
        ids = list(np.asarray(tok(s, add_special_tokens=False).input_ids, dtype=np.int64))
        gram.reset()
        out = lm.generate(ids, max_new_tokens=2048, stop_ids=stop, processor=gram)
        text = tok.decode(out, skip_special_tokens=False)
        call = parse_call(text)
        if call is None:
            print(f"[turn {turn}] no tool call — final answer. STOP.", flush=True)
            print("  ", repr(tok.decode(out, skip_special_tokens=True)[:200]), flush=True)
            break
        name, args = call
        result = dispatch(name, args)
        short = result.split("\n")[0]
        print(f"[turn {turn}] CALL {name}({list(args)}) -> {short}", flush=True)
        messages.append({"role": "user", "content": text.split("<channel|>")[-1]})  # assistant action (post-thought)
        messages.append({"role": "user", "content": f"[Result of {name}]\n{result}"})
        if name == "run_pytest" and "exit_code=0" in result:
            print("\n*** GREEN — pytest exit_code=0 ***", flush=True)
            break

    print("\n=== sandbox ===", flush=True)
    for f in ("primes.py", "test_primes.py"):
        pth = os.path.join(SANDBOX, f)
        if os.path.exists(pth):
            print(f"--- {f} ---\n{open(pth).read()}", flush=True)


if __name__ == "__main__":
    main()
