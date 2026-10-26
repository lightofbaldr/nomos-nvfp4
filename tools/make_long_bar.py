#!/usr/bin/env python3
"""Build tools/gold_bar_tokens_long.json — the LONG-CONTEXT bar tier.

WHY (2026-08-03, Adam-directed). The standard 12-prompt bar (gold_bar_tokens.json) has prompts of
54-216 tokens and generates 64: the deepest position any gate ever reaches is ~280. Gemma-4's
sliding window is 1024, so on the standard bar THE WINDOW NEVER FILLS — every window-boundary
behavior (KV ring correctness, PV quant-block anchoring, acceptance at depth) is structurally
unreachable by it, and the ModCon demo will run far deeper contexts than anything we measure.
X's inexactness passing 12/12 wasn't even luck: below 1024 its defect cannot exist.

Content: built from THIS repo's own Mojo sources and docs (self-contained, no external text), with
Mojo-writing questions per Adam — that's the actual ModCon audience workload. One synthetic agentic
ops transcript. Prompts carry a `text` field alongside `ids` so nobody has to detokenize to answer
"what do we test with" again.

Targets (prompt tokens): mojo1k ~1.3K (just past the window), prose4k ~4K, agent8k ~8K,
mojoreview20k ~20K. Run with NOMOS_KV_SWA=1 and NOMOS_MAX_SEQ>=24576 — without the ring, 20K+ of
INT4 KV (~5-8GB) does not fit beside the NVFP4 weights on a 24GB card.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOK_DIR = os.environ.get("TOK_DIR", os.path.expanduser("~/models/gemma-4-31b-it"))
OUT = os.path.join(HERE, "gold_bar_tokens_long.json")

tok = AutoTokenizer.from_pretrained(TOK_DIR)


def read(path, max_chars):
    with open(os.path.join(REPO, path), errors="replace") as f:
        return f.read()[:max_chars]


def build(user_text):
    # apply_chat_template(tokenize=True) returns a BatchEncoding in this transformers version —
    # list() of it iterates dict KEYS (2 entries), which produced 2-"token" prompts on the first
    # attempt. Take .input_ids explicitly.
    enc = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                  add_generation_prompt=True, tokenize=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    assert len(ids) > 200, f"suspiciously short prompt ({len(ids)} toks) — template ate the content"
    return [int(x) for x in ids]


prompts = []

# ── mojo1k [code] ~1.3K: one kernel excerpt + a how-do-I-write-Mojo question. Sits just past the
#    1024 window so even the SHORTEST tier prompt exercises a full window.
mojo1k = (
    "Here is an excerpt from a Mojo GPU kernel in our inference engine:\n\n```mojo\n"
    + read("lib/kv_cache_quant.mojo", 3600)
    + "\n```\n\nI'm new to Mojo. Explain how to write a GPU kernel like this in Mojo: how "
    "`ctx.enqueue_function`, `grid_dim`/`block_dim`, `UnsafePointer`, and warp-level reductions "
    "with `shuffle_xor` fit together. Then write a small new Mojo kernel that computes a per-row "
    "L2 norm of a [n_rows, dim] fp32 buffer using one warp per row, following the same idioms."
)
prompts.append(dict(id="mojo1k", bucket="code", text=mojo1k))

# ── prose4k [memory_prose] ~4K: our own docs + a synthesis question.
prose4k = (
    "Below are two internal engineering documents from an LLM inference kernel project.\n\n"
    "=== DOCUMENT 1: RUNBOOK ===\n" + read("docs/RUNBOOK.md", 9000)
    + "\n\n=== DOCUMENT 2: MEASUREMENT TRAPS ===\n" + read("docs/MEASUREMENT_TRAPS.md", 6000)
    + "\n\nSummarize, for a new engineer joining this project: (1) the three most important "
    "configuration rules and why each exists, (2) the measurement failure modes they must know "
    "before quoting any benchmark number, and (3) what 'lossless' does and does not guarantee here."
)
prompts.append(dict(id="prose4k", bucket="memory_prose", text=prose4k))

# ── agent8k [agentic] ~8K: synthetic multi-turn ops transcript with tool outputs, ending in a
#    next-action question. Built from realistic material (build logs, gate output shapes).
_tool_log = read("docs/MEASUREMENT_TRAPS.md", 4000)
_src = read("lib/engine_prefill.mojo", 12000)
agent8k = (
    "You are an SRE agent with tools: ssh_run(host, cmd), read_file(host, path), "
    "restart_service(host, name), git(cmd). Below is the session so far.\n\n"
    "[turn 1] user: The inference box 'gold' is serving at half its usual tok/s since last "
    "night's deploy. Find out why.\n"
    "[turn 2] you called ssh_run('gold', 'git -C ~/nomos log --oneline -5') ->\n"
    "  d043291 docs: exact-Y ties the champion\n  db92375 Merge: batched multirow append\n"
    "  30b887f spec loop: DEAD-DRAFTER guard\n  879b81f SPEC_VB=7 everywhere\n  2b47694 tools: stop tail -16\n"
    "[turn 3] you called ssh_run('gold', 'nvidia-smi --query-gpu=clocks.sm,temperature.gpu "
    "--format=csv,noheader') -> 2617 MHz, 83 C\n"
    "[turn 4] you called read_file('gold', 'docs/MEASUREMENT_TRAPS.md') ->\n" + _tool_log + "\n"
    "[turn 5] you called ssh_run('gold', 'systemctl is-active gemma4-chat nomos-serve') -> "
    "active\\nactive\n"
    "[turn 6] you called read_file('gold', 'lib/engine_prefill.mojo') (excerpt) ->\n" + _src + "\n\n"
    "[turn 7] user: OK — so given all of that, what is your diagnosis of the slowdown, what one "
    "command do you run next to confirm it, and what would you roll back if confirmed? Answer as "
    "a numbered plan."
)
prompts.append(dict(id="agent8k", bucket="agentic", text=agent8k))

# ── mojoreview20k [code] ~20K: several full Mojo files + write-new-Mojo questions. The deep-context
#    Mojo-authoring prompt — the ModCon audience workload at demo-realistic depth.
mojoreview20k = (
    "Here are several Mojo source files from a production GPU inference kernel.\n\n"
    "=== FILE 1: lib/fp4_gemv_v2.mojo ===\n```mojo\n" + read("lib/fp4_gemv_v2.mojo", 26000) + "\n```\n\n"
    "=== FILE 2: lib/attention_gpu_int8.mojo ===\n```mojo\n" + read("lib/attention_gpu_int8.mojo", 26000) + "\n```\n\n"
    "=== FILE 3: lib/kv_cache_quant.mojo (excerpt) ===\n```mojo\n" + read("lib/kv_cache_quant.mojo", 14000) + "\n```\n\n"
    "You are teaching a workshop on writing Mojo GPU kernels. Using these files as your corpus: "
    "(1) explain the core patterns a Mojo kernel author must know — `@parameter for` unrolling, "
    "compile-time `comptime` constants, `InlineArray` accumulators, warp shuffles, and how host "
    "launch wrappers relate to kernels; (2) identify two idioms here you would teach as best "
    "practice and one you would warn about; (3) then WRITE, in Mojo, a new warp-per-row kernel "
    "that dequantizes an INT8 row (per-row fp32 scale) and accumulates a dot product against an "
    "fp32 vector, with the host-side launch wrapper, in the exact style of these files."
)
prompts.append(dict(id="mojoreview20k", bucket="code", text=mojoreview20k))

out = {"weights": {"agentic": 0.35, "code": 0.40, "memory_prose": 0.25}, "prompts": []}
for p in prompts:
    ids = build(p["text"])
    out["prompts"].append(dict(id=p["id"], bucket=p["bucket"], ids=ids, text=p["text"]))
    print(f"  {p['id']:>14} [{p['bucket']:>12}] {len(ids):>6} prompt tokens")

tot = sum(out["weights"].values())
assert abs(tot - 1.0) < 1e-9, f"weights sum {tot}"
json.dump(out, open(OUT, "w"))
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
