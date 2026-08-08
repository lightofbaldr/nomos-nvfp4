#!/usr/bin/env python3
"""DFlash synthetic goldens: z-lab reference drafter (fp32, CPU, deterministic) on fixed-seed
synthetic inputs — pins ALL drafter-internal math for the Mojo port without needing the target.

Stages captured (per codex's ambiguity list, all source-pinned from z-lab dflash/model.py):
  ctx_fused   = hidden_norm(fc(target_hidden_concat))        [1, CTX, 5376]   (model.py:334)
  layer{i}_out = decoder layer i output                       [1, BLOCK, 5376]
  final_norm  = norm(last hidden)                             [1, BLOCK, 5376] (model.py:347)
Inputs (seeded): target_hidden [1, CTX, 6*5376], noise_embedding [1, BLOCK, 5376],
position_ids = arange(CTX+... actually [0, CTX+BLOCK) sliced per reference call semantics:
the generate loop passes position_ids for [draft_cache_len, start+block) — for a fresh cache
that is [0, CTX+BLOCK) where context occupies [0,CTX) and the block [CTX, CTX+BLOCK).
Attention: is_causal=False, attention_mask=None (model.py:118) — bidirectional; sliding(2048)
on layers 0-3 (irrelevant at CTX+BLOCK << 2048 but noted), full on layer 4.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

REF_REPO_ENV = os.environ.get("DFLASH_REF_REPO", "")
if not REF_REPO_ENV:
    raise SystemExit("set DFLASH_REF_REPO to a checkout containing dflash/model.py")
REF_REPO = Path(REF_REPO_ENV)
sys.path.insert(0, str(REF_REPO))
from dflash.model import DFlashDraftModel

CKPT = os.environ.get("DFLASH_CKPT", "")
if not CKPT:
    raise SystemExit("set DFLASH_CKPT to the DFlash checkpoint directory")
OUT = os.environ.get(
    "DFLASH_GOLDEN_OUT",
    str(Path(__file__).resolve().with_name("dflash_synth_goldens.npz")),
)
CTX, BLOCK, H, TAPS = 8, 16, 5376, 6

torch.manual_seed(20260704)
model = DFlashDraftModel.from_pretrained(CKPT, torch_dtype=torch.float32, device_map="cpu")
model.eval()
print(f"loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params fp32 cpu", flush=True)

target_hidden = torch.randn(1, CTX, TAPS * H, dtype=torch.float32) * 0.5
noise_embedding = torch.randn(1, BLOCK, H, dtype=torch.float32) * 0.5
position_ids = torch.arange(CTX + BLOCK).unsqueeze(0)

caps = {}
def cap(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        caps[name] = t.detach().float().numpy().copy()
    return hook

hooks = [model.layers[i].register_forward_hook(cap(f"layer{i}_out")) for i in range(5)]
# fc/hidden_norm captured directly below (model applies them at forward top, model.py:334)

with torch.inference_mode():
    fc_out = model.fc(target_hidden)
    ctx_fused = model.hidden_norm(fc_out)
    final = model(
        position_ids=position_ids,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )
for h in hooks: h.remove()

save = {
    "target_hidden": target_hidden.numpy(), "noise_embedding": noise_embedding.numpy(),
    "position_ids": position_ids.numpy(),
    "fc_out": fc_out.numpy(), "ctx_fused": ctx_fused.numpy(),
    "final_norm": final.detach().float().numpy(),
}
save.update(caps)
np.savez_compressed(OUT, **save)
print("captured:", {k: list(v.shape) for k, v in save.items()}, flush=True)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)
