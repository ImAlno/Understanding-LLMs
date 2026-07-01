"""
E04 — Quantize a model and measure the drift.
=============================================
The real test: quantize every Linear weight of a GPT and see how far the output logits move from
fp32. int8 barely budges; int4 drifts more but quarters the size.

Run:  python e04_quantize_model.py
"""
import sys
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn as nn
from gpt import GPT, random_batch


@torch.no_grad()
def quantize_linears(model, bits):
    qmax = 2 ** (bits - 1) - 1
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            scale = w.abs().amax(1, keepdim=True) / qmax
            mod.weight.data = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


torch.manual_seed(0)
model = GPT().eval()
idx = random_batch()
with torch.no_grad():
    ref = model(idx)
n = sum(p.numel() for p in model.parameters())

print(f"{'format':>7} {'size':>9} {'logit drift':>13}")
print(f"{'fp16':>7} {n * 2 / 1e6:>6.2f} MB {'ref':>13}")
for bits in (8, 4):
    mq = copy.deepcopy(model)
    quantize_linears(mq, bits)
    with torch.no_grad():
        drift = ((mq(idx) - ref).pow(2).mean().sqrt() / ref.std()).item()
    print(f"{'int' + str(bits):>7} {n * bits / 8 / 1e6:>6.2f} MB {drift:>13.4f}")

print("\n✓ int8: half the size, output barely moves. int4: quarter the size, a few % drift.")
