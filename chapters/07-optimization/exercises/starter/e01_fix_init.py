"""
E01 — Fix a broken initialization (STARTER scaffold).
=====================================================
Set FIXED_SCALE so the model starts near -log(1/2) ≈ 0.69 instead of confidently wrong.
Reference: ../solutions/e01_fix_init.py.

Run:  python e01_fix_init.py
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch, torch.nn as nn, torch.nn.functional as F
from optimizers import make_moons, Adam

X, y = make_moons()

def model(output_scale):
    torch.manual_seed(1337)
    m = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
    with torch.no_grad():
        m[-1].weight *= output_scale
    return m

def train(m, steps=150):
    params = list(m.parameters()); opt = Adam(lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(m(X), y); m.zero_grad(); loss.backward()
        opt.step(params, [p.grad for p in params])
    return loss.item()

# The broken model multiplies its output layer by 20 → a confident-wrong start.
# ✍️ TODO: choose a scale that makes the model start near 0.69 (hint: don't blow up the output).
FIXED_SCALE = None      # replace
if FIXED_SCALE is None:
    raise SystemExit("Set FIXED_SCALE (a sane output-layer scale), then run again.")

print(f"a good init starts near -log(1/2) = {math.log(2):.3f}\n")
for scale, label in [(30.0, "broken (output x30)"), (FIXED_SCALE, "your fix")]:
    m = model(scale)
    init = F.cross_entropy(m(X), y).item()
    final = train(m)
    print(f"  {label:22} init {init:6.3f}  ->  after 150 steps {final:.4f}")
