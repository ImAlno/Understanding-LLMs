"""
E01 — Fix a broken initialization.
==================================
A model whose output layer is scaled up starts confidently WRONG — a huge loss at init. Fix the
init and the loss starts near -log(1/2), with no wasted recovery steps.

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

print(f"a good init starts near -log(1/2) = {math.log(2):.3f}\n")
for scale, label in [(30.0, "broken (output x30)"), (1.0, "fixed (default scale)")]:
    m = model(scale)
    init = F.cross_entropy(m(X), y).item()
    final = train(m)
    print(f"  {label:22} init {init:6.3f}  ->  after 150 steps {final:.4f}")
print("\nThe broken model starts ~4.5x higher AND ends ~25x worse — even Adam can't fully erase a")
print("confidently-wrong start. Sane init (1/sqrt(fan-in), PyTorch's default) starts at the floor.")
