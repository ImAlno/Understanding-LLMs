"""
Initialization — why the starting point matters.
=================================================
Two failure modes a bad initialization causes, and how to see them:

  1. Loss at init. A well-initialized classifier starts out unsure — its loss should be about
     -log(1/n_classes), the loss of uniform guessing. Scale the output layer up and it starts
     wildly *confident and wrong*, so the loss is huge and the first many steps are wasted just
     recovering (the "hockey-stick" loss curve).

  2. Saturated activations. Too-large weights push tanh to ±1, where its gradient is ~0 — so
     those neurons stop learning ("dead" gradients).

Run:  python init_demo.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizers import make_moons

X, y = make_moons()
n_classes = 2
print(f"Uniform guessing has loss -log(1/{n_classes}) = {math.log(n_classes):.3f} — a good init starts near it.\n")

print("loss at initialization:")
for scale, label in [(1.0, "good init (default)"), (30.0, "bad init (output layer ×30)")]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, n_classes))
    with torch.no_grad():
        model[-1].weight *= scale
    loss = F.cross_entropy(model(X), y)
    print(f"  {label:30} {loss.item():7.3f}")

print("\ntanh saturation (fraction of activations stuck near ±1, where the gradient ~vanishes):")
for scale, label in [(0.5, "good init (small weights)"), (3.0, "bad init (large weights)")]:
    torch.manual_seed(0)
    W = torch.randn(2, 500) * scale
    h = torch.tanh(X @ W)
    print(f"  {label:30} {(h.abs() > 0.99).float().mean().item():6.1%}")

print("\nBad init wastes the first steps un-saturating and recovering from confident-wrong guesses.")
print("Good init starts near the theoretical loss, with healthy gradients flowing everywhere.")
