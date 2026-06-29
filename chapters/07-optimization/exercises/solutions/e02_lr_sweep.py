"""
E02 — Sweep the learning rate.
==============================
The learning rate is the most important knob: too small crawls, too big diverges.

Run:  python e02_lr_sweep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import math
from optimizers import train_with, Adam

print("Adam on two-moons, final loss after 300 steps, across learning rates:")
for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    final = train_with(Adam(lr=lr))[-1]
    if math.isnan(final) or final > 1.0:
        tag = "  <- diverged (too big)"
    elif final > 0.05:
        tag = "  <- crawling (too small)" if lr < 0.01 else "  <- unstable (too big)"
    else:
        tag = ""
    print(f"  lr {lr:<8} final {final:.4f}{tag}")
print("\nThere's a sweet spot (~0.01-0.1 here): below it learning crawls, above it the loss")
print("blows up. Finding this band is what a learning-rate sweep / range test is for.")
