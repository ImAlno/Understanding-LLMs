"""
E03 — Momentum's beta.
======================
beta controls how much past velocity to keep. 0 = plain SGD; higher = more smoothing/speed,
until too high overshoots.

Run:  python e03_momentum_beta.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from optimizers import train_with, Momentum

print("Momentum on two-moons (lr=0.1), final loss after 300 steps, across beta:")
for beta in [0.0, 0.5, 0.9, 0.99]:
    final = train_with(Momentum(lr=0.1, beta=beta))[-1]
    print(f"  beta {beta:<5} final {final:.4f}")
print("\nbeta=0 is plain SGD; raising it averages more gradients and descends faster — here it's")
print("monotonic, with 0.99 best on this easy task. On harder loss surfaces too-high beta makes")
print("the velocity overshoot, so ~0.9 is the usual safe default.")
