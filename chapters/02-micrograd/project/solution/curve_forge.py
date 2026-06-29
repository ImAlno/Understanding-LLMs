"""
Curve Forge — REFERENCE SOLUTION.
=================================
Train a micrograd MLP to trace a wiggly curve, and save a picture of the fit.

Run:  python curve_forge.py
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import random
from nn import MLP

random.seed(42)


def target(x):
    return 0.8 * math.sin(3.0 * x)


# training data: N points evenly spaced on [-2, 2]
N = 25
xs = [-2.0 + 4.0 * i / (N - 1) for i in range(N)]
ys = [target(x) for x in xs]

model = MLP(1, [16, 16, 1])
print(f"{len(model.parameters())} parameters, training on {N} points...\n")

lr = 0.1
for step in range(150):
    preds = [model([x]) for x in xs]
    # 1) mean-squared-error loss
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) * (1.0 / N)
    # 2) backprop
    model.zero_grad()
    loss.backward()
    # 3) update
    for p in model.parameters():
        p.data += -lr * p.grad
    if step % 30 == 0 or step == 149:
        print(f"step {step:3d} | loss {loss.data:.5f}")

print(f"\nfinal loss {loss.data:.5f}")

# ---- draw the fit ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = [-2.0 + 4.0 * i / 199 for i in range(200)]
    plt.figure(figsize=(8, 5))
    plt.plot(grid, [target(x) for x in grid], label="target  0.8·sin(3x)", linewidth=2)
    plt.plot(grid, [model([x]).data for x in grid], "--", label="network", linewidth=2)
    plt.scatter(xs, ys, s=15, c="k", zorder=3, label="training points")
    plt.legend(); plt.title(f"Curve Forge — final loss {loss.data:.4f}")
    plt.tight_layout(); plt.savefig("curve_fit.png", dpi=120)
    print("saved curve_fit.png — the dashed curve should hug the solid one.")
except ImportError:
    print("(install matplotlib to see the plot)")
