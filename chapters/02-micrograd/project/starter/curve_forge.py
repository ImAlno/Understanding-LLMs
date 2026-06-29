"""
Curve Forge — STARTER scaffold.
===============================
Fill in the 3 TODOs in the training loop. Everything else (engine, model, plotting) is
done. A full reference is in ../solution/curve_forge.py.

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


N = 25
xs = [-2.0 + 4.0 * i / (N - 1) for i in range(N)]
ys = [target(x) for x in xs]

model = MLP(1, [16, 16, 1])
print(f"{len(model.parameters())} parameters, training on {N} points...\n")

lr = 0.1
for step in range(150):
    preds = [model([x]) for x in xs]

    # ✍️ TODO #1: mean-squared error — the AVERAGE of (pred - y)**2 over all points.
    #    (the MEAN: divide by N — not a plain sum like the train demo)
    loss = None

    if loss is None:
        raise SystemExit("Fill in TODO #1 (the loss) first, then run again.")

    model.zero_grad()
    # ✍️ TODO #2: backprop — one call on `loss` to fill every parameter's .grad

    # ✍️ TODO #3: update — for each p in model.parameters(), step p.data a small amount
    #             AGAINST its gradient (use lr)

    if step % 10 == 0 or step == 149:
        print(f"step {step:3d} | loss {loss.data:.5f}")

print(f"\nfinal loss {loss.data:.5f}")

# ---- draw the fit (given) ----
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
