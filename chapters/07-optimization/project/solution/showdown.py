"""
The Optimizer Showdown — REFERENCE SOLUTION.
============================================
Train SGD, Momentum, and Adam on two-moons, record their full loss CURVES, and plot them on one
chart. The shape of each curve tells the story.

    python showdown.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from optimizers import SGD, Momentum, Adam, train_with

opts = [("SGD", SGD(lr=0.5)), ("Momentum", Momentum(lr=0.1)), ("Adam", Adam(lr=0.05))]
curves = {name: train_with(opt, steps=300) for name, opt in opts}

for name, c in curves.items():
    print(f"  {name:9} start {c[0]:.3f} -> final {c[-1]:.4f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, c in curves.items():
        plt.plot(c, label=name)
    plt.yscale("log")
    plt.xlabel("step"); plt.ylabel("loss (log scale)")
    plt.title("Optimizer showdown on two-moons")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("showdown.png", dpi=120)
    print("\nsaved showdown.png — Adam's curve plunges fastest and settles lowest.")
except ImportError:
    print("(install matplotlib to see the chart)")
