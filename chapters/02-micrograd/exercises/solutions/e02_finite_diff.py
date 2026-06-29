"""
E02 — Check backprop against finite differences (no PyTorch needed).
====================================================================
A gradient is "how much the output moves when you nudge an input." So we can check our
engine by literally nudging each input and measuring — it must match `.grad`.

Run:  python e02_finite_diff.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from engine import Value


def forward(a_val, b_val):
    """Return just the output number for given inputs."""
    a, b = Value(a_val), Value(b_val)
    out = (a * b + b ** 2).tanh() * 2.0
    return out.data


# backprop gradients
a, b = Value(1.5), Value(-2.0)
out = (a * b + b ** 2).tanh() * 2.0
out.backward()

# finite-difference gradients: nudge one input by h, see how much `out` changes
h = 1e-6
base = forward(1.5, -2.0)
fd_a = (forward(1.5 + h, -2.0) - base) / h
fd_b = (forward(1.5, -2.0 + h) - base) / h

print(f"  a.grad: backprop={a.grad:+.5f}   finite-diff={fd_a:+.5f}   "
      f"{'✅' if abs(a.grad - fd_a) < 1e-3 else '❌'}")
print(f"  b.grad: backprop={b.grad:+.5f}   finite-diff={fd_b:+.5f}   "
      f"{'✅' if abs(b.grad - fd_b) < 1e-3 else '❌'}")
print("\nThey agree — backprop computes exactly what a derivative *means*.")
