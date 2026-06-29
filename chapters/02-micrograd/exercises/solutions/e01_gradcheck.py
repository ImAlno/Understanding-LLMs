"""
E01 — Gradient-check the engine against PyTorch.
================================================
Build the SAME expression with our Value engine and with torch tensors, backprop both,
and assert every gradient agrees. This is the gold-standard test for an autograd engine.

Run:  python e01_gradcheck.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from engine import Value
import torch


def f(a, b):
    """One expression, written so it works on Values OR torch tensors."""
    c = a * b
    d = c + b ** 2 - a
    e = d.tanh() * 3.0
    return (e - c) ** 2 / 2.0


# --- our engine ---
a, b = Value(1.5), Value(-2.0)
out = f(a, b)
out.backward()

# --- pytorch (double precision for an exact comparison) ---
at = torch.tensor([1.5], dtype=torch.double, requires_grad=True)
bt = torch.tensor([-2.0], dtype=torch.double, requires_grad=True)
outt = f(at, bt)
outt.backward()

rows = [
    ("forward", out.data, outt.item()),
    ("a.grad", a.grad, at.grad.item()),
    ("b.grad", b.grad, bt.grad.item()),
]
all_ok = True
for name, mine, theirs in rows:
    ok = abs(mine - theirs) < 1e-6
    all_ok &= ok
    print(f"  {'✅' if ok else '❌'} {name:8s} engine={mine:+.6f}  torch={theirs:+.6f}")
print("\nALL MATCH — your engine computes the same gradients as PyTorch."
      if all_ok else "\nMISMATCH — check your _backward rules.")
