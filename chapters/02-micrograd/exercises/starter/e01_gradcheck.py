"""
E01 — Gradient-check your engine against PyTorch (STARTER scaffold).
===================================================================
Fill in the 3 TODOs. A full reference is in ../solutions/e01_gradcheck.py.

Run:  python e01_gradcheck.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from engine import Value
import torch


def f(a, b):
    """One expression that works on BOTH Values and torch tensors (they share the
    operators + - * ** and .tanh()).

    TODO #1: build your OWN tangle of a few ops and return it. Start trivial to confirm
    the setup works (e.g. `return a * b + b`), then make it gnarlier — throw in `**`, a
    subtraction, and a `.tanh()` — so the gradient is interesting.
    """
    return ...  # <-- replace with your expression


# --- our engine ---
a, b = Value(1.5), Value(-2.0)
out = f(a, b)
# ✍️ TODO #2: backprop `out` so that a.grad and b.grad get filled in.

# --- pytorch (double precision for an exact match) ---
at = torch.tensor([1.5], dtype=torch.double, requires_grad=True)
bt = torch.tensor([-2.0], dtype=torch.double, requires_grad=True)
outt = f(at, bt)
# ✍️ TODO #2 (again): backprop the torch version too.

# ✍️ TODO #3: for each of (forward value, a.grad, b.grad), check the engine value is
#    within 1e-6 of the torch value. The torch values are: outt.item(), at.grad.item(),
#    bt.grad.item(). Print a ✅ / ❌ for each.
print("fill in the TODOs to compare your engine's gradients with PyTorch's")
