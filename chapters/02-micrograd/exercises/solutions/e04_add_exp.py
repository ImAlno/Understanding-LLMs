"""
E04 — Add a new operation: exp().
=================================
Every operation in the engine follows the same template — forward, then a local
`_backward`. Here's `exp` (already in engine.py), and a gradient-check proving the rule
`d(e^x)/dx = e^x` is right.

If you're adding it to your OWN engine, this is the method:

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad      # d(e^x)/dx = e^x = out.data
        out._backward = _backward
        return out

Run:  python e04_add_exp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from engine import Value
import torch

# a small expression that uses exp
x = Value(0.7)
y = (x * 2).exp() + x
y.backward()

xt = torch.tensor([0.7], dtype=torch.double, requires_grad=True)
yt = (xt * 2).exp() + xt
yt.backward()

ok_y = abs(y.data - yt.item()) < 1e-6
ok_g = abs(x.grad - xt.grad.item()) < 1e-6
print(f"  value:  engine={y.data:.6f}  torch={yt.item():.6f}   {'✅' if ok_y else '❌'}")
print(f"  x.grad: engine={x.grad:.6f}  torch={xt.grad.item():.6f}   {'✅' if ok_g else '❌'}")
print("\nThe new op backprops correctly — exp's local derivative is just its own output.")
