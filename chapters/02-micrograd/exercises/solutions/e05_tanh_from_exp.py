"""
E05 — Build tanh from exp, and watch the gradient come out identical.
=====================================================================
tanh(x) = (e^{2x} - 1) / (e^{2x} + 1). We build it as a composite of smaller ops and
confirm both the value AND the gradient match the direct tanh() — because the chain
rule composes automatically. This is why we only implement *primitive* operations.

Run:  python e05_tanh_from_exp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from engine import Value

X = 0.8

# direct: one tanh node
a = Value(X)
t_direct = a.tanh()
t_direct.backward()
g_direct = a.grad

# composite: built from exp, +, -, /
b = Value(X)
e2x = (b * 2).exp()
t_comp = (e2x - 1) / (e2x + 1)
t_comp.backward()
g_comp = b.grad

print(f"  value:  direct={t_direct.data:.6f}  composed={t_comp.data:.6f}   "
      f"{'✅' if abs(t_direct.data - t_comp.data) < 1e-9 else '❌'}")
print(f"  grad:   direct={g_direct:.6f}  composed={g_comp:.6f}   "
      f"{'✅' if abs(g_direct - g_comp) < 1e-9 else '❌'}")
print("\nSame value, same gradient. backprop doesn't care how you spell tanh —")
print("the chain rule flows through five nodes exactly as it would through one.")
