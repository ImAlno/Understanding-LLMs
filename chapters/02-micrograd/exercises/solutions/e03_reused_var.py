"""
E03 — Why `_backward` accumulates with `+=`: a reused variable.
===============================================================
When one Value feeds a computation more than once, its gradient is the SUM of the
contributions from every path. That's why every `_backward` does `self.grad += ...`
instead of `=`. Here's the smallest example that proves it.

Run:  python e03_reused_var.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from engine import Value

# d = a * a.  By hand: d/da = a + a = 2a. At a=3 that's 6.
# In the graph, `a` is BOTH inputs to the multiply, so its grad gets a contribution
# from each — 3 from one, 3 from the other — and += sums them to 6.
a = Value(3.0)
d = a * a
d.backward()
print(f"  d = a*a  →  a.grad = {a.grad}   (should be 2a = 6)   "
      f"{'✅' if a.grad == 6.0 else '❌'}")

# e = a + a.  By hand: d/da = 1 + 1 = 2.
a = Value(3.0)
e = a + a
e.backward()
print(f"  e = a+a  →  a.grad = {a.grad}   (should be 2)         "
      f"{'✅' if a.grad == 2.0 else '❌'}")

print("\nBoth rely on `+=`. Change it to `=` in engine.py and these become 3 and 1 —")
print("wrong. Accumulation is what makes reused variables (and real nets) work.")
