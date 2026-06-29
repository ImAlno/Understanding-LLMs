# Chapter 2 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** Each imports the reference engine
from `../../code/engine.py`.

| Script | Covers | Shows |
|--------|--------|-------|
| [`e01_gradcheck.py`](./e01_gradcheck.py) | E01 | A tangled expression in your engine **and** PyTorch — every gradient matches to ~1e-6. |
| [`e02_finite_diff.py`](./e02_finite_diff.py) | E02 | Backprop gradients match the "nudge and measure" finite difference. |
| [`e03_reused_var.py`](./e03_reused_var.py) | E03 | `d = a*a` gives `a.grad = 2a` — only because `_backward` accumulates with `+=`. |
| [`e04_add_exp.py`](./e04_add_exp.py) | E04 | Adding `exp()` (forward + local rule) and gradient-checking it vs PyTorch. |
| [`e05_tanh_from_exp.py`](./e05_tanh_from_exp.py) | E05 | `tanh` built from `exp` and `+ - * /` gives the identical value *and* gradient. |

```bash
python chapters/02-micrograd/exercises/solutions/e01_gradcheck.py
```
