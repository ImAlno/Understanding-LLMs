# Chapter 2 — Exercises

These deepen your grip on backprop. **Try each yourself first** (extend your notebook
engine, or start a fresh file). Worked, runnable solutions are in
[`solutions/`](./solutions/) — peek only after you've struggled a little. 🙂

The solutions import the reference engine from `../code/engine.py`, so run them from
anywhere:

```bash
python chapters/02-micrograd/exercises/solutions/e01_gradcheck.py
```

---

### E01 — Gradient-check your engine against PyTorch 🥇 *(most important)*
Build one tangled expression out of `+`, `-`, `*`, `**`, and `tanh` — in **your engine**
*and* in **PyTorch** — call `.backward()` on both, and assert every gradient matches.
This is how you *prove* your engine is correct (and how everyone debugs autograd).

- 🧩 **Scaffold:** [`starter/e01_gradcheck.py`](./starter/e01_gradcheck.py) (3 TODOs).
- ✅ **Solution:** [`solutions/e01_gradcheck.py`](./solutions/e01_gradcheck.py).

  <details><summary>💡 Hint 1 — the setup</summary>

  Write the expression once as a function `f(a, b)` so you can call it with your
  `Value`s and with `torch.tensor(..., requires_grad=True)` (use `dtype=torch.double`
  for an exact match). PyTorch tensors support the same `+ - * ** .tanh()`.
  </details>

  <details><summary>💡 Hint 2 — the comparison</summary>

  After `out.backward()` in each, compare `a.grad` to `a_torch.grad.item()` with
  `abs(...) < 1e-6`. Compare the forward values too (`out.data` vs `out_torch.item()`).
  </details>

### E02 — Check gradients with finite differences
No PyTorch needed: verify your engine's `.grad` against the "nudge and measure" method
from the lesson. Build an expression, `backward()` it, then nudge each input by a tiny
`h` and confirm `(out_new − out)/h` ≈ that input's `.grad`.

- **Why it matters:** finite differences are the ground truth a derivative *means*. If
  backprop and the nudge disagree, backprop is wrong.

  <details><summary>💡 Hint — the one gotcha</summary>

  You can't reuse the same graph for the nudge — once it's built, its `.data` values are
  fixed. Wrap the forward pass in a function `forward(a_val, b_val)` that builds **fresh**
  `Value`s and returns the output *number*, so you can call it at `a` and again at `a + h`.
  </details>
- ✅ Solution: [`solutions/e02_finite_diff.py`](./solutions/e02_finite_diff.py).

### E03 — Why `+=` matters: a reused variable
Backprop `d = a * a` with `a = 3`. What *should* `a.grad` be? Compute it by hand, then
check your engine. Then try `e = a + a`.

- **The point:** `a` feeds the multiply *twice*, so its gradient is the **sum** of both
  paths (`a.grad = 2a = 6`). This only works because `_backward` uses `+=`. Replace the
  `+=` with `=` in your engine and watch this break — that's the lesson.
- ✅ Solution: [`solutions/e03_reused_var.py`](./solutions/e03_reused_var.py).

### E04 — Add a new operation: `exp()`
Give **your notebook engine** a new method, the exponential `e^x`. (The reference
`engine.py` already ships `exp`, so you're checking your own version — or checking it
against PyTorch.) You write the forward (`math.exp(self.data)`) and — the real exercise —
the `_backward` rule, then gradient-check it.

- **Why it's neat:** `exp` has the most elegant derivative in math — `d(e^x)/dx = e^x` —
  so the local derivative is simply the output value itself.
- ✅ Solution: [`solutions/e04_add_exp.py`](./solutions/e04_add_exp.py).

  <details><summary>💡 Hint 1 — the shape of any new op</summary>

  Every op follows the same template: compute `out = Value(<forward>, (self,), 'exp')`,
  define a local `_backward()` that adds `<local derivative> * out.grad` to `self.grad`,
  attach it with `out._backward = _backward`, and `return out`.
  </details>

  <details><summary>💡 Hint 2 — the local derivative</summary>

  For `exp`, the local derivative *is* the output (`e^x`). In code that's `out.data`, so:
  `self.grad += out.data * out.grad`.
  </details>

### E05 — Build `tanh` from smaller pieces *(stretch)*
`tanh(x) = (e^{2x} − 1) / (e^{2x} + 1)`. Using your new `exp()` (E04) plus `+ - * /`,
build tanh *as a composite expression* and confirm both the value **and the gradient**
match your direct `tanh()`.

- **The "aha":** backprop doesn't care whether `tanh` is one node or five — the chain
  rule composes automatically and gives the identical gradient. That's why you only ever
  need to implement *primitive* ops.

  <details><summary>💡 Hint</summary>

  Make two separate inputs with the same value (say `0.8`). For the **composite**, compute
  `e2x = (x * 2).exp()` once, then `t = (e2x - 1) / (e2x + 1)`, and `t.backward()`. For the
  **direct** one, take a fresh `x2` and run `x2.tanh().backward()`. Compare `x.grad` from
  each — value and gradient should match to many decimals.
  </details>
- ✅ Solution: [`solutions/e05_tanh_from_exp.py`](./solutions/e05_tanh_from_exp.py).
