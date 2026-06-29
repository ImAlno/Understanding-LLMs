# Chapter 5 — Exercises

Probe what makes a Transformer work by **removing or scaling each piece** and watching the val
loss. **Try first**; solutions are in [`solutions/`](./solutions/).

All four share one configurable GPT, [`solutions/mini_gpt.py`](./solutions/mini_gpt.py) — its
`train(...)` lets you toggle `residual`, `norm`, `n_layer`, `n_head`, and `activation`. Run from
anywhere:

```bash
python chapters/05-transformer/exercises/solutions/e01_residuals.py
```

Each trains a couple of small GPTs (~25-30s per model on a CPU).

---

### E01 — Ablate the residual connections 🥇
Train a 4-layer GPT **with** and **without** the `x = x + sublayer(x)` residuals. Without them,
gradients can't reach the early layers and the deep model trains much worse — the single most
important trick for depth.

- 🧩 **Scaffold:** [`starter/e01_residuals.py`](./starter/e01_residuals.py).
- ✅ **Solution:** [`solutions/e01_residuals.py`](./solutions/e01_residuals.py) — expect a clear
  win for residuals (~2.1 vs ~2.6).

  <details><summary>💡 Hint</summary>

  `mini_gpt.train(...)` returns `(val_loss, model)`. Call it once with `residual=True` and once
  with `residual=False`, keeping everything else equal, and print both losses.
  </details>

### E02 — Ablate LayerNorm
Train with and without LayerNorm. **Heads up — the result is the lesson:** at this toy scale,
with residuals already present, removing LayerNorm barely moves the loss (it can even land a
hair lower). LayerNorm buys *stability* at depth and long training, not a lower loss on a tiny
model — a good reminder that ablations don't always show the textbook effect on a toy.

- ✅ Solution: [`solutions/e02_layernorm.py`](./solutions/e02_layernorm.py).

### E03 — Does depth help? Scale `n_layer`
Train at 1, 2, and 4 layers and compare. More blocks = more rounds of "communicate, then
think" — and it goes lower here. (The only reason stacking this deep even trains is the
residual + LayerNorm scaffolding from E01/E02.)

- ✅ Solution: [`solutions/e03_depth.py`](./solutions/e03_depth.py).

### E04 — ReLU vs GELU
GPT-2 uses GELU (a smooth ReLU) in the feed-forward net. Swap the activation and compare — the
gap is small at this scale, which is itself worth knowing.

- ✅ Solution: [`solutions/e04_activation.py`](./solutions/e04_activation.py).

> 🧠 **Takeaway:** residuals and depth clearly help; LayerNorm and the exact activation are
> about stability and scale. You're now reasoning about architectures the way researchers do —
> by ablation.
