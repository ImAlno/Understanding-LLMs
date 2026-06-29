# Chapter 5 — Exercise Solutions

Runnable ablations. **Peek only after you've tried.** All share
[`mini_gpt.py`](./mini_gpt.py) (a configurable GPT) and the data from `../../code/gpt.py`; each
trains a couple of small GPTs (~25-30s per model).

| Script | Knob | Typical result |
|--------|------|----------------|
| [`e01_residuals.py`](./e01_residuals.py) | `residual` on/off | ~2.13 with, ~2.61 without — residuals clearly win. |
| [`e02_layernorm.py`](./e02_layernorm.py) | `norm` on/off | A wash at this scale (~2.13 either way) — LayerNorm is about stability at depth. |
| [`e03_depth.py`](./e03_depth.py) | `n_layer` 1/2/4 | 2.31 → 2.25 → 2.13 — deeper goes lower. |
| [`e04_activation.py`](./e04_activation.py) | `relu` vs `gelu` | ~2.13 vs ~2.13 — close at this scale. |

```bash
python chapters/05-transformer/exercises/solutions/e01_residuals.py
```
