# Chapter 3 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** Each imports the data helpers from
`../../code/mlp.py`.

| Script | Covers | Shows |
|--------|--------|-------|
| [`e01_tune.py`](./e01_tune.py) | E01 | Sweeps a few configs and prints their **dev** losses — more context + a bigger net helps. |
| [`e02_embed_viz.py`](./e02_embed_viz.py) | E02 | Trains with `n_embd=2` and plots the 27 character embeddings (`embeddings.png`) — vowels cluster. |
| [`e03_gelu.py`](./e03_gelu.py) | E03 | `tanh` vs **GELU**, side by side. |
| [`e04_overfit.py`](./e04_overfit.py) | E04 | Trains on 200 names: train loss → ~0 while dev loss stays high — overfitting, made visible. |
| [`e05_deeper.py`](./e05_deeper.py) | E05 | A two-hidden-layer MLP. |

```bash
python chapters/03-ngram-mlp/exercises/solutions/e01_tune.py
```
