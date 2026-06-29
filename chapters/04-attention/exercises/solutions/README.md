# Chapter 4 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** E02–E05 import the data/vocab from
`../../code/attention.py` and train a small head (a few seconds each).

| Script | Covers | Shows |
|--------|--------|-------|
| [`e01_avg_proof.py`](./e01_avg_proof.py) | E01 | The triangular matmul gives exactly the loop's running average (to ~1e-7). |
| [`e02_visualize.py`](./e02_visualize.py) | E02 | A heatmap of one head's `(T, T)` attention weights — the causal triangle, made visible (`attention.png`). |
| [`e03_no_mask.py`](./e03_no_mask.py) | E03 | With vs without the causal mask: the no-mask model "cheats" to a much lower (meaningless) loss. |
| [`e04_block_size.py`](./e04_block_size.py) | E04 | Val loss vs context length — more context helps. |
| [`e05_multihead.py`](./e05_multihead.py) | E05 | Single head vs 4-head attention. |

```bash
python chapters/04-attention/exercises/solutions/e01_avg_proof.py
```
