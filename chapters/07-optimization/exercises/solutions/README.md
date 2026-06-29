# Chapter 7 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All import
[`../../code/optimizers.py`](../../code/optimizers.py) (the optimizers + the two-moons task).

| Script | Shows |
|--------|-------|
| [`e01_fix_init.py`](./e01_fix_init.py) | Broken init (×30) starts ~4.5× higher and ends ~25× worse than a sane init. |
| [`e02_lr_sweep.py`](./e02_lr_sweep.py) | Final loss vs learning rate — crawling below ~0.01, diverging above ~1. |
| [`e03_momentum_beta.py`](./e03_momentum_beta.py) | Higher `beta` descends faster (0.23 → 0.0000 across 0 → 0.99). |
| [`e04_schedule.py`](./e04_schedule.py) | A warmup→cosine LR curve (and why it's about stability, not toy-loss). |

```bash
python chapters/07-optimization/exercises/solutions/e01_fix_init.py
```
