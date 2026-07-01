# Chapter 10 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All run on a plain CPU (gloo + single-process
simulations); E02 launches real processes.

| Script | Shows |
|--------|-------|
| [`e01_average_grads.py`](./e01_average_grads.py) | Averaging shard-gradients == the full-batch gradient (diff ~6e-8). |
| [`e02_allreduce.py`](./e02_allreduce.py) | A real 4-process `gloo` all-reduce — every rank ends with the same mean. |
| [`e03_sampler.py`](./e03_sampler.py) | No sampler → 100% overlap; `r::N` sharding → disjoint + complete. |
| [`e04_zero_table.py`](./e04_zero_table.py) | Per-GPU memory by ZeRO stage across model sizes (which fit an 80 GB card). |

```bash
python chapters/10-speed-distributed/exercises/solutions/e03_sampler.py
```
