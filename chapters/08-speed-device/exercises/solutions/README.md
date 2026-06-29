# Chapter 8 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All import
[`../../code/benchmark.py`](../../code/benchmark.py) and run on any device (best on a GPU — Colab).

| Script | Shows |
|--------|-------|
| [`e01_same_device.py`](./e01_same_device.py) | The CPU-batch/GPU-model `RuntimeError`, and the `.to(device)` fix. |
| [`e02_crossover.py`](./e02_crossover.py) | CPU wins for small matmuls, GPU for big — the crossover (~1024 on a laptop GPU). |
| [`e03_synchronize.py`](./e03_synchronize.py) | Timing without `synchronize` looks ~100s× too fast (you timed the launch). |
| [`e04_transfer.py`](./e04_transfer.py) | Moving a tensor CPU→GPU costs about as much as a matmul on it. |

```bash
python chapters/08-speed-device/exercises/solutions/e02_crossover.py
```
