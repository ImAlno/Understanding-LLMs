# Chapter 11 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** Each imports the pipeline helpers from
[`../../code/`](../../code/) and works on a small temp corpus (no downloads).

| Script | Shows |
|--------|-------|
| [`e01_get_batch.py`](./e01_get_batch.py) | The memmap random-window sampler: `(x, y)`, `y` shifted one right, int64. |
| [`e02_shift.py`](./e02_shift.py) | `x` above `y`, aligned — every x token predicts the y token below it. |
| [`e03_no_leak.py`](./e03_no_leak.py) | Train windows never reach the val region (split before windowing). |
| [`e04_packing.py`](./e04_packing.py) | `uint16` `.bin` round-trips losslessly at 2 bytes/token; wrong dtype → garbage. |

```bash
python chapters/11-datasets/exercises/solutions/e02_shift.py
```
