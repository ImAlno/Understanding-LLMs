# Chapter 9 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All run on any device — the formats behave
identically on a CPU (E04 imports `get_device` from [`../../code/mixed_precision_train.py`](../../code/mixed_precision_train.py)).

| Script | Shows |
|--------|-------|
| [`e01_memory.py`](./e01_memory.py) | A 100M-param model: 0.40 GB in fp32, 0.20 GB in fp16/bf16 — half. |
| [`e02_overflow.py`](./e02_overflow.py) | fp16 overflows around 2¹⁶ (max 65504); bf16 holds to ~2¹²⁸. |
| [`e03_loss_scaling.py`](./e03_loss_scaling.py) | A `1e-8` gradient: lost at scale 1, recovered once the scale lifts it into fp16's range. |
| [`e04_autocast_dtypes.py`](./e04_autocast_dtypes.py) | Inside autocast: matmul → bf16, but softmax / layernorm / sum stay fp32. |

```bash
python chapters/09-speed-precision/exercises/solutions/e01_memory.py
```
