# Chapter 13 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All run on a CPU; E04 imports the GPT from
[`../../code/gpt.py`](../../code/gpt.py).

| Script | Shows |
|--------|-------|
| [`e01_quantize.py`](./e01_quantize.py) | The int8 map: floats → integers `[13,-54,127,-6,79]` → recovered. |
| [`e02_per_channel.py`](./e02_per_channel.py) | Per-channel vs per-tensor on outliers — ~7× less error at int8. |
| [`e03_bits.py`](./e03_bits.py) | int8→int2 sweep: error climbs (0.007 → 0.8) as size falls (7 → 1.8 GB). |
| [`e04_quantize_model.py`](./e04_quantize_model.py) | Quantize a GPT: int8 drift 0.004 / half size, int4 0.074 / quarter size. |

```bash
python chapters/13-inference-quantization/exercises/solutions/e03_bits.py
```
