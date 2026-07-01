# Chapter 12 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All run on a CPU; E02–E04 import the GPT from
[`../../code/gpt.py`](../../code/gpt.py).

| Script | Shows |
|--------|-------|
| [`e01_implement_cache.py`](./e01_implement_cache.py) | Append k/v to a cache; cached attention == full attention (exact). |
| [`e02_identical.py`](./e02_identical.py) | On the GPT, `generate_cached` == `generate_naive`, every time. |
| [`e03_speedup.py`](./e03_speedup.py) | Speedup climbs with length: ~2× at 64 tokens, ~5× at 480. |
| [`e04_memory.py`](./e04_memory.py) | Cache size by model × context; Llama-2-7B @ 32k (17 GB) > its weights (14 GB). |

```bash
python chapters/12-inference-kv-cache/exercises/solutions/e03_speedup.py
```
