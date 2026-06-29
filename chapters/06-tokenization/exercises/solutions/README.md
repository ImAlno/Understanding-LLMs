# Chapter 6 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All import the reference
[`../../code/bpe.py`](../../code/bpe.py).

| Script | Shows |
|--------|-------|
| [`e01_compression.py`](./e01_compression.py) | Compression climbs with vocab size (1.00→1.89→2.35→2.72×), with diminishing returns. Saves `compression.png`. |
| [`e02_roundtrip.py`](./e02_roundtrip.py) | `decode(encode(text)) == text` for accents, emoji, and unseen Chinese — bytes mean nothing is out-of-vocabulary. |
| [`e03_inspect_merges.py`](./e03_inspect_merges.py) | The first merges (common letter pairs) and longest tokens (whole speaker headers). |
| [`e04_regex.py`](./e04_regex.py) | Basic (trailing-space `'e '`) vs Regex (leading-space `' Citizen'`) tokenization. |

```bash
python chapters/06-tokenization/exercises/solutions/e01_compression.py
```
