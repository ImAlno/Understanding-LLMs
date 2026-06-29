# 🔤 Mini-Project — Train Your Own Tokenizer

You built BPE from scratch in the notebook. Now train a **real** (regex-splitting) tokenizer on
a text of *your* choice and analyze it — the same workflow used to build GPT-2's tokenizer, just
smaller.

> **How this works:** [`starter/tokenizer_report.py`](./starter/tokenizer_report.py) has two
> TODOs — **train** the tokenizer and **measure** its compression — then it prints a report for
> you. A full reference is in [`solution/tokenizer_report.py`](./solution/tokenizer_report.py).

## 🎯 What it does

```bash
python starter/tokenizer_report.py        # ~5s on a CPU
```

It trains a `RegexTokenizer` on Shakespeare and prints:

```
=== your tokenizer ===
  round-trip:  ✓
  compression: 30000 bytes -> 16429 tokens  (1.83x)
  sample:      'First Citizen: we are all resolved rather to die ...'
               -> 25 tokens: ['First', ' Citizen', ':', ' we', ' are', ' all', ' re', 's', 'ol', ...]
  longest learned tokens: [b'CORIOLANUS', b'MENENIUS', b'CORIOLAN', b'SICINIUS', b' Citizen', b' Marcius', b'CORIOLA', b' people']
```

## 🛠️ Your TODOs

1. **Train** `tok` on `train_text` to `VOCAB_SIZE` (one call — see `RegexTokenizer.train`).
2. **Measure** compression: the number of UTF-8 bytes in `held`, and the number of tokens
   `tok.encode(held)` produces.

## ✅ Checking your work

- The round-trip assertion must pass (it's the correctness gate).
- Compression should be roughly **1.8×** at vocab 512 (the regex split trades a little
  compression for cleaner, word-aligned tokens).
- The sample should split into clean, leading-space word units.
- Compare against `solution/tokenizer_report.py`.

## 🚀 Make it yours (extensions)

- **Your own text.** Point `TEXT_PATH` at any `.txt` — a favorite book, your code, song lyrics.
  See what tokens BPE invents for *your* data (try Python source — it'll learn `    ` and `):`).
- **Bigger vocab.** Push `VOCAB_SIZE` to 2000+ and watch compression climb (and training slow).
- **Basic vs Regex.** Swap in `BasicTokenizer` and compare the longest tokens — does it glue
  words to spaces?
- **🏆 Stretch — plug it into your GPT.** Re-encode Shakespeare with your tokenizer, set the
  Chapter 5 GPT's `vocab_size` to your tokenizer's, and train. Each sequence now holds ~2× more
  text — your first taste of why real models tokenize. (Generation must `decode` with your
  tokenizer too.)

A tokenizer is the unglamorous front door of every LLM — and now you've built one end to end.
Next: [Chapter 7 — Optimization](../../07-optimization/), where we make training itself
faster and more stable. 🚪
