# Chapter 6 — Exercises

Poke at the tokenizer you built. **Try first**; solutions are in [`solutions/`](./solutions/).
All import the reference [`../code/bpe.py`](../code/bpe.py), so run from anywhere:

```bash
python chapters/06-tokenization/exercises/solutions/e01_compression.py
```

---

### E01 — The vocab-size ↔ compression curve 🥇
Train BPE at several vocab sizes (256, 512, 1024, 2048) and measure the **compression** on
held-out text (bytes ÷ tokens). Plot it — you'll see the gains shrink as the vocab grows.

- 🧩 **Scaffold:** [`starter/e01_compression.py`](./starter/e01_compression.py).
- ✅ **Solution:** [`solutions/e01_compression.py`](./solutions/e01_compression.py) (saves `compression.png`).

  <details><summary>💡 Hint</summary>

  For each `vs` in your list: make a `BasicTokenizer()`, `tok.train(train_text, vs)`, then the
  ratio is `len(held.encode("utf-8")) / len(tok.encode(held))`. Collect the ratios and plot.
  </details>

### E02 — Round-trip stress test
Encode then decode a battery of tricky strings — accents, emoji, even Chinese the tokenizer
never saw — and confirm every one comes back **exactly**.

- **Why it matters:** it proves the payoff of working in *bytes* — nothing is ever
  out-of-vocabulary, so unseen text falls back to single-byte tokens and still round-trips.
- ✅ Solution: [`solutions/e02_roundtrip.py`](./solutions/e02_roundtrip.py).

### E03 — Inspect what BPE learned
Print the first merges (the commonest letter pairs) and the **longest tokens** in the vocab.
On Shakespeare, BPE rediscovers English structure — and even turns whole speaker headers like
`MENENIUS:\n` into single tokens.

- ✅ Solution: [`solutions/e03_inspect_merges.py`](./solutions/e03_inspect_merges.py).

### E04 — Regex splitting (the GPT-2 way)
Compare the `BasicTokenizer` (merges across spaces) with the `RegexTokenizer` (splits into
word-ish chunks first). Watch the difference: Basic learns trailing-space tokens like `'e '`;
Regex learns **leading-space** word units like `' the'`, `' Citizen'` — exactly how real GPT
tokenizers work.

- ✅ Solution: [`solutions/e04_regex.py`](./solutions/e04_regex.py).

> 🧠 **Takeaway:** a tokenizer is just bytes + merges + a dictionary — but those choices (vocab
> size, splitting) shape how the model sees *all* text.
