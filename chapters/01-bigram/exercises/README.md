# Chapter 1 — Exercises

These cement the chapter's ideas. **Try each one yourself first** — open a notebook
or a fresh `.py` file and tinker. When you're stuck or want to check your answer,
the worked solutions live in [`solutions/`](./solutions/). Reading the solution
*after* struggling for a few minutes is a great way to learn; reading it first is
not. 🙂

Each solution is a small, runnable script. From the repo root:

```bash
python chapters/01-bigram/exercises/solutions/e01_trigram.py
```

---

### E01 — Build a trigram model 🥇 *(most important)*
A bigram uses **one** previous character. A **trigram** uses **two**. Build a
trigram model *by counting*: make a `27×27×27` count tensor `N`, where
`N[i, j, k]` is "how often did character `k` follow the pair `(i, j)`?", normalize
along the last dimension, and compute the loss.

- **Question:** is the loss lower than the bigram's ~2.45? Do the sampled names
  look more name-like? (They should — more context = better predictions.)

  <details>
  <summary>💡 Hint 1 — getting started</summary>

  Pad each name with **two** start tokens and one end token:
  `['.', '.'] + list(word) + ['.']`. A trigram is then any three consecutive
  characters in that padded list.
  </details>

  <details>
  <summary>💡 Hint 2 — the key lines</summary>

  Make a 3-D table `N = torch.zeros((27, 27, 27))`. Walk triples with
  `for a, b, c in zip(chs, chs[1:], chs[2:])` and do `N[stoi[a], stoi[b], stoi[c]] += 1`.
  To get probabilities, normalize along the **last** dimension:
  `P = (N + 1).float(); P /= P.sum(dim=2, keepdim=True)`.
  </details>

- 🧩 **Prefer a scaffold?** Start from [`starter/e01_trigram.py`](./starter/e01_trigram.py)
  — the setup is done, with 3 TODOs to fill (a gentler on-ramp than a blank file).
- ✅ **Full solution:** [`solutions/e01_trigram.py`](./solutions/e01_trigram.py)
  (try the hints and the scaffold first!)

### E02 — Train / dev / test split
So far we measured loss on the *same* names we trained on — that's like grading an
exam you've already seen. Shuffle the names and split them **80% train / 10% dev /
10% test**. Build the counts on **train only**, then report the loss on all three
splits.

- **Question:** is the test loss higher than the train loss? Why does evaluating on
  *held-out* data give a more honest measure of quality?

  <details>
  <summary>💡 Hint — how to start</summary>

  Reuse your work: wrap your **counting** code and your **loss** code from the lesson
  as two small functions, then call them on three different word-lists. Split with:
  `import random; random.seed(42); random.shuffle(words)`, then
  `n1, n2 = int(0.8*len(words)), int(0.9*len(words))` and slice
  `words[:n1]`, `words[n1:n2]`, `words[n2:]`.
  </details>
- Solution: [`solutions/e02_03_split_and_smoothing.py`](./solutions/e02_03_split_and_smoothing.py)

### E03 — Tune the smoothing on the dev set
The `+1` smoothing was arbitrary. Try several values
(`0.01, 0.1, 1, 3, 10, 100, ...`), build the model on **train**, and pick the value
that gives the lowest **dev** loss. Only *then* report the **test** loss for that
chosen value.

- **Key idea:** you tune knobs ("hyperparameters") on **dev**, and touch **test**
  only once at the very end. This discipline prevents you from fooling yourself —
  you'll rely on it for the rest of the course.
- Solution: same file as E02.

### E04 — Delete the one-hot, just index
In `bigram_nn.py` we compute `F.one_hot(xs, 27).float() @ W`. Show — with
`torch.allclose` — that this is **exactly** equal to `W[xs]` (plain row indexing).

- **Why it matters:** "look up a row" *is* what an **embedding** layer does, and
  it's how every real model handles inputs (Chapter 3). One-hot + matmul is just a
  wasteful way to write the same thing.

  <details>
  <summary>💡 Hint</summary>

  `W[xs]` indexes `W` with a whole *tensor* of ids at once: for each id in `xs` it
  grabs that row of `W`, producing a `(len(xs), 27)` result — the same thing the
  one-hot-times-`W` produces. Build both and compare them with `torch.allclose(...)`.
  </details>
- Solution: [`solutions/e04_05_nn_refinements.py`](./solutions/e04_05_nn_refinements.py)

### E05 — Use `F.cross_entropy`
Replace the hand-written `softmax → log → mean` with the single call
`F.cross_entropy(logits, ys)`. Confirm it gives the same loss, then train with it.

- **Why it matters:** `cross_entropy` is faster and **numerically stable** (it
  avoids `exp()` of large logits overflowing to `inf`). Real training loops always
  use it. We'll rely on it from Chapter 3 onward.
- Solution: same file as E04.

### E06 — Sample from your trigram *(stretch)*
Take your E01 trigram model and generate 20 names from it (start from `(., .)` and
slide your two-character window forward as you sample). Put them next to 20 bigram
names. Can you *feel* the difference an extra character of memory makes?

- The new trick here is the **sliding two-character window** — the bigram loop only
  tracked one index, so this is genuinely new. Hints if you want them:

  <details>
  <summary>💡 Hint 1 — the sliding window</summary>

  Keep **two** indices for the current context, `i, j`, both starting at `0` (the
  `(. .)` start). Sample the next id `k` from `P[i, j]`. If `k == 0` (end token),
  stop; otherwise emit it and **slide the window**: `i, j = j, k`. Repeat.
  </details>

  <details>
  <summary>💡 Hint 2 — the full loop</summary>

  ```python
  g = torch.Generator().manual_seed(2147483647)
  i, j, out = 0, 0, []
  while True:
      k = torch.multinomial(P[i, j], num_samples=1, generator=g).item()
      if k == 0:
          break
      out.append(itos[k])
      i, j = j, k          # slide the 2-char window forward
  print("".join(out))
  ```
  </details>
- **Reflection:** if two characters of context help this much, imagine *hundreds*.
  That craving for more context is exactly what leads us to the MLP (Ch 3),
  attention (Ch 4), and the Transformer (Ch 5).
