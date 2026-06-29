# Chapter 3 — Exercises

Push the MLP further. **Try each yourself first**; runnable solutions are in
[`solutions/`](./solutions/) — peek after you've struggled. They import the data helpers
from `../code/mlp.py`, so run them from anywhere:

```bash
python chapters/03-ngram-mlp/exercises/solutions/e01_tune.py
```

> 💡 **Reuse the harness.** Every exercise builds on the same train-an-MLP loop. Fastest
> start: copy the setup from [`../code/mlp.py`](../code/mlp.py) (load → `build_dataset` →
> init params → train loop) and change only the one thing the exercise asks. E01 ships a
> [`starter/e01_tune.py`](./starter/e01_tune.py) you can adapt for the others.

---

### E01 — Tune the model on the dev set 🥇 *(most important)*
The config (`block_size`, `n_embd`, `n_hidden`, `steps`, learning rate) was a first guess.
**Sweep it** — change one thing at a time, train, and read the **dev** loss — to find a
setup that beats the baseline. (At the quick **8k-step** budget these exercises use, the
baseline lands around **~2.23**; the full 30k-step run in `mlp.py` reaches ~2.15 — so
`steps` is itself a lever.) Touch the *test* set only once, at the very end.

- 🧩 **Scaffold:** [`starter/e01_tune.py`](./starter/e01_tune.py).
- ✅ **Solution:** [`solutions/e01_tune.py`](./solutions/e01_tune.py).

  <details><summary>💡 Hint 1 — what to turn</summary>

  Biggest levers, roughly in order: **more context** (`block_size` 3 → 4 → 5+), a **bigger
  hidden layer** (`n_hidden`), a **richer embedding** (`n_embd`), and simply **training
  longer** (`steps`). Change ONE at a time so you know what helped.
  </details>

  <details><summary>💡 Hint 2 — the method</summary>

  Wrap "build data → init params → train → return dev loss" in a function `run(block_size,
  n_embd, n_hidden, steps)`, then call it for a few configs and print a little table of dev
  losses. Reuse `build_dataset(words, stoi, block_size)` from `mlp.py`.
  </details>

### E02 — Visualize the learned embeddings
Set `n_embd = 2`, train, then **scatter-plot the 27 rows of `C`** with each point labelled
by its character. You'll *see* the structure the model invented — vowels tend to cluster,
`.` sits off on its own, etc.

- **Why it matters:** this is the payoff of embeddings made visible — similar characters
  end up near each other because that lowers the loss.

  <details><summary>💡 Hint — the plotting bit (matplotlib)</summary>

  After training, get the table as plain numbers with `Cd = C.detach()`, then:
  ```python
  import matplotlib.pyplot as plt
  plt.scatter(Cd[:, 0], Cd[:, 1])
  for i in range(27):
      plt.text(Cd[i, 0].item(), Cd[i, 1].item(), itos[i])
  plt.savefig("embeddings.png")
  ```
  </details>
- ✅ Solution: [`solutions/e02_embed_viz.py`](./solutions/e02_embed_viz.py) (saves `embeddings.png`).

### E03 — Swap `tanh` for GELU
Modern nets rarely use `tanh`. Replace it with **GELU** (`F.gelu`) — or ReLU (`F.relu`) —
retrain, and compare the dev loss.

- **Why it matters:** GELU is the activation in GPT and most modern LLMs. See if it helps
  here (it's usually a small win) — and you'll meet it again from Chapter 5 on.

  <details><summary>💡 Hint</summary>

  Only one line changes: where you compute the hidden layer, use
  `h = F.gelu(x @ W1 + b1)` instead of `torch.tanh(...)`. Train both and print the two dev
  losses side by side.
  </details>
- ✅ Solution: [`solutions/e03_gelu.py`](./solutions/e03_gelu.py).

### E04 — Watch a model overfit
Train on just a **tiny slice** of the data (say 200 names) for many steps. The **train**
loss will dive toward 0 while the **dev** loss *rises*. That gap is **overfitting** —
memorizing instead of generalizing.

- **Why it matters:** it makes "why we hold out a dev set" visceral. A model that aces the
  training data but flunks new data has learned nothing useful.

  <details><summary>💡 Hint</summary>

  Build the dataset from a small slice (`words[:200]`). Then **drop the minibatch** — the set
  is tiny, so train on the *whole* thing each step: `loss = F.cross_entropy(forward(Xtr), Ytr)`
  with no `randint`. Print the dev loss every so often and watch it pull away from train.
  </details>
- ✅ Solution: [`solutions/e04_overfit.py`](./solutions/e04_overfit.py).

### E05 — A deeper MLP *(stretch)*
Add a **second hidden layer** (so: embed → linear → tanh → linear → tanh → linear →
logits). Does the extra depth lower the dev loss?

- **Reflection:** more layers ≠ automatically better — depth needs the right size, init,
  and training (the stuff of Chapters 5 and 7). See what you find.

  <details><summary>💡 Hint</summary>

  Add `W2b, b2b` for the new hidden layer between the existing hidden and the output, and
  insert another `h = torch.tanh(h @ W2b + b2b)` step. Remember to add the new tensors to
  your `params` list so they get trained.
  </details>
- ✅ Solution: [`solutions/e05_deeper.py`](./solutions/e05_deeper.py).
