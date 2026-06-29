# Chapter 1 — Exercise Solutions

Worked, runnable solutions. **Peek only after you've had a real go yourself.**

| Script | Covers | What it shows |
|--------|--------|---------------|
| [`e01_trigram.py`](./e01_trigram.py) | E01 (+E06) | A counting **trigram** model: builds a 27×27×27 table, reports the loss (~**2.09**, clearly below the bigram's 2.45), and samples names. |
| [`e02_03_split_and_smoothing.py`](./e02_03_split_and_smoothing.py) | E02, E03 | An 80/10/10 **train/dev/test** split, honest held-out loss, and **tuning the smoothing** on dev before reporting test. |
| [`e04_05_nn_refinements.py`](./e04_05_nn_refinements.py) | E04, E05 | That `one_hot @ W == W[xs]` (the **embedding** trick), and that `F.cross_entropy` matches the manual loss — then trains with it. |

Run any of them from the repo root, e.g.:

```bash
python chapters/01-bigram/exercises/solutions/e01_trigram.py
python chapters/01-bigram/exercises/solutions/e02_03_split_and_smoothing.py
python chapters/01-bigram/exercises/solutions/e04_05_nn_refinements.py
```

> Exact numbers may vary by a hair across machines/PyTorch versions, but the story
> won't: trigram beats bigram, test loss ≳ train loss, and the NN matches counting.
