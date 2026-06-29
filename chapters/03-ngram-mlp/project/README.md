# 🔥 Mini-Project — The Name Forge MK II

In Chapter 1 you built **The Name Forge**, a bigram name generator. Time for the upgrade:
the same idea, now powered by the **MLP** you just built — three characters of context and
learned embeddings. The names come out noticeably better, and you keep the fun knobs.

> **How this works:** you finish the model in
> [`starter/name_forge2.py`](./starter/name_forge2.py) — the data, vocab, parameters, and
> command-line interface are wired up; you write the **training loop** and the **sampling**
> (the two things that are really "the model"). A full reference is in
> [`solution/name_forge2.py`](./solution/name_forge2.py).

## 🎯 What it does

```bash
python starter/name_forge2.py --n 10                 # 10 fresh names
python starter/name_forge2.py --n 8 --start k        # names that begin with 'k'
python starter/name_forge2.py --temperature 0.6      # play it safe (more common names)
python starter/name_forge2.py --temperature 1.5      # go wild
```

It trains a small MLP on the names (a few seconds), then samples — with the same
**`--start`** and **`--temperature`** controls as the Chapter 1 forge.

## 🛠️ Your TODOs

Open `starter/name_forge2.py`. Two pieces:

1. **The training loop** — the forward → backward → update you now know by heart: minibatch
   loss with `cross_entropy`, `loss.backward()`, then nudge every param down its gradient
   (the learning rate is provided).
2. **The sampling step** — given the next-character probabilities, **sample** one with
   `torch.multinomial`, then **slide** the context window. (Temperature and the `--start`
   handling are already wired up around your two lines.)

## ✅ Checking your work

- Training should print a falling loss and end well under the bigram's 2.45.
- The names should look better than Chapter 1's (e.g. `kaeli`, `jahmer`, `nellara`).
- `--start k` should always begin with `k`; lower `--temperature` → safer/blander names.
- Compare against `solution/name_forge2.py` if anything looks off.

## 🚀 Extensions (open-ended)

- **Save your roster.** Add `--save names.txt` to append the generated names to a file —
  your story's cast.
- **Bigger context.** Bump `block_size` to 4 or 5 and retrain — do the names improve?
- **Pick the best.** Generate 100 names and keep only the ones the model thinks are most
  likely (lowest per-name loss).
- **A web button.** Skip ahead in spirit to Chapter 16: wrap the trained model in a tiny
  [FastAPI](https://fastapi.tiangolo.com/) endpoint and forge names from a webpage.

When your MK II forge is humming, you've trained and deployed a real neural language model.
On to [Chapter 4 — Attention](../../04-attention/), where context goes from *three*
characters to *all* of them. 🧠
