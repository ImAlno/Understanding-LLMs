# 🔥 Mini-Project — The Name Forge

Time to make something *yours*. You'll build **The Name Forge**: a little
command-line tool that conjures brand-new names for your story's characters, on
demand, with knobs you can turn. It's the bigram model from the lesson, wrapped in
a real (tiny) program.

> **How this works:** you build it yourself from a scaffold. The file
> [`starter/name_forge.py`](./starter/name_forge.py) has the boring parts done
> (loading data, the command-line interface) and **four `TODO` functions for you to
> fill in**. A complete, runnable reference lives in
> [`solution/name_forge.py`](./solution/name_forge.py) — use it to check yourself
> *after* you've tried, not before.

## 🎯 What it should do

When finished, your tool runs like this:

```bash
# 10 names from the forge
python starter/name_forge.py

# 5 names that start with 'k'
python starter/name_forge.py --n 5 --start k

# crank the "creativity" up (more surprising) or down (safer)
python starter/name_forge.py --temperature 1.6
python starter/name_forge.py --temperature 0.5
```

Example output (yours will differ — that's the point!):

```
🔥 The Name Forge — 5 names (temp=1.0, smoothing=1.0)
   kayle
   kor
   kimi
   khalee
   ka

model loss (NLL) = 2.4544
```

## 🛠️ Your four TODOs

Open `starter/name_forge.py`. Each `TODO` has hints in the comments.

1. **`build_probs(words, stoi, V, smoothing)`** — count the bigrams into a `V×V`
   tensor, add `smoothing`, and normalize each row so it sums to 1. *Return `P`.*
   (You did exactly this in the lesson — reuse it.)

2. **`apply_temperature(p, temperature)`** — given a probability vector `p`, return
   a re-weighted one. **Temperature** controls randomness: raise each probability
   to the power `1/temperature` and renormalize.
   - `temperature = 1.0` → unchanged.
   - `< 1.0` → sharper (the model plays it safe, picks common letters).
   - `> 1.0` → flatter (wilder, more surprising names).
   - *Why it works:* the power `1/T` stretches or squashes the gaps between
     probabilities. For `T < 1` the power is `> 1`, so the big probabilities grow
     *relatively* bigger (sharper, safer); for `T > 1` the power is `< 1`, so they
     even out (flatter, wilder). Renormalizing keeps it a valid distribution.
   - *Hint:* `q = p ** (1.0 / temperature); return q / q.sum()`
   - *Aside:* real LLMs usually apply temperature to the **logits** (before softmax);
     doing it to the final probabilities here is a simpler equivalent for our purpose.

3. **`sample_name(P, stoi, itos, start, temperature, g)`** — generate one name. If
   `start` is given, the name must begin with that letter (set your current index
   to `stoi[start]` and emit it); otherwise begin at the `.` token (index 0). Then
   loop: apply temperature to the current row, `torch.multinomial` the next index,
   stop at `.`. *Return the string.*

4. **`model_loss(words, P, stoi)`** — the average negative log-likelihood over all
   bigrams (straight from the lesson). *Return a float.*

Run `python starter/name_forge.py` as you go — it will tell you which TODO is next.

## ✅ Checking your work

- Names should look *name-ish* (start/end believably, mostly pronounceable).
- `--start k` should always produce names beginning with `k`.
- Lower `--temperature` → blander, more repetitive; higher → weirder.
- Your `model_loss` should print ≈ **2.45** at `smoothing=1.0`.
- Compare behavior against `solution/name_forge.py` if anything seems off.

## 🚀 Extensions (optional, open-ended — stretch yourself)

No full solutions for these — they're yours to explore. Ideas:

- **Trigram Forge.** Use two characters of context (see exercise E01). The names
  get noticeably better.
- **Themed forges.** Make a `data/` file of, say, elf names or villain names and
  train on it with `--data myfile.txt`. Same code, totally different vibe — your
  first taste of how *data* shapes a model.
- **"Sounds like" seeding.** Let the user pass `--contains` and keep sampling until
  a name contains that substring.
- **Save your favorites.** Add `--save names.txt` to append the generated names to
  a file — your story's character roster.
- **A web button.** Skip ahead in spirit to Chapter 16: wrap `sample_name` in a
  tiny [FastAPI](https://fastapi.tiangolo.com/) endpoint and hit it from a webpage.

When you're happy with your Forge, you've truly understood Chapter 1. On to
[Chapter 2 — Micrograd](../../02-micrograd/), where we open up `loss.backward()`
and build backpropagation ourselves. 🧠
