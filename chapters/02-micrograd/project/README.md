# 🔥 Mini-Project — Curve Forge

You built an autograd engine and trained it on four points. Now make it do something you
can *see*: train a small network to trace a **wiggly curve**, and watch the fit improve.

> **How this works:** you finish the training loop yourself in
> [`starter/curve_forge.py`](./starter/curve_forge.py) (the engine, the data, and the
> plot are wired up — you write the 3 lines that *are* learning). A full reference is in
> [`solution/curve_forge.py`](./solution/curve_forge.py) — peek only after you try.

## 🎯 What it does

It fits a network `f(x)` to a target curve `0.8 · sin(3x)` over `x ∈ [−2, 2]`, then saves
a picture of the network's curve laid over the true one:

```bash
python starter/curve_forge.py
# → trains, prints the loss dropping, and saves curve_fit.png
```

The network is `MLP(1, [16, 16, 1])` — one input `x`, two hidden layers, one output —
about 320 `Value` parameters, all tuned by *your* backprop.

## 🛠️ Your TODOs (the training loop)

Open `starter/curve_forge.py`. Inside the loop, fill in:

1. **The loss** — mean-squared error over all points: the *average* of `(pred − y)²`.
   ⚠️ This is the **mean** — divide by `N`. (The train demo used a plain *sum*; if you
   copy that here, your loss will read ~25× too big and look broken when it isn't.)
2. **Backprop** — the one call that fills every parameter's `.grad` (you know this one).
3. **The update** — nudge every parameter a small step *against* its gradient (the
   learning rate `lr` is already set for you).

These are the same three moves as Chapter 1 and the train demo — now on a real curve.

## ✅ Checking your work

- The printed loss should fall steadily (from ~0.3 toward ~0.01 or lower).
- `curve_fit.png` should show the dashed "network" curve hugging the solid "target" sine.
- Compare against `solution/curve_forge.py` if the fit looks off.

> ⏱️ Heads-up: micrograd is pure-Python and scalar, so this takes a few seconds to a
> minute — that slowness is *exactly* why Chapter 3 moves to tensors.

## 🚀 Extensions (open-ended)

- **Harder curves.** Change `target(x)` to something wilder (`sin(5x)`, a sum of waves).
  Does it still fit? Do you need a bigger network or more steps?
- **Kill the nonlinearity.** Make every neuron linear (return `act` instead of
  `act.tanh()` in `nn.py`). Watch the network collapse to a straight line — proof, in a
  picture, of *why* §7 said nonlinearity matters.
- **Learning-rate hunt.** Try `lr = 0.01`, `0.3`, `1.0`. Find where it learns fastest and
  where it explodes.
- **Animate it.** Save a frame every few steps and stitch them into a gif of the curve
  snapping into shape.

When your network traces the curve, you've used a from-scratch autograd engine to train a
real (if tiny) model. On to [Chapter 3 — the N-gram MLP](../../03-ngram-mlp/), where we
trade scalars for tensors and start generating language again. 🧠
