# 🏁 Mini-Project — The Optimizer Showdown

You implemented SGD, Momentum, and Adam. Now **race them on one chart** — the *shape* of each
loss curve tells you, at a glance, why Adam is the default.

> **How this works:** [`starter/showdown.py`](./starter/showdown.py) has one TODO — train each
> optimizer and record its full loss **curve** — then the plotting is done for you. A full
> reference is in [`solution/showdown.py`](./solution/showdown.py).

## 🎯 What it does

```bash
python starter/showdown.py
```

It trains the same network on two-moons with each optimizer, recording the loss at *every* step,
and plots all three curves on a log scale, saving `showdown.png` to your current directory:

```
  SGD       start 0.705 -> final 0.0110
  Momentum  start 0.705 -> final 0.0037
  Adam      start 0.705 -> final 0.0001
```

## 🛠️ Your TODO

Build `curves` — a dict mapping each optimizer's name to its full list of per-step losses.
`train_with(opt, steps=300)` (from `optimizers.py`) trains and returns that list.

## ✅ Checking your work

- Three curves, all starting at ~0.69 and descending.
- **Adam's curve plunges fastest and settles lowest**; SGD's is the slowest, with visible
  wiggle; Momentum sits between them.
- Compare against `solution/showdown.py`.

## 🚀 Extensions (open-ended)

- **Learning-rate sweep.** Plot Adam at several LRs on one chart — see the crawling, the sweet
  spot, and the divergence from Exercise 2, as *curves*.
- **Add AdamW.** Include `AdamW(weight_decay=...)` and see how decay changes the late curve.
- **Add the schedule.** Overlay a warmup+cosine run (Exercise 4) and compare its shape.
- **Harder task.** Make the two moons noisier or closer together — the optimizer gap widens as
  the problem gets harder (exactly why it matters for real transformers).

You now understand the machinery that turns a model architecture into a *trained* model. Next:
[Chapter 8 — Need for Speed I: Device](../../08-speed-device/), where we make all of this run on
a GPU. ⚡
