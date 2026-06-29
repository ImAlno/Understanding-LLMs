# 🔬 Mini-Project — The Attention Microscope

You built an attention head; now look *through* it. Train the model, then feed it a prompt
and **watch which earlier characters it leans on** when predicting the next one. Attention
stops being an abstraction the moment you can see it pointing at things.

> **How this works:** you finish two pieces in
> [`starter/attention_microscope.py`](./starter/attention_microscope.py) — the **training
> loop** and the line that **pulls out the attention weights** — then the provided display
> draws them as bars. A full reference is in
> [`solution/attention_microscope.py`](./solution/attention_microscope.py).

## 🎯 What it does

```bash
python starter/attention_microscope.py
```

It trains a single-head model on Shakespeare (~10s), then for each example prompt prints the
attention weights of the **last** character as a bar chart:

```
prompt: "To be, or not to b"  (predicting the next character)
  ...
  't'    0.002
  'o'    0.001
  ' '    0.014 █
  'b'    0.971 ██████████████████████████████████████████████████████████
```

— literally *which characters the model is looking at* to guess what comes next. (And you'll
discover something: see below.)

> 🔬 **What you'll find — and why it's the perfect setup for Chapter 5.** The single head puts
> almost all its weight on the **current** character. On its own, one head mostly learns a
> *smarter bigram* — it beats the plain bigram only thanks to the small slice of context it
> does use plus the position embedding. **Rich, long-range attention is exactly what stacking
> many heads and layers (the Transformer, Chapter 5) buys you.** Run the multi-head extension
> and watch heads begin to specialize.

## 🛠️ Your TODOs

1. **The training loop** — the reset → backward → update you know, with `AdamW`.
2. **Read out the attention** — the head stashes its weights in `model.head.last_wei` (shape
   `(B, T, T)`) on each forward pass; grab the row for the **last** position (what the final
   character attends to) and hand it to the display.

## ✅ Checking your work

- Training should print a falling loss (well under the ~2.5 bigram).
- Each prompt should print one bar per character, weights summing to ~1.
- The biggest bar will be the **current** (last) character — that's the discovery, not a bug.
  The thin weights on earlier characters are the bit of context it adds on top.
- Compare against `solution/attention_microscope.py`.

## 🚀 Extensions (open-ended)

- **Multi-head microscope.** Use the multi-head model from exercise E05 and show each head's
  attention separately — different heads often specialize.
- **Any position.** Show the attention of a *middle* character, not just the last.
- **Heatmap.** Plot the full `(T, T)` weight matrix with matplotlib (see exercise E02).
- **Bigger model.** Train longer / wider and see whether the attention sharpens.

Seeing attention point at things is the best intuition you can have walking into
[Chapter 5 — the Transformer](../../05-transformer/), where we stack many of these heads into
a real GPT. 🧠
