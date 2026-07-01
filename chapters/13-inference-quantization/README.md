# Chapter 13 — Inference II: Quantization

> Chapter 12 made generation *fast*; this makes the model *small*. A trained model is a pile of
> floating-point numbers, and floats are expensive: a 7-billion-parameter model is 14 GB in fp16 —
> too big for most GPUs, slow to load, slow to move. **Quantization** stores those weights as small
> **integers** instead — int8 (1 byte) halves the model, int4 (½ byte) quarters it — for almost no
> loss in quality. The whole idea is one linear map: pick a **scale**, divide, round to an integer,
> and multiply back. We build it from scratch, watch the error stay tiny at int8, discover why
> **per-channel** scaling matters (the outlier problem behind LLM.int8/AWQ), quantize a real model,
> and place **GPTQ** and **AWQ** as the production refinements.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/13-inference-quantization/code/explore.ipynb)

> 💻 **No GPU needed.** Quantization is arithmetic on weight tensors — it all runs on a CPU, and the
> size savings are exact. (On a real GPU, the smaller weights are also *faster* to load and move,
> which is much of the speedup.)

**You will be able to:**
- explain why weights are the expensive part of a model, and how quantization shrinks them;
- implement **symmetric int8 / int4** quantize-dequantize from the map `q = round(w/scale)`, and say
  what the **scale** (and **zero-point**) are for;
- explain why **per-channel** scaling crushes far less than **per-tensor** — the **outlier** problem;
- quantize a whole model's **Linear** weights and measure the **logit drift** vs fp32 (int8 nearly
  lossless, int4 more);
- reason about the **size / speed / quality** trade-off, **PTQ vs QAT**, and place **GPTQ**, **AWQ**,
  and **LLM.int8()** as the methods that make int4 work.

**Prerequisites:** a trained model's weights are `nn.Linear` matrices (Chapter 5); floating-point
formats and bytes-per-number (Chapter 9); and the inference framing (Chapter 12).

**Time:** ~2 hours. **Hardware:** any laptop CPU.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)**: implement quantize/dequantize, compare int8 vs
> int4 error, see per-channel beat per-tensor on outliers, then quantize a GPT and watch the logit
> drift and the memory drop. "✍️ Your turn", "▶️ Run this", "▶️ Check your work" cells. Watch for:
> ✏️ **In the notebook → Step N**.

---

## 0. Setup

Two scripts, both CPU, both self-contained:

- [`code/quantize.py`](./code/quantize.py) — the core map, int8 vs int4, per-tensor vs per-channel, and
  the memory table.
- [`code/quantize_model.py`](./code/quantize_model.py) — quantize a compact GPT ([`code/gpt.py`](./code/gpt.py))
  and measure the output drift.

```bash
python chapters/13-inference-quantization/code/quantize.py
python chapters/13-inference-quantization/code/quantize_model.py
```

---

## A 2-minute primer: weights are the expensive part

A trained Transformer *is* its weights — hundreds of millions to hundreds of billions of numbers,
learned and then frozen. At inference time those numbers don't change; they just get read, over and
over, for every token. So their **storage format** is a pure cost you can optimize:

- **Memory.** fp16 is 2 bytes per weight. A 7B model is 14 GB — it won't fit on a 12 GB GPU at all.
- **Bandwidth.** Chapter 8's lesson: generation is often *memory-bound* — the bottleneck is reading
  the weights from memory, not the math. Smaller weights = less to read = faster.

Quantization attacks both by storing each weight in **fewer bits** — as a small integer instead of a
float. int8 (1 byte) halves everything; int4 (½ byte) quarters it. The trick is doing that without
wrecking the model, and it turns out you can get *remarkably* close to lossless.

---

## 1. The core: the quantization map

> ✏️ **In the notebook → Step 1.** Implement it.

Quantization is a linear map from a range of floats to a small set of integers. Symmetric int8 maps
onto `-127 … 127` (int8 actually has 256 codes, `-128 … 127`; the *symmetric* scheme centres on 0 and
uses `±127`, and we clamp to the full `[-128, 127]` for safety). To map a weight tensor into that
range:

1. Find a **scale** — how much float each integer step is worth: `scale = max(|w|) / 127`.
2. **Quantize**: divide by the scale and round to the nearest integer, `q = round(w / scale)`
   (clamped to the int range). These integers are what you *store* — 1 byte each.
3. **Dequantize** to use the weight: `w_hat = q × scale` — an approximation of the original.

Real output from [`quantize.py`](./code/quantize.py) on a tiny vector:

```
weights   : [0.1, -0.42, 0.98, -0.05, 0.61]
scale     : 0.00772   (= max|w| / 127)
integers  : [13, -54, 127, -6, 79]     <- what's actually stored (1 byte each)
recovered : [0.1003, -0.4167, 0.98, -0.0463, 0.6096]   (≈ the originals)
```

Picture it as snapping every weight onto a grid of evenly-spaced points, `scale` apart:

```
  float weights:   -0.42        -0.05   0.1        0.61     0.98
                     │            │      │           │        │
  int8 grid:   ──•───┼──•──•──•───┼──•───┼──•──•──•──┼──•──•──┼──►   (256 points, spacing = scale)
   (integer)      -54           -6      13          79      127
```

The largest-magnitude weight (`0.98`) maps to the extreme integer `127`; everything else is a
fraction of that, snapped to the nearest grid point. Dequantizing recovers values within a
half-grid-step of the originals. That
step — the **quantization error** — is the whole story of quality, and it's set by two things: how
many bits (§2), and how the scale is chosen (§3).

Why does throwing away precision like this barely hurt? Because a trained network is **robust to
small weight noise** — millions of weights share the work, and no single one needs to be exact.
Quantization essentially adds a little rounding noise to every weight, and the model mostly shrugs it
off (the same robustness that let bf16 work in Chapter 9). That tolerance is what makes the whole
trick possible.

> 💡 **Zero-point (asymmetric quantization).** Symmetric quantization assumes the values are roughly
> centered on 0 (weights usually are). When they're *not* — like ReLU activations, which are all ≥ 0
> — you waste half your integers on negatives. **Asymmetric** quantization adds a **zero-point**: an
> integer offset so the range `[min, max]` maps onto the full `[0, 255]`. Same idea, one extra
> number. Weights → symmetric; skewed activations → asymmetric.

---

## 2. int8 vs int4: fewer bits, more error

> ✏️ **In the notebook → Step 2.**

The bit count is the number of integer levels you get: int8 gives `2⁸ = 256` levels, int4 only
`2⁴ = 16`. Fewer levels means a coarser grid, so each weight rounds to something further away.
Concretely, int4's grid is **16× coarser** than int8's: if weights span `[−1, 1]`, int8's points sit
~`0.008` apart but int4's are ~`0.14` apart — so two distinct weights like `0.30` and `0.34` collapse
to the *same* int4 value. On a plain random weight matrix (per-tensor), the relative error jumps as
you drop bits:

```
  int8 :  0.0094     (≈ 1% — you'd struggle to tell the model apart from fp16)
  int4 :  0.1684     (≈ 17% — real error; needs the smarter methods of §7 to stay usable)
```

That gap is why **int8 is nearly free** — it's the default "just make it smaller" setting — while
**int4 is aggressive**: it halves the model *again* but the error is large enough that naive rounding
noticeably hurts, which is exactly the problem GPTQ and AWQ (§7) solve.

---

## 3. Per-tensor vs per-channel: the outlier problem

> ✏️ **In the notebook → Step 3.**

Here's the subtlety that makes real quantization work. If you use **one scale for the whole tensor**
(per-tensor), that scale is set by the single largest-magnitude weight. LLM weight matrices have a
few **outliers** — channels with values 10–20× the rest — so one huge scale gets chosen, and every
*ordinary* weight becomes a tiny fraction of it that rounds to almost the same few integers. The
outliers hog all the precision; everything else is crushed.

Concretely: if one outlier weight is `20` and the rest cluster around `0.5`, per-tensor picks
`scale = 20/127 ≈ 0.16`. Now weights of `0.45`, `0.50`, and `0.55` all become `round(w/0.16) = 3` —
three different weights collapsed to one dequantized value (`0.47`). The outlier ate the precision
budget.

The fix is **per-channel** scaling: give each *row* (output channel) its own scale. Now an outlier
row gets its own big scale and the normal rows get small, tight scales — nobody is crushed. On a
matrix with a couple of outlier rows, the difference is stark ([`quantize.py`](./code/quantize.py)):

```
              per-tensor   per-channel
  int8            0.0443       0.0063        <- 7× less error
  int4            0.3133       0.1198        <- 2.6× less error
```

Per-channel is almost always worth its tiny cost (one extra scale per row). This same insight —
*outliers need to be handled separately* — is the seed of **LLM.int8()** (keep the outlier
dimensions in fp16, quantize the rest) and **AWQ** (§7). (In real models the troublesome outliers are
usually in the *activations* — or the weights that get multiplied by large activations — rather than
in the weight rows of our demo, but it's the same idea, just aimed at where the big values actually
live.)

---

## 4. Quantizing a whole model

> ✏️ **In the notebook → Step 4.**

Real quantization targets a model's **Linear weights** — the big matmuls, which are the vast bulk of
the parameters (in our GPT, **97%**). We quantize each `nn.Linear`'s weight per-channel, leave the
activations in float (the standard "weight-only" recipe for LLM inference), run the model, and measure
how far the output **logits drift** from the fp32 model — the typical logit change relative to the
logits' own spread (a relative RMS, so `0.0042` is a ~0.4% wobble). Real output from
[`quantize_model.py`](./code/quantize_model.py):

```
  format       size   logit drift
    fp16    1.24 MB   0.0000 (ref)
    int8    0.62 MB       0.0042        <- half the size, output barely moves (0.4%)
    int4    0.31 MB       0.0744        <- quarter the size, a few % drift
```

There it is, the whole trade-off in three rows. **int8 halves the model and the output is
essentially unchanged** — a 0.4% drift is imperceptible in the generated text. **int4 quarters it**
for a few percent of drift — visible, sometimes fine, and the target of the sharpening methods in §7.
(The `size` column is the idealized "all weights at this format"; since 97% are Linear, that's what
you actually get, give or take the un-quantized embeddings.)

A note on *what we're doing here*: we **simulate** quantization by replacing each weight with its
quantized-then-dequantized value — still an fp32 tensor — so we can measure the output error on a CPU
in pure PyTorch. Real int8 storage keeps the *packed integers* (and uses int8 matmul kernels), which
is where the actual memory drop and speedup come from; our demo is about the *quality and size math*,
not the bytes on disk. That's also why the model's RAM doesn't literally halve when you run this — the
`size` column is the analytic size, not a measured one.

What does a "0.4%" or "7%" logit drift *mean* for the text? At int8, the model picks the same next
token almost every time — you couldn't tell its stories from the fp16 original. At naive int4 it
occasionally picks a *different* token, so the writing stays fluent but wanders a bit further from the
fp16 version — which is exactly why int4 is paired with GPTQ/AWQ (§7), to pull that drift back down.

---

## 5. The trade-off, and when to reach for each

Quantization is a three-way trade — **size**, **speed**, **quality** — and the bit width is the dial:

- **int8** — the safe default. ~2× smaller, faster (less to move), quality drop usually
  unmeasurable. If you just want a model to fit and run, quantize to int8 and move on.
- **int4** — aggressive. ~4× smaller, lets a 34B model run on a single 24 GB GPU (or a 70B on a
  48 GB card) — but naive int4
  hurts enough that you use **GPTQ or AWQ** (§7) to claw the quality back. This is how most big open
  models are shipped for local use.
- **Below int4** (int3, int2, 1-bit) — active research; the quality cost climbs steeply.

The quality "cost" is also task-dependent: quantization nibbles at the model's sharpest, least-robust
behaviors first, so easy generation survives int4 comfortably while precise reasoning or long-context
recall degrades sooner.

In practice this is a whole ecosystem. You download a model as a **GGUF** or **GPTQ/AWQ** file — the
weights pre-quantized to 4, 5, or 8 bits with per-group scales — and a runtime (llama.cpp, vLLM)
loads it. A 70B model that needs 140 GB in fp16 ships as a ~40 GB 4-bit file that runs on one
high-end GPU, or slowly on a laptop's CPU. Quantization is *the* reason you can run serious models
locally at all.

---

## 6. PTQ vs QAT

There are two moments you can quantize:

- **Post-Training Quantization (PTQ)** — take an already-trained fp16 model and quantize it, no
  retraining. Fast (minutes), needs at most a little **calibration** data to pick good scales. This
  is what we did, and what GPTQ/AWQ are — it's how almost all deployed LLMs are quantized.
  (**Calibration** = running a few representative samples through the model to observe the value
  ranges, so you can choose good scales. It's needed for *activation* quantization and for GPTQ/AWQ —
  but *not* for our weight-only method, whose `max|w|` scale reads the weights directly.)
- **Quantization-Aware Training (QAT)** — simulate the quantization *during* training, so the model
  learns weights that survive it. Higher quality at very low bits, but you have to (re)train, which
  for a large model is enormous. Reserved for when PTQ isn't good enough.

For LLMs, PTQ dominates because retraining a giant model just to quantize it rarely pays off.

---

## 7. GPTQ, AWQ, and making int4 work

Naive round-to-nearest int4 (§2) loses too much. The production one-shot methods are cleverer, and
all attack the same target: **minimize the error in the layer's *output* `Wx`, not just in the
weights `W`.** A weight that gets multiplied by a large activation matters more, so protect it.

- **GPTQ** — quantize a layer's weights **one column at a time**, and after rounding each column,
  *nudge the not-yet-quantized columns* to compensate for the error just introduced (using
  second-order/Hessian information — the *curvature* of the layer's error — from a little calibration
  data). The rounding errors partly cancel
  instead of accumulating, so int4 GPTQ lands close to fp16 — this is the workhorse for shipping 4-bit
  models. (A faithful implementation is beyond our from-scratch scope; the *idea* is error-compensated
  rounding.)
- **AWQ (Activation-aware Weight Quantization)** — notice that only a small fraction of weight
  channels are "salient" (they see large activations). **Scale those channels up before quantizing**
  (and scale the matching activations down to compensate), so the important weights get more of the
  precision budget. Cheap, calibration-light, and excellent at int4.
- **LLM.int8()** — for int8 specifically: keep the handful of **outlier** activation dimensions in
  fp16 and quantize the rest to int8, so the outliers (§3) can't wreck the quantization. Made int8
  reliable for the largest models.

They're all the same lesson as §3, pushed harder: *precision is a budget — spend it where it matters
(the outliers and the salient channels), not uniformly.*

---

## 🔌 How this plugs into the Storyteller

Your trained Storyteller is a stack of `nn.Linear` weights. Quantizing them to int8 halves the file
and speeds up loading and generation, with text you can't tell from the fp16 original — and int4
(with GPTQ/AWQ) shrinks it enough to run a much bigger Storyteller on modest hardware. Together with
Chapter 12's KV-cache, this is the deployment toolkit: **the KV-cache makes each token cheap, and
quantization makes the whole model cheap to hold** — which is exactly what the mini-project measures.

---

## 🐛 Building it yourself: what trips people up

- **Per-tensor on outliers.** One global scale set by an outlier crushes every normal weight (§3).
  Default to **per-channel** — it's a one-line change and a big accuracy win.
- **Quantizing everything, including LayerNorm/embeddings blindly.** The safe, standard target is the
  **Linear weights**. Quantizing small, sensitive tensors (norms) or the embedding table can hurt more
  than it saves.
- **Forgetting to clamp.** After `round(w/scale)` you must **clamp** to the integer range, or a stray
  value overflows int8's ±127 and wraps to garbage.
- **Measuring weight error, not output error.** What matters is how the *model's output* drifts, not
  how the weights drift — a weight can move a lot with little output effect (and vice versa). That's
  the whole premise of GPTQ/AWQ (§7).
- **Expecting int4 to be free.** int8 is nearly lossless; int4 is a real trade you manage with
  per-channel + GPTQ/AWQ, not a free lunch.

---

## 🤔 Common questions

- **Does quantization change the model's answers?** A little — it's an approximation, unlike the
  KV-cache which was exact. At int8 the drift is imperceptible; at int4 it's small but real, and the
  §7 methods minimize it.
- **Why quantize weights but not activations?** Weights are fixed and read constantly (so shrinking
  them is a pure win), and they're better-behaved (fewer wild outliers) than activations. "Weight-only"
  int8/int4 with float activations is the standard LLM-inference recipe; activation quantization is
  harder and used more selectively.
- **What actually gets stored?** The integers (1 byte for int8, packed to ½ byte for int4) plus the
  scales (one per channel — a tiny overhead). At run time you dequantize (or use special int8/int4
  matmul kernels) to compute.
- **int8 or int4?** int8 if you can afford the size — it's basically free quality-wise. int4 (with
  GPTQ/AWQ) when you need to fit a big model on small hardware.
- **Is this really how local LLMs run?** Yes — the GGUF/GPTQ/AWQ files you download to run a model on
  a laptop are exactly this: the weights quantized to 4–8 bits, with per-channel (or per-group) scales.
- **Do the scales cost much memory?** No. Per-channel is one scale per row — for a `4096×4096` weight
  that's 4,096 extra floats against 16M weights, under 0.1%. Per-*group* (a scale per, say, 128
  weights) is finer still and standard in 4-bit formats, for a slightly larger but still tiny overhead.
- **Why not just train in int4 from the start?** You can (that's QAT, §6), but training needs the fine
  gradients that low precision destroys — the mixed-precision story of Chapter 9. It's far cheaper to
  train in bf16 and quantize *after*, which is why PTQ dominates.

## ✅ Check your understanding

<details>
<summary>1. Write the three steps of symmetric int8 quantize-dequantize.</summary>

`scale = max(|w|) / 127`; `q = clamp(round(w / scale), -128, 127)` (store the integers `q`);
`w_hat = q × scale` (dequantize to use). The error is the gap between `w` and `w_hat`.
</details>

<details>
<summary>2. Why does int4 have so much more error than int8?</summary>

int8 has `2⁸ = 256` integer levels, int4 only `2⁴ = 16` — a far coarser grid, so each weight rounds to
something further away. Fewer bits = fewer levels = bigger rounding error.
</details>

<details>
<summary>3. Why is per-channel scaling so much more accurate than per-tensor?</summary>

Per-tensor uses one scale set by the largest (outlier) weight, which crushes every ordinary weight
into a few integers. Per-channel gives each row its own scale, so outlier rows and normal rows are
each quantized on an appropriate grid — nobody is crushed.
</details>

<details>
<summary>4. What do we quantize in a model, and what do we measure to judge the damage?</summary>

We quantize the **Linear weights** (the bulk of the parameters), weight-only, leaving activations in
float. We judge it by how far the **output logits drift** from the fp32 model — output error, not
weight error.
</details>

<details>
<summary>5. In one line each, what do GPTQ and AWQ do?</summary>

**GPTQ**: quantize columns one at a time, compensating the remaining weights for each rounding error
(minimizing output error). **AWQ**: scale up the "salient" weight channels (those with large
activations) before quantizing so they keep more precision.
</details>

## 🎓 Key takeaways

- A model *is* its weights; quantization stores them as **integers** — int8 halves the model, int4
  quarters it — via the map `q = round(w/scale)`, `w_hat = q·scale`.
- **Fewer bits = more error** (int8 ≈ 1%, int4 ≈ 17% per-tensor); **per-channel** scaling beats
  per-tensor by a lot because it handles **outliers** (the seed of LLM.int8/AWQ).
- Quantizing a whole model's Linear weights: **int8 barely drifts the output (~0.4%) at half the
  size; int4 drifts a few % at quarter the size** — the size/speed/quality trade.
- Almost all LLM quantization is **PTQ** (no retraining); **GPTQ** (error-compensated rounding) and
  **AWQ** (protect salient channels) are what make **int4** land close to fp16.
- With Chapter 12's KV-cache, this is the deployment toolkit: fast tokens *and* a small, cheap model.

## 📖 New vocabulary

`quantization` · `int8` / `int4` · `scale` · `zero-point` · `symmetric` vs `asymmetric` ·
`quantize` / `dequantize` · `per-tensor` vs `per-channel` · `outlier` · `weight-only quantization` ·
`logit drift` · `PTQ` vs `QAT` · `calibration` · `GPTQ` · `AWQ` · `LLM.int8()`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/13-inference-quantization/code/explore.ipynb))
   — implement quantize/dequantize, compare int8/int4, see per-channel win, quantize a GPT. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): implement the quant map, show per-channel beats
   per-tensor on outliers, compare int8/int4 error & memory, and quantize a model to measure the
   drift. Tiered hints + solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Quantize Your Storyteller"** — quantize a GPT's
   Linear weights to int8 and int4, and measure the three-way trade-off: model size, and output drift
   vs the fp32 model.

## 🔗 Go deeper (optional)

- 📄 [Dettmers et al. (2022), *LLM.int8()*](https://arxiv.org/abs/2208.07339) — the outlier problem
  and how to keep int8 lossless at scale.
- 📄 [Frantar et al. (2022), *GPTQ*](https://arxiv.org/abs/2210.17323) — accurate one-shot 4-bit
  quantization via error compensation.
- 📄 [Lin et al. (2023), *AWQ*](https://arxiv.org/abs/2306.00978) — activation-aware weight
  quantization; protect the salient channels.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 12 — KV-cache](../12-inference-kv-cache/) | [Syllabus](../../README.md#-syllabus) | [Chapter 14 — Finetuning: SFT](../14-finetuning-sft/) |
