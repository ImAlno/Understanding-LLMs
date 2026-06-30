# Chapter 9 — Need for Speed II: Precision (fp16, bf16, fp8)

> Same model, half the bits, roughly **twice the speed and half the memory** — for almost no loss
> in quality. The trick is **mixed precision**: do the heavy matmuls in a fast 16-bit number format
> while keeping a full-precision copy of the weights where it matters. This chapter is about *what
> those formats are*, *why fewer bits is faster*, the two ways low precision bites (range and
> precision), and the small tricks — **loss scaling** and an **fp32 master copy** — that make it
> all just work. Then we flip it on with PyTorch's **autocast** in three lines.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/09-speed-precision/code/explore.ipynb)

> 💻 **No GPU? No problem.** The *formats* — what each can and can't represent — are fully visible
> on a CPU, and that's the heart of this chapter. The **speedup** itself needs an NVIDIA GPU's
> **Tensor Cores**, so click the badge to open the notebook in **Google Colab** (*Runtime → Change
> runtime type → GPU*) to feel it. Everything still *runs* on a CPU or Apple GPU — you just won't
> see the wall-clock win.

**You will be able to:**
- read a floating-point format as **sign · exponent · mantissa**, and say what the exponent
  (range) and mantissa (precision) each buy;
- compare **fp32, fp16, bf16** (and **fp8**) — and explain why bf16, not fp16, is the modern default;
- name the two failure modes of low precision — **overflow/underflow** (range) and **swamping**
  (precision) — and recognize them in code;
- explain why mixed precision keeps an **fp32 master copy** of the weights and (for fp16) uses
  **loss scaling**;
- turn on mixed precision with **`torch.autocast`** + **`GradScaler`**, and measure the speedup,
  the memory savings, and the (un)changed loss curve.

**Prerequisites:** the training loop and optimizer (Chapter 7), and Chapter 8's **device** model
(`get_device()`, `.to(device)`, the async/`synchronize` timing rule, VRAM). No new math — just a
careful look at how a number is stored in bits.

**Time:** ~2 hours. **Hardware:** a CPU is enough to *see the formats*; a free Colab GPU to *feel
the speedup*.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** (the Colab notebook above): inspect the formats,
> watch fp16 overflow and bf16 swamp, implement **loss scaling**, wrap a forward pass in
> **autocast**, then run a mixed-precision training loop and compare. "✍️ Your turn", "▶️ Run this",
> "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

Two reference scripts, both runnable anywhere (no GPU required to *run* them):

- [`code/formats.py`](./code/formats.py) — the formats up close: bit budgets, what numbers round
  to, the overflow/underflow/swamping demos, and the memory math. **All CPU-visible** — this is the
  conceptual core.
- [`code/mixed_precision_train.py`](./code/mixed_precision_train.py) — a device-agnostic training
  loop in fp32 vs mixed precision, comparing time and the loss curve.

```bash
.venv/bin/python chapters/09-speed-precision/code/formats.py
.venv/bin/jupyter lab        # open chapters/09-speed-precision/code/explore.ipynb
```

---

## A 2-minute primer: what a floating-point number *is*

Chapter 8 made the GPU fast. This chapter makes each *number* cheaper. To see how, we have to open
up how a decimal like `3.14159` is actually stored.

A floating-point number is **binary scientific notation**. In school you write `6.022 × 10²³` —
a **sign**, some **significant digits** (`6.022`), and an **exponent** (`23`). A computer does the
same in binary, splitting a fixed budget of bits into three parts:

```
   sign │   exponent    │        mantissa
    ±   │  how big (2^?) │  the significant bits
```

- The **sign** is one bit: positive or negative.
- The **exponent** sets the **range** — how enormous or how tiny the number can be (it's the power
  of 2 the value is scaled by). More exponent bits → wider range.
- The **mantissa** (or "significand") sets the **precision** — how many significant bits you keep,
  i.e. how *finely* you can distinguish nearby numbers. More mantissa bits → finer steps.

That's the whole trade-off of this chapter in one sentence: **a number's bits are split between
range (exponent) and precision (mantissa), and when you shrink the budget you have to sacrifice
one, the other, or both.** Everything below is choosing *which* sacrifice you can afford.

---

## 1. The formats: fp32, fp16, bf16

> ✏️ **In the notebook → Step 1.** Inspect the formats yourself.

The default everywhere so far has been **fp32** — 32 bits, split `1 + 8 + 23`. It's accurate and
roomy, but every number costs 4 bytes to store and move. The two 16-bit formats halve that to
2 bytes — and they split the *same* 16 bits in two very different ways:

| format | bits (S·E·M) | bytes | max value | smallest normal | mantissa (≈precision) |
|--------|:------------:|:-----:|----------:|----------------:|:---------------------:|
| **fp32** | 1 · 8 · 23 | 4 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 23 bits (~7 digits) |
| **fp16** (half) | 1 · 5 · 10 | 2 | **6.55 × 10⁴** | 6.1 × 10⁻⁵ | 10 bits (~3 digits) |
| **bf16** (bfloat16) | 1 · 8 · 7 | 2 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 7 bits (~2 digits) |

(These are the numbers `formats.py` prints from `torch.finfo`, rounded to a couple of significant
figures.) Read the two 16-bit rows carefully, because the whole chapter lives in the difference:

- **fp16** spends its bits on the **mantissa** (10 bits — decent precision) but starves the
  **exponent** (5 bits), so its range is *tiny*: the biggest number it can hold is **65 504**.
  Go higher and it **overflows to infinity**.
- **bf16** does the opposite: it keeps fp32's full **8-bit exponent** — so its range is identical
  to fp32, up to ~10³⁸ — and pays for it with a **coarse 7-bit mantissa**. It almost never
  overflows, but it rounds aggressively.

The name gives it away: **bf16 = "brain float," literally fp32 with the bottom 16 mantissa bits
chopped off.** Same exponent, half the significant bits. That single design choice — *keep the
range, sacrifice precision* — is why bf16 won, as we'll see in §7.

Store `3.14159265` in each and watch the precision evaporate (real output):

```
fp32: 3.1415927410...     (≈7 good digits)
fp16: 3.140625            (≈3 good digits)
bf16: 3.140625            (≈2-3 good digits — even coarser steps)
```

Both 16-bit formats land on `3.140625` here — that's the nearest value each can represent. The
point isn't the exact digits; it's that **half the bits buys you roughly half the significant
figures.** For training, that turns out to be plenty — *if* you avoid the two traps in §2.

---

## 2. The two ways low precision bites: range and precision

> ✏️ **In the notebook → Step 2.** Make fp16 overflow and bf16 swamp.

Two distinct things go wrong when numbers get small budgets. Knowing which is which is the whole
skill.

**(a) Range — overflow and underflow (fp16's weakness).** fp16's max is 65 504. Store something
bigger and it doesn't clamp; it becomes **infinity**:

```
store 70000:   fp32 → 70000.0     fp16 → inf  ← overflowed!     bf16 → 70144.0
```

The mirror problem is **underflow**: a number too *small* rounds to zero. A gradient like `1e-8` —
entirely ordinary deep in a network — vanishes in fp16:

```
store 1e-8:    fp32 → 1.0e-8      fp16 → 0.0  ← gradient lost!  bf16 → 1.0e-8
```

(fp16 doesn't snap straight to 0 at the `6.1 × 10⁻⁵` "smallest normal" in the table above — below
that it keeps going, at *degraded* precision, through so-called **subnormal** values down to about
`6 × 10⁻⁸`, and only *then* hits 0. So `1e-8` is genuinely lost, but a gradient like `1e-7` or
`1e-5` still survives — which is exactly why the *moderate* loss scale of §5 is enough to rescue
them.)

A gradient that underflows to `0` is a weight that stops learning — a silent, training-wrecking
bug. bf16 sails through both cases because it kept fp32's exponent (its range). This is exactly why
fp16 needs the **loss scaling** of §5 and bf16 does not.

**(b) Precision — swamping (bf16's weakness).** With only 7 mantissa bits, bf16 can't tell apart
numbers that are close *relative to their size*. Add a small number to a big one and the small one
falls off the bottom and is simply **lost**:

```
compute 256 + 0.5:   fp32 → 256.5     fp16 → 256.5     bf16 → 256.0  ← the 0.5 was swamped!
```

At magnitude 256, bf16's representable values are spaced **2 apart** — its 7 mantissa bits split
each power-of-2 interval into `2⁷ = 128` even steps, and the interval from 256 to 512 spans 256, so
each step is `256 / 128 = 2`. So `256.5` rounds straight back to `256` — as if you never added
anything. fp16,
with its finer mantissa, keeps the `0.5`. This is why the *accumulation* of many small updates —
like summing a million gradients, or a weight update of `1e-4` onto a weight of `2.0` — is the part
of training you **don't** want to do in 16-bit. Which brings us to the master copy (§4).

> 💡 **The one-line summary.** *fp16 has fine steps but a short ruler (overflow/underflow). bf16 has
> a long ruler but coarse steps (swamping).* fp32 has both and costs double. Mixed precision uses
> the cheap formats for the bulk work and fp32 for the few places the coarse one would hurt.

---

## 3. Why fewer bits is faster

A 16-bit number is **half the bytes** of a 32-bit one, and that pays off twice:

- **Half the memory.** Weights, gradients, optimizer state, activations — all roughly halve. A
  model that needed 16 GB to train (Chapter 8's estimate) drops toward ~8 GB, so you fit a bigger
  model or a bigger batch in the same VRAM.
- **Half the data to move.** Recall Chapter 8's punchline: for many operations the bottleneck isn't
  the math, it's **memory bandwidth** — how fast the GPU feeds numbers to its cores. Halve the bytes
  per number and you halve that traffic, which directly speeds up the memory-bound parts.

But the biggest win is special hardware. Modern NVIDIA GPUs have **Tensor Cores** — units built to
multiply 16-bit matrices several times faster than they do 32-bit ones. A bf16/fp16 matmul on a
Tensor Core can run **2–8× faster** than the fp32 version of the same matmul. Since a Transformer is
mostly matmuls, that flows straight through to training time.

That last speedup is **GPU-specific**: a CPU or an Apple GPU has no Tensor Cores, so
`mixed_precision_train.py` will show ~1× there (it still runs, and the loss still tracks — you just
need a CUDA GPU, i.e. Colab, to see the wall-clock win). Same lesson as Chapter 8: the *ideas* are
visible everywhere; the *speedup* needs the right hardware.

---

## 4. The fp32 master copy: where full precision still matters

Here's the tension. We want the speed of 16-bit, but §2(b) just showed that *accumulating small
updates* in 16-bit loses them to swamping. And the optimizer update is exactly that: each step nudges
a weight by a tiny amount (`weight -= lr * grad`, often a change of `1e-4` or smaller). In bf16, that
nudge onto a sizable weight can round away to nothing — the weight would never move, and training
would stall. (Concretely: `2.0 + 1e-4` in bf16 → `2.0` — the update vanished, because bf16's steps
near 2.0 are about `0.016` apart, far coarser than a `1e-4` nudge.)

The fix is the cornerstone of mixed precision: **keep a full-precision (fp32) master copy of the
weights.** The flow each step:

1. Make a 16-bit copy of the weights for the **forward and backward** passes (the fast matmuls).
2. Compute 16-bit gradients.
3. **Apply the optimizer update to the fp32 master weights** — so the tiny nudges accumulate at full
   precision and nothing is swamped.
4. Next step, cast the updated fp32 weights down to 16-bit again.

So "mixed" precision is literally *mixed*: the heavy, bandwidth-bound matmuls run in 16-bit for
speed, while the delicate running total of the weights stays in fp32 for correctness. You get most
of the speed and almost none of the numerical pain.

A clarification that matters once you use `autocast` (§6): there's no *second* stored copy of the
weights. They simply **stay in fp32** — that resident fp32 copy *is* the master — and autocast casts
them down to 16-bit on the fly *inside each matmul*, then discards the 16-bit version. (Steps 1 and 4
above are that transient cast, not a kept tensor.) So the "master copy" costs no extra memory; it's
just the weights, left in full precision while the heavy ops borrow a cheap 16-bit view of them.

---

## 5. Loss scaling: rescuing fp16's tiny gradients

> ✏️ **In the notebook → Step 3.** Implement loss scaling.

§2(a) showed fp16 underflowing a `1e-8` gradient to `0`. Many real gradients live in that danger
zone — too small for fp16's short ruler. **Loss scaling** is the cure, and it's beautifully simple:
**multiply the loss by a big constant before `backward()`, then divide it back out before the
optimizer step.**

Why it works: the chain rule is linear, so scaling the loss by `S` scales *every* gradient by `S`
too. Pick `S` (say 1024) big enough to lift the tiny gradients up into fp16's representable range —
*before* they get rounded — then unscale by the same `S` in fp32 afterwards, leaving the real update
unchanged. You can see it on one number (real output from `formats.py`):

```
fp16(1e-8)              = 0.0          ← underflowed, gradient lost
fp16(1e-8 * 1024) / 1024 = 1.0e-8      ← scaled up, survived fp16, unscaled back: recovered!
```

That's the entire idea. In code it's three touches around the backward pass:

```python
scaled_loss = loss * scale       # (1) scale UP, so small grads land in fp16's range
scaled_loss.backward()           #     gradients are now S× too big, but representable
# ... then, before opt.step():
for p in params:
    p.grad /= scale              # (2) unscale in fp32, recovering the true gradients
opt.step()                       # (3) normal update on the fp32 master weights
```

Real AMP adds one more safety valve: if a gradient *overflows* anyway (becomes `inf`/`nan`), it
**skips that step** and **shrinks `S`** for next time; if many steps pass cleanly, it grows `S` back.
That's "dynamic" loss scaling, and `GradScaler` (§6) does it for you.

**The headline:** loss scaling is an **fp16-only** chore. **bf16 keeps fp32's range**, so its tiny
gradients never underflow, so it needs **no scaling at all** — one of the big reasons bf16 is simpler
to train with (§7).

---

## 6. Automatic Mixed Precision: `autocast` + `GradScaler`

> ✏️ **In the notebook → Steps 4–5.** Wrap a forward pass, then run the full loop.

You almost never wire up the master copy and loss scaling by hand. PyTorch's **AMP** (Automatic
Mixed Precision) does it with two tools:

- **`torch.autocast(device_type, dtype)`** — a context manager. Inside it, PyTorch automatically
  runs each operation in the *right* precision: matmuls in fast 16-bit, but numerically sensitive
  ops — sums, softmax, normalizations, the loss (all *reductions*, exposed to the swamping of
  §2b) — stay in fp32. You don't pick per-op; autocast knows the safe choices.
- **`torch.amp.GradScaler`** — does the dynamic loss scaling of §5 around the backward pass. Needed
  for **fp16**; for **bf16** you create it with `enabled=False` (a no-op) because bf16 needs no
  scaling.

The whole training step becomes:

```python
scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

for xb, yb in loader:
    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=dtype):   # fast 16-bit forward
        loss = loss_fn(model(xb), yb)
    scaler.scale(loss).backward()    # scale up (fp16) → backward; no-op for bf16
    scaler.step(opt)                 # unscale, then step (skips the step if grads overflowed)
    scaler.update()                  # adjust the scale factor for next time
```

Compared to your Chapter 5/7 loop, that's **three changed lines**: wrap the forward in `autocast`,
and route `backward`/`step` through the `scaler`. Everything else — the model, the optimizer, the
data — is untouched. That's the payoff of AMP: near-fp32 quality at near-16-bit speed, for a
three-line diff.

`mixed_precision_train.py` runs this loop in fp32 and in mixed precision from the **same
initialization**, and prints the two loss curves side by side (real MPS output):

```
 step    fp32 loss   mixed loss
    0       2.2530       2.2527
    7       1.8975       1.8976
   15       1.3925       1.3924
   22       0.8201       0.8201
   29       0.3033       0.3032
```

The curves track to 3–4 decimals — **mixed precision barely moves the loss.** On this Apple GPU the
speedup is ~1× (no Tensor Cores); on a Colab T4 the same code trains markedly faster while these
loss numbers stay essentially identical. That combination — *same curve, less time and memory* — is
the entire reason mixed precision is standard practice.

---

## 7. bf16 vs fp16: which to use

For years fp16 was the only fast 16-bit option, and the loss-scaling machinery of §5 grew up around
its narrow range. Then hardware added **bf16**, and the calculus changed. Side by side:

| | **fp16** | **bf16** |
|---|---|---|
| Range | tiny (max 65 504) | full (≈fp32, ~10³⁸) |
| Precision | finer (10 mantissa bits) | coarser (7 mantissa bits) |
| Loss scaling | **required** (or grads underflow) | **not needed** |
| Overflow risk | real (must watch activations) | virtually none |
| Verdict | legacy / memory-tightest | **the modern default** |

The trade is **precision for range**, and training turns out to care far more about *range* than
about the last couple of mantissa bits: gradients span many orders of magnitude (where bf16's range
saves you), but each individual number rarely needs 10 bits of precision (where fp16's finer mantissa
would help). So bf16 trains more robustly *and* without the loss-scaling bookkeeping — strictly
simpler. The catch is hardware support: bf16 needs a reasonably modern GPU (NVIDIA Ampere/2020+,
TPUs, recent Apple silicon). Where it's available — which is most places that matter — **reach for
bf16 first**; fp16 is for older GPUs or when you're squeezing the absolute most out of memory.

---

## 8. fp8 and the frontier (a peek)

The logic doesn't stop at 16 bits. The newest training hardware (NVIDIA Hopper/Blackwell) supports
**fp8** — *eight* bits per number, in two flavours that, just like fp16-vs-bf16, trade range against
precision:

- **e4m3** — 1 sign · 4 exponent · 3 mantissa, max ≈ 448 (more precision, less range).
- **e5m2** — 1 sign · 5 exponent · 2 mantissa, max ≈ 57 344 (more range, less precision).

With only 2–3 mantissa bits, fp8 is *far* too coarse to use everywhere, so frontier training uses it
surgically — fp8 for the bulk matmuls, bf16/fp32 for everything sensitive — with **per-tensor scaling
factors** to keep each tensor inside fp8's sliver of range. It's the same idea you've now learned,
pushed to its limit: spend the fewest bits you can get away with on each piece of the computation.
You won't write fp8 in this course, but the principle — *precision is a resource you budget, not a
constant* — is exactly what the rest of the speed arc is about.

---

## 🔌 How this plugs into the Storyteller

Nothing about your Chapter 5 GPT changes structurally. You wrap the forward pass in `autocast`, route
`backward`/`step` through a `GradScaler`, and the **same** model trains ~2× faster in ~half the VRAM
on a real GPU — with a loss curve you can overlay on the fp32 one. Combined with Chapter 8's device
move, this is what makes the *next* chapters (bigger models, more data, longer training) actually
affordable: precision is free real estate. The mini-project adds exactly this to your GPT and
measures all three — speed, memory, and the unchanged loss.

---

## 🐛 Building it yourself: what trips people up

The formats are subtle and the API has a few sharp edges:

- **Casting the whole model with `.half()` instead of `autocast`.** `model.half()` makes *everything*
  fp16 — including the weight updates, which then get swamped (§4). Training goes unstable. Use
  `autocast` (which keeps the fp32 master copy and sensitive ops in fp32), not a blanket cast.
- **fp16 without a `GradScaler`.** Gradients underflow to 0, and the loss mysteriously stalls or goes
  `NaN`. With fp16 you *need* the scaler; with bf16 you don't (create it `enabled=False`).
- **Timing without `synchronize()`.** Same trap as Chapter 8 — GPU calls are async, so an un-synced
  timer makes mixed precision look infinitely fast (or no faster). Sync before you stop the clock.
- **Expecting a speedup on a CPU or Apple GPU.** No Tensor Cores → no matmul speedup. The code runs
  and the loss matches; the wall-clock win needs a CUDA GPU.
- **Doing manual reductions in 16-bit outside `autocast`.** A hand-rolled `sum()` over many values in
  bf16 will swamp the small contributions. Let autocast keep reductions in fp32, or accumulate in
  fp32 yourself.
- **Reading `inf`/`nan` as a model bug.** Early in fp16 training an overflow is often just the loss
  scale being too high — the `GradScaler` is *designed* to catch it, skip the step, and back off.

---

## 🤔 Common questions

- **Does mixed precision make my model worse?** Almost never measurably. The forward/backward run in
  16-bit but the *weights* are updated in fp32 (§4) and sensitive ops stay fp32 (autocast), so the
  loss curve tracks fp32 closely (§6). The tiny numerical differences wash out.
- **fp16 or bf16 — which do I pick?** **bf16** if your GPU supports it (Ampere/2020+, TPUs, recent
  Apple) — it needs no loss scaling and rarely overflows. fp16 only for older GPUs or extreme memory
  pressure. (§7.)
- **Why keep the weights in fp32 — isn't that the memory we're trying to save?** With `autocast`
  the weights simply *are* fp32 — there's no second copy. What goes 16-bit is the transient cast
  inside matmuls and the *activations* (which, with the batch, dominate training memory). Keeping
  the weights themselves in fp32 is what stops tiny updates from being swamped (§4), at no real
  memory cost.
- **What does `autocast` actually decide?** Per operation: matmuls → 16-bit (fast, safe);
  reductions, softmax, normalization, the loss → fp32 (precision-sensitive). You don't choose
  per-op; it uses a vetted list.
- **Why didn't I see a speedup on my laptop?** No **Tensor Cores**. The bf16 *math* runs, and the
  loss matches, but the wall-clock win needs an NVIDIA GPU (Colab's free T4). Same hardware caveat as
  Chapter 8's GPU speedup.
- **Is this really how big models are trained?** Yes — bf16 (or fp16) mixed precision via AMP is
  standard for essentially every modern LLM; the frontier adds fp8 for the matmuls (§8). The
  `autocast` you just learned is the production API.

---

## ✅ Check your understanding

<details>
<summary>1. A format's bits are split into sign, exponent, and mantissa. Which sets the range, and which the precision?</summary>

The **exponent** sets the **range** (how big/small a number can be — the power of 2 it scales by);
the **mantissa** sets the **precision** (how many significant bits, i.e. how finely you can tell
nearby numbers apart). The sign is just ±.
</details>

<details>
<summary>2. fp16 and bf16 are both 16 bits. How do they differ, and which would <i>overflow</i> storing 70000?</summary>

fp16 is `1·5·10` (small exponent → tiny range, max 65 504; finer mantissa). bf16 is `1·8·7` (fp32's
exponent → full range; coarse mantissa). **fp16 overflows** 70000 to `inf`; bf16 stores it fine
(~70144) because it kept fp32's range.
</details>

<details>
<summary>3. What is loss scaling, and why does fp16 need it while bf16 doesn't?</summary>

Multiply the loss by a large `S` before `backward()` (so tiny gradients land in fp16's representable
range instead of underflowing to 0), then divide the gradients by `S` before the optimizer step.
**fp16** needs it because its range is tiny; **bf16** keeps fp32's range, so its gradients don't
underflow — no scaling needed.
</details>

<details>
<summary>4. Why does mixed precision keep an fp32 "master copy" of the weights?</summary>

The optimizer update is a *tiny* nudge (`lr * grad`) added to each weight. In 16-bit, such a small
addition onto a larger weight can be **swamped** (rounded away), so the weight would never move.
Accumulating the updates in an fp32 master copy keeps them from being lost.
</details>

<details>
<summary>5. In the AMP loop, what do <code>autocast</code> and <code>GradScaler</code> each do?</summary>

`autocast` runs each op in the right dtype automatically (matmuls in 16-bit, sensitive ops in fp32).
`GradScaler` does dynamic loss scaling around the backward pass (scale up → backward → unscale →
step, skipping steps whose gradients overflowed). The scaler is a no-op for bf16.
</details>

## 🎓 Key takeaways

- A float is **sign · exponent · mantissa** — exponent = **range**, mantissa = **precision**. A
  smaller bit budget forces a sacrifice.
- **fp16** = fine precision, **tiny range** (overflow/underflow). **bf16** = full range, **coarse
  precision** (swamping). **bf16 is the modern default** — it needs no loss scaling.
- Fewer bits → **half the memory** and **half the bandwidth**, and 16-bit matmuls run **2–8× faster**
  on **Tensor Cores** (a CUDA-GPU win).
- Mixed precision keeps an **fp32 master copy** of the weights (so tiny updates aren't swamped) and,
  for fp16, uses **loss scaling** (so tiny gradients don't underflow).
- **AMP** turns it on in ~3 lines: `torch.autocast` around the forward, `GradScaler` around
  `backward`/`step`. Same loss curve, less time and memory.

## 📖 New vocabulary

`floating point` · `sign / exponent / mantissa` · `fp32` · `fp16 (half)` · `bf16 (bfloat16)` ·
`fp8 (e4m3/e5m2)` · `range` vs `precision` · `overflow` / `underflow` · `swamping` · `Tensor Cores` ·
`memory bandwidth` · `mixed precision` · `fp32 master weights` · `loss scaling` (static/dynamic) ·
`autocast` · `GradScaler` · `AMP`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/09-speed-precision/code/explore.ipynb))
   — inspect the formats, trigger overflow/swamping, implement loss scaling, wrap a forward in
   autocast, and run the mixed-precision loop. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): measure the memory savings, find fp16's overflow
   cliff (and bf16 surviving it), rescue an underflowing gradient with loss scaling, and watch which
   dtype `autocast` picks per op. Tiered hints + solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Mixed-Precision Your GPT"** — add `autocast` +
   `GradScaler` to your Chapter 5 GPT, and measure the speedup, the memory drop, and that the loss
   curve is unchanged. (Run it on Colab for the real numbers.)

## 🔗 Go deeper (optional)

- 📄 [Micikevicius et al. (2017), *Mixed Precision Training*](https://arxiv.org/abs/1710.03740) — the
  paper that introduced fp16 training with loss scaling and an fp32 master copy.
- 📄 [PyTorch — *What Every User Should Know About Mixed Precision*](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
  — the practical guide to `autocast`/`GradScaler` and bf16 vs fp16.
- 📄 [Google — *bfloat16: The secret to high performance on Cloud TPUs*](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
  — why "keep the range, drop the precision" works for deep learning.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 8 — Device](../08-speed-device/) | [Syllabus](../../README.md#-syllabus) | [Chapter 10 — Distributed](../10-speed-distributed/) |
