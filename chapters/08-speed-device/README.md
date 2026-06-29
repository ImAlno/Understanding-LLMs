# Chapter 8 — Need for Speed I: Device (CPU, GPU, CUDA)

> Your models have run on a laptop CPU this whole course. Real training runs on a **GPU** — often
> tens to hundreds of times faster. This chapter is about *where the compute happens*: the
> `device` abstraction, how to move a model and its data onto a GPU, the speedup (and its
> surprising crossover), and what **CUDA** actually is.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/08-speed-device/code/explore.ipynb)

> 💻 **No GPU? No problem.** Everything here runs on a CPU (you just won't see the speedup), on
> an **Apple Silicon GPU** (MPS), or on a free **NVIDIA GPU** — click the badge above to open
> this chapter's notebook in **Google Colab**, set *Runtime → Change runtime type → GPU*, and run
> it. That's the recommended way to feel the GPU speedup and meet real CUDA.

**You will be able to:**
- explain *why* a GPU is fast (massive parallelism) and when it *isn't* (small work);
- use the **`device`** abstraction — `get_device()`, `.to(device)`, `torch.cuda.is_available()`;
- move a model and its data to a GPU and train there (the **device-agnostic** pattern);
- time GPU code honestly (the **async / `synchronize`** gotcha) and avoid **transfer overhead**;
- say what **CUDA**, a **kernel**, and **VRAM** are.

**Prerequisites:** the training loop and tensors. No GPU required to *read* or to run the
CPU/Apple-GPU paths; a free Colab GPU to feel the CUDA speedup.

**Time:** ~2 hours.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** (the Colab notebook above): detect your device,
> move tensors and a model onto it, hit the same-device rule, benchmark CPU vs GPU, and train on
> the device. "✍️ Your turn", "▶️ Run this", "▶️ Check your work" cells. Watch for: ✏️ **Step N**.

---

## 0. Setup

Reference code: [`code/benchmark.py`](./code/benchmark.py) (the CPU-vs-GPU speedup) and
[`code/device_train.py`](./code/device_train.py) (the device-agnostic training loop). Both run
anywhere; on a GPU they light up. The *ideas* (device, transfer, sync, VRAM) are identical on every
machine — only the speedup needs a GPU to actually *see*, which is what Colab is for.

---

## A 2-minute primer: why a GPU?

A neural network's heavy lifting is **matrix multiplication**, and a matmul is *embarrassingly
parallel* — each output number is an independent dot product, millions of them, all computable at
once. Two very different chips:

- A **CPU** has a handful of big, fast cores (great for sequential logic).
- A **GPU** has *thousands* of small, slower cores (great for doing the same simple thing to a lot
  of data at once).

An analogy: the CPU is a few math professors — each brilliant and fast, able to follow complicated
sequential reasoning. The GPU is a stadium of ten thousand schoolchildren who can each only multiply
two numbers. Ask for one hard, step-by-step calculation and the professors win easily. Ask for a
million *independent* multiplications — exactly what a matmul is — and the stadium finishes in the
time it takes one professor to clear their throat. Neural nets are almost entirely the second kind
of work, which is why they live on GPUs.

None of this makes the CPU obsolete — it's the right tool for sequential logic, small data, and
orchestrating the whole program (including telling the GPU what to do). The GPU is a *co-processor*
you hand the parallel-heavy parts to, not a replacement.

|              | CPU                          | GPU                              |
|--------------|------------------------------|----------------------------------|
| **Cores**    | a handful (4–32), big & fast | thousands, small & slow          |
| **Best at**  | sequential logic, branching  | the *same* op on lots of data    |
| **A matmul** | a few cores, mostly in turn  | thousands of dot products at once |
| **Analogy**  | a few professors             | a stadium of schoolchildren      |

For a big matmul the GPU's thousands of cores crush the CPU's handful. **CUDA** is NVIDIA's system
for programming those cores; on Colab you get an NVIDIA GPU and CUDA for free.

---

## 1. The `device`

> ✏️ **In the notebook → Steps 1–2.** Detect your device and move tensors to it.

In PyTorch, every tensor lives on a **device** — `"cpu"`, `"cuda"` (an NVIDIA GPU), or `"mps"`
(an Apple Silicon GPU). You ask what's available and pick the best one:

```python
def get_device():
    if torch.cuda.is_available():            # NVIDIA GPU (Colab, real training)
        return "cuda"
    if torch.backends.mps.is_available():    # Apple Silicon GPU
        return "mps"
    return "cpu"                             # always works
```

(The order matters: we check `cuda` first because an NVIDIA GPU is the fastest option and what real
training uses; `mps` — Apple's GPU — next; and `cpu` last as the always-works fallback. `get_device`
just returns the best thing present.)

To put a tensor or a model on a device, use **`.to(device)`**. One subtle gotcha: for a **tensor**
you must *reassign* (`x = x.to(device)`) — `x.to(device)` on its own returns a moved copy and is
silently lost — but a **model** moves *in place*, so `model.to(device)` alone is enough:

```python
device = get_device()
x = torch.randn(1000, 1000).to(device)       # or: torch.randn(1000, 1000, device=device)
model = MyModel().to(device)
```

That's the whole abstraction. Writing `device = get_device()` once and `.to(device)` on your data
and model is the **device-agnostic pattern**: the same code runs on a CPU, an Apple GPU, or a CUDA
GPU on Colab — you change *nothing else*.

Under the hood, `.to(device)` **copies** the tensor's numbers into that device's memory — a real
data transfer if it crosses the CPU↔GPU boundary (see §5). That's *why* the tensor case returns a
new tensor and the original stays put. Re-running `.to(device)` on something already there is a cheap
no-op, so calling it defensively never hurts.

---

## 2. The one rule: same device

> ✏️ **In the notebook → Step 3.** Hit the rule, then fix it.

Operations need all their tensors on the **same** device. Mix them and PyTorch stops you:

```python
a = torch.randn(3, device="cpu")
b = torch.randn(3, device=device)   # on the GPU
a + b
# RuntimeError: Expected all tensors to be on the same device ...
```

This is the #1 beginner GPU error: the model is on the GPU but a batch of data is still on the
CPU. The fix is always the same — move the stray tensor: `a.to(device) + b`. In a real training
loop (like your Chapter 5 GPT, which loops over batches) that means moving **each batch** to the
device before the forward pass:

```python
model = model.to(device)                 # once
for xb, yb in loader:                     # every step:
    xb, yb = xb.to(device), yb.to(device) # move THIS batch onto the device
    loss = loss_fn(model(xb), yb)         # ...now model and data are on the same device
```

The mental model: think of `cpu` and `cuda` as two separate workshops. An operation can only combine
tensors that are *in the same workshop* — there's no reaching across. `.to(...)` ships a tensor to
the other workshop. So "all tensors on the same device" really means "do the work where the data is,
and bring any stragglers over first." Once the model lives on the GPU, every batch has to make the
trip too — which is the loop above.

---

## 3. The speedup — and how it depends on the GPU

> ✏️ **In the notebook → Step 4.** Benchmark it.

Time an `n×n` matmul on the CPU and the GPU as `n` grows. On this **Apple laptop (MPS)**:

```
 size    CPU ms    GPU ms  speedup
  256      0.03      0.11    0.2x      <- CPU wins! the laptop GPU's overhead isn't worth it
 1024      1.22      1.11    1.1x      <- about even (the crossover, here)
 4096     74.71     46.94    1.6x      <- GPU wins, but only ~1.6x
```

And on Colab's **datacenter GPU (a T4)** — the same code, much stronger hardware (real numbers
from a reader's run):

```
 size    CPU ms    GPU ms  speedup
  256      0.41      0.06    7.2x      <- the T4 already wins here — no CPU-winning crossover
 1024     26.47      0.84   31.7x
 4096   1053.13     39.70   26.5x
```

(These are *two different machines*, so their CPU columns differ — Colab's CPU is much slower than
this laptop's. The `speedup` is computed from the un-rounded times, so it won't exactly match
dividing the rounded `ms`. And the small dip from 31.7× to 26.5× is just measurement noise — the
trend is what matters.)

Two lessons. **(1)** The GPU's advantage **grows with the size of the work** — every GPU call has
a fixed launch overhead, so for tiny matmuls the speedup is small. On a *weak* GPU that overhead
can make the CPU actually win below a **crossover size** (the laptop's `0.2×` at 256); a *strong*
GPU is fast enough to win even there, so its crossover sits below this whole range. **(2)** How
big the speedup gets depends on the GPU: a laptop's integrated GPU tops out around **~2×**, while
the **T4 hits ~30×** — datacenter GPUs have far more cores and **memory bandwidth** (how fast they
can read and write their own memory). The more powerful the GPU, the smaller the work where it
starts paying off, and the larger the eventual speedup.

To feel the *scale* of parallelism: a `4096×4096` matmul computes `4096² ≈ 16.8 million` output
numbers, each a 4096-long dot product — about **69 billion** multiply-adds. The CPU's ~8 cores grind
through them mostly in sequence; the T4's ~2,560 cores chew thousands at once. That's the whole story
of the ~26×, and why the gap *widens* as the matrices grow: more independent work is exactly what a
GPU is built to swallow.

When you run the project on your own hardware, the *shape* of the curve is the lesson: speedup starts
below 1 (CPU wins — overhead dominates), crosses 1 at your **crossover size**, and climbs as the work
grows. Where it crosses and how high it climbs is a fingerprint of your GPU — a weak laptop GPU
crosses late and plateaus near 2×; a datacenter card crosses almost immediately and soars past 25×.

One honest caveat: these are *pure matmul* numbers. Real training also runs activations, data
loading, and the occasional CPU-bound step, so end-to-end speedups are usually a bit smaller than the
raw matmul figure — but still the difference between hours and weeks for a large model.

A subtlety worth knowing: for many workloads the bottleneck isn't the cores at all — it's **memory
bandwidth**, how fast the GPU can read its own VRAM to feed those cores. A matmul reuses data heavily,
so it's *compute*-bound; but many ops (activations, normalizations) touch each number once and move
on, so they're *memory*-bound. That's why "more cores" doesn't always mean "proportionally faster,"
and why the rest of the speed arc obsesses over *moving less data*, not just doing less math.

---

## 4. Timing GPU code honestly: `synchronize`

> ✏️ **In the notebook → Step 4.** This is *why* the benchmark is correct.

A subtle trap: **GPU calls are asynchronous.** When you write `c = a @ b`, PyTorch *queues* the
work on the GPU and returns immediately — before it's done. So this lies:

```python
t0 = time.perf_counter()
c = a @ b                    # returns instantly; the GPU is still working
dt = time.perf_counter() - t0    # measures launch time, not compute time — way too fast
```

You must **wait for the GPU to finish** before stopping the clock:

```python
torch.cuda.synchronize()     # (or torch.mps.synchronize() on Apple) — block until the GPU is done
```

Forgetting this is the most common GPU-benchmarking mistake — it makes the GPU look impossibly
fast. (You don't normally call `synchronize` in training; it's for *timing*.)

What the bug looks like in numbers: time a big matmul *without* `synchronize` and you might read
`0.05 ms` — faster than physically possible — because you only timed how long it took to *queue* the
work. Add `synchronize()` and the same matmul honestly reports, say, `40 ms`. That first number isn't
"the GPU is amazing," it's "the clock stopped before the work had started."

Why is the GPU asynchronous at all? So the CPU can keep *queuing* the next operations while the GPU
chews through the current ones — the two run in parallel instead of the CPU sitting idle. It's a
feature, not a quirk; it only becomes a trap when you *time* something and forget the work is still
in flight.

---

## 5. The cost of moving data

A GPU has its **own memory**, separate from your CPU's RAM. Moving data across the
CPU↔GPU boundary (`.to(device)`, `.cpu()`, `.item()`, `print(tensor)`) is **slow** relative to
compute. Two habits follow:

- **Keep data on the GPU.** Move a batch over once, do all the work there. Don't bounce tensors
  back and forth. (A classic slow pattern: compute on the GPU, pull the result to the CPU to do one
  small thing, push it back — three boundary crossings where zero were needed.)
- **Don't `.item()` / `.cpu()` in the hot loop** (the per-step inner training loop). Pulling the
  loss back to the CPU *every step* to print it forces a sync and a transfer; do it every N steps.

How slow is "slow"? Moving a tensor across the bus between CPU and GPU runs at maybe ~10 GB/s, while
the GPU *computes* at teraflops. So copying a 100 MB batch costs ~10 ms — about the same as a sizable
matmul. Do that once per step and it's invisible; do it several times per step (a stray `.item()`, a
`print`, bouncing data back and forth) and transfer can dominate your runtime even though "the GPU
is fast."

A surprising amount of "my GPU training is slow" is really "I'm copying data across the boundary
too often."

---

## 6. Memory: VRAM

A GPU's memory (**VRAM**) holds the model's parameters, the optimizer state, *and* the
activations saved for backprop — and it's limited (Colab's T4 has 16 GB; a laptop GPU often less).
Run out and you get the dreaded **`CUDA out of memory`**.

A quick way to estimate: each parameter is **4 bytes** (fp32), so a 1-billion-parameter model needs
~4 GB just for weights — then roughly *another* 4 GB for gradients, ~8 GB for AdamW's two optimizer
states (`m` and `v` from Chapter 7), plus the activations saved for backprop. That's why a "1B model"
can need ~16–20 GB to *train* but only ~2–4 GB to *run* (inference needs no gradients, no optimizer
state, no saved activations). Chapter 9's half-precision roughly halves every one of those.

The knobs that use less: a **smaller
batch size**, a **smaller model**, and tricks from later chapters (mixed precision in
[Chapter 9](../09-speed-precision/), which roughly halves it). For this course's models, a free
Colab GPU has memory to spare.

Picture two separate pools: your computer's **RAM** (where data is loaded) and the GPU's **VRAM**
(where the model trains) — the same boundary that makes transfers slow in §5. When you hit `CUDA out
of memory`, the usual fixes, in order: halve the batch size (cheapest knob, usually enough), shrink
the model, or reach for Chapter 9's mixed precision. The error often appears a few steps *in* rather
than immediately — because it's the first **backward** pass that allocates all the saved-activation
memory. The full message reads something like `CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0;
15.78 GiB total capacity; 14.9 GiB already allocated; ...)` — it tells you how much it *wanted* and
how much was *free*, which is your clue for how much to trim.

---

## 7. What CUDA actually is

**CUDA** is NVIDIA's platform for running code on the GPU. A function that runs on the GPU is a
**kernel**, and it runs in thousands of parallel **threads** — each handling one small piece, e.g.
one output element of a matmul. (Those threads are organized into **blocks** (small groups that
run together) and a **grid** (all the blocks for the job) — the structure you'd specify if you
*wrote* a kernel, as in the project's stretch.) When you call
`a @ b` on a `cuda` tensor, PyTorch launches a highly-optimized matmul *kernel* for you.

Picture it for a GELU over a million numbers (the project's stretch kernel): you launch a million
threads, one per element; thread `i` reads `x[i]`, computes `gelu(x[i])`, and writes `out[i]`. The
threads are grouped into blocks of, say, 256, so the grid is `1,000,000 / 256 ≈ 3,907` blocks. Every
thread runs the *same* one-line formula on *different* data — that "same instruction, many data"
shape is precisely what the hardware accelerates, and it's why writing a kernel is mostly "compute
element `i`" plus a little bookkeeping to work out *which* `i` each thread is.

You don't have to *write* CUDA to use a GPU — `.to("cuda")` and PyTorch's built-in kernels get you
the speedup. But you *can* write your own: PyTorch can compile an inline CUDA kernel at runtime
(`torch.utils.cpp_extension.load_inline`), which is the project's stretch goal — and exactly what
the deeper speed chapters build on. In practice you'll write a kernel roughly *never* — PyTorch's
built-ins cover almost everything and are faster than what you'd hand-write — so the stretch exists to
*demystify* the layer beneath `a @ b`, not because you'll need it. But seeing that a kernel is "just a
function each thread runs on one element" takes the magic out of CUDA, and is the doorway to `llm.c`
and the frontier work later in the course. (`mps` is Apple's equivalent; `ROCm` is AMD's. PyTorch hides
them all behind the same `device`.)

Between your `a @ b` and the silicon sits a stack: PyTorch calls into NVIDIA's libraries (**cuBLAS**
for matmuls, **cuDNN** for neural-net ops), which dispatch hand-tuned kernels for your exact GPU. You
stand on years of low-level optimization every time you write one line of tensor code — which is why
"just use PyTorch's built-ins" beats hand-rolling almost always, right up until you reach the
frontier where `llm.c` and custom kernels live.

---

## 🔌 How this plugs into the Storyteller

Nothing about your GPT changes — you write `device = get_device()`, `.to(device)` the model and
each batch, and the *same* Chapter 5 code now trains on a GPU many times faster. That's what makes
the later chapters (bigger models, more data) practical: the speed to actually train them.

This is the first of the **speed arc**. Chapter 8 is *where* compute happens (the device); Chapter 9
is *how precisely* you compute (fp16/bf16 — ~2× faster, half the memory); Chapter 10 is *across how
many* devices (multi-GPU). Together they're the difference between training a toy and training
something real — the same Chapter 5 architecture, just run fast enough to matter.

---

## 🐛 Building it yourself: what trips people up

The GPU gotchas are a rite of passage — here are the ones that bite first:

- **The same-device error.** "Expected all tensors to be on the same device" almost always means a
  batch is still on the CPU while the model is on the GPU. Move *each batch* inside the loop:
  `xb, yb = xb.to(device), yb.to(device)`.
- **`x.to(device)` silently does nothing.** For a **tensor** you must *reassign* —
  `x = x.to(device)` — because `.to` returns a *moved copy*. (A **model** moves in place, so
  `model.to(device)` alone is fine. The asymmetry trips everyone once.)
- **Timing without `synchronize()`.** You'll measure launch time, not compute time, and conclude your
  GPU is 1000× faster than it is. Sync before you stop the clock.
- **`.item()` in the inner loop.** Each call forces a CPU↔GPU sync. Accumulate on the GPU and only
  pull the number across every N steps.
- **`CUDA out of memory` mid-run.** Often the *batch size* is too big — or you're holding onto
  tensors (appending `loss` instead of `loss.item()` to a list keeps the whole computation graph
  alive). Smaller batches and `.item()`-ing logged values fix most cases.

---

## 🤔 Common questions

- **Do I need an NVIDIA GPU?** To run *CUDA*, yes — but you don't need to *own* one: Colab gives
  you a free one. Apple Silicon Macs also have a usable GPU (`mps`). And everything here runs
  (slower) on a CPU.
- **Why was the GPU slower for the small matmul?** Launch overhead: every GPU call costs a fixed
  amount to dispatch. For tiny work that overhead is bigger than the compute it saves, so the CPU
  wins until the work is large enough.
- **Why does my GPU timing look too good to be true?** You probably forgot `synchronize()` — you
  measured how fast the work was *queued*, not *done*.
- **What's `mps`?** Apple Silicon's GPU backend (Metal). It's a real GPU and gives a real (if
  smaller) speedup; the device-agnostic code treats it just like CUDA.
- **Is the speedup really 50×?** On a big model and a datacenter GPU, yes; on a laptop's
  integrated GPU it's more like 2×. It scales with the GPU and the size of the work.
- **CPU, CUDA, MPS — do I write different code for each?** No — that's the whole point of the
  device-agnostic pattern: `get_device()` picks the best available, `.to(device)` moves things, and
  the rest of your code is identical. Write once, run anywhere.
- **Why is moving data so slow when the GPU is so fast?** The GPU is fast at *computing* on data it
  already holds; getting data *to* it crosses a relatively narrow bus. Compute is cheap, movement is
  expensive — the opposite of most people's intuition, and the theme of the whole speed arc.
- **Does a GPU make a *small* model train faster?** Not necessarily — for tiny work the launch
  overhead can let the CPU win (the crossover). GPUs pay off once the work per step is big enough to
  keep thousands of cores busy.

## ✅ Check your understanding

<details>
<summary>1. What does the device-agnostic pattern look like, in three pieces?</summary>

`device = get_device()` once; `.to(device)` on the model; `.to(device)` on the data (each batch).
Everything else in the code is unchanged.
</details>

<details>
<summary>2. Why is the GPU *slower* than the CPU for a small matmul?</summary>

Every GPU call has fixed launch overhead. For small work that overhead outweighs the parallel
speedup, so the CPU wins until the matrices are big enough (the "crossover").
</details>

<details>
<summary>3. Why must you call <code>synchronize()</code> before timing GPU code?</summary>

GPU calls are asynchronous — they queue work and return immediately. Without `synchronize()` you
measure how fast you *launched* the work, not how long it took to *run*, making the GPU look
impossibly fast.
</details>

<details>
<summary>4. Name two ways to reduce <code>CUDA out of memory</code>.</summary>

A smaller batch size and a smaller model (also: mixed precision, Chapter 9). VRAM holds
parameters, optimizer state, and saved activations — all of which shrink with those.
</details>

<details>
<summary>5. Why does <code>x.to(device)</code> sometimes "do nothing," while <code>model.to(device)</code> works?</summary>

For a **tensor**, `.to(device)` returns a *moved copy* and leaves the original alone — you must
reassign: `x = x.to(device)`. A **model** (`nn.Module`) moves its parameters *in place*, so
`model.to(device)` on its own is enough. Forgetting to reassign a tensor is a classic silent bug.
</details>

<details>
<summary>6. You time a GPU matmul at 0.05 ms — faster than physically possible. What happened?</summary>

You forgot `synchronize()`. GPU calls are asynchronous, so you measured how long it took to *queue*
the work, not to *run* it. Call `torch.cuda.synchronize()` before stopping the clock for an honest
number.
</details>

## 🎓 Key takeaways

- A GPU is fast because a matmul is **massively parallel**; it's *slow* for small work because of
  **launch overhead** — there's a crossover size.
- Every tensor lives on a **`device`**; the **device-agnostic pattern** is `get_device()` +
  `.to(device)` on model and data, and **everything must be on the same device**.
- GPU calls are **asynchronous** — `synchronize()` before timing, and avoid needless
  **CPU↔GPU transfers** in the hot loop.
- **VRAM** is limited; smaller batches/models (and Chapter 9's precision) fit more.
- **CUDA** runs **kernels** in thousands of **threads**; `.to("cuda")` + PyTorch's kernels give
  you the speedup, and you *can* write your own.

## 📖 New vocabulary

`device` · `CPU` / `GPU` · `CUDA` · `MPS` · `.to(device)` · `device-agnostic` · `parallelism` ·
`launch overhead` · `crossover` · `asynchronous` · `synchronize` · `host↔device transfer` ·
`VRAM` · `out of memory` · `kernel` · `thread` / `block` / `grid`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/08-speed-device/code/explore.ipynb))
   — detect your device, move tensors/models, hit the same-device rule, benchmark, and train.
2. **Exercises** — [`exercises/`](./exercises/): fix a same-device error, find the crossover, see
   how `synchronize` changes a timing, and measure transfer overhead. A starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Benchmark Your GPU"** — plot the CPU-vs-GPU
   speedup curve on *your* hardware (run it on Colab for the real numbers), and find your crossover.
   Stretch: write a one-line **CUDA kernel** with `load_inline`.

## 🔗 Go deeper (optional)

- 📄 [NVIDIA — An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- 📄 [Horace He — Making Deep Learning Go Brrrr (From First Principles)](https://horace.io/brrr_intro.html)
  — why data movement, not compute, is usually the bottleneck.
- 💻 [karpathy/llm.c](https://github.com/karpathy/llm.c) — GPT training in raw C/CUDA, the
  end point of this speed arc.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 7 — Optimization](../07-optimization/) | [Syllabus](../../README.md#-syllabus) | [Chapter 9 — Need for Speed II: Precision](../09-speed-precision/) |
