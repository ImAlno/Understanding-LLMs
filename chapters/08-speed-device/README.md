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
anywhere; on a GPU they light up.

---

## A 2-minute primer: why a GPU?

A neural network's heavy lifting is **matrix multiplication**, and a matmul is *embarrassingly
parallel* — each output number is an independent dot product, millions of them, all computable at
once. Two very different chips:

- A **CPU** has a handful of big, fast cores (great for sequential logic).
- A **GPU** has *thousands* of small, slower cores (great for doing the same simple thing to a lot
  of data at once).

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

To put a tensor or a model on a device, use **`.to(device)`** (tensors are returned moved;
`nn.Module`s move in place):

```python
device = get_device()
x = torch.randn(1000, 1000).to(device)       # or: torch.randn(1000, 1000, device=device)
model = MyModel().to(device)
```

That's the whole abstraction. Writing `device = get_device()` once and `.to(device)` on your data
and model is the **device-agnostic pattern**: the same code runs on a CPU, an Apple GPU, or a CUDA
GPU on Colab — you change *nothing else*.

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
CPU. The fix is always the same — move the stray tensor: `a.to(device) + b`. In a training loop
that means moving **each batch** to the device before the forward pass.

---

## 3. The speedup — and its crossover

> ✏️ **In the notebook → Step 4.** Benchmark it.

Time an `n×n` matmul on the CPU and the GPU as `n` grows. On this Apple laptop (MPS):

```
 size    CPU ms    GPU ms  speedup
  256      0.03      0.11    0.2x      <- CPU wins! GPU overhead isn't worth it
 1024      1.22      1.11    1.1x      <- about even (the crossover)
 4096     74.71     46.94    1.6x      <- GPU wins
```

Two lessons. **(1)** For *small* work the GPU is **slower** — every GPU call has launch overhead,
and for a tiny matmul that overhead dominates. There's a **crossover size** below which the CPU
wins. **(2)** The speedup *grows* with the size of the work. On a laptop's integrated GPU it tops
out around ~2×; on a **datacenter GPU like Colab's T4 the same benchmark shows tens of times** —
because real GPUs have far more cores and memory bandwidth. Run the notebook on Colab to see it.

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

---

## 5. The cost of moving data

A GPU has its **own memory**, separate from your CPU's RAM. Moving data across the
CPU↔GPU boundary (`.to(device)`, `.cpu()`, `.item()`, `print(tensor)`) is **slow** relative to
compute. Two habits follow:

- **Keep data on the GPU.** Move a batch over once, do all the work there. Don't bounce tensors
  back and forth.
- **Don't `.item()` / `.cpu()` in the hot loop.** Pulling the loss back to the CPU *every step* to
  print it forces a sync and a transfer; do it every N steps instead.

A surprising amount of "my GPU training is slow" is really "I'm copying data across the boundary
too often."

---

## 6. Memory: VRAM

A GPU's memory (**VRAM**) holds the model's parameters, the optimizer state, *and* the
activations saved for backprop — and it's limited (Colab's T4 has 16 GB; a laptop GPU often less).
Run out and you get the dreaded **`CUDA out of memory`**. The knobs that use less: a **smaller
batch size**, a **smaller model**, and tricks from later chapters (mixed precision in
[Chapter 9](../09-speed-precision/), which roughly halves it). For this course's models, a free
Colab GPU has memory to spare.

---

## 7. What CUDA actually is

**CUDA** is NVIDIA's platform for running code on the GPU. A function that runs on the GPU is a
**kernel**, and it runs in thousands of parallel **threads** (organized into *blocks* and a
*grid*) — each thread handling one small piece, e.g. one output element of a matmul. When you call
`a @ b` on a `cuda` tensor, PyTorch launches a highly-optimized matmul *kernel* for you.

You don't have to *write* CUDA to use a GPU — `.to("cuda")` and PyTorch's built-in kernels get you
the speedup. But you *can* write your own: PyTorch can compile an inline CUDA kernel at runtime
(`torch.utils.cpp_extension.load_inline`), which is the project's stretch goal — and exactly what
the deeper speed chapters build on. (`mps` is Apple's equivalent; `ROCm` is AMD's. PyTorch hides
them all behind the same `device`.)

---

## 🔌 How this plugs into the Storyteller

Nothing about your GPT changes — you write `device = get_device()`, `.to(device)` the model and
each batch, and the *same* Chapter 5 code now trains on a GPU many times faster. That's what makes
the later chapters (bigger models, more data) practical: the speed to actually train them.

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
