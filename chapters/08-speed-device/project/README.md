# ⚡ Mini-Project — Benchmark Your GPU

You've seen the GPU speedup; now **measure it on your own hardware** and plot the curve. The shape
— slow for small work, winning big for large — is the whole story of when a GPU helps.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/08-speed-device/code/explore.ipynb)

> **Best run on Colab** (Runtime → Change runtime type → GPU) for real CUDA numbers — but it also
> runs on an Apple GPU (MPS) or just prints a note on a pure CPU.

> **How this works:** [`starter/benchmark_gpu.py`](./starter/benchmark_gpu.py) has one TODO —
> build the list of CPU-vs-GPU speedups — then it finds your crossover and plots the curve. A
> reference is in [`solution/benchmark_gpu.py`](./solution/benchmark_gpu.py).

## 🎯 What it does

```bash
python starter/benchmark_gpu.py
```

It times an `n×n` matmul on the CPU and the GPU for `n` from 128 to 4096, computes the speedup at
each, prints your **crossover** (the size where the GPU starts winning), and saves the
speedup-vs-size curve to `gpu_speedup.png` in the current directory.

## 🛠️ Your TODO

Build `speedups` — for each size `n`, the ratio `CPU_ms / GPU_ms` (use `bench_matmul("cpu", n)`
and `bench_matmul(device, n)`).

## ✅ Checking your work

- The curve should rise from **below 1** (CPU wins on small matmuls) through **break-even** to
  **above 1** (GPU wins on big ones).
- Your **crossover** and top speedup depend on your hardware: a laptop GPU tops out around ~2×; a
  Colab T4 goes tens of times higher and crosses over at a smaller size.
- Compare against `solution/benchmark_gpu.py`.

## 🚀 Stretch — write a real CUDA kernel

[`solution/cuda_gelu_stretch.py`](./solution/cuda_gelu_stretch.py) compiles a **hand-written CUDA
kernel for GELU** at runtime (`torch.utils.cpp_extension.load_inline`), where each GPU thread
computes one element, and checks it against PyTorch's GELU.

> ⚠️ **CUDA-only, and needs `ninja`** — it requires an NVIDIA GPU, so run it on **Colab** (it
> prints a note and exits on CPU/Apple GPU). Colab compiles the kernel with the `ninja` build
> tool, which isn't pre-installed — run `!pip install ninja` in a cell first. It's written against
> the standard `load_inline` pattern but was *not* testable on the author's machine (no local
> CUDA) — if your Colab run needs a tweak, let us know.

This is the doorway to the rest of the speed arc: PyTorch's fast ops are kernels like this, and
[`llm.c`](https://github.com/karpathy/llm.c) is a whole GPT written in them.

## Other extensions

- **Train on the GPU.** Move your Chapter 5 GPT to `device` and time a training step vs CPU.
- **Bigger matrices.** Push `n` to 8192+ on Colab and watch the speedup keep climbing.
- **Memory.** Print `torch.cuda.memory_allocated() / 1e9` GB before/after building a big model.

Next: [Chapter 9 — Need for Speed II: Precision](../../09-speed-precision/), where fp16/bf16 makes
the GPU ~2× faster again (and halves the memory). 🏎️
