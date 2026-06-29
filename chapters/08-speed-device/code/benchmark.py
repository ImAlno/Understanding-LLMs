"""
CPU vs GPU: the device, and the speedup.
========================================
A neural network is mostly matrix multiplies, and a matrix multiply is *embarrassingly parallel*
— millions of independent multiply-adds. A CPU has a handful of fast cores; a GPU has thousands
of slower ones, so for big enough matrices it wins enormously. But for *small* work the GPU's
launch overhead makes it slower — there's a crossover.

This file finds that crossover on whatever device you have:
  • "cuda"  — an NVIDIA GPU (what Colab gives you, and what real training uses)
  • "mps"   — Apple Silicon's built-in GPU (Metal)
  • "cpu"   — no GPU; you'll see CPU times only (run this on Colab for the GPU column)

Run:  python benchmark.py        (locally, or on Colab with Runtime -> GPU)
"""
import time
import torch


def get_device():
    """Pick the best available device, in order: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Wait for the GPU to actually finish — GPU calls are queued asynchronously, so timing
    without this measures only how fast we *launched* the work, not how fast it *ran*."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_matmul(device, n, reps=10):
    """Average milliseconds for one n×n matmul on `device`."""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b
    sync(device)                                  # warm up (first call includes setup)
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    sync(device)                                  # make sure all reps finished before we stop the clock
    return (time.perf_counter() - t0) / reps * 1000


def main():
    dev = get_device()
    note = "  (no GPU here — run on Colab with Runtime → GPU for the GPU column)" if dev == "cpu" else ""
    print(f"GPU device on this machine: {dev}{note}\n")
    print(f"{'size':>6} {'CPU ms':>9} {'GPU ms':>9} {'speedup':>8}")
    for n in [256, 512, 1024, 2048, 4096]:
        ct = bench_matmul("cpu", n)
        if dev != "cpu":
            gt = bench_matmul(dev, n)
            print(f"{n:>6} {ct:>9.2f} {gt:>9.2f} {ct / gt:>6.1f}x")
        else:
            print(f"{n:>6} {ct:>9.2f} {'—':>9} {'—':>8}")
    print("\nThe GPU's advantage grows with size. For tiny matmuls a weak laptop GPU can even lose")
    print("(its launch overhead isn't worth it); big matmuls win. And a datacenter GPU (Colab's T4)")
    print("wins by far more than a laptop — tens of times, and at every size, not ~2x.")


if __name__ == "__main__":
    main()
