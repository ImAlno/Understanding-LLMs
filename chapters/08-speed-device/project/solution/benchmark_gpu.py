"""
Benchmark Your GPU — REFERENCE SOLUTION.
========================================
Sweep matrix sizes, measure the GPU speedup over the CPU at each, plot the curve, and find the
crossover (where the GPU starts winning).

    python benchmark_gpu.py        (run on Colab with a GPU for the real numbers)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from benchmark import get_device, bench_matmul

device = get_device()
sizes = [128, 256, 512, 1024, 2048, 4096]

if device == "cpu":
    print("No GPU detected — run this on Colab (Runtime → Change runtime type → GPU) for the curve.")
    raise SystemExit(0)

speedups = []
print(f"benchmarking matmul on '{device}' vs CPU:")
for n in sizes:
    ct = bench_matmul("cpu", n)
    gt = bench_matmul(device, n)
    speedups.append(ct / gt)
    print(f"  {n:>5}x{n:<5}  {ct / gt:5.2f}x")

cross = next((n for n, s in zip(sizes, speedups) if s > 1), None)
print(f"\ncrossover (GPU starts winning): {('~' + str(cross)) if cross else 'not reached in this range'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(sizes, speedups, "o-")
    plt.axhline(1, color="gray", ls="--", label="break-even (GPU = CPU)")
    plt.xscale("log", base=2)
    plt.xlabel("matrix size n"); plt.ylabel(f"speedup ({device} vs CPU)")
    plt.title(f"GPU speedup grows with the work ({device})")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("gpu_speedup.png", dpi=120)
    print("saved gpu_speedup.png to the current directory")
except ImportError:
    print("(install matplotlib to see the curve)")
