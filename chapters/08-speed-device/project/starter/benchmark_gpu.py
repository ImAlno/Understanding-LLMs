"""
Benchmark Your GPU — STARTER scaffold.
======================================
Fill the TODO: build the list of speedups. The plotting is done for you.
Reference: ../solution/benchmark_gpu.py.

    python benchmark_gpu.py        (run on Colab with a GPU)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from benchmark import get_device, bench_matmul

device = get_device()
sizes = [128, 256, 512, 1024, 2048, 4096]

if device == "cpu":
    print("No GPU detected — run this on Colab (Runtime → GPU) for the curve.")
    raise SystemExit(0)

# ✍️ TODO: for each n in `sizes`, time a CPU and a GPU matmul (bench_matmul(dev, n)) and append the
#          speedup (CPU ms / GPU ms) to `speedups`.
speedups = []      # replace
if not speedups:
    raise SystemExit("Fill the TODO (build the speedups list), then run again.")

cross = next((n for n, s in zip(sizes, speedups) if s > 1), None)
print(f"crossover (GPU starts winning): {('~' + str(cross)) if cross else 'not reached in this range'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(sizes, speedups, "o-")
    plt.axhline(1, color="gray", ls="--", label="break-even")
    plt.xscale("log", base=2)
    plt.xlabel("matrix size n"); plt.ylabel(f"speedup ({device} vs CPU)")
    plt.title(f"GPU speedup vs work size ({device})")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("gpu_speedup.png", dpi=120)
    print("saved gpu_speedup.png")
except ImportError:
    pass
