"""
E02 — Find the crossover.
=========================
Below some matrix size the CPU wins (GPU launch overhead); above it the GPU wins, by more as the
work grows.

Run:  python e02_crossover.py        (locally, or on Colab with a GPU)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from benchmark import get_device, bench_matmul

device = get_device()
if device == "cpu":
    print("No GPU here — run on Colab (Runtime → GPU). On a GPU: small = CPU wins, big = GPU wins.")
else:
    print(f"finding the crossover on '{device}':")
    for n in [128, 256, 512, 1024, 2048, 4096]:
        ct = bench_matmul("cpu", n)
        gt = bench_matmul(device, n)
        winner = "GPU" if gt < ct else "CPU"
        print(f"  {n:>5}x{n:<5}: CPU {ct:7.2f} ms | GPU {gt:7.2f} ms  ->  {winner} wins ({ct / gt:.1f}x)")
    print("\nThe crossover is where the GPU starts winning. On a weak GPU (a laptop) it's around")
    print("1024-2048 (and noisy near there); on a strong one (a datacenter GPU) it's so small the")
    print("GPU wins at EVERY size here. If you see no 'CPU wins' row at all, your GPU is strong.")
