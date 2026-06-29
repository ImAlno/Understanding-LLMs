"""
E04 — The cost of moving data.
==============================
A GPU has its own memory; moving tensors across the CPU<->GPU boundary is not free. Often it's
comparable to real compute — which is why you keep data on the GPU.

Run:  python e04_transfer.py        (locally, or on Colab with a GPU)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
from benchmark import get_device, sync

device = get_device()
if device == "cpu":
    print("No GPU here — there's no CPU<->GPU boundary to cross. Run on Colab/MPS to measure it.")
else:
    n = 4096
    x_cpu = torch.randn(n, n)
    t0 = time.perf_counter(); x = x_cpu.to(device); sync(device); t_move = (time.perf_counter() - t0) * 1000
    a = torch.randn(n, n, device=device); a @ a; sync(device)                       # warmup
    t0 = time.perf_counter(); a @ a; sync(device); t_mm = (time.perf_counter() - t0) * 1000
    print(f"moving a {n}x{n} tensor CPU -> {device}: {t_move:6.2f} ms")
    print(f"a {n}x{n} matmul on {device}:           {t_mm:6.2f} ms")
    print("\nMoving data costs real time — so move a batch over once and do all the work there;")
    print("don't .item()/.cpu()/print() tensors every step in the hot loop.")
