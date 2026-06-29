"""
E03 — Why timing needs synchronize.
===================================
GPU calls are asynchronous — they return before the work finishes. Time without synchronize and
the GPU looks impossibly fast.

Run:  python e03_synchronize.py        (locally, or on Colab with a GPU)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
from benchmark import get_device, sync

device = get_device()
if device == "cpu":
    print("No GPU here — on a CPU there's nothing async to wait for. Run on Colab/MPS to see the trap.")
else:
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    a @ b; sync(device)                                     # warmup
    t0 = time.perf_counter(); a @ b; t_no = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); a @ b; sync(device); t_yes = (time.perf_counter() - t0) * 1000
    print(f"WITHOUT synchronize: {t_no:6.2f} ms   (looks ~{t_yes / t_no:.0f}x faster than reality!)")
    print(f"WITH    synchronize: {t_yes:6.2f} ms   (the honest time)")
    print("\nWithout sync you measure how fast the work was *launched*, not how long it *ran*.")
