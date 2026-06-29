"""
E01 — Fix the same-device error.
================================
The #1 GPU bug: the model is on the GPU but the data is still on the CPU. The fix is always to
move the stray tensor with .to(device).

Run:  python e01_same_device.py        (locally, or on Colab with a GPU)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch, torch.nn as nn
from benchmark import get_device

device = get_device()
model = nn.Linear(10, 2).to(device)
batch = torch.randn(4, 10)              # created on the CPU — the bug, if the model is on a GPU

try:
    out = model(batch)
    print(f"[{device}] model(batch) worked — on CPU there's only one device, so no mismatch.")
except RuntimeError as e:
    print(f"[{device}] model(batch) FAILED -> {str(e).splitlines()[0][:70]}")

out = model(batch.to(device))           # the fix: same device as the model
print(f"[{device}] model(batch.to(device)) -> output on {out.device}. Fixed.")
print("\nIn a real training loop, that means moving EACH batch to the device before the forward pass.")
