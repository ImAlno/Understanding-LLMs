"""
E01 — Fix the same-device error (STARTER scaffold).
===================================================
Fill the TODO so the forward pass works on any device. Reference: ../solutions/e01_same_device.py.

Run:  python e01_same_device.py        (locally, or on Colab with a GPU)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch, torch.nn as nn
from benchmark import get_device

device = get_device()
model = nn.Linear(10, 2).to(device)     # the model is on `device`
batch = torch.randn(4, 10)              # the batch is on the CPU

# ✍️ TODO: run the model on the batch — but first put the batch on the SAME device as the model.
out = None      # replace
if out is None:
    raise SystemExit("Fill the TODO (move the batch onto the device, then call the model), then run again.")

print(f"[{device}] output is on {out.device}. Fixed — the batch and model are on the same device.")
