"""
The device-agnostic training pattern.
=====================================
The same training loop runs on a CPU, an Apple GPU (MPS), or an NVIDIA GPU (CUDA, e.g. on Colab)
— you change exactly one thing: the `device`. The rule is simple: the **model and the data must
be on the same device**. Put both there with `.to(device)` and everything else is unchanged.

Run:  python device_train.py        (locally, or on Colab with Runtime -> GPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark import get_device

device = get_device()
print(f"training on: {device}\n")

# ---- data: created right on the device ----
torch.manual_seed(0)
X = torch.randn(2000, 20, device=device)
w_true = torch.randn(20, 1, device=device)
y = X @ w_true + 0.1 * torch.randn(2000, 1, device=device)

# ---- model: moved to the device with .to(device) ----
model = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(500):
    loss = F.mse_loss(model(X), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step:3d} | loss {loss.item():.4f}")

print(f"\nfinal loss {loss.item():.4f} — and every tensor lived on '{device}'.")
print("The only device-specific code: get_device(), and .to(device) on the data and the model.")
print("Flip the device and this same loop runs on a CPU, an Apple GPU, or a CUDA GPU on Colab.")
