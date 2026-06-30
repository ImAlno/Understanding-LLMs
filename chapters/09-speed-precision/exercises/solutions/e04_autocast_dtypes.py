"""
E04 — What autocast does to each operation.
===========================================
`torch.autocast` doesn't blindly cast everything to 16-bit. It runs matmuls (the heavy, parallel
work) in fast 16-bit, but keeps numerically sensitive ops — softmax, normalization, sums, the loss
— in fp32. See it per op. (This runs anywhere; on a CUDA GPU the 16-bit matmul is also *faster*.)

Run:  python e04_autocast_dtypes.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn.functional as F
from mixed_precision_train import get_device

device = get_device()
x = torch.randn(64, 256, device=device)
w = torch.randn(256, 256, device=device)

print(f"[{device}] dtype of each op INSIDE autocast(bf16):")
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    ops = [
        ("matmul  x @ w", x @ w),
        ("softmax", F.softmax(x, dim=-1)),
        ("layernorm", F.layer_norm(x, (256,))),
        ("sum", x.sum()),
    ]
    for label, out in ops:
        kind = "16-bit (lowered for speed)" if out.dtype == torch.bfloat16 else "fp32 (kept precise)"
        print(f"  {label:>14}: {str(out.dtype).split('.')[-1]:>9}   {kind}")

print("\nautocast lowers the matmul but protects the precision-sensitive ops — you don't pick per op,")
print("it uses a vetted list. That's how you get 16-bit speed without 16-bit instability.")
