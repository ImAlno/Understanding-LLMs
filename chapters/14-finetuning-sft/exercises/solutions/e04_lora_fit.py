"""
E04 — LoRA fits a low-rank target (and can't fit a full-rank one).
==================================================================
LoRA's bet is that the change fine-tuning needs is LOW-RANK. Show it: LoRA nails a target that
differs from its frozen base by a rank-8 update, but leaves big error when the difference is
full-rank (rank 64) — which rank-8 simply can't represent.

Run:  python e04_lora_fit.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn as nn
from lora import LoRALinear


def fit(change_rank, r=8, steps=400):
    """Freeze a random base, target = base + a `change_rank`-rank update, train LoRA to match."""
    torch.manual_seed(0)
    base = nn.Linear(64, 64, bias=False)                 # LoRA freezes THIS base
    if change_rank == 8:
        change = (torch.randn(64, 8) @ torch.randn(8, 64)) * 0.1     # rank-8 update
    else:
        change = torch.randn(64, 64) * 0.1                            # full-rank (64) update
    W_target = base.weight.data + change                 # so target - base == change
    X = torch.randn(512, 64); Y = X @ W_target.T

    lora = LoRALinear(base, r)
    opt = torch.optim.Adam([lora.A, lora.B], lr=0.02)
    for _ in range(steps):
        loss = ((lora(X) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


print(f"LoRA (rank 8) fitting a rank-8 change    : final loss {fit(8):.6f}")
print(f"LoRA (rank 8) fitting a full-rank change : final loss {fit(64):.6f}")
print("\nIt nails the low-rank change and can't fully capture the full-rank one — which is fine, because")
print("the adaptations fine-tuning actually needs ARE low-rank. That's the whole reason LoRA works.")
