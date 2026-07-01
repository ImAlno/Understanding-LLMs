"""
LoRA from scratch: fine-tune with tiny low-rank adapters.
=========================================================
Full fine-tuning updates *every* weight — for a big model, that means storing and optimizing billions
of parameters (and a full copy per task). **LoRA** (Low-Rank Adaptation) instead **freezes** the
original weight `W` and learns a small **low-rank** update `B·A` beside it:

    output = W·x  +  (B·A)·x          with A: r×k,  B: d×r,  and r ≪ d, k

Only `A` and `B` are trained — a tiny fraction of the parameters. The idea (the "low-rank
hypothesis"): the *change* a task needs is simple, so a low-rank matrix captures it.

This file builds `LoRALinear`, counts how few parameters it trains, and shows it can exactly fit a
low-rank weight update.

Run:  python lora.py
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear and add a trainable rank-`r` update B·A."""
    def __init__(self, base: nn.Linear, r: int = 8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                          # freeze the original weight (and bias)
        d, k = base.out_features, base.in_features
        self.A = nn.Parameter(torch.randn(r, k) * 0.02)      # small random
        self.B = nn.Parameter(torch.zeros(d, r))             # zero -> the adapter starts as a no-op

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T      # W·x + B·A·x


def full_params(d, k):
    return d * k                                             # the weight matrix (ignoring bias)


def lora_params(d, k, r):
    return r * (d + k)                                       # A (r×k) + B (d×r)


def main():
    torch.manual_seed(0)

    # 1) how few parameters LoRA trains — and how the win grows with the matrix size
    r = 8
    print(f"Trainable parameters, full fine-tuning vs LoRA (rank {r}):")
    print(f"{'layer':>14} {'full':>12} {'LoRA':>10} {'LoRA %':>8}")
    for d in (128, 1024, 4096):
        f, l = full_params(d, d), lora_params(d, d, r)
        print(f"  {d}×{d:<8} {f:>12,} {l:>10,} {100 * l / f:>7.1f}%")
    print("The bigger the layer, the tinier LoRA's share — for a whole 7B model it's well under 1%.")

    # 2) LoRA can exactly fit a low-rank weight update (the kind fine-tuning actually needs)
    d = k = 96
    base = nn.Linear(d, k, bias=False)
    W_target = base.weight.data + (torch.randn(k, r) @ torch.randn(r, d)) * 0.1   # W + a rank-r change
    X = torch.randn(512, d)
    Y = X @ W_target.T                                       # the "task": reproduce the adapted layer

    lora = LoRALinear(base, r)
    opt = torch.optim.Adam([lora.A, lora.B], lr=0.02)        # train ONLY the adapter
    first = None
    for _ in range(400):
        loss = ((lora(X) - Y) ** 2).mean()
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"\nLoRA fitting a rank-{r} target: loss {first:.3f} -> {loss.item():.6f}")
    print("It nails it — because the update the task needs really is low-rank. That's the whole bet.")


if __name__ == "__main__":
    main()
