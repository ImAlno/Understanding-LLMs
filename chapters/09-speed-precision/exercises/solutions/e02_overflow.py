"""
E02 — Find fp16's overflow cliff.
=================================
fp16 spends only 5 bits on the exponent, so its range is tiny: the biggest number it can hold is
65504. Push past that and it doesn't clamp — it becomes infinity. bf16 kept fp32's 8-bit exponent,
so it shrugs the same values off.

Run:  python e02_overflow.py
"""
import torch

print("largest power of 2 each format can hold (before it overflows to inf):")
for name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    e = 0
    while torch.isfinite(torch.tensor(2.0 ** e, dtype=dt)):
        e += 1
    print(f"  {name}: 2^{e - 1} is finite, 2^{e} -> inf   (true max is {torch.finfo(dt).max:.3g})")

print("\nconcrete values:")
for v in [60000, 65504, 70000]:
    f = torch.tensor(float(v), dtype=torch.float16).item()
    b = torch.tensor(float(v), dtype=torch.bfloat16).item()
    flag = "  <- fp16 overflowed!" if f == float("inf") else ""
    print(f"  {v:>6}:  fp16 -> {f!r:>10}   bf16 -> {b!r}{flag}")

print("\nThis is why fp16 needs care with large activations, and why bf16 (full range) is safer.")
