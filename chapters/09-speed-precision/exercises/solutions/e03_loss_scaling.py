"""
E03 — Loss scaling: how big a scale do you need?
================================================
A gradient too small for fp16 underflows to 0. Multiply it UP into fp16's range before storing,
then divide back out in fp32, and it survives. Watch which scales rescue a 1e-8 gradient — too
small a scale doesn't help; a big enough one does (and bigger is more accurate).

Run:  python e03_loss_scaling.py
"""
import torch

g = 1e-8
naive = torch.tensor(g, dtype=torch.float16).item()
print(f"gradient g = {g}")
print(f"fp16(g) with no scaling = {naive}   (underflowed — gradient lost)\n")

print(f"{'scale':>12} {'recovered g':>14}   result")
for power in range(0, 25, 4):
    scale = 2.0 ** power
    recovered = torch.tensor(g * scale, dtype=torch.float16).item() / scale   # scale up, unscale
    result = "still lost" if recovered == 0 else "recovered"
    print(f"  2^{power:<2} = {scale:>8.0f} {recovered:>14.3e}   {result}")

print("\nToo small a scale leaves the gradient underflowed; once the scale lifts g into fp16's range")
print("it comes back. (bf16 keeps fp32's range, so it never underflows and needs no scaling at all.)")
