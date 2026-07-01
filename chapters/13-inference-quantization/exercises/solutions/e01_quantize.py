"""
E01 — Implement quantize / dequantize.
======================================
The core of quantization: map floats onto an integer grid, then back. Symmetric int8 uses the
integers -127..127. Prove the recovered floats land within a rounding step of the originals.

Run:  python e01_quantize.py
"""
import torch

w = torch.tensor([0.10, -0.42, 0.98, -0.05, 0.61])
qmax = 127

scale = w.abs().max() / qmax                              # how much float each integer step is worth
q = torch.clamp(torch.round(w / scale), -128, 127)        # the stored integers
w_hat = q * scale                                         # dequantized approximation

print(f"scale     : {scale.item():.5f}")
print(f"integers  : {q.int().tolist()}   (1 byte each)")
print(f"recovered : {[round(x, 4) for x in w_hat.tolist()]}")

assert q.int().tolist() == [13, -54, 127, -6, 79]
assert torch.allclose(w, w_hat, atol=scale.item())        # every weight within one step
print("✓ Quantized to int8 and recovered within a rounding step — the whole map.")
