"""
E01 — Implement quantize / dequantize (STARTER scaffold).
=========================================================
Fill the TODO: the three-line symmetric int8 map. Reference: ../solutions/e01_quantize.py.

Run:  python e01_quantize.py
"""
import torch

w = torch.tensor([0.10, -0.42, 0.98, -0.05, 0.61])
qmax = 127

# ✍️ TODO: symmetric int8 quantize -> dequantize
#   scale = max|w| / qmax ; q = round(w/scale) clamped to [-128, 127] ; w_hat = q * scale
scale = None      # replace
q = None          # replace
w_hat = None      # replace

if w_hat is None:
    raise SystemExit("Fill the TODO (scale, q, w_hat), then run again.")

print(f"integers  : {q.int().tolist()}")
print(f"recovered : {[round(x, 4) for x in w_hat.tolist()]}")
assert q.int().tolist() == [13, -54, 127, -6, 79], "check scale = max|w|/127 and round/clamp"
assert torch.allclose(w, w_hat, atol=scale.item())
print("✓ Correct — the int8 quantize/dequantize map.")
