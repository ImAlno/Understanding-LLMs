"""
E01 — Half the bits, half the memory (STARTER scaffold).
========================================================
Fill the TODO: build `sizes` — for each format, the storage cost of N parameters.
Reference: ../solutions/e01_memory.py.

Run:  python e01_memory.py
"""
import torch

N = 100_000_000   # 100M parameters — a small-ish model

print(f"storing {N:,} parameters in each format:")

# ✍️ TODO: for each (name, dtype) below, compute bytes-per-number with
#          torch.zeros(1, dtype=dt).element_size(), then GB = bytes_each * N / 1e9,
#          and append (name, gb) to `sizes`.
sizes = []      # replace
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    pass        # replace with the computation + sizes.append((name, gb))

if not sizes:
    raise SystemExit("Fill the TODO (build the `sizes` list), then run again.")

for name, gb in sizes:
    print(f"  {name}: {gb:.2f} GB")
print(f"\nfp16/bf16 use {sizes[1][1] / sizes[0][1]:.0%} of fp32's memory — about twice the model fits.")
