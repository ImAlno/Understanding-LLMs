"""
E01 — Half the bits, half the memory.
=====================================
A 16-bit number is 2 bytes; fp32 is 4. A tensor's storage cost is just bytes-per-number x count,
so switching to fp16/bf16 halves the memory for weights, gradients, optimizer state, and
activations. Compute it for a model-sized parameter count.

Run:  python e01_memory.py
"""
import torch

N = 100_000_000   # 100M parameters — a small-ish model

print(f"storing {N:,} parameters in each format:")
sizes = []
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    bytes_each = torch.zeros(1, dtype=dt).element_size()   # 4, 2, 2 — no big allocation
    gb = bytes_each * N / 1e9
    sizes.append((name, gb))
    print(f"  {name}: {bytes_each} bytes/number  ->  {gb:.2f} GB")

fp32_gb = sizes[0][1]
print(f"\nfp16/bf16 use {sizes[1][1] / fp32_gb:.0%} of fp32's memory — so about twice the model")
print("(or twice the batch) fits in the same VRAM. And remember (Ch8): half the bytes also means")
print("half the data to move, which is most of where the GPU speedup comes from.")
