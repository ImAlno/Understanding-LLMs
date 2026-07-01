"""
E03 — Bit width vs error and size.
==================================
Fewer bits = fewer integer levels = coarser grid = more error, but smaller. Sweep the bit width and
see the error rise as the size falls — the core trade-off.

Run:  python e03_bits.py
"""
import torch


def quant_dequant(w, bits):                              # per-channel symmetric
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(1, keepdim=True) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


torch.manual_seed(0)
w = torch.randn(256, 256)

print(f"{'bits':>5} {'levels':>8} {'rel error':>10} {'7B size':>10}")
for bits in (8, 6, 4, 3, 2):
    wh = quant_dequant(w, bits)
    err = ((w - wh).pow(2).mean().sqrt() / w.std()).item()
    size_gb = 7e9 * (bits / 8) / 1e9
    print(f"{bits:>5} {2 ** bits:>8} {err:>10.4f} {size_gb:>8.1f} GB")

print("\n✓ int8 is near-lossless; error climbs steeply below int4 — which is why 4-bit (with")
print("  GPTQ/AWQ) is the usual floor for good quality, and int8 the safe default.")
