"""
E02 — Per-channel beats per-tensor on outliers.
===============================================
LLM weight matrices have a few outlier rows. One per-tensor scale, set by the biggest outlier,
crushes every ordinary weight. Per-channel — one scale per row — fixes it. Measure the gap.

Run:  python e02_per_channel.py
"""
import torch


def quant_dequant(w, bits, per_channel):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(1, keepdim=True) / qmax if per_channel else w.abs().max() / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def rel_error(w, w_hat):
    return ((w - w_hat).pow(2).mean().sqrt() / w.std()).item()


torch.manual_seed(0)
w = torch.randn(64, 64)
w[7] *= 20; w[30] *= 15                                   # outlier rows, like real LLM weights

print(f"{'':>8} {'per-tensor':>12} {'per-channel':>12}")
for bits in (8, 4):
    pt = rel_error(w, quant_dequant(w, bits, per_channel=False))
    pc = rel_error(w, quant_dequant(w, bits, per_channel=True))
    print(f"  int{bits:<4} {pt:>12.4f} {pc:>12.4f}   ({pt / pc:.1f}x less error)")

pt4 = rel_error(w, quant_dequant(w, 4, False))
pc4 = rel_error(w, quant_dequant(w, 4, True))
assert pc4 < pt4, "per-channel should beat per-tensor"
print("\n✓ Per-channel gives each row its own scale, so the outliers can't crush the rest.")
