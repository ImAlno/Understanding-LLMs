"""
Quantization from scratch: floats -> small integers -> floats.
=============================================================
A trained model's weights are floats (fp16 = 2 bytes each). Quantization stores them as small
**integers** (int8 = 1 byte, int4 = half a byte) so the model is smaller and cheaper to move. The
core is a linear map: pick a **scale**, divide, round to an integer, and multiply back to recover an
approximation.

    q     = round(w / scale)   clamped to the integer range      (quantize)
    w_hat = q * scale                                             (dequantize)

This file shows the map on a few numbers, then the two things that decide the error: how many bits,
and whether the scale is shared across a whole tensor (per-tensor) or per row (per-channel).

Run:  python quantize.py
"""
import torch


def quantize_dequantize(w, bits, per_channel=False):
    """Symmetric quantize then dequantize. Returns the reconstructed float tensor `w_hat`."""
    qmax = 2 ** (bits - 1) - 1                       # int8 -> 127, int4 -> 7 (signed, symmetric)
    if per_channel:
        scale = w.abs().amax(dim=1, keepdim=True) / qmax   # one scale per row (output channel)
    else:
        scale = w.abs().max() / qmax                        # one scale for the whole tensor
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)   # the stored integers
    return q * scale                                           # dequantized back to float


def rel_error(w, w_hat):
    """Relative RMS reconstruction error (how far the quantized weights drifted, ÷ their spread)."""
    return ((w - w_hat).pow(2).mean().sqrt() / w.std()).item()


def main():
    torch.manual_seed(0)

    # 1) the map, on a few numbers
    print("Quantizing a small vector to int8:")
    v = torch.tensor([0.10, -0.42, 0.98, -0.05, 0.61])
    qmax = 127
    scale = v.abs().max() / qmax
    q = torch.clamp(torch.round(v / scale), -128, 127)
    print(f"  weights   : {[round(x, 2) for x in v.tolist()]}")
    print(f"  scale     : {scale.item():.5f}  (= max|w| / 127)")
    print(f"  integers  : {q.int().tolist()}  <- what's actually stored (1 byte each)")
    print(f"  recovered : {[round(x, 4) for x in (q * scale).tolist()]}  (≈ the originals)")

    # 2) more bits = less error (plain matrix, per-tensor — isolating the effect of the bit count)
    torch.manual_seed(0)
    wp = torch.randn(64, 64)
    print("\nBit count vs error (per-tensor, no outliers):")
    for bits in (8, 4):
        print(f"  int{bits}: relative error {rel_error(wp, quantize_dequantize(wp, bits)):.4f}")

    # 3) per-channel = much less error than per-tensor, especially with outliers
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    w[7] *= 20; w[30] *= 15                          # a couple of big "outlier" rows, like real LLM weights
    print("\nReconstruction error (relative RMS) on a matrix with outlier rows:")
    print(f"{'':>10} {'per-tensor':>12} {'per-channel':>12}")
    for bits in (8, 4):
        pt = rel_error(w, quantize_dequantize(w, bits, per_channel=False))
        pc = rel_error(w, quantize_dequantize(w, bits, per_channel=True))
        print(f"  int{bits:<6} {pt:>12.4f} {pc:>12.4f}")
    print("Per-tensor lets the outliers set one huge scale that crushes the small weights;")
    print("per-channel gives each row its own scale, so nothing is crushed. int8 is near-lossless.")

    # 4) memory
    print("\nModel size by weight format (7B parameters):")
    for name, bytes_per in (("fp16", 2), ("int8", 1), ("int4", 0.5)):
        print(f"  {name}: {7e9 * bytes_per / 1e9:>5.1f} GB")
    print("int8 halves the model; int4 quarters it — the whole point of quantization.")


if __name__ == "__main__":
    main()
