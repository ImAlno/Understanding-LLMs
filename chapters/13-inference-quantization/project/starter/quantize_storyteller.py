"""
Quantize Your Storyteller — STARTER scaffold.
=============================================
Fill the one TODO: the per-channel quantize/dequantize of a weight tensor. Everything else — walking
the model's Linear layers, measuring size and output drift — is done for you.
Reference: ../solution/quantize_storyteller.py.

Run:  python quantize_storyteller.py
"""
import sys
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn as nn
from gpt import GPT, random_batch


def quantize_dequantize(w, bits):
    """Per-channel symmetric quantize -> dequantize of a weight tensor."""
    qmax = 2 ** (bits - 1) - 1
    # ✍️ TODO: scale = per-row max|w| / qmax ; q = round(w/scale) clamped to [-qmax-1, qmax] ; return q*scale
    #   scale = w.abs().amax(dim=1, keepdim=True) / qmax
    #   return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    return w        # replace  (currently returns the weight UNCHANGED — no quantization yet)


@torch.no_grad()
def quantize_model(model, bits):
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data = quantize_dequantize(mod.weight.data, bits)


def main():
    torch.manual_seed(0)
    model = GPT().eval()
    idx = random_batch()
    with torch.no_grad():
        ref = model(idx)
    n = sum(p.numel() for p in model.parameters())

    rows = []
    for bits in (8, 4):
        mq = copy.deepcopy(model)
        quantize_model(mq, bits)
        with torch.no_grad():
            drift = ((mq(idx) - ref).pow(2).mean().sqrt() / ref.std()).item()
        rows.append((bits, n * bits / 8 / 1e6, drift))

    if rows[1][2] < 0.01:                                    # int4 drift ≈ 0 means nothing was quantized
        raise SystemExit("The output didn't change — you haven't quantized yet. Fill the ✍️ TODO in "
                         "quantize_dequantize, then re-run.")

    print(f"{'format':>8} {'size':>10} {'logit drift':>13}")
    print(f"{'fp16':>8} {n * 2 / 1e6:>7.2f} MB {'0.0000 (ref)':>13}")
    for bits, size, drift in rows:
        print(f"{'int' + str(bits):>8} {size:>7.2f} MB {drift:>13.4f}")
    print("\n✅ Quantized your Storyteller — half the size at int8 (imperceptible drift), a quarter at int4.")


if __name__ == "__main__":
    main()
