"""
Quantize a whole model, and measure what it costs.
==================================================
Real quantization is applied to a model's **Linear** weights (the big matmuls). We quantize each
`nn.Linear`'s weight per-channel to int8, then int4, run the model, and measure how far the output
logits **drift** from the fp32 model — plus how much smaller the model gets.

The result is the headline of the chapter: int8 is nearly lossless (tiny drift, half the size); int4
drifts more but quarters the size. (We quantize *weights only* and keep activations in float — the
standard recipe for LLM inference.)

Run:  python quantize_model.py
"""
import copy
import torch
import torch.nn as nn
from gpt import GPT, random_batch


@torch.no_grad()
def quantize_linears(model, bits):
    """Replace each Linear weight with its per-channel symmetric int{bits} reconstruction, in place."""
    qmax = 2 ** (bits - 1) - 1
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            scale = w.abs().amax(dim=1, keepdim=True) / qmax     # per output channel (row)
            mod.weight.data = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def main():
    torch.manual_seed(0)
    model = GPT().eval()
    idx = random_batch()

    with torch.no_grad():
        ref = model(idx)                                          # the fp32 reference logits

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params\n")
    print(f"{'format':>8} {'size':>10} {'logit drift':>13}")
    print(f"{'fp16':>8} {n_params * 2 / 1e6:>7.2f} MB {'0.0000 (ref)':>13}")

    for bits in (8, 4):
        mq = copy.deepcopy(model)
        quantize_linears(mq, bits)
        with torch.no_grad():
            out = mq(idx)
        drift = ((out - ref).pow(2).mean().sqrt() / ref.std()).item()
        size = n_params * (bits / 8) / 1e6
        print(f"{'int' + str(bits):>8} {size:>7.2f} MB {drift:>13.4f}")

    print("\nint8: half the size, output barely moves (drift < 1%). int4: quarter the size, a few % drift —")
    print("visible but often fine, and the sweet spot for serving big models (with GPTQ/AWQ to sharpen it).")


if __name__ == "__main__":
    main()
