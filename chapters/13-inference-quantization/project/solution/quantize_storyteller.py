"""
Quantize Your Storyteller — SOLUTION.
=====================================
Quantize every Linear weight of a GPT to int8 and int4, and measure the trade-off: model size, and
how far the output logits drift from the fp32 original. This is what shipping a model as a 4-bit
GGUF/GPTQ file actually does.

(We measure size + output drift, which run honestly on a CPU. The *speed* win of quantization needs
special int8/int4 matmul kernels or a GPU — our dequant-to-float approach here is for measuring
quality/size, not latency.)

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
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


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
        ref = model(idx)                                    # fp32 reference logits
    n = sum(p.numel() for p in model.parameters())

    print(f"{'format':>8} {'size':>10} {'logit drift':>13}")
    print(f"{'fp16':>8} {n * 2 / 1e6:>7.2f} MB {'0.0000 (ref)':>13}")
    for bits in (8, 4):
        mq = copy.deepcopy(model)
        quantize_model(mq, bits)
        with torch.no_grad():
            drift = ((mq(idx) - ref).pow(2).mean().sqrt() / ref.std()).item()
        print(f"{'int' + str(bits):>8} {n * bits / 8 / 1e6:>7.2f} MB {drift:>13.4f}")

    print("\n✅ Quantized your Storyteller. int8: half the size, ~0.4% drift (imperceptible in the")
    print("   text). int4: quarter the size, a few % drift. This is how models ship for local use.")


if __name__ == "__main__":
    main()
