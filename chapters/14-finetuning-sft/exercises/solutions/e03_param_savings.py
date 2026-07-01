"""
E03 — LoRA's parameter savings grow with size.
===============================================
LoRA trains r·(d+k) parameters instead of d·k. For small layers that's a modest saving; for the big
layers of a real model it's a tiny fraction. Tabulate it, and estimate a whole model.

Run:  python e03_param_savings.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from lora import full_params, lora_params

r = 8
print(f"Trainable parameters, full fine-tuning vs LoRA (rank {r}):\n")
print(f"{'layer':>14} {'full':>14} {'LoRA':>10} {'LoRA %':>8}")
for d in (128, 1024, 4096, 8192):
    f, l = full_params(d, d), lora_params(d, d, r)
    print(f"  {d}×{d:<8} {f:>14,} {l:>10,} {100 * l / f:>7.2f}%")

# a rough whole-model estimate: a 7B model is ~mostly square-ish Linear weights of width ~4096
model_full = 7e9
approx_layers = model_full / (4096 * 4096)                # how many 4096×4096-equivalent layers
model_lora = approx_layers * lora_params(4096, 4096, r)
print(f"\n~7B model: full-FT trains {model_full/1e9:.0f}B params; LoRA (r={r}) trains ~{model_lora/1e6:.1f}M "
      f"({100*model_lora/model_full:.2f}%).")
print("Well under 1% — that's why you can fine-tune a huge model on modest hardware.")
