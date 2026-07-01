"""
E02 — LoRALinear: frozen base, no-op start.
===========================================
Two properties that make LoRA behave: (1) it starts as a NO-OP (B=0, so B·A=0 and the output equals
the frozen base), and (2) only A and B are trainable — the base weight is frozen.

Run:  python e02_lora_module.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn as nn
from lora import LoRALinear

torch.manual_seed(0)
base = nn.Linear(64, 64)
lora = LoRALinear(base, r=8)

x = torch.randn(4, 64)

# 1) starts as a no-op: at init (B=0), the adapter output equals the frozen base's
print("output == base at init:", torch.allclose(lora(x), base(x)))
assert torch.allclose(lora(x), base(x))

# 2) only A and B are trainable; the base weight/bias are frozen
trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
frozen = [n for n, p in lora.named_parameters() if not p.requires_grad]
print("trainable:", trainable)
print("frozen   :", frozen)
assert set(trainable) == {"A", "B"}
assert all("base" in n for n in frozen)
print("✓ LoRA starts as a no-op and trains only the tiny A, B — the base is frozen.")
