"""
Instruction-Tune Your Storyteller — STARTER scaffold.
=====================================================
Fill the one TODO: apply LoRA to the model (replace its Linear layers with LoRALinear adapters). The
freeze, SFT loop, and before/after are done for you. Reference: ../solution/instruction_tune.py.

Run:  python instruction_tune.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPT, PAIRS, build_vocab, make_example, respond
from lora import LoRALinear


def apply_lora(model, r=8):
    """Replace every nn.Linear in the model with a LoRALinear (frozen base + trainable adapter)."""
    # ✍️ TODO: wrap each nn.Linear in LoRALinear(child, r). Collect the layers to replace FIRST,
    #          then setattr them — mutating the module tree while iterating it recurses forever.
    #   to_wrap = [(parent, name, child)
    #              for parent in list(model.modules())
    #              for name, child in parent.named_children()
    #              if isinstance(child, nn.Linear)]
    #   for parent, name, child in to_wrap:
    #       setattr(parent, name, LoRALinear(child, r))
    return model        # replace  (currently does nothing — no adapters are added)


def main():
    torch.manual_seed(0)
    stoi, itos, V, block = build_vocab()
    examples = [make_example(i, r, stoi) for i, r in PAIRS]

    model = GPT(V, block)
    for p in model.parameters():
        p.requires_grad = False                          # freeze the whole base model
    apply_lora(model, r=8)                               # add trainable LoRA adapters

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise SystemExit("No trainable parameters — apply_lora didn't add any adapters. "
                         "Fill the ✍️ TODO, then re-run.")

    total = sum(p.numel() for p in model.parameters())
    print(f"trainable (LoRA only): {trainable:,} / {total:,} ({100 * trainable / total:.0f}% here)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    for _ in range(600):
        loss = sum(F.cross_entropy(model(torch.tensor(x).unsqueeze(0)).view(-1, V),
                                   torch.tensor(y), ignore_index=-100) for x, y in examples) / len(examples)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = sum(respond(model, i, stoi, itos) == r for i, r in PAIRS)
    for i, r in PAIRS:
        got = respond(model, i, stoi, itos)
        print(f"  '{i}' -> {got!r}   {'✓' if got == r else '✗'}")
    print(f"\n✅ {correct}/{len(PAIRS)} correct — instruction-tuned with LoRA (loss {loss.item():.4f}).")


if __name__ == "__main__":
    main()
