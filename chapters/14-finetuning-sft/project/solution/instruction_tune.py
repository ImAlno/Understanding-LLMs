"""
Instruction-Tune Your Storyteller — SOLUTION.
=============================================
Put it all together: freeze a GPT, bolt LoRA adapters onto its Linear layers, and SFT it (with loss
masking) into an instruction-follower — training only a sliver of the parameters.

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
    # collect first, THEN replace — mutating the tree while iterating it recurses forever
    to_wrap = [(parent, name, child)
               for parent in list(model.modules())
               for name, child in parent.named_children()
               if isinstance(child, nn.Linear)]
    for parent, name, child in to_wrap:
        setattr(parent, name, LoRALinear(child, r))
    return model


def main():
    torch.manual_seed(0)
    stoi, itos, V, block = build_vocab()
    examples = [make_example(i, r, stoi) for i, r in PAIRS]

    model = GPT(V, block)
    for p in model.parameters():
        p.requires_grad = False                          # freeze the whole base model
    apply_lora(model, r=8)                               # add trainable LoRA adapters

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable (LoRA only): {trainable:,} / {total:,} "
          f"({100 * trainable / total:.0f}% on this tiny model; well under 1% on a real one)")

    print(f"\nbefore SFT: {[respond(model, i, stoi, itos) for i, _ in PAIRS[:2]]}   (babble)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    for _ in range(600):
        loss = sum(F.cross_entropy(model(torch.tensor(x).unsqueeze(0)).view(-1, V),
                                   torch.tensor(y), ignore_index=-100) for x, y in examples) / len(examples)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    print(f"\nafter LoRA-SFT (loss {loss.item():.4f}):")
    correct = 0
    for i, r in PAIRS:
        got = respond(model, i, stoi, itos)
        correct += (got == r)
        print(f"  '{i}' -> {got!r}   {'✓' if got == r else '✗'}")
    print(f"\n✅ {correct}/{len(PAIRS)} correct — instruction-tuned with LoRA (the base never moved).")


if __name__ == "__main__":
    main()
