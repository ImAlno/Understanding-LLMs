"""
E01 — Train a reward model with the Bradley-Terry loss (STARTER).
================================================================
Fill in `bt_loss`: given the reward model `rm`, return the Bradley-Terry loss that pushes the CHOSEN
reply's score above the REJECTED reply's. Reference solution: ../solutions/e01_bradley_terry.py.

Run:  python e01_bradley_terry.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn.functional as F
from gpt import PREFS, RewardModel, reward, build_vocab

stoi, itos, V, block = build_vocab()


def bt_loss(rm, prompt, chosen, rejected):
    # ✍️ TODO: return -log sigmoid( score(chosen) - score(rejected) )
    #   score a reply with:  reward(rm, prompt, response, stoi)
    #   use F.logsigmoid(...)
    raise NotImplementedError("Implement the Bradley-Terry loss, then delete this line.")


torch.manual_seed(1)
rm = RewardModel(V, block)
opt = torch.optim.AdamW(rm.parameters(), lr=3e-3)
for _ in range(300):
    loss = sum(bt_loss(rm, p, c, r) for p, c, r in PREFS) / len(PREFS)
    opt.zero_grad(); loss.backward(); opt.step()
rm.eval()

with torch.no_grad():
    margins = [(reward(rm, p, c, stoi) - reward(rm, p, r, stoi)).item() for p, c, r in PREFS]
acc = sum(m > 0 for m in margins) / len(margins)
print(f"reward-model accuracy: {acc * 100:.0f}%   |   mean margin (chosen - rejected): {sum(margins) / len(margins):+.2f}")
print("✓ Bradley-Terry taught the reward model to rank chosen above rejected."
      if acc == 1 else "✗ Check the loss: -log sigmoid(score(chosen) - score(rejected)).")
