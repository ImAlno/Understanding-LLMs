"""
E01 — Train a reward model with the Bradley-Terry loss.
=======================================================
A reward model turns preference *comparisons* into a *score*. Implement the Bradley-Terry loss so the
model scores the chosen reply above the rejected one, then train it and check it ranks all pairs right.

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
    # Bradley-Terry: maximize sigmoid(score(chosen) - score(rejected)) == minimize -log of it
    return -F.logsigmoid(reward(rm, prompt, chosen, stoi) - reward(rm, prompt, rejected, stoi))


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
print("✓ Bradley-Terry taught the reward model to rank chosen above rejected — a score RL can climb."
      if acc == 1 else "✗ Check the loss: it should be -log sigmoid(score(chosen) - score(rejected)).")
