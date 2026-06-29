"""
The Attention Microscope — REFERENCE SOLUTION.
==============================================
Train a single-head model, then see which earlier characters it attends to.

    python attention_microscope.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import attention as A

vocab_size, train_data, val_data, encode = A.vocab_size, A.train_data, A.val_data, A.encode
torch.manual_seed(1337)
block_size, n_embd, batch_size, steps = 32, 64, 32, 3000


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    return (torch.stack([d[i:i + block_size] for i in ix]),
            torch.stack([d[i + 1:i + block_size + 1] for i in ix]))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.last_wei = None

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()          # stash the weights so the microscope can read them
        return wei @ self.value(x)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.head = Head()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.lm_head(self.head(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


model = LM()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---- TODO #1: the training loop ----
print(f"training the microscope's model ({steps} steps)...")
for step in range(steps):
    _, loss = model(*get_batch("train"))
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"done (loss {loss.item():.3f})\n")


@torch.no_grad()
def microscope(prompt):
    ids = encode(prompt)[-block_size:]
    model(torch.tensor([ids]))                       # one forward pass populates head.last_wei
    # ---- TODO #2: the weights of the LAST position (what the final char attends to) ----
    wei = model.head.last_wei[0, -1]                 # (T,)
    print(f'prompt: "{prompt}"  (predicting the next character)')
    for ch, w in zip(prompt[-len(ids):], wei.tolist()):
        label = "\\n" if ch == "\n" else ch
        print(f"  {label!r:6} {w:.3f} {'█' * round(w * 60)}")
    print()


for p in ["To be, or not to b", "First Citizen", "ROMEO:\nWherefore art thou Rome"]:
    microscope(p)
