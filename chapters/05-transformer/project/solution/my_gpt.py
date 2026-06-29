"""
Your First GPT — REFERENCE SOLUTION.
====================================
Assemble the Transformer blocks (from code/gpt.py) into a full GPT, train it on Shakespeare,
and generate a paragraph of your own.

    python my_gpt.py        (~2 minutes on a CPU)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpt as G                       # data, vocab, get_batch, decode, and the Block you built

vocab_size, block_size = G.vocab_size, G.block_size
n_embd, n_head, n_layer = G.n_embd, G.n_head, G.n_layer
get_batch, decode = G.get_batch, G.decode


class MyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[G.Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


torch.manual_seed(1337)
model = MyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"training your GPT ({sum(p.nelement() for p in model.parameters())/1e3:.0f}K params, ~2 min)...")
for step in range(3000):
    _, loss = model(*get_batch("train"))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

sample = decode(model.generate(torch.zeros((1, 1), dtype=torch.long), 1000)[0].tolist())
print("\n--- your Shakespeare ---\n" + sample)
Path("my_first_gpt.txt").write_text(sample)
print("\n(saved to my_first_gpt.txt)")
