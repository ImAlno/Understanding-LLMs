"""
Your First GPT — STARTER scaffold.
==================================
Assemble the full GPT and train it. Fill the two TODOs. The Transformer blocks are already
built (in code/gpt.py) — your job is to wire them into a model and train it.
Reference: ../solution/my_gpt.py.

    python my_gpt.py        (~2 minutes on a CPU, once it's filled in)
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
        # ✍️ TODO #1: build x — add token embeddings and position embeddings (use torch.arange(T)),
        #             run x through self.blocks, then through self.ln_f
        x = None
        if x is None:
            raise NotImplementedError("Fill in TODO #1: the forward pass (embeddings → blocks → ln_f).")
        logits = self.lm_head(x)
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
print("training your GPT (~2 min, once it's filled in)...")
loss = None
for step in range(3000):
    # ✍️ TODO #2: one training step — get a "train" batch, forward through the model for the
    #             loss, zero the gradients, backward, optimizer step
    if loss is None:
        raise SystemExit("Fill in TODO #2 (the training step) — and TODO #1 — then run again.")
    if step % 500 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

sample = decode(model.generate(torch.zeros((1, 1), dtype=torch.long), 1000)[0].tolist())
print("\n--- your Shakespeare ---\n" + sample)
Path("my_first_gpt.txt").write_text(sample)
print("\n(saved to my_first_gpt.txt)")
