"""
A tiny self-contained GPT for the DDP project (Chapter 5 architecture, shrunk).
==============================================================================
Small and dependency-free so the project can focus on the *distributed* wiring, not the model.
Import GPT and full_batch from here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK = 65, 64, 4, 2, 32


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK) * float("-inf"), diagonal=1))

    def forward(self, x):
        T = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T])
        x = x + a
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB)

    def forward(self, idx, targets):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        return F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))


def full_batch(device="cpu"):
    """The whole (fixed) batch — every rank makes the same one, then takes its own slice."""
    g = torch.Generator().manual_seed(123)
    idx = torch.randint(0, VOCAB, (32, BLOCK), generator=g).to(device)
    targets = torch.randint(0, VOCAB, (32, BLOCK), generator=g).to(device)
    return idx, targets


def new_model():
    torch.manual_seed(0)
    return GPT()


if __name__ == "__main__":
    m = new_model()
    xb, yb = full_batch()
    print(f"tiny GPT: {sum(p.numel() for p in m.parameters()):,} params | init loss {m(xb, yb).item():.3f}")
