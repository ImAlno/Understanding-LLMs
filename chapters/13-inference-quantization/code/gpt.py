"""
A compact GPT to quantize (Chapter 5 architecture, shrunk).
==========================================================
Small and dependency-free so the quantization scripts have a real model to compress. Import `GPT`
and `random_batch` from here. (Random init is fine — we measure how much quantization *drifts* the
output, which doesn't need a trained model.)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK = 65, 128, 4, 3, 64
HEAD_DIM = N_EMBD // N_HEAD


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.proj = (nn.Linear(N_EMBD, N_EMBD) for _ in range(4))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = (lin(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2) for lin in (self.q, self.k, self.v))
        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM) + mask, dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, N_EMBD))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = Attention()
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB)

    def forward(self, idx):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def random_batch(batch_size=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch_size, BLOCK), generator=g)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = GPT().eval()
    n = sum(p.numel() for p in m.parameters())
    n_lin = sum(p.numel() for mod in m.modules() if isinstance(mod, nn.Linear) for p in mod.parameters())
    print(f"GPT: {n:,} params, of which {n_lin:,} ({100 * n_lin / n:.0f}%) are in Linear layers (what we quantize)")
