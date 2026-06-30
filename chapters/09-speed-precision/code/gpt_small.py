"""
A compact, self-contained GPT — the Chapter 5 architecture, shrunk.
==================================================================
Same pieces you built in Chapter 5 (token + position embeddings → Transformer blocks → lm_head),
just small and dependency-free (no dataset to load) so the Chapter 9 project can focus on the
*mixed-precision* wiring rather than the model. Import `GPT` and `random_batch` from here.

Run:  python gpt_small.py     # a quick sanity check that it trains
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK = 65, 128, 4, 2, 64


class Block(nn.Module):
    """Pre-norm Transformer block: communicate (attention) then think (feed-forward), each residual."""
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK) * float("-inf"), diagonal=1))

    def forward(self, x):
        T = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T])   # causal self-attention
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

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.head(self.ln_f(self.blocks(x)))
        if targets is None:
            return x, None
        loss = F.cross_entropy(x.reshape(-1, VOCAB), targets.reshape(-1))
        return x, loss


def random_batch(batch_size, device):
    """A batch of random token sequences and their next-token targets (no dataset needed)."""
    idx = torch.randint(0, VOCAB, (batch_size, BLOCK), device=device)
    targets = torch.randint(0, VOCAB, (batch_size, BLOCK), device=device)
    return idx, targets


if __name__ == "__main__":
    torch.manual_seed(0)
    m = GPT()
    xb, yb = random_batch(8, "cpu")
    _, loss = m(xb, yb)
    print(f"GPT params: {sum(p.numel() for p in m.parameters()):,}")
    print(f"one forward: loss {loss.item():.3f}  (≈ ln(65) = {torch.log(torch.tensor(65.0)).item():.3f} at init)")
