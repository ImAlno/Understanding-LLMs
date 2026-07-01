"""
A compact GPT that can generate two ways: naive, and with a KV-cache.
=====================================================================
Same Chapter 5 architecture (token+pos embeddings → blocks → head), but with hand-written attention
so the **KV-cache** is visible. The model exposes two generators:

  • generate_naive  — the obvious way: every step, run the WHOLE sequence-so-far through the model,
                      recomputing the keys/values of every past token from scratch. O(T²) total.
  • generate_cached — cache each token's key and value the first time it's computed, so each step
                      only computes the ONE new token's q/k/v and attends over the cache. O(T).

Both produce the *identical* tokens (greedy decoding) — the cache is a pure speed optimization.

Run:  python gpt.py        # sanity check: the two generators agree
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, N_EMBD, N_HEAD, N_LAYER, BLOCK = 65, 128, 4, 3, 512
HEAD_DIM = N_EMBD // N_HEAD


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(N_EMBD, N_EMBD)
        self.k = nn.Linear(N_EMBD, N_EMBD)
        self.v = nn.Linear(N_EMBD, N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x, cache, pos_offset):
        """x: (B, T, C). `cache` is None (no caching) or a dict {'k','v'} to append to and reuse.
        `pos_offset` is the absolute position of x's first token (for the causal mask)."""
        B, T, _ = x.shape
        # project to per-head queries/keys/values, shape (B, n_head, T, head_dim)
        q = self.q(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        if cache is not None:
            if cache["k"] is not None:                       # append the new k/v to the cached past
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v                    # cache now holds ALL keys/values so far

        Tk = k.size(2)                                       # total keys = past (cached) + current
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)   # (B, n_head, T, Tk)
        # causal mask: query at absolute position (pos_offset+i) may attend to keys at 0..that pos.
        qpos = torch.arange(pos_offset, pos_offset + T).view(T, 1)
        kpos = torch.arange(Tk).view(1, Tk)
        att = att.masked_fill(kpos > qpos, float("-inf"))
        out = F.softmax(att, dim=-1) @ v                     # (B, n_head, T, head_dim)
        return self.proj(out.transpose(1, 2).reshape(B, T, N_EMBD))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = Attention()
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x, cache, pos_offset):
        x = x + self.attn(self.ln1(x), cache, pos_offset)
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB)

    def forward(self, idx, caches=None, pos_offset=0):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(pos_offset, pos_offset + T))
        for block, cache in zip(self.blocks, caches or [None] * N_LAYER):
            x = block(x, cache, pos_offset)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate_naive(self, idx, n_new):
        """Each step re-runs the whole sequence through the model (recomputing all past k/v)."""
        for _ in range(n_new):
            logits = self(idx[:, -BLOCK:])                   # full forward over everything so far
            nxt = logits[:, -1].argmax(-1, keepdim=True)     # greedy: pick the top next token
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def generate_cached(self, idx, n_new):
        """PREFILL the prompt once (filling the cache), then DECODE one token at a time using it."""
        caches = [{"k": None, "v": None} for _ in range(N_LAYER)]
        logits = self(idx, caches, pos_offset=0)             # prefill: process the whole prompt once
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([idx, nxt], dim=1)
        for _ in range(n_new - 1):                           # decode: feed only the newest token
            logits = self(out[:, -1:], caches, pos_offset=out.size(1) - 1)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GPT().eval()
    prompt = torch.randint(0, VOCAB, (1, 8))
    a = model.generate_naive(prompt.clone(), 120)
    b = model.generate_cached(prompt.clone(), 120)
    print(f"naive and cached generate identical tokens: {torch.equal(a, b)}  (both {a.size(1)} long)")
    print("The KV-cache is a pure speed optimization — same output, less work.")
