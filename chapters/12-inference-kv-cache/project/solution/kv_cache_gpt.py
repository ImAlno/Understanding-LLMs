"""
KV-Cache Your GPT — SOLUTION.
=============================
A self-contained GPT with two generators. The KV-cache lives in the attention (the two `torch.cat`
lines). We prove cached generation is identical to naive, then benchmark the tokens/second win.

Run:  python kv_cache_gpt.py
"""
import math
import time
import torch
import torch.nn as nn

VOCAB, C, NH, NL, BLOCK = 65, 128, 4, 3, 512
HD = C // NH


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.proj = (nn.Linear(C, C) for _ in range(4))

    def forward(self, x, cache, po):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, NH, HD).transpose(1, 2)
        k = self.k(x).view(B, T, NH, HD).transpose(1, 2)
        v = self.v(x).view(B, T, NH, HD).transpose(1, 2)
        if cache is not None:
            if cache["k"] is not None:                        # <-- the KV-cache: append new k/v to the past
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
        qp = torch.arange(po, po + T).view(T, 1); kp = torch.arange(Tk).view(1, Tk)
        att = att.masked_fill(kp > qp, float("-inf"))
        return self.proj((torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.at = Attn()
        self.ff = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

    def forward(self, x, c, po):
        x = x + self.at(self.ln1(x), c, po)
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, C); self.pos = nn.Embedding(BLOCK, C)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.lnf = nn.LayerNorm(C); self.head = nn.Linear(C, VOCAB)

    def forward(self, idx, caches=None, po=0):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(po, po + T))
        for b, c in zip(self.blocks, caches or [None] * NL):
            x = b(x, c, po)
        return self.head(self.lnf(x))

    @torch.no_grad()
    def generate_naive(self, idx, n):
        for _ in range(n):
            idx = torch.cat([idx, self(idx[:, -BLOCK:])[:, -1].argmax(-1, keepdim=True)], 1)
        return idx

    @torch.no_grad()
    def generate_cached(self, idx, n):
        caches = [{"k": None, "v": None} for _ in range(NL)]
        out = torch.cat([idx, self(idx, caches, 0)[:, -1].argmax(-1, keepdim=True)], 1)   # prefill
        for _ in range(n - 1):                                                            # decode
            out = torch.cat([out, self(out[:, -1:], caches, out.size(1) - 1)[:, -1].argmax(-1, keepdim=True)], 1)
        return out


def main():
    torch.manual_seed(0)
    model = GPT().eval()
    prompt = torch.randint(0, VOCAB, (1, 8))

    # 1) correctness: cached must equal naive
    a = model.generate_naive(prompt.clone(), 200)
    b = model.generate_cached(prompt.clone(), 200)
    print(f"identical output (naive vs cached): {torch.equal(a, b)}")
    assert torch.equal(a, b), "cache changed the output — the cache logic is wrong"

    # 2) tokens/second, cached
    n = 400
    for gen, name in [(model.generate_naive, "naive"), (model.generate_cached, "cached")]:
        gen(prompt.clone(), 8)                                   # warm up
        t0 = time.perf_counter(); gen(prompt.clone(), n); dt = time.perf_counter() - t0
        print(f"  {name:>6}: {n / dt:6.0f} tokens/sec  ({dt * 1000:.0f} ms for {n} tokens)")
    print("\n✅ Same output, more tokens/second — you KV-cached your GPT.")


if __name__ == "__main__":
    main()
