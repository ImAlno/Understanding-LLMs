"""
E02 — Cached generation is bit-for-bit identical to naive.
=========================================================
The KV-cache is an optimization, not an approximation. On a full GPT, greedy generation with the
cache produces the EXACT same tokens as the naive loop — that's the correctness test.

Run:  python e02_identical.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
from gpt import GPT, VOCAB

torch.manual_seed(0)
model = GPT().eval()

for prompt_len in [1, 8, 20]:
    prompt = torch.randint(0, VOCAB, (1, prompt_len))
    a = model.generate_naive(prompt.clone(), 100)
    b = model.generate_cached(prompt.clone(), 100)
    same = torch.equal(a, b)
    print(f"prompt of {prompt_len:>2} tokens -> generate 100:  naive == cached: {same}")
    assert same

print("\nIdentical every time — the cache changes the *work*, never the *answer*.")
