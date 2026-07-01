"""
E03 — The speedup grows with length.
====================================
Naive generation is O(T²) (recompute all past k/v every step); cached is O(T). So the longer you
generate, the bigger the win. Time both at growing lengths and watch the ratio climb.

Run:  python e03_speedup.py        (a few seconds)
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
from gpt import GPT, VOCAB


def ms(fn):
    fn()                                        # warm up
    t0 = time.perf_counter(); fn(); return (time.perf_counter() - t0) * 1000


torch.manual_seed(0)
model = GPT().eval()
prompt = torch.randint(0, VOCAB, (1, 8))

print(f"{'tokens':>8} {'naive ms':>10} {'cached ms':>11} {'speedup':>9}")
prev = None
for n in [64, 128, 256, 480]:
    tn = ms(lambda: model.generate_naive(prompt.clone(), n))
    tc = ms(lambda: model.generate_cached(prompt.clone(), n))
    print(f"{n:>8} {tn:>10.0f} {tc:>11.0f} {tn / tc:>8.1f}x")
    prev = tn / tc

print("\nThe speedup rises with length — because naive redoes O(T²) work and the cache only O(T).")
