"""
The KV-cache speedup — and how it grows with length.
====================================================
Naive generation recomputes every past token's keys/values on every step: total work ~O(T²).
The KV-cache computes each token's k/v once: total work ~O(T). So the *longer* you generate, the
bigger the win. This proves the two agree, then times them at growing lengths.

Run:  python benchmark.py        (a few seconds on a CPU)
"""
import time
import torch
from gpt import GPT, VOCAB


def timed(fn):
    fn()                                            # warm up (first call includes setup)
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


torch.manual_seed(0)
model = GPT().eval()
prompt = torch.randint(0, VOCAB, (1, 8))

# 1) correctness: the cache must not change the output
a = model.generate_naive(prompt.clone(), 100)
b = model.generate_cached(prompt.clone(), 100)
print(f"identical output (naive vs cached): {torch.equal(a, b)}\n")

# 2) speed, at growing generation lengths
print(f"{'tokens':>8} {'naive ms':>10} {'cached ms':>11} {'speedup':>9}")
for n in [64, 128, 256, 480]:                       # stay within the model's context window (BLOCK)
    tn = timed(lambda: model.generate_naive(prompt.clone(), n))
    tc = timed(lambda: model.generate_cached(prompt.clone(), n))
    print(f"{n:>8} {tn:>10.0f} {tc:>11.0f} {tn / tc:>8.1f}x")

print("\nThe speedup climbs with length: naive redoes O(T²) work, the cache only O(T).")
print("(On a GPU serving real chats, this is the difference between snappy and unusable.)")
