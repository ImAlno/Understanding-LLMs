"""
E03 — The #1 DDP bug: forgetting to shard the data.
====================================================
DDP only speeds things up if each rank sees a DIFFERENT slice of the data. Forget the
`DistributedSampler` and every rank iterates the whole dataset — so you compute the same gradient
N times, get no speedup, and silently use the wrong effective batch. This shows the difference.

Run:  python e03_sampler.py
"""
N = 4
dataset = list(range(16))          # 16 example indices

# WRONG — no sharding: every rank iterates the whole dataset
wrong = [set(dataset) for _ in range(N)]

# RIGHT — DistributedSampler-style: rank r takes indices r, r+N, r+2N, …
right = [set(dataset[r::N]) for r in range(N)]


def report(name, shards):
    sizes = [len(s) for s in shards]
    # do any two ranks share an index?
    overlap = any(shards[i] & shards[j] for i in range(N) for j in range(i + 1, N))
    covered = set().union(*shards) == set(dataset)
    print(f"{name}: each rank sees {sizes} examples | overlap between ranks: {overlap} | covers all 16: {covered}")


print(f"dataset of {len(dataset)} examples, {N} ranks:\n")
report("WITHOUT sampler", wrong)
report("WITH sampler   ", right)
print("\nWithout a sampler every rank sees all 16 (100% overlap) — N× redundant work, no speedup.")
print("With one, the 16 split into disjoint shards of 4 that together cover everything. Always shard.")
