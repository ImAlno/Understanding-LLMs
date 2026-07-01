# Chapter 10 — Exercises

Feel how distributed training works. **Try first**; solutions are in [`solutions/`](./solutions/).
Everything runs on a **plain CPU** (the `gloo` backend and single-process simulations) — no GPU
needed. E02 launches real processes and takes a few seconds to start.

```bash
python chapters/10-speed-distributed/exercises/solutions/e01_average_grads.py
```

---

### E01 — Average the gradients 🥇
The heart of DDP: prove (in one process) that averaging the shard-gradients equals the whole-batch
gradient — so DDP gives the same answer, in parallel.

- 🧩 **Scaffold:** [`starter/e01_average_grads.py`](./starter/e01_average_grads.py).
- ✅ **Solution:** [`solutions/e01_average_grads.py`](./solutions/e01_average_grads.py).

  <details><summary>💡 Hint</summary>

  `averaged = torch.stack(shard_grads).mean(dim=0)` — average *across* the ranks (dim 0).
  </details>

### E02 — A real all-reduce across processes
The same average, for real: N `gloo` processes each hold a gradient; an `all_reduce` gives every
rank the identical mean. This is exactly what DDP does under `loss.backward()`.

- ✅ Solution: [`solutions/e02_allreduce.py`](./solutions/e02_allreduce.py).

  <details><summary>💡 Hint</summary>

  `dist.all_reduce(t, op=dist.ReduceOp.SUM)` then `t /= world` — SUM ÷ N is the average.
  </details>

### E03 — Catch the `DistributedSampler` bug
The #1 DDP mistake: forgetting to shard the data, so every rank trains on the *same* examples.
Compare "no sampler" (100% overlap, no speedup) with proper `r::N` sharding (disjoint, complete).

- ✅ Solution: [`solutions/e03_sampler.py`](./solutions/e03_sampler.py).

  <details><summary>💡 Hint</summary>

  Rank `r`'s shard is `dataset[r::N]`. Check the shards are pairwise-disjoint and together cover the
  whole dataset.
  </details>

### E04 — A ZeRO memory table
Compute per-GPU memory across ZeRO stages for several model sizes, and see which become trainable at
each stage (and which still need more GPUs).

- ✅ Solution: [`solutions/e04_zero_table.py`](./solutions/e04_zero_table.py).

  <details><summary>💡 Hint</summary>

  16 bytes/param (2 fp16 params + 2 fp16 grads + 12 optimizer). Divide `optim` by N at ZeRO-1,
  also `grads` at ZeRO-2, also `params` at ZeRO-3.
  </details>

> 🧠 **Takeaway:** DDP = average the shard-gradients (exact); a `DistributedSampler` keeps the
> shards disjoint; ZeRO shards the redundant state so big models fit. All three run on your laptop.
