# 🌐 Mini-Project — DDP Your GPT

You've seen the pieces; now **wrap a real GPT in DistributedDataParallel** and prove the payoff that
makes distributed training trustworthy: `N` workers produce the *identical* loss curve as one worker
on the whole batch — just (on real hardware) `N×` faster.

> 💻 **No GPU needed.** This runs `N` real processes on your CPU with the `gloo` backend. On real
> GPUs you'd swap to `nccl` and launch with `torchrun`; the code is otherwise identical. The one
> thing a laptop can't show is the *speedup* (that needs `N` real GPUs) — but the **loss parity** is
> fully visible here.

> **How this works:** [`starter/ddp_gpt.py`](./starter/ddp_gpt.py) trains the tiny GPT from
> [`gpt_tiny.py`](./gpt_tiny.py) across 4 processes and has **one TODO** — wrap the model in DDP.
> A reference is in [`solution/ddp_gpt.py`](./solution/ddp_gpt.py).

## 🎯 What it does

```bash
python starter/ddp_gpt.py
```

Spawns 4 ranks, gives each a disjoint shard of the batch, trains for 30 steps, and compares rank 0's
full-batch loss curve against a single process trained on the whole batch. If DDP is wired correctly,
the two curves are **identical** (max gap ~5e-7) — because averaging the shard-gradients equals the
full-batch gradient.

## 🛠️ Your TODO

In `worker()`, wrap the model so its gradients synchronize across ranks:

```python
ddp = DDP(model)          # instead of `ddp = model`
```

The data is already sharded (`X[rank::world]`). Until you wrap in DDP, the ranks train independently
on their shards, the curves *don't* match, and the script tells you to fill the TODO.

## ✅ Checking your work

- **Loss parity** (visible on CPU): the DDP curve should match the single-process curve to ~1e-7.
  That's the guarantee — DDP changes the *speed*, never the *answer*.
- **Unfilled → refusal:** without the DDP wrap, rank 0 trains on only 1/4 of the data, the curves
  diverge, and the starter exits asking you to fix it (no false success).
- Compare against [`solution/ddp_gpt.py`](./solution/ddp_gpt.py).

## 🚀 Stretch

- **Launch with `torchrun`.** Rewrite the script to read `rank`/`world_size` from the environment
  (`int(os.environ["RANK"])`, `int(os.environ["WORLD_SIZE"])`) and run it with
  `torchrun --nproc_per_node=4 ddp_gpt.py` — the way real jobs start (no `mp.spawn`).
- **Add a real `DistributedSampler`.** Feed the GPT batches through a `DataLoader` +
  `DistributedSampler` instead of manual `X[rank::world]` slicing, and call `sampler.set_epoch(e)`.
- **Scale the model up and try FSDP.** Grow `gpt_tiny.py`, then swap `DDP` for
  `torch.distributed.fsdp.FullyShardedDataParallel` (ZeRO-3) and confirm training still matches.
- **On real GPUs:** switch the backend to `nccl`, put each rank on its own GPU, and *time* it — now
  you'll see the `N×` speedup the CPU can't show.

Next: [Chapter 11 — Datasets](../../11-datasets/), where we turn from *training fast* to *what to
feed the model*. 🍽️
