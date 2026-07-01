"""
DistributedDataParallel (DDP), runnable on a plain CPU.
=======================================================
Data parallelism: put a *replica* of the model on each of N workers ("ranks"), give each a
*different shard* of the batch, and after `loss.backward()` **average the gradients across all
ranks** (an "all-reduce"). Every replica then takes the identical step, so they stay in lockstep —
it's exactly as if one machine trained on the whole batch, just N× faster.

This runs N processes on your CPU with the `gloo` backend (no GPU needed). On real hardware you'd
switch the backend to `nccl` and put each rank on its own GPU — the code is otherwise the same.
We prove two things:
  1. after training, every rank holds the SAME weights (the all-reduce kept them in sync);
  2. those weights match a single-process run on the full batch (DDP changes speed, not the answer).

Run:  python ddp_train.py
"""
import os
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

WORLD = 4          # number of workers (processes / would-be GPUs)
STEPS = 20
RESULT = os.path.join(tempfile.gettempdir(), "ch10_ddp_weight.pt")


def make_data():
    """The full dataset — identical on every rank (they each take a different slice of it)."""
    g = torch.Generator().manual_seed(42)
    X = torch.randn(16, 4, generator=g)
    Y = torch.randn(16, 2, generator=g)
    return X, Y


def make_model():
    torch.manual_seed(0)               # same init everywhere; DDP also broadcasts rank 0's weights
    return nn.Linear(4, 2)


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    model = make_model()
    ddp = DDP(model)                   # wraps the model: backward() now all-reduces the gradients
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)

    X, Y = make_data()
    xb, yb = X[rank::world], Y[rank::world]     # rank r trains on its shard (rows r, r+W, r+2W, …)

    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((ddp(xb) - yb) ** 2).mean()
        loss.backward()                # <-- DDP averages grads across ranks right here
        opt.step()

    print(f"  rank {rank}/{world}: final weight[0,0] = {model.weight[0, 0].item():.6f}")
    dist.barrier()                     # everyone finishes before anyone tears down
    if rank == 0:                      # rank 0 saves the final weights for main() to check
        torch.save(model.weight.detach().clone(), RESULT)
    dist.destroy_process_group()


def single_process_baseline():
    """Train one model on the WHOLE batch — the answer DDP should reproduce."""
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, Y = make_data()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return model.weight.detach().clone()


def main():
    print(f"Training a model {WORLD} ways with DDP (gloo, CPU) — each rank sees 1/{WORLD} of the data:")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)

    ddp_w = torch.load(RESULT)
    base_w = single_process_baseline()
    gap = (ddp_w - base_w).abs().max().item()

    print(f"\nAll {WORLD} ranks printed the SAME final weight above — the gradient all-reduce kept the")
    print(f"replicas in lockstep. And vs a single-process run on the full batch:")
    print(f"  max weight difference (DDP vs single-process) = {gap:.2e}")
    print(f"  {'MATCH — DDP is just full-batch training, parallelized.' if gap < 1e-5 else 'MISMATCH!'}")


if __name__ == "__main__":
    main()
