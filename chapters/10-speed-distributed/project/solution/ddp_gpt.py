"""
DDP Your GPT — SOLUTION.
========================
Train the tiny GPT with DistributedDataParallel across N processes, and prove the loss curve is
identical to a single process training on the whole batch — because DDP averages the shard
gradients, which equals the full-batch gradient. (gloo/CPU here; swap to nccl/GPU for the speedup.)

Run:  python ddp_gpt.py
"""
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt_tiny import GPT, new_model, full_batch

WORLD = 4
STEPS = 30
RESULT = os.path.join(tempfile.gettempdir(), "ch10_ddp_gpt_traj.pt")


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29508"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    model = new_model()                       # same seed -> identical init on every rank
    ddp = DDP(model)                          # gradients now all-reduce inside backward()
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    X, Y = full_batch()
    xb, yb = X[rank::world], Y[rank::world]   # this rank's disjoint shard

    traj = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ddp(xb, yb)
        loss.backward()                       # <-- shard grads averaged across ranks here
        opt.step()
        if rank == 0:                         # rank 0 logs the FULL-batch loss of the synced model
            with torch.no_grad():
                traj.append(model(X, Y).item())

    dist.barrier()
    if rank == 0:
        torch.save(traj, RESULT)
    dist.destroy_process_group()


def single_process():
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    X, Y = full_batch()
    traj = []
    for _ in range(STEPS):
        opt.zero_grad()
        model(X, Y).backward()
        opt.step()
        with torch.no_grad():                 # log the SAME thing the DDP run does: post-step full-batch loss
            traj.append(model(X, Y).item())
    return traj


def main():
    print(f"Training the tiny GPT with DDP across {WORLD} ranks (gloo, CPU)...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    ddp_traj = torch.load(RESULT)
    base_traj = single_process()

    gap = max(abs(a - b) for a, b in zip(ddp_traj, base_traj))
    print(f"\n{'step':>6} {'DDP loss':>10} {'1-process':>11}")
    for s in [0, STEPS // 2, STEPS - 1]:
        print(f"{s:>6} {ddp_traj[s]:>10.4f} {base_traj[s]:>11.4f}")
    print(f"\nmax gap over training: {gap:.2e}")
    print("MATCH — {} ranks give the identical loss curve as one process on the full batch.".format(WORLD)
          if gap < 1e-4 else "mismatch!")


if __name__ == "__main__":
    main()
