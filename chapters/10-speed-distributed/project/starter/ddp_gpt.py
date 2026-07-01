"""
DDP Your GPT — STARTER scaffold.
================================
Fill the one TODO: wrap the model in DistributedDataParallel so the gradients synchronize across
ranks. The data is already sharded for you. Reference: ../solution/ddp_gpt.py.

Run:  python ddp_gpt.py
"""
import os
import tempfile
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt_tiny import GPT, new_model, full_batch

WORLD = 4
STEPS = 30
RESULT = os.path.join(tempfile.gettempdir(), "ch10_ddp_gpt_starter.pt")


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29509"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    model = new_model()
    # ✍️ TODO: wrap `model` in DistributedDataParallel so backward() all-reduces the gradients.
    ddp = model            # replace with: DDP(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    X, Y = full_batch()
    xb, yb = X[rank::world], Y[rank::world]     # this rank's disjoint shard (already done for you)

    traj = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ddp(xb, yb)
        loss.backward()
        opt.step()
        if rank == 0:
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
        with torch.no_grad():
            traj.append(model(X, Y).item())
    return traj


def main():
    print(f"Training the tiny GPT across {WORLD} ranks (gloo, CPU)...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    ddp_traj = torch.load(RESULT)
    base_traj = single_process()
    gap = max(abs(a - b) for a, b in zip(ddp_traj, base_traj))

    if gap > 1e-4:
        print(f"\nDDP loss {ddp_traj[-1]:.4f} vs single-process {base_traj[-1]:.4f} — they DON'T match "
              f"(gap {gap:.3f}).")
        raise SystemExit("Gradients aren't syncing — fill the ✍️ TODO (wrap the model in DDP), then re-run.")
    print(f"\nDDP loss curve matches single-process (max gap {gap:.2e}) — {WORLD} ranks, one answer. ✅")


if __name__ == "__main__":
    main()
