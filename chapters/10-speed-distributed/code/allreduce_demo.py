"""
Collective operations: all-reduce and broadcast.
================================================
Distributed training is built on a few "collective" operations — ways for N workers to combine
their tensors. The two that matter most:

  • all-reduce : every rank contributes a tensor; every rank gets back the combined result
                 (SUM or AVERAGE). This is how DDP averages gradients.
  • broadcast  : rank 0's tensor is copied to every other rank. This is how DDP starts all
                 replicas from the same weights.

Runs N CPU processes with the `gloo` backend (no GPU). Real GPU training uses `nccl`; same code.

Run:  python allreduce_demo.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    # 1) all-reduce SUM: each rank starts with its own value; all end with the total.
    t = torch.tensor([float(rank + 1)])          # ranks hold 1, 2, 3, 4
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"all_reduce SUM of [1,2,3,4] -> every rank now has {t.item()}  (= 1+2+3+4)")

    # 2) all-reduce AVG: the same, divided by world size — exactly how DDP combines gradients.
    g = torch.tensor([float(rank + 1)])
    dist.all_reduce(g, op=dist.ReduceOp.SUM)
    g /= world                                   # SUM then / N  ==  average
    if rank == 0:
        print(f"all_reduce AVG of [1,2,3,4] -> every rank now has {g.item()}  (= the mean, 2.5)")

    # 3) broadcast: rank 0's value is copied to everyone (how replicas start in sync).
    b = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
    dist.broadcast(b, src=0)
    print(f"  rank {rank}: after broadcast from rank 0, my value = {b.item()}")

    dist.barrier()                               # all finish before teardown (avoids a shutdown race)
    dist.destroy_process_group()


if __name__ == "__main__":
    print(f"Launching {WORLD} workers (gloo, CPU)...\n")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    print("\nEvery worker ended up with the same combined values — that's the whole basis of DDP:")
    print("broadcast the weights once, then all-reduce (average) the gradients every step.")
