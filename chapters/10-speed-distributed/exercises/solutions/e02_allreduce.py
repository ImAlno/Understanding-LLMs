"""
E02 — A real all-reduce across processes.
=========================================
E01 simulated the average in one process. Here it's REAL: N processes, each with its own gradient,
combined with a gloo all-reduce so every rank ends up with the identical averaged gradient — exactly
what DDP does under the hood.

Run:  python e02_allreduce.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29507"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    # each rank pretends to have computed a different gradient
    grad = torch.tensor([float(rank), float(rank * 10)])
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad /= world                                   # SUM / N == average

    print(f"  rank {rank}: averaged gradient = {grad.tolist()}")   # every rank prints the SAME thing
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print(f"{WORLD} ranks, each with gradient [rank, rank*10]; all-reduce should give the mean on every rank:")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    print(f"\nExpected on every rank: [{sum(range(WORLD)) / WORLD}, {sum(r * 10 for r in range(WORLD)) / WORLD}]")
    print("All ranks agree — that's the gradient synchronization DDP does every step.")
