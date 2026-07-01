"""
ZeRO: shard the training state so a model too big for one GPU still fits.
========================================================================
Plain data parallelism (DDP) replicates EVERYTHING on every GPU — weights, gradients, and optimizer
state — all identical across the N GPUs. That's hugely redundant. **ZeRO** shards those pieces
across the workers instead, so each GPU holds only 1/N of them:

  ZeRO-1: shard the optimizer state         (biggest win — Adam's state is the largest piece)
  ZeRO-2: + shard the gradients
  ZeRO-3: + shard the parameters themselves  (this is PyTorch's FSDP)

For a P-parameter model in mixed-precision Adam, the standard per-parameter accounting is 16 bytes:
  fp16 params (2) + fp16 grads (2) + fp32 master weights (4) + Adam m (4) + Adam v (4).
This computes the per-GPU memory at each stage.

Run:  python zero_memory.py
"""


def per_gpu_gb(P, N, stage):
    params = 2            # fp16 parameters
    grads = 2             # fp16 gradients
    optim = 4 + 4 + 4     # fp32 master weights + Adam's m + Adam's v
    if stage >= 1:
        optim /= N        # ZeRO-1: shard optimizer state across the N GPUs
    if stage >= 2:
        grads /= N        # ZeRO-2: also shard gradients
    if stage >= 3:
        params /= N       # ZeRO-3 (FSDP): also shard parameters
    return (params + grads + optim) * P / 1e9


P = 7.5e9    # a 7.5-billion-parameter model
N = 8        # 8 GPUs

print(f"{P/1e9:.1f}B-parameter model, {N} GPUs, mixed-precision Adam — memory PER GPU:")
for stage, name in [(0, "plain DDP (no sharding)"),
                    (1, "ZeRO-1 (optimizer state)"),
                    (2, "ZeRO-2 (+ gradients)"),
                    (3, "ZeRO-3 / FSDP (+ parameters)")]:
    print(f"  {name:>28}: {per_gpu_gb(P, N, stage):6.1f} GB")

print(f"\nPlain DDP needs {per_gpu_gb(P, N, 0):.0f} GB/GPU — impossible on a 40 GB card. ZeRO-3 brings")
print(f"it to {per_gpu_gb(P, N, 3):.0f} GB, so the SAME model now fits. Sharding trades a little extra")
print("communication for a big drop in per-GPU memory — that's how billion-parameter models train.")
