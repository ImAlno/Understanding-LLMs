"""
E04 — A ZeRO memory table across model sizes.
=============================================
ZeRO shards the training state (optimizer → gradients → parameters) across N GPUs. Compute the
per-GPU memory for several model sizes and watch which ones become trainable at each stage.

Run:  python e04_zero_table.py
"""


def per_gpu_gb(P, N, stage):
    params, grads, optim = 2, 2, 12    # bytes/param: fp16 params, fp16 grads, (fp32 master + Adam m + v)
    if stage >= 1:
        optim /= N
    if stage >= 2:
        grads /= N
    if stage >= 3:
        params /= N
    return (params + grads + optim) * P / 1e9


N = 8
GPU_MEM = 80        # an 80 GB card
stages = [(0, "DDP"), (1, "ZeRO-1"), (2, "ZeRO-2"), (3, "ZeRO-3")]

print(f"Per-GPU memory (GB) on {N} GPUs — ✓ = fits on an {GPU_MEM} GB card:\n")
print(f"{'model':>8} " + " ".join(f"{name:>10}" for _, name in stages))
for P in [1e9, 7.5e9, 30e9, 70e9]:
    cells = []
    for stage, _ in stages:
        gb = per_gpu_gb(P, N, stage)
        cells.append(f"{gb:6.0f}{'✓' if gb <= GPU_MEM else '✗'} ")
    print(f"{P/1e9:>5g}B  " + " ".join(f"{c:>10}" for c in cells))

print(f"\nBigger models need higher ZeRO stages to fit. Note the optimizer state (ZeRO-1) is the")
print("biggest single saving — 12 of the 16 bytes/param — so it's the first and cheapest knob.")
