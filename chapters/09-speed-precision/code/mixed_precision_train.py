"""
Mixed-precision training: same model, half the bits.
====================================================
"Mixed precision" = do the heavy matmuls in a fast 16-bit format (bf16/fp16) while keeping a
full-precision fp32 copy of the weights for the optimizer. PyTorch automates it with two tools:

  • torch.autocast(...)  — a context manager: inside it, each op runs in the best dtype
                           automatically (matmuls in 16-bit, sensitive reductions in fp32).
  • torch.amp.GradScaler — only for fp16: multiplies the loss up so tiny gradients don't
                           underflow to zero, then unscales before the optimizer step.
                           bf16 has fp32's range, so it needs NO scaler.

This trains a small matmul-heavy model in fp32, then in mixed precision, and compares time + loss.
The SPEEDUP needs an NVIDIA GPU's Tensor Cores (run on Colab) — on a CPU/Apple GPU the loop still
runs and the loss still matches, you just won't see the wall-clock win.

Run:  python mixed_precision_train.py        (locally, or on Colab with Runtime -> GPU)
"""
import time
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_model_and_data(device):
    """A matmul-heavy MLP — wide layers, so 16-bit matmuls have something to speed up."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2048, 2048), nn.ReLU(),
        nn.Linear(2048, 2048), nn.ReLU(),
        nn.Linear(2048, 10),
    ).to(device)
    x = torch.randn(1024, 2048, device=device)
    y = torch.randint(0, 10, (1024,), device=device)
    return model, x, y


def train(device, steps=30, mixed=False, dtype=torch.bfloat16):
    """Run `steps` of training; return (ms per step, per-step loss list). `mixed=True` wraps the
    forward pass in autocast. A GradScaler is used only for fp16 on CUDA (a no-op otherwise)."""
    model, x, y = make_model_and_data(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5)   # gentle, so the loss curve is visible
    loss_fn = nn.CrossEntropyLoss()

    use_scaler = mixed and dtype == torch.float16 and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    def one_step():
        opt.zero_grad()
        # autocast: inside here, matmuls run in `dtype`; CrossEntropy stays fp32 automatically.
        with torch.autocast(device_type=device, dtype=dtype, enabled=mixed):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()   # scale up (fp16) so small grads survive; no-op otherwise
        scaler.step(opt)                # unscales, then steps (skips the step if grads overflowed)
        scaler.update()                 # adjust the scale factor for next time
        return loss.item()

    one_step(); sync(device)            # warm up (first step includes setup/compilation)
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        losses.append(one_step())
    sync(device)
    return (time.perf_counter() - t0) / steps * 1000, losses


def main():
    device = get_device()
    # bf16 is the modern default (no scaler needed); on CUDA you could also use fp16 + a scaler.
    mixed_dtype = torch.bfloat16
    print(f"device: {device}    mixed-precision dtype: {str(mixed_dtype).split('.')[-1]}")
    if device == "cpu":
        print("(No GPU here — the loop runs and the loss tracks fp32, but the speedup needs an")
        print(" NVIDIA GPU's Tensor Cores. Run on Colab with Runtime -> GPU to feel it.)")

    fp32_ms, fp32_losses = train(device, mixed=False)
    mp_ms, mp_losses = train(device, mixed=True, dtype=mixed_dtype)

    # Starting from the same seed, the two loss curves should track almost exactly.
    print(f"\nLoss curves (same init) — mixed precision tracks fp32 step for step:")
    print(f"{'step':>6} {'fp32 loss':>12} {'mixed loss':>12}")
    for s in [0, len(fp32_losses) // 4, len(fp32_losses) // 2, 3 * len(fp32_losses) // 4, -1]:
        label = len(fp32_losses) - 1 if s == -1 else s
        print(f"{label:>6} {fp32_losses[s]:>12.4f} {mp_losses[s]:>12.4f}")

    print(f"\n{'fp32 (baseline)':>16}: {fp32_ms:6.2f} ms/step")
    print(f"{'mixed precision':>16}: {mp_ms:6.2f} ms/step")
    print(f"{'speedup':>16}: {fp32_ms / mp_ms:6.2f}x   (big on a CUDA GPU; ~1x on CPU/Apple GPU)")


if __name__ == "__main__":
    main()
