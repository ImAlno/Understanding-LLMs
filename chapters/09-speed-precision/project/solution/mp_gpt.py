"""
Mixed-Precision Your GPT — SOLUTION.
====================================
Train a (compact) GPT in fp32, then in mixed precision, from the same initialization. Measure the
loss parity, the time per step, and — on a CUDA GPU — the peak memory. The whole change is wrapping
the forward in `autocast` and routing backward/step through a `GradScaler`.

Run:  python mp_gpt.py        (best on Colab with Runtime -> GPU)
"""
import sys
import time
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
from gpt_small import GPT, random_batch


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


def train(device, steps=40, mixed=False, dtype=torch.bfloat16):
    torch.manual_seed(0)
    model = GPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # GradScaler matters only for fp16 (bf16 keeps fp32's range); a no-op otherwise.
    scaler = torch.amp.GradScaler(device, enabled=(mixed and dtype == torch.float16 and device == "cuda"))
    xb, yb = random_batch(32, device)

    def step():
        opt.zero_grad()
        # The whole mixed-precision change: run the forward under autocast.
        ctx = torch.autocast(device_type=device, dtype=dtype) if mixed else contextlib.nullcontext()
        with ctx:
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()   # scale up (fp16) then backward; no-op for bf16
        scaler.step(opt)                # unscale + step (skips the step if grads overflowed)
        scaler.update()
        return loss.item()

    step(); sync(device)                # warm up
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], time.perf_counter()
    for _ in range(steps):
        losses.append(step())
    sync(device)
    ms = (time.perf_counter() - t0) / steps * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    return ms, losses, peak_gb


def main():
    device = get_device()
    dtype = torch.bfloat16          # the modern default; on CUDA you could also use fp16 + a scaler
    print(f"device: {device}   mixed dtype: bf16\n")

    fp32_ms, fp32_losses, fp32_gb = train(device, mixed=False)
    mp_ms, mp_losses, mp_gb = train(device, mixed=True, dtype=dtype)

    print("loss parity (same init — mixed should track fp32):")
    print(f"{'step':>6} {'fp32':>10} {'mixed':>10}")
    for s in [0, len(fp32_losses) // 2, -1]:
        i = len(fp32_losses) - 1 if s == -1 else s
        print(f"{i:>6} {fp32_losses[i]:>10.4f} {mp_losses[i]:>10.4f}")
    gap = max(abs(a - b) for a, b in zip(fp32_losses, mp_losses))
    print(f"max loss gap over training: {gap:.4f}  (tiny — same curve)\n")

    print(f"{'fp32':>8}: {fp32_ms:6.2f} ms/step" + (f"   peak {fp32_gb:.2f} GB" if fp32_gb else ""))
    print(f"{'mixed':>8}: {mp_ms:6.2f} ms/step" + (f"   peak {mp_gb:.2f} GB" if mp_gb else ""))
    if device == "cuda":
        print(f"\nspeedup: {fp32_ms / mp_ms:.2f}x   |   memory: {mp_gb / fp32_gb:.0%} of fp32")
        print("Same loss curve, faster and lighter — that's the whole win.")
    else:
        print(f"\nspeedup here: {fp32_ms / mp_ms:.2f}x — on a CPU/Apple GPU there are no Tensor Cores,")
        print("so mixed precision is ~1x or even slower. The LOSS PARITY is the point you can see")
        print("locally; run on Colab (Runtime -> GPU) for the real speed and memory win.")


if __name__ == "__main__":
    main()
