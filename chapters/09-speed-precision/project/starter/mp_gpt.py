"""
Mixed-Precision Your GPT — STARTER scaffold.
============================================
Fill the one TODO: wrap the forward pass in `autocast` when training in mixed precision. The
GradScaler is already wired (a no-op for bf16). Reference: ../solution/mp_gpt.py.

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
    scaler = torch.amp.GradScaler(device, enabled=(mixed and dtype == torch.float16 and device == "cuda"))
    xb, yb = random_batch(32, device)
    seen = {"dtype": None}

    def step():
        opt.zero_grad()
        # ✍️ TODO: when `mixed` is True, make `ctx` a torch.autocast block
        #          (device_type=device, dtype=dtype); otherwise keep the no-op nullcontext.
        ctx = contextlib.nullcontext()      # replace
        with ctx:
            logits, loss = model(xb, yb)
        if seen["dtype"] is None:
            seen["dtype"] = logits.dtype
        scaler.scale(loss).backward()       # (already wired) scale up for fp16, no-op for bf16
        scaler.step(opt)
        scaler.update()
        return loss.item()

    step(); sync(device)                    # warm up
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], time.perf_counter()
    for _ in range(steps):
        losses.append(step())
    sync(device)
    ms = (time.perf_counter() - t0) / steps * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    return ms, losses, peak_gb, seen["dtype"]


def main():
    device = get_device()
    print(f"device: {device}   mixed dtype: bf16\n")

    fp32_ms, fp32_losses, fp32_gb, _ = train(device, mixed=False)
    mp_ms, mp_losses, mp_gb, mp_dtype = train(device, mixed=True)

    if mp_dtype == torch.float32:
        raise SystemExit("Mixed precision isn't on yet — fill the ✍️ TODO in train() (wrap the "
                         "forward in torch.autocast when `mixed`), then run again.")

    print("loss parity (same init — mixed should track fp32):")
    gap = max(abs(a - b) for a, b in zip(fp32_losses, mp_losses))
    print(f"  fp32  {fp32_losses[0]:.3f} -> {fp32_losses[-1]:.3f}")
    print(f"  mixed {mp_losses[0]:.3f} -> {mp_losses[-1]:.3f}   (max gap {gap:.4f})\n")
    print(f"  fp32 : {fp32_ms:6.2f} ms/step")
    print(f"  mixed: {mp_ms:6.2f} ms/step   ({fp32_ms / mp_ms:.2f}x — big on CUDA, ~1x on CPU/MPS)")


if __name__ == "__main__":
    main()
