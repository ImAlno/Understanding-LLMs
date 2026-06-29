# Chapter 8 — Exercises

Feel how the GPU behaves. **Try first**; solutions are in [`solutions/`](./solutions/). All import
the helpers from [`../code/benchmark.py`](../code/benchmark.py) and are **device-agnostic** — they
run on a CPU, an Apple GPU, or a CUDA GPU.

```bash
python chapters/08-speed-device/exercises/solutions/e01_same_device.py
```

> 💻 **For the real numbers, run these on Colab with a GPU** (E02–E04 only have a story to tell
> when a GPU is present — on a pure CPU they print a "run on Colab" note).

---

### E01 — Fix the same-device error 🥇
A model on the GPU, a batch on the CPU → `RuntimeError`. Fix it the way you always do: move the
stray tensor.

- 🧩 **Scaffold:** [`starter/e01_same_device.py`](./starter/e01_same_device.py).
- ✅ **Solution:** [`solutions/e01_same_device.py`](./solutions/e01_same_device.py).

  <details><summary>💡 Hint</summary>

  Put the batch on the same device as the model before the forward pass: `model(batch.to(device))`.
  </details>

### E02 — Find the crossover
Benchmark CPU vs GPU matmul across sizes and find where the GPU *starts* winning. (On a laptop
GPU it's around 1024; on a datacenter GPU it's smaller and the high-end speedup is far bigger.)

- ✅ Solution: [`solutions/e02_crossover.py`](./solutions/e02_crossover.py).

### E03 — Why timing needs `synchronize`
Time a big matmul **without** then **with** `synchronize`. Without it, the GPU looks absurdly fast
(you measured the *launch*, not the *work*) — often **hundreds of times** off.

- ✅ Solution: [`solutions/e03_synchronize.py`](./solutions/e03_synchronize.py).

### E04 — The cost of moving data
Compare the time to **move** a big tensor CPU→GPU against the time to **compute** with it. They're
often comparable — which is why you keep data on the GPU and don't `.item()` in the hot loop.

- ✅ Solution: [`solutions/e04_transfer.py`](./solutions/e04_transfer.py).

> 🧠 **Takeaway:** the GPU is fast for big parallel work, but launch overhead, the async timing
> trap, and CPU↔GPU transfers all bite — and all three are things you control.
