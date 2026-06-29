# Chapter 4 — Exercises

Cement attention. **Try each yourself first**; solutions are in [`solutions/`](./solutions/).
Most import the data and vocab from `../code/attention.py`, so run from anywhere:

```bash
python chapters/04-attention/exercises/solutions/e01_avg_proof.py
```

> 💡 **Reuse the harness.** E02–E05 each train a small attention model — copy the model +
> training loop from [`../code/attention.py`](../code/attention.py) and change the one thing
> the exercise asks.

---

### E01 — Prove the matmul trick equals a loop 🥇
The whole chapter rests on "a triangular matrix multiply averages the past." *Prove it*:
compute the running average of a `(B, T, C)` tensor two ways — an explicit double loop, and
`wei @ x` — and assert they're equal.

- 🧩 **Scaffold:** [`starter/e01_avg_proof.py`](./starter/e01_avg_proof.py).
- ✅ **Solution:** [`solutions/e01_avg_proof.py`](./solutions/e01_avg_proof.py).

  <details><summary>💡 Hint 1 — the loop</summary>

  For each batch `b` and position `t`, the running average is `x[b, :t+1].mean(0)` (mean over
  positions `0…t`). Fill a `xbow_loop` tensor of the same shape.
  </details>

  <details><summary>💡 Hint 2 — the matmul</summary>

  `tril = torch.tril(torch.ones(T, T))`, then `wei = tril / tril.sum(1, keepdim=True)`, then
  `wei @ x`. Compare with `torch.allclose(..., atol=1e-6)`.
  </details>

### E02 — Visualize the attention weights
Train a small head, then **plot the `(T, T)` attention matrix** for one sequence as a
heatmap. You'll *see* the lower-triangular causal structure and which positions each token
leans on.

- **Why it matters:** the attention weights are the model's "what am I looking at?" — made
  visible. (This is the mini-project's core, on training wheels.)

  <details><summary>💡 Hint — getting the weights out</summary>

  Have the head stash its weights (`self.last_wei = wei.detach()`) so you can grab them after
  a forward pass. Then `plt.imshow(last_wei[0])`. Black cells above the diagonal = the masked
  future.
  </details>
- ✅ Solution: [`solutions/e02_visualize.py`](./solutions/e02_visualize.py) (saves `attention.png`).

### E03 — Break the causal mask
Delete the `masked_fill(...)` line so every token can see the *future*, and retrain. The
**train** loss plunges (the model cheats — it can read the answer) but the model is useless
for generation. This is *why* the mask exists.

- **Why it matters:** it makes the causal rule visceral — a next-token predictor must not see
  the next token.

  <details><summary>💡 Hint</summary>

  Train two models, identical except one skips the `masked_fill` line. Print both val losses.
  The no-mask one will be much lower — and that low number is meaningless, because at
  generation time there *is* no future to peek at.
  </details>
- ✅ Solution: [`solutions/e03_no_mask.py`](./solutions/e03_no_mask.py).

### E04 — Does more context help? Tune `block_size`
Train the model at a few context lengths (`block_size` = 8, 16, 32) and compare the **val**
loss. Does more context help? (It does here — but only because the model is big enough to use
it; a tinier one barely benefits, which is part of why capacity matters.)

- ✅ Solution: [`solutions/e04_block_size.py`](./solutions/e04_block_size.py).

### E05 — Multi-head attention *(stretch → Chapter 5)*
One head learns one kind of relationship. **Multi-head** runs several smaller heads in
parallel and concatenates them, so the model can attend to different things at once. Build
`n` heads of size `n_embd // n` and compare to the single head.

- **Reflection:** this is the first piece of the full Transformer you'll assemble next chapter.

  <details><summary>💡 Hint</summary>

  (`nn.ModuleList` is just a list of layers that `nn.Module` tracks for you — like `nn.Linear`,
  so their weights get trained.) Build the heads with
  `self.heads = nn.ModuleList([Head(n_embd // n) for _ in range(n)])`, and in `forward`
  concatenate their outputs along the last dim: `torch.cat([h(x) for h in self.heads], dim=-1)`.
  The concatenated size comes back to `n_embd`.
  </details>
- ✅ Solution: [`solutions/e05_multihead.py`](./solutions/e05_multihead.py).
