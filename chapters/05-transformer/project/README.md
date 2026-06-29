# 🏛️ Mini-Project — Your First GPT

You built every piece in the notebook. Now **assemble the whole thing yourself** and train it:
a real, end-to-end GPT that writes Shakespeare. This is the capstone of the first five chapters.

> **How this works:** the Transformer **blocks** are already built (in
> [`../code/gpt.py`](../code/gpt.py)). In
> [`starter/my_gpt.py`](./starter/my_gpt.py) you wire them into a GPT and train it — two TODOs:
> the **forward pass** (embeddings → blocks → final norm → head) and the **training loop**. A
> full reference is in [`solution/my_gpt.py`](./solution/my_gpt.py).

## 🎯 What it does

```bash
python starter/my_gpt.py        # ~2 minutes on a laptop CPU
```

It trains an ~817K-parameter GPT on Shakespeare, prints the falling loss, then generates a
paragraph and saves it to `my_first_gpt.txt` (illustrative — your sample will differ, since
generation draws random tokens):

```
KING EWIO:
Etwar, our, inte is-cry out musterford,
He have to no at see hold see
...
```

## 🛠️ Your TODOs

1. **The forward pass** — add the token and position embeddings, run the result through
   `self.blocks`, apply the final `self.ln_f`, then `self.lm_head` for logits. (The loss line
   is already there.)
2. **The training loop** — the get-batch → forward → `zero_grad` → `backward` → `step` you've
   done since Chapter 3, now driving a full Transformer.

## ✅ Checking your work

- The loss should fall from ~4.2 toward **~1.7** (well under Chapter 4's 2.34).
- The sample should have **speaker names, line breaks, and word-shaped words** — recognizable
  (if broken) English, not the character-soup of Chapter 4.
- Compare against `solution/my_gpt.py`.

## 🚀 Extensions (open-ended)

- **Prompt it.** Start generation from your own text (encode a string instead of the zero
  token) and see it continue in Shakespeare's voice.
- **Scale it up.** Bump `n_embd`, `n_layer`, `block_size` in `code/gpt.py` and train longer —
  watch the text sharpen (it'll get slower on a CPU).
- **Temperature.** Divide the logits by a `temperature` before softmax; compare 0.5 (timid) vs
  1.2 (wild).
- **Save & reload.** `torch.save(model.state_dict(), "gpt.pt")` so you can generate again
  without retraining.

You just built and trained a GPT from scratch — the same architecture as ChatGPT. From here the
course is about making it **better**: real tokenization ([Chapter 6](../../06-tokenization/)),
speed, scale, and the finetuning that turns a text-continuer into an assistant. 🎉
