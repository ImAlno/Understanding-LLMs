# Chapter 07 — Optimization — Training That Actually Works

> Good models are made by good training. We tame initialization, optimizers, and learning-rate schedules.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We dig into **why** training succeeds or stalls: sane **initialization** so activations don't explode/vanish, the **AdamW** optimizer, **learning-rate warmup + cosine decay**, gradient clipping, and weight decay. Your GPT trains faster and to a lower loss.

## Key concepts

- weight initialization (He/Xavier)
- Adam & AdamW
- learning-rate warmup & cosine schedules
- gradient clipping
- weight decay & regularization

## 🔧 Mini-project (planned)

Take your Chapter 5 GPT and run an optimization ablation: fix the init, switch SGD→AdamW, add warmup+cosine. Plot the loss curves and quantify each improvement.

## 📚 Resources

- [🎥 Karpathy — makemore Part 3: Activations, Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- [📄 Kingma & Ba 2014 — Adam](https://arxiv.org/abs/1412.6980)
- [📄 Loshchilov & Hutter 2017 — AdamW](https://arxiv.org/abs/1711.05101)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 06](../06-tokenization/) | [Syllabus](../../README.md#-syllabus) | [Ch 08](../08-speed-device/) |
