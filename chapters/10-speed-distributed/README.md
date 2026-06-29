# Chapter 10 — Need for Speed III: Distributed — DDP & ZeRO

> One GPU isn't enough for big models. We split the work across many.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We learn **data parallelism** (DDP): replicate the model, split the batch, and synchronize gradients. Then **ZeRO**-style sharding of optimizer states/gradients/parameters to fit bigger models, and a peek at tensor/model parallelism (Megatron).

## Key concepts

- data parallelism & DDP
- gradient all-reduce / synchronization
- ZeRO sharding stages
- model/tensor parallelism
- scaling laws intuition

## 🔧 Mini-project (planned)

Wrap your training in PyTorch **DDP** and run it across multiple processes/GPUs (or simulate locally). Verify the loss matches single-GPU and measure throughput scaling.

## 📚 Resources

- [📄 PyTorch — Getting Started with DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [📄 Rajbhandari et al. 2019 — ZeRO](https://arxiv.org/abs/1910.02054)
- [📄 Shoeybi et al. 2019 — Megatron-LM](https://arxiv.org/abs/1909.08053)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 09](../09-speed-precision/) | [Syllabus](../../README.md#-syllabus) | [Ch 11](../11-datasets/) |
