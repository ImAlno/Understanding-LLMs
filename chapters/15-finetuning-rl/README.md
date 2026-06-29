# Chapter 15 — Finetuning II: RL — RLHF, PPO & DPO

> We align the Storyteller with *preferences* — making it write the stories people actually like.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

Beyond imitation, we optimize for **human preferences**. We cover the classic **RLHF** pipeline (reward model + **PPO**) and the simpler, popular **DPO**, which trains directly on preferred-vs-rejected pairs with no separate reward model or RL loop.

## Key concepts

- reward modeling
- RLHF pipeline
- PPO (policy optimization)
- DPO (direct preference optimization)
- alignment & its pitfalls

## 🔧 Mini-project (planned)

Collect (or synthesize) preference pairs over your Storyteller's outputs and align it with **DPO**. Show before/after samples where the aligned model writes clearly better stories.

## 📚 Resources

- [🎥 Karpathy — Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
- [📄 Ouyang et al. 2022 — InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155)
- [📄 Rafailov et al. 2023 — DPO](https://arxiv.org/abs/2305.18290)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 14](../14-finetuning-sft/) | [Syllabus](../../README.md#-syllabus) | [Ch 16](../16-deployment/) |
