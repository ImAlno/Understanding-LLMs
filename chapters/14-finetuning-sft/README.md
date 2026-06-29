# Chapter 14 — Finetuning I: SFT, PEFT & LoRA

> A base model predicts text; a *chat* model follows instructions. We teach the Storyteller to take requests.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We do **supervised finetuning (SFT)**: train on (instruction → response) pairs formatted with a **chat template**, so the model learns to answer rather than merely continue. Then **LoRA** — finetune via tiny low-rank adapters instead of all the weights, on a single GPU.

## Key concepts

- base vs. instruct/chat models
- chat templates & special tokens
- supervised finetuning (SFT)
- parameter-efficient finetuning (PEFT)
- LoRA & QLoRA

## 🔧 Mini-project (planned)

Finetune your pretrained Storyteller to follow prompts like *“Write a story about a brave little robot.”* Implement LoRA and compare it to full finetuning in cost and quality.

## 📚 Resources

- [📄 Hu et al. 2021 — LoRA](https://arxiv.org/abs/2106.09685)
- [📄 Dettmers et al. 2023 — QLoRA](https://arxiv.org/abs/2305.14314)
- [📄 HuggingFace — Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 13](../13-inference-quantization/) | [Syllabus](../../README.md#-syllabus) | [Ch 15](../15-finetuning-rl/) |
