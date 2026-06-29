# Chapter 16 — Deployment — The Storyteller Web App

> The payoff: wrap the model in an API and a web UI, so anyone can chat with your Storyteller.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We serve the model behind a **FastAPI** backend with **streaming** responses, build a minimal **chat web UI**, and discuss real serving (batching, the KV-cache from Chapter 12, and tools like **vLLM**). This mirrors **nanochat**'s `chat_web` — your own little ChatGPT.

## Key concepts

- serving a model behind an API
- token streaming (SSE/websockets)
- a minimal chat web UI
- batching & throughput
- vLLM & production serving

## 🔧 Mini-project (planned)

Ship it: a FastAPI endpoint that streams tokens from your finetuned Storyteller, plus a simple web page to chat with it. Generate and illustrate (Ch 17) a story live in the browser.

## 📚 Resources

- [💻 karpathy/nanochat — the capstone reference](https://github.com/karpathy/nanochat)
- [📄 Kwon et al. 2023 — vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [📄 FastAPI documentation](https://fastapi.tiangolo.com/)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 15](../15-finetuning-rl/) | [Syllabus](../../README.md#-syllabus) | [Ch 17](../17-multimodal/) |
