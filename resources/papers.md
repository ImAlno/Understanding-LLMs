# 📚 Papers & Written Tutorials (per chapter)

A curated, beginner-friendly reading list — one to four resources per chapter.
We favor the *seminal paper* (so you can say you read the original) paired with
a *gentle explainer* (so it actually makes sense the first time).

> Every link was verified to resolve in 2026. A few items are PDFs or
> conference pages rather than arXiv (noted where relevant). Read these
> **alongside** the chapter, not before — each chapter tells you when.

---

## Chapter 01 — Bigram / Character-Level Language Modeling
- [Shannon (1948), "A Mathematical Theory of Communication"](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — The founding paper of information theory; introduces entropy and the n-gram/Markov view of language that underlies every LM.
- [Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Ch. 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — The clearest free textbook chapter on n-grams, smoothing, and perplexity.
- [Karpathy, makemore](https://github.com/karpathy/makemore) — The hands-on repo this chapter is modeled on.

## Chapter 02 — Micrograd / Backpropagation / Autograd
- [Karpathy, "building micrograd" (video)](https://www.youtube.com/watch?v=VMj-3S1tku0) — The flagship from-scratch walkthrough of autograd and backprop.
- [Karpathy, micrograd](https://github.com/karpathy/micrograd) — The ~100-line autograd engine, the code companion to the video.
- [Olah, "Calculus on Computational Graphs: Backpropagation"](https://colah.github.io/posts/2015-08-Backprop/) — Beautifully illustrated explanation of *why* reverse-mode is efficient.

## Chapter 03 — N-gram MLP & GELU
- [Bengio et al. (2003), "A Neural Probabilistic Language Model"](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — Introduces learned embeddings + an MLP language model; the blueprint for this chapter.
- [Hendrycks & Gimpel (2016), "Gaussian Error Linear Units (GELUs)"](https://arxiv.org/abs/1606.08415) — The activation function used throughout GPT-style models.

## Chapter 04 — Attention
- [Bahdanau et al. (2014), "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) — The paper that introduced attention.
- [Alammar, "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) — The classic visual, intuition-first explainer of self-attention.
- [Alammar, "Visualizing A Neural Machine Translation Model (Seq2seq with Attention)"](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — Animated walkthrough of attention inside seq2seq.

## Chapter 05 — Transformer
- [Vaswani et al. (2017), "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — The original Transformer: self-attention, multi-head attention, positional encodings.
- [Radford et al. (2019), "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — The decoder-only architecture our course mirrors.
- [Ba, Kiros, Hinton (2016), "Layer Normalization"](https://arxiv.org/abs/1607.06450) — The normalization used in every Transformer block.
- [He et al. (2015), "Deep Residual Learning"](https://arxiv.org/abs/1512.03385) — Residual/skip connections, what makes deep stacks trainable.

## Chapter 06 — Tokenization
- [Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) — Brought Byte-Pair Encoding (BPE) to NLP.
- [Kudo & Richardson (2018), "SentencePiece"](https://arxiv.org/abs/1808.06226) — Language-independent subword tokenizer used across modern LLMs.
- [Karpathy, minbpe](https://github.com/karpathy/minbpe) — Minimal, clean BPE code (GPT-2/GPT-4 style) to build alongside.

## Chapter 07 — Optimization
- [Kingma & Ba (2014), "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) — The default optimizer for deep learning.
- [Loshchilov & Hutter (2017), "Decoupled Weight Decay Regularization (AdamW)"](https://arxiv.org/abs/1711.05101) — The optimizer actually used to train LLMs.
- [He et al. (2015), "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852) — He/Kaiming initialization.
- [Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks"](https://proceedings.mlr.press/v9/glorot10a.html) — Xavier/Glorot init.
- [Loshchilov & Hutter (2016), "SGDR: Warm Restarts"](https://arxiv.org/abs/1608.03983) — The cosine learning-rate schedule.

## Chapter 08 — Device / GPU
- [NVIDIA, "An Even Easier Introduction to CUDA"](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) — The gentlest hands-on intro to CUDA threads and blocks.
- [Horace He, "Making Deep Learning Go Brrrr From First Principles"](https://horace.io/brrr_intro.html) — Compute vs. memory-bandwidth vs. overhead bottlenecks, demystified.
- [Modal, "GPU Glossary"](https://modal.com/gpu-glossary) — Browsable reference for GPU hardware and the memory hierarchy.

## Chapter 09 — Mixed Precision
- [Micikevicius et al. (2017), "Mixed Precision Training"](https://arxiv.org/abs/1710.03740) — FP16 training with loss scaling and an FP32 master copy.
- [PyTorch, "What Every User Should Know About Mixed Precision Training"](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) — Practical fp16 vs bf16 vs TF32 with code.

## Chapter 10 — Distributed Training
- [PyTorch, "Getting Started with Distributed Data Parallel"](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) — Hands-on DDP: gradient syncing, `torchrun`.
- [Rajbhandari et al. (2019), "ZeRO"](https://arxiv.org/abs/1910.02054) — Sharding optimizer states/gradients/params to slash memory.
- [Shoeybi et al. (2019), "Megatron-LM"](https://arxiv.org/abs/1909.08053) — Tensor (intra-layer) model parallelism.

## Chapter 11 — Datasets
- [Gao et al. (2020), "The Pile"](https://arxiv.org/abs/2101.00027) — The classic curated 800GB pretraining corpus.
- [HuggingFace, "FineWeb: decanting the web"](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — Beginner-friendly walkthrough of building a 15T-token web dataset.
- [Eldan & Li (2023), "TinyStories"](https://arxiv.org/abs/2305.07759) — Tiny models learn fluent English on synthetic kids' stories. **Our capstone dataset.**
- [HuggingFace, "Cosmopedia"](https://huggingface.co/blog/cosmopedia) — Recipe for generating synthetic pretraining data.

## Chapter 12 — Inference / KV-Cache
- [HuggingFace, "KV Caching Explained"](https://huggingface.co/blog/not-lain/kv-caching) — Why/how the KV cache speeds up decoding, with code.
- [HuggingFace, "KV Cache from scratch in nanoVLM"](https://huggingface.co/blog/kv-cache) — Step-by-step implementation (prefill vs decode).
- [Pope et al. (2022), "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — Inference partitioning and memory/latency tradeoffs.

## Chapter 13 — Quantization
- [Dettmers et al. (2022), "LLM.int8()"](https://arxiv.org/abs/2208.07339) — Outlier-aware 8-bit inference (behind bitsandbytes).
- [Frantar et al. (2022), "GPTQ"](https://arxiv.org/abs/2210.17323) — One-shot 3–4 bit weight quantization.
- [Lin et al. (2023), "AWQ"](https://arxiv.org/abs/2306.00978) — Activation-aware weight quantization (MLSys 2024 best paper).
- [HuggingFace, "A Gentle Introduction to 8-bit Matrix Multiplication"](https://huggingface.co/blog/hf-bitsandbytes-integration) — Beginner-friendly quantization walkthrough.

## Chapter 14 — SFT / PEFT / LoRA
- [Hu et al. (2021), "LoRA: Low-Rank Adaptation"](https://arxiv.org/abs/2106.09685) — The foundational parameter-efficient finetuning method.
- [Dettmers et al. (2023), "QLoRA"](https://arxiv.org/abs/2305.14314) — 4-bit base model + LoRA to finetune on a single GPU.
- [Wei et al. (2021), "Finetuned Language Models Are Zero-Shot Learners (FLAN)"](https://arxiv.org/abs/2109.01652) — The instruction-tuning paper.
- [HuggingFace, "Chat templates"](https://huggingface.co/docs/transformers/main/en/chat_templating) — Formatting chat data for SFT.

## Chapter 15 — RLHF / PPO / DPO
- [Ouyang et al. (2022), "Training language models to follow instructions (InstructGPT)"](https://arxiv.org/abs/2203.02155) — The canonical RLHF pipeline (SFT → reward model → PPO).
- [Schulman et al. (2017), "Proximal Policy Optimization (PPO)"](https://arxiv.org/abs/1707.06347) — The RL algorithm used in RLHF.
- [Rafailov et al. (2023), "Direct Preference Optimization (DPO)"](https://arxiv.org/abs/2305.18290) — Align directly on preferences, no reward model or RL loop.

## Chapter 16 — Deployment
- [Kwon et al. (2023), "PagedAttention / vLLM"](https://arxiv.org/abs/2309.06180) — KV-cache memory management for high-throughput serving.
- [FastAPI documentation](https://fastapi.tiangolo.com/) — The Python framework we wrap our model in.
- [vLLM documentation](https://docs.vllm.ai/en/latest/) — Standing up an OpenAI-compatible inference server.

## Chapter 17 — Multimodal
- [van den Oord et al. (2017), "Neural Discrete Representation Learning (VQ-VAE)"](https://arxiv.org/abs/1711.00937) — Learned discrete codes; the basis for tokenizing images.
- [Esser et al. (2020), "Taming Transformers (VQGAN)"](https://arxiv.org/abs/2012.09841) — VQ codebook + adversarial training + a transformer prior.
- [Peebles & Xie (2022), "Scalable Diffusion Models with Transformers (DiT)"](https://arxiv.org/abs/2212.09748) — The transformer backbone behind modern image/video models.
- [Lilian Weng, "What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — The go-to math-grounded intro to diffusion.
- [HuggingFace, "The Annotated Diffusion Model"](https://huggingface.co/blog/annotated-diffusion) — A DDPM implemented from scratch in PyTorch.
