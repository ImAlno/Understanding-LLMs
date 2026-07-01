# Chapter 14 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All run on a CPU, importing from
[`../../code/`](../../code/).

| Script | Shows |
|--------|-------|
| [`e01_loss_mask.py`](./e01_loss_mask.py) | The SFT target: response trained (`hello!<end>`), prompt masked (`___`). |
| [`e02_lora_module.py`](./e02_lora_module.py) | LoRA starts as a no-op (`output == base`) and trains only `A`, `B`. |
| [`e03_param_savings.py`](./e03_param_savings.py) | full vs LoRA params: 12.5% → 0.20% as layers grow; ~0.39% for a 7B. |
| [`e04_lora_fit.py`](./e04_lora_fit.py) | LoRA fits a rank-8 change (→ 0.000000) but not a full-rank one (→ 0.39). |

```bash
python chapters/14-finetuning-sft/exercises/solutions/e04_lora_fit.py
```
