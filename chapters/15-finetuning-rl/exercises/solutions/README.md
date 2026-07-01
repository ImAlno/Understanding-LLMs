# Chapter 15 — Exercise Solutions

Runnable solutions. **Peek only after you've tried.** All run on a CPU, importing from
[`../../code/`](../../code/).

| Script | Shows |
|--------|-------|
| [`e01_bradley_terry.py`](./e01_bradley_terry.py) | Bradley-Terry trains a reward model to rank chosen > rejected (100%, mean margin +10.7). |
| [`e02_dpo_loss.py`](./e02_dpo_loss.py) | The DPO loss aligns the curt model to warm — 5/5 replies, no reward model, no RL. |
| [`e03_over_optimization.py`](./e03_over_optimization.py) | Too-small β over-optimizes: the margin climbs (+26 → +182) while the text decays to mush. |
| [`e04_frozen_reference.py`](./e04_frozen_reference.py) | DPO moves the policy while leaving the reference byte-for-byte frozen. |

```bash
python chapters/15-finetuning-rl/exercises/solutions/e02_dpo_loss.py
```
