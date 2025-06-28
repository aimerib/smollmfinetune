---
# R3-5  Mini-MoE Adapter Spike
Status: **Todo**
Ring: R3
Created: 2025-06-27
---

## Goal
Train and evaluate a tiny (≈60 M params) dual-head Mixture-of-Experts model (“smollm2-Mini”) with LoRA adapters to prove that the R4 architecture’s dual-head loss, control tokens, and adapter hot-swap work end-to-end on commodity hardware (MacBook M-series / 24 GB consumer GPU).

## Context
This task is the first stepping-stone between the production Devkit (R3) and the research Narrative-LLM (R4). It de-risks key components—dual heads, MoE capacity, control token coverage—at small scale before weekend A100 runs.

## Acceptance Criteria
- [ ] **Model**: 60 M param Transformer (6 layers, 8 heads) with dual output heads (text & action) and MoE FFNs (`num_experts=4`, `top_k=2`).
- [ ] **Dataset**: 10 k synthetic samples in unified R4 schema produced via existing Devkit; at least 5 control tokens per 1 k tokens.
- [ ] **Training Script**: `scripts/run_mini_moe_sft.py` that trains for ≤2 hours on single GPU/CPU, logs to WandB.
- [ ] **Metrics**: JSON-correctness ≥80 %, personality alignment ≥0.6 on eval harness (R4-3.5).
- [ ] **Adapters**: After training, export LoRA adapter `adapter_mini_moe.safetensors`; can be hot-swapped into inference script with <1 s swap time.
- [ ] **Documentation**: `docs/spikes/mini_moe_adapter.md` summarising config, compute cost, and results.

## Implementation Notes
```text
• Use HuggingFace Transformers `GPTNeoXConfig` for prototype; set `num_experts` and `num_experts_per_tok`.
• Dual heads can be simple Linear layers tied to shared hidden states.
• Re-use DualHeadLoss (R4-3) implementation.
• Training script very similar to run_sft.py but with small model & dataset paths hard-coded for convenience.
• Provide Makefile target `make spike-mini-moe` for one-command run.
```

## Checklist / Steps
1. Design config in `narrative_engine/configs/mini_moe.json`.
2. Implement `scripts/run_mini_moe_sft.py` (argparse & wandb).
3. Generate synthetic dataset via Dataset Studio; save under `datasets/spikes/mini/`.
4. Train, export adapter, run inference sanity check.
5. Document findings & move card to `completed/` when metrics met.

## References
Prereqs: R3-3 (Data Analysis), R3-4 (Observability).  Feeds into R4-0.1 Dual-Head Tiny-Llama Spike. 