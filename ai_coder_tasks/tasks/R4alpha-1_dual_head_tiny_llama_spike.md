---
# R4α-1  Dual-Head Tiny-Llama Spike
Status: **Todo**
Ring: R4α
Created: 2025-06-27
---

## Goal
Validate that the Narrative-LLM dual-head architecture scales from the Mini-MoE (R3-9-1) to a 1.3 B-param Tiny-Llama subset (12 layers). Demonstrate stable training with the DualHeadLoss and achieve JSON-correctness ≥85 % on eval harness.

## Context
Success here green-lights scaling to 7 B and beyond. Failure triggers redesign before expensive pretraining.

## Acceptance Criteria
- [ ] Fork Tiny-Llama v0 weights; retain embeddings + 12 transformer blocks.
- [ ] Patch model with action_head identical dim to lm_head.
- [ ] LoRA-only fine-tune on 50 k dataset samples for 500 steps on single A100 (40 GB) via run_sft_tiny_llama.py.
- [ ] WandB run logs loss curves; converges (<1.0 CE) without divergence.
- [ ] Eval harness (R4-3.5) scores: JSON ≥85 %, personality alignment ≥0.7.
- [ ] Spike report `docs/spikes/tiny_llama_dual_head.md` with findings & next steps.

## Implementation Notes
```text
• Use unsloth or bitsandbytes 4-bit loading to fit in memory.
• Re-use telemetry_sdk for logging.
• Gradient checkpointing on.
```

## Checklist / Steps
1. Script `scripts/run_tiny_llama_sft.py` (argparse).
2. Modify NarrativeLLM.build_base_model to optionally load pretrained.
3. Launch training on rented A100; capture telemetry.
4. Run eval harness; document.

## References
Depends on: R3-9-1, R3-9-3 Telemetry SDK. 