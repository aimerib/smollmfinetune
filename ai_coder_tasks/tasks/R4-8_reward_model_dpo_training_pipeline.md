---
# R4-8: Reward Model & DPO Training Pipeline
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

## Goal
Implement the training pipelines for the Reward Model (RM) and the Direct Preference Optimization (DPO) phase, using the collected preference data to align the SFT model.

## Context
This is the core of the alignment process. First, we train a Reward Model to learn the patterns in the human preference data. Then, we use that RM as a loss function in the DPO algorithm to fine-tune the SFT model to produce outputs that would score highly on the RM.

## Acceptance Criteria
### Reward Model Training:
- [ ] A script `scripts/run_rm_training.py` is created.
- [ ] It loads a base model (like the NarrativeLLM but with a single scalar output head) and the preference dataset.
- [ ] It trains the model to predict which response is better (higher score for chosen, lower for rejected).

### DPO Training:
- [ ] A script `scripts/run_dpo.py` is created, likely using a library like TRL (DPOTrainer).
- [ ] It loads the SFT model, the preference dataset, and the trained Reward Model (or uses the implicit RM of DPO).
- [ ] It runs the DPO training loop, which fine-tunes the SFT model based on the preference pairs.
- [ ] It saves the final, aligned model adapter.

## Implementation Notes
```text
• TDD Instructions:
  - For both run_rm_training.py and run_dpo.py, follow the same TDD pattern as in R4-6:
    - Test that the script can be executed with --help.
    - Write an integration test that runs a single training step on a tiny dummy dataset without crashing.
    - Assert that model weights change after the step.
```

## Checklist / Steps
1. Create reward model training script with argument parsing
2. Implement RM architecture with scalar output head
3. Create RM training loop with preference pair loss
4. Create DPO training script using TRL or similar
5. Implement DPO training loop with preference optimization
6. Add model saving and checkpointing
7. Write comprehensive tests for both pipelines
8. Integrate with WandB for training metrics

## References
- Proposal §5.2: DPO
- TRL DPOTrainer documentation 