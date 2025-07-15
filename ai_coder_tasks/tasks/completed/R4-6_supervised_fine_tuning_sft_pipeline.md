---
# R4-6: Supervised Fine-Tuning (SFT) Pipeline
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

## Goal
Develop a training script that orchestrates the SFT phase, taking the prepared dataset and training the Narrative-LLM using the dual-head loss function, while logging metrics to WandB.

## Context
This is the first major training phase. It takes the base pre-trained model and teaches it the specific format of your narrative conversations, including how to generate both prose and tool-use actions. All subsequent alignment tuning (DPO) depends on a successful SFT run.

## Acceptance Criteria
- [ ] A script `scripts/run_sft.py` is created.
- [ ] The script can parse command-line arguments for configuration (e.g., model path, data path, learning rate).
- [ ] It correctly initializes the NarrativeLLM (R4-1), the DatasetProcessor (R4-2), the DualHeadLoss (R4-3), and a standard PyTorch optimizer.
- [ ] It contains a training loop that iterates through the data, performs forward and backward passes, and updates the model weights.
- [ ] It integrates with wandb to log training/validation loss and other relevant metrics (e.g., learning rate).
- [ ] The script saves model checkpoints at regular intervals.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Failing Test - Script Execution): In tests/scripts/test_run_sft.py, write a test that attempts to run the scripts/run_sft.py script with a --help flag. This will fail if the script or its argument parser doesn't exist.
  - Green (Passing Test - Script Execution): Create the script with a basic argparse setup.
  - Red (Failing Test - Training Loop Step): Write an integration test that runs the script with arguments pointing to tiny, dummy versions of the model and dataset. The test should assert that the training loop can complete at least one step without crashing. Mock the wandb.init call to prevent actual logging.
  - Green (Passing Test - Training Loop Step): Implement the core training loop logic: data loading, model forward pass, loss calculation, and optimizer step.
  - Refinement: Add tests to ensure that model weights actually change after a training step and that a checkpoint file is created.
```

## Checklist / Steps
1. Create scripts/run_sft.py with argument parsing
2. Implement model, dataset, and loss function initialization
3. Create training loop with forward/backward passes
4. Add WandB integration for metrics logging
5. Implement checkpoint saving functionality
6. Write comprehensive tests for training pipeline
7. Add validation loop and metrics

## References
- Proposal §5.2: Training Phases (SFT)
- Completed tasks: R4-1, R4-2, R4-3 