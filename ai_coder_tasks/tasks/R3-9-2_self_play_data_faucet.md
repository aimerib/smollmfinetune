---
# R3-9-2  Self-Play Data Faucet
Status: **Todo**
Ring: R3.9
Created: 2025-06-27
---

## Goal
Automate nightly self-play runs using the current SmolLM2 adapters to continuously generate conversation data in the unified R4 schema, seeding the research dataset with minimal human effort.

## Context
High-quality, model-authored data is needed before scaling to R4 experiments. A lightweight cron-driven harness that spins up two character adapters in dialogue, logs the turns, and stores to `datasets/self_play/` will turn the Devkit into a "data faucet" that improves with every model iteration.

## Acceptance Criteria
- [ ] **Harness Script**: `scripts/run_self_play_cron.py` can be called with `--world`, `--charA`, `--charB`, `--turns`.
- [ ] **Scheduler**: GitHub Action (`.github/workflows/self_play.yml`) runs nightly on cheap `ubuntu-latest` with CPU inference and pushes resulting JSONL to DVC remote.
- [ ] **Output Format**: Each dialog stored as `DatasetSample` list conforming to R4 schema; validated via `DatasetProcessor.validate_batch()`.
- [ ] **Quality Filters**: Post-run filter removes samples where JSON-correctness <90 % or length <4 turns using evaluation harness.
- [ ] **Metrics Dashboard**: WandB project `self_play_faucet` logs tokens_generated, good_samples, avg_quality.

## Implementation Notes
```text
• Use HuggingFace's `TextIteratorStreamer` for streaming inference on CPU.
• Generate two assistant responses per user turn; alternate "speaker" adapters.
• Control token `<scene_night>` injected every 10 turns to hit coverage targets.
• DVC command in workflow: `dvc add datasets/self_play/nightly_{date}.jsonl && git commit -m "data: nightly self-play"`.
```

## Checklist / Steps
1. Implement harness with argparse + wandb.
2. Add validation call to R4-3.5 eval harness.
3. Create GitHub Action cron job (runs at 03:00 UTC).
4. Configure DVC remote (s3 or local for now).
5. Verify first nightly run generates ≥5 k tokens and passes filters.

## References
Depends on: R3-2.5 (Data Collection Foundation), R3-4 (Observability) for logging hooks. 