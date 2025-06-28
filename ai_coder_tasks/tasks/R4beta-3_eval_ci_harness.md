---
# R4β-3  Evaluation Harness CI Job
Status: **Todo**
Ring: R4β
Created: 2025-06-27
---

## Goal
Catch regressions early by running a lightweight training step + evaluation harness in CI for every pull request touching model or dataset code, blocking merges that drop key quality metrics.

## Context
With multiple researchers iterating, quality can silently degrade. A 5-minute GPU-less job can train a toy model for ~50 steps on a micro-dataset and run JSON correctness, personality alignment, and speed tests to ensure nothing fundamental broke.

## Acceptance Criteria
- [ ] **GitHub Workflow** `.github/workflows/eval_ci.yml` triggered on `pull_request`.
- [ ] **Steps**:
      1. Checkout repo & install minimal deps with pip cache.
      2. Generate 100-sample toy dataset via `DatasetProcessor` (CPU).
      3. Train smollm2-tiny (4 layers) for 50 steps using `run_sft_ci.py` (CPU, fp32).
      4. Run evaluation harness (`scripts/run_basic_evaluation.py`) against checkpoint.
      5. Parse JSON correctness & personality alignment; fail job if <70 % / <0.5.
- [ ] **Skip Logic**: Workflow auto-skips if only docs or markdown changed.
- [ ] **Badges**: Add CI shield to README showing last run status.
- [ ] **Unit Test**: `tests/scripts/test_run_sft_ci.py` asserts script exits 0 and produces checkpoint file within 120 s on CI runner.

## Implementation Notes
```text
• Use huggingface `--no_cuda` training to stay inside 7 GB RAM.
• job.<container> ubuntu-latest, set `pytest -q tests/scripts/test_run_sft_ci.py`.
• Telemetry SDK still records run -> allows local repro.
```

## Checklist / Steps
1. Create `scripts/run_sft_ci.py` (hard-coded tiny config).
2. Add basic evaluation CLI flags for quick mode.
3. Write GitHub Action with job matrix python 3.11.
4. Add badge to root README.

## References
Depends on R4-3.5 evaluation harness foundation; uses telemetry_sdk (R3-9-3). 