---
# R4-9: Evaluation Harness & Safety Layer
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

## Goal
Extend the basic evaluation foundation (R4-3.5) with comprehensive quality metrics and implement a safety layer for content filtering.

## Context
"You can't improve what you don't measure." This task builds the automated suite of tests that will run against every new model checkpoint to ensure quality and prevent regressions. It operationalizes the success criteria from the design document.

## Acceptance Criteria
- [ ] A directory `narrative_engine/evaluation/` is created.
- [ ] It contains separate modules for each evaluation domain:
  - [ ] `eval_json_correctness.py`: Generates prompts requiring tool use and checks if the output is valid JSON.
  - [ ] `eval_coherence.py`: Uses an LLM-as-judge to check for contradictions in long conversations.
  - [ ] `eval_latency.py`: Measures time-to-first-token and per-token generation speed.
  - [ ] `eval_memory_consistency.py`: Validates memory head outputs and formation accuracy.
- [ ] A main script `scripts/run_evaluation.py` orchestrates running all evaluations against a specified model checkpoint.
- [ ] A `SafetyLayer` class is implemented that can be wrapped around the model's generation function to perform input/output filtering based on blocklists.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Failing Test - JSON Eval): In tests/narrative_engine/evaluation/test_json_correctness.py, write a test that calls the evaluation function with a correct JSON string and asserts the score is 1.0. Then add a test with malformed JSON and assert the score is 0.0.
  - Green (Passing Test - JSON Eval): Implement the JSON validation logic.
  - Repeat: Follow this pattern for each evaluation module, writing focused unit tests for the core logic.
  - Safety Layer: Write unit tests for the SafetyLayer, feeding it text that should be blocked and text that should be allowed, and asserting the correct behavior.
  - Memory Eval: Test memory head outputs for proper embedding normalization and metadata validation.
```

## Checklist / Steps
1. Create evaluation directory structure
2. Implement JSON correctness evaluation module
3. Implement coherence evaluation with LLM-as-judge
4. Implement latency measurement module
5. **NEW**: Add memory consistency evaluation for triple-head architecture
6. Create main evaluation orchestration script
7. Implement SafetyLayer class with filtering logic
8. Write comprehensive tests for all evaluation components
9. Add configuration for evaluation thresholds and blocklists

## References
- Depends on: R4-3.5 (Evaluation Foundation)
- Uses: Triple-head architecture with Generation + Control + Memory heads
- Proposal §7: Evaluation & Validation
- Proposal §9: Work-Package Skeleton (Items 8 & 9) 