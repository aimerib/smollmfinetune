---
# R0-2  Add Unit Test for Dataset Generation
Status: **Todo**
Ring: R0
Created: 2025-06-18
---

## Goal
Automated test ensures `DatasetManager.generate_dataset()` returns correct number of samples and basic structure.

## Context
After R0-1 restores generation, we want CI confidence the function keeps working during refactors.

## Acceptance Criteria
- [ ] Pytest file `tests/test_dataset_generation.py` exists.
- [ ] Test spins up `DatasetManager` with dummy character card, requests 5 samples, asserts:
  * length == 5
  * each sample has `messages` list length ≥ 3 (`system`,`user`,`assistant`).
  * no exception raised.
- [ ] `pytest` passes locally.

## Implementation Notes
```text
Use a mock client to avoid external API cost:
• monkeypatch `DatasetManager.client.generate` to return deterministic stub text.
```

## Steps
1. Create `tests/` directory with `__init__.py`.
2. Implement test per above.
3. Add `pytest` to dev dependencies if missing.

## References
- Task R0-1. 