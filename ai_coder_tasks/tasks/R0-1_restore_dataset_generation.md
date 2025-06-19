---
# R0-1  Restore Dataset Generation
Status: **In-Progress**
Ring: R0
Created: 2025-06-18
---

## Goal
`page_dataset_preview → Fast Mode` button works again (samples appear in preview grid).

## Context
Refactor split `DatasetManager` into `utils/generation/*`, but `DatasetManager.generate_dataset()` is now missing. Legacy implementation lives in `app/utils/dataset.old.py` (≈5 k lines).

## Acceptance Criteria
- [ ] Calling `st.session_state.dataset_manager.generate_dataset()` returns a list of sample dicts.
- [ ] Streamlit UI no longer raises AttributeError.
- [ ] Quick manual test: generate 10 samples in Fast mode.

## Implementation Notes
  1. Identify functions still used by UI:
     • `generate_dataset`
     • `generate_fast_templated_dataset`
     • `generate_interactive_batch`
     • `generate_factual_qa_dataset`
  2. Cut-and-paste bodies from `dataset.old.py` into **appropriate new modules**:
     • core logic → `utils/generation/base_manager.py` or `nsfw_manager.py`
     • high-level façade methods → `utils/dataset/manager.py`
  3. Remove any direct OpenAI calls that duplicate `openai_client`; switch to existing client.
  5. Ensure no circular imports.

## Guard-rails & Gotchas
* If moving code enlarges a file beyond 1 000 LOC, split into helpers.
* Delete the legacy `dataset.old.py` only after R0-2 tests pass.

## Steps
1. Copy required methods into new modules as per plan.
2. Update imports in `app/app.py` and elsewhere.
3. Run Streamlit and generate 10 samples in Fast mode.
4. Commit with message `refactor: migrate legacy dataset generation code`. (For Human to complete after verification of changes)

## References
- Chat 2025-06-18 – "RING 0 baseline"
- Code at `app/utils/dataset.old.py`. 