---
# R5-2: The Iceberg Model (Subtext Generation)
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Re-architect the Narrative-LLM and its training pipeline to generate both "surface" dialogue and hidden "subtext," representing the character's true, unspoken intentions, thereby creating deep psychological realism.

## Context
Inspired by Hemingway's "Iceberg Theory," this feature imbues characters with a rich inner life. The model will learn to separate what a character says from what they mean, a critical step towards creating truly believable, layered personas rather than simple text generators.

## Acceptance Criteria
- [ ] Schema Update: The `Turn` model in `narrative_engine/data_schema.py` is updated with a new field: `subtext: Optional[str] = None`.
- [ ] Architecture Modification:
  - [ ] The `NarrativeLLM` in `narrative_engine/model.py` is modified to include a third output head: `subtext_head`.
  - [ ] The model's forward pass will now return three sets of logits: `text_logits`, `action_logits`, and `subtext_logits`.
- [ ] Loss Function Upgrade: The `DualHeadLoss` is refactored into a `MultiHeadLoss` (`narrative_engine/loss.py`) capable of calculating a joint loss across all three heads, using an expanded loss mask.
- [ ] Data Pipeline Update: The `DatasetProcessor` (`narrative_engine/data_pipeline.py`) is updated to handle the subtext field, correctly tagging its tokens for the new `subtext_head` in the loss mask.
- [ ] UI for Creators: The Dataset Studio and live chat interfaces gain a "🎭 Director's View" toggle that, when enabled, displays the generated subtext beneath the surface dialogue for review and editing.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Schema): Update the Pydantic model tests to verify that the Turn model now correctly handles the subtext field.
  - Red (Model): Update the test_forward_pass_shapes test in tests/narrative_engine/test_model.py to assert that the model's output dictionary now contains subtext_logits of the correct shape.
  - Red (Loss): Create tests/narrative_engine/test_multi_head_loss.py. Write a test that provides dummy data for all three heads and asserts that the loss is calculated correctly based on the masking.
  - Green (All): Implement the architectural, loss, and data pipeline changes to make the backend tests pass.
  - Red/Green (UI): Using streamlit.testing.v1.AppTest, write a test for the chat UI. Pre-populate a message with subtext. Assert that the subtext is initially hidden. Simulate clicking the "Director's View" toggle and assert that the subtext is now visible in the rendered output.
```

## Checklist / Steps
1. Update Turn model schema to include subtext field
2. Modify NarrativeLLM architecture to add subtext_head
3. Refactor DualHeadLoss to MultiHeadLoss for three heads
4. Update DatasetProcessor to handle subtext tokens
5. Add Director's View toggle to UI interfaces
6. Implement subtext display/hide functionality
7. Write comprehensive tests for all components
8. Update training pipeline to handle three-head loss

## References
Depends on R4-1 (Model Architecture), R4-2 (Data Pipeline), and R4-3 (Loss Function). 