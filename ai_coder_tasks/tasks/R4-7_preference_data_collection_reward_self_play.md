---
# R4-7: Preference Data Collection (Reward & Self-Play)
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

## Goal
Create the infrastructure to generate and collect preference data: a "Self-Play Harness" for automated data generation and a "Reward Collector" UI for human feedback.

## Context
To align the model with human preferences (DPO), we first need to collect those preferences. This involves generating conversational snippets (either from the model playing against itself or from curated prompts) and having humans choose the better response. This task builds the tools for both steps.

## Acceptance Criteria
### Self-Play Harness:
- [ ] A script `scripts/run_self_play.py` is created.
- [ ] It can load two instances of the NarrativeLLM (or one instance with two different adapters) to converse with each other.
- [ ] It logs the full conversation traces in the unified dataset format to a specified output directory.

### Reward Collector:
- [ ] A simple web application (e.g., a new Streamlit page `page_reward_labeling.py`) is created.
- [ ] It reads conversation snippets, displays two alternative assistant responses (chosen vs. rejected style), and allows a user to select the preferred one.
- [ ] Selections are saved to a database or a structured file (e.g., `preference_pairs.jsonl`), storing the prompt, chosen response, and rejected response.

## Implementation Notes
```text
• TDD Instructions:
  - Self-Play Harness: Write a test that runs run_self_play.py for a single turn. Mock the model's generate method to return deterministic output. Assert that an output log file is created with the expected content.
  - Reward Collector (UI): Using streamlit.testing.v1.AppTest, write a test that loads the reward labeling page. Assert that it correctly displays a prompt and two buttons for the responses. Simulate a button click and assert that the corresponding data is written to a (mocked) database or file.
```

## Checklist / Steps
1. Create self-play harness script with model loading
2. Implement conversation generation between model instances
3. Add logging functionality for conversation traces
4. Create Streamlit page for reward collection UI
5. Implement preference selection and storage
6. Write tests for both harness and UI components
7. Integrate with existing dataset format

## References
- Proposal §5.2: Reward Modelling
- Proposal §9: Work-Package Skeleton (Items 6 & 7)
- Existing Task: R1-11_ppo_finetune_pipeline.md (for preference data schema) 