---
# R1-6  Prompt Builder Enhancement
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Include personality, goals, relationships, a random lore fact, and multi-turn chat support (with optional NSFW style cues) in all generation prompts.

## Acceptance Criteria
- [ ] **Prompt Builder** `utils/generation/prompt_builder.py`
      • `build_prompt(character, mode="chat"|"nsfw"|"qa", **opts)`
      • Injects Big-Five adjectives, top goals, relationship stance, random lore fact (`world_lore["facts"]`).
      • Adds `[NSFW:<style>]` tag when mode == "nsfw" (styles: soft, explicit, kink).
      • Helper `build_conversation_turn(turn_idx, role, text, **meta)` for iterative loops.
- [ ] **Dataset generation**
      • `DatasetManager.generate_dataset` gains `turns:int=4`, `assistant_options:int=1`.
      • Supports multi-turn chat (user ↔ assistant) and option branching (`assistant_options>1`).
      • All generation pathways (incl. `NSFWGenerationManager`) delegate to Prompt Builder.
- [ ] **Schema update** (HF Dataset)
      • Add columns `prompt_built` (bool), `lore_fact` (str), `nsfw_style` (str, nullable), `turn_id` (int), `option_id` (int, nullable), `role` (str).
      • Migration script tags legacy rows with `prompt_built=False`.
- [ ] **NSFW quality path**
      • `content_evaluation.evaluate_nsfw_quality` scores alignment between requested `nsfw_style` and produced text (`style_score`, `safety`, `coherence`).
      • Failing samples are discarded or flagged.
- [ ] **UI additions** (Dataset Generation page)
      • Slider "Number of turns" (2-8).
      • Slider "Assistant options per turn" (1-6, >1 enables preference collection).
      • Preview pane streams generated conversation; creators can mark preferred option → logged for R1-11 GRPO/PPO pipeline.
- [ ] **Unit tests**
      • `tests/test_prompt_builder.py`: ensures builder output contains trait adjectives and lore snippet; snapshot for `[NSFW:soft]`.
      • `tests/test_multi_turn_generation.py`: dataset with `turns=4`, `assistant_options=3` has expected rows and columns.
- [ ] **Documentation**
      • `docs/prompt_format.md` and `docs/dataset_format.md` updated with templates and multi-turn schema examples.
      • `notebooks/PromptBuilder_Demo.ipynb` expanded with multi-turn, NSFW example.

## References
- Relies on R1-1, R1-2, R1-3, R1-11. 