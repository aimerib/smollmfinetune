---
# R1-10  Control Token Vocabulary and UI
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Implement a core set of special "control tokens" that give creators fine-grained command over generated content. This includes tokens for mood, actions, and NSFW styles, all manageable through the UI.

## Acceptance Criteria

### 1. Token Definition & Management
- [ ] Create `core_data/tokens.json` to define the default list of control tokens (id, description, category, ui_icon).
- [ ] This list **must** include NSFW style tokens like `<nsfw_soft>`, `<nsfw_explicit>`, matching the styles defined in `R1-6`.
- [ ] `WorldManager` (`utils/world.py`) will be responsible for managing tokens on a per-world basis, copying the default set on creation.

### 2. Tokenizer Patching
- [ ] Create a new directory: `scripts/`.
- [ ] Inside, create `scripts/patch_tokenizer.py`. This script loads a base model's tokenizer, adds the tokens from `core_data/tokens.json`, and saves the result to a cached directory (e.g., `.cache/tokenizers/<base_model_name>-patched`).
- [ ] The main application (`app.py`) will check for a patched tokenizer and run the script if it doesn't exist.

### 3. Prompt and Dataset Integration
- [ ] The `PromptBuilder` (`utils/generation/prompt_builder.py`) will be updated to include a natural language hint for any token used in a prompt (e.g., "The following dialogue uses the token <mood_happy>, which means the character is cheerful."). This is critical for teaching the model the token's meaning.
- [ ] This hint should only be inserted once at the beginning of a multi-turn conversation.

### 4. UI Components
- [ ] On the **Dataset Generation** page (`page_dataset_preview` in `app.py`), add a UI widget: "🎭 Insert Control Token".
- [ ] This widget will be a searchable dropdown menu, allowing creators to easily find and insert tokens into the text input areas for multi-turn chat generation.
- [ ] The page will also display "Token Coverage" statistics after generation, showing how many times each token was used and warning if any are under-utilized.

### 5. Runtime Export
- [ ] The `TrainingManager.export_runtime_packet` function must be updated to include the world-specific `tokens.json` file in the exported runtime packet.

## Implementation Notes
- **Data Collator**: When fine-tuning, the `DataCollatorForLanguageModeling` must be configured *not* to mask these special tokens.
- **Example Tokens**:
  ```json
  [
    {"token": "<mood_happy>", "category": "mood", "description": "Character speaks in a cheerful, upbeat tone.", "ui_icon": "😊"},
    {"token": "<stage_whisper>", "category": "action", "description": "Character speaks softly or whispers.", "ui_icon": "🤫"},
    {"token": "<nsfw_soft>", "category": "nsfw", "description": "Scene contains soft, romantic, or suggestive content.", "ui_icon": "🌶️"}
  ]
  ```

## References
- This card directly implements the `[NSFW:<style>]` tag requirement from `R1-6`.
- It provides the mechanism for the `runtime_prompt_constructor` in `R2-2`.
- Aligns with the "Devkit" vision from the `overview.md`.

## Context
We want tighter control than pure prose.  A curated set (~10-15) provides:
* Mood control ( `<mood_happy>`, `<mood_angry>` … )
* Stage directions (`<stage_whisper>`, `<shout>` )
* Scene markers (`<scene_tavern>`, `<scene_night>` )
* Content tags (`<nsfw_soft>`, `<nsfw_fade>` )

Tokens must appear plenty in training data, must be present in tokenizer, and must be exported so runtime can parse.

## Guard-rails & Gotchas
* If **`utils/world.py`** is not present yet, _block_ PR and check R1-1 task status.  Tokens live in world scope.
* Any new token must trigger tokenizer patch **once per base model**; cache patched tokenizers in `.cache/tokenizers`.
* Legacy datasets without tokens should still load (fallback: empty `tokens.json`).
* Fine-tuning: ensure `DataCollatorForLanguageModeling` does **NOT** mask special tokens (`mlm=False` covers this but double-check).

## Suggested Implementation Steps
```text
1. Design `core_tokens.json` (example below).
2. Write patch_tokenizer.py:
   - load tokenizer
   - for t in core_tokens if t not in tokenizer.get_vocab(): tokenizer.add_tokens([t])
   - save as <orig_name>-patched
3. Modify WorldManager.create_world() to copy default core_tokens.json unless overridden.
4. In utils/generation/prompt_builder:
   def apply_control_tokens(text, chosen_tokens):
       # insert NL semantics header once
5. Interactive UI:
   • st.selectbox("Insert token", tokens, on_change=append_to_prompt)
6. Update dataset stats & warnings (page_dataset_preview).
```

### Example `core_tokens.json`
```json
[
  {"token": "<mood_happy>",  "category": "mood",  "description": "character speaks in a cheerful tone", "ui_icon": "😊"},
  {"token": "<mood_angry>",  "category": "mood",  "description": "character is irritated or shouting",   "ui_icon": "😠"},
  {"token": "<stage_whisper>", "category": "stage", "description": "spoken softly / whispering",       "ui_icon": "🤫"}
]
```

## Unit Tests
- `tests/test_tokenizer_patch.py` ensures tokens added and can be encoded/decoded round-trip without loss.
- `tests/test_prompt_builder_tokens.py` builds prompt with `<mood_happy>` and asserts descriptive header present.

## Roll-out Plan
1. Merge behind feature flag `ENABLE_CONTROL_TOKENS = False` in `.env`.  
2. After verifying dataset generation & training with a small character, flip flag to `True`.

## References
* Overview §4 Vision
* Conversation 2025-06-18 "special tokens discussion"
* HuggingFace tokenizer add_tokens docs 