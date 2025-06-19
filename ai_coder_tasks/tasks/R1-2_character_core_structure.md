---
# R1-2  CharacterCore File Structure
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Move from single SillyTavern JSON to multi-file character folder inside a world.
```
characters/<char_name>/
├── character_core.json   # static facts, personality, goals
├── mes_example.txt       # example dialogue
└── assets/ (future)
```

## Context
We need two things:
1. A **new canonical format** (`character_core.json` + companions) for all characters _inside a world_.
2. A backward-compat import path so creators can drag-and-drop a SillyTavern card (or older JSON) and have the devkit auto-convert it into this new folder layout, using existing helper functions + LLM extraction for richer fields.

## Acceptance Criteria
- [ ] `character_core.json` schema (Pydantic `CharacterCore`):
  * `name`, `description`, `scenario`, `backstory`, `appearance`
  * `personality_traits` – dict with keys `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism` (0-1 floats)
  * `goals` – list[str]
  * `relationships` – list[{`name`, `affinity` (-100..100)}]
  * `tags` – optional list[str] (genres, species, etc.)
  * `imports` – metadata about original source (e.g., `{"source":"sillytavern","raw_file":"kaelen.json"}`)
- [ ] `utils/character.py`
  * `CharacterManager.import_sillytavern_card(path) → CharacterCore`
    - Re-use existing parsing logic.
    - Calls helper in `utils/character/analysis.py` or LLM to infer personality (scores + goals) if missing.
  * `save_character(core: CharacterCore, world_path)`  → writes folder & files.
- [ ] UI update (stub): on "Character Management" page a **Convert from SillyTavern** button that shows diff then saves.
- [ ] Unit test: round-trip import of sample SillyTavern card produces valid CharacterCore and writes files.

## Implementation Notes
```python
from pydantic import BaseModel
class Personality(BaseModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

class Relationship(BaseModel):
    name: str
    affinity: int  # -100..100

class CharacterCore(BaseModel):
    name: str
    description: str
    scenario: str = ""
    backstory: str = ""
    personality_traits: Personality = Personality()
    goals: list[str] = []
    relationships: list[Relationship] = []
    tags: list[str] = []
    imports: dict = {}
    appearance: str = ""
```

### Auto-Populate via LLM
Use existing `character_analysis.extract_character_knowledge()` which already returns `traits`, `goals`, etc.  For missing Big-5 scores, call new helper `llm_estimate_big5(description, mes_example)`.

### Folder layout (updated)
```
characters/kaelen/
├── character_core.json
├── mes_example.txt        # single or multi-turn examples
└── assets/
    └── portrait.png       # optional future asset
```

### Migration Script (one-shot)
Add `scripts/convert_cards.py` to batch-convert all old `.json` cards in a folder for early adopters.

## Open Questions
1. Do we want a separate `appearance` field (height/hair/etc.) now or later?
2. How many Big-5 decimals do we store (0.00 vs 0.0)?

## Steps
1. Define Pydantic model `CharacterCore`.
2. Refactor load/upload path in Streamlit page.
3. During synthetic-sample generation the prompt-builder already injects character_core.description and personality. Extend it to include appearance if non-empty:
```python
system_info = (
    f"Name: {core.name}\n"
    f"Description: {core.description}\n"
    f"Appearance: {core.appearance}\n"      # new
    f"Personality (Big-5): {trait_summary}\n"
)
```

## References
- R1 vision. 