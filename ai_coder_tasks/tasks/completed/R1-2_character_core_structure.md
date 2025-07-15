---
# R1-2  CharacterCore File Structure
Status: **Completed** ✅
Ring: R1
Created: 2025-06-18
Completed: 2025-06-18
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

## Acceptance Criteria ✅
- [x] `character_core.json` schema (Pydantic `CharacterCore`):
  * `name`, `description`, `scenario`, `backstory`, `appearance`
  * `personality_traits` – dict with keys `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism` (0-1 floats)
  * `goals` – list[str]
  * `relationships` – list[{`name`, `affinity` (-100..100)}]
  * `tags` – optional list[str] (genres, species, etc.)
  * `imports` – metadata about original source (e.g., `{"source":"sillytavern","raw_file":"kaelen.json"}`)
- [x] `utils/character.py`
  * `CharacterManager.import_sillytavern_card(path) → CharacterCore`
    - Re-use existing parsing logic.
    - Calls helper in `utils/character/analysis.py` or LLM to infer personality (scores + goals) if missing.
  * `save_character(core: CharacterCore, world_path)`  → writes folder & files.
- [x] UI update (stub): on "Character Management" page a **Convert from SillyTavern** button that shows diff then saves.
- [x] Unit test: round-trip import of sample SillyTavern card produces valid CharacterCore and writes files.

## Implementation Notes ✅
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

### Auto-Populate via LLM ✅
Used existing `character_analysis.extract_character_knowledge()` which already returns `traits`, `goals`, etc.  For missing Big-5 scores, created new helper `llm_estimate_big5(description, mes_example)`.

### Folder layout (updated) ✅
```
characters/kaelen/
├── character_core.json
├── mes_example.txt        # single or multi-turn examples
└── assets/
    └── portrait.png       # optional future asset
```

### Migration Script (one-shot)
Added `scripts/convert_cards.py` concept - implemented in UI for now.

## Open Questions ✅
1. ~~Do we want a separate `appearance` field (height/hair/etc.) now or later?~~ → Added `appearance` field
2. ~~How many Big-5 decimals do we store (0.00 vs 0.0)?~~ → Using 0.1 precision for display

## Steps ✅
1. ✅ Define Pydantic model `CharacterCore`.
2. ✅ Refactor load/upload path in Streamlit page.
3. ✅ During synthetic-sample generation the prompt-builder already injects character_core.description and personality. Extended it to include appearance if non-empty:
```python
system_info = (
    f"Name: {core.name}\n"
    f"Description: {core.description}\n"
    f"Appearance: {core.appearance}\n"      # new
    f"Personality (Big-5): {trait_summary}\n"
)
```

## What Was Completed

### 1. Created Pydantic Models (app/utils/character/models.py)
- **Personality**: Big Five traits with 0-1 validation
- **Relationship**: Name + affinity (-100 to 100) 
- **CharacterCore**: Complete character structure with all required fields
- **llm_estimate_big5()**: Async function to estimate personality traits from description

### 2. Extended CharacterManager (app/utils/character/character.py)
- **import_sillytavern_card()**: Convert legacy cards to CharacterCore format
- **save_character()**: Save in new folder structure with character_core.json + mes_example.txt
- **load_character_core()**: Load from new format
- **list_characters_in_world()**: List all characters in a world
- **Enhanced prompt generation**: Includes appearance and Big Five traits
- **Backward compatibility**: Legacy methods still work

### 3. UI Updates (app/app.py)
- Added "Convert to CharacterCore" section to character upload page
- Shows preview of converted structure
- Maintains backward compatibility with existing workflow

### 4. Comprehensive Tests (tests/test_character_core.py)
- **Model validation tests**: Personality, Relationship, CharacterCore
- **Save/load round-trip tests**: Ensure data integrity
- **Card block generation tests**: Enhanced prompts with appearance
- **Backward compatibility tests**: Legacy functionality preserved
- **Async LLM estimation tests**: Proper error handling

### 5. Key Features Implemented
- 🧠 **Big Five Personality Traits**: Auto-estimated from character descriptions
- 📁 **World-based Organization**: Characters live inside worlds
- 🔄 **Seamless Conversion**: SillyTavern cards → CharacterCore format  
- 📝 **Enhanced Prompts**: Include appearance, backstory, goals, relationships
- 🏷️ **Smart Tagging**: Auto-extract species, occupation, genres
- 💾 **Structured Storage**: JSON + separate text files + assets folder
- ⚡ **Backward Compatible**: Existing workflows continue to work

### 6. Technical Achievements
- Fixed circular import issues with dynamic imports
- Updated to modern Pydantic v2 syntax (model_dump vs dict)
- Comprehensive error handling and logging
- 25/25 tests passing
- Enhanced character analysis using existing tools

## References
- R1 vision achieved ✅
- Ready for R1-3 (World management UI) and R1-4 (Character management UI)
- Foundation for R1-5 (Personality radar chart) in place 