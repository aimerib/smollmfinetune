---
# R1-1  WorldManager & Project Layout
Status: **Completed** ✅
Ring: R1
Created: 2025-06-18
Completed: 2025-06-18
---

## Goal
Introduce a `WorldManager` that handles a directory like:
```
worlds/
└── my_world/
    ├── world_lore.json
    └── characters/
        └── <char>/ (see R1-2)
```
so authors can create/select worlds inside the devkit.

## Context
We need a top-level container for shared lore (facts the LLM can't contradict).  Current app treats each card as isolated.

## Acceptance Criteria
- [x] `utils/world.py` with class `WorldManager` (load, save, list_worlds, create_world).
- [x] Default storage path `content/worlds/` (configurable by env).
- [x] Support **optional nested faction timelines** in the data model (backend only; no UI yet).
- [x] Auto-increment `meta.version` when saving world lore.
- [x] Unit test: create temp dir, create world, write lore with faction timeline, reload == same.

## Implementation Notes
### Data Model Additions
```jsonc
world_lore.json minimal example
{
  "meta": {"version": 1},
  "facts": {"magic_system": "Elemental runes"},
  "factions": [
    {
      "name": "Moonfall",
      "timeline": [
        {"year": 310, "event": "Moonfall rebellion"}
      ]
    }
  ],
  "timeline": [
    {"year": 300, "event": "Founding of Solaris"}
  ],
  "places": [
    {"name": "Whispering Peaks", "description": "Tall, misty...", "npcs": [{"name": "Elara", "description": "Queen of the Dwarven folk, Elara..."}], "events": [{"name":"Moonlight Festival", "description": "Every 100 years the Dwarven folk celebrate...", "random": "true"}]}
  ]
}
```

## Steps
1. Add new file and class.
2. Wire into `CharacterManager` (if active world not set, use default world).

## References
- Overview §R1. 

## UX & AI-Assisted World-Building
### Creator Flow (happy-path)
1. **Create World** – Author clicks "➕ New World", enters a name, chooses a colour banner/icon.
2. **Lore Dashboard** – Shows three editable tabs:
   • Facts (key/value)  
   • Timeline  
   • Places of Interest
3. **AI Helper buttons** (powered by existing generation client):
   * "Draft back-story" – feeds current facts to LLM, returns paragraph suggestions.
   * "Generate 10 conflict ideas" – for plot hooks.
   * "Fill empty fields" – LLM proposes values for blank keys.
4. **Places of Interest** (powered by existing generation client): 
   * Creators can add a name a description of the place, and optionally a list of NPCs (non player and non DoRA/LoRA characters) and their descriptions, a list of likely events and their descriptions to happen here (nudge to the llm, not necessarily plot beats). This should also have options for LLM assisted generation of content to help content creators.
   * LLM assisted generation takes into consideration any available already existing information about the world to help with this.
5. **Save → WorldManager.write_lore()** and refresh sidebar count of characters.


### AI Prompt Snippets
*Back-end helper in `utils/world_ai.py`*
```python
def suggest_fact(prompt, client):
    return await client.generate(prompt, max_tokens=120, temperature=0.8)
```

---
## ✅ COMPLETION SUMMARY

**What was completed:**

1. **WorldManager Implementation** (`app/utils/world.py`):
   - Full dataclass-based data model for WorldLore, Factions, Places, NPCs, etc.
   - Complete CRUD operations: create_world, load_world, save_world_lore, list_worlds
   - Auto-versioning system for world lore updates
   - Proper JSON serialization/deserialization with orjson
   - Configurable worlds root path (defaults to `content/worlds/`)

2. **CharacterManager Integration** (`app/utils/character/character.py`):
   - WorldManager dependency injection in constructor
   - Automatic default world creation and management
   - World context injection in character card blocks (`<WORLD_CONTEXT>`)
   - Methods for world switching and lore retrieval
   - Enhanced character summaries with current world info

3. **Comprehensive Test Suite** (`tests/test_world_manager.py`):
   - 8 test cases covering all WorldManager functionality
   - Complex faction timeline and place data serialization testing
   - CharacterManager integration testing
   - Proper test isolation with temporary directories
   - World context injection testing

4. **Project Structure Improvements**:
   - Updated overview.md with testing standards and TDD guidelines
   - Clarified that ALL tests go in `/tests/` directory
   - Marked codebase as "TDD-Ready" with proper separation of concerns
   - Updated roadmap status (R0 complete → R1 in progress)

**Key Features:**
- **Structured World Lore**: Facts, timelines, factions with nested timelines, places with NPCs and events
- **Version Management**: Auto-incrementing version numbers on save
- **Character Integration**: Characters now operate within world contexts
- **Test-Driven Foundation**: Comprehensive test coverage enables confident future development

**TDD Assessment:** ✅ **YES, the codebase is now TDD-ready**
- Modular, well-separated components
- Comprehensive test coverage patterns established  
- Clear interfaces and dependency injection
- Proper mocking capabilities for external dependencies
- Test infrastructure in place for rapid red-green-refactor cycles

**Next Steps**: Ready for R1-2 (Character core structure) and R1-3 (World management UI)
