---
# R1-1  WorldManager & Project Layout
Status: **Todo**
Ring: R1
Created: 2025-06-18
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
- [ ] `utils/world.py` with class `WorldManager` (load, save, list_worlds, create_world).
- [ ] Default storage path `content/worlds/` (configurable by env).
- [ ] Support **optional nested faction timelines** in the data model (backend only; no UI yet).
- [ ] Auto-increment `meta.version` when saving world lore.
- [ ] Unit test: create temp dir, create world, write lore with faction timeline, reload == same.

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
