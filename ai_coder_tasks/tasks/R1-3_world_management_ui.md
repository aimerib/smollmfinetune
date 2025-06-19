---
# R1-3  🌍 World Management Page
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Add a new Streamlit sidebar option "🌍 World Management" where creators can:
1. Create / select a world.
2. Visual-edit `world_lore.json` (facts, timeline, places) and trigger AI-helper actions.

## Acceptance Criteria
- [ ] Page file `app/pages/world_management.py` registered in nav (uses `st.experimental_set_query_params` so deep-link works).
- [ ] Sidebar shows `selectbox` of worlds and a `➕ New World` button.
- [ ] Main area has **three tabs**: Facts | Timeline | Places.
  * **Facts**: AgGrid/data_editor for key-value pairs + AI buttons `Draft back-story`, `Fill empty fields`.
  * **Timeline**: editable table (`year`,`event`) with optional `faction` column (hidden if unused).  Add `Generate 5 historical events` button.
  * **Places**: list editable; each row opens an accordion to edit NPCs/events; AI helper `Suggest NPCs`.
- [ ] Changes are persisted via `WorldManager.write_lore()`; unsaved changes warning on page exit.
- [ ] When switching worlds, diff modal shows unsaved → saved delta.
- [ ] Unit test with `streamlit.testing` (or pure function test) ensures `write_lore()` called after UI edit simulation.

## Implementation Notes
```python
# skeleton
import streamlit as st
from utils.world import WorldManager
wm: WorldManager = st.session_state.world_manager

worlds = wm.list_worlds()
selected_world = st.sidebar.selectbox("World", worlds, index=0)
core = wm.load_world(selected_world)

if st.sidebar.button("➕ New World"):
    new_name = st.text_input("World name")
    if st.button("Create"):
        wm.create_world(new_name)
        st.rerun()
```

### Component helpers (put in `app/components/world_ui.py`)
* `edit_facts(core)` – returns updated dict.
* `edit_timeline(core)`
* `edit_places(core)`
All return tuple `(updated_core, modified_flag)`.

### AI helper calls
```python
from utils.world_ai import suggest_fact, suggest_events
async def on_suggest_events():
    new_events = await suggest_events(core.facts, n=5)
    core.timeline.extend(new_events)
```

### Diff modal snippet
```python
diff = DeepDiff(old_core.dict(), new_core.dict(), ignore_order=True)
if diff and st.sidebar.button("View unsaved changes"):
    st.json(diff, expanded=True)
```

## Steps
1. Create page stub and register in `render_sidebar` menu.
2. Implement Facts tab with data_editor.
3. Implement Timeline tab (table + AI generate btn).
4. Implement Places tab with nested edit dialog.
5. Wire save & diff logic.
6. Add unsaved-changes guard (`st.session_state._dirty = True`).
7. Write unit test for save.

### Preference Logging
- Inline suggestion component must emit an event:
  ```jsonc
  {
    "world": <world_name>,
    "field": "facts.timeline",   // dot path
    "prompt": <prompt_used_to_generate>,
    "options": ["optA","optB",...],
    "chosen": "optA",
    "timestamp": ISO8601
  }
  ```
  Stored in `content/worlds/<world>/preference_logs/` as NDJSON.
- `WorldManager.aggregate_preferences()` returns ready-to-train list of (prompt, chosen, rejected) tuples.
- Training page should display a badge:  _"🔄 234 preference pairs available – Enable PPO fine-tune?"_ 

## References
- Relies on R1-1. 