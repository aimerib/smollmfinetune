---
# R1-4  Revamp Character Management UI
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Replace current "Character Upload" page with a **Character Management Studio** that works with new CharacterCore folders and lives under the selected world.

## Page Location
`app/pages/character_management.py` (called from sidebar).

## Tabs & Widgets
| Tab | UI Elements | AI-Assist Actions |
|-----|-------------|-------------------|
| Profile | text_inputs for name, description, scenario; large textarea for back-story; appearance textarea | "Rewrite description" (LLM paraphrase), "Suggest back-story" |
| Personality | R1-5 radar component + numeric sliders (0.0-1.0, step 0.05) | "Auto-estimate from examples" button invokes `llm_estimate_big5` |
| Goals & Relationships | Goals: editable list with add/remove rows; Relationships: table (target char dropdown, affinity slider) | "Brainstorm 3 new goals" (LLM) |
| Examples | file_uploader for `mes_example.txt`; inline code editor if file already exists; show token count | "Generate additional example" button (uses prompt builder) |

### Global Toolbar
• Save ✔️
• Duplicate ↩️ (clone character)  
• Delete 🗑️ (with confirmation)

### Inline Suggestion UX
Whenever a text field is empty or cursor is inside field, show small "✨" button; clicking calls LLM, displays 2–3 suggestions inline; acceptance logs preference event.

## Acceptance Criteria
- [ ] Selecting a character loads all four tabs with data from `character_core.json` + example file.
- [ ] Drag-and-drop SillyTavern card triggers auto-convert workflow (see R1-2) and opens editor.
- [ ] Save writes JSON & txt; increments `meta.version` in world lore.
- [ ] Unsaved changes prompt when navigating away.
- [ ] Preference events emitted for each AI suggestion (same schema as R1-3).
- [ ] Unit test: create dummy CharacterCore, open page via Streamlit testing API, change slider, save, reload, value persists.

## Implementation Notes
```python
# example field writer
if st.button("✨ Suggest", key="descr_ai"):
    suggestions = llm_suggest_description(core)
    chosen = st.radio("Pick one", suggestions)
    if st.button("Use", key="descr_use"):
        core.description = chosen
        log_preference_event(...)
```

## Steps
1. New page skeleton; add to sidebar under world section.
2. Implement Profile tab.
3. Integrate personality editor component (R1-5) + radar chart.
4. Build Goals & Relationships tables with `st.data_editor`.
5. Examples tab with file read/write.
6. AI helper buttons (use existing OpenAI client).
7. Save & diff logic.
8. Unit test.

## Dependencies
- R1-2 CharacterCore structure
- R1-5 radar component
- Preference logging standard from R1-3.

## References
- Depends on R1-2 & R1-5 for radar. 