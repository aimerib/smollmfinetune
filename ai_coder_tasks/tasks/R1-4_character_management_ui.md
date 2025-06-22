---
# R1-4a  Character Management Studio (Tabbed Interface)
Status: **In-Progress** 
Ring: R1
Created: 2025-06-18
Updated: 2025-01-14
---

## Goal
Replace current "Character Upload" page with a **Character Management Studio** that works with new CharacterCore folders and lives under the selected world, using a traditional tabbed interface with AI-powered suggestions.

## Context
This is Phase A of the character management system. Subsequent phases (R1-4b through R1-4e) will add conversational interfaces, live synthesis, world integration, and advanced intelligence. This phase establishes the solid foundation with all core functionality in a familiar tabbed format.

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
- [x] Selecting a character loads all four tabs with data from `character_core.json` + example file.
- [x] Drag-and-drop SillyTavern card triggers auto-convert workflow (see R1-2) and opens editor.
- [x] Save writes JSON & txt; increments `meta.version` in world lore.
- [ ] Unsaved changes prompt when navigating away.
- [x] Preference events emitted for each AI suggestion (same schema as R1-3).
- [ ] Unit test: create dummy CharacterCore, open page via Streamlit testing API, change slider, save, reload, value persists.
- [x] All AI suggestion functions implemented with structured outputs
- [x] Navigation integration with modern st.navigation system

## Implementation Notes
```python
# AI suggestions with structured outputs (COMPLETED)
async def llm_suggest_description(core: CharacterCore) -> List[str]:
    suggestions = await client.generate(prompt, response_format=DescriptionSuggestions.model_json_schema())
    return suggestions.suggestions

# UI pattern for suggestions (COMPLETED)
if st.button("✨", key="desc_ai_btn"):
    suggestions = asyncio.run(llm_suggest_description(core))
    st.session_state.desc_suggestions = suggestions
    # Show radio buttons for selection...
```

## Remaining Work
1. ~~New page skeleton; add to sidebar under world section.~~ ✅
2. ~~Implement Profile tab.~~ ✅
3. ~~Integrate personality editor component (R1-5) + radar chart.~~ ✅
4. ~~Build Goals & Relationships tables with `st.data_editor`.~~ ✅
5. ~~Examples tab with file read/write.~~ ✅
6. ~~AI helper buttons (use existing OpenAI client).~~ ✅
7. ~~Save & diff logic.~~ ✅
8. **Add unsaved changes detection** (remaining)
9. **Complete unit tests** (remaining)

## Next Phases
- **R1-4b**: Conversational Character Builder (guided AI interview)
- **R1-4c**: Live Character Synthesis (real-time character preview)
- **R1-4d**: Contextual World Integration (world-aware suggestions)
- **R1-4e**: Advanced Character Intelligence (preference learning, relationship mapping)

## Dependencies
- R1-2 CharacterCore structure ✅
- R1-5 radar component ✅
- Preference logging standard from R1-3 ✅
- Modern navigation system ✅

## References
- Depends on R1-2 & R1-5 for radar ✅
- UX Assessment: Foundation for conversational interfaces (R1-4b+) 