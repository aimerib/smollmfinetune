---
# R5-3: The Living Interface
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Enable the Narrative-LLM to directly influence and manipulate the user interface by emitting special tool calls, making the platform itself feel like a magical, responsive part of the story world.

## Context
Inspired by a Disney Imagineer's approach, this feature dissolves the boundary between the narrative and the interface. The character is no longer confined to a text box but can change the application's theme, present items to the user, and use typography to express emotion, creating a deeply immersive and delightful experience.

## Acceptance Criteria
- [ ] New Tool Definitions: Define the schemas for new tool calls in `narrative_engine/api/tool_schema.py`:
  - [ ] `SetUIAttribute(theme: str, accent_color: str, font_style: str)`
  - [ ] `PresentItem(item_name: str, item_id: str, icon: str)`
- [ ] Orchestrator Update: The Orchestrator is modified to intercept these specific UI-related tool calls. Instead of sending them to a game engine, it places them into a dedicated command queue in the user's session state (e.g., `st.session_state.ui_commands`).
- [ ] Frontend Handler: The main Streamlit application (`app.py`) includes a handler function that runs at the start of every page render. This function checks the `ui_commands` queue, executes the commands (e.g., by injecting CSS with `st.markdown` or by adding new elements to a sidebar), and then clears the queue.
- [ ] Expressive Typography: The chat display logic is updated to check for control tokens like `<stage_whisper>` or `<shout>` and apply inline CSS to modify the font size and style of that specific message.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Orchestrator): In tests/orchestrator/test_tool_handling.py, simulate the model generating a SetUIAttribute tool call. Assert that the Orchestrator correctly parses it and adds a corresponding dictionary to a mocked session_state.ui_commands list.
  - Green (Orchestrator): Implement the interception and queueing logic in the Orchestrator.
  - Red (UI): In tests/ui/test_living_interface.py, use AppTest to run the main app. Pre-populate the session_state.ui_commands queue with a command to change an accent color. Assert that the final rendered HTML contains a <style> tag with the expected CSS variable override.
  - Green (UI): Implement the frontend handler in app.py to process the queue and inject the CSS.
  - Repeat for the PresentItem command, asserting that a new button with the specified icon appears in the rendered output.
```

## Checklist / Steps
1. Define tool schemas for UI manipulation
2. Update Orchestrator to intercept UI tool calls
3. Implement UI command queueing system
4. Create frontend handler for processing UI commands
5. Add CSS injection and dynamic element creation
6. Implement expressive typography with control tokens
7. Write comprehensive tests for all UI interactions
8. Add sidebar and theme manipulation capabilities

## References
Builds on the tool-use framework from R4. 