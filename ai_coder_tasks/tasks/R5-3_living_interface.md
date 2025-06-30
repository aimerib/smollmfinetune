---
# R5-3  Living Interface (Director's View)
Status: **Todo**
Ring: R5
Created: 2025-06-19
Updated: 2025-01-16 (Triple-Head Architecture Integration)
---

## Goal
Deliver a real-time "Director's View" Streamlit page that visualizes world state, agent subtext (R5-2), live actions, and **triple-head architecture outputs** (generation quality, emotional control, memory formation), turning the platform into an interactive god-view debugging and storytelling console.

## Context
Writers need to observe and steer the emergent simulation. A graphical, auto-refreshing dashboard that surfaces subtext, relationships, actions, and **real-time triple-head model outputs** is critical for validating R5 proactive agents and ecology behavior. The interface must provide insights into all three model heads: generation content, emotional control, and memory formation.

## 3. How? (The Implementation)
This will be a new page in the Streamlit application that reads data asynchronously 
from the `RuntimeStateManager` and triple-head model outputs.

1. **Create the Enhanced `Director's View` Page:**
   - Add a new page: `app/pages/directors_view.py`.
   - This page will establish connections to the `RuntimeStateManager` and triple-head model monitoring.

2. **Live World State Map with Triple-Head Indicators:**
   - The main component will be a visual map with real-time triple-head status indicators.
   - Use `streamlit-agraph` or `plotly` for node-graph visualization.
   - Periodically fetch latest state and triple-head outputs.
   - **Node Styling**: 
     - Node color = overall health across all three heads
     - Border thickness = generation head performance
     - Glow effect = emotional control head activity
     - Badge icons = memory formation activity
   - Clicking on an agent node opens detailed triple-head analysis.

3. **Enhanced Agent Detail Sidebar with Triple-Head Panels:**
   - When an agent is selected, sidebar displays:
       - **Generation Head Panel**: Content quality metrics, response coherence scores
       - **Control Head Panel**: Active emotional state, control token usage, mood tracking
       - **Memory Head Panel**: Recent memory formations, memory retrieval patterns
       - Traditional state info (inventory, status, goals)
       - Live-updating action log with head-specific annotations

4. **NEW: Triple-Head Model Monitoring Dashboard:**
   - **Generation Head Monitor**: Response quality, creativity scores, factual accuracy
   - **Control Head Monitor**: Emotional appropriateness, personality consistency, control token effectiveness
   - **Memory Head Monitor**: Memory formation rate, retrieval accuracy, long-term consistency
   - **Cross-Head Coordination**: Visualization of how the three heads work together

5. **Asynchronous Triple-Head Data Fetching:**
   - Non-blocking updates for all three model heads
   - Separate refresh rates for different data types
   - WebSocket connections for real-time triple-head model output streaming

## 4. How to Test?

- **UI Tests (`tests/ui/test_directors_view.py`):**
   - Create an `AppTest` for the new page.
   - Mock the `RuntimeStateManager` interface.
   - In the test setup, have the mock `StateManager` return a predefined set of 
   agents and locations.
   - Test that the graph visualizer renders the correct number of nodes.
   - Simulate clicking a node and test that the sidebar appears and displays the 
   correct information from the mock state.
   - Test the auto-refresh mechanism by updating the mock `StateManager`'s data and 
   asserting that the UI reflects the change after a refresh cycle.
   
## Acceptance Criteria
- [ ] **Page**: `app/pages/directors_view.py` registered in navigation.
- [ ] **Graph Visualiser**: Plotly / streamlit-agraph node-graph rendering locations + agents, refresh ≤5 s.
- [ ] **Sidebar Detail**: On node click, sidebar shows agent state, last 5 actions, last 5 subtext lines (from `StateManager.get_subtext`).
- [ ] **Auto-Refresh**: Uses async polling without blocking UI; CPU usage <30 % on MacBook.
- [ ] **Search & Filter**: Text box filters nodes by name/id; checkbox toggles NPC/PC visibility.
- [ ] **Unit/UI Tests**: `tests/ui/test_directors_view.py` with mocked `StateManager`; asserts graph node count, sidebar content, refresh cycle.
- [ ] **NEW - Memory Visualization**:
  - [ ] Memory formation bubbles appear above characters during significant moments
  - [ ] Color coding: warm colors (positive valence), cool colors (negative valence)
  - [ ] Bubble size represents memory strength/importance
  - [ ] Opacity indicates persistence likelihood
  - [ ] Floating emotion icons (😊, 😨, 💕, 😤) for quick recognition
- [ ] **NEW - Emotional State Panel**:
  - [ ] Real-time emotional momentum tracker showing active emotions
  - [ ] Decay visualization showing how emotions fade over turns
  - [ ] Surprise score meter (0-1 scale) with recent spike history
- [ ] **NEW - Memory Timeline**:
  - [ ] Scrollable timeline of formed memories for selected agent
  - [ ] Filter by memory type (episodic, semantic, emotional, procedural)
  - [ ] Click memory to see full details and formation context

## Implementation Notes
```text
• Expose websocket or polling endpoint in RuntimeStateManager returning JSON snapshot.
• Use `st.experimental_rerun` guarded by `time.sleep` for simple refresh or `asyncio` + `st.experimental_data_editor` if available.
• Derive node colours: player=blue, NPC=grey, hostile=red.
• Memory Visualization:
  - Subscribe to memory formation events from NarrativeLLM
  - Use Plotly animations for floating bubbles
  - Store last 100 memories per agent in circular buffer
  - Color map: HSL where H=120*(valence+1)/2 (red to green)
• Emotional State:
  - Poll model.get_emotional_state_summary() every update cycle
  - Use st.progress bars for decay visualization
  - Sparkline chart for surprise score history
• TDD Instructions:
  - Red (Orchestrator): In tests/orchestrator/test_tool_handling.py, simulate the 
  model generating a SetUIAttribute tool call. Assert that the Orchestrator correctly 
  parses it and adds a corresponding dictionary to a mocked session_state.ui_commands 
  list.
  - Green (Orchestrator): Implement the interception and queueing logic in the 
  Orchestrator.
  - Red (UI): In tests/ui/test_living_interface.py, use AppTest to run the main app. 
  Pre-populate the session_state.ui_commands queue with a command to change an accent 
  color. Assert that the final rendered HTML contains a <style> tag with the expected 
  CSS variable override.
  - Green (UI): Implement the frontend handler in app.py to process the queue and 
  inject the CSS.
  - Repeat for the PresentItem command, asserting that a new button with the 
  specified icon appears in the rendered output.
```

## Checklist / Steps
1. Add REST/ws endpoint `/_snapshot` in StateManager API.
2. Build graph component util in `components/world_graph.py`.
3. Implement page with sidebar interactions.
4. Write mock StateManager fixture + UI tests.
5. Add nav entry icon 🎬 and tooltip.
6. Define tool schemas for UI manipulation
7. Update Orchestrator to intercept UI tool calls
8. Implement UI command queueing system
9. Create frontend handler for processing UI commands
10. Add CSS injection and dynamic element creation
11. Implement expressive typography with control tokens
12. Write comprehensive tests for all UI interactions
13. Add sidebar and theme manipulation capabilities
14. **NEW**: Create memory bubble animation component
15. **NEW**: Implement emotional state tracking panel
16. **NEW**: Build memory timeline visualization
17. **NEW**: Add memory formation event system
18. **NEW**: Create surprise score visualization widget

## References
Consumes subtext from R5-2, world state from R4-12, actions from R4-13, **memory formation from R4-5**.