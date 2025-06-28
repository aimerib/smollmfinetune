# R5-1: The Narrative Scripting Engine (N-Script)

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R4-12, R4-13

---

## 1. Goal

To create a simple, event-driven scripting system that allows designers to define deterministic narrative logic (e.g., "if-this-then-that") that interacts with the emergent, agent-driven world.

---

## 2. Why? (The Story)

While we want an emergent world, designers still need control. They need to be able to write "if-this-then-that" logic for critical story moments, tutorials, or safety rails. For example: "IF the player enters the Dragon's Lair for the first time, THEN force the Dragon to deliver its monologue." N-Script is the tool that gives designers this power, blending scripted events with the emergent simulation. It's the safety net and the director's tool for our AI improv actors.

---

## 3. How? (The Implementation)

N-Script will be implemented as a set of components that integrate with the `AgenticLoopFramework`. The initial version will use a simple YAML-based DSL for readability and ease of parsing.

1.  **Define the N-Script DSL (Domain Specific Language):**
    -   Design a human-readable YAML syntax for scripts.
    -   The script will define `Triggers` and `Actions`.
    -   **Example `dragon_monologue.nscript`:**
        ```yaml
        script_id: dragon_monologue_intro
        trigger:
          type: ON_ENTER_LOCATION
          location_id: "dragon_lair_entrance"
          actor_filter: "player" # Only triggers for the player
          once: true # This script only runs once per campaign
        actions:
          - type: FORCE_ACTION
            target_agent_id: "dragon_boss"
            action:
              # This uses the same Action Schema from R4-13
              type: SpeakToAction
              message: "Frail mortal, you have stumbled into my domain..."
          - type: SET_STATE
            target_agent_id: "dragon_boss"
            state_patch:
              # This directly modifies the agent's state in the StateManager
              status: "hostile"
        ```

2.  **Create the `ScriptManager`:**
    -   This service resides in `narrative_engine/nscript.py`.
    -   It loads all `.nscript` files from a world's `/scripts` directory at startup.
    -   It uses a library like `PyYAML` to parse them into validated `Script` data objects.

3.  **Implement the `TriggerMonitor`:**
    -   Also in `narrative_engine/nscript.py`, this component subscribes to the event stream from the `RuntimeStateManager` (R4-12).
    -   When an event occurs (e.g., state change, location change), it checks if any loaded script's trigger conditions are met.

4.  **Implement the `ActionExecutor`:**
    -   When a trigger is matched, the `TriggerMonitor` passes the corresponding `actions` list to the `ActionExecutor`.
    -   The `ActionExecutor` is responsible for executing the action.
        -   For a `FORCE_ACTION`, it directly calls the `agent.act()` method from the `AgenticLoopFramework` (R4-13), bypassing that agent's `perceive` and `think` steps for that one tick.
        -   For a `SET_STATE`, it calls the `StateManager.update_state()` method directly.

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_nscript.py`):**
    -   Test the N-Script parser: can it correctly parse valid YAML script files and reject invalid ones (e.g., missing required keys)?
    -   Test the `TriggerMonitor`: given a state change event from a mock `StateManager`, does it correctly identify and fire the matched script?
    -   Test the `ActionExecutor`: does it make the correct calls to the (mocked) agent and `StateManager` interfaces based on the action list?
-   **Integration Test:**
    -   Create a test scenario with the `AgenticLoopFramework`. Let an agent move into a location, which should fire an N-Script trigger from a loaded test script.
    -   Assert that the trigger forces another agent to perform a specific action.
    -   Verify the final state in the `StateManager` is correct.

## Acceptance Criteria
### DSL & Parsing
- [ ] YAML-based `.nscript` schema with JSON-Schema validation file in `narrative_engine/nscript_schema.json`.
- [ ] Parser `ScriptManager.load_scripts(world_path)` raises `ScriptValidationError` on invalid file.

### Trigger Runtime
- [ ] `TriggerMonitor` subscribes to `StateManager.event_bus`; latency <50 ms/event.
- [ ] Opt-in debug logs showing matched script id & actions executed.

### Action Execution
- [ ] Support at minimum `FORCE_ACTION`, `SET_STATE`, `EMIT_EVENT` action types.
- [ ] Unit tests reaching 90 % branch coverage across parser, monitor, executor.

### Integration & Tooling
- [ ] Example script folder added to `content/worlds/Default World/scripts/` with at least two scripts including `dragon_monologue_intro`.
- [ ] Docs page `docs/nscript.md` explaining syntax + examples.
- [ ] Streamlit "Scripts" tab in World Management page listing scripts & validation status.

## Implementation Notes
```text
• Use pydantic v2 for in-memory Script objects.
• Leverage watchdog to reload scripts at runtime when files change (hot-reload for designers).
• Ensure thread-safe interaction with StateManager via asyncio.Queue event bus.
```

## Checklist / Steps
1. Define JSON-Schema + pydantic models.
2. Implement ScriptManager load & validation.
3. Build TriggerMonitor + ActionExecutor.
4. Write unit tests with pytest fixtures.
5. Create example scripts & documentation.
6. Add Scripts tab to world_management UI.
7. Achieve >90 % test coverage then move card to completed.

## References
Builds on R4-12 StateManager and R4-13 Agentic Loop.