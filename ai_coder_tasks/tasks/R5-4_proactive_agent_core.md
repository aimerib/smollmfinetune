# R5-4: Proactive Agent Core

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Large
- **Related-Tasks:** R4-13, R4-12

---

## 1. Goal

To implement the first concrete version of a proactive agent, `ProactiveAgent`, which uses the `AgenticLoopFramework` to autonomously pursue goals within the simulated world.

---

## 2. Why? (The Story)

This is where the magic happens. A `ProactiveAgent` is a character with a will of its own. Instead of waiting for user input, it has internal goals (e.g., "Find the missing artifact," "Become friends with Character B") and actively works to achieve them. This task turns our characters from reactive puppets into self-motivated actors, which is the foundational technology for creating a truly living world.

---

## 3. How? (The Implementation)

This task involves creating a concrete implementation of the `BaseAgent` interface defined in R4-13.

1.  **Create `ProactiveAgent` Class:**
    -   In `narrative_engine/agent.py`, create the class `ProactiveAgent(BaseAgent)`.
    -   The `__init__` method will take the agent's core data (personality, goals) from the character file.

2.  **Implement `perceive()` Method:**
    -   This method will query the `RuntimeStateManager` to build a "context" for the agent.
    -   It will get:
        -   The agent's own state (`self.state`).
        -   The state of all other agents in the same location.
        -   A list of recent events that happened in the location.
    -   This information is bundled into a `Perception` data object.

3.  **Implement `think()` Method:**
    -   This is the core logic. It constructs a detailed prompt for the R4 Narrative Engine.
    -   The prompt will include:
        -   **Personality:** The agent's Big Five traits.
        -   **Goals:** A list of the agent's long-term and short-term goals from its state.
        -   **Perception:** The formatted context from the `perceive()` step.
        -   **Instructions:** A call to action, asking the model to decide on the best next action to make progress towards its goals, and to provide subtext for that action.
    -   It then calls the Narrative Engine, and parses the response into the `ThinkResult(action, subtext)` object, as defined in R5-2.

4.  **Implement `act()` Method:**
    -   This method receives the `Action` object from the `think()` step.
    -   It acts as a dispatcher. It translates the high-level `Action` into one or more low-level calls to the `RuntimeStateManager`.
    -   For `MoveToAction(location)`, it calls `state_manager.update_state(self.id, {"location": location})`.
    -   For `SpeakToAction(target, msg)`, it creates a new "speech" event in the `StateManager`'s event log.

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_proactive_agent.py`):**
    -   This class requires extensive unit testing.
    -   **Test `perceive`:** Mock the `StateManager` and verify that the agent constructs the correct `Perception` object based on the mock data.
    -   **Test `think`:** Mock the R4 model. Verify that the agent constructs the correct prompt based on its state and perception. Provide a canned model response and assert that it is parsed into the correct `ThinkResult`.
    -   **Test `act`:** For each type of `Action`, verify that the `act` method makes the correct corresponding calls to the mocked `StateManager`.
-   **Integration Test:**
    -   Set up the `AgenticLoopFramework` with one `ProactiveAgent`. Run the scheduler for one tick.
    -   Assert that the agent correctly perceived its environment, "thought" (using a canned model response), and modified the `StateManager`'s state according to its action.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Agent Loop): In tests/narrative_engine/test_agent_service.py, instantiate the AgentLoop with a dummy character. Mock the LLM client. Call the .run_single_cycle() method. Assert that the LLM was called twice (once for Reflect, once for Plan) with the correctly formatted prompts.
  - Green (Agent Loop): Implement the core Perceive-Reflect-Plan logic.
  - Red (Action): Write a test where the mocked "Plan" output is to send a message. Assert that the .run_single_cycle() method results in a new entry being added to a mocked proactive message queue.
  - Green (Action): Implement the "Act" step logic.
  - Red/Green (UI): In a UI test, pre-populate the proactive message queue. Assert that the application renders a notification element with the correct text.
```

## Checklist / Steps
1. Create AgentLoop class with cognitive cycle structure
2. Implement Perceive step to gather character/world data
3. Implement Reflect step with LLM-generated monologue
4. Implement Plan step with actionable planning
5. Implement Act step with state updates and messaging
6. Create proactive message queue/database
7. Add UI notifications for proactive messages
8. Implement asynchronous agent execution
9. Write comprehensive tests for agent loop

## References
Relies on character_core.json goals (R1-2) and world_lore.json (R1-1). 