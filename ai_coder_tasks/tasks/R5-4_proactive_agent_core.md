# R5-4: Proactive Agent Core

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Large
- **Related-Tasks:** R4-13, R4-12, R4-6

---

## 1. Goal

To implement the first concrete version of a proactive agent, `ProactiveAgent`, which uses the `AgenticLoopFramework` to autonomously pursue goals within the simulated world, leveraging the full triple-head architecture for generation, emotional control, and memory formation.

---

## 2. Why? (The Story)

This is where the magic happens. A `ProactiveAgent` is a character with a will of its own. Instead of waiting for user input, it has internal goals (e.g., "Find the missing artifact," "Become friends with Character B") and actively works to achieve them. This task turns our characters from reactive puppets into self-motivated actors, which is the foundational technology for creating a truly living world. The triple-head architecture enables characters to not only act and speak, but also form persistent memories and maintain emotional states.

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
        -   **Recent memories**: Query the agent's memory bank for relevant context.
        -   **Current emotional state**: Retrieve emotional momentum from previous interactions.
    -   This information is bundled into a `Perception` data object.

3.  **Implement `think()` Method:**
    -   This is the core logic. It constructs a detailed prompt for the triple-head Narrative Engine (R4-6).
    -   The prompt will include:
        -   **Personality:** The agent's Big Five traits.
        -   **Goals:** A list of the agent's long-term and short-term goals from its state.
        -   **Perception:** The formatted context from the `perceive()` step.
        -   **Memory Context:** Relevant memories to inform decision-making.
        -   **Instructions:** A call to action, asking the model to decide on the best next action to make progress towards its goals, and to provide subtext for that action.
    -   It then calls the triple-head Narrative Engine, and processes all three outputs:
        -   **Generation Head**: Parses into dialogue/action text
        -   **Control Head**: Extracts emotional state and control tokens
        -   **Memory Head**: Processes memory formation signals
    -   Returns an enhanced `ThinkResult(action, subtext, emotional_state, memory_formation)` object.

4.  **Implement `act()` Method:**
    -   This method receives the `Action` object from the `think()` step.
    -   It acts as a dispatcher. It translates the high-level `Action` into one or more low-level calls to the `RuntimeStateManager`.
    -   For `MoveToAction(location)`, it calls `state_manager.update_state(self.id, {"location": location})`.
    -   For `SpeakToAction(target, msg)`, it creates a new "speech" event in the `StateManager`'s event log.
    -   **Memory Processing**: If memory formation is triggered, store new memories in the agent's memory bank.
    -   **Emotional State Update**: Update the agent's emotional momentum based on control head outputs.

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_proactive_agent.py`):**
    -   This class requires extensive unit testing.
    -   **Test `perceive`:** Mock the `StateManager` and verify that the agent constructs the correct `Perception` object based on the mock data, including memory and emotional context.
    -   **Test `think`:** Mock the triple-head Narrative Engine. Verify that the agent constructs the correct prompt based on its state and perception. Provide canned responses for all three heads and assert correct parsing into `ThinkResult`.
    -   **Test `act`:** For each type of `Action`, verify that the `act` method makes the correct corresponding calls to the mocked `StateManager`, including memory storage and emotional state updates.
    -   **Test Memory Integration**: Verify that memory formation from the memory head is properly processed and stored.
    -   **Test Emotional Continuity**: Verify that emotional states persist and evolve across multiple think-act cycles.
-   **Integration Test:**
    -   Set up the `AgenticLoopFramework` with one `ProactiveAgent`. Run the scheduler for multiple ticks.
    -   Assert that the agent correctly perceived its environment, "thought" using all three heads, and modified the `StateManager`'s state according to its action.
    -   Verify that memories are formed and emotional states evolve over time.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Agent Loop): In tests/narrative_engine/test_agent_service.py, instantiate the AgentLoop with a dummy character. Mock the triple-head LLM client. Call the .run_single_cycle() method. Assert that the LLM was called with correctly formatted prompts and all three heads processed.
  - Green (Agent Loop): Implement the core Perceive-Reflect-Plan logic with triple-head support.
  - Red (Action): Write a test where the mocked triple-head output includes memory formation. Assert that memories are stored and emotional state is updated.
  - Green (Action): Implement the "Act" step logic with memory and emotional processing.
  - Red/Green (UI): In a UI test, verify that the Living Interface can display agent emotional states and memory formation in real-time.
```

## Checklist / Steps
1. Create AgentLoop class with cognitive cycle structure
2. Implement Perceive step to gather character/world data and memory context
3. Implement Reflect step with LLM-generated monologue using triple-head architecture
4. Implement Plan step with actionable planning and emotional state consideration
5. Implement Act step with state updates, messaging, memory formation, and emotional updates
6. Create proactive message queue/database with memory integration
7. Add UI notifications for proactive messages with emotional context
8. Implement asynchronous agent execution with memory persistence
9. **NEW**: Add memory formation processing from memory head outputs
10. **NEW**: Implement emotional state tracking and persistence
11. **NEW**: Add triple-head output validation and error handling
12. Write comprehensive tests for agent loop with all three heads

## References
Relies on character_core.json goals (R1-2), world_lore.json (R1-1), and triple-head Narrative Engine (R4-6).
Integrates with memory system (R4-5) and emotional control tokens (R4-6). 