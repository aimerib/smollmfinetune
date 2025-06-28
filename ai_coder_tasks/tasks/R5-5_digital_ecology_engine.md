# R5-5: Digital Ecology Engine (Relationship Manager)

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R4-12, R5-4

---

## 1. Goal

To implement a `RelationshipManager` that observes agent interactions and updates their relationship statuses in the `RuntimeStateManager`, creating a dynamic social fabric in the world.

---

## 2. Why? (The Story)

A world isn't just a collection of individuals; it's a web of relationships. Friends, rivals, lovers, and enemies. The `Digital Ecology Engine`, in its first version, is a system that makes these relationships dynamic. When one agent gives another a gift, their affinity should increase. If they insult them, it should decrease. This system allows for the organic evolution of social dynamics, leading to emergent alliances, rivalries, and drama.

---

## 3. How? (The Implementation)

This system will be a new service that subscribes to events from the `RuntimeStateManager` and modifies agent state in response.

1.  **Define Relationship Schema:**
    -   In the `RuntimeStateManager`'s schema for an agent, add a `relationships` dictionary.
    -   This will map other agent IDs to a `Relationship` object.
    -   **Example `Relationship` object:** `{ "affinity": 0.75, "status": "Friendly" }` (Affinity is on a -1.0 to 1.0 scale).

2.  **Create the `RelationshipManager` Service:**
    -   This is a new class in `narrative_engine/ecology.py`.
    -   It subscribes to the `StateManager`'s event stream, just like the N-Script `TriggerMonitor`.

3.  **Implement Interaction Analysis:**
    -   The `RelationshipManager` will listen for `Action` events, particularly `SpeakToAction`.
    -   When an action occurs between two agents, it will perform a "social analysis." This involves a new, specialized prompt to the R4 Narrative Engine.
    -   **Example Analysis Prompt:**
        ```
        Character A (Openness: 0.8) said to Character B (Agreeableness: 0.3): "That was a foolish move."

        Analyze the social impact of this statement. On a scale of -1.0 (very negative) to 1.0 (very positive), what is the affinity change? Respond with JSON.

        {"affinity_change": -0.2}
        ```

4.  **Update State:**
    -   The `RelationshipManager` takes the `affinity_change` from the model's response.
    -   It then calls `StateManager.update_state()` for both agents involved, adjusting the `affinity` score in their respective `relationships` maps.
    -   It can also update the `status` string based on affinity thresholds (e.g., if affinity drops below -0.5, status becomes "Hostile").

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_relationship_manager.py`):**
    -   Test the interaction analysis logic. Mock the event stream and the R4 model call.
    -   Provide a sample interaction event. Verify that the correct analysis prompt is generated.
    -   Provide a canned model response (e.g., `{"affinity_change": 0.1}`) and assert that the `RelationshipManager` makes the correct `update_state` calls to the mocked `StateManager` with the new affinity scores.
-   **Integration Test:**
    -   In a test with the `AgenticLoopFramework`, have one agent perform a `SpeakToAction` directed at another.
    -   Have a `RelationshipManager` running and listening.
    -   After the tick, verify that the `relationships` map in the `StateManager` has been updated for both agents.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Simulation Logic): In tests/scripts/test_ecology_simulation.py, create a temporary world with two dummy characters. Run the simulation function. Mock the LLM call to return a deterministic outcome JSON. Assert that the function correctly parses this outcome.
  - Green (Simulation Logic): Implement the core simulation and outcome-parsing logic.
  - Red (State Mutation): In an integration test, run the full script on a temporary directory. After the script finishes, load the character and world files from that directory and assert that their contents have been modified exactly as specified by the mocked LLM outcome.
  - Green (State Mutation): Implement the logic that calls the Manager classes to save the updated data to disk/DB.
```

## Checklist / Steps
1. Create ecology simulation script framework
2. Implement world and character selection logic
3. Create LLM-driven interaction simulation
4. Implement outcome parsing and interpretation
5. Update WorldManager and CharacterManager for thread safety
6. Add atomic world state mutation functionality
7. Create background job scheduling capability
8. Write comprehensive tests for simulation logic
9. Add logging and monitoring for ecosystem changes

## References
Directly builds upon the Agent Core (R5-4) concept but applies it between NPCs. 