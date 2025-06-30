# R5-5: Digital Ecology Engine (Relationship Manager)

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R4-12, R5-4, R4-6

---

## 1. Goal

To implement a `RelationshipManager` that observes agent interactions and updates their relationship statuses in the `RuntimeStateManager`, creating a dynamic social fabric in the world by leveraging the full triple-head architecture for sophisticated relationship analysis.

---

## 2. Why? (The Story)

A world isn't just a collection of individuals; it's a web of relationships. Friends, rivals, lovers, and enemies. The `Digital Ecology Engine`, in its first version, is a system that makes these relationships dynamic. When one agent gives another a gift, their affinity should increase. If they insult them, it should decrease. With the triple-head architecture, we can now analyze not just what was said, but the emotional undertones (control head) and how memorable the interaction was (memory head), creating richer and more nuanced relationship dynamics.

---

## 3. How? (The Implementation)

This system will be a new service that subscribes to events from the `RuntimeStateManager` and modifies agent state in response.

1.  **Define Enhanced Relationship Schema:**
    -   In the `RuntimeStateManager`'s schema for an agent, add a `relationships` dictionary.
    -   This will map other agent IDs to a `Relationship` object.
    -   **Enhanced `Relationship` object:** 
        ```json
        {
            "affinity": 0.75, 
            "status": "Friendly",
            "emotional_history": ["happy", "curious", "fond"],
            "memory_significance": 0.8,
            "last_interaction": "2024-01-15T10:30:00Z",
            "interaction_count": 15
        }
        ```

2.  **Create the `RelationshipManager` Service:**
    -   This is a new class in `narrative_engine/ecology.py`.
    -   It subscribes to the `StateManager`'s event stream, just like the N-Script `TriggerMonitor`.

3.  **Implement Triple-Head Interaction Analysis:**
    -   The `RelationshipManager` will listen for `Action` events, particularly `SpeakToAction`.
    -   When an action occurs between two agents, it will perform a "social analysis" using the triple-head Narrative Engine (R4-6).
    -   **Enhanced Analysis Process:**
        - **Generation Head**: Analyzes the semantic content and intent of the interaction
        - **Control Head**: Extracts emotional undertones and mood indicators
        - **Memory Head**: Determines how memorable and significant the interaction is
    -   **Example Enhanced Analysis:**
        ```
        Character A (Openness: 0.8) said to Character B (Agreeableness: 0.3): "That was a foolish move."
        
        GENERATION ANALYSIS: Analyze the social impact and intent.
        CONTROL ANALYSIS: What emotions are present? Extract mood and relationship control tokens.
        MEMORY ANALYSIS: How memorable is this interaction? Rate significance and emotional impact.
        
        Respond with structured analysis including affinity change, emotional state, and memory importance.
        ```

4.  **Process Triple-Head Outputs:**
    -   **Generation Head Output**: Semantic relationship analysis and affinity changes
    -   **Control Head Output**: Emotional state tokens (e.g., `<mood_annoyed>`, `<relationship_tension>`)
    -   **Memory Head Output**: Memory formation data for significant interactions
    -   The `RelationshipManager` combines all three outputs for comprehensive relationship updates.

5.  **Update Enhanced State:**
    -   Updates both agents' relationship data using insights from all three heads:
        - **Affinity**: From generation and control head analysis
        - **Emotional History**: From control head emotional state tracking
        - **Memory Significance**: From memory head importance scoring
        - **Interaction Patterns**: Accumulated data for long-term relationship evolution
    -   Relationship status updates consider emotional context and memory significance

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_relationship_manager.py`):**
    -   Test the triple-head interaction analysis logic. Mock the event stream and the triple-head Narrative Engine.
    -   Provide a sample interaction event. Verify that the correct analysis prompt is generated for all three heads.
    -   Provide canned responses for all three heads and assert that the `RelationshipManager` makes the correct `update_state` calls with comprehensive relationship data.
    -   **Test Memory Integration**: Verify that significant interactions (high memory head scores) have longer-lasting relationship effects.
    -   **Test Emotional Continuity**: Verify that emotional states from control head influence relationship evolution over time.
-   **Integration Test:**
    -   In a test with the `AgenticLoopFramework`, have one agent perform a `SpeakToAction` directed at another.
    -   Have a `RelationshipManager` running and listening to triple-head outputs.
    -   After the tick, verify that the `relationships` map in the `StateManager` has been updated with emotional history, memory significance, and enhanced affinity data.
    -   **Multi-Interaction Test**: Run multiple interactions and verify that relationship patterns evolve based on emotional and memory context.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Triple-Head Analysis): In tests/narrative_engine/test_relationship_manager.py, create a test that calls the triple-head analysis function. Mock all three head outputs. Assert that the relationship update includes data from generation, control, and memory heads.
  - Green (Triple-Head Analysis): Implement the enhanced analysis logic using all three heads.
  - Red (Memory Integration): Write a test where memory head indicates high significance. Assert that the relationship update has enhanced persistence and memory tracking.
  - Green (Memory Integration): Implement memory-aware relationship persistence.
  - Red (Emotional Context): Test that control head emotional states influence relationship dynamics over multiple interactions.
  - Green (Emotional Context): Implement emotional state integration in relationship evolution.
```

## Checklist / Steps
1. Create enhanced relationship schema with emotional and memory tracking
2. Implement triple-head interaction analysis system
3. Create comprehensive LLM prompts for all three heads
4. Implement outcome parsing for generation, control, and memory outputs
5. **NEW**: Add emotional state tracking to relationship evolution
6. **NEW**: Implement memory-based relationship persistence
7. **NEW**: Create relationship pattern analysis using control head data
8. Update WorldManager and CharacterManager for enhanced relationship data
9. Add atomic world state mutation functionality
10. Create background job scheduling capability with triple-head processing
11. **NEW**: Add relationship visualization data for Living Interface (R5-3)
12. Write comprehensive tests for triple-head relationship analysis
13. Add logging and monitoring for ecosystem changes with emotional and memory context

## References
Directly builds upon the Agent Core (R5-4) and integrates with triple-head Narrative Engine (R4-6).
Uses memory system (R4-5) for relationship persistence and control tokens for emotional analysis. 