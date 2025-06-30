# R5-1: The Narrative Scripting Engine (N-Script)

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R4-12, R4-13, R4-6

---

## 1. Goal

To create a simple, event-driven scripting system that allows designers to define deterministic narrative logic (e.g., "if-this-then-that") that interacts with the emergent, agent-driven world, leveraging the full triple-head architecture for sophisticated content generation, emotional control, and memory manipulation.

---

## 2. Why? (The Story)

While we want an emergent world, designers still need control. They need to be able to write "if-this-then-that" logic for critical story moments, tutorials, or safety rails. With the triple-head architecture, designers can now script not just what happens, but how it should be expressed emotionally (control head), what should be remembered (memory head), and how it should be narrated (generation head). For example: "IF the player enters the Dragon's Lair for the first time, THEN force the Dragon to deliver its monologue with menacing emotional undertones AND ensure this moment forms a vivid memory." N-Script is the tool that gives designers this multi-dimensional power, blending scripted events with the emergent simulation.

---

## 3. How? (The Implementation)

N-Script will be implemented as a set of components that integrate with the `AgenticLoopFramework` and triple-head architecture. The initial version will use a YAML-based DSL enhanced for multi-head control.

1.  **Define the Enhanced N-Script DSL:**
    -   Design a human-readable YAML syntax with triple-head support.
    -   Scripts define `Triggers` (can include head-specific conditions) and `Actions` (can target specific heads).
    -   **Enhanced Example `dragon_monologue.nscript`:**
        ```yaml
        script_id: dragon_monologue_intro
        trigger:
          type: ON_ENTER_LOCATION
          location_id: "dragon_lair_entrance"
          actor_filter: "player"
          once: true
          # NEW: Triple-head trigger conditions
          conditions:
            memory_significance: "> 0.7"  # Only trigger if entry is memorable
            emotional_state: "curious|fearful"  # Player must be emotionally engaged
        actions:
          - type: TRIPLE_HEAD_ACTION
            target_agent_id: "dragon_boss"
            generation_params:
              style: "menacing_monologue"
              tone: "ancient_wisdom"
            control_params:
              emotions: ["intimidating", "mysterious"]
              mood_shift: "hostile"
              control_tokens: ["<presence_overwhelming>", "<voice_booming>"]
            memory_params:
              importance: 0.9
              emotional_impact: 0.8
              tags: ["first_encounter", "dragon_lair", "monologue"]
            action:
              type: SpeakToAction
              message: "Frail mortal, you have stumbled into my domain..."
          - type: SET_TRIPLE_HEAD_STATE
            target_agent_id: "dragon_boss"
            generation_state:
              narrative_focus: "dramatic_presence"
            control_state:
              dominant_emotion: "menacing"
              energy_level: 0.8
            memory_state:
              context_salience: 0.9
              emotional_tagging: true
        ```

2.  **Create the Enhanced `ScriptManager`:**
    -   Resides in `narrative_engine/nscript.py` with triple-head integration.
    -   Loads and validates `.nscript` files with head-specific schemas.
    -   Interfaces with triple-head model architecture (R4-6).

3.  **Implement the Triple-Head `TriggerMonitor`:**
    -   Subscribes to events from `RuntimeStateManager` AND triple-head model outputs.
    -   **New Trigger Types**:
        - `ON_MEMORY_FORMATION`: Triggers when significant memories are formed
        - `ON_EMOTIONAL_STATE`: Triggers based on emotional control head outputs
        - `ON_GENERATION_QUALITY`: Triggers based on generation head performance
        - `ON_HEAD_COORDINATION`: Triggers when multiple heads achieve specific coordination patterns
    -   Evaluates complex multi-head conditions for script activation.

4.  **Implement the Triple-Head `ActionExecutor`:**
    -   Executes actions that can target individual heads or coordinate across all three:
        - `TRIPLE_HEAD_ACTION`: Coordinates all three heads for specific narrative moments
        - `GENERATION_OVERRIDE`: Temporarily override generation head parameters
        - `CONTROL_INJECTION`: Inject specific emotional states via control head
        - `MEMORY_FORMATION`: Force specific memory creation with custom parameters
        - `HEAD_SYNCHRONIZATION`: Ensure all three heads work in harmony for critical moments

5.  **NEW: Head-Specific Script Features:**
    -   **Generation Scripts**: Control narrative style, pacing, and content quality
    -   **Control Scripts**: Manage emotional arcs, mood transitions, and personality expression
    -   **Memory Scripts**: Orchestrate memory formation, retrieval, and long-term consistency
    -   **Cross-Head Scripts**: Coordinate complex interactions between all three heads

---

## 4. How to Test?

-   **Enhanced Unit Tests (`tests/narrative_engine/test_nscript.py`):**
    -   Test triple-head script parsing: validate head-specific parameters and actions
    -   Test enhanced `TriggerMonitor`: verify it responds to memory, emotional, and generation events
    -   Test `ActionExecutor`: ensure correct calls to all three model heads
    -   **New Tests**: Memory formation triggers, emotional state conditions, generation quality thresholds
-   **Triple-Head Integration Test:**
    -   Create test scenarios where scripts coordinate all three heads
    -   Verify memory formation, emotional control, and generation quality work together
    -   Test cross-head synchronization for complex narrative moments

## Acceptance Criteria
### Enhanced DSL & Parsing
- [ ] **Triple-Head YAML Schema**: Extended `.nscript` schema with generation, control, and memory parameters
- [ ] **Head-Specific Validation**: JSON-Schema validation for each head's parameters in `narrative_engine/nscript_schema.json`
- [ ] **Multi-Head Parser**: `ScriptManager.load_scripts()` validates all head-specific configurations

### Enhanced Trigger Runtime
- [ ] **Multi-Head Event Subscription**: `TriggerMonitor` subscribes to events from all three model heads
- [ ] **NEW Trigger Types**: Support for memory, emotional, generation, and coordination triggers
- [ ] **Head-Specific Debugging**: Debug logs showing which head triggered script activation

### Triple-Head Action Execution
- [ ] **Core Actions**: Support `TRIPLE_HEAD_ACTION`, `GENERATION_OVERRIDE`, `CONTROL_INJECTION`, `MEMORY_FORMATION`
- [ ] **Head Coordination**: Actions that synchronize multiple heads for narrative moments
- [ ] **Enhanced Coverage**: Unit tests reaching 90% branch coverage across all three heads

### Integration & Enhanced Tooling
- [ ] **Triple-Head Examples**: Example scripts in `content/worlds/Default World/scripts/` demonstrating all three heads
- [ ] **Enhanced Documentation**: `docs/nscript.md` with triple-head syntax and head-specific examples
- [ ] **Multi-Head UI**: Scripts tab showing head-specific validation status and performance metrics

### NEW: Head-Specific Features
- [ ] **Generation Control**: Scripts that fine-tune narrative style and content quality
- [ ] **Emotional Orchestration**: Scripts that manage character emotional arcs and mood transitions
- [ ] **Memory Management**: Scripts that control what gets remembered and how memories influence behavior
- [ ] **Performance Monitoring**: Real-time metrics for script effectiveness across all three heads

## Implementation Notes
```text
• Use pydantic v2 for enhanced Script objects with head-specific nested models.
• Integrate with triple-head model outputs for trigger conditions.
• Ensure thread-safe interaction with all three model heads.
• Add head-specific hot-reload capability for rapid iteration.
• Create head-specific debugging and monitoring interfaces.
```

## Checklist / Steps
1. **ENHANCED**: Define triple-head JSON-Schema + pydantic models
2. **ENHANCED**: Implement ScriptManager with multi-head load & validation
3. **NEW**: Build triple-head TriggerMonitor with memory/emotional/generation triggers
4. **NEW**: Create enhanced ActionExecutor with head-specific actions
5. **NEW**: Implement head coordination and synchronization logic
6. **ENHANCED**: Write comprehensive unit tests covering all three heads
7. **NEW**: Create triple-head example scripts & enhanced documentation
8. **NEW**: Build multi-head Scripts UI with performance monitoring
9. **ENHANCED**: Achieve >90% test coverage across all head-specific functionality

## References
Builds on R4-12 StateManager, R4-13 Agentic Loop, and **R4-6 Triple-Head Architecture**.
Integrates with **R4-5 Memory System** and **R1-10 Control Token Vocabulary**.