# R4-13: Agentic Loop Framework

- **Ring:** R4
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Large
- **Related-Tasks:** R4-12, R5-4, R5-5, R5-6

---

## 1. Goal

To create the core "heartbeat" of the narrative engine: a framework that gives non-player characters (NPCs) proactive, autonomous behavior by running them through a continuous `perceive -> think -> act` cycle.

---

## 2. Why? (The Story)

In most games, NPCs are static; they wait for the player to talk to them. Our vision is a world where characters have lives. They wake up, go to work, pursue their own goals, and interact with each other, even when the player isn't around. The `AgenticLoopFramework` is the engine that drives this. It's the scheduler that gives each character a "turn" to do something, creating the illusion of a persistent, living world. It's the foundation upon which all R5 features like the `ProactiveAgentCore` will be built.

---

## 3. How? (The Implementation)

1.  **Define the Core `Agent` Interface:**
    -   Create an abstract base class `BaseAgent` in `narrative_engine/agent.py`.
    -   It will define the methods for the cycle:
        -   `perceive(state_manager: StateManager) -> Perception`: Gathers relevant information from the world state (e.g., "What's in my location? What happened recently?").
        -   `think(perception: Perception) -> Action`: Makes a call to the R4 Narrative Engine with the perception and its own goals to decide on a course of action.
        -   `act(action: Action, state_manager: StateManager) -> ActionResult`: Executes the chosen action, which results in one or more updates to the `StateManager`.

2.  **Develop the `Action` Schema:**
    -   Create a structured format for actions, which are the output of the `think` phase.
    -   These should be data classes, not just text strings. Examples:
        -   `MoveToAction(target_location: str)`
        -   `SpeakToAction(target_agent_id: str, message: str)`
        -   `TakeItemAction(item_id: str)`
        -   `UpdateGoalAction(new_goal_description: str)`

3.  **Implement the `Scheduler` / `Ticker`:**
    -   Create a `Scheduler` class that manages the main loop.
    -   It will run in its own thread or async process.
    -   **On each tick:**
        1.  Get a list of all active agents from the `StateManager`.
        2.  Determine which agent(s) should act in this tick (e.g., round-robin, priority-based).
        3.  For each active agent, call `agent.perceive()`.
        4.  Call `agent.think()` with the perception.
        5.  Call `agent.act()` with the resulting action, which updates the `StateManager`.
    -   The tick rate should be configurable (e.g., one world tick every 10 real-world seconds).

4.  **Integration with Narrative Engine:**
    -   The `think` method is the key integration point. It will be responsible for:
        -   Constructing a detailed prompt for the R4 model, including the agent's personality, goals, current state, and recent perceptions.
        -   Making the API call to the model.
        -   Parsing the model's text output into a structured `Action` object. This is a critical and complex step that may require few-shot prompting or a dedicated parsing model.

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_agentic_loop.py`):**
    -   Test the `Scheduler` logic: does it correctly identify and cycle through agents?
    -   Test the `Agent` interface methods individually.
    -   Mock the `StateManager` and the R4 model call. Provide a canned model response to the `think` method and verify that it correctly parses it into an `Action` object.
    -   Verify that the `act` method correctly translates an `Action` object into the appropriate calls to the (mocked) `StateManager`.
-   **Integration Tests:**
    -   Create a test with a real (in-memory) `StateManager` and 2-3 mock agents.
    -   Run the scheduler for a few ticks and assert that the world state in the `StateManager` has been modified in the expected way based on the agents' canned actions. 