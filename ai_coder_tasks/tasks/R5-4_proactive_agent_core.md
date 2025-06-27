---
# R5-4: The Proactive Agent Core
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Implement a persistent, goal-oriented "agent loop" for each character, giving them an inner life and the ability to act proactively, including initiating conversations with the user.

## Context
This feature is the definitive step in transforming characters from reactive chatbots into believable, living actors. By giving each character a persistent, asynchronous cognitive loop (Perceive-Reflect-Plan-Act), they gain true agency, capable of pursuing their own goals and driving the narrative forward independently.

## Acceptance Criteria
- [ ] Agent Service: A new service is created at `narrative_engine/agent_service.py`, containing an `AgentLoop` class.
- [ ] Cognitive Cycle: The `AgentLoop` implements the four-step cycle:
  - [ ] Perceive: Gathers data from the character's core file, the world lore, and recent conversation history.
  - [ ] Reflect: Uses an LLM call to generate a private "internal monologue" summarizing its state and goals.
  - [ ] Plan: Uses a second LLM call to form a simple, actionable plan based on its reflection.
  - [ ] Act: Can update its own internal state (e.g., mood) or queue a proactive message for the user.
- [ ] Proactive Messaging: A new database table or queue (`proactive_messages`) is created. The agent's "Act" step can place messages here.
- [ ] UI Notifications: The main application UI is updated to poll for new proactive messages for the current user and display a non-intrusive notification (e.g., "Kaelen has a message for you...").
- [ ] Asynchronous Execution: The agent loops for all characters run in a separate background process or worker (e.g., using Celery or `asyncio.TaskGroup`).

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