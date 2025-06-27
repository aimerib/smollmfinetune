---
# R5-4: The Proactive Agent Core
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Build a production-ready autonomous agent system that enables characters to pursue goals, form memories, evolve relationships, and initiate interactions based on their personality and world state.

## Context
To create truly living characters, we need agents that operate independently of user input. This system provides characters with persistent goals, memory, and decision-making capabilities while maintaining performance and preventing unwanted behavior.

## Acceptance Criteria

### High-Performance Agent Engine:
- [ ] Async agent scheduler with configurable execution intervals
- [ ] Memory-efficient agent state management with persistent storage
- [ ] Goal-driven behavior trees with dynamic priority adjustment
- [ ] Resource management and execution limits per agent
- [ ] Health monitoring and automatic agent recovery

### Production Agent Behaviors:
- [ ] **Goal Pursuit**: Long-term objective tracking and execution
- [ ] **Relationship Management**: Dynamic relationship modeling and evolution
- [ ] **Memory Formation**: Automatic episodic and semantic memory creation
- [ ] **Mood Evolution**: Persistent emotional state with realistic changes
- [ ] **Proactive Messaging**: Context-aware user outreach and conversation initiation

### Integration & Safety:
- [ ] Message queue system with delivery guarantees
- [ ] User notification preferences and consent management
- [ ] Agent behavior boundaries and content filtering
- [ ] Performance monitoring and resource usage tracking
- [ ] Rollback system for problematic agent actions

### Configuration & Control:
- [ ] Agent behavior editor with visual workflow designer
- [ ] Real-time agent monitoring dashboard
- [ ] Agent activity logs and decision explanations
- [ ] User controls for agent interaction frequency
- [ ] A/B testing framework for agent behavior optimization

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