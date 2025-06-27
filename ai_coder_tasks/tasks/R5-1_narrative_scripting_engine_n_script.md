---
# R5-1: The Narrative Scripting Engine (N-Script)
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Build a production-ready scripting language and execution engine that enables power users to create deterministic narrative logic, quest systems, and character behavior trees with a simple, readable syntax.

## Context
Professional game writers need precise control over story pacing and character interactions. N-Script provides a lightweight, event-driven programming language specifically designed for narrative control, bridging the gap between LLM creativity and traditional game scripting.

## Acceptance Criteria
### Scripting Language Engine:
- [ ] PLY-based parser in `narrative_engine/scripting/parser.py` with formal grammar
- [ ] AST compiler in `narrative_engine/scripting/compiler.py` with bytecode generation
- [ ] High-performance executor in `narrative_engine/scripting/executor.py` with event loop
- [ ] Comprehensive error handling with line-number specific debugging
- [ ] Hot-reloading of scripts without server restart

### Production Language Features:
- [ ] **Variables**: `SET $trust_level = character.trust + 10`
- [ ] **Conditionals**: `IF $trust_level > 50 THEN show_secret_dialog`
- [ ] **Loops**: `WHILE inventory.has("key") DO unlock_doors`
- [ ] **Functions**: `DEFINE check_relationship(name) RETURN character(name).trust`
- [ ] **Events**: `ON user_message, ON character_response, ON world_state_change`

### Integration Architecture:
- [ ] Event dispatcher system integrated with main conversation loop
- [ ] Script state persistence between conversations
- [ ] Performance monitoring and script execution limits
- [ ] Sandbox execution environment for security
- [ ] Cache optimization for frequently executed scripts

### Developer Experience:
- [ ] VS Code extension for N-Script syntax highlighting and debugging
- [ ] Script validation and linting in real-time
- [ ] Interactive debugger with breakpoints and variable inspection
- [ ] Comprehensive documentation with examples and tutorials
- [ ] Script testing framework with unit test support

## Implementation Notes
```text
• TDD Instructions:
  - Red (Parser): In tests/narrative_engine/scripting/test_parser.py, write a test that feeds a valid N-Script string to the parser and asserts that it produces a correct, non-empty AST. Write another test with invalid syntax and assert it raises a ParsingError.
  - Green (Parser): Implement the N-Script parser to make the tests pass.
  - Red (Executor): In tests/narrative_engine/scripting/test_executor.py, create a dummy event and a pre-defined AST. Instantiate the Executor with the AST. Fire the event. Assert that the correct action is "called" (i.e., returned by the executor's process_event method).
  - Green (Executor): Implement the executor logic to correctly evaluate conditions and identify the right actions to take.
  - Integration (Orchestrator): Write a test for the Orchestrator where a scripted rule should modify the final prompt sent to the LLM (e.g., by adding a memory or control token).
```

## Checklist / Steps
1. Design N-Script language syntax and grammar
2. Implement parser for N-Script strings to AST
3. Create executor for running AST against events
4. Integrate executor with Orchestrator
5. Add N-Script editor tab to world management UI
6. Implement script file saving/loading
7. Write comprehensive tests for parser and executor
8. Add syntax highlighting and error reporting

## References
This is a new, foundational feature for Ring 5. 