---
# R5-1: The Narrative Scripting Engine (N-Script)
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Implement "N-Script," a simple, event-driven scripting language that allows power-user creators to define deterministic narrative logic and control the flow of the story with precision.

## Context
Inspired by Steve Wozniak's philosophy of providing elegant, powerful tools, this feature adds a "hacker mode" for creators. It provides a deterministic, low-level escape hatch from the probabilistic nature of the LLM, enabling the creation of intricate, clockwork-like narrative devices that the standard UI doesn't support.

## Acceptance Criteria
### Parser Implementation:
- [ ] A new module `narrative_engine/scripting/parser.py` is created.
- [ ] It contains a parser that can take an N-Script string and output a structured Abstract Syntax Tree (AST).

### Executor Implementation:
- [ ] A new module `narrative_engine/scripting/executor.py` is created.
- [ ] The Orchestrator is equipped with an instance of the Executor, which loads the parsed AST for the current world.
- [ ] The Executor checks its rules against events fired by the platform on each turn (e.g., `on_turn_start`, `on_tool_call`).

### UI Integration:
- [ ] A new "📜 N-Script" tab is added to the `page_world_management.py` UI.
- [ ] This tab contains an `st.code_editor` with syntax highlighting for N-Script, allowing creators to write and save scripts to a `world_script.nscript` file within the world's directory.

### Core Language Features:
- [ ] Events: `ON turn(number)`, `ON tool_call(name)`.
- [ ] Conditions: `IF character('name').relationship.trust < value`.
- [ ] Actions: `THEN inject_memory("text")`, `THEN force_control_token("<token>")`, `THEN trigger_event("event_name")`.

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