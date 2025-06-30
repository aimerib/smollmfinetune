# R5-2: Iceberg Model (Subtext Generation)

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R4-3, R4-13

---

## 1. Goal

To modify the Narrative Engine and agent framework to generate a character's "internal monologue" (subtext) in addition to their spoken dialogue or actions. This subtext is visible to the player/director but not to other characters.

---

## 2. Why? (The Story)

What a character says is only half the story. What they *think* is where the real drama lies. Does the character say "I'm fine" while thinking "I can't believe they betrayed me"? This "Iceberg Model" creates dramatic irony and deepens the player's connection to the characters by exposing their inner world. It also provides a powerful debugging tool for writers, showing what the character's motivations truly are.

---

## 3. How? (The Implementation)

This feature will be implemented by modifying the `think` step of the `AgenticLoopFramework` and leveraging the triple-head output capability from R4-6.

1.  **Update the `Agent.think()` Method Signature:**
    -   In `narrative_engine/agent.py`, modify the `think` method in the `BaseAgent` interface.
    -   It should now return a `ThinkResult` object instead of just an `Action`.
    -   The `ThinkResult` data class will be: `ThinkResult(action: Action, subtext: str)`.

2.  **Modify the Prompt for Dual Output:**
    -   In the `think` method's implementation, adjust the prompt sent to the Narrative Engine (R4 model).
    -   The prompt will now explicitly ask for two outputs, clearly separated.
    -   **Example Prompt Suffix:**
        ```
        ...Based on the above, generate the character's next action and their private inner thoughts.

        [SUBTEXT]
        (The character's inner monologue and true feelings)
        [/SUBTEXT]

        [ACTION]
        (The structured action the character will perform, in the specified format)
        [/ACTION]
        ```

3.  **Implement Parsing Logic:**
    -   After receiving the response from the LLM, the `think` method must parse the output.
    -   It will extract the content between the `[SUBTEXT]` tags and the `[ACTION]` tags.
    -   The action content will be parsed into a structured `Action` object as before.
    -   The subtext content will be stored as a simple string.

4.  **Log the Subtext:**
    -   The `AgenticLoopFramework`'s main scheduler will be updated to handle the new `ThinkResult` object.
    -   After calling `agent.think()`, it will pass the `action` to the `agent.act()` method as before.
    -   The `subtext` string will be logged to a new, dedicated stream, perhaps in the `RuntimeStateManager`, tagged with the character's ID and the current timestamp. It is then the responsibility of the UI (R5-3) to fetch and display this information.

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_agent_think.py`):**
    -   Create a new test file for this functionality.
    -   Mock the R4 model call. Provide a canned response that includes both `[SUBTEXT]` and `[ACTION]` blocks.
    -   Call the `agent.think()` method and assert that it returns a `ThinkResult` object with the correctly parsed `Action` and `subtext` string.
    -   Test edge cases: what if the model fails to generate one of the blocks? The system should handle this gracefully (e.g., subtext is an empty string).
-   **Integration Test:**
    -   In an integration test for the `AgenticLoopFramework`, check that after a tick, the subtext log in the (mocked) `StateManager` has been updated correctly. 

## Acceptance Criteria
- [ ] **API Change**: `BaseAgent.think()` returns `ThinkResult(action: Action, subtext: str)`; dataclass lives in `narrative_engine/types.py`.
- [ ] **Prompt Format**: Prompt suffix with `[SUBTEXT]` / `[ACTION]` tags lives in `utils/prompt_templates.py`; unit test snapshot asserts tags present.
- [ ] **Parser**: Robust regex parser raises `SubtextParseError` on malformed tags; 100 % branch coverage.
- [ ] **State Logging**: Subtext stored via `StateManager.add_subtext(agent_id, text, ts)`; retrieval API added.
- [ ] **UI Hook**: Director's View (R5-3) fetches & displays subtext; placeholder if none.
- [ ] **Tests**: `tests/narrative_engine/test_subtext_parser.py` (unit) and integration test with AgentLoop verifying subtext log update.

## Implementation Notes
```text
• Keep triple-head architecture optional: if model lacks `[SUBTEXT]`, gracefully fallback with empty string.
• Use pydantic to validate ThinkResult.
• For existing model weights, insert tagged prompt but rely on model to copy guidance tokens until retrained.
```

## Checklist / Steps
1. Create ThinkResult dataclass & update BaseAgent.
2. Add prompt template with tags.
3. Implement parser + tests.
4. Extend StateManager with subtext log table.
5. Update AgentLoop to log subtext.
6. Adjust Director's View UI to render subtext.

## References
Depends on triple-head capability (R4-6) and feeds R5-3 Living Interface for display. 