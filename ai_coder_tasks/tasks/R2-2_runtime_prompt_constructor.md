---
# R2-2  Runtime Prompt Constructor
Status: **Todo**
Ring: R2
Created: 2025-06-18
---

## Goal
Create a stateful `RuntimePromptConstructor` class that builds prompts for a character at runtime, mirroring the logic of the training-time `PromptBuilder` but incorporating dynamic, turn-by-turn state.

## Acceptance Criteria

### 1. Class Implementation
- [ ] Create a new module: `utils/runtime/prompt_constructor.py`.
- [ ] Implement a class `RuntimePromptConstructor`.
      - The constructor `__init__(self, packet_path: str)` will load all the assets from a character's runtime packet (`R2-1`).
      - A method `construct(self, conversation_history: List[Dict], dynamic_state: Dict) -> str` will build the final prompt string.

### 2. Dynamic State Handling
- [ ] The `dynamic_state` dictionary is crucial. It should be able to process keys such as:
      - `current_mood`: (e.g., "happy", "annoyed") which can influence adjective choice.
      - `relationship_to_user`: A dictionary of scores (e.g., `{ "trust": 0.7, "attraction": 0.2 }`).
      - `recent_events`: A list of short summaries of what has just happened.
      - `forced_control_tokens`: A list of control tokens the runtime wants to inject.

### 3. Prompt Construction Logic
- [ ] The constructor must replicate the core logic of the `PromptBuilder` from `R1-6`:
      - It uses the Big-Five scores from `character_core.json` to generate personality adjectives.
      - It injects character goals.
      - It can select and inject a relevant lore fact.
- [ ] It must correctly format the `conversation_history` into the prompt, following the model's chat template.
- [ ] It must handle `control_tokens` from `tokens.json` seamlessly.

### 4. Unit Testing
- [ ] Create a unit test that builds a fake runtime packet.
- [ ] The test will instantiate the `RuntimePromptConstructor` and call `construct` with various dynamic states to ensure the output prompt string is correctly and consistently formatted.

## Implementation Notes
This component is the "brain" of the character at runtime. It's the final step that translates all the authored data and dynamic state into something the LLM can understand. Its output should be directly passable to a tokenizer and then the model.

```python
# Example Usage
from utils.runtime.prompt_constructor import RuntimePromptConstructor

# At the start of a session
constructor = RuntimePromptConstructor(packet_path="runtime_packets/my_character")

# For each turn...
history = [{"role": "user", "content": "Hello there!"}]
state = {"current_mood": "curious", "relationship_to_user": {"trust": 0.5}}

prompt = constructor.construct(history, state)
# -> to tokenizer -> to model
```

## References
- This is the direct runtime counterpart to `R1-6` (Prompt Builder).
- It consumes the packet created by `R2-1` (Export Runtime Packet). 