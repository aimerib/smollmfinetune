---
### **R4-4: Narrative Engine - Adapter Loader & Hot-Swapping**
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

#### **Goal**
Integrate a mechanism into the Narrative-LLM to dynamically load, unload, and combine LoRA/DoRA adapters ("Cartridges") at runtime for persona and genre switching, leveraging a paged KV-cache for efficiency.

#### **Context**
This task directly connects your existing "Cartridge" concept to the new engine. It enables the hot-swapping of character personalities, which is a core tenet of your project vision. It ensures that the adapters trained in Ring 1 can be used with the powerful new engine from Ring 4.

#### **Acceptance Criteria**
- [ ] The `NarrativeLLM` class is extended with methods like `load_adapter(path, adapter_name)` and `set_active_adapter(adapter_name)`.
- [ ] Integration with a library like PEFT or a custom implementation allows for loading `.safetensors` adapters into the model.
- [ ] The forward pass is modified to correctly use the active adapter's weights.
- [ ] A mechanism for handling the paged KV-cache is implemented to prevent cache invalidation when swapping adapters (e.g., using vLLM or a similar serving framework's capabilities).
- [ ] The system can handle combining multiple adapters, as specified by `persona_mix` in the dataset schema.

#### **TDD Instructions**
1.  **Red (Failing Test - Method Exists):** In `tests/narrative_engine/test_model.py`, add a test `test_load_adapter_method()` that gets an instance of `NarrativeLLM` and tries to call `.load_adapter()`. It will fail with an `AttributeError`.
2.  **Green (Passing Test - Method Exists):** Add the stub methods (`load_adapter`, `set_active_adapter`) to the `NarrativeLLM` class.
3.  **Red (Failing Test - State Change):** Write a more advanced test, `test_adapter_loading_changes_weights()`.
    - Create a tiny dummy adapter file (`.safetensors`) using the PEFT library.
    - Instantiate the base `NarrativeLLM`.
    - Record the initial weights of a specific layer (e.g., a query projection).
    - Call `load_adapter()` with the path to the dummy adapter.
    - Assert that the weights of the target layer have now changed. This will require mocking file I/O or creating a real dummy file in a temp directory.
4.  **Green (Passing Test - State Change):** Implement the actual adapter loading logic using PEFT's `load_peft_model` or similar functions to make the test pass.
5.  **TDD for Hot-Swapping:** Repeat the cycle for `set_active_adapter`, asserting that the model's forward pass output changes when a different adapter is selected. Testing the paged KV-cache integration will likely require an integration test with the serving framework (e.g., vLLM).

#### **References**
*   Proposal §4: Persona-Genre Adapters
*   Proposal §8: Deployment Considerations (paged KV-cache)
*   Existing Tasks: All tasks that produce `.safetensors` adapters.
