---
### **R4-1: Narrative Engine - Model Architecture Scaffolding**
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

#### **Goal**
Implement the core PyTorch classes for the Narrative-LLM, establishing the foundational code structure for the model, its configuration, and its distinct layers as defined in the architecture specification.

#### **Context**
This is the first concrete step in building the custom Narrative-LLM. It translates the architectural diagram from the design document into a tangible, albeit not yet trainable, set of Python classes. All subsequent development (loss functions, training loops, data pipelines) will depend on this skeleton.

#### **Acceptance Criteria**
- [ ] A new directory `narrative_engine/` exists at the project root.
- [ ] A `NarrativeLLMConfig` dataclass is defined in `narrative_engine/config.py`, containing all key hyperparameters from the design doc (token_dim, session_embedding_dim, n_layers, n_experts, etc.).
- [ ] A `NarrativeLLM` class, inheriting from `torch.nn.Module`, is defined in `narrative_engine/model.py`.
- [ ] The `NarrativeLLM` constructor correctly initializes all major sub-modules based on the config:
    - Token, Position, and Session ID Embeddings.
    - A stack of Transformer Blocks (can be a stub initially).
    - A Cross-Attention layer for external memory (stub).
    - Two separate output heads: `lm_head` (for text) and `action_head` (for JSON).
- [ ] The model's `forward()` method has a defined signature that accepts `input_ids`, `attention_mask`, `session_id`, and `external_memory_states`. It returns a dictionary with logits from both heads.

#### **TDD Instructions**
1.  **Red (Failing Test - Imports):** In a new file `tests/narrative_engine/test_model.py`, write a test `test_model_imports()` that tries to `from narrative_engine.model import NarrativeLLM` and `from narrative_engine.config import NarrativeLLMConfig`. This will fail as the files/classes don't exist.
2.  **Green (Passing Test - Imports):** Create the empty files and class definitions to make the import test pass.
3.  **Red (Failing Test - Instantiation):** Write a test `test_model_instantiation()` that creates a default `NarrativeLLMConfig` object and attempts to instantiate the `NarrativeLLM(config)` model. This will fail as the `__init__` method is not implemented.
4.  **Green (Passing Test - Instantiation):** Implement the `__init__` method in `NarrativeLLM`, adding the required layers (even as simple `nn.Linear` stubs) to satisfy the instantiation.
5.  **Red (Failing Test - Forward Pass):** Write a test `test_forward_pass_shapes()` that creates a dummy config, instantiates the model, passes correctly-shaped dummy tensors for all inputs, and asserts that the output dictionary contains two tensors (`text_logits` and `action_logits`) of the expected shapes. This will fail as the `forward` method is incomplete.
6.  **Green (Passing Test - Forward Pass):** Implement the `forward` method to pass the dummy data through the layers and produce outputs of the correct shape.

#### **References**
*   Depends on: R4-0 (Prototype Validation)
*   Proposal §4: Model Architecture Specification
*   Proposal §9: Work-Package Skeleton (Item 1)
