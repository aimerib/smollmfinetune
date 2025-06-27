---
### **R4-3: Narrative Engine - Dual-Head Output & Loss Function**
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

#### **Goal**
Implement a custom loss function that can calculate a joint loss from the model's dual output heads (free-text and action JSON), correctly routing gradients based on tagged data spans.

#### **Context**
A standard cross-entropy loss won't work for our dual-head model. We need a specialized loss function that knows which parts of the output correspond to conversational text and which correspond to structured tool calls. This is the heart of training the model to be bilingual in prose and actions.

#### **Acceptance Criteria**
- [ ] A new module `narrative_engine/loss.py` is created.
- [ ] It contains a `DualHeadLoss` class, inheriting from `torch.nn.Module`.
- [ ] The `forward()` method of `DualHeadLoss` accepts the model's output logits, the ground-truth labels, and the `loss_mask` (from the data pipeline).
- [ ] It calculates cross-entropy loss separately for the text predictions and action predictions, using the mask to ignore irrelevant tokens for each head.
- [ ] It returns a single, combined scalar loss value (e.g., a weighted sum of the two losses).

#### **TDD Instructions**
1.  **Red (Failing Test - Basic Loss):** In `tests/narrative_engine/test_loss.py`, write `test_dual_head_loss_calculation()`. Create dummy logits, labels, and a loss mask where some tokens are tagged for the text head and some for the action head. Instantiate the `DualHeadLoss` and pass these tensors to it. Assert that the output is a single scalar tensor. This will fail because the class doesn't exist.
2.  **Green (Passing Test - Basic Loss):** Implement the `DualHeadLoss` class with a basic implementation to make the test pass.
3.  **Red (Failing Test - Correct Masking):** Write a more specific test, `test_loss_masking_logic()`.
    - Create a scenario where the action logits are perfect but text logits are wrong. The loss should be > 0.
    - Create a scenario where the text logits are perfect but action logits are wrong. The loss should be > 0.
    - Create a scenario where both are perfect. The loss should be exactly 0.
    This will drive the correct implementation of the masking logic inside the loss function.
4.  **Green (Passing Test - Correct Masking):** Refine the `DualHeadLoss` implementation to correctly apply the masks and compute the separate losses before combining them, ensuring all tests pass.

#### **References**
*   Proposal §4: Model Architecture Specification (Dual Output Heads)
*   Proposal §9: Work-Package Skeleton (Item 4)
*   New Task: `R4-2: Narrative Engine - Unified Dataset Pipeline` (produces the loss mask)
