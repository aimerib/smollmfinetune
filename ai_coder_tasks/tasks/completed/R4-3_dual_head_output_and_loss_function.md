---
### **R4-3: Narrative Engine - Dual-Head Output & Loss Function**
Status: **Completed** ✅
Ring: R4
Created: 2025-06-19
---

#### **Goal**
Implement a custom loss function that can calculate a joint loss from the model's dual output heads (free-text and action JSON), correctly routing gradients based on tagged data spans.

#### **Context**
A standard cross-entropy loss won't work for our dual-head model. We need a specialized loss function that knows which parts of the output correspond to conversational text and which correspond to structured tool calls. This is the heart of training the model to be bilingual in prose and actions.

#### **Acceptance Criteria**
- [x] A new module `narrative_engine/loss.py` is created.
- [x] It contains a `DualHeadLoss` class, inheriting from `torch.nn.Module`.
- [x] The `forward()` method of `DualHeadLoss` accepts the model's output logits, the ground-truth labels, and the `loss_mask` (from the data pipeline).
- [x] It calculates cross-entropy loss separately for the text predictions and action predictions, using the mask to ignore irrelevant tokens for each head.
- [x] It returns a single, combined scalar loss value (e.g., a weighted sum of the two losses).

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

---

## **Completion Summary**

**Completed by:** Claude (AI Assistant)
**Date:** 2024-12-18

### What Was Implemented:

1. **Created `narrative_engine/loss.py`** with the `DualHeadLoss` class that:
   - Properly routes gradients to the correct head based on channel masks
   - Handles cases where all tokens belong to one channel (detaches unused heads)
   - Supports configurable weights for text vs action losses
   - Includes comprehensive logging for debugging
   - Provides a `compute_per_channel_losses` method for analysis

2. **Key Design Decisions:**
   - Used detachment to prevent gradient flow to unused heads
   - Implemented proper handling of edge cases (e.g., no tokens for a channel)
   - Added validation for input tensor shapes
   - Used ignore_index pattern for masked tokens

3. **Created comprehensive test suite** in `tests/narrative_engine/test_loss.py`:
   - Tests basic functionality (scalar output)
   - Tests masking logic (perfect predictions yield ~0 loss)
   - Tests loss mask exclusion (masked tokens don't contribute)
   - Tests gradient routing (only active heads get gradients)
   - Tests weighted loss combinations

4. **Integration points**:
   - Created `narrative_engine/__init__.py` for clean imports
   - Created integration test showing how to use with model and data pipeline
   - Demonstrated how to integrate into existing model classes

### Technical Highlights:

The dual-head loss function is the critical innovation that enables the model to be "bilingual" - speaking both natural language and structured actions. Key features:

- **Gradient Routing**: Only the head responsible for a token receives gradients
- **Channel-Aware**: Uses channel masks from the data pipeline to identify text vs action tokens
- **Loss Masking**: Respects loss masks to exclude non-training tokens (padding, user turns)
- **Flexible Weighting**: Allows different importance for text vs action predictions

### Usage Example:
```python
loss_fn = DualHeadLoss(text_weight=1.0, action_weight=1.5)
loss = loss_fn(
    text_logits=model_outputs["text_logits"],
    action_logits=model_outputs["action_logits"], 
    labels=labels,
    loss_mask=loss_mask,
    channel_mask=channel_mask
)
```

This implementation provides the foundation for training models that can seamlessly switch between conversational responses and structured tool use - a key capability for creating more capable and interactive AI characters.
