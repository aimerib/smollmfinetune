---
# R5-2: The Iceberg Model (Subtext Generation)
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Implement a three-head model architecture that generates surface dialogue, internal thoughts, and emotional state simultaneously, providing rich psychological depth for character interactions and enabling advanced narrative features.

## Context
Characters need believable inner lives that inform their external behavior. By training the model to generate both what characters say and what they think/feel, we enable sophisticated features like emotional intelligence, character development arcs, and realistic relationship dynamics.

## Acceptance Criteria

### Model Architecture Enhancement:
- [ ] Three-head architecture: `dialogue_head`, `thought_head`, `emotion_head`
- [ ] Shared attention layers with head-specific projection layers
- [ ] Attention masking to prevent heads from accessing each other's outputs
- [ ] Memory-efficient implementation with gradient checkpointing
- [ ] Configurable head weighting for different training phases

### Training Pipeline Updates:
- [ ] Multi-head loss function with balanced weighting strategies
- [ ] Curriculum learning: train dialogue first, then add inner layers
- [ ] Data augmentation pipeline for generating internal thoughts
- [ ] Validation metrics for each head independently
- [ ] Early stopping based on composite loss improvement

### Production Features:
- [ ] Real-time emotion tracking and character mood persistence
- [ ] Relationship dynamic modeling based on internal thoughts
- [ ] Character development arc tracking through thought evolution
- [ ] Context-aware subtext generation based on conversation history
- [ ] Performance optimization for triple-head inference

### Creator Tools:
- [ ] Character psychology editor for defining thought patterns
- [ ] Emotion calibration interface with personality trait integration
- [ ] Relationship matrix visualization showing hidden dynamics
- [ ] Character development timeline with internal state evolution
- [ ] A/B testing for different subtext generation strategies

## Implementation Notes
```text
• TDD Instructions:
  - Red (Schema): Update the Pydantic model tests to verify that the Turn model now correctly handles the subtext field.
  - Red (Model): Update the test_forward_pass_shapes test in tests/narrative_engine/test_model.py to assert that the model's output dictionary now contains subtext_logits of the correct shape.
  - Red (Loss): Create tests/narrative_engine/test_multi_head_loss.py. Write a test that provides dummy data for all three heads and asserts that the loss is calculated correctly based on the masking.
  - Green (All): Implement the architectural, loss, and data pipeline changes to make the backend tests pass.
  - Red/Green (UI): Using streamlit.testing.v1.AppTest, write a test for the chat UI. Pre-populate a message with subtext. Assert that the subtext is initially hidden. Simulate clicking the "Director's View" toggle and assert that the subtext is now visible in the rendered output.
```

## Checklist / Steps
1. Update Turn model schema to include subtext field
2. Modify NarrativeLLM architecture to add subtext_head
3. Refactor DualHeadLoss to MultiHeadLoss for three heads
4. Update DatasetProcessor to handle subtext tokens
5. Add Director's View toggle to UI interfaces
6. Implement subtext display/hide functionality
7. Write comprehensive tests for all components
8. Update training pipeline to handle three-head loss

## References
Depends on R4-1 (Model Architecture), R4-2 (Data Pipeline), and R4-3 (Loss Function). 