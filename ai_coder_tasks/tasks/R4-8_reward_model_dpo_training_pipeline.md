---
# R4-8: Triple-Head Reward Model & DPO Training Pipeline
Status: **Todo**
Ring: R4
Created: 2025-06-19
Updated: 2025-01-16 (Triple-Head Architecture Integration)
---

## Goal
Implement comprehensive training pipelines for Triple-Head Reward Models and Direct Preference Optimization (DPO) across all three model heads (generation, control, memory), using head-specific preference data to align each component independently while maintaining coordination.

## Context
This is the core of the alignment process for the triple-head architecture. We need separate reward models for each head plus coordination between them. First, we train head-specific Reward Models to learn patterns in human preferences for generation quality, emotional control, and memory formation. Then, we use coordinated DPO training to fine-tune all three heads while maintaining their interdependencies.

## Acceptance Criteria

### Triple-Head Reward Model Training:
- [ ] **Generation Head RM**: Script `scripts/run_generation_rm_training.py` for content quality reward model
- [ ] **Control Head RM**: Script `scripts/run_control_rm_training.py` for emotional appropriateness reward model  
- [ ] **Memory Head RM**: Script `scripts/run_memory_rm_training.py` for memory consistency reward model
- [ ] **Coordinated RM**: Script `scripts/run_coordinated_rm_training.py` for cross-head harmony evaluation
- [ ] **Head-Specific Architecture**: Each RM adapted for its head's output format (text/tokens/embeddings)
- [ ] **Preference Data Integration**: Load head-specific preference datasets from R4-7

### Advanced Triple-Head DPO Training:
- [ ] **Head-Specific DPO**: Scripts for `run_generation_dpo.py`, `run_control_dpo.py`, `run_memory_dpo.py`
- [ ] **Coordinated DPO**: Script `scripts/run_triple_head_dpo.py` for joint optimization
- [ ] **Head Balancing**: Dynamic weighting system to balance improvements across all three heads
- [ ] **Cross-Head Consistency**: Ensure DPO training maintains coordination between heads
- [ ] **Memory-Aware Training**: Special handling for memory head's embedding and metadata outputs

### NEW: Advanced Training Features:
- [ ] **Head-Specific Learning Rates**: Adaptive learning rates for each head based on performance
- [ ] **Cross-Head Regularization**: Prevent one head from degrading others during training
- [ ] **Memory Consistency Loss**: Additional loss term for memory head coherence
- [ ] **Emotional Arc Preservation**: Maintain character emotional consistency during control head training
- [ ] **Multi-Objective Optimization**: Balance competing objectives across all three heads

### Enhanced Model Saving & Loading:
- [ ] **Head-Specific Checkpoints**: Save/load individual head adapters independently
- [ ] **Coordinated Checkpoints**: Save full triple-head state with head relationships
- [ ] **Rollback Capability**: Revert individual heads if performance degrades
- [ ] **A/B Testing Support**: Maintain multiple head versions for comparison

## Implementation Architecture

### Generation Head Reward Model:
- **Input**: Text responses from generation head
- **Output**: Scalar reward for content quality, creativity, factual accuracy
- **Training Data**: Generation-specific preference pairs from R4-7
- **Architecture**: Language model with classification head

### Control Head Reward Model:
- **Input**: Control tokens and emotional state information
- **Output**: Scalar reward for emotional appropriateness and personality consistency
- **Training Data**: Control-specific preference pairs from R4-7
- **Architecture**: Token sequence classifier with emotional state encoding

### Memory Head Reward Model:
- **Input**: Memory embeddings (768-dim) + metadata (4-dim)
- **Output**: Scalar reward for memory accuracy and consistency
- **Training Data**: Memory-specific preference pairs from R4-7
- **Architecture**: Embedding classifier with metadata integration

### Coordinated Reward Model:
- **Input**: Combined outputs from all three heads
- **Output**: Scalar reward for cross-head harmony and overall character coherence
- **Training Data**: Cross-head preference pairs from R4-7
- **Architecture**: Multi-modal fusion model

## Implementation Notes
```text
• TDD Instructions:
  - Head-Specific Training: Test each head's RM training script independently. Mock head-specific outputs and verify appropriate loss calculation.
  - Coordinated Training: Test joint DPO script with all three heads. Verify that improvements in one head don't degrade others.
  - Cross-Head Consistency: Test that coordinated training maintains relationships between heads.
  - Memory Integration: Verify memory head training handles both embeddings and metadata correctly.
  - Performance Tracking: Ensure all training scripts log head-specific metrics to WandB.
```

## Enhanced Checklist / Steps
1. **NEW**: Create generation head reward model architecture and training script
2. **NEW**: Create control head reward model for emotional/personality evaluation  
3. **NEW**: Create memory head reward model for embedding and metadata evaluation
4. **NEW**: Implement coordinated reward model for cross-head harmony
5. **NEW**: Create head-specific DPO training scripts with appropriate loss functions
6. **NEW**: Implement joint triple-head DPO training with cross-head coordination
7. **NEW**: Add head-specific learning rate adaptation and balancing
8. **NEW**: Implement cross-head regularization to prevent degradation
9. **ENHANCED**: Create comprehensive model saving/loading for all heads
10. **NEW**: Add memory consistency loss and emotional arc preservation
11. **NEW**: Implement multi-objective optimization across heads
12. **NEW**: Create A/B testing support for head-specific improvements
13. **ENHANCED**: Write comprehensive tests covering all three heads and coordination
14. **NEW**: Add advanced monitoring and metrics for each head's performance

## Head-Specific Training Configurations

### Generation Head DPO:
- **Objective**: Maximize content quality, creativity, and factual accuracy
- **Loss Function**: Standard DPO loss on generation head outputs
- **Regularization**: Maintain consistency with control and memory heads
- **Metrics**: BLEU, ROUGE, factual accuracy, creativity scores

### Control Head DPO:
- **Objective**: Optimize emotional appropriateness and personality consistency
- **Loss Function**: DPO loss on control token distributions
- **Regularization**: Preserve emotional arc coherence
- **Metrics**: Emotional accuracy, personality consistency, mood transition smoothness

### Memory Head DPO:
- **Objective**: Improve memory formation accuracy and retrieval consistency
- **Loss Function**: DPO loss on memory embeddings + metadata
- **Regularization**: Maintain long-term memory coherence
- **Metrics**: Memory accuracy, retrieval precision, consistency over time

### Coordinated DPO:
- **Objective**: Optimize overall character coherence and head coordination
- **Loss Function**: Weighted combination of all three head losses plus coordination term
- **Regularization**: Cross-head consistency constraints
- **Metrics**: Overall character quality, cross-head harmony, user satisfaction

## References
- **Triple-Head Architecture**: R4-6 SFT Pipeline for model architecture
- **Preference Data**: R4-7 Preference Collection for head-specific training data
- **Memory System**: R4-5 External Memory API for memory head integration
- **Control Tokens**: R1-10 Control Token Vocabulary for control head training
- **Original Proposal**: §5.2 DPO (now enhanced for triple-head architecture)
- **TRL Integration**: Enhanced DPOTrainer for multi-head optimization 