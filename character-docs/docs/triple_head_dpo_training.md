# Triple-Head DPO Training Pipeline

This document describes the reward model and DPO (Direct Preference Optimization) training pipeline for the triple-head architecture.

## Overview

The triple-head architecture consists of:
1. **Generation Head**: Produces text output
2. **Control Head**: Manages emotional and cognitive control tokens  
3. **Memory Head**: Generates memory embeddings and metadata

Each head can be trained independently or coordinately using preference data collected through the UI.

## Reward Models

### Generation Head Reward Model
Evaluates text quality, creativity, and factual accuracy.

```bash
python scripts/run_generation_rm_training.py \
    --preference-dir preference_data \
    --output-dir models/generation_rm \
    --num-epochs 3 \
    --use-wandb
```

### Control Head Reward Model
Evaluates emotional appropriateness and personality consistency.

```bash
python scripts/run_control_rm_training.py \
    --preference-dir preference_data \
    --output-dir models/control_rm \
    --num-epochs 5 \
    --use-wandb
```

### Memory Head Reward Model
Evaluates memory formation accuracy and consistency.

```bash
python scripts/run_memory_rm_training.py \
    --preference-dir preference_data \
    --output-dir models/memory_rm \
    --num-epochs 5 \
    --use-wandb
```

## DPO Training

### Generation Head DPO
Standard DPO for text generation quality.

```bash
python scripts/run_generation_dpo.py \
    --model-path models/sft_checkpoint \
    --preference-dir preference_data \
    --output-dir models/generation_dpo \
    --beta 0.1 \
    --num-epochs 3
```

### Coordinated Triple-Head DPO
Simultaneous optimization of all three heads with cross-head regularization.

```bash
python scripts/run_triple_head_dpo.py \
    --model-path models/sft_checkpoint \
    --preference-dir preference_data \
    --output-dir models/triple_head_dpo \
    --generation-weight 1.0 \
    --control-weight 0.8 \
    --memory-weight 0.6 \
    --coordination-weight 0.4
```

## Preference Data Format

Preferences are stored as JSONL files in the preference directory:

### Generation Preferences (`generation_preferences.jsonl`)
```json
{
    "conversation_id": "conv_123",
    "prompt": "Tell me a story",
    "chosen": "Once upon a time...",
    "rejected": "I don't know any stories",
    "content_quality": 8,
    "creativity": 9,
    "factual_accuracy": 7,
    "coordination": 8
}
```

### Control Preferences (`control_preferences.jsonl`)
```json
{
    "conversation_id": "conv_123",
    "emotional_appropriateness": 9,
    "personality_consistency": 8,
    "mood_matching": 8,
    "coordination": 8
}
```

### Memory Preferences (`memory_preferences.jsonl`)
```json
{
    "conversation_id": "conv_123",
    "memory_accuracy": 8,
    "memory_consistency": 9,
    "formation_quality": 8,
    "coordination": 8
}
```

## Integration with UI

The training pipeline integrates seamlessly with the Streamlit UI:

1. **Preference Collection**: Use the triple-head reward labeling interface to rate model outputs
2. **Automatic RLHF**: After SFT training, RLHF (GRPO/PPO/DPO) can run automatically
3. **Algorithm Selection**: Choose between GRPO, PPO, or DPO in the training configuration

### UI Configuration

In the training config page, enable RLHF and select DPO:

```python
# Training configuration
enable_rlhf = True
algorithm = "DPO"
beta = 0.1
learning_rate = 1e-5
```

## Implementation Details

### Head-Specific Loss Functions

1. **Generation DPO Loss**: Standard DPO loss on text logits
2. **Control DPO Loss**: Multi-label classification with emotional arc preservation
3. **Memory DPO Loss**: Cosine similarity for embeddings + MSE for metadata

### Cross-Head Regularization

Prevents one head from dominating during training:

```python
reg_loss = variance(generation_loss, control_loss, memory_loss)
```

### Dynamic Learning Rates

Each head can have different learning rates:
- Generation: 1e-5
- Control: 5e-6 (lower for stability)
- Memory: 2e-6 (lowest for embedding stability)

## Best Practices

1. **Data Collection**: Collect at least 100 preference pairs before starting DPO
2. **Head Balancing**: Start with default weights and adjust based on performance
3. **Monitoring**: Use WandB to track head-specific losses and rewards
4. **Evaluation**: Test each head independently after training

## Troubleshooting

### Common Issues

1. **Insufficient Preferences**: Need at least 100 preference pairs
2. **Memory Head Instability**: Reduce memory learning rate or weight
3. **Coordination Loss Too High**: Reduce coordination weight

### Debug Commands

Check preference data:
```bash
python scripts/aggregate_preferences.py \
    --base-path content/worlds \
    --output datasets/preferences.arrow
```

Test reward model:
```python
from narrative_engine.reward_models import GenerationRewardModel
model = GenerationRewardModel()
# Test inference
``` 