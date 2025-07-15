# 🌊 Diffusion Multimodal Training Wizard

The Diffusion Training Wizard is a comprehensive React-based interface for training cutting-edge diffusion models that generate text, speech, control tokens, and memory vectors simultaneously. This wizard guides you through the entire training pipeline from dataset generation to production deployment.

## Overview

The diffusion multimodal model represents a major advancement in character AI training. Unlike traditional autoregressive models that generate text sequentially, this architecture uses a diffusion process to generate all modalities (text, speech, control tokens, memory) simultaneously with cross-modal attention.

### Key Features

- **🧠 Multimodal Generation**: Simultaneous generation of text, speech, control tokens, and memory vectors
- **🔗 Cross-Modal Attention**: Advanced attention mechanisms that align different modalities
- **⚡ Scalable Training**: From 25M parameter validation models to 400M parameter production models
- **📊 Real-time Monitoring**: Live training metrics and progress tracking
- **🎛️ Guided Configuration**: Step-by-step wizard with intelligent defaults
- **💾 Training Control**: Pause, resume, and stop training at any time

## Getting Started

### Prerequisites

- React application running with the wizard component
- Backend API server with diffusion training endpoints
- GPU with at least 8GB VRAM (16GB+ recommended for large models)
- Characters created in your character management system
- Optional: World definitions for context-aware training

### Accessing the Wizard

1. Navigate to the **Diffusion Training** page in your application
2. Ensure your backend API is running and accessible
3. Have at least one character created in your system

## Step-by-Step Guide

### Step 1: Model Configuration

The first step allows you to select your model size and core parameters.

#### Model Size Selection

| Size | Parameters | Memory | Speed | Best For |
|------|------------|---------|-------|----------|
| **Small** | ~25M | ~100MB | Fast | Testing & Validation |
| **Medium** | ~100M | ~400MB | Medium | Development |
| **Large** | ~400M | ~1.6GB | Slow | Production |

#### Core Training Parameters

- **Batch Size**: Number of samples processed simultaneously
  - `2`: Safe for most GPUs (8GB VRAM)
  - `4`: Recommended for most use cases (12GB+ VRAM)
  - `8`: High-end GPUs (16GB+ VRAM)
  - `16`: Multi-GPU setups

- **Learning Rate**: Controls training step size
  - `5e-5`: Conservative, stable training
  - `1e-4`: Recommended default
  - `2e-4`: Aggressive, faster convergence
  - `5e-4`: Experimental, may be unstable

- **Guidance Scale**: Controls how strongly the model follows prompts
  - `3.0`: Subtle guidance, more creative
  - `7.5`: Recommended balance
  - `10.0`: Strong guidance, more controlled
  - `15.0`: Very strong, may reduce creativity

#### Advanced Options

- **Cross-Modal Alignment Weight**: How much to weight alignment between modalities
  - `0.1`: Light alignment, more modality independence
  - `0.3`: Recommended balance
  - `0.5`: Strong alignment
  - `0.8`: Very strong, may over-constrain

- **Inference Steps**: Quality vs speed tradeoff for generation
  - `20`: Fast inference, lower quality
  - `50`: Recommended balance
  - `100`: High quality, slower
  - `200`: Maximum quality, very slow

- **Enable EMA**: Exponential Moving Average for stable training (recommended: ✅)

### Step 2: Dataset Generation

Generate synthetic multimodal training data tailored to your characters.

#### Dataset Configuration

- **Number of Characters**: How many characters to include
  - `10`: Quick test (500 conversations)
  - `25`: Small scale (1,250 conversations)
  - `50`: Medium scale (2,500 conversations)
  - `100`: Large scale (5,000+ conversations)

- **Conversations per Character**: Training data per character
  - `25`: Quick validation
  - `50`: Small scale training
  - `100`: Medium scale training
  - `500`: Large scale training

- **Multimodal Samples Ratio**: Percentage with speech/audio
  - `10%`: Text-heavy dataset
  - `30%`: Recommended balance
  - `50%`: Equal text and multimodal
  - `80%`: Multimodal-heavy dataset

#### Dataset Size Estimation

The wizard automatically calculates:
- Total conversations
- Number of multimodal vs text-only samples
- Estimated generation time (~100 conversations per hour)

### Step 3: Model Validation

Before training, the wizard validates your model architecture.

#### Validation Checks

1. **Configuration Validation**: Ensures all parameters are valid
2. **Model Creation**: Tests that the architecture can be instantiated
3. **Forward Pass**: Validates tensor shapes and operations
4. **Memory Requirements**: Estimates GPU memory needs
5. **Dataset Availability**: Checks for training data

#### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Configuration error | Check parameter ranges and types |
| Model creation failed | Verify GPU drivers and PyTorch installation |
| Forward pass failed | Usually dimension mismatch - check config |
| Insufficient memory | Reduce batch size or use smaller model |
| No datasets found | Generate a dataset first |

### Step 4: Incremental Training

Train your model through progressive phases.

#### Training Phases

1. **Small Scale Training** (5,000 steps)
   - 25M parameter model
   - 10 characters
   - Quick validation of architecture

2. **Medium Scale Training** (20,000 steps)
   - 100M parameter model
   - 25 characters
   - Development and iteration

3. **Large Scale Training** (50,000 steps)
   - 400M parameter model
   - 50 characters
   - Full-quality training

#### Training Controls

- **▶️ Start Training**: Begin training with current configuration
- **⏸️ Pause Training**: Temporarily stop training (preserves state)
- **▶️ Resume Training**: Continue from paused state
- **⏹️ Stop Training**: Permanently stop training

#### Training Metrics

Real-time monitoring includes:
- **Step**: Current training step
- **Loss**: Overall training loss (lower is better)
- **Learning Rate**: Current learning rate (may decay over time)
- **Alignment**: Cross-modal alignment loss

### Step 5: Production Training

Final large-scale training run for production deployment.

#### Production Configuration

- **400M Parameters**: Full-size model
- **100 Characters**: Large character dataset
- **50,000 Conversations**: Comprehensive training data
- **100,000 Steps**: Extended training for quality

#### Production Metrics

Extended metrics for production monitoring:
- **Total Loss**: Overall training objective
- **Text Loss**: Text generation quality
- **Speech Loss**: Speech synthesis quality
- **Alignment Loss**: Cross-modal coordination

## API Integration

The wizard integrates with these backend endpoints:

### Dataset Generation
```
POST /api/v1/diffusion/dataset/generate
```

### Model Validation
```
POST /api/v1/diffusion/model/validate
```

### Training Control
```
POST /api/v1/diffusion/training/start
POST /api/v1/diffusion/training/pause
POST /api/v1/diffusion/training/resume
POST /api/v1/diffusion/training/stop
```

### Monitoring
```
GET /api/v1/diffusion/training/status
GET /api/v1/diffusion/training/metrics
GET /api/v1/diffusion/training/history
```

### Model Management
```
GET /api/v1/diffusion/model/checkpoints
POST /api/v1/diffusion/model/export
```

## Training Best Practices

### 🎯 Model Size Selection

- **Start Small**: Always validate with small models first
- **Scale Gradually**: Progress through small → medium → large
- **Consider Resources**: Match model size to available GPU memory

### 📊 Dataset Quality

- **Character Diversity**: Include varied personality types
- **Balanced Modalities**: Don't over-weight text or speech
- **World Context**: Use consistent world lore for coherence
- **Quality over Quantity**: Better to have fewer high-quality conversations

### ⚡ Training Optimization

- **Monitor Alignment**: Cross-modal alignment loss should decrease
- **Use EMA**: Enable exponential moving average for stability
- **Save Frequently**: Set reasonable save_steps for backup
- **Early Stopping**: Stop if loss plateaus or increases

### 🔧 Troubleshooting

#### Common Training Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Loss not decreasing | Learning rate too low | Increase learning rate |
| Loss exploding | Learning rate too high | Decrease learning rate |
| Poor alignment | Cross-modal weight too low | Increase cross-modal weight |
| Overfitting | Too many steps for data size | Reduce max_steps |
| Out of memory | Batch size too large | Reduce batch_size |

#### Performance Tips

- **GPU Utilization**: Monitor GPU usage (should be 80-90%)
- **Mixed Precision**: Enable if available for speed/memory
- **Gradient Accumulation**: Use for effectively larger batch sizes
- **Checkpointing**: Use gradient checkpointing for memory efficiency

## Advanced Configuration

### Custom Model Architectures

For advanced users, you can modify the diffusion architecture:

```python
# Custom config example
config = get_small_config()
config.transformer.num_layers = 8
config.transformer.num_attention_heads = 12
config.modalities.cross_modal_attention_layers = [2, 4, 6]
```

### Specialized Training

- **Domain Adaptation**: Fine-tune pre-trained models on specific domains
- **Multi-Character**: Train single model for multiple character voices
- **Conditional Generation**: Add conditioning tokens for controlled generation

## Monitoring and Evaluation

### Training Metrics Dashboard

The wizard provides real-time monitoring:
- Loss curves for each modality
- Cross-modal alignment progression
- Training speed and efficiency
- GPU memory utilization

### Quality Evaluation

Post-training evaluation includes:
- Character voice consistency
- Multimodal coherence
- Speech quality metrics
- Control token accuracy

## Deployment

### Model Export Options

1. **ONNX Format**: For optimized inference
2. **TensorRT**: For NVIDIA GPU acceleration
3. **Cartridge**: For runtime deployment

### Runtime Integration

Trained models can be exported as "cartridges" for use in:
- Character runtime engines
- Interactive applications
- Production deployments

## Troubleshooting

### Common Wizard Issues

#### Wizard Won't Load
- Check React app is running
- Verify component imports
- Check browser console for errors

#### API Connection Failed
- Ensure backend server is running
- Check API endpoint URLs
- Verify CORS configuration

#### Training Won't Start
- Validate model configuration
- Check dataset availability
- Verify GPU availability

### Performance Issues

#### Slow Training
- Reduce batch size
- Use smaller model
- Check GPU utilization

#### Memory Issues
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model size

## Support and Resources

### Getting Help

1. Check this documentation first
2. Review console logs for errors
3. Check training metrics for anomalies
4. Consult the troubleshooting section

### Additional Resources

- [Diffusion Architecture Documentation](./diffusion-architecture.md)
- [Character Creation Guide](./core-concepts.md)
- [Training Best Practices](./training-guide.md)

---

**Happy Training!** 🚀 The diffusion wizard makes advanced multimodal training accessible while providing the flexibility for expert customization. 