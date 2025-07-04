# R6-5: Multimodal NarrativeLM Architecture Extension
Status: **Todo**
Ring: R6
Created: 2025-01-20
---

## Goal
Extend NarrativeLM's triple-head architecture to quad-head with integrated speech generation capabilities, providing seamless integration with the React+FastAPI platform for real-time multimodal interaction.

## Context
**Post-Migration**: This task assumes completion of R6-3.1, R6-3.2, R6-3.3 (Architecture Migration) and R6-4 (Performance Optimization)

The unified platform now supports real-time streaming and performance optimization. This task extends the core NarrativeLM architecture to natively support speech generation, enabling true multimodal character interaction within the Dreamcast console experience.

**Current Architecture**: Triple-Head (Generation + Control + Memory)  
**Target Architecture**: Quad-Head (Generation + Control + Memory + Speech)

## Acceptance Criteria

### Quad-Head Architecture Implementation
- [ ] **Speech Head Integration**: Add speech generation head to existing triple-head architecture
- [ ] **Shared Backbone**: Extend transformer backbone to support speech generation
- [ ] **Multi-Task Training**: Implement training pipeline for all four heads simultaneously
- [ ] **Real-time Inference**: Support streaming inference for speech generation
- [ ] **React Integration**: Provide React components for monitoring and controlling speech generation

### Speech Head Design
- [ ] **Mel-Spectrogram Prediction**: Generate 80-dimensional mel-spectrograms at 25ms resolution
- [ ] **Discrete Quantization**: 4-bit quantization per mel-bin following dMel methodology
- [ ] **Temporal Modeling**: Causal attention with 1000-frame context window
- [ ] **Cross-Modal Attention**: Text-speech alignment for synchronized generation
- [ ] **Character Voice Conditioning**: Character-specific voice generation

### Training Infrastructure
- [ ] **Multi-Task Loss Function**: Balanced loss across all four heads
- [ ] **Curriculum Learning**: Progressive training strategy for multimodal learning
- [ ] **Data Pipeline**: Text-speech aligned datasets with control annotations
- [ ] **Distributed Training**: Support for multi-GPU training of large model
- [ ] **Training Monitoring**: React dashboard for training progress and metrics

### Platform Integration
- [ ] **FastAPI Endpoints**: API endpoints for model training and inference
- [ ] **WebSocket Streaming**: Real-time speech generation via WebSocket
- [ ] **React Training UI**: Interactive training interface with real-time monitoring
- [ ] **Model Management**: Version control and deployment of trained models
- [ ] **Performance Monitoring**: Real-time metrics and performance tracking

## Technical Architecture Design

### Current Triple-Head → Quad-Head Extension
```
Generation Head: Text token prediction (existing)
Control Head: Narrative control token prediction (existing)  
Memory Head: Memory update operations (existing)
Speech Head: Mel-spectrogram frame prediction (NEW)
```

### Speech Head Implementation
```python
class QuadHeadNarrativeLM(nn.Module):
    def __init__(self):
        self.shared_backbone = TransformerBackbone(hidden_dim=768)
        self.generation_head = GenerationHead(vocab_size=50000)
        self.control_head = ControlHead(control_vocab=200)
        self.memory_head = MemoryHead(memory_dim=512)
        self.speech_head = SpeechHead(mel_bins=80, quantization_bits=4)
        
    def forward(self, input_ids, speech_frames=None):
        # Shared representation
        hidden_states = self.shared_backbone(input_ids)
        
        # Multi-head prediction
        text_logits = self.generation_head(hidden_states)
        control_logits = self.control_head(hidden_states)
        memory_updates = self.memory_head(hidden_states)
        
        # Speech generation with cross-attention to text
        if self.training or speech_frames is not None:
            speech_logits = self.speech_head(hidden_states, speech_frames)
            return text_logits, control_logits, memory_updates, speech_logits
        return text_logits, control_logits, memory_updates
```

### React Training Dashboard
```typescript
const MultimodalTrainingDashboard: React.FC = () => {
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics>();
  const [isTraining, setIsTraining] = useState(false);
  const wsRef = useRef<WebSocket>();
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/training-progress');
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      const metrics = JSON.parse(event.data);
      setTrainingMetrics(metrics);
    };
    
    return () => ws.close();
  }, []);
  
  return (
    <div className="training-dashboard">
      <TrainingControls 
        onStart={startTraining} 
        onStop={stopTraining}
        isTraining={isTraining}
      />
      <MultiHeadLossChart metrics={trainingMetrics} />
      <SpeechQualityMonitor metrics={trainingMetrics?.speech} />
      <ModelPerformanceMetrics metrics={trainingMetrics} />
    </div>
  );
};
```

## Implementation Notes
```text
• Architecture Design:
  - Extend existing triple-head without breaking backward compatibility
  - Speech head uses discrete mel-spectrogram quantization
  - Cross-modal attention for text-speech alignment
  - Character-specific voice conditioning via embeddings
  
• Training Strategy:
  - Curriculum learning: text → multimodal gradually
  - Multi-task loss balancing across all heads
  - Distributed training for large model sizes
  - Real-time monitoring via React dashboard
  
• Platform Integration:
  - FastAPI endpoints for training and inference
  - WebSocket streaming for real-time generation
  - React components for training visualization
  - Model versioning and deployment pipeline
  
• Performance Considerations:
  - Gradient checkpointing for memory efficiency
  - Mixed-precision training (FP16)
  - Streaming inference for real-time interaction
  - CUDA graph optimization for speed
```

## TDD Instructions
- **Model Tests**: Test quad-head architecture and forward pass
- **Training Tests**: Test multi-task training pipeline
- **API Tests**: Test FastAPI endpoints for training and inference
- **React Tests**: Test training dashboard and monitoring components
- **Integration Tests**: Test end-to-end multimodal generation

## Checklist / Steps
1. **Design quad-head architecture** extending existing triple-head model
2. **Implement speech head** with mel-spectrogram prediction
3. **Create multi-task training pipeline** with curriculum learning
4. **Implement FastAPI endpoints** for model training and management
5. **Create React training dashboard** with real-time monitoring
6. **Add WebSocket streaming** for real-time speech generation
7. **Implement model versioning** and deployment system
8. **Create data pipeline** for text-speech aligned training data
9. **Add distributed training** support for multi-GPU setups
10. **Implement performance monitoring** and metrics collection
11. **Create speech quality evaluation** metrics and monitoring
12. **Add character voice conditioning** and customization
13. **Implement streaming inference** for real-time generation
14. **Create comprehensive testing** for all components
15. **Add documentation** for multimodal architecture and training

## References
- Depends on: R6-4 (Performance Optimization)
- Enables: R6-6 (Advanced Custom Speech Architecture)
- Architecture: See overview.mdc architecture diagram
- Model Architecture: `narrative_engine/model.py`
- Training Infrastructure: `narrative_engine/` training modules
- Speech Research: dMel quantization methodology
