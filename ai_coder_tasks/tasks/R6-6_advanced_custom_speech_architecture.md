# R6-6: Advanced Custom Speech Architecture (Research Alternative)
Status: **Todo**
Ring: R6
Created: 2025-01-20
---

## Goal
Develop novel speech synthesis architecture optimized specifically for narrative generation within the unified React+FastAPI platform, providing an advanced alternative to Orpheus with deep platform integration.

## Context
**Post-Migration**: This task assumes completion of R6-3.1, R6-3.2, R6-3.3 (Architecture Migration) and R6-5 (Multimodal Architecture Extension)

The unified platform now supports multimodal generation and streaming. This task develops a custom speech architecture that leverages the Dreamcast platform's capabilities for immersive, narrative-focused speech synthesis with deep React UI integration.

**Research Alternative**: Custom flow-matching architecture optimized for narrative contexts, with React-based training and monitoring interfaces.

## Acceptance Criteria

### Custom Speech Architecture
- [ ] **Flow-Matching Generator**: Continuous mel-spectrogram prediction using flow matching
- [ ] **Narrative-Aware Attention**: Attention mechanisms optimized for story contexts
- [ ] **Character Conditioning**: Deep character voice integration with personality traits
- [ ] **Real-time Inference**: Streaming generation compatible with platform WebSocket architecture
- [ ] **Platform Integration**: Native integration with React+FastAPI infrastructure

### React Training Interface
- [ ] **Training Dashboard**: React interface for monitoring flow-matching training
- [ ] **Architecture Visualization**: Interactive visualization of model architecture
- [ ] **Hyperparameter Tuning**: Real-time hyperparameter adjustment interface
- [ ] **Quality Monitoring**: Real-time speech quality metrics and visualization
- [ ] **Model Comparison**: A/B testing interface for different architectures

### Advanced Features Implementation
- [ ] **Zero-Shot Voice Cloning**: Speaker embedding extraction from reference audio
- [ ] **Real-Time Voice Conversion**: Dynamic character voice switching mid-sentence
- [ ] **Multi-Language Support**: Language-specific mel-spectrogram predictors
- [ ] **Environmental Effects**: Acoustic environment modeling and application
- [ ] **Emotion-Aware Morphing**: Real-time voice morphing based on emotional state

### Platform Integration
- [ ] **FastAPI Endpoints**: API endpoints for custom model training and inference
- [ ] **WebSocket Streaming**: Real-time speech generation via platform WebSocket
- [ ] **React Components**: Voice synthesis components integrated with platform UI
- [ ] **Model Management**: Version control and deployment within platform infrastructure
- [ ] **Performance Monitoring**: Integration with platform monitoring and analytics

## Technical Architecture Design

### Flow-Matching Speech Generator
```python
class NarrativeFlowMatchingTTS(nn.Module):
    """Custom flow-matching TTS optimized for narrative generation"""
    
    def __init__(self, hidden_dim=768, num_mel_bins=80):
        super().__init__()
        self.text_encoder = NarrativeTextEncoder(hidden_dim)
        self.flow_matcher = ContinuousFlowMatcher(hidden_dim, num_mel_bins)
        self.character_conditioner = CharacterConditioner(hidden_dim)
        self.narrative_attention = NarrativeAwareAttention(hidden_dim)
        
    def forward(self, text_tokens, character_id, narrative_context, target_mel=None):
        # Encode text with narrative awareness
        text_hidden = self.text_encoder(text_tokens, narrative_context)
        
        # Apply character conditioning
        char_conditioned = self.character_conditioner(text_hidden, character_id)
        
        # Apply narrative-aware attention
        attended_features = self.narrative_attention(
            char_conditioned, narrative_context
        )
        
        # Flow matching for mel-spectrogram generation
        if self.training and target_mel is not None:
            loss = self.flow_matcher.compute_loss(attended_features, target_mel)
            return loss
        else:
            mel_output = self.flow_matcher.generate(attended_features)
            return mel_output
```

### React Training Dashboard
```typescript
const FlowMatchingTrainingDashboard: React.FC = () => {
  const [trainingMetrics, setTrainingMetrics] = useState<FlowMatchingMetrics>();
  const [modelArchitecture, setModelArchitecture] = useState<ArchitectureConfig>();
  const [isTraining, setIsTraining] = useState(false);
  const wsRef = useRef<WebSocket>();
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/flow-matching-training');
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'training_metrics') {
        setTrainingMetrics(data.metrics);
      } else if (data.type === 'architecture_update') {
        setModelArchitecture(data.architecture);
      }
    };
    
    return () => ws.close();
  }, []);
  
  return (
    <div className="flow-matching-dashboard">
      <TrainingControls 
        onStart={startFlowMatchingTraining}
        onStop={stopTraining}
        isTraining={isTraining}
      />
      <FlowMatchingLossChart metrics={trainingMetrics} />
      <ArchitectureVisualizer architecture={modelArchitecture} />
      <SpeechQualityMonitor metrics={trainingMetrics?.quality} />
      <HyperparameterTuner 
        onUpdate={updateHyperparameters}
        currentParams={trainingMetrics?.hyperparams}
      />
    </div>
  );
};
```

### Narrative-Aware Components
```python
class NarrativeAwareAttention(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.text_attention = nn.MultiheadAttention(hidden_dim, 12)
        self.control_modulator = nn.Linear(200, hidden_dim)  # 200 control tokens
        self.character_embeddings = nn.Embedding(1000, hidden_dim)  # 1000 characters
        self.narrative_projector = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, speech_hidden, text_hidden, control_tokens, character_id, narrative_context):
        # Character-specific attention bias
        char_bias = self.character_embeddings(character_id)
        
        # Narrative context integration
        narrative_features = self.narrative_projector(narrative_context)
        
        # Control-modulated attention
        control_modulation = self.control_modulator(control_tokens)
        modulated_text = text_hidden + control_modulation + narrative_features
        
        # Cross-modal attention with character and narrative bias
        attended_output, alignment = self.text_attention(
            query=speech_hidden + char_bias,
            key=modulated_text,
            value=modulated_text
        )
        return attended_output, alignment
```

## Implementation Notes
```text
• Platform Architecture:
  - Custom FastAPI endpoints for flow-matching training and inference
  - React components for training visualization and model management
  - WebSocket integration for real-time training monitoring
  - Integration with platform's character and narrative systems
  
• Flow-Matching Design:
  - Continuous mel-spectrogram prediction for high quality
  - Narrative-aware positional encoding for story context
  - Character-conditioned layer normalization for voice consistency
  - Multi-scale attention for phoneme and prosody modeling
  
• Training Strategy:
  - Mixed-precision training (FP16) for memory efficiency
  - Curriculum learning from simple to complex narratives
  - Real-time monitoring via React dashboard
  - A/B testing framework for architecture comparison
  
• Platform Integration:
  - Native integration with emotion control system
  - Character voice consistency with existing systems
  - Seamless switching between Orpheus and custom architecture
  - Export capabilities for cartridge integration
```

## TDD Instructions
- **Model Tests**: Test flow-matching architecture and forward pass
- **Training Tests**: Test custom training pipeline and curriculum learning
- **API Tests**: Test FastAPI endpoints for training and inference
- **React Tests**: Test training dashboard and visualization components
- **Integration Tests**: Test platform integration and voice consistency

## Checklist / Steps
1. **Design flow-matching architecture** optimized for narrative generation
2. **Implement narrative-aware attention** mechanisms and components
3. **Create React training dashboard** with real-time monitoring
4. **Implement FastAPI endpoints** for model training and management
5. **Add character conditioning** and voice consistency features
6. **Create zero-shot voice cloning** system with speaker embeddings
7. **Implement real-time voice conversion** and morphing capabilities
8. **Add multi-language support** with language-specific predictors
9. **Create environmental effects** and acoustic modeling
10. **Implement A/B testing framework** for architecture comparison
11. **Add comprehensive evaluation** metrics and monitoring
12. **Create model versioning** and deployment system
13. **Implement streaming inference** for real-time generation
14. **Add platform integration** with existing character and narrative systems
15. **Create comprehensive documentation** and user guides

## References
- Depends on: R6-5 (Multimodal Architecture Extension)
- Enables: R6-7 (Multi-Character Conversation)
- Architecture: See overview.mdc architecture diagram
- Platform Integration: React+FastAPI unified architecture
- Research: Flow-Omni methodology and dMel quantization