## R6-5: Multimodal NarrativeLM Architecture Extension

**Objective**: Extend NarrativeLM's triple-head architecture to quad-head with integrated speech generation capabilities

**Technical Architecture Design**:

**Current Triple-Head → Quad-Head Extension**:
```
Generation Head: Text token prediction (existing)
Control Head: Narrative control token prediction (existing)
Memory Head: Memory update operations (existing)
Speech Head: Mel-spectrogram frame prediction (NEW)
```

**Speech Head Detailed Implementation**:

**Core Architecture**:
- **Base Design**: Multi-layer transformer decoder with speech-specific modifications
- **Model Dimensions**: 768 hidden units, 12 attention heads, 6 decoder layers
- **Input Processing**: Receives shared transformer representations from backbone
- **Output Specification**: 80-dimensional mel-spectrogram frames at 25ms resolution
- **Temporal Modeling**: Causal attention with 1000-frame context window

**Speech Tokenization Strategy** (Based on Visatronic/dMel Research):
- **Approach**: Discrete mel-spectrogram quantization following dMel methodology
- **Quantization**: 4-bit quantization per mel-bin (16 discrete levels)
- **Codebook**: Evenly spaced values in [mel_min, mel_max] range computed across dataset
- **Frame Structure**: 80 mel-bins × 4-bit quantization = 320 discrete tokens per frame
- **Embedding**: Each discrete value mapped via learnable embedding to 768-dim space

**Multi-Head Integration Architecture**:
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

**Speech Head Internal Architecture**:
```python
class SpeechHead(nn.Module):
    def __init__(self, mel_bins=80, quantization_bits=4):
        self.mel_bins = mel_bins
        self.num_discrete_values = 2 ** quantization_bits  # 16 levels
        
        # Speech-specific processing layers
        self.speech_projector = nn.Linear(768, 512)
        self.temporal_attention = nn.MultiheadAttention(512, 8)
        self.mel_decoders = nn.ModuleList([
            nn.Linear(512, self.num_discrete_values) 
            for _ in range(mel_bins)
        ])
        
        # Cross-modal attention for text-speech alignment
        self.cross_attention = nn.MultiheadAttention(512, 8)
        
    def forward(self, hidden_states, text_context):
        # Project to speech space
        speech_features = self.speech_projector(hidden_states)
        
        # Temporal modeling within speech
        speech_features, _ = self.temporal_attention(
            speech_features, speech_features, speech_features
        )
        
        # Cross-attention with text for alignment
        aligned_features, _ = self.cross_attention(
            speech_features, text_context, text_context
        )
        
        # Predict discrete values for each mel-bin independently
        mel_predictions = []
        for i, decoder in enumerate(self.mel_decoders):
            mel_predictions.append(decoder(aligned_features))
        
        return torch.stack(mel_predictions, dim=-1)  # [batch, seq, 16, 80]
```

**Training Configuration**:
- **Multi-Task Loss Function**:
  ```python
  total_loss = (
      1.0 * text_generation_loss +      # Cross-entropy for text
      0.5 * control_prediction_loss +   # Cross-entropy for controls  
      0.3 * memory_update_loss +        # MSE for memory operations
      2.0 * speech_generation_loss      # Cross-entropy for mel-frames
  )
  ```
- **Curriculum Learning**: 
  - Phase 1 (0-50k steps): Text + Control + Memory heads only
  - Phase 2 (50k-100k steps): Add speech head with 50% probability
  - Phase 3 (100k+ steps): Full multimodal training
- **Data Requirements**: Text-speech aligned pairs with control annotations
- **Optimization**: AdamW (lr=1e-4), gradient clipping (max_norm=1.0), warmup schedule

**Speech Generation Pipeline**:
1. **Input Processing**: Text tokens processed through shared backbone
2. **Cross-Modal Attention**: Speech head attends to text representations
3. **Mel-Frame Prediction**: Generate discrete mel-spectrogram values autoregressively
4. **Vocoder Integration**: Convert discrete mel-frames to continuous spectrograms
5. **Audio Synthesis**: HiFi-GAN vocoder generates final waveform

**Technical Integration Details**:
- **Frame Alignment**: 25ms mel-frames aligned with ~3-4 text tokens (assuming 150ms per token)
- **Attention Masking**: Causal masking for autoregressive speech generation
- **Memory Efficiency**: Gradient checkpointing for large sequence lengths
- **Streaming Support**: Frame-by-frame generation for real-time synthesis

**Deliverables**:
- Quad-head NarrativeLM architecture implementation
- Speech head with mel-spectrogram prediction
- Multi-task training pipeline with curriculum learning
- Speech-text alignment and synchronization system
- Vocoder integration and audio synthesis pipeline

**Success Criteria**:
- Successfully extend tri-head to quad-head without degrading existing performance
- Generate coherent mel-spectrograms synchronized with text generation
- Achieve reasonable speech quality after vocoder conversion
- Maintain real-time inference capability for interactive use
