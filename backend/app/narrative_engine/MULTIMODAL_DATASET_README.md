# Synthetic Multimodal Dataset Generation for NarrativeLM

This document describes the synthetic dataset generation pipeline for bootstrapping the pretraining of the multimodal NarrativeLM with its quad-head architecture (text generation, control tokens, memory operations, and speech synthesis).

## Overview

The multimodal NarrativeLM requires aligned training data across four modalities:
1. **Text Generation**: Natural language narrative text
2. **Control Tokens**: Emotional and narrative control signals
3. **Memory Operations**: Memory vectors with importance/surprise/valence metadata
4. **Speech Synthesis**: Discrete mel-spectrograms (4-bit quantized, 80 mel bins)

Our synthetic dataset generation pipeline creates this aligned multimodal data at scale, enabling bootstrapping of the model before fine-tuning on high-quality real data.

## Architecture

### Quad-Head NarrativeLM Architecture

```
Shared Transformer Backbone
           ↓
    ┌──────┴──────┬──────────┬───────────┐
    ↓             ↓          ↓           ↓
Generation    Control    Memory      Speech
  Head         Head       Head        Head
    ↓             ↓          ↓           ↓
  Text        Control    Memory    Mel-Spectrogram
 Tokens       Tokens     Vectors      Frames
```

### Speech Head Specifications

- **Input**: Shared transformer representations (768-dim)
- **Output**: 80-dimensional mel-spectrogram frames at 25ms resolution
- **Quantization**: 4-bit per mel bin (16 discrete levels)
- **Alignment**: Cross-attention between text and speech representations
- **Frame Rate**: ~40 frames per second (25ms hop length at 22.05kHz)

## Dataset Generation Pipeline

### 1. Character Generation

The pipeline generates diverse narrative characters based on archetypes:

```python
character_archetypes = ["hero", "mentor", "trickster", "shadow", "herald", "shapeshifter"]
```

Each character includes:
- **Name and background**
- **Big Five personality traits** (0-1 scores)
- **Speaking style and voice characteristics**
- **Emotional tendencies**
- **Goals and relationships**

### 2. Narrative Text Generation

Six types of narrative content are generated:

1. **Dialogue**: Character conversations
2. **Monologue**: Internal reflections
3. **Action Scene**: Dynamic sequences
4. **Emotional Moment**: Vulnerable character moments
5. **Memory Recall**: Past experiences
6. **World Description**: Environmental narrative

### 3. Control Token Extraction

Control tokens are automatically extracted based on:

- **Emotion Detection**: `[EMOTION:happy]`, `[EMOTION:sad]`, etc.
- **Pace Control**: `[PACE:fast]`, `[PACE:slow]`
- **Energy Levels**: `[ENERGY:high]`, `[TENSION:high]`
- **Character Traits**: Based on Big Five personality scores

### 4. Speech Synthesis

Three TTS provider options:

1. **Orpheus-TTS** (Default)
   - 3B parameter model with emotion tags
   - Tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<gasp>`, etc.
   - Best for emotional expression

2. **XTTS v2**
   - Zero-shot voice cloning
   - Best for consistent character voices
   - Requires reference audio

3. **Bark**
   - Highly expressive synthesis
   - Non-verbal sounds support
   - Multi-language capabilities

### 5. Mel-Spectrogram Processing

Audio is converted to discrete mel-spectrograms:

```python
# Parameters
n_mels = 80          # Number of mel bins
n_fft = 1024         # FFT size
hop_length = 256     # ~11.6ms at 22050 Hz
win_length = 1024    # ~46.4ms at 22050 Hz
quantization_bits = 4 # 16 discrete levels
```

### 6. Memory Generation

Memory vectors include:
- **768-dim embedding vector** (normalized)
- **Importance score** (0-1)
- **Surprise factor** (0-1)
- **Emotional valence** (-1 to 1)
- **Persistence score** (0-1)

Memory generation probability: 40% of samples

### 7. Text-Speech Alignment

Simple linear alignment (production would use forced alignment):
- Maps text tokens to mel-spectrogram frames
- Ensures temporal correspondence
- Enables cross-modal attention during training

## Usage

### Basic Generation

```bash
# Generate 1000 samples with mock TTS
python scripts/generate_multimodal_dataset.py \
    --num-samples 1000 \
    --num-characters 20 \
    --output-dir multimodal_dataset
```

### With Real TTS

```bash
# Generate with Orpheus-TTS
python scripts/generate_multimodal_dataset.py \
    --num-samples 1000 \
    --use-real-tts \
    --tts-model orpheus \
    --save-audio
```

### Custom Narrative Types

```bash
# Focus on specific narrative types
python scripts/generate_multimodal_dataset.py \
    --narrative-types dialogue emotional_moment memory_recall \
    --num-samples 5000
```

## Output Format

### Directory Structure

```
multimodal_dataset/
├── chunk_0.json       # First 1000 samples
├── chunk_1.json       # Next 1000 samples
├── ...
├── metadata.json      # Dataset metadata
├── manifest.json      # Training manifest
└── audio/            # Optional audio files
    ├── 0.wav
    ├── 1.wav
    └── ...
```

### Sample Format

Each sample contains:

```json
{
  "text": "Character dialogue or narrative text",
  "tokens": [101, 2023, 2003, ...],
  "mel_frames": [[...], [...], ...],      // [num_frames, 80]
  "discrete_mel": [[...], [...], ...],    // [num_frames, 80] quantized
  "control_tokens": [0, 0, 1, 0, ...],    // Multi-hot encoding
  "control_sequence": ["[EMOTION:joy]", "[PACE:normal]"],
  "memory_vector": [...],                  // 772 values (768 + 4)
  "memory_importance": 0.7,
  "memory_surprise": 0.4,
  "memory_valence": 0.6,
  "memory_persistence": 0.8,
  "text_to_mel_alignment": [[0, 0], [0, 1], ...],
  "character_id": "char_hero_a1b2c3d4",
  "narrative_context": {
    "type": "dialogue",
    "chapter": 3,
    "scene": 15,
    "tension": 0.6,
    "emotion_state": {"joy": 0.7, "sadness": 0.1, ...}
  },
  "session_id": "synthetic_0",
  "turn_index": 0
}
```

## Training Integration

### Curriculum Learning Strategy

1. **Phase 1 (0-50k steps)**: Text + Control + Memory heads only
2. **Phase 2 (50k-100k steps)**: Add speech head with 50% probability
3. **Phase 3 (100k+ steps)**: Full multimodal training

### Loss Weighting

```python
total_loss = (
    1.0 * text_generation_loss +      # Standard language modeling
    0.5 * control_prediction_loss +   # Control token prediction
    0.3 * memory_update_loss +        # Memory operations
    2.0 * speech_generation_loss      # Speech synthesis (higher weight)
)
```

### Data Loading

```python
from narrative_engine.data_loader import MultimodalDataLoader

# Load from manifest
loader = MultimodalDataLoader(
    manifest_path="multimodal_dataset/manifest.json",
    batch_size=16,
    shuffle=True
)

for batch in loader:
    # batch contains aligned data for all four heads
    text_ids = batch["text_tokens"]
    control_labels = batch["control_tokens"]
    memory_targets = batch["memory_vectors"]
    speech_frames = batch["discrete_mel"]
    alignments = batch["alignments"]
```

## Quality Considerations

### Synthetic Data Limitations

1. **Text Quality**: LLM-generated text may lack nuance of human writing
2. **Emotion Accuracy**: Heuristic control token extraction is imperfect
3. **Speech Naturalness**: Mock TTS produces simplified audio
4. **Alignment Precision**: Linear alignment is approximate

### Recommended Workflow

1. **Bootstrap Phase**: Train on 100k+ synthetic samples
2. **Quality Filter**: Use trained model to filter synthetic data
3. **Real Data**: Fine-tune on high-quality real narration/voice acting
4. **Iterative Improvement**: Generate better synthetic data with improved model

## Advanced Features

### Character Voice Consistency

```python
# Ensure consistent voice across character appearances
character_voice_map = {
    "char_hero_123": "voice_reference_hero.wav",
    "char_mentor_456": "voice_reference_mentor.wav"
}
```

### Emotional Arc Tracking

```python
# Track emotional progression through narrative
emotional_arc = track_character_emotions(
    character_id="char_hero_123",
    samples=character_samples
)
```

### World Consistency

```python
# Maintain consistent world facts across samples
world_facts = {
    "setting": "Medieval fantasy",
    "magic_system": "Elemental based",
    "technology_level": "Pre-industrial"
}
```

## Performance Optimization

### Parallel Generation

```python
# Generate samples in parallel
async def generate_parallel(num_workers=4):
    tasks = []
    for i in range(num_workers):
        task = generate_batch(samples_per_worker)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return combine_results(results)
```

### Caching Strategies

1. **Character Cache**: Reuse generated characters
2. **TTS Cache**: Cache common phrases
3. **Mel-Spectrogram Cache**: Reuse similar audio segments

## Troubleshooting

### Common Issues

1. **Memory Usage**: Large datasets may require chunked processing
2. **TTS Latency**: Use mock TTS for rapid prototyping
3. **Alignment Errors**: Verify token-frame correspondence

### Debug Mode

```bash
# Generate small dataset with verbose logging
python scripts/generate_multimodal_dataset.py \
    --num-samples 10 \
    --output-dir debug_dataset \
    --save-audio \
    --log-level DEBUG
```

## Future Enhancements

1. **Forced Alignment**: Use Montreal Forced Aligner for precise text-speech alignment
2. **Emotion Classifier**: Replace heuristics with trained emotion detection
3. **Voice Conversion**: Add voice style transfer capabilities
4. **Multi-Speaker Scenes**: Support conversations with multiple characters
5. **Environmental Audio**: Add ambient sounds and effects
6. **Prosody Control**: Fine-grained control over speech rhythm and intonation

## Citation

If you use this synthetic dataset generation pipeline, please cite:

```bibtex
@software{narrativelm_multimodal_2024,
  title={Synthetic Multimodal Dataset Generation for NarrativeLM},
  author={SmolLM Finetune Team},
  year={2024},
  url={https://github.com/yourusername/narrativelm}
}
``` 