# Ring 6: Voice Integration Tasks (Updated)

## R6-1: Orpheus-TTS Integration Foundation

**Objective**: Establish core Orpheus-TTS integration infrastructure for the narrative AI platform

**Technical Requirements**:
- Install and configure Orpheus-TTS (3B finetuned model) 
- Set up inference environment with proper GPU allocation (15GB+ VRAM recommended)
- Implement basic text-to-speech generation pipeline
- Create voice model loading and caching system

**Key Implementation Details**:
- **Model Selection**: Use `canopylabs/orpheus-3b-0.1-ft` (finetuned production model)
- **Emotion Tag Support**: Built-in tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
- **Performance Target**: <200ms latency for real-time narrative interaction
- **Architecture**: Llama-3B backbone enables seamless LLM integration
- **Licensing**: Apache 2.0 allows commercial deployment

**Integration Points**:
- Interface with existing control token system
- Map narrative context to appropriate emotion tags
- Implement streaming audio output for responsive interaction
- Create fallback mechanisms for model unavailability

**Deliverables**:
- Orpheus-TTS service wrapper with API endpoints
- Basic emotion tag injection from narrative context
- Performance benchmarking and latency optimization
- Documentation for voice generation pipeline

**Success Criteria**:
- Generate speech from text with <2 second total latency
- Successfully map control tokens to Orpheus emotion tags
- Demonstrate stable operation under load
- Audio output quality meets narrative standards