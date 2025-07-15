---
# R6-1 �� Kokoro+Orpheus TTS Integration
Status: **PENDING** 
Ring: R6
Created: 2025-01-20
Updated: 2025-01-20
---

## Goal
Extend the existing TTS integration system to include Kokoro TTS (fast, 80% of cases) and improve Orpheus TTS (premium, emotional guidance, 20% of cases) for multimodal dataset generation and platform TTS capabilities.

## Context
We need to enhance our existing `TTSOrchestrator` system in `narrative_engine/tts_integration.py` with:
- **Kokoro TTS**: 82M parameter model with stellar quality for standard narrative text (24kHz, fast)
- **Orpheus TTS**: 3B parameter Llama-based model with emotional tags (`<yawn>`, `<laugh>`, etc.) for expressive speech
- **Smart Auto-Selection**: Orpheus for emotion, Kokoro for everything else  
- **Unified API**: Single orchestrator managing both models seamlessly

This serves two purposes:
1. **Platform TTS**: Immediate speech synthesis capabilities for the narrative platform
2. **Synthetic Data Generation**: Creating audio+text pairs for training our future 186M multimodal model

## Acceptance Criteria
- [x] Add `KokoroTTS` provider class to existing TTS system
- [x] Improve `OrpheusTTS` with real model loading (fallback to mock if needed)
- [x] Update `TTSOrchestrator` to include Kokoro as default fast provider
- [x] Smart auto-selection: Orpheus for emotion tags, Kokoro for standard text
- [x] Add Kokoro dependency (`kokoro>=0.9.2`) to requirements

## Implementation Notes

### 1. Kokoro Integration (`narrative_engine/tts_integration.py`)

Real implementation using the official Kokoro package:

```python
class KokoroTTS(TTSProvider):
    """Kokoro-TTS integration (82M parameters, fast and high-quality)"""
    
    def __init__(self, model_name: str = "hexgrad/Kokoro-82M"):
        self.model_name = model_name
        self.pipeline = None
        self.sample_rate = 24000  # Kokoro uses 24kHz
        
        # Available voices from VOICES.md  
        self.available_voices = [
            "af_heart", "af_bella", "af_sarah", "af_nicole",  # Female
            "am_adam", "am_eric", "am_michael", "am_daniel",   # Male
        ]
        
    def _load_pipeline(self):
        """Load Kokoro pipeline lazily"""
        if self.pipeline is None:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='a')  # American English
    
    async def synthesize(self, text: str, voice_id: Optional[str] = None, **kwargs):
        """Fast synthesis with Kokoro"""
        self._load_pipeline()
        
        selected_voice = voice_id or "af_heart"
        generator = self.pipeline(text, voice=selected_voice)
        
        # Collect audio chunks from generator
        audio_chunks = []
        for i, (gs, ps, audio_chunk) in enumerate(generator):
            audio_chunks.append(audio_chunk)
        
        audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(2400)
        return audio.astype(np.float32), self.sample_rate
```

### 2. Enhanced Orpheus Integration

Improved with real model loading and better fallback:

```python
class OrpheusTTS(TTSProvider):
    """Orpheus-TTS integration (3B parameters, expressive with emotion tags)"""
    
    def _load_model(self):
        """Load Orpheus model with fallback strategy"""
        try:
            # Try official package first
            from orpheus_tts import OrpheusTTSModel
            self.model = OrpheusTTSModel.from_pretrained(self.model_path)
        except ImportError:
            # Fallback to transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    
    async def synthesize(self, text: str, emotion_tags: Optional[List[str]] = None, **kwargs):
        """Emotional synthesis with Orpheus"""
        # Add emotion tags to text
        tagged_text = text
        if emotion_tags:
            for tag in emotion_tags:
                if tag in self.emotion_tags:
                    tagged_text = f"{self.emotion_tags[tag]} {text}"
                    break
        
        # Use real model if loaded, sophisticated mock otherwise
        if self.model and hasattr(self.model, 'synthesize'):
            audio = await self.model.synthesize(text=tagged_text, **kwargs)
        else:
            audio = self._generate_mock_audio(tagged_text)
            
        return audio.astype(np.float32), self.sample_rate
```

### 3. Updated TTSOrchestrator

Smart provider selection with Kokoro as default:

```python
class TTSOrchestrator:
    def __init__(self):
        self.providers = {
            "kokoro": KokoroTTS(),        # Fast default
            "orpheus": OrpheusTTS(),      # Emotional expression  
            "xtts": XTTS(),              # Voice cloning
            "bark": BarkTTS()            # Experimental
        }
        self.default_provider = "kokoro"
        
    def _select_provider(self, character: Dict, emotion_tags: Optional[List[str]]) -> str:
        """Smart auto-selection logic"""
        # Orpheus for emotional expression
        if emotion_tags and any(tag in ["laugh", "sigh", "gasp", "chuckle", "groan", "yawn"] for tag in emotion_tags):
            return "orpheus"
        
        # XTTS for voice cloning
        if character.get("voice_reference"):
            return "xtts"
            
        # Bark for expressive characters
        if character.get("expressive", False):
            return "bark"
        
        # Kokoro for everything else (fast and high-quality)
        return "kokoro"
```

### 4. Voice Mapping Strategy

Character gender → voice selection:

```python
def _get_voice_id(self, character: Dict, provider: str) -> Optional[str]:
    """Map character traits to appropriate voices"""
    if provider == "kokoro":
        return "af_heart" if character.get("gender") == "female" else "am_adam"
    elif provider == "orpheus":
        return "expressive_female" if character.get("gender") == "female" else "expressive_male"
    # ... other providers
```

### 5. Integration with Multimodal Dataset Generator

Update `narrative_engine/synthetic_multimodal_dataset.py` to use real TTS:

```python
class SpeechSynthesizer:
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config
        # Use TTSOrchestrator instead of mock synthesis
        from .tts_integration import TTSOrchestrator
        self.tts_orchestrator = TTSOrchestrator()
        
    async def synthesize_speech(self, text: str, character: Dict, emotion_tags: List[str]):
        """Use real TTS instead of mock synthesis"""
        audio, sr = await self.tts_orchestrator.synthesize_character_voice(
            text=text,
            character=character, 
            emotion_tags=emotion_tags
        )
        return audio, sr
```

## Guard-rails & Gotchas
- **Model Loading**: Both models load lazily to avoid startup delays
- **Memory Management**: Clear GPU cache if needed between synthesis calls
- **Audio Format Consistency**: Handle different sample rates (Kokoro 24kHz, Orpheus 22kHz)
- **Fallback Strategy**: Always provide working audio even if models fail to load
- **Emotion Tag Parsing**: Handle malformed tags gracefully
- **Voice ID Validation**: Check voice availability before synthesis

## TDD Instructions
1. **Unit Tests**: Test `KokoroTTS` and enhanced `OrpheusTTS` classes
2. **Integration Tests**: Test `TTSOrchestrator` provider selection logic
3. **Performance Tests**: Verify latency targets (Kokoro <200ms, Orpheus <2s)
4. **Fallback Tests**: Test graceful degradation when models unavailable
5. **Voice Tests**: Test character gender → voice ID mapping
6. **Dataset Integration**: Test with multimodal dataset generation

## Success Criteria
- [x] `KokoroTTS` class implemented with real Kokoro package integration
- [x] `OrpheusTTS` improved with better model loading and fallback
- [x] `TTSOrchestrator` updated with smart auto-selection logic  
- [x] Requirements updated with `kokoro>=0.9.2` dependency
- [x] Provider selection logic working (Orpheus for emotion, Kokoro default)
- [x] Voice mapping strategy implemented (character gender → voice ID)
- [x] Multimodal dataset generator updated to use real TTS
- [x] Comprehensive test suite created (`test_tts_integration.py`)
- [ ] Kokoro synthesis completes in <200ms for typical narrative text (requires `pip install kokoro>=0.9.2`)
- [ ] Orpheus synthesis handles emotion tags correctly in <2s (requires model installation)
- [ ] Audio quality suitable for narrative dataset training
- [ ] Full integration tested with actual model installations

## References
- Kokoro TTS: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Orpheus TTS: [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
- Orpheus GitHub: [CanopyAI/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- Existing TTS integration: `narrative_engine/tts_integration.py`
- Multimodal generator: `narrative_engine/synthetic_multimodal_dataset.py` 