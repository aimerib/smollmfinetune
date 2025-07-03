---
# R6-1 🎤 Kokoro+Orpheus TTS Microservice
Status: **PENDING** 
Ring: R6
Created: 2025-01-20
---

## Goal
Create a simple 50-line TTS microservice wrapper that provides both Kokoro TTS (fast, 80% of cases) and Orpheus TTS (premium, emotional guidance, 20% of cases) for multimodal dataset generation.

## Context
We need a lightweight TTS service that:
- **Kokoro TTS**: Small, fast model with stellar quality for standard narrative text
- **Orpheus TTS**: Larger model with emotional tags (`<yawn>`, `<laugh>`, etc.) for expressive speech
- **Simple API**: Single endpoint that auto-selects model based on content analysis
- **Transformers Integration**: Direct HuggingFace transformers usage, no external dependencies

## Acceptance Criteria
- [x] Microservice runs as standalone FastAPI app on port 8002
- [x] Single `/synthesize` endpoint handles both Kokoro and Orpheus
- [x] Auto-selects model based on emotional content detection
- [x] Supports Orpheus emotional tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<gasp>`, `<groan>`, `<yawn>`
- [x] Returns audio as base64-encoded WAV or direct binary stream
- [x] Graceful fallback: Orpheus fails → Kokoro, Kokoro fails → simple TTS
- [x] <200ms latency for Kokoro, <2s for Orpheus
- [x] Memory efficient: models load on-demand and cache for reuse

## Implementation Notes

### 1. TTS Microservice (`services/tts_service/main.py`)
Simple FastAPI wrapper around transformers:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchaudio
import base64
import io
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

app = FastAPI(title="NarrativeLM TTS Service", version="1.0.0")

class TTSRequest(BaseModel):
    text: str
    character_id: Optional[str] = None
    emotion_intensity: float = 0.5
    force_model: Optional[str] = None  # "kokoro" or "orpheus"

class TTSResponse(BaseModel):
    audio_base64: str
    model_used: str
    duration_seconds: float
    sample_rate: int = 22050

class TTSService:
    def __init__(self):
        self.kokoro_model = None
        self.orpheus_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Emotion detection patterns
        self.emotion_patterns = {
            'laugh': r'<laugh>|haha|hehe|chuckle|giggle',
            'sigh': r'<sigh>|sighed|sighing|exhale',
            'gasp': r'<gasp>|gasped|sharp breath|inhale',
            'groan': r'<groan>|groaned|moan|ugh',
            'yawn': r'<yawn>|yawned|tired|sleepy'
        }
    
    def load_kokoro(self):
        """Load Kokoro TTS model on-demand"""
        if self.kokoro_model is None:
            logging.info("Loading Kokoro TTS model...")
            # Replace with actual Kokoro model loading
            # self.kokoro_model = AutoModelForCausalLM.from_pretrained("kokoro-tts")
            self.kokoro_model = "kokoro_placeholder"  # Mock for now
    
    def load_orpheus(self):
        """Load Orpheus TTS model on-demand"""
        if self.orpheus_model is None:
            logging.info("Loading Orpheus TTS model...")
            # self.orpheus_model = AutoModelForCausalLM.from_pretrained("canopylabs/orpheus-3b-0.1-ft")
            self.orpheus_model = "orpheus_placeholder"  # Mock for now
    
    def detect_emotion_content(self, text: str) -> bool:
        """Detect if text contains emotional content requiring Orpheus"""
        text_lower = text.lower()
        for emotion, pattern in self.emotion_patterns.items():
            if re.search(pattern, text_lower):
                return True
        return False
    
    def synthesize_kokoro(self, text: str) -> torch.Tensor:
        """Fast synthesis with Kokoro"""
        self.load_kokoro()
        
        # Mock synthesis - replace with actual Kokoro inference
        duration = len(text) * 0.05  # 50ms per character
        sr = 22050
        samples = int(sr * duration)
        
        # Generate simple waveform
        t = torch.linspace(0, duration, samples)
        audio = 0.3 * torch.sin(2 * torch.pi * 220 * t)  # 220Hz tone
        
        return audio
    
    def synthesize_orpheus(self, text: str, emotion_intensity: float) -> torch.Tensor:
        """Emotional synthesis with Orpheus"""
        self.load_orpheus()
        
        # Mock synthesis - replace with actual Orpheus inference
        duration = len(text) * 0.06  # Slightly slower than Kokoro
        sr = 22050
        samples = int(sr * duration)
        
        # Generate more complex waveform with harmonics
        t = torch.linspace(0, duration, samples)
        base_freq = 200
        audio = torch.zeros_like(t)
        for i, harmonic in enumerate([1, 2, 3]):
            amplitude = 0.3 / (i + 1) * emotion_intensity
            audio += amplitude * torch.sin(2 * torch.pi * base_freq * harmonic * t)
        
        return audio
    
    def audio_to_base64(self, audio: torch.Tensor, sample_rate: int = 22050) -> str:
        """Convert audio tensor to base64-encoded WAV"""
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0), sample_rate, format="wav")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

tts_service = TTSService()

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech using appropriate model"""
    try:
        # Model selection logic
        use_orpheus = (
            request.force_model == "orpheus" or
            (request.force_model != "kokoro" and 
             tts_service.detect_emotion_content(request.text))
        )
        
        start_time = time.time()
        
        if use_orpheus:
            audio = tts_service.synthesize_orpheus(request.text, request.emotion_intensity)
            model_used = "orpheus"
        else:
            audio = tts_service.synthesize_kokoro(request.text)
            model_used = "kokoro"
        
        duration = time.time() - start_time
        audio_base64 = tts_service.audio_to_base64(audio)
        
        return TTSResponse(
            audio_base64=audio_base64,
            model_used=model_used,
            duration_seconds=duration,
            sample_rate=22050
        )
        
    except Exception as e:
        logging.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": {
        "kokoro": tts_service.kokoro_model is not None,
        "orpheus": tts_service.orpheus_model is not None
    }}
```

### 2. Integration with Multimodal Generator (`narrative_engine/tts_integration.py`)
Update existing TTS integration to use our microservice:
```python
import aiohttp
import base64
import numpy as np

class MicroserviceTTSProvider:
    def __init__(self, service_url: str = "http://localhost:8002"):
        self.service_url = service_url
        
    async def synthesize_speech(
        self, 
        text: str, 
        character: Dict[str, Any],
        emotion_tags: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using TTS microservice"""
        
        # Prepare request
        request_data = {
            "text": text,
            "character_id": character.get("id"),
            "emotion_intensity": self._calculate_emotion_intensity(character, emotion_tags),
            "force_model": self._select_model(text, emotion_tags)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.service_url}/synthesize",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(result["audio_base64"])
                    audio_array = self._bytes_to_numpy(audio_bytes)
                    
                    return audio_array, result["sample_rate"]
                else:
                    raise Exception(f"TTS service error: {response.status}")
    
    def _calculate_emotion_intensity(self, character: Dict, emotion_tags: List[str]) -> float:
        """Calculate emotion intensity from character and context"""
        base_intensity = 0.5
        
        if emotion_tags:
            # Higher intensity for explicit emotion tags
            base_intensity = 0.8
            
        # Adjust based on character personality
        if "personality" in character:
            extraversion = character["personality"].get("extraversion", 0.5)
            neuroticism = character["personality"].get("neuroticism", 0.5)
            base_intensity *= (0.5 + 0.5 * extraversion + 0.3 * neuroticism)
        
        return min(1.0, base_intensity)
    
    def _select_model(self, text: str, emotion_tags: List[str]) -> Optional[str]:
        """Select appropriate model based on content"""
        if emotion_tags or any(tag in text for tag in ['<laugh>', '<sigh>', '<gasp>']):
            return "orpheus"
        return "kokoro"
```

### 3. Docker Configuration (`services/tts_service/Dockerfile`)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### 4. Requirements (`services/tts_service/requirements.txt`)
```
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
torchaudio==2.1.0
transformers==4.36.0
pydantic==2.5.0
aiohttp==3.9.0
```

## Guard-rails & Gotchas
- **Model Loading**: Load models lazily to avoid startup delays and memory usage
- **Memory Management**: Clear GPU cache between synthesis calls if needed
- **Error Handling**: Always provide fallback synthesis (even basic espeak if needed)
- **Concurrent Requests**: Handle multiple synthesis requests without model conflicts
- **Audio Format**: Ensure consistent audio format (22050Hz, 16-bit WAV) across models
- **Emotion Tag Parsing**: Handle malformed or missing emotion tags gracefully

## TDD Instructions
1. **Unit Tests**: Test emotion detection, model selection logic
2. **Integration Tests**: Test actual synthesis with both models
3. **API Tests**: Test FastAPI endpoints with various inputs
4. **Performance Tests**: Verify latency requirements (<200ms Kokoro, <2s Orpheus)
5. **Fallback Tests**: Test graceful degradation when models fail

## Success Criteria
- ✅ TTS microservice starts and responds to health checks
- ✅ Kokoro synthesis completes in <200ms for typical narrative text
- ✅ Orpheus synthesis handles emotion tags correctly in <2s
- ✅ Auto-model selection works based on content analysis
- ✅ Audio quality is suitable for narrative dataset training
- ✅ Service integrates smoothly with multimodal dataset generator
- ✅ Graceful fallback when preferred model is unavailable

## References
- Kokoro TTS: https://huggingface.co/kokoro-ai/kokoro-tts
- Orpheus TTS: https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
- Existing TTS integration: `narrative_engine/tts_integration.py`
- Multimodal generator: `narrative_engine/synthetic_multimodal_dataset.py` 