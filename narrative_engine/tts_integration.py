"""
TTS Integration for Multimodal NarrativeLM

This module provides integration with various TTS systems for speech synthesis,
including Kokoro-TTS (fast), Orpheus-TTS (expressive), XTTS, Bark, and others.
"""

import asyncio
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torchaudio
import requests
import json

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract base class for TTS providers"""
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text"""
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        pass


class KokoroTTS(TTSProvider):
    """Kokoro-TTS integration (82M parameters, fast and high-quality)"""
    
    def __init__(self, model_name: str = "hexgrad/Kokoro-82M"):
        self.model_name = model_name
        self.pipeline = None
        self.sample_rate = 24000  # Kokoro uses 24kHz
        
        # Available voices from VOICES.md
        self.available_voices = [
            "af_heart", "af_bella", "af_sarah", "af_nicole",  # Female voices
            "am_adam", "am_eric", "am_michael", "am_daniel",   # Male voices
        ]
        
    def _load_pipeline(self):
        """Load Kokoro pipeline lazily"""
        if self.pipeline is None:
            try:
                logger.info(f"Loading Kokoro-TTS pipeline...")
                from kokoro import KPipeline
                self.pipeline = KPipeline(lang_code='a')  # 'a' for American English
                logger.info("Kokoro-TTS pipeline loaded successfully")
            except ImportError:
                logger.error("Kokoro not installed. Run: pip install kokoro>=0.9.2")
                raise
            except Exception as e:
                logger.error(f"Failed to load Kokoro pipeline: {e}")
                raise
    
    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using Kokoro-TTS"""
        self._load_pipeline()
        
        # Select voice (default to af_heart for female, am_adam for male)
        selected_voice = voice_id or "af_heart"
        if selected_voice not in self.available_voices:
            logger.warning(f"Voice {selected_voice} not available, using af_heart")
            selected_voice = "af_heart"
        
        # Note: Kokoro doesn't use emotion tags directly - it's optimized for speed
        # Emotion should be conveyed through text content and voice selection
        
        try:
            logger.info(f"Synthesizing with Kokoro: {text[:50]}... (voice: {selected_voice})")
            
            # Generate audio using Kokoro pipeline
            generator = self.pipeline(text, voice=selected_voice)
            
            # Kokoro returns a generator, we need to collect all audio
            audio_chunks = []
            for i, (gs, ps, audio_chunk) in enumerate(generator):
                audio_chunks.append(audio_chunk)
                logger.debug(f"Generated chunk {i}: {gs}, {ps}")
            
            # Concatenate all chunks
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                # Fallback empty audio
                audio = np.zeros(int(self.sample_rate * 0.1))  # 100ms silence
            
            return audio.astype(np.float32), self.sample_rate
            
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}")
            # Generate fallback audio
            return self._generate_fallback_audio(text)
    
    def _generate_fallback_audio(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate fallback audio if Kokoro fails"""
        duration = len(text) * 0.05  # 50ms per character
        sr = self.sample_rate
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simple sine wave fallback
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # 220Hz tone
        logger.warning("Using fallback audio synthesis")
        
        return audio.astype(np.float32), sr
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available Kokoro voices"""
        voices = []
        for voice_id in self.available_voices:
            # Infer gender and name from voice ID
            if voice_id.startswith('af_'):
                gender = 'female'
                name = voice_id.replace('af_', '').title()
            elif voice_id.startswith('am_'):
                gender = 'male' 
                name = voice_id.replace('am_', '').title()
            else:
                gender = 'neutral'
                name = voice_id.title()
                
            voices.append({
                "id": voice_id,
                "name": f"Kokoro {name}",
                "gender": gender,
                "provider": "kokoro"
            })
        
        return voices


class OrpheusTTS(TTSProvider):
    """Orpheus-TTS integration (3B parameters, expressive with emotion tags)"""
    
    def __init__(self, model_path: str = "canopylabs/orpheus-3b-0.1-ft", use_production_service: bool = True):
        self.model_path = model_path
        self.use_production_service = use_production_service
        self.production_service = None
        self.model = None
        self.tokenizer = None
        self.sample_rate = 22050  # Standard TTS sample rate
        
        # Emotion tag mapping for Orpheus
        self.emotion_tags = {
            "laugh": "<laugh>",
            "chuckle": "<chuckle>", 
            "sigh": "<sigh>",
            "cough": "<cough>",
            "sniffle": "<sniffle>",
            "groan": "<groan>",
            "yawn": "<yawn>",
            "gasp": "<gasp>"
        }
        
    def _load_model(self):
        """Load Orpheus model lazily"""
        if self.model is None:
            try:
                logger.info(f"Loading Orpheus-TTS model from {self.model_path}")
                
                # Try to import Orpheus-specific modules first
                try:
                    # This would be the ideal import if they have a proper Python package
                    from orpheus_tts import OrpheusTTSModel
                    self.model = OrpheusTTSModel.from_pretrained(self.model_path)
                    logger.info("Loaded Orpheus using official package")
                except ImportError:
                    # Fallback to transformers if no official package
                    logger.info("Official Orpheus package not found, trying transformers...")
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    logger.info("Loaded Orpheus using transformers")
                    
            except Exception as e:
                logger.error(f"Failed to load Orpheus model: {e}")
                # Keep model as None, will use fallback
                self.model = None
    
    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using Orpheus-TTS"""
        
        # Try production service first if enabled
        if self.use_production_service:
            try:
                if self.production_service is None:
                    from .orpheus_production_service import get_orpheus_service, SynthesisRequest
                    self.production_service = await get_orpheus_service()
                
                # Create synthesis request
                request = SynthesisRequest(
                    text=text,
                    voice_id=voice_id,
                    emotion_tags=emotion_tags,
                    request_id=f"tts_{int(time.time() * 1000)}"
                )
                
                # Use production service
                result = await self.production_service.synthesize(request)
                logger.info(f"Production Orpheus synthesis: {result.duration_seconds:.3f}s")
                
                return result.audio, result.sample_rate
                
            except Exception as e:
                logger.warning(f"Production service failed, falling back to direct model: {e}")
        
        # Fallback to direct model usage
        self._load_model()
        
        # Add emotion tags to text for Orpheus
        tagged_text = text
        if emotion_tags:
            for tag in emotion_tags:
                if tag in self.emotion_tags:
                    tagged_text = f"{self.emotion_tags[tag]} {text}"
                    break  # Only use first emotion tag
        
        if self.model is not None:
            try:
                logger.info(f"Synthesizing with Orpheus: {tagged_text[:50]}...")
                
                # If using official Orpheus package
                if hasattr(self.model, 'synthesize'):
                    audio = await self.model.synthesize(
                        text=tagged_text,
                        voice_id=voice_id,
                        temperature=temperature
                    )
                else:
                    # Using transformers - would need actual Orpheus inference code
                    # For now, return mock until we have the real implementation
                    logger.warning("Orpheus loaded via transformers, using mock synthesis")
                    audio = self._generate_mock_audio(tagged_text)
                
                return audio.astype(np.float32), self.sample_rate
                
            except Exception as e:
                logger.error(f"Orpheus synthesis failed: {e}")
                return self._generate_mock_audio(tagged_text)
        else:
            logger.warning("Orpheus model not loaded, using mock synthesis")
            return self._generate_mock_audio(tagged_text)
    
    def _generate_mock_audio(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate mock audio for testing"""
        sr = self.sample_rate
        duration = len(text) * 0.05  # 50ms per character
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)
        
        logger.info(f"Generated mock audio for '{text[:20]}...'")
        return audio.astype(np.float32), sr
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available Orpheus voices (mocked for now)"""
        return [
            {"id": "default", "name": "Orpheus Default", "gender": "neutral", "provider": "orpheus"},
            {"id": "narrator", "name": "Orpheus Narrator", "gender": "male", "provider": "orpheus"},
            {"id": "expressive_female", "name": "Orpheus Expressive Female", "gender": "female", "provider": "orpheus"},
            {"id": "expressive_male", "name": "Orpheus Expressive Male", "gender": "male", "provider": "orpheus"}
        ]


class XTTS(TTSProvider):
    """Coqui XTTS v2 integration"""
    
    def __init__(self, api_url: Optional[str] = None):
        self.api_url = api_url or "http://localhost:5000"  # Local XTTS server
        self.voices = {}
        
    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        language: str = "en",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using XTTS"""
        
        # Prepare request
        payload = {
            "text": text,
            "language": language,
            "speaker_wav": voice_id,  # Path to reference audio
            "temperature": kwargs.get("temperature", 0.7),
            "length_penalty": kwargs.get("length_penalty", 1.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 2.0),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.85)
        }
        
        try:
            # Make API request
            response = requests.post(
                f"{self.api_url}/tts",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # Extract audio from response
                audio_data = response.content
                # Convert to numpy array
                audio = np.frombuffer(audio_data, dtype=np.float32)
                return audio, 22050  # XTTS uses 22050 Hz
            else:
                logger.error(f"XTTS API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"XTTS synthesis error: {e}")
        
        # Fallback to mock audio
        return self._generate_fallback_audio(text)
    
    def _generate_fallback_audio(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate fallback audio if XTTS fails"""
        duration = len(text) * 0.05
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)
        return audio.astype(np.float32), sr
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voice references"""
        # In production, scan voice reference directory
        return [
            {"id": "reference_1.wav", "name": "Voice 1", "gender": "female"},
            {"id": "reference_2.wav", "name": "Voice 2", "gender": "male"}
        ]


class BarkTTS(TTSProvider):
    """Suno Bark integration for expressive speech"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_model(self):
        """Load Bark model lazily"""
        if self.model is None:
            logger.info("Loading Bark model...")
            # In production:
            # from transformers import BarkModel, BarkProcessor
            # self.processor = BarkProcessor.from_pretrained("suno/bark")
            # self.model = BarkModel.from_pretrained("suno/bark").to(self.device)
            pass
    
    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using Bark"""
        self._load_model()
        
        # Bark uses voice presets
        voice_preset = voice_id or "v2/en_speaker_0"
        
        # Add non-verbal sounds based on emotion tags
        if emotion_tags:
            if "laugh" in emotion_tags:
                text = f"{text} [laughs]"
            elif "sigh" in emotion_tags:
                text = f"[sighs] {text}"
        
        # Mock synthesis
        logger.info(f"Synthesizing with Bark: {text[:50]}...")
        
        # Generate audio with Bark characteristics
        duration = len(text) * 0.06  # Bark is slightly slower
        sr = 24000  # Bark uses 24kHz
        t = np.linspace(0, duration, int(sr * duration))
        
        # Bark-style audio (more expressive)
        base_freq = 180
        audio = 0.3 * np.sin(2 * np.pi * base_freq * t)
        
        # Add expressiveness
        expression = 1 + 0.3 * np.sin(2 * np.pi * 3 * t)
        audio *= expression
        
        # Add character
        audio += 0.1 * np.sin(2 * np.pi * base_freq * 1.5 * t)
        
        return audio.astype(np.float32), sr
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get Bark voice presets"""
        voices = []
        for lang in ["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"]:
            for i in range(10):  # 10 speakers per language
                voices.append({
                    "id": f"v2/{lang}_speaker_{i}",
                    "name": f"{lang.upper()} Speaker {i}",
                    "language": lang
                })
        return voices


class TTSOrchestrator:
    """Orchestrates multiple TTS providers for optimal speech synthesis"""
    
    def __init__(self):
        self.providers = {
            "kokoro": KokoroTTS(),
            "orpheus": OrpheusTTS(),
            "xtts": XTTS(),
            "bark": BarkTTS()
        }
        self.default_provider = "kokoro"
        
    async def synthesize_character_voice(
        self,
        text: str,
        character: Dict[str, Any],
        emotion_tags: Optional[List[str]] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech for a character using appropriate TTS provider"""
        
        # Select provider based on requirements
        if provider is None:
            provider = self._select_provider(character, emotion_tags)
        
        if provider not in self.providers:
            logger.warning(f"Unknown provider {provider}, using default")
            provider = self.default_provider
        
        # Get character voice ID
        voice_id = self._get_voice_id(character, provider)
        
        # Synthesize speech
        tts = self.providers[provider]
        audio, sr = await tts.synthesize(
            text=text,
            voice_id=voice_id,
            emotion_tags=emotion_tags,
            **kwargs
        )
        
        # Post-process audio based on character traits
        audio = self._apply_character_effects(audio, sr, character)
        
        return audio, sr
    
    def _select_provider(
        self, 
        character: Dict[str, Any], 
        emotion_tags: Optional[List[str]]
    ) -> str:
        """Select best TTS provider for character and context"""
        
        # Orpheus for emotional expression (high priority)
        if emotion_tags and any(tag in ["laugh", "sigh", "gasp", "chuckle", "groan", "yawn"] for tag in emotion_tags):
            return "orpheus"
        
        # XTTS for voice cloning (if reference available)
        if character.get("voice_reference"):
            return "xtts"
        
        # Bark for non-verbal sounds and expressiveness
        if character.get("expressive", False):
            return "bark"
        
        # Kokoro for everything else (fast and high-quality default)
        return self.default_provider
    
    def _get_voice_id(self, character: Dict[str, Any], provider: str) -> Optional[str]:
        """Get appropriate voice ID for character and provider"""
        
        if provider == "kokoro":
            # Map character traits to Kokoro voices
            if character.get("gender") == "female":
                return "af_heart"
            elif character.get("gender") == "male":
                return "am_adam"
            return "af_heart"
            
        elif provider == "orpheus":
            # Map character traits to Orpheus voices
            if character.get("gender") == "female":
                return "expressive_female"
            elif character.get("gender") == "male":
                return "expressive_male"
            return "narrator"
            
        elif provider == "xtts":
            # Use voice reference if available
            return character.get("voice_reference")
            
        elif provider == "bark":
            # Select Bark preset based on character
            lang = character.get("language", "en")
            speaker_idx = hash(character.get("name", "")) % 10
            return f"v2/{lang}_speaker_{speaker_idx}"
        
        return None
    
    def _apply_character_effects(
        self, 
        audio: np.ndarray, 
        sr: int, 
        character: Dict[str, Any]
    ) -> np.ndarray:
        """Apply character-specific audio effects"""
        
        personality = character.get("personality", {})
        
        # Adjust pitch based on personality
        if personality.get("extraversion", 0.5) > 0.7:
            # More energetic, slightly higher pitch
            audio = self._pitch_shift(audio, sr, semitones=1)
        elif personality.get("neuroticism", 0.5) > 0.7:
            # More nervous, add slight tremolo
            audio = self._add_tremolo(audio, sr, rate=4, depth=0.05)
        
        # Adjust dynamics based on confidence
        confidence = personality.get("conscientiousness", 0.5)
        if confidence < 0.3:
            # Less confident, quieter with more variation
            envelope = 1 + 0.2 * np.sin(2 * np.pi * 0.5 * np.arange(len(audio)) / sr)
            audio *= envelope * 0.8
        
        return audio
    
    def _pitch_shift(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Simple pitch shifting (placeholder)"""
        # In production, use librosa.effects.pitch_shift
        return audio  # Placeholder
    
    def _add_tremolo(self, audio: np.ndarray, sr: int, rate: float, depth: float) -> np.ndarray:
        """Add tremolo effect"""
        t = np.arange(len(audio)) / sr
        tremolo = 1 + depth * np.sin(2 * np.pi * rate * t)
        return audio * tremolo


class MicroserviceTTSProvider(TTSProvider):
    """A placeholder for the R6-1 TTS microservice provider."""
    def __init__(self, service_url: str):
        self.service_url = service_url

    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text"""
        # This is a mock implementation
        return np.zeros(1), 22050

    async def synthesize_speech(
        self,
        text: str,
        character: Dict[str, Any],
        emotion_tags: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, int]:
        return await self.synthesize(text, character.get("id"), emotion_tags)

    def get_available_voices(self) -> List[Dict[str, Any]]:
        return [{"id": "mock_voice", "name": "Mock Voice", "gender": "neutral", "provider": "microservice"}]


# Utility functions for mel-spectrogram processing
def audio_to_mel_spectrogram(
    audio: np.ndarray, 
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024
) -> np.ndarray:
    """Convert audio to mel-spectrogram using consistent parameters"""
    import librosa
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=0,
        fmax=8000
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to [time, freq]
    return mel_spec_db.T


def quantize_mel_spectrogram(
    mel_spec: np.ndarray,
    num_bits: int = 4
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Quantize mel-spectrogram to discrete values for NarrativeLM"""
    num_levels = 2 ** num_bits
    
    # Compute global min/max
    mel_min = np.min(mel_spec)
    mel_max = np.max(mel_spec)
    
    # Create quantization levels
    levels = np.linspace(mel_min, mel_max, num_levels)
    
    # Quantize
    discrete_mel = np.zeros_like(mel_spec, dtype=np.int32)
    for i in range(mel_spec.shape[1]):
        mel_bin = mel_spec[:, i]
        indices = np.searchsorted(levels, mel_bin)
        indices = np.clip(indices, 0, num_levels - 1)
        discrete_mel[:, i] = indices
    
    # Create codebook
    codebook = {
        "mel_min": float(mel_min),
        "mel_max": float(mel_max),
        "num_levels": num_levels,
        "levels": levels.tolist()
    }
    
    return discrete_mel, codebook


# Demo function
async def demo_tts_synthesis():
    """Demonstrate TTS synthesis with different providers"""
    
    orchestrator = TTSOrchestrator()
    
    # Test character
    character = {
        "name": "Elena",
        "gender": "female",
        "personality": {
            "extraversion": 0.8,
            "conscientiousness": 0.7,
            "neuroticism": 0.3
        },
        "expressive": True
    }
    
    # Test text with emotion
    text = "Oh my goodness! I can't believe we finally found it!"
    emotion_tags = ["surprise", "joy"]
    
    # Synthesize with automatic provider selection
    audio, sr = await orchestrator.synthesize_character_voice(
        text=text,
        character=character,
        emotion_tags=emotion_tags
    )
    
    logger.info(f"Synthesized audio: {len(audio)} samples at {sr} Hz")
    
    # Convert to mel-spectrogram
    mel_spec = audio_to_mel_spectrogram(audio, sr)
    discrete_mel, codebook = quantize_mel_spectrogram(mel_spec)
    
    logger.info(f"Mel-spectrogram shape: {mel_spec.shape}")
    logger.info(f"Discrete mel shape: {discrete_mel.shape}")
    logger.info(f"Codebook levels: {codebook['num_levels']}")
    
    return audio, sr, mel_spec, discrete_mel


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_tts_synthesis()) 