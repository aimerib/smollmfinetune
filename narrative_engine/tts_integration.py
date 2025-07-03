"""
TTS Integration for Multimodal NarrativeLM

This module provides integration with various TTS systems for speech synthesis,
including Orpheus-TTS, XTTS, Bark, and others.
"""

import asyncio
import logging
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


class OrpheusTTS(TTSProvider):
    """Orpheus-TTS integration (3B finetuned model)"""
    
    def __init__(self, model_path: str = "canopylabs/orpheus-3b-0.1-ft"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.vocoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Emotion tag mapping
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
            logger.info(f"Loading Orpheus-TTS model from {self.model_path}")
            # Placeholder for actual model loading
            # In production:
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self.vocoder = load_hifigan_vocoder()
            pass
    
    async def synthesize(
        self, 
        text: str, 
        voice_id: Optional[str] = None,
        emotion_tags: Optional[List[str]] = None,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using Orpheus-TTS"""
        self._load_model()
        
        # Add emotion tags to text
        if emotion_tags:
            for tag in emotion_tags:
                if tag in self.emotion_tags:
                    text = f"{self.emotion_tags[tag]} {text}"
                    break  # Only use first emotion tag
        
        # Mock synthesis for now
        # In production, use actual Orpheus inference
        logger.info(f"Synthesizing with Orpheus: {text[:50]}...")
        
        # Generate mock audio
        duration = len(text) * 0.05  # 50ms per character estimate
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create more realistic speech-like waveform
        base_freq = 200 if voice_id and "female" in voice_id.lower() else 120
        harmonics = [1, 2, 3, 4, 5]  # Fundamental + harmonics
        audio = np.zeros_like(t)
        
        for i, harmonic in enumerate(harmonics):
            amplitude = 0.5 / (i + 1)  # Decreasing amplitude for harmonics
            audio += amplitude * np.sin(2 * np.pi * base_freq * harmonic * t)
        
        # Add formant-like modulation
        formant_mod = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)  # 5Hz modulation
        audio *= formant_mod
        
        # Add slight noise for realism
        audio += 0.02 * np.random.randn(len(audio))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32), sr
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voice configurations"""
        return [
            {"id": "default", "name": "Default Voice", "gender": "neutral"},
            {"id": "narrator", "name": "Narrator", "gender": "male"},
            {"id": "character_female", "name": "Female Character", "gender": "female"},
            {"id": "character_male", "name": "Male Character", "gender": "male"}
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
            "orpheus": OrpheusTTS(),
            "xtts": XTTS(),
            "bark": BarkTTS()
        }
        self.default_provider = "orpheus"
        
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
        
        # Orpheus for emotional expression
        if emotion_tags and any(tag in ["laugh", "sigh", "gasp"] for tag in emotion_tags):
            return "orpheus"
        
        # Bark for non-verbal sounds and expressiveness
        if character.get("expressive", False):
            return "bark"
        
        # XTTS for voice cloning
        if character.get("voice_reference"):
            return "xtts"
        
        return self.default_provider
    
    def _get_voice_id(self, character: Dict[str, Any], provider: str) -> Optional[str]:
        """Get appropriate voice ID for character and provider"""
        
        if provider == "orpheus":
            # Map character traits to Orpheus voices
            if character.get("gender") == "female":
                return "character_female"
            elif character.get("gender") == "male":
                return "character_male"
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