"""
Synthetic Multimodal Dataset Generation Pipeline for NarrativeLM

This module generates synthetic multimodal training data for the quad-head NarrativeLM,
including text, speech (mel-spectrograms), control tokens, and memory operations.
"""

import asyncio
import json
import logging
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import librosa
import soundfile as sf
from collections import defaultdict
import hashlib
import uuid

# Import existing infrastructure
from .data_schema import DatasetSample, Turn
from .config import NarrativeLLMConfig
from .character_voice_integration import CharacterVoiceSynthesizer
from app.utils.openai_client import get_client
from app.utils.dataset import character_analysis, prompt_generators
from app.utils.control_tokens import load_control_tokens

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """A single multimodal training sample for quad-head NarrativeLM"""
    # Text data
    text: str
    tokens: List[int]
    
    # Speech data (discretized mel-spectrograms)
    mel_frames: np.ndarray  # [num_frames, 80] - 80 mel bins
    discrete_mel: np.ndarray  # [num_frames, 80] - 4-bit quantized
    
    # Control tokens (multi-hot encoding)
    control_tokens: np.ndarray  # [num_control_tokens]
    control_sequence: List[str]  # Actual control token names
    
    # Memory operations
    memory_vector: np.ndarray  # [768 + 4] - embedding + metadata
    memory_importance: float
    memory_surprise: float
    memory_valence: float
    memory_persistence: float
    
    # Alignment data
    text_to_mel_alignment: List[Tuple[int, int]]  # [(token_idx, mel_frame_idx)]
    
    # Metadata
    character_id: str
    narrative_context: Dict[str, Any]
    session_id: str
    turn_index: int
    voice_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticGenerationConfig:
    """Configuration for synthetic multimodal dataset generation"""
    # Generation parameters
    num_samples: int = 10000
    narrative_types: List[str] = field(default_factory=lambda: [
        "dialogue", "monologue", "action_scene", "emotional_moment", 
        "memory_recall", "world_description"
    ])
    
    # Character parameters
    num_characters: int = 50
    character_archetypes: List[str] = field(default_factory=lambda: [
        "hero", "mentor", "trickster", "shadow", "herald", "shapeshifter"
    ])
    
    # TTS parameters
    tts_model: str = "orpheus"  # or "xtts", "bark", etc.
    tts_emotion_tags: List[str] = field(default_factory=lambda: [
        "<laugh>", "<chuckle>", "<sigh>", "<gasp>", "<groan>", "<yawn>"
    ])
    
    # Mel-spectrogram parameters
    sample_rate: int = 22050
    n_mels: int = 80
    hop_length: int = 256  # ~11.6ms at 22050 Hz
    win_length: int = 1024  # ~46.4ms at 22050 Hz
    n_fft: int = 1024
    
    # Quantization parameters
    quantization_bits: int = 4  # 16 discrete levels per mel bin
    
    # Control token parameters
    control_token_probability: float = 0.7
    emotion_intensity_range: Tuple[float, float] = (0.3, 0.9)
    
    # Memory parameters
    memory_generation_probability: float = 0.4
    memory_embedding_dim: int = 768
    
    # Output parameters
    output_dir: Path = Path("synthetic_multimodal_dataset")
    save_audio: bool = True  # Save raw audio for validation
    save_mel_images: bool = False  # Save mel-spectrogram visualizations


class CharacterGenerator:
    """Generates diverse narrative characters for synthetic data"""
    
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config
        self.client = get_client()
        
    async def generate_character(self, archetype: str) -> Dict[str, Any]:
        """Generate a character based on archetype"""
        prompt = f"""Create a detailed character profile for a {archetype} character.
        
Include:
- Name
- Personality (Big Five traits as scores 0-1)
- Background story (2-3 sentences)
- Speaking style and voice characteristics
- Emotional tendencies
- Key relationships
- Goals and motivations

Format as JSON."""

        try:
            response = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant specializing in character creation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                response_format={"type": "json_object"}
            )
            
            character = json.loads(response)
            
            # Ensure Big Five traits are present
            if "personality" not in character:
                character["personality"] = {}
            
            for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                if trait not in character["personality"]:
                    character["personality"][trait] = random.uniform(0.3, 0.8)
            
            # Add character ID
            character["id"] = f"char_{archetype}_{uuid.uuid4().hex[:8]}"
            
            return character
            
        except Exception as e:
            logger.error(f"Error generating character: {e}")
            # Return a default character
            return self._get_default_character(archetype)
    
    def _get_default_character(self, archetype: str) -> Dict[str, Any]:
        """Return a default character if generation fails"""
        return {
            "id": f"char_{archetype}_default",
            "name": f"Default {archetype.title()}",
            "personality": {
                "openness": 0.6,
                "conscientiousness": 0.7,
                "extraversion": 0.5,
                "agreeableness": 0.6,
                "neuroticism": 0.4
            },
            "background": f"A mysterious {archetype} with hidden depths.",
            "speaking_style": "Clear and direct",
            "goals": ["Fulfill their role in the narrative"]
        }


class NarrativeTextGenerator:
    """Generates narrative text with control tokens and structure"""
    
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config
        self.client = get_client()
        self.control_tokens = load_control_tokens()
        
    async def generate_narrative_turn(
        self, 
        character: Dict[str, Any],
        narrative_type: str,
        context: Optional[str] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Generate a narrative turn with text and control tokens"""
        
        # Create narrative prompt based on type
        prompt = self._create_narrative_prompt(character, narrative_type, context)
        
        # Generate text
        text = await self._generate_text(character, prompt)
        
        # Analyze text for control tokens
        control_tokens = self._extract_control_tokens(text, character, narrative_type)
        
        # Generate narrative context
        narrative_context = {
            "type": narrative_type,
            "chapter": random.randint(1, 10),
            "scene": random.randint(1, 50),
            "tension": random.uniform(0.2, 0.9),
            "emotion_state": self._analyze_emotion_state(text, character)
        }
        
        return text, control_tokens, narrative_context
    
    def _create_narrative_prompt(self, character: Dict[str, Any], 
                                narrative_type: str, context: Optional[str]) -> str:
        """Create a prompt for narrative generation"""
        prompts = {
            "dialogue": f"Write a short dialogue where {character['name']} is having a conversation. Their personality: {character.get('speaking_style', 'thoughtful')}.",
            "monologue": f"Write an internal monologue for {character['name']} reflecting on their goals: {character.get('goals', ['unknown'])}.",
            "action_scene": f"Describe {character['name']} in an action sequence that reveals their character.",
            "emotional_moment": f"Write an emotional moment for {character['name']} that shows vulnerability.",
            "memory_recall": f"Write {character['name']} recalling an important memory from their past.",
            "world_description": f"Describe the world through {character['name']}'s eyes, showing their unique perspective."
        }
        
        base_prompt = prompts.get(narrative_type, prompts["dialogue"])
        if context:
            base_prompt += f"\n\nContext: {context}"
            
        return base_prompt
    
    async def _generate_text(self, character: Dict[str, Any], prompt: str) -> str:
        """Generate narrative text"""
        try:
            messages = [
                {"role": "system", "content": f"You are writing as {character['name']}. {character.get('background', '')}"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat_complete(
                messages=messages,
                max_tokens=200,
                temperature=0.8
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "The character paused, gathering their thoughts."
    
    def _extract_control_tokens(self, text: str, character: Dict[str, Any], 
                               narrative_type: str) -> List[str]:
        """Extract appropriate control tokens based on text analysis"""
        control_tokens = []
        text_lower = text.lower()
        
        # Emotion detection
        emotion_map = {
            "happy": ["joy", "laugh", "smile", "delight", "cheerful"],
            "sad": ["cry", "tear", "sorrow", "grief", "melancholy"],
            "angry": ["rage", "fury", "anger", "frustrated", "irritated"],
            "fear": ["afraid", "scared", "terrified", "anxious", "nervous"],
            "surprise": ["shocked", "amazed", "astonished", "unexpected"]
        }
        
        for emotion, keywords in emotion_map.items():
            if any(keyword in text_lower for keyword in keywords):
                control_tokens.append(f"[EMOTION:{emotion}]")
        
        # Pace detection
        if narrative_type == "action_scene" or "!" in text:
            control_tokens.append("[PACE:fast]")
        elif narrative_type in ["memory_recall", "emotional_moment"]:
            control_tokens.append("[PACE:slow]")
        
        # Character-specific tokens based on personality
        personality = character.get("personality", {})
        if personality.get("extraversion", 0.5) > 0.7:
            control_tokens.append("[ENERGY:high]")
        elif personality.get("neuroticism", 0.5) > 0.7:
            control_tokens.append("[TENSION:high]")
        
        return control_tokens
    
    def _analyze_emotion_state(self, text: str, character: Dict[str, Any]) -> Dict[str, float]:
        """Analyze emotional state from text"""
        # Simple heuristic analysis - in production, use a proper emotion classifier
        emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "disgust": 0.0
        }
        
        text_lower = text.lower()
        
        # Joy indicators
        joy_words = ["happy", "joy", "laugh", "smile", "wonderful", "great"]
        emotions["joy"] = min(1.0, sum(word in text_lower for word in joy_words) * 0.3)
        
        # Sadness indicators
        sad_words = ["sad", "cry", "tear", "sorrow", "lonely", "miss"]
        emotions["sadness"] = min(1.0, sum(word in text_lower for word in sad_words) * 0.3)
        
        # Add personality influence
        personality = character.get("personality", {})
        if personality.get("neuroticism", 0.5) > 0.6:
            emotions["fear"] += 0.2
            emotions["sadness"] += 0.1
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions


class MelSpectrogramProcessor:
    """Processes audio into discrete mel-spectrograms for the speech head"""
    
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config
        self.num_discrete_values = 2 ** config.quantization_bits
        
    def audio_to_mel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Convert audio to mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to [time, freq]
        mel_spec_db = mel_spec_db.T
        
        return mel_spec_db
    
    def quantize_mel(self, mel_spec: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize mel-spectrogram to discrete values"""
        # Compute min/max for quantization
        mel_min = np.min(mel_spec)
        mel_max = np.max(mel_spec)
        
        # Create quantization levels
        levels = np.linspace(mel_min, mel_max, self.num_discrete_values)
        
        # Quantize each mel bin
        discrete_mel = np.zeros_like(mel_spec, dtype=np.int32)
        for i in range(mel_spec.shape[1]):  # For each mel bin
            # Find nearest quantization level
            mel_bin = mel_spec[:, i]
            indices = np.searchsorted(levels, mel_bin)
            indices = np.clip(indices, 0, self.num_discrete_values - 1)
            discrete_mel[:, i] = indices
        
        # Store codebook info
        codebook = {
            "mel_min": float(mel_min),
            "mel_max": float(mel_max),
            "num_levels": self.num_discrete_values,
            "levels": levels.tolist()
        }
        
        return discrete_mel, codebook
    
    def align_text_to_mel(
        self, 
        text: str, 
        tokens: List[int], 
        mel_frames: int,
        speech_duration: float
    ) -> List[Tuple[int, int]]:
        """Create alignment between text tokens and mel frames"""
        # Simple linear alignment - in production, use forced alignment
        alignments = []
        
        if len(tokens) == 0:
            return alignments
        
        frames_per_token = mel_frames / len(tokens)
        
        for i, token in enumerate(tokens):
            start_frame = int(i * frames_per_token)
            end_frame = int((i + 1) * frames_per_token)
            
            # Add alignment points
            for frame in range(start_frame, min(end_frame, mel_frames)):
                alignments.append((i, frame))
        
        return alignments


class MemoryGenerator:
    """Generates memory vectors and metadata for the memory head"""
    
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config
        self.embedding_dim = config.memory_embedding_dim
        
    def generate_memory_vector(
        self, 
        text: str,
        character: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """Generate memory vector and metadata"""
        
        # Generate memory embedding (random for now, use text encoder in production)
        memory_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)  # Normalize
        
        # Calculate metadata based on narrative context
        importance = self._calculate_importance(text, narrative_context)
        surprise = self._calculate_surprise(text, character, narrative_context)
        valence = self._calculate_valence(text, narrative_context)
        persistence = self._calculate_persistence(narrative_context)
        
        return memory_embedding, importance, surprise, valence, persistence
    
    def _calculate_importance(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate memory importance score"""
        importance = 0.5  # Base importance
        
        # Increase for certain narrative types
        if context["type"] in ["emotional_moment", "memory_recall"]:
            importance += 0.3
        
        # Increase for high tension moments
        importance += context.get("tension", 0.5) * 0.2
        
        # Check for important keywords
        important_words = ["always", "never", "promise", "secret", "truth", "betray"]
        if any(word in text.lower() for word in important_words):
            importance += 0.2
        
        return min(1.0, importance)
    
    def _calculate_surprise(self, text: str, character: Dict[str, Any], 
                          context: Dict[str, Any]) -> float:
        """Calculate surprise factor"""
        surprise = 0.3  # Base surprise
        
        # Check for surprise indicators
        surprise_words = ["suddenly", "unexpected", "shocked", "amazed", "never thought"]
        if any(word in text.lower() for word in surprise_words):
            surprise += 0.4
        
        # Action scenes often have surprises
        if context["type"] == "action_scene":
            surprise += 0.2
        
        return min(1.0, surprise)
    
    def _calculate_valence(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate emotional valence (-1 to 1)"""
        emotion_state = context.get("emotion_state", {})
        
        positive = emotion_state.get("joy", 0) + emotion_state.get("surprise", 0) * 0.5
        negative = emotion_state.get("sadness", 0) + emotion_state.get("anger", 0) + \
                  emotion_state.get("fear", 0) + emotion_state.get("disgust", 0)
        
        valence = (positive - negative) / max(positive + negative, 1.0)
        
        return np.clip(valence, -1.0, 1.0)
    
    def _calculate_persistence(self, context: Dict[str, Any]) -> float:
        """Calculate how long memory should persist"""
        persistence = 0.5  # Base persistence
        
        # Emotional moments and memories should persist longer
        if context["type"] in ["emotional_moment", "memory_recall"]:
            persistence += 0.3
        
        # High importance moments persist longer
        if context.get("tension", 0.5) > 0.7:
            persistence += 0.2
        
        return min(1.0, persistence)


class MultimodalDatasetGenerator:
    """Main class for generating synthetic multimodal dataset"""
    
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config
        self.character_generator = CharacterGenerator(config)
        self.text_generator = NarrativeTextGenerator(config)
        self.character_voice_synthesizer = CharacterVoiceSynthesizer()
        self.mel_processor = MelSpectrogramProcessor(config)
        self.memory_generator = MemoryGenerator(config)
        
        # Control token setup
        self.control_tokens = load_control_tokens()
        self.control_token_to_id = {token['token']: i for i, token in enumerate(self.control_tokens)}
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def generate_dataset(self) -> List[MultimodalSample]:
        """Generate complete synthetic multimodal dataset"""
        logger.info(f"Starting synthetic dataset generation: {self.config.num_samples} samples")
        
        # Generate characters
        characters = await self._generate_characters()
        
        # Generate samples
        samples = []
        for i in range(self.config.num_samples):
            try:
                # Select random character and narrative type
                character = random.choice(characters)
                narrative_type = random.choice(self.config.narrative_types)
                
                # Generate sample
                sample = await self._generate_sample(
                    character=character,
                    narrative_type=narrative_type,
                    sample_index=i
                )
                
                if sample:
                    samples.append(sample)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{self.config.num_samples} samples")
                    
            except Exception as e:
                logger.error(f"Error generating sample {i}: {e}")
                continue
        
        logger.info(f"Generated {len(samples)} valid samples")
        
        # Save dataset
        self._save_dataset(samples)
        
        return samples
    
    async def _generate_characters(self) -> List[Dict[str, Any]]:
        """Generate diverse characters"""
        characters = []
        
        for archetype in self.config.character_archetypes:
            num_chars = max(1, self.config.num_characters // len(self.config.character_archetypes))
            
            for _ in range(num_chars):
                character = await self.character_generator.generate_character(archetype)
                characters.append(character)
        
        logger.info(f"Generated {len(characters)} characters")
        return characters
    
    async def _generate_sample(
        self, 
        character: Dict[str, Any],
        narrative_type: str,
        sample_index: int
    ) -> Optional[MultimodalSample]:
        """Generate a single multimodal sample"""
        
        # Generate narrative text with control tokens
        text, control_tokens, narrative_context = await self.text_generator.generate_narrative_turn(
            character=character,
            narrative_type=narrative_type
        )
        
        # Tokenize text (mock tokenization for demo)
        tokens = self._tokenize_text(text)
        
        # Character-aware speech synthesis
        audio, sr, voice_metadata = await self.character_voice_synthesizer.synthesize_character_speech(
            text=text,
            character=character,
            narrative_context={
                "type": narrative_type,
                "tension": random.uniform(0.3, 0.9),
                "scene_type": narrative_type
            }
        )
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_processor.audio_to_mel(audio, sr)
        discrete_mel, codebook = self.mel_processor.quantize_mel(mel_spec)
        
        # Create text-mel alignment
        speech_duration = len(audio) / sr
        alignment = self.mel_processor.align_text_to_mel(
            text=text,
            tokens=tokens,
            mel_frames=mel_spec.shape[0],
            speech_duration=speech_duration
        )
        
        # Generate control token encoding
        control_encoding = self._encode_control_tokens(control_tokens)
        
        # Generate memory vector if appropriate
        if random.random() < self.config.memory_generation_probability:
            memory_vec, importance, surprise, valence, persistence = \
                self.memory_generator.generate_memory_vector(text, character, narrative_context)
        else:
            # Default memory values
            memory_vec = np.zeros(self.config.memory_embedding_dim, dtype=np.float32)
            importance = surprise = valence = persistence = 0.0
        
        # Create sample
        sample = MultimodalSample(
            text=text,
            tokens=tokens,
            mel_frames=mel_spec,
            discrete_mel=discrete_mel,
            control_tokens=control_encoding,
            control_sequence=control_tokens,
            memory_vector=np.concatenate([memory_vec, [importance, surprise, valence, persistence]]),
            memory_importance=importance,
            memory_surprise=surprise,
            memory_valence=valence,
            memory_persistence=persistence,
            text_to_mel_alignment=alignment,
            character_id=character["id"],
            narrative_context=narrative_context,
            session_id=f"synthetic_{sample_index}",
            turn_index=0
        )
        sample.voice_metadata = voice_metadata
        
        # Optionally save audio
        if self.config.save_audio:
            audio_path = self.config.output_dir / f"audio/{sample_index}.wav"
            audio_path.parent.mkdir(exist_ok=True)
            sf.write(str(audio_path), audio, sr)
        
        return sample
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Mock tokenization - replace with actual tokenizer"""
        # Simple character-level tokenization for demo
        return [ord(c) for c in text]
    
    def _encode_control_tokens(self, control_tokens: List[str]) -> np.ndarray:
        """Encode control tokens as multi-hot vector"""
        encoding = np.zeros(len(self.control_token_to_id), dtype=np.float32)
        
        for token in control_tokens:
            if token in self.control_token_to_id:
                encoding[self.control_token_to_id[token]] = 1.0
        
        return encoding
    
    def _save_dataset(self, samples: List[MultimodalSample]):
        """Save dataset to disk"""
        # Save samples in chunks for efficient loading
        chunk_size = 1000
        
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            chunk_data = []
            
            for sample in chunk:
                # Convert to serializable format
                sample_dict = {
                    "text": sample.text,
                    "tokens": sample.tokens,
                    "mel_frames": sample.mel_frames.tolist(),
                    "discrete_mel": sample.discrete_mel.tolist(),
                    "control_tokens": sample.control_tokens.tolist(),
                    "control_sequence": sample.control_sequence,
                    "memory_vector": sample.memory_vector.tolist(),
                    "text_to_mel_alignment": sample.text_to_mel_alignment,
                    "character_id": sample.character_id,
                    "narrative_context": sample.narrative_context,
                    "session_id": sample.session_id,
                    "turn_index": sample.turn_index,
                    "voice_metadata": sample.voice_metadata
                }
                chunk_data.append(sample_dict)
            
            # Save chunk
            chunk_path = self.config.output_dir / f"chunk_{i // chunk_size}.json"
            with open(chunk_path, 'w') as f:
                json.dump(chunk_data, f)
        
        # Save metadata
        metadata = {
            "num_samples": len(samples),
            "config": {
                "narrative_types": self.config.narrative_types,
                "num_characters": self.config.num_characters,
                "sample_rate": self.config.sample_rate,
                "n_mels": self.config.n_mels,
                "quantization_bits": self.config.quantization_bits
            },
            "generation_date": datetime.now().isoformat()
        }
        
        metadata_path = self.config.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {self.config.output_dir}")


# CLI interface
async def main():
    """Generate synthetic multimodal dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic multimodal dataset for NarrativeLM")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--num-characters", type=int, default=50, help="Number of unique characters")
    parser.add_argument("--output-dir", type=str, default="synthetic_multimodal_dataset", help="Output directory")
    parser.add_argument("--save-audio", action="store_true", help="Save audio files")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SyntheticGenerationConfig(
        num_samples=args.num_samples,
        num_characters=args.num_characters,
        output_dir=Path(args.output_dir),
        save_audio=args.save_audio
    )
    
    # Generate dataset
    generator = MultimodalDatasetGenerator(config)
    await generator.generate_dataset()


if __name__ == "__main__":
    asyncio.run(main()) 