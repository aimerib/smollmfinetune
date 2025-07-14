"""
Multimodal Dataset Pipeline for Quad-Head Training

This module provides comprehensive data preprocessing for text-speech aligned training:
- Text-speech temporal alignment using forced alignment
- Mel-spectrogram preprocessing and quantization
- Character voice conditioning data preparation
- Cross-modal attention data structures
- Batch processing for efficient training
"""

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import librosa
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class TextSpeechAlignment:
    """Represents alignment between text tokens and speech frames"""
    text_tokens: List[str]
    token_start_times: List[float]  # Start time for each token in seconds
    token_end_times: List[float]    # End time for each token in seconds
    mel_frames: np.ndarray          # Shape: (time_frames, mel_bins)
    frame_times: List[float]        # Time for each mel frame
    character_id: str
    total_duration: float
    
    def get_token_frame_alignment(self) -> List[Tuple[int, List[int]]]:
        """Get mapping from token index to corresponding mel frame indices"""
        alignments = []
        for i, (start_time, end_time) in enumerate(zip(self.token_start_times, self.token_end_times)):
            # Find mel frames that overlap with this token's time span
            frame_indices = []
            for j, frame_time in enumerate(self.frame_times):
                if start_time <= frame_time < end_time:
                    frame_indices.append(j)
            alignments.append((i, frame_indices))
        return alignments


@dataclass
class MultimodalTrainingSample:
    """Single training sample for quad-head model"""
    # Text data
    input_ids: torch.Tensor           # Input text tokens
    attention_mask: torch.Tensor      # Text attention mask
    text_labels: torch.Tensor         # Text generation targets
    
    # Speech data  
    mel_frames: torch.Tensor          # Mel-spectrogram frames (time, mel_bins)
    speech_labels: torch.Tensor       # Quantized mel targets for speech head
    speech_attention_mask: torch.Tensor  # Speech frame attention mask
    
    # Cross-modal alignment
    text_to_speech_alignment: torch.Tensor  # Alignment matrix (text_len, speech_len)
    
    # Character conditioning
    character_id: str
    character_embedding_idx: int
    
    # Control and memory (from existing pipeline)
    control_labels: Optional[torch.Tensor] = None
    memory_labels: Optional[torch.Tensor] = None
    
    # Metadata
    sample_id: str = ""
    duration: float = 0.0


class CharacterVoiceRegistry:
    """Registry for character voice profiles and conditioning"""
    
    def __init__(self):
        self.character_voices: Dict[str, Dict[str, Any]] = {}
        self.character_to_idx: Dict[str, int] = {}
        self.idx_to_character: Dict[int, str] = {}
        self._next_idx = 0
    
    def register_character(self, character_id: str, voice_profile: Dict[str, Any]) -> int:
        """Register a character voice profile and return embedding index"""
        if character_id not in self.character_to_idx:
            self.character_to_idx[character_id] = self._next_idx
            self.idx_to_character[self._next_idx] = character_id
            self._next_idx += 1
        
        self.character_voices[character_id] = voice_profile
        return self.character_to_idx[character_id]
    
    def get_character_idx(self, character_id: str) -> Optional[int]:
        """Get embedding index for character"""
        return self.character_to_idx.get(character_id)
    
    def get_voice_profile(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get voice profile for character"""
        return self.character_voices.get(character_id)


class SpeechPreprocessor:
    """Handles mel-spectrogram preprocessing and quantization"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_mels: int = 80,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_fft: int = 1024,
                 quantization_bits: int = 4):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram from audio"""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
        
        return normalized.T  # Shape: (time_frames, mel_bins)
    
    def quantize_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Quantize mel-spectrogram to discrete levels"""
        # Scale to quantization levels
        quantized = np.round(mel_spec * (self.quantization_levels - 1))
        return quantized.astype(np.int32)
    
    def get_frame_times(self, audio_length: int) -> List[float]:
        """Get time stamps for each mel frame"""
        n_frames = (audio_length - self.win_length) // self.hop_length + 1
        frame_times = [(i * self.hop_length) / self.sample_rate for i in range(n_frames)]
        return frame_times


class ForcedAligner:
    """Handles forced alignment between text and speech"""
    
    def __init__(self):
        # In a real implementation, this would use a proper forced alignment model
        # like Montreal Forced Alignment (MFA) or similar
        self.alignment_model = None
    
    async def align_text_speech(self, 
                               text: str, 
                               audio: np.ndarray, 
                               sample_rate: int) -> TextSpeechAlignment:
        """
        Perform forced alignment between text and speech
        
        For now, implements a simple heuristic alignment.
        In production, this would use a proper phoneme-based aligner.
        """
        # Simple tokenization
        tokens = text.split()
        
        # Simple heuristic: distribute tokens evenly across audio duration
        audio_duration = len(audio) / sample_rate
        token_duration = audio_duration / len(tokens)
        
        token_start_times = [i * token_duration for i in range(len(tokens))]
        token_end_times = [(i + 1) * token_duration for i in range(len(tokens))]
        
        # Extract mel-spectrogram
        preprocessor = SpeechPreprocessor(sample_rate=sample_rate)
        mel_frames = preprocessor.extract_mel_spectrogram(audio)
        frame_times = preprocessor.get_frame_times(len(audio))
        
        return TextSpeechAlignment(
            text_tokens=tokens,
            token_start_times=token_start_times,
            token_end_times=token_end_times,
            mel_frames=mel_frames,
            frame_times=frame_times,
            character_id="",  # Will be set by caller
            total_duration=audio_duration
        )


class MultimodalDatasetPipeline:
    """Main pipeline for processing multimodal training data"""
    
    def __init__(self, 
                 tokenizer=None,
                 max_text_length: int = 512,
                 max_speech_length: int = 1000,
                 character_registry: Optional[CharacterVoiceRegistry] = None):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_speech_length = max_speech_length
        self.character_registry = character_registry or CharacterVoiceRegistry()
        self.forced_aligner = ForcedAligner()
        self.speech_preprocessor = SpeechPreprocessor()
        
    async def process_text_speech_pair(self,
                                     text: str,
                                     audio_path: str,
                                     character_id: str,
                                     voice_profile: Optional[Dict[str, Any]] = None) -> MultimodalTrainingSample:
        """Process a single text-speech pair into training sample"""
        
        # Register character if needed
        if voice_profile:
            character_idx = self.character_registry.register_character(character_id, voice_profile)
        else:
            character_idx = self.character_registry.get_character_idx(character_id)
            if character_idx is None:
                character_idx = self.character_registry.register_character(character_id, {})
        
        # Load audio
        audio, sample_rate = librosa.load(audio_path, sr=self.speech_preprocessor.sample_rate)
        
        # Perform forced alignment
        alignment = await self.forced_aligner.align_text_speech(text, audio, sample_rate)
        alignment.character_id = character_id
        
        # Tokenize text
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            # Simple word-based tokenization for testing
            tokens = text.split()[:self.max_text_length]
            input_ids = torch.tensor([hash(token) % 50000 for token in tokens])
            attention_mask = torch.ones(len(tokens))
            
            # Pad to max length
            pad_length = self.max_text_length - len(tokens)
            if pad_length > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
        
        # Process speech data
        mel_frames = torch.from_numpy(alignment.mel_frames).float()
        
        # Truncate or pad speech to max length
        if mel_frames.shape[0] > self.max_speech_length:
            mel_frames = mel_frames[:self.max_speech_length]
        else:
            pad_length = self.max_speech_length - mel_frames.shape[0]
            mel_frames = torch.cat([mel_frames, torch.zeros(pad_length, mel_frames.shape[1])])
        
        # Quantize for speech labels
        quantized_mel = self.speech_preprocessor.quantize_mel_spectrogram(mel_frames.numpy())
        speech_labels = torch.from_numpy(quantized_mel).long()
        
        # Create speech attention mask
        actual_frames = min(len(alignment.mel_frames), self.max_speech_length)
        speech_attention_mask = torch.cat([
            torch.ones(actual_frames),
            torch.zeros(self.max_speech_length - actual_frames)
        ])
        
        # Create cross-modal alignment matrix
        alignment_matrix = self._create_alignment_matrix(
            alignment.get_token_frame_alignment(),
            input_ids.shape[0],
            mel_frames.shape[0]
        )
        
        return MultimodalTrainingSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_labels=input_ids.clone(),  # For autoregressive text generation
            mel_frames=mel_frames,
            speech_labels=speech_labels,
            speech_attention_mask=speech_attention_mask,
            text_to_speech_alignment=alignment_matrix,
            character_id=character_id,
            character_embedding_idx=character_idx,
            sample_id=f"{character_id}_{hash(text) % 100000}",
            duration=alignment.total_duration
        )
    
    def _create_alignment_matrix(self, 
                               token_frame_alignment: List[Tuple[int, List[int]]],
                               text_length: int,
                               speech_length: int) -> torch.Tensor:
        """Create cross-modal alignment matrix"""
        alignment_matrix = torch.zeros(text_length, speech_length)
        
        for token_idx, frame_indices in token_frame_alignment:
            if token_idx < text_length:
                for frame_idx in frame_indices:
                    if frame_idx < speech_length:
                        alignment_matrix[token_idx, frame_idx] = 1.0
        
        # Normalize each row to sum to 1 (soft alignment)
        row_sums = alignment_matrix.sum(dim=1, keepdim=True)
        
        # For rows with no alignment, create uniform distribution
        zero_rows = (row_sums == 0).squeeze()
        alignment_matrix[zero_rows] = 1.0 / speech_length
        
        # Normalize non-zero rows
        non_zero_rows = ~zero_rows
        if non_zero_rows.any():
            alignment_matrix[non_zero_rows] = alignment_matrix[non_zero_rows] / row_sums[non_zero_rows]
        
        return alignment_matrix
    
    async def process_dataset(self, 
                            dataset_config: Dict[str, Any],
                            output_path: str,
                            batch_size: int = 32) -> Dict[str, Any]:
        """Process entire dataset for multimodal training"""
        
        samples = []
        processing_stats = {
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "total_duration": 0.0,
            "characters": set(),
            "avg_text_length": 0.0,
            "avg_speech_length": 0.0
        }
        
        # Process samples from dataset config
        for item in dataset_config.get("samples", []):
            try:
                sample = await self.process_text_speech_pair(
                    text=item["text"],
                    audio_path=item["audio_path"],
                    character_id=item["character_id"],
                    voice_profile=item.get("voice_profile")
                )
                samples.append(sample)
                
                # Update stats
                processing_stats["successful_samples"] += 1
                processing_stats["total_duration"] += sample.duration
                processing_stats["characters"].add(sample.character_id)
                processing_stats["avg_text_length"] += sample.input_ids.shape[0]
                processing_stats["avg_speech_length"] += sample.mel_frames.shape[0]
                
            except Exception as e:
                logger.error(f"Failed to process sample {item}: {e}")
                processing_stats["failed_samples"] += 1
        
        processing_stats["total_samples"] = len(samples)
        processing_stats["characters"] = list(processing_stats["characters"])
        
        if processing_stats["successful_samples"] > 0:
            processing_stats["avg_text_length"] /= processing_stats["successful_samples"]
            processing_stats["avg_speech_length"] /= processing_stats["successful_samples"]
        
        # Save processed dataset
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save samples in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_file = output_path / f"batch_{i // batch_size:04d}.pt"
            torch.save(batch, batch_file)
        
        # Save metadata
        metadata = {
            "processing_stats": processing_stats,
            "character_registry": {
                "character_to_idx": self.character_registry.character_to_idx,
                "character_voices": self.character_registry.character_voices
            },
            "pipeline_config": {
                "max_text_length": self.max_text_length,
                "max_speech_length": self.max_speech_length,
                "speech_config": {
                    "sample_rate": self.speech_preprocessor.sample_rate,
                    "n_mels": self.speech_preprocessor.n_mels,
                    "quantization_bits": self.speech_preprocessor.quantization_bits
                }
            }
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Processed {processing_stats['successful_samples']} samples")
        logger.info(f"Total duration: {processing_stats['total_duration']:.2f} seconds")
        logger.info(f"Characters: {processing_stats['characters']}")
        
        return processing_stats


def create_multimodal_pipeline(tokenizer=None, **kwargs) -> MultimodalDatasetPipeline:
    """Factory function to create multimodal dataset pipeline"""
    return MultimodalDatasetPipeline(tokenizer=tokenizer, **kwargs)


# Example usage and testing functions
async def create_sample_dataset_config() -> Dict[str, Any]:
    """Create a sample dataset configuration for testing"""
    return {
        "samples": [
            {
                "text": "Hello, how are you today?",
                "audio_path": "/path/to/audio1.wav",
                "character_id": "alice",
                "voice_profile": {
                    "pitch_mean": 220.0,
                    "pitch_range": 50.0,
                    "voice_style": "friendly"
                }
            },
            {
                "text": "I'm doing great, thanks for asking!",
                "audio_path": "/path/to/audio2.wav", 
                "character_id": "bob",
                "voice_profile": {
                    "pitch_mean": 150.0,
                    "pitch_range": 30.0,
                    "voice_style": "casual"
                }
            }
        ]
    }


if __name__ == "__main__":
    # Example usage
    pipeline = create_multimodal_pipeline()
    print("Multimodal dataset pipeline created successfully!") 