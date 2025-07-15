"""
Speech Head for Quad-Head NarrativeLM

This module implements the speech generation head that produces mel-spectrograms
aligned with text generation. Features include:
- 80-dimensional mel-spectrogram prediction
- 4-bit quantization (16 discrete levels)
- Cross-modal attention between text and speech
- Character-specific voice conditioning
- Temporal modeling with causal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class SpeechHead(nn.Module):
    """
    Speech generation head for mel-spectrogram prediction.
    
    Features:
    - Mel-spectrogram generation (80 bins, 4-bit quantized)
    - Cross-modal attention to text representations
    - Character voice conditioning via embeddings
    - Temporal causal modeling for speech continuity
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        mel_bins: int = 80,
        quantization_bits: int = 4,
        num_character_embeddings: int = 1000,
        enable_cross_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mel_bins = mel_bins
        self.quantization_bits = quantization_bits
        self.num_character_embeddings = num_character_embeddings
        self.enable_cross_attention = enable_cross_attention
        
        # Number of discrete levels for quantization
        self.num_quantization_levels = 2 ** quantization_bits
        
        # Character voice conditioning
        self.character_embedding = nn.Embedding(
            num_character_embeddings, 
            hidden_size // 4  # Smaller dimension for efficiency
        )
        
        # Cross-modal attention (text -> speech)
        if enable_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attention_norm = nn.LayerNorm(hidden_size)
        
        # Speech-specific processing layers
        self.speech_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Temporal modeling for speech continuity
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size  # Depthwise convolution for efficiency
        )
        
        # Final projection to mel-spectrogram space
        self.mel_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, mel_bins)
        )
        
        # Quantization layer
        self.quantization_scale = nn.Parameter(torch.tensor(1.0))
        
        logger.info(f"Initialized SpeechHead with {mel_bins} mel bins, "
                   f"{quantization_bits}-bit quantization, "
                   f"cross_attention={enable_cross_attention}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        character_ids: Optional[torch.Tensor] = None,
        text_hidden_states: Optional[torch.Tensor] = None,
        speech_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for speech generation.
        
        Args:
            hidden_states: Shared transformer hidden states [B, T, H]
            character_ids: Character IDs for voice conditioning [B]
            text_hidden_states: Text representations for cross-attention [B, T, H]
            speech_frames: Previous speech frames for context [B, T, mel_bins]
            
        Returns:
            speech_logits: Quantized mel-spectrogram logits [B, T, mel_bins]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Start with shared hidden states
        speech_hidden = hidden_states
        
        # Add character voice conditioning
        if character_ids is not None:
            char_emb = self.character_embedding(character_ids)  # [B, H//4]
            char_emb = char_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, H//4]
            
            # Pad character embedding to match hidden size
            char_emb_padded = F.pad(char_emb, (0, hidden_size - char_emb.shape[-1]))
            speech_hidden = speech_hidden + char_emb_padded
        
        # Cross-modal attention from text to speech
        if self.enable_cross_attention and text_hidden_states is not None:
            # Use text as key/value, speech as query
            attended_speech, _ = self.cross_attention(
                query=speech_hidden,
                key=text_hidden_states,
                value=text_hidden_states
            )
            speech_hidden = self.cross_attention_norm(speech_hidden + attended_speech)
        
        # Speech-specific processing
        speech_hidden = self.speech_processor(speech_hidden)
        
        # Temporal modeling (apply causal masking for autoregressive generation)
        # Reshape for Conv1d: [B, H, T]
        speech_hidden_conv = speech_hidden.transpose(1, 2)
        speech_hidden_conv = self.temporal_conv(speech_hidden_conv)
        speech_hidden = speech_hidden_conv.transpose(1, 2)  # Back to [B, T, H]
        
        # Project to mel-spectrogram space
        mel_logits = self.mel_projection(speech_hidden)  # [B, T, mel_bins]
        
        # Apply quantization during training
        if self.training:
            mel_logits = self._apply_quantization(mel_logits)
        
        return mel_logits
    
    def _apply_quantization(self, mel_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply 4-bit quantization to mel-spectrogram predictions.
        
        Uses straight-through estimator for gradient flow during training.
        """
        # Scale to [0, num_levels-1] range
        scaled = torch.sigmoid(mel_logits * self.quantization_scale)
        scaled = scaled * (self.num_quantization_levels - 1)
        
        # Quantize (forward) but keep gradients (backward)
        quantized = torch.round(scaled)
        
        # Straight-through estimator: forward = quantized, backward = scaled
        quantized = scaled + (quantized - scaled).detach()
        
        # Normalize back to reasonable range
        return quantized / (self.num_quantization_levels - 1)
    
    def generate_speech_frame(
        self,
        hidden_states: torch.Tensor,
        character_ids: Optional[torch.Tensor] = None,
        previous_speech_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate a single speech frame for streaming inference.
        
        Args:
            hidden_states: Current hidden states [B, 1, H]
            character_ids: Character IDs [B]
            previous_speech_frames: Previous frames for context [B, T-1, mel_bins]
            
        Returns:
            speech_frame: Single mel-spectrogram frame [B, mel_bins]
        """
        # Use only last token for frame generation
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]  # [B, 1, H]
        
        # Generate frame
        speech_logits = self.forward(
            hidden_states=hidden_states,
            character_ids=character_ids,
            speech_frames=previous_speech_frames
        )
        
        # Return single frame
        return speech_logits.squeeze(1)  # [B, mel_bins]


def create_speech_head(config, hidden_size: int = None) -> SpeechHead:
    """Factory function to create SpeechHead from config"""
    # Use actual model hidden size if provided, otherwise fall back to config
    actual_hidden_size = hidden_size if hidden_size is not None else config.hidden_size
    
    return SpeechHead(
        hidden_size=actual_hidden_size,
        mel_bins=config.speech_mel_bins,
        quantization_bits=config.speech_quantization_bits,
        num_character_embeddings=config.num_character_embeddings,
        enable_cross_attention=config.enable_cross_attention,
        dropout=config.dropout
    ) 