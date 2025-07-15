"""
Quad-Head NarrativeLM Model

Extends the existing triple-head NarrativeLM architecture with integrated speech generation:
- Generation Head: Text token prediction (existing)
- Control Head: Emotional/cognitive control tokens (existing)  
- Memory Head: External memory vectors + metadata (existing)
- Speech Head: Mel-spectrogram generation for native speech synthesis (NEW)

This enables real-time multimodal character interaction with text and speech
generated simultaneously from a unified model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import logging

from .model import NarrativeLLM
from .speech_head import SpeechHead, create_speech_head
from .config import NarrativeLLMConfig

logger = logging.getLogger(__name__)


class QuadHeadNarrativeLM(NarrativeLLM):
    """
    Quad-head NarrativeLM with integrated speech generation.
    
    Extends the existing triple-head architecture with a fourth head for
    native mel-spectrogram generation, enabling real-time multimodal
    character interaction.
    
    Architecture:
    - Shared transformer backbone (inherited)
    - Generation head: Text prediction (inherited)
    - Control head: Emotional tokens (inherited)
    - Memory head: Memory vectors (inherited)
    - Speech head: Mel-spectrogram generation (NEW)
    """
    
    def __init__(self, config: NarrativeLLMConfig, control_tokens_path: Optional[str] = None):
        # Initialize the base triple-head model
        super().__init__(config, control_tokens_path)
        
        # Add speech head if enabled
        if config.enable_speech_head:
            # Pass the actual model hidden size (from base_model) instead of config
            self.speech_head = create_speech_head(config, hidden_size=self.hidden_size)
            logger.info(f"Added speech head to quad-head architecture with hidden_size={self.hidden_size}")
        else:
            self.speech_head = None
            logger.info("Speech head disabled - using triple-head architecture")
        
        # Update model description
        self.__doc__ = """
        Quad-head NarrativeLLM with:
        - Generation head for text
        - Control head for emotions
        - Memory head for memories
        - Speech head for voice synthesis
        """
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        session_id: Optional[torch.Tensor] = None,
        external_memory_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        control_labels: Optional[torch.Tensor] = None,
        memory_labels: Optional[torch.Tensor] = None,
        speech_labels: Optional[torch.Tensor] = None,  # NEW: Speech targets
        character_ids: Optional[torch.Tensor] = None,  # NEW: Character conditioning
        speech_frames: Optional[torch.Tensor] = None,  # NEW: Previous speech context
        recirculation_tokens: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with quad-head outputs.
        
        Args:
            input_ids: Token IDs for input text
            attention_mask: Attention mask
            session_id: Session identifier for stateful context
            external_memory_states: External memory for cross-attention
            labels: Target tokens for generation head
            control_labels: Target control tokens (multi-hot encoded)
            memory_labels: Target memory vectors for memory head
            speech_labels: Target mel-spectrograms for speech head (NEW)
            character_ids: Character IDs for voice conditioning (NEW)
            speech_frames: Previous speech frames for context (NEW)
            recirculation_tokens: Control tokens from previous turn
            
        Returns:
            Dictionary with all four head outputs and losses
        """
        
        # Get triple-head outputs from parent class
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            session_id=session_id,
            external_memory_states=external_memory_states,
            labels=labels,
            control_labels=control_labels,
            memory_labels=memory_labels,
            recirculation_tokens=recirculation_tokens,
            **kwargs
        )
        
        # Add speech head output if enabled
        if self.speech_head is not None:
            # Get hidden states from the forward pass
            hidden_states = outputs['hidden_states']  # [B, T, H]
            
            # Generate speech logits
            speech_logits = self.speech_head(
                hidden_states=hidden_states,
                character_ids=character_ids,
                text_hidden_states=hidden_states,  # Use same hidden states for cross-attention
                speech_frames=speech_frames
            )
            
            # Add to outputs
            outputs['speech_logits'] = speech_logits
            
            # Calculate speech loss if targets provided
            if speech_labels is not None:
                speech_loss = self._calculate_speech_loss(speech_logits, speech_labels)
                outputs['losses']['speech_loss'] = speech_loss
                
                # Update total loss
                current_total = outputs['losses'].get('total_loss', 0)
                speech_weight = getattr(self.config, 'speech_loss_weight', 0.5)
                outputs['losses']['total_loss'] = current_total + (speech_weight * speech_loss)
        
        return outputs
    
    def _calculate_speech_loss(
        self, 
        speech_logits: torch.Tensor, 
        speech_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate speech generation loss.
        
        Uses MSE loss for mel-spectrogram regression with optional
        spectral loss for better perceptual quality.
        """
        # Basic MSE loss for mel-spectrogram prediction
        mse_loss = F.mse_loss(speech_logits, speech_labels)
        
        # Optional: Add spectral loss for better perceptual quality
        # This encourages similar frequency characteristics
        if hasattr(self.config, 'use_spectral_loss') and self.config.use_spectral_loss:
            # Simple spectral loss - encourage similar frequency distributions
            speech_fft = torch.fft.fft(speech_logits, dim=-1).abs()
            target_fft = torch.fft.fft(speech_labels, dim=-1).abs()
            spectral_loss = F.mse_loss(speech_fft, target_fft)
            
            return mse_loss + 0.1 * spectral_loss
        
        return mse_loss
    
    def generate_with_control(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        user_input: str = "",
        previous_context: str = "",
        character_ids: Optional[torch.Tensor] = None,  # NEW: Character conditioning
        generate_speech: bool = False,  # NEW: Whether to generate speech
        recirculation_tokens: Optional[List[str]] = None,
        generate_memory: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with optional speech synthesis.
        
        Extends the parent's generate_with_control method to include
        simultaneous speech generation when requested.
        """
        
        # Get text generation from parent
        generation_result = super().generate_with_control(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            user_input=user_input,
            previous_context=previous_context,
            recirculation_tokens=recirculation_tokens,
            generate_memory=generate_memory,
            **kwargs
        )
        
        # Add speech generation if requested and available
        if generate_speech and self.speech_head is not None:
            speech_frames = self._generate_speech_for_text(
                input_ids=generation_result['generated_ids'],
                character_ids=character_ids,
                temperature=temperature
            )
            generation_result['speech_frames'] = speech_frames
        
        return generation_result
    
    def _generate_speech_for_text(
        self,
        input_ids: torch.Tensor,
        character_ids: Optional[torch.Tensor] = None,
        temperature: float = 0.7
    ) -> torch.Tensor:
        """
        Generate speech frames for the given text tokens.
        
        This method aligns speech generation with text generation,
        producing mel-spectrogram frames that correspond to the text.
        """
        with torch.no_grad():
            # Get hidden states for the text
            outputs = self.base_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Generate speech frames
            speech_logits = self.speech_head(
                hidden_states=hidden_states,
                character_ids=character_ids,
                text_hidden_states=hidden_states
            )
            
            # Apply temperature sampling for speech generation
            if temperature > 0:
                speech_logits = speech_logits / temperature
                speech_frames = torch.sigmoid(speech_logits)  # Convert to probabilities
            else:
                speech_frames = torch.sigmoid(speech_logits)
            
            return speech_frames
    
    def generate_speech_frame(
        self,
        input_ids: torch.Tensor,
        character_ids: Optional[torch.Tensor] = None,
        previous_speech_frames: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate a single speech frame for streaming inference.
        
        This method enables real-time speech generation by producing
        one mel-spectrogram frame at a time.
        """
        if self.speech_head is None:
            raise ValueError("Speech head not enabled - cannot generate speech frames")
        
        with torch.no_grad():
            # Get hidden states for current input
            outputs = self.base_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Generate single frame
            speech_frame = self.speech_head.generate_speech_frame(
                hidden_states=hidden_states,
                character_ids=character_ids,
                previous_speech_frames=previous_speech_frames
            )
            
            return speech_frame
    
    def get_speech_head_parameters(self) -> int:
        """Get number of parameters in speech head"""
        if self.speech_head is None:
            return 0
        return sum(p.numel() for p in self.speech_head.parameters())
    
    def get_total_parameters(self) -> Dict[str, int]:
        """Get parameter count breakdown for all heads"""
        param_counts = {
            'base_model': sum(p.numel() for p in self.base_model.parameters()),
            'control_head': sum(p.numel() for p in self.control_head.parameters()),
            'memory_head': sum(p.numel() for p in self.memory_head.parameters()),
            'speech_head': self.get_speech_head_parameters()
        }
        param_counts['total'] = sum(param_counts.values())
        return param_counts


def create_quad_head_model(
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    control_tokens_path: Optional[str] = None,
    enable_speech_head: bool = True,
    **config_kwargs
) -> QuadHeadNarrativeLM:
    """
    Factory function to create QuadHeadNarrativeLM with speech capabilities.
    
    Args:
        base_model_name: Base transformer model to use
        control_tokens_path: Path to control tokens configuration
        enable_speech_head: Whether to enable speech generation
        **config_kwargs: Additional configuration parameters
        
    Returns:
        QuadHeadNarrativeLM model instance
    """
    config = NarrativeLLMConfig(
        base_model_name=base_model_name,
        enable_speech_head=enable_speech_head,
        **config_kwargs
    )
    
    model = QuadHeadNarrativeLM(config, control_tokens_path)
    
    logger.info(f"Created QuadHeadNarrativeLM with {model.get_total_parameters()['total']:,} parameters")
    
    return model 