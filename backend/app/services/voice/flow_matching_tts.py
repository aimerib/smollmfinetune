"""
Flow-Matching Text-to-Speech Architecture for Narrative Generation

This module implements a sophisticated flow-matching based TTS system specifically
optimized for narrative content and character-conditioned voice synthesis.

Key Components:
- NarrativeTextEncoder: Transformer-based text encoding with narrative context awareness
- ContinuousFlowMatcher: Flow matching for mel-spectrogram generation  
- NarrativeAwareAttention: Story-context optimized attention mechanisms
- CharacterConditioner: Character-specific voice trait conditioning
- SpeakerEmbeddingExtractor: Zero-shot voice cloning capabilities
- FlowMatchingTrainer: Training infrastructure with curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow-matching TTS model"""
    # Model architecture
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_mel_bins: int = 80
    max_sequence_length: int = 2048
    
    # Flow matching parameters
    noise_schedule: str = "cosine"  # cosine, linear, sigmoid
    num_inference_steps: int = 50
    min_noise_level: float = 1e-4
    max_noise_level: float = 1.0
    
    # Character conditioning
    num_character_classes: int = 100
    character_embedding_dim: int = 256
    personality_trait_dim: int = 5  # Big Five traits
    
    # Speaker embedding
    speaker_embedding_dim: int = 512
    enable_zero_shot: bool = True
    
    # Training parameters
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class NarrativeTextEncoder(nn.Module):
    """
    Transformer-based text encoder optimized for narrative content.
    
    Features:
    - Narrative context awareness through specialized attention
    - Story progression understanding
    - Character relationship modeling
    """
    
    def __init__(self, config: FlowMatchingConfig, vocab_size: int = 32000):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Text embedding layers
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_dim)
        
        # Narrative context embedding
        self.narrative_context_embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Transformer layers with narrative-aware attention
        self.transformer_layers = nn.ModuleList([
            NarrativeAwareAttention(config) for _ in range(config.num_layers)
        ])
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Output projection for mel-spectrogram conditioning
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        narrative_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text tokens with narrative context awareness.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            narrative_context: Story context embeddings [batch_size, context_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Text representations [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embedding layers
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Add narrative context if provided
        if narrative_context is not None:
            context_embeds = self.narrative_context_embedding(narrative_context)
            # Broadcast context across sequence length
            context_embeds = context_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            hidden_states = hidden_states + context_embeds
            
        hidden_states = self.dropout(hidden_states)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            
        # Final normalization and projection
        hidden_states = self.layer_norm(hidden_states)
        return self.output_projection(hidden_states)


class NarrativeAwareAttention(nn.Module):
    """
    Attention mechanism optimized for narrative content understanding.
    
    Features:
    - Story progression awareness
    - Character relationship modeling
    - Temporal consistency across narrative segments
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        
        # Ensure hidden_dim is divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            # Dynamically adjust num_heads to be compatible
            self.num_heads = max(1, self.hidden_dim // 64)  # Use 64 as a reasonable head dimension
            if self.hidden_dim % self.num_heads != 0:
                # If still not divisible, round down to nearest factor
                for i in range(self.num_heads, 0, -1):
                    if self.hidden_dim % i == 0:
                        self.num_heads = i
                        break
        
        self.head_dim = self.hidden_dim // self.num_heads
        
        # Multi-head attention components
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Narrative-specific attention weights
        self.narrative_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.story_progression_weight = nn.Parameter(torch.ones(1))
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply narrative-aware attention.
        
        Args:
            hidden_states: Input representations [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Updated representations [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Apply multi-head attention
        attn_output, _ = self.attention(
            query=hidden_states,
            key=hidden_states, 
            value=hidden_states,
            key_padding_mask=attention_mask
        )
        
        # Apply narrative gating
        narrative_weights = torch.sigmoid(self.narrative_gate(hidden_states))
        attn_output = attn_output * narrative_weights * self.story_progression_weight
        
        hidden_states = residual + attn_output
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)
        
        return hidden_states


class ContinuousFlowMatcher(nn.Module):
    """
    Flow matching for continuous mel-spectrogram generation.
    
    Uses optimal transport and continuous normalizing flows for
    high-quality speech synthesis with smooth interpolation.
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        self.num_mel_bins = config.num_mel_bins
        self.hidden_dim = config.hidden_dim
        
        # Flow network for mel-spectrogram prediction
        self.flow_network = nn.Sequential(
            nn.Linear(config.num_mel_bins + config.hidden_dim + 1, config.hidden_dim),  # +1 for time
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_mel_bins)
        )
        
        # Noise schedule parameters
        if config.noise_schedule == "cosine":
            self.register_buffer("alphas", self._cosine_schedule(config.num_inference_steps))
        elif config.noise_schedule == "linear":
            self.register_buffer("alphas", self._linear_schedule(config.num_inference_steps))
        else:
            self.register_buffer("alphas", self._sigmoid_schedule(config.num_inference_steps))
            
    def _cosine_schedule(self, timesteps: int) -> torch.Tensor:
        """Create cosine noise schedule"""
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def _linear_schedule(self, timesteps: int) -> torch.Tensor:
        """Create linear noise schedule"""
        beta_start, beta_end = 0.0001, 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
        
    def _sigmoid_schedule(self, timesteps: int) -> torch.Tensor:
        """Create sigmoid noise schedule"""
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (0.02 - 0.0001) + 0.0001
    
    def forward(
        self,
        mel_noisy: torch.Tensor,
        text_embeddings: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise/velocity for flow matching.
        
        Args:
            mel_noisy: Noisy mel-spectrograms [batch_size, time, mel_bins]
            text_embeddings: Text representations [batch_size, time, hidden_dim]
            timestep: Current timestep [batch_size]
            
        Returns:
            Predicted velocity [batch_size, time, mel_bins]
        """
        batch_size, time_steps, _ = mel_noisy.shape
        
        # Normalize timestep
        t_normalized = timestep.float() / self.config.num_inference_steps
        t_embed = t_normalized.unsqueeze(1).unsqueeze(2).expand(batch_size, time_steps, 1)
        
        # Concatenate inputs: mel + text + time
        flow_input = torch.cat([mel_noisy, text_embeddings, t_embed], dim=-1)
        
        # Predict velocity through flow network
        velocity = self.flow_network(flow_input)
        
        return velocity
    
    def sample(
        self,
        text_embeddings: torch.Tensor,
        num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate mel-spectrograms using flow matching sampling.
        
        Args:
            text_embeddings: Text representations [batch_size, time, hidden_dim]
            num_inference_steps: Number of sampling steps
            
        Returns:
            Generated mel-spectrograms [batch_size, time, mel_bins]
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
            
        batch_size, time_steps, _ = text_embeddings.shape
        device = text_embeddings.device
        
        # Start from random noise
        mel_current = torch.randn(
            batch_size, time_steps, self.num_mel_bins, 
            device=device, dtype=text_embeddings.dtype
        )
        
        # Sampling loop
        dt = 1.0 / num_inference_steps
        for step in range(num_inference_steps):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Predict velocity
            velocity = self.forward(mel_current, text_embeddings, t)
            
            # Update using Euler integration
            mel_current = mel_current + velocity * dt
            
        return mel_current


class CharacterConditioner(nn.Module):
    """
    Character-specific voice conditioning system.
    
    Integrates personality traits, character identity, and voice consistency
    for believable character voices across narrative contexts.
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        
        # Character identity embeddings
        self.character_embedding = nn.Embedding(
            config.num_character_classes, config.character_embedding_dim
        )
        
        # Personality trait processing (Big Five)
        self.personality_processor = nn.Sequential(
            nn.Linear(config.personality_trait_dim, config.character_embedding_dim),
            nn.GELU(),
            nn.Linear(config.character_embedding_dim, config.character_embedding_dim)
        )
        
        # Voice characteristic control
        self.voice_controller = nn.Sequential(
            nn.Linear(config.character_embedding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh()  # Bounded output for stable conditioning
        )
        
        # Adaptive layer normalization for character conditioning
        self.adaptive_norm = AdaptiveLayerNorm(config.hidden_dim, config.character_embedding_dim)
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        character_id: torch.Tensor,
        personality_traits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply character conditioning to text embeddings.
        
        Args:
            text_embeddings: Text representations [batch_size, seq_len, hidden_dim]
            character_id: Character IDs [batch_size]
            personality_traits: Big Five traits [batch_size, 5]
            
        Returns:
            Character-conditioned embeddings [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = text_embeddings.shape
        
        # Get character embeddings
        char_embeds = self.character_embedding(character_id)  # [batch_size, char_embed_dim]
        
        # Process personality traits if provided
        if personality_traits is not None:
            personality_embeds = self.personality_processor(personality_traits)
            # Combine character and personality
            combined_embeds = torch.cat([char_embeds, personality_embeds], dim=-1)
        else:
            # Use zero personality if not provided
            zero_personality = torch.zeros_like(char_embeds)
            combined_embeds = torch.cat([char_embeds, zero_personality], dim=-1)
        
        # Generate voice conditioning
        voice_conditioning = self.voice_controller(combined_embeds)  # [batch_size, hidden_dim]
        
        # Apply adaptive normalization with character conditioning
        conditioned_embeddings = self.adaptive_norm(text_embeddings, char_embeds)
        
        # Add voice conditioning
        voice_conditioning = voice_conditioning.unsqueeze(1).expand(-1, seq_len, -1)
        conditioned_embeddings = conditioned_embeddings + voice_conditioning
        
        return conditioned_embeddings


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization conditioned on character embeddings"""
    
    def __init__(self, hidden_dim: int, conditioning_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale_transform = nn.Linear(conditioning_dim, hidden_dim)
        self.shift_transform = nn.Linear(conditioning_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        scale = self.scale_transform(conditioning).unsqueeze(1)
        shift = self.shift_transform(conditioning).unsqueeze(1)
        return normalized * (1 + scale) + shift


class SpeakerEmbeddingExtractor(nn.Module):
    """
    Extract speaker embeddings for zero-shot voice cloning.
    
    Enables the model to adapt to new voices without retraining
    by learning a compact speaker representation space.
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        
        # Speaker encoder network
        self.speaker_encoder = nn.Sequential(
            nn.Conv1d(config.num_mel_bins, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, config.speaker_embedding_dim),
            nn.Tanh()
        )
        
        # Speaker adaptation layers
        self.speaker_adapter = nn.Sequential(
            nn.Linear(config.speaker_embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def extract_speaker_embedding(self, reference_mel: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from reference mel-spectrogram.
        
        Args:
            reference_mel: Reference mel-spectrograms [batch_size, time, mel_bins]
            
        Returns:
            Speaker embeddings [batch_size, speaker_embedding_dim]
        """
        # Transpose for conv1d: [batch_size, mel_bins, time]
        mel_transposed = reference_mel.transpose(1, 2)
        
        # Extract speaker characteristics
        speaker_embedding = self.speaker_encoder(mel_transposed)
        
        return speaker_embedding
    
    def adapt_embeddings(
        self, 
        text_embeddings: torch.Tensor, 
        speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Adapt text embeddings with speaker characteristics.
        
        Args:
            text_embeddings: Text representations [batch_size, seq_len, hidden_dim]
            speaker_embedding: Speaker embeddings [batch_size, speaker_embedding_dim]
            
        Returns:
            Speaker-adapted embeddings [batch_size, seq_len, hidden_dim]
        """
        # Generate speaker conditioning
        speaker_conditioning = self.speaker_adapter(speaker_embedding)
        
        # Apply to text embeddings
        speaker_conditioning = speaker_conditioning.unsqueeze(1)
        adapted_embeddings = text_embeddings + speaker_conditioning
        
        return adapted_embeddings


class NarrativeFlowMatchingTTS(nn.Module):
    """
    Main flow-matching TTS model for narrative generation.
    
    Integrates all components for end-to-end character-conditioned
    speech synthesis optimized for storytelling contexts.
    """
    
    def __init__(self, config: FlowMatchingConfig, vocab_size: int = 32000):
        super().__init__()
        self.config = config
        
        # Core components
        self.text_encoder = NarrativeTextEncoder(config, vocab_size)
        self.flow_matcher = ContinuousFlowMatcher(config)
        self.character_conditioner = CharacterConditioner(config)
        
        # Optional speaker embedding for zero-shot cloning
        if config.enable_zero_shot:
            self.speaker_extractor = SpeakerEmbeddingExtractor(config)
        else:
            self.speaker_extractor = None
            
    def forward(
        self,
        input_ids: torch.Tensor,
        character_id: torch.Tensor,
        target_mel: Optional[torch.Tensor] = None,
        personality_traits: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        narrative_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            character_id: Character IDs [batch_size]
            target_mel: Target mel-spectrograms for training [batch_size, time, mel_bins]
            personality_traits: Big Five traits [batch_size, 5]
            reference_mel: Reference mel for zero-shot cloning [batch_size, ref_time, mel_bins]
            narrative_context: Story context [batch_size, context_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            timestep: Training timestep [batch_size]
            
        Returns:
            Dictionary with model outputs
        """
        # 1. Encode text with narrative awareness
        text_embeddings = self.text_encoder(
            input_ids=input_ids,
            narrative_context=narrative_context,
            attention_mask=attention_mask
        )
        
        # 2. Apply character conditioning
        conditioned_embeddings = self.character_conditioner(
            text_embeddings=text_embeddings,
            character_id=character_id,
            personality_traits=personality_traits
        )
        
        # 3. Apply speaker adaptation if reference provided
        if reference_mel is not None and self.speaker_extractor is not None:
            speaker_embedding = self.speaker_extractor.extract_speaker_embedding(reference_mel)
            conditioned_embeddings = self.speaker_extractor.adapt_embeddings(
                conditioned_embeddings, speaker_embedding
            )
        
        # 4. Generate or predict mel-spectrograms
        outputs = {}
        
        if target_mel is not None and timestep is not None:
            # Training mode: predict velocity
            if target_mel.size(1) != conditioned_embeddings.size(1):
                # Adjust target mel length to match text embeddings
                target_length = conditioned_embeddings.size(1)
                if target_mel.size(1) < target_length:
                    # Pad target mel
                    padding = target_length - target_mel.size(1)
                    target_mel = F.pad(target_mel, (0, 0, 0, padding), mode='constant', value=0)
                else:
                    # Truncate target mel
                    target_mel = target_mel[:, :target_length, :]
                    
            # Add noise for training
            noise = torch.randn_like(target_mel)
            alpha = self.flow_matcher.alphas[timestep].view(-1, 1, 1)
            mel_noisy = target_mel * torch.sqrt(alpha) + noise * torch.sqrt(1 - alpha)
            
            # Predict velocity
            predicted_velocity = self.flow_matcher(mel_noisy, conditioned_embeddings, timestep)
            outputs['predicted_velocity'] = predicted_velocity
            outputs['target_velocity'] = noise  # True velocity is the noise
            outputs['mel_noisy'] = mel_noisy
            
        else:
            # Inference mode: generate mel-spectrograms
            generated_mel = self.flow_matcher.sample(conditioned_embeddings)
            outputs['generated_mel'] = generated_mel
            
        outputs['text_embeddings'] = text_embeddings
        outputs['conditioned_embeddings'] = conditioned_embeddings
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        character_id: torch.Tensor,
        personality_traits: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        narrative_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate mel-spectrograms for given text and character.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            character_id: Character IDs [batch_size]
            personality_traits: Big Five traits [batch_size, 5]
            reference_mel: Reference mel for zero-shot cloning [batch_size, ref_time, mel_bins]
            narrative_context: Story context [batch_size, context_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            num_inference_steps: Number of sampling steps
            
        Returns:
            Generated mel-spectrograms [batch_size, time, mel_bins]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                character_id=character_id,
                personality_traits=personality_traits,
                reference_mel=reference_mel,
                narrative_context=narrative_context,
                attention_mask=attention_mask
            )
            return outputs['generated_mel']


class FlowMatchingTrainer:
    """
    Training infrastructure for flow-matching TTS with curriculum learning
    and advanced optimization strategies.
    """
    
    def __init__(
        self, 
        model: NarrativeFlowMatchingTTS,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000
    ):
        self.model = model
        self.config = model.config
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000)
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Training metrics
        self.training_metrics = {
            'velocity_loss': [],
            'mel_accuracy': [],
            'character_consistency': [],
            'speaker_similarity': []
        }
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single training step.
        
        Args:
            batch: Training batch with required fields
            
        Returns:
            Loss tensor and metrics dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Generate random timesteps for this batch
        batch_size = batch['text_tokens'].size(0)
        timestep = torch.randint(
            0, self.config.num_inference_steps, 
            (batch_size,), device=batch['text_tokens'].device
        )
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['text_tokens'],
            character_id=batch['character_id'],
            target_mel=batch['target_mel'],
            personality_traits=batch.get('personality_traits'),
            reference_mel=batch.get('reference_mel'),
            narrative_context=batch.get('narrative_context'),
            timestep=timestep
        )
        
        # Compute velocity loss (main training objective)
        velocity_loss = F.mse_loss(
            outputs['predicted_velocity'], 
            outputs['target_velocity']
        )
        
        # Additional metrics
        mel_pred = outputs['mel_noisy'] + outputs['predicted_velocity']
        
        # Ensure target_mel matches the prediction shape
        target_mel = batch['target_mel']
        if target_mel.size(1) != mel_pred.size(1):
            if target_mel.size(1) < mel_pred.size(1):
                padding = mel_pred.size(1) - target_mel.size(1)
                target_mel = F.pad(target_mel, (0, 0, 0, padding), mode='constant', value=0)
            else:
                target_mel = target_mel[:, :mel_pred.size(1), :]
        
        mel_accuracy = 1.0 / (1.0 + F.mse_loss(mel_pred, target_mel).item())
        
        # Character consistency metric (simplified)
        character_consistency = torch.cosine_similarity(
            outputs['conditioned_embeddings'].mean(dim=1),
            outputs['text_embeddings'].mean(dim=1)
        ).mean().item()
        
        # Backward pass
        velocity_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step with warmup
        if self.step_count < self.warmup_steps:
            lr_scale = min(1.0, (self.step_count + 1) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * 1e-4
        
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1
        
        # Update metrics
        metrics = {
            'velocity_loss': velocity_loss.item(),
            'mel_accuracy': mel_accuracy,
            'character_consistency': character_consistency,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        for key, value in metrics.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)
        
        return velocity_loss, metrics
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {}
        for key, values in self.training_metrics.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values[-100:])  # Last 100 steps
                stats[f'{key}_std'] = np.std(values[-100:])
                stats[f'{key}_latest'] = values[-1]
        
        stats['total_steps'] = self.step_count
        stats['current_lr'] = self.optimizer.param_groups[0]['lr']
        
        return stats 