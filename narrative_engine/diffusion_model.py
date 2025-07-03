"""
Diffusion Multimodal Model for NarrativeLLM

A comprehensive diffusion-based multimodal model that generates text, speech (mel-spectrograms),
control tokens, and memory vectors simultaneously with cross-modal attention and conditioning.

This model works with the existing multimodal dataset format and integrates with the 
TrainingManager infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .diffusion_config import DiffusionMultimodalConfig
from .config import NarrativeLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class DiffusionOutput:
    """Output structure for diffusion model"""
    # Predicted noise/denoised outputs for each modality
    text_prediction: torch.Tensor
    speech_prediction: torch.Tensor 
    control_prediction: torch.Tensor
    memory_prediction: torch.Tensor
    
    # Loss components
    text_loss: Optional[torch.Tensor] = None
    speech_loss: Optional[torch.Tensor] = None
    control_loss: Optional[torch.Tensor] = None
    memory_loss: Optional[torch.Tensor] = None
    alignment_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    
    # Additional outputs
    cross_modal_attention_weights: Optional[Dict[str, torch.Tensor]] = None
    timestep_embeddings: Optional[torch.Tensor] = None


class NoiseScheduler:
    """DDPM-style noise scheduler with modality-specific scaling"""
    
    def __init__(self, config: DiffusionMultimodalConfig):
        self.config = config
        self.num_train_timesteps = config.scheduler.num_train_timesteps
        self.num_inference_steps = config.scheduler.num_inference_steps
        
        # Create beta schedule
        if config.scheduler.beta_schedule == "linear":
            self.betas = torch.linspace(
                config.scheduler.beta_start, 
                config.scheduler.beta_end, 
                self.num_train_timesteps
            )
        elif config.scheduler.beta_schedule == "scaled_linear":
            self.betas = torch.linspace(
                config.scheduler.beta_start ** 0.5, 
                config.scheduler.beta_end ** 0.5, 
                self.num_train_timesteps
            ) ** 2
        elif config.scheduler.beta_schedule == "squaredcos_cap_v2":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta_schedule: {config.scheduler.beta_schedule}")
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Variance schedule
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Modality-specific scaling factors
        self.modality_scales = {
            'text': config.scheduler.text_beta_scale,
            'speech': config.scheduler.speech_beta_scale,
            'control': config.scheduler.control_beta_scale,
            'memory': config.scheduler.memory_beta_scale
        }
    
    def _cosine_beta_schedule(self):
        """Cosine beta schedule from improved DDPM"""
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
        
        betas = []
        for i in range(self.num_train_timesteps):
            t1 = i / self.num_train_timesteps
            t2 = (i + 1) / self.num_train_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor, modality: str = 'text') -> torch.Tensor:
        """Add noise to original samples according to the schedule"""
        device = original_samples.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        # Apply modality-specific scaling
        scale = self.modality_scales.get(modality, 1.0)
        sqrt_alpha_prod = (alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1) * scale
        sqrt_one_minus_alpha_prod = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1) * scale
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             modality: str = 'text') -> torch.Tensor:
        """Perform one denoising step"""
        device = sample.device
        
        # Get current and previous alpha values
        alpha_prod_t = self.alphas_cumprod[timestep].to(device)
        alpha_prod_t_prev = self.alphas_cumprod_prev[timestep].to(device) if timestep > 0 else torch.tensor(1.0).to(device)
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute prediction type
        if self.config.scheduler.prediction_type == "epsilon":
            # Predict original sample from noise
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.scheduler.prediction_type == "v_prediction":
            # v-parameterization
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            pred_original_sample = model_output
        
        # Clip predicted original sample
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[timestep]) / (1 - alpha_prod_t)
        current_sample_coeff = self.alphas[timestep] ** 0.5 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        return pred_prev_sample
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference"""
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings"""
    
    def __init__(self, embedding_dim: int, max_positions: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal embeddings for timesteps"""
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class TextModalityHead(nn.Module):
    """Text modality head for diffusion in continuous embedding space"""
    
    def __init__(self, config: DiffusionMultimodalConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.modalities.text_embedding_dim
        self.hidden_size = config.transformer.hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_size)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, self.embedding_dim)
        )
        
        # Text encoder for conditioning (frozen)
        self.text_encoder = nn.Linear(768, self.embedding_dim)  # Assume 768-dim text features
    
    def forward(self, noisy_text: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_text: [batch_size, seq_len, embedding_dim] - noisy text embeddings
            hidden_states: [batch_size, seq_len, hidden_size] - transformer hidden states
        """
        # Project transformer hidden states to text embedding space
        denoised_text = self.output_projection(hidden_states)
        return denoised_text


class SpeechModalityHead(nn.Module):
    """Speech modality head for mel-spectrogram diffusion"""
    
    def __init__(self, config: DiffusionMultimodalConfig):
        super().__init__()
        self.config = config
        self.mel_bins = config.modalities.speech_mel_bins
        self.hidden_size = config.transformer.hidden_size
        
        # Temporal convolutions for speech processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.mel_bins, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, self.hidden_size, kernel_size=3, padding=1)
        )
        
        # Input projection
        self.input_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.SiLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, self.mel_bins, kernel_size=3, padding=1)
        )
    
    def forward(self, noisy_speech: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_speech: [batch_size, max_frames, mel_bins] - noisy mel-spectrograms
            hidden_states: [batch_size, max_frames, hidden_size] - transformer hidden states
        """
        batch_size, max_frames, _ = hidden_states.shape
        
        # Process through temporal convolutions
        # hidden_states: [batch_size, max_frames, hidden_size] -> [batch_size, hidden_size, max_frames]
        x = hidden_states.transpose(1, 2)
        x = self.output_projection[0](x.transpose(1, 2)).transpose(1, 2)  # Linear layer first
        x = self.output_projection[1](x)  # SiLU
        x = self.output_projection[2](x)  # Conv1d
        x = self.output_projection[3](x)  # SiLU  
        x = self.output_projection[4](x)  # Conv1d
        
        # Output: [batch_size, mel_bins, max_frames] -> [batch_size, max_frames, mel_bins]
        denoised_speech = x.transpose(1, 2)
        return denoised_speech


class ControlModalityHead(nn.Module):
    """Control token modality head for emotional/narrative control"""
    
    def __init__(self, config: DiffusionMultimodalConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.modalities.control_vocab_size
        self.embedding_dim = config.modalities.control_embedding_dim
        self.hidden_size = config.transformer.hidden_size
        
        # Control token embeddings
        self.control_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Input projection
        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_size)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, self.embedding_dim)
        )
    
    def forward(self, noisy_control: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_control: [batch_size, max_tokens, embedding_dim] - noisy control embeddings
            hidden_states: [batch_size, max_tokens, hidden_size] - transformer hidden states
        """
        denoised_control = self.output_projection(hidden_states)
        return denoised_control


class MemoryModalityHead(nn.Module):
    """Memory modality head for memory vector generation"""
    
    def __init__(self, config: DiffusionMultimodalConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.modalities.memory_embedding_dim
        self.metadata_dim = config.modalities.memory_metadata_dim
        self.hidden_size = config.transformer.hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(self.embedding_dim + self.metadata_dim, self.hidden_size)
        
        # Separate heads for embedding and metadata
        self.embedding_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, self.embedding_dim)
        )
        
        self.metadata_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, self.metadata_dim),
            nn.Sigmoid()  # Bound metadata values to [0,1]
        )
    
    def forward(self, noisy_memory: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_memory: [batch_size, max_vectors, embedding_dim + metadata_dim] - noisy memory
            hidden_states: [batch_size, max_vectors, hidden_size] - transformer hidden states
        """
        # Split predictions
        denoised_embedding = self.embedding_head(hidden_states)
        denoised_metadata = self.metadata_head(hidden_states)
        
        # Normalize embeddings
        denoised_embedding = F.normalize(denoised_embedding, p=2, dim=-1)
        
        # Combine
        denoised_memory = torch.cat([denoised_embedding, denoised_metadata], dim=-1)
        return denoised_memory


class CrossModalAttention(nn.Module):
    """Cross-attention between different modalities"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, hidden_size]
            key: [batch_size, seq_len_k, hidden_size]  
            value: [batch_size, seq_len_v, hidden_size]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores += attention_mask
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attention_weights


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with time conditioning and cross-modal attention"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Cross-modal attention (if this layer supports it)
        self.has_cross_attention = layer_idx in config.cross_attention_layers
        if self.has_cross_attention:
            self.cross_attention = CrossModalAttention(config.hidden_size, config.num_attention_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size) 
        if self.has_cross_attention:
            self.norm_cross = nn.LayerNorm(config.hidden_size)
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(self, hidden_states: torch.Tensor, time_emb: torch.Tensor,
                cross_modal_states: Optional[Dict[str, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            time_emb: [batch_size, time_embedding_dim]  
            cross_modal_states: Dict of other modality states for cross-attention
        """
        # Apply time conditioning
        time_cond = self.time_mlp(time_emb).unsqueeze(1)  # [batch_size, 1, hidden_size]
        hidden_states = hidden_states + time_cond
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, _ = self.self_attention(hidden_states, hidden_states, hidden_states,
                                           key_padding_mask=attention_mask)
        hidden_states = residual + attn_output
        
        # Cross-modal attention
        if self.has_cross_attention and cross_modal_states is not None:
            for modality, states in cross_modal_states.items():
                residual = hidden_states
                hidden_states_norm = self.norm_cross(hidden_states)
                cross_output, _ = self.cross_attention(hidden_states_norm, states, states)
                hidden_states = residual + cross_output
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states


class DiffusionMultimodalModel(nn.Module):
    """
    Main diffusion multimodal model for simultaneous generation of text, speech, 
    control tokens, and memory vectors with cross-modal attention.
    """
    
    def __init__(self, config: DiffusionMultimodalConfig):
        super().__init__()
        self.config = config
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(config)
        
        # Timestep embedding
        self.time_embedding = TimestepEmbedding(config.transformer.time_embedding_dim)
        
        # Modality heads
        self.text_head = TextModalityHead(config)
        self.speech_head = SpeechModalityHead(config)
        self.control_head = ControlModalityHead(config)
        self.memory_head = MemoryModalityHead(config)
        
        # Character conditioning
        if config.use_character_conditioning:
            self.character_embedding = nn.Embedding(1000, config.character_embedding_dim)  # Support up to 1000 characters
            self.character_projection = nn.Linear(config.character_embedding_dim, config.transformer.hidden_size)
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(config.transformer, i) 
            for i in range(config.transformer.num_layers)
        ])
        
        # Input projections for each modality
        self.text_input_proj = nn.Linear(config.modalities.text_embedding_dim, config.transformer.hidden_size)
        self.speech_input_proj = nn.Linear(config.modalities.speech_mel_bins, config.transformer.hidden_size)
        self.control_input_proj = nn.Linear(config.modalities.control_embedding_dim, config.transformer.hidden_size)
        self.memory_input_proj = nn.Linear(config.modalities.memory_embedding_dim + config.modalities.memory_metadata_dim, config.transformer.hidden_size)
        
        # Positional encodings
        self.text_pos_embedding = nn.Parameter(torch.randn(1, config.modalities.text_max_sequence_length, config.transformer.hidden_size))
        self.speech_pos_embedding = nn.Parameter(torch.randn(1, config.modalities.speech_max_frames, config.transformer.hidden_size))
        self.control_pos_embedding = nn.Parameter(torch.randn(1, config.modalities.control_max_tokens, config.transformer.hidden_size))
        self.memory_pos_embedding = nn.Parameter(torch.randn(1, config.modalities.memory_max_vectors, config.transformer.hidden_size))
        
        # EMA for stable training
        if config.use_ema:
            self.ema_decay = config.ema_decay
            self.register_buffer('ema_enabled', torch.tensor(True))
            # EMA parameters will be registered automatically
        
        # Alignment projection heads - project all modalities to same space for alignment loss
        alignment_dim = 256  # Common dimension for alignment
        self.text_alignment_proj = nn.Linear(config.modalities.text_embedding_dim, alignment_dim)
        self.speech_alignment_proj = nn.Linear(config.modalities.speech_mel_bins, alignment_dim)
        self.control_alignment_proj = nn.Linear(config.modalities.control_embedding_dim, alignment_dim)
        self.memory_alignment_proj = nn.Linear(config.modalities.memory_embedding_dim + config.modalities.memory_metadata_dim, alignment_dim)
    
    def forward(self, 
                # Noisy inputs
                noisy_text: torch.Tensor,
                noisy_speech: torch.Tensor, 
                noisy_control: torch.Tensor,
                noisy_memory: torch.Tensor,
                
                # Conditioning
                timesteps: torch.Tensor,
                character_ids: Optional[torch.Tensor] = None,
                
                # Ground truth for training
                clean_text: Optional[torch.Tensor] = None,
                clean_speech: Optional[torch.Tensor] = None,
                clean_control: Optional[torch.Tensor] = None,
                clean_memory: Optional[torch.Tensor] = None,
                
                # Masks
                text_mask: Optional[torch.Tensor] = None,
                speech_mask: Optional[torch.Tensor] = None,
                control_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                
                return_loss: bool = True) -> DiffusionOutput:
        """
        Forward pass of the diffusion multimodal model.
        
        Args:
            noisy_text: [batch_size, text_seq_len, text_embedding_dim]
            noisy_speech: [batch_size, speech_frames, mel_bins]
            noisy_control: [batch_size, control_tokens, control_embedding_dim]
            noisy_memory: [batch_size, memory_vectors, memory_dim + metadata_dim]
            timesteps: [batch_size] - timestep for each sample
            character_ids: [batch_size] - character conditioning
        """
        batch_size = noisy_text.shape[0]
        device = noisy_text.device
        
        # Time embeddings
        time_emb = self.time_embedding(timesteps)
        
        # Character conditioning
        if self.config.use_character_conditioning and character_ids is not None:
            char_emb = self.character_embedding(character_ids)
            char_proj = self.character_projection(char_emb)
            time_emb = time_emb + char_proj
        
        # Project inputs to hidden space and add positional embeddings
        text_hidden = self.text_input_proj(noisy_text) + self.text_pos_embedding[:, :noisy_text.shape[1], :]
        speech_hidden = self.speech_input_proj(noisy_speech) + self.speech_pos_embedding[:, :noisy_speech.shape[1], :]
        control_hidden = self.control_input_proj(noisy_control) + self.control_pos_embedding[:, :noisy_control.shape[1], :]
        memory_hidden = self.memory_input_proj(noisy_memory) + self.memory_pos_embedding[:, :noisy_memory.shape[1], :]
        
        # Store modality states for cross-attention
        modality_states = {
            'text': text_hidden,
            'speech': speech_hidden,
            'control': control_hidden,
            'memory': memory_hidden
        }
        
        # Process each modality through transformer blocks
        processed_states = {}
        
        for modality, hidden_states in modality_states.items():
            current_states = hidden_states
            
            # Prepare cross-modal states (exclude current modality)
            cross_modal_states = {k: v for k, v in modality_states.items() if k != modality}
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                current_states = block(
                    current_states, 
                    time_emb, 
                    cross_modal_states if block.has_cross_attention else None
                )
            
            processed_states[modality] = current_states
        
        # Generate predictions through modality heads
        text_pred = self.text_head(noisy_text, processed_states['text'])
        speech_pred = self.speech_head(noisy_speech, processed_states['speech'])
        control_pred = self.control_head(noisy_control, processed_states['control'])
        memory_pred = self.memory_head(noisy_memory, processed_states['memory'])
        
        # Create output
        output = DiffusionOutput(
            text_prediction=text_pred,
            speech_prediction=speech_pred,
            control_prediction=control_pred,
            memory_prediction=memory_pred,
            timestep_embeddings=time_emb
        )
        
        # Compute losses if training
        if return_loss and all(clean is not None for clean in [clean_text, clean_speech, clean_control, clean_memory]):
            losses = self._compute_losses(
                output, 
                clean_text, clean_speech, clean_control, clean_memory,
                text_mask, speech_mask, control_mask, memory_mask
            )
            
            output.text_loss = losses['text_loss']
            output.speech_loss = losses['speech_loss'] 
            output.control_loss = losses['control_loss']
            output.memory_loss = losses['memory_loss']
            output.alignment_loss = losses['alignment_loss']
            output.total_loss = losses['total_loss']
        
        return output
    
    def _compute_losses(self, output: DiffusionOutput,
                       clean_text: torch.Tensor, clean_speech: torch.Tensor,
                       clean_control: torch.Tensor, clean_memory: torch.Tensor,
                       text_mask: Optional[torch.Tensor] = None,
                       speech_mask: Optional[torch.Tensor] = None,
                       control_mask: Optional[torch.Tensor] = None,
                       memory_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute diffusion losses for all modalities"""
        
        loss_config = self.config.loss
        
        # Basic diffusion losses (MSE by default)
        if loss_config.diffusion_loss_type == "mse":
            loss_fn = F.mse_loss
        elif loss_config.diffusion_loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_config.diffusion_loss_type == "huber":
            loss_fn = F.smooth_l1_loss
        else:
            loss_fn = F.mse_loss
        
        # Compute individual modality losses
        text_loss = loss_fn(output.text_prediction, clean_text, reduction='none')
        if text_mask is not None:
            text_loss = text_loss * text_mask.unsqueeze(-1)
        text_loss = text_loss.mean() * loss_config.text_loss_weight
        
        speech_loss = loss_fn(output.speech_prediction, clean_speech, reduction='none')
        if speech_mask is not None:
            speech_loss = speech_loss * speech_mask.unsqueeze(-1)
        speech_loss = speech_loss.mean() * loss_config.speech_loss_weight
        
        control_loss = loss_fn(output.control_prediction, clean_control, reduction='none')
        if control_mask is not None:
            control_loss = control_loss * control_mask.unsqueeze(-1)
        control_loss = control_loss.mean() * loss_config.control_loss_weight
        
        memory_loss = loss_fn(output.memory_prediction, clean_memory, reduction='none')
        if memory_mask is not None:
            memory_loss = memory_loss * memory_mask.unsqueeze(-1)
        memory_loss = memory_loss.mean() * loss_config.memory_loss_weight
        
        # Cross-modal alignment losses
        alignment_loss = self._compute_alignment_losses(output, loss_config)
        
        # Total loss
        total_loss = text_loss + speech_loss + control_loss + memory_loss + alignment_loss
        
        return {
            'text_loss': text_loss,
            'speech_loss': speech_loss,
            'control_loss': control_loss,
            'memory_loss': memory_loss,
            'alignment_loss': alignment_loss,
            'total_loss': total_loss
        }
    
    def _compute_alignment_losses(self, output: DiffusionOutput, loss_config) -> torch.Tensor:
        """Compute cross-modal alignment losses"""
        alignment_loss = torch.tensor(0.0, device=output.text_prediction.device)
        
        # Project all modalities to common alignment space
        text_avg = output.text_prediction.mean(dim=1)  # [batch_size, text_embedding_dim]
        speech_avg = output.speech_prediction.mean(dim=1)  # [batch_size, mel_bins]
        control_avg = output.control_prediction.mean(dim=1)  # [batch_size, control_embedding_dim]
        memory_avg = output.memory_prediction.mean(dim=1)  # [batch_size, memory_dim]
        
        # Project to common space
        text_proj = F.normalize(self.text_alignment_proj(text_avg), p=2, dim=1)
        speech_proj = F.normalize(self.speech_alignment_proj(speech_avg), p=2, dim=1)
        control_proj = F.normalize(self.control_alignment_proj(control_avg), p=2, dim=1)
        memory_proj = F.normalize(self.memory_alignment_proj(memory_avg), p=2, dim=1)
        
        # Text-speech alignment
        if loss_config.text_speech_alignment_weight > 0:
            cos_sim = F.cosine_similarity(text_proj, speech_proj, dim=1)
            text_speech_loss = (1 - cos_sim).mean() * loss_config.text_speech_alignment_weight
            alignment_loss = alignment_loss + text_speech_loss
        
        # Text-control alignment
        if loss_config.text_control_alignment_weight > 0:
            cos_sim = F.cosine_similarity(text_proj, control_proj, dim=1)
            text_control_loss = (1 - cos_sim).mean() * loss_config.text_control_alignment_weight
            alignment_loss = alignment_loss + text_control_loss
        
        # Text-memory alignment
        if loss_config.text_memory_alignment_weight > 0:
            cos_sim = F.cosine_similarity(text_proj, memory_proj, dim=1)
            text_memory_loss = (1 - cos_sim).mean() * loss_config.text_memory_alignment_weight
            alignment_loss = alignment_loss + text_memory_loss
        
        return alignment_loss
    
    @torch.no_grad()
    def generate(self, 
                 batch_size: int = 1,
                 character_ids: Optional[torch.Tensor] = None,
                 guidance_scale: float = None,
                 num_inference_steps: int = None,
                 device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Generate samples using DDPM sampling.
        
        Args:
            batch_size: Number of samples to generate
            character_ids: Character conditioning
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
        """
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.scheduler.num_inference_steps
        
        # Initialize random noise for all modalities
        text_shape = (batch_size, self.config.modalities.text_max_sequence_length, self.config.modalities.text_embedding_dim)
        speech_shape = (batch_size, self.config.modalities.speech_max_frames, self.config.modalities.speech_mel_bins)
        control_shape = (batch_size, self.config.modalities.control_max_tokens, self.config.modalities.control_embedding_dim)
        memory_shape = (batch_size, self.config.modalities.memory_max_vectors, self.config.modalities.memory_embedding_dim + self.config.modalities.memory_metadata_dim)
        
        text_noise = torch.randn(text_shape, device=device)
        speech_noise = torch.randn(speech_shape, device=device)
        control_noise = torch.randn(control_shape, device=device)
        memory_noise = torch.randn(memory_shape, device=device)
        
        # Set inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        
        # Current samples (start from noise)
        current_text = text_noise
        current_speech = speech_noise
        current_control = control_noise
        current_memory = memory_noise
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            timestep_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if guidance_scale > 1.0 and character_ids is not None:
                # Conditional prediction
                cond_output = self.forward(
                    current_text, current_speech, current_control, current_memory,
                    timestep_tensor, character_ids, return_loss=False
                )
                
                # Unconditional prediction (no character conditioning)
                uncond_output = self.forward(
                    current_text, current_speech, current_control, current_memory,
                    timestep_tensor, None, return_loss=False
                )
                
                # Apply guidance
                text_pred = uncond_output.text_prediction + guidance_scale * (cond_output.text_prediction - uncond_output.text_prediction)
                speech_pred = uncond_output.speech_prediction + guidance_scale * (cond_output.speech_prediction - uncond_output.speech_prediction)
                control_pred = uncond_output.control_prediction + guidance_scale * (cond_output.control_prediction - uncond_output.control_prediction)
                memory_pred = uncond_output.memory_prediction + guidance_scale * (cond_output.memory_prediction - uncond_output.memory_prediction)
            else:
                # Direct prediction
                output = self.forward(
                    current_text, current_speech, current_control, current_memory,
                    timestep_tensor, character_ids, return_loss=False
                )
                text_pred = output.text_prediction
                speech_pred = output.speech_prediction
                control_pred = output.control_prediction
                memory_pred = output.memory_prediction
            
            # Denoise each modality
            current_text = self.noise_scheduler.step(text_pred, t, current_text, modality='text')
            current_speech = self.noise_scheduler.step(speech_pred, t, current_speech, modality='speech')
            current_control = self.noise_scheduler.step(control_pred, t, current_control, modality='control')
            current_memory = self.noise_scheduler.step(memory_pred, t, current_memory, modality='memory')
        
        return {
            'text': current_text,
            'speech': current_speech,
            'control': current_control,
            'memory': current_memory
        }
    
    def update_ema(self):
        """Update EMA parameters"""
        if not self.config.use_ema or not self.ema_enabled:
            return
        
        decay = self.ema_decay
        for name, param in self.named_parameters():
            if param.requires_grad:
                ema_name = f"ema_{name.replace('.', '_')}"
                if hasattr(self, ema_name):
                    ema_param = getattr(self, ema_name)
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
                else:
                    # Initialize EMA parameter
                    self.register_buffer(ema_name, param.data.clone())


def create_diffusion_model(config: DiffusionMultimodalConfig) -> DiffusionMultimodalModel:
    """Factory function to create diffusion multimodal model"""
    return DiffusionMultimodalModel(config) 