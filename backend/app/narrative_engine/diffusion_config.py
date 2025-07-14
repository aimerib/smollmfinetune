from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch


@dataclass
class DiffusionSchedulerConfig:
    """Configuration for noise scheduling in diffusion process"""
    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "scaled_linear", "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    
    # Modality-specific scheduling
    text_beta_scale: float = 1.0
    speech_beta_scale: float = 1.0  
    control_beta_scale: float = 0.8
    memory_beta_scale: float = 0.6


@dataclass
class ModalityConfig:
    """Configuration for individual modality heads"""
    # Text Modality
    text_embedding_dim: int = 768
    text_max_sequence_length: int = 512
    text_continuous_space: bool = True  # Use continuous embeddings for diffusion
    
    # Speech Modality  
    speech_mel_bins: int = 80
    speech_max_frames: int = 1000
    speech_frame_rate: int = 40  # frames per second
    
    # Control Modality
    control_vocab_size: int = 64
    control_embedding_dim: int = 256
    control_max_tokens: int = 32
    
    # Memory Modality
    memory_embedding_dim: int = 768
    memory_metadata_dim: int = 4
    memory_max_vectors: int = 16


@dataclass 
class DiffusionTransformerConfig:
    """Configuration for the diffusion transformer backbone"""
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    
    # Cross-attention between modalities
    cross_attention_layers: List[int] = None  # Which layers have cross-attention
    modality_fusion_method: str = "cross_attention"  # "cross_attention", "concat", "add"
    
    # Conditioning
    time_embedding_dim: int = 256
    condition_embedding_dim: int = 512
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    def __post_init__(self):
        if self.cross_attention_layers is None:
            # Add cross-attention to every 4th layer
            self.cross_attention_layers = list(range(3, self.num_layers, 4))


@dataclass
class DiffusionLossConfig:
    """Configuration for diffusion training losses"""
    # Basic diffusion loss weights
    text_loss_weight: float = 1.0
    speech_loss_weight: float = 1.0
    control_loss_weight: float = 0.8
    memory_loss_weight: float = 0.6
    
    # Cross-modal alignment losses
    text_speech_alignment_weight: float = 0.3
    text_control_alignment_weight: float = 0.2
    text_memory_alignment_weight: float = 0.2
    
    # Consistency losses
    temporal_consistency_weight: float = 0.1
    semantic_consistency_weight: float = 0.1
    
    # Loss types
    diffusion_loss_type: str = "mse"  # "mse", "l1", "huber"
    alignment_loss_type: str = "cosine"  # "cosine", "mse", "contrastive"


@dataclass
class DiffusionMultimodalConfig:
    """Complete configuration for diffusion multimodal model"""
    # Model identification
    model_name: str = "DiffusionNarrativeLLM"
    model_version: str = "1.0"
    
    # Subconfigurations
    scheduler: DiffusionSchedulerConfig = None
    modalities: ModalityConfig = None
    transformer: DiffusionTransformerConfig = None
    loss: DiffusionLossConfig = None
    
    # Training configuration
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Sampling configuration
    guidance_scale: float = 7.5  # For classifier-free guidance
    guidance_probability: float = 0.1  # Probability of unconditional training
    
    # Character conditioning
    character_embedding_dim: int = 256
    use_character_conditioning: bool = True
    use_style_conditioning: bool = True
    
    # Memory and context
    max_context_length: int = 2048
    use_memory_conditioning: bool = True
    memory_retrieval_k: int = 5
    
    # Advanced features
    use_ema: bool = True  # Exponential moving average
    ema_decay: float = 0.9999
    use_self_conditioning: bool = True
    self_conditioning_probability: float = 0.5
    
    def __post_init__(self):
        if self.scheduler is None:
            self.scheduler = DiffusionSchedulerConfig()
        if self.modalities is None:
            self.modalities = ModalityConfig()
        if self.transformer is None:
            self.transformer = DiffusionTransformerConfig()
        if self.loss is None:
            self.loss = DiffusionLossConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiffusionMultimodalConfig':
        """Create from dictionary"""
        # Recursively create nested dataclasses
        if 'scheduler' in data and isinstance(data['scheduler'], dict):
            data['scheduler'] = DiffusionSchedulerConfig(**data['scheduler'])
        if 'modalities' in data and isinstance(data['modalities'], dict):
            data['modalities'] = ModalityConfig(**data['modalities'])
        if 'transformer' in data and isinstance(data['transformer'], dict):
            data['transformer'] = DiffusionTransformerConfig(**data['transformer'])
        if 'loss' in data and isinstance(data['loss'], dict):
            data['loss'] = DiffusionLossConfig(**data['loss'])
        
        return cls(**data)
    
    def get_total_parameters(self) -> int:
        """Estimate total model parameters"""
        # -----------------------------------------------
        # Embedding + Positional
        # -----------------------------------------------
        embedding_params = (
            self.modalities.text_embedding_dim * self.transformer.hidden_size +
            self.modalities.speech_mel_bins * self.transformer.hidden_size +
            self.modalities.control_embedding_dim * self.transformer.hidden_size +
            (self.modalities.memory_embedding_dim + self.modalities.memory_metadata_dim) * self.transformer.hidden_size
        )

        # -----------------------------------------------
        # Transformer Blocks
        # Each block contains:
        #   – Q, K, V, O projections  (4 * h^2)
        #   – FFN first & second layer (2 * h * f)
        #   – Biases for all linear layers (negligible but included)
        # -----------------------------------------------
        h = self.transformer.hidden_size
        f = self.transformer.intermediate_size
        layers = self.transformer.num_layers
        attn_linear = 4 * h * h
        ffn_linear = 2 * h * f
        per_layer = attn_linear + ffn_linear
        transformer_params = per_layer * layers

        # -----------------------------------------------
        # Time-conditioning & modality projections
        # -----------------------------------------------
        time_cond_params = self.transformer.time_embedding_dim * h * 2  # 2-layer MLP

        # Add a small overhead (biases & layer norms) ≈ 2% of main params
        overhead = int(0.02 * (transformer_params + embedding_params))

        total = transformer_params + embedding_params + time_cond_params + overhead
        return total
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if self.transformer.hidden_size % self.transformer.num_attention_heads != 0:
            issues.append("hidden_size must be divisible by num_attention_heads")
        
        if self.scheduler.num_inference_steps > self.scheduler.num_train_timesteps:
            issues.append("num_inference_steps should not exceed num_train_timesteps")
        
        if self.gradient_accumulation_steps < 1:
            issues.append("gradient_accumulation_steps must be >= 1")
        
        return issues


# Predefined configurations for different scales
def get_small_config() -> DiffusionMultimodalConfig:
    """Small model configuration for testing"""
    return DiffusionMultimodalConfig(
        transformer=DiffusionTransformerConfig(
            hidden_size=512,
            num_layers=12,
            num_attention_heads=8,
            intermediate_size=2048,
            time_embedding_dim=512  # Match hidden_size to avoid dimension mismatch
        ),
        modalities=ModalityConfig(
            text_embedding_dim=512,
            text_max_sequence_length=256,
            speech_max_frames=500
        )
    )


def get_medium_config() -> DiffusionMultimodalConfig:
    """Medium model configuration"""
    return DiffusionMultimodalConfig(
        transformer=DiffusionTransformerConfig(
            hidden_size=1024,
            num_layers=24, 
            num_attention_heads=16,
            intermediate_size=4096,
            time_embedding_dim=1024  # Align with hidden_size to avoid dim mismatch
        )
    )


def get_large_config() -> DiffusionMultimodalConfig:
    """Large model configuration"""
    return DiffusionMultimodalConfig(
        transformer=DiffusionTransformerConfig(
            hidden_size=1536,
            num_layers=36,
            num_attention_heads=24,
            intermediate_size=6144,
            time_embedding_dim=1536  # Align with hidden_size to avoid dim mismatch
        ),
        modalities=ModalityConfig(
            text_embedding_dim=1536,
            speech_max_frames=2000,
            memory_embedding_dim=1536
        )
    ) 