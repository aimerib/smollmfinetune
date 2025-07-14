# narrative_engine/config.py
from dataclasses import dataclass

@dataclass
class NarrativeLLMConfig:
    """
    Configuration for the Narrative-LLM.
    This dataclass is the single source of truth for all model hyperparameters.
    """
    # --- Base Model ---
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # --- Vocabulary and Embedding ---
    vocab_size: int = 32000  # Size of the tokenizer vocabulary
    token_dim: int = 4096  # Dimensionality of token embeddings (d_model)
    max_position_embeddings: int = 4096 # Maximum sequence length
    
    # --- Session Management ---
    # The session embedding allows the model to be stateful across turns 
    # without needing the full history in the prompt.
    max_session_ids: int = 1024 # Max concurrent sessions the model can distinguish
    session_embedding_dim: int = 256 # Dimensionality of the session embedding

    # --- Transformer Backbone ---
    num_hidden_layers: int = 32 # Number of transformer blocks
    num_attention_heads: int = 32 # Number of heads in multi-head attention
    num_key_value_heads: int = 8 # Number of heads for key/value in Grouped-Query Attention
    hidden_size: int = 4096 # Dimensionality of the hidden layers (d_model)
    intermediate_size: int = 14336 # Dimensionality of the feed-forward layer
    
    # --- MoE (Mixture of Experts) ---
    # Used in the feed-forward layers for scalable capacity.
    num_experts: int = 8 # Number of experts in each MoE layer
    num_experts_per_tok: int = 2 # Number of experts to route each token to (top-k)

    # --- External Memory Cross-Attention ---
    # Allows the model to attend to information from a vector database.
    memory_key_value_dim: int = 1024 # Dimensionality of keys/values from external memory
    num_memory_attention_heads: int = 8 # Number of heads for cross-attention
    
    # --- C.L.A.R.A. Loop Features ---
    # Triple-head architecture and emotional recirculation
    control_head_dim: int = 256  # Dimensionality of control token head
    control_vocab_size: int = 64  # Max control tokens
    recirculation_layers: int = 2  # Layers for emotional context injection
    surprise_threshold: float = 0.7  # Threshold for surprise detection
    decay_steps: int = 4  # How many turns to track emotional momentum
    
    # --- Speech Head (Fourth Head) ---
    # Configuration for native speech synthesis capabilities
    enable_speech_head: bool = False  # Enable/disable speech head
    speech_mel_bins: int = 80  # Number of mel-frequency bins
    speech_quantization_bits: int = 4  # Quantization bits (4-bit = 16 levels)
    speech_hop_length: int = 256  # Hop length for mel-spectrogram (25ms at 22050Hz)
    speech_sample_rate: int = 22050  # Target sample rate for speech synthesis
    speech_frame_context: int = 1000  # Context window for temporal modeling
    num_character_embeddings: int = 1000  # Max characters for voice conditioning
    enable_cross_attention: bool = True  # Text-speech cross-modal attention
    enable_streaming: bool = False  # Streaming speech generation capability
    
    # --- Adapters (LoRA/DoRA) ---
    # Configuration for dynamically loaded persona adapters.
    lora_r: int = 16 # Rank for LoRA decomposition
    lora_alpha: int = 32 # Alpha for LoRA scaling
    lora_dropout: float = 0.05
    
    # --- General ---
    dropout: float = 0.1 # Dropout rate for regularization
    initializer_range: float = 0.02 # Std dev for weight initialization
    rms_norm_eps: float = 1e-6 # Epsilon for RMSNorm 