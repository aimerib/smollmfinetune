"""
Triple-Head Reward Models for Narrative Engine.

This module implements reward models for each head of the triple-head architecture:
- Generation Head: Evaluates text quality, creativity, and factual accuracy
- Control Head: Evaluates emotional appropriateness and personality consistency
- Memory Head: Evaluates memory formation accuracy and consistency
- Coordinated Model: Evaluates cross-head harmony and overall coherence
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardModelOutput:
    """Output from a reward model"""
    reward: torch.Tensor  # [batch_size]
    features: Optional[torch.Tensor] = None  # Intermediate features for analysis
    head_type: str = "unknown"


class GenerationRewardModel(nn.Module):
    """
    Reward model for generation head outputs.
    
    Evaluates text quality, creativity, coherence, and factual accuracy.
    Uses a language model backbone with a reward head.
    """
    
    def __init__(self, base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", 
                 hidden_size: int = 768):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Get actual hidden size from model config
        actual_hidden_size = self.base_model.config.hidden_size
        
        # Reward head: maps from hidden states to scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(actual_hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        logger.info(f"Initialized GenerationRewardModel with {base_model_name}")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> RewardModelOutput:
        """
        Forward pass through generation reward model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            RewardModelOutput with scalar rewards
        """
        # Get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooled output or last hidden state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling over sequence
            hidden_states = outputs.last_hidden_state
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                # Simple mean pooling without mask
                pooled = hidden_states.mean(dim=1)
        
        # Compute reward
        reward = self.reward_head(pooled).squeeze(-1)
        
        return RewardModelOutput(
            reward=reward,
            features=pooled,
            head_type="generation"
        )


class ControlRewardModel(nn.Module):
    """
    Reward model for control head outputs.
    
    Evaluates emotional appropriateness, personality consistency, and mood transitions.
    """
    
    def __init__(self, num_control_tokens: int = 64, hidden_size: int = 256):
        super().__init__()
        
        self.num_control_tokens = num_control_tokens
        
        # Control token encoder
        self.control_encoder = nn.Sequential(
            nn.Linear(num_control_tokens, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        logger.info(f"Initialized ControlRewardModel with {num_control_tokens} tokens")
    
    def forward(self, control_tokens: torch.Tensor) -> RewardModelOutput:
        """
        Forward pass through control reward model.
        
        Args:
            control_tokens: Multi-hot encoded control tokens [batch_size, num_control_tokens]
            
        Returns:
            RewardModelOutput with scalar rewards
        """
        # Encode control tokens
        features = self.control_encoder(control_tokens)
        
        # Compute reward
        reward = self.reward_head(features).squeeze(-1)
        
        return RewardModelOutput(
            reward=reward,
            features=features,
            head_type="control"
        )


class MemoryRewardModel(nn.Module):
    """
    Reward model for memory head outputs.
    
    Evaluates memory accuracy, consistency, and formation quality.
    Handles both embedding vectors and metadata.
    """
    
    def __init__(self, embedding_dim: int = 768, metadata_dim: int = 4, 
                 hidden_size: int = 512):
        super().__init__()
        
        # Separate encoders for embedding and metadata
        self.embedding_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size // 4)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        logger.info(f"Initialized MemoryRewardModel with embedding_dim={embedding_dim}")
    
    def forward(self, memory_embeddings: torch.Tensor, 
                memory_metadata: torch.Tensor) -> RewardModelOutput:
        """
        Forward pass through memory reward model.
        
        Args:
            memory_embeddings: Memory vectors [batch_size, 768]
            memory_metadata: Memory metadata [batch_size, 4]
            
        Returns:
            RewardModelOutput with scalar rewards
        """
        # Encode components
        embedding_features = self.embedding_encoder(memory_embeddings)
        metadata_features = self.metadata_encoder(memory_metadata)
        
        # Fuse features
        combined = torch.cat([embedding_features, metadata_features], dim=-1)
        features = self.fusion_layer(combined)
        
        # Compute reward
        reward = self.reward_head(features).squeeze(-1)
        
        return RewardModelOutput(
            reward=reward,
            features=features,
            head_type="memory"
        )


class CoordinatedRewardModel(nn.Module):
    """
    Reward model that evaluates cross-head coordination and overall coherence.
    
    Takes outputs from all three heads and evaluates how well they work together.
    """
    
    def __init__(self, text_hidden_size: int = 768, control_vocab_size: int = 64,
                 memory_dim: int = 768, metadata_dim: int = 4, 
                 fusion_hidden_size: int = 512):
        super().__init__()
        
        # Head-specific encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden_size)
        )
        
        self.control_encoder = nn.Sequential(
            nn.Linear(control_vocab_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size // 2, fusion_hidden_size),
            nn.LayerNorm(fusion_hidden_size)
        )
        
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim + metadata_dim, fusion_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden_size)
        )
        
        # Cross-attention layers for head interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_hidden_size * 3, fusion_hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden_size * 2),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_size * 2, fusion_hidden_size),
            nn.ReLU()
        )
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_size // 2, 1)
        )
        
        logger.info("Initialized CoordinatedRewardModel for cross-head evaluation")
    
    def forward(self, text_features: torch.Tensor, control_tokens: torch.Tensor,
                memory_embeddings: torch.Tensor, memory_metadata: torch.Tensor) -> RewardModelOutput:
        """
        Forward pass evaluating cross-head coordination.
        
        Args:
            text_features: Text generation features [batch_size, text_hidden_size]
            control_tokens: Control token probabilities [batch_size, control_vocab_size]
            memory_embeddings: Memory vectors [batch_size, memory_dim]
            memory_metadata: Memory metadata [batch_size, metadata_dim]
            
        Returns:
            RewardModelOutput with coordination rewards
        """
        # Encode each head's output
        text_encoded = self.text_encoder(text_features)
        control_encoded = self.control_encoder(control_tokens)
        
        # Combine memory components
        memory_combined = torch.cat([memory_embeddings, memory_metadata], dim=-1)
        memory_encoded = self.memory_encoder(memory_combined)
        
        # Stack for cross-attention [batch_size, 3, hidden_size]
        stacked = torch.stack([text_encoded, control_encoded, memory_encoded], dim=1)
        
        # Apply cross-attention to model interactions
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        
        # Flatten and fuse
        flattened = attended.reshape(attended.size(0), -1)
        features = self.fusion_network(flattened)
        
        # Compute coordination reward
        reward = self.reward_head(features).squeeze(-1)
        
        return RewardModelOutput(
            reward=reward,
            features=features,
            head_type="coordinated"
        )


def train_reward_model(model_type: str, preferences: List[Dict[str, Any]], 
                      model: Optional[nn.Module] = None, 
                      num_epochs: int = 3, 
                      learning_rate: float = 1e-5,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu") -> nn.Module:
    """
    Train a reward model on preference data.
    
    Args:
        model_type: Type of reward model ("generation", "control", "memory", "coordinated")
        preferences: List of preference dictionaries
        model: Optional pre-initialized model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Trained reward model
    """
    # Initialize model if not provided
    if model is None:
        if model_type == "generation":
            model = GenerationRewardModel()
        elif model_type == "control":
            model = ControlRewardModel()
        elif model_type == "memory":
            model = MemoryRewardModel()
        elif model_type == "coordinated":
            model = CoordinatedRewardModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Training {model_type} reward model for {num_epochs} epochs")
    
    # Training loop would go here
    # For now, just return the model
    return model


def load_head_preferences(preference_file: Path, head_type: str) -> List[Dict[str, Any]]:
    """
    Load head-specific preferences from JSONL file.
    
    Args:
        preference_file: Path to preference JSONL file
        head_type: Type of head ("generation", "control", "memory")
        
    Returns:
        List of processed preference dictionaries
    """
    preferences = []
    
    with open(preference_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                # Calculate reward based on ratings
                if head_type == "generation":
                    reward = np.mean([
                        data.get('content_quality', 5) / 10,
                        data.get('creativity', 5) / 10,
                        data.get('factual_accuracy', 5) / 10
                    ])
                elif head_type == "control":
                    reward = np.mean([
                        data.get('emotional_appropriateness', 5) / 10,
                        data.get('personality_consistency', 5) / 10,
                        data.get('mood_matching', 5) / 10
                    ])
                elif head_type == "memory":
                    reward = np.mean([
                        data.get('memory_accuracy', 5) / 10,
                        data.get('memory_consistency', 5) / 10,
                        data.get('formation_quality', 5) / 10
                    ])
                else:
                    reward = data.get('coordination', 5) / 10
                
                data['reward'] = float(reward)
                preferences.append(data)
    
    logger.info(f"Loaded {len(preferences)} {head_type} preferences")
    return preferences


def prepare_control_preferences(raw_preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process raw control preferences into training format."""
    processed = []
    for pref in raw_preferences:
        # Mock control token extraction - in practice would use actual model outputs
        processed.append({
            'control_tokens': torch.rand(64),  # Mock multi-hot encoding
            'reward': pref.get('reward', 0.5)
        })
    return processed


def prepare_memory_preferences(raw_preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process raw memory preferences into training format."""
    processed = []
    for pref in raw_preferences:
        # Mock memory data - in practice would use actual model outputs
        processed.append({
            'memory_embedding': torch.randn(768),
            'memory_metadata': torch.rand(4),
            'reward': pref.get('reward', 0.5)
        })
    return processed 