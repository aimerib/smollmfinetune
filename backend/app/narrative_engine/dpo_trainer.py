"""
Triple-Head DPO (Direct Preference Optimization) Trainer.

Implements DPO training for the triple-head architecture with:
- Head-specific DPO losses
- Cross-head regularization
- Coordinated optimization
- Dynamic learning rate adaptation
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from tqdm import tqdm

from .model import NarrativeLLM
from .loss import TripleHeadLoss

logger = logging.getLogger(__name__)


@dataclass
class TripleHeadDPOConfig:
    """Configuration for triple-head DPO training"""
    head_type: str = "all"  # "generation", "control", "memory", or "all"
    learning_rate: float = 1e-5
    beta: float = 0.1  # KL penalty coefficient
    
    # Head-specific learning rates
    head_specific_lr: Dict[str, float] = field(default_factory=lambda: {
        "generation": 1e-5,
        "control": 5e-6,
        "memory": 2e-6
    })
    
    # Loss weights for coordinated training
    generation_weight: float = 1.0
    control_weight: float = 0.8
    memory_weight: float = 0.6
    coordination_weight: float = 0.4
    
    # Advanced features
    cross_head_regularization: float = 0.01
    emotional_arc_weight: float = 0.2
    memory_consistency_weight: float = 0.3
    
    # Training parameters
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    
    def __post_init__(self):
        """Validate configuration"""
        valid_head_types = ["generation", "control", "memory", "all"]
        if self.head_type not in valid_head_types:
            raise ValueError(f"Invalid head_type: {self.head_type}. Must be one of {valid_head_types}")


def compute_generation_dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    padding_value: int = -100
) -> torch.Tensor:
    """
    Compute DPO loss for generation head.
    
    Standard DPO loss on text generation quality.
    """
    # Get log probabilities
    policy_chosen_logprobs = get_batch_logps(policy_chosen_logits, labels, padding_value)
    policy_rejected_logprobs = get_batch_logps(policy_rejected_logits, labels, padding_value)
    ref_chosen_logprobs = get_batch_logps(ref_chosen_logits, labels, padding_value)
    ref_rejected_logprobs = get_batch_logps(ref_rejected_logits, labels, padding_value)
    
    # Compute DPO loss
    pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
    
    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
    
    return loss


def compute_control_dpo_loss(
    policy_chosen_probs: torch.Tensor,
    policy_rejected_probs: torch.Tensor,
    ref_chosen_probs: torch.Tensor,
    ref_rejected_probs: torch.Tensor,
    beta: float = 0.1,
    emotional_arc_weight: float = 0.2
) -> torch.Tensor:
    """
    Compute DPO loss for control head.
    
    Handles multi-label control token distributions with emotional arc preservation.
    """
    # Ensure probabilities are in valid range [0, 1]
    eps = 1e-8
    policy_chosen_probs = torch.clamp(policy_chosen_probs, min=eps, max=1-eps)
    policy_rejected_probs = torch.clamp(policy_rejected_probs, min=eps, max=1-eps)
    ref_chosen_probs = torch.clamp(ref_chosen_probs, min=eps, max=1-eps)
    ref_rejected_probs = torch.clamp(ref_rejected_probs, min=eps, max=1-eps)
    
    # Convert probabilities to log probabilities (with stability)
    policy_chosen_logprobs = torch.log(policy_chosen_probs).sum(dim=-1)
    policy_rejected_logprobs = torch.log(policy_rejected_probs).sum(dim=-1)
    ref_chosen_logprobs = torch.log(ref_chosen_probs).sum(dim=-1)
    ref_rejected_logprobs = torch.log(ref_rejected_probs).sum(dim=-1)
    
    # Compute DPO loss
    pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
    
    base_loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
    
    # Add emotional arc preservation term
    # Penalize large changes in emotional state distributions
    # Use clamped probabilities for KL divergence
    emotional_consistency = F.kl_div(
        torch.log(policy_chosen_probs),
        torch.log(ref_chosen_probs),
        log_target=True,
        reduction='batchmean'
    )
    
    total_loss = base_loss + emotional_arc_weight * emotional_consistency
    
    return total_loss


def compute_memory_dpo_loss(
    policy_chosen_embedding: torch.Tensor,
    policy_rejected_embedding: torch.Tensor,
    ref_chosen_embedding: torch.Tensor,
    ref_rejected_embedding: torch.Tensor,
    policy_chosen_metadata: torch.Tensor,
    policy_rejected_metadata: torch.Tensor,
    ref_chosen_metadata: torch.Tensor,
    ref_rejected_metadata: torch.Tensor,
    beta: float = 0.1,
    embedding_weight: float = 0.7,
    metadata_weight: float = 0.3
) -> torch.Tensor:
    """
    Compute DPO loss for memory head.
    
    Handles both embedding vectors and metadata with proper weighting.
    """
    # Compute embedding similarities
    policy_chosen_sim = F.cosine_similarity(policy_chosen_embedding, ref_chosen_embedding, dim=-1)
    policy_rejected_sim = F.cosine_similarity(policy_rejected_embedding, ref_rejected_embedding, dim=-1)
    ref_chosen_sim = torch.ones_like(policy_chosen_sim)  # Perfect similarity for reference
    ref_rejected_sim = F.cosine_similarity(ref_rejected_embedding, ref_chosen_embedding, dim=-1)
    
    # Convert similarities to log probabilities
    policy_chosen_logprobs = torch.log(0.5 * (policy_chosen_sim + 1) + 1e-8)
    policy_rejected_logprobs = torch.log(0.5 * (policy_rejected_sim + 1) + 1e-8)
    ref_chosen_logprobs = torch.log(0.5 * (ref_chosen_sim + 1) + 1e-8)
    ref_rejected_logprobs = torch.log(0.5 * (ref_rejected_sim + 1) + 1e-8)
    
    # Compute embedding DPO loss
    pi_logratios_emb = policy_chosen_logprobs - policy_rejected_logprobs
    ref_logratios_emb = ref_chosen_logprobs - ref_rejected_logprobs
    embedding_loss = -F.logsigmoid(beta * (pi_logratios_emb - ref_logratios_emb)).mean()
    
    # Compute metadata loss (MSE-based)
    metadata_chosen_loss = F.mse_loss(policy_chosen_metadata, ref_chosen_metadata)
    metadata_rejected_loss = F.mse_loss(policy_rejected_metadata, ref_rejected_metadata)
    metadata_loss = metadata_chosen_loss - metadata_rejected_loss  # Prefer chosen
    
    # Combine losses
    total_loss = embedding_weight * embedding_loss + metadata_weight * metadata_loss
    
    return total_loss


def compute_cross_head_regularization(
    generation_loss: torch.Tensor,
    control_loss: torch.Tensor,
    memory_loss: torch.Tensor,
    target_balance: float = 1.0
) -> torch.Tensor:
    """
    Compute regularization to prevent one head from dominating.
    
    Encourages balanced improvement across all heads.
    """
    losses = torch.stack([generation_loss, control_loss, memory_loss])
    mean_loss = losses.mean()
    
    # Penalize deviation from mean
    variance = ((losses - mean_loss) ** 2).mean()
    
    # Also penalize if total loss is too high
    magnitude_penalty = torch.relu(mean_loss - target_balance)
    
    reg_loss = variance + 0.1 * magnitude_penalty
    
    return reg_loss


def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, 
                    padding_value: int = -100) -> torch.Tensor:
    """Get log probabilities for a batch of sequences."""
    # Shift labels for autoregressive loss
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    # Mask padding
    mask = (labels != padding_value)
    
    # Get per-token log probabilities
    per_token_logps = torch.gather(
        logits.log_softmax(-1), 
        dim=2, 
        index=labels.unsqueeze(2)
    ).squeeze(2)
    
    # Average over non-padding tokens
    return (per_token_logps * mask).sum(-1) / mask.sum(-1)


class GenerationDPOTrainer(Trainer):
    """DPO trainer for generation head"""
    
    def __init__(self, model: NarrativeLLM, config: TripleHeadDPOConfig, 
                 ref_model: Optional[NarrativeLLM] = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.config = config
        self.ref_model = ref_model or model  # Use same model if no reference provided
        self.beta = config.beta
    
    def compute_dpo_loss(self, model_outputs: Dict[str, torch.Tensor],
                        ref_outputs: Dict[str, torch.Tensor],
                        labels: torch.Tensor) -> torch.Tensor:
        """Compute DPO loss for generation head"""
        # Split chosen and rejected samples (assuming paired in batch)
        batch_size = model_outputs['text_logits'].shape[0] // 2
        
        policy_chosen_logits = model_outputs['text_logits'][:batch_size]
        policy_rejected_logits = model_outputs['text_logits'][batch_size:]
        ref_chosen_logits = ref_outputs['text_logits'][:batch_size]
        ref_rejected_logits = ref_outputs['text_logits'][batch_size:]
        
        chosen_labels = labels[:batch_size]
        
        return compute_generation_dpo_loss(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            ref_chosen_logits=ref_chosen_logits,
            ref_rejected_logits=ref_rejected_logits,
            labels=chosen_labels,
            beta=self.beta
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss for DPO training"""
        # Get model outputs
        outputs = model(**inputs)
        
        # Get reference model outputs
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
        
        # Compute DPO loss
        loss = self.compute_dpo_loss(outputs, ref_outputs, inputs['labels'])
        
        return (loss, outputs) if return_outputs else loss


class CoordinatedDPOTrainer(Trainer):
    """DPO trainer that optimizes all three heads together"""
    
    def __init__(self, model: NarrativeLLM, config: TripleHeadDPOConfig,
                 ref_model: Optional[NarrativeLLM] = None, **kwargs):
        # Don't pass head-specific weights to base Trainer
        trainer_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['generation_weight', 'control_weight', 
                                     'memory_weight', 'coordination_weight']}
        super().__init__(model=model, **trainer_kwargs)
        self.config = config
        self.ref_model = ref_model or model
        
        # Head weights from config
        self.generation_weight = config.generation_weight
        self.control_weight = config.control_weight
        self.memory_weight = config.memory_weight
        self.coordination_weight = config.coordination_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute coordinated DPO loss across all heads"""
        # Get model outputs
        outputs = model(**inputs)
        
        # Get reference model outputs
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
        
        # Assume batch is organized as [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
        batch_size = outputs['text_logits'].shape[0] // 2
        
        # 1. Generation head loss
        gen_loss = compute_generation_dpo_loss(
            policy_chosen_logits=outputs['text_logits'][:batch_size],
            policy_rejected_logits=outputs['text_logits'][batch_size:],
            ref_chosen_logits=ref_outputs['text_logits'][:batch_size],
            ref_rejected_logits=ref_outputs['text_logits'][batch_size:],
            labels=inputs['labels'][:batch_size],
            beta=self.config.beta
        )
        
        # 2. Control head loss
        control_loss = compute_control_dpo_loss(
            policy_chosen_probs=outputs['control_logits'][:batch_size],
            policy_rejected_probs=outputs['control_logits'][batch_size:],
            ref_chosen_probs=ref_outputs['control_logits'][:batch_size],
            ref_rejected_probs=ref_outputs['control_logits'][batch_size:],
            beta=self.config.beta,
            emotional_arc_weight=self.config.emotional_arc_weight
        )
        
        # 3. Memory head loss
        memory_loss = compute_memory_dpo_loss(
            policy_chosen_embedding=outputs['memory_embedding'][:batch_size],
            policy_rejected_embedding=outputs['memory_embedding'][batch_size:],
            ref_chosen_embedding=ref_outputs['memory_embedding'][:batch_size],
            ref_rejected_embedding=ref_outputs['memory_embedding'][batch_size:],
            policy_chosen_metadata=outputs['memory_metadata'][:batch_size],
            policy_rejected_metadata=outputs['memory_metadata'][batch_size:],
            ref_chosen_metadata=ref_outputs['memory_metadata'][:batch_size],
            ref_rejected_metadata=ref_outputs['memory_metadata'][batch_size:],
            beta=self.config.beta
        )
        
        # 4. Cross-head regularization
        reg_loss = compute_cross_head_regularization(
            generation_loss=gen_loss,
            control_loss=control_loss,
            memory_loss=memory_loss
        )
        
        # Combine losses
        total_loss = (
            self.generation_weight * gen_loss +
            self.control_weight * control_loss +
            self.memory_weight * memory_loss +
            self.coordination_weight * reg_loss
        )
        
        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss/generation": gen_loss.item(),
                "loss/control": control_loss.item(),
                "loss/memory": memory_loss.item(),
                "loss/regularization": reg_loss.item(),
                "loss/total": total_loss.item()
            })
        
        return (total_loss, outputs) if return_outputs else total_loss


def prepare_dpo_dataset(preferences: List[Dict[str, Any]], 
                       tokenizer: PreTrainedTokenizer,
                       max_length: int = 512) -> Dataset:
    """
    Prepare preference data for DPO training.
    
    Converts preference pairs into tokenized format.
    """
    processed_data = []
    
    for pref in preferences:
        prompt = pref['prompt']
        chosen = pref['chosen']
        rejected = pref['rejected']
        
        # Tokenize chosen and rejected
        # In practice, would need proper formatting for the model
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        processed_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'chosen_text': chosen_text,
            'rejected_text': rejected_text
        })
    
    return Dataset.from_list(processed_data)


def load_head_specific_preferences(preference_dir: Path, 
                                 head_type: str) -> List[Dict[str, Any]]:
    """Load preferences for a specific head"""
    preference_file = preference_dir / f"{head_type}_preferences.jsonl"
    
    if not preference_file.exists():
        logger.warning(f"Preference file not found: {preference_file}")
        return []
    
    preferences = []
    with open(preference_file, 'r') as f:
        for line in f:
            if line.strip():
                preferences.append(json.loads(line))
    
    logger.info(f"Loaded {len(preferences)} {head_type} preferences")
    return preferences


def load_model_for_dpo(model_path: str, head_type: str = "all") -> NarrativeLLM:
    """Load model for DPO training"""
    from .config import NarrativeLLMConfig
    
    # Load configuration
    config = NarrativeLLMConfig()
    
    # Create model
    model = NarrativeLLM(config)
    
    # Load checkpoint if path exists
    if Path(model_path).exists():
        # Load adapter or full model
        logger.info(f"Loading model from {model_path}")
        # Implementation would depend on how models are saved
    
    return model 