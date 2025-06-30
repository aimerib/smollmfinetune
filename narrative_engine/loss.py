"""
Dual-Head Loss Function for Narrative Engine

This module implements the custom loss function that handles the model's
dual output heads (free-text and action JSON), correctly routing gradients
based on tagged data spans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MaskedCrossEntropy(torch.autograd.Function):
    """
    Custom autograd function that applies cross-entropy loss with gradient masking.
    This ensures that gradients are completely blocked (not just zeroed) when mask is 0.
    """
    
    @staticmethod
    def forward(ctx, logits, labels, has_tokens, ignore_index=-100):
        """
        Forward pass: compute cross-entropy loss if has_tokens, else return 0.
        """
        if has_tokens:
            # Normal cross-entropy calculation
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=ignore_index,
                reduction='mean'
            )
            ctx.save_for_backward(torch.tensor(True))
        else:
            # No tokens, return zero loss
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            ctx.save_for_backward(torch.tensor(False))
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: only propagate gradients if we had tokens in forward.
        """
        has_tokens, = ctx.saved_tensors
        
        if has_tokens:
            # Let autograd handle the gradient computation
            return grad_output, None, None, None
        else:
            # Block all gradients
            return None, None, None, None


class DualHeadLoss(nn.Module):
    """
    Custom loss function for dual-head Narrative-LLM.
    
    This loss function:
    - Calculates separate losses for text and action predictions
    - Uses channel masks to route gradients to the correct head
    - Applies loss masks to ignore irrelevant tokens (e.g., padding, user turns)
    - Combines the losses with configurable weights
    
    The key innovation is that it knows which parts of the output correspond
    to conversational text and which correspond to structured tool calls,
    enabling the model to be "bilingual" in prose and actions.
    """
    
    def __init__(self, text_weight: float = 1.0, action_weight: float = 1.0, 
                 ignore_index: int = -100):
        """
        Initialize the dual-head loss function.
        
        Args:
            text_weight: Weight for text generation loss
            action_weight: Weight for action generation loss
            ignore_index: Label value to ignore in loss calculation
        """
        super().__init__()
        self.text_weight = text_weight
        self.action_weight = action_weight
        self.ignore_index = ignore_index
        
        logger.info(f"Initialized DualHeadLoss with text_weight={text_weight}, "
                   f"action_weight={action_weight}")
    
    def forward(
        self, 
        text_logits: torch.Tensor,
        action_logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        channel_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the combined loss for dual-head outputs.
        
        Args:
            text_logits: Logits from text generation head [batch, seq_len, vocab_size]
            action_logits: Logits from action generation head [batch, seq_len, num_actions]
            labels: Ground truth token IDs [batch, seq_len]
            loss_mask: Binary mask for tokens to include in loss [batch, seq_len]
            channel_mask: Channel indicator (0=text, 1=action) [batch, seq_len]
            
        Returns:
            Combined scalar loss value
        """
        # Validate inputs
        batch_size, seq_len = labels.shape
        assert text_logits.shape[:2] == (batch_size, seq_len), \
            f"Text logits shape mismatch: {text_logits.shape} vs labels {labels.shape}"
        assert action_logits.shape[:2] == (batch_size, seq_len), \
            f"Action logits shape mismatch: {action_logits.shape} vs labels {labels.shape}"
        assert loss_mask.shape == labels.shape, \
            f"Loss mask shape mismatch: {loss_mask.shape} vs labels {labels.shape}"
        assert channel_mask.shape == labels.shape, \
            f"Channel mask shape mismatch: {channel_mask.shape} vs labels {labels.shape}"
        
        # Get vocabulary sizes
        text_vocab_size = text_logits.shape[-1]
        action_vocab_size = action_logits.shape[-1]
        
        # Create separate masks for text and action channels
        text_mask = (channel_mask == 0).float() * loss_mask
        action_mask = (channel_mask == 1).float() * loss_mask
        
        # Count tokens for each channel
        text_token_count = (text_mask > 0).sum()
        action_token_count = (action_mask > 0).sum()
        
        # Initialize losses
        total_loss = torch.tensor(0.0, device=text_logits.device, dtype=torch.float32, requires_grad=True)
        
        # Compute text loss only if there are text tokens
        if text_token_count > 0:
            # Prepare text labels
            text_labels = labels.clone()
            text_labels[text_mask == 0] = self.ignore_index
            
            # Compute text loss
            text_loss = F.cross_entropy(
                text_logits.reshape(-1, text_vocab_size),
                text_labels.reshape(-1),
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            
            # Add weighted text loss
            total_loss = total_loss + self.text_weight * text_loss
        else:
            # No text tokens, detach to prevent gradient flow
            text_logits = text_logits.detach()
        
        # Compute action loss only if there are action tokens
        if action_token_count > 0:
            # Prepare action labels
            action_labels = labels.clone()
            action_labels[action_mask == 0] = self.ignore_index
            
            # Ensure action labels are within valid range
            action_labels = torch.where(
                action_labels != self.ignore_index,
                action_labels.clamp(max=action_vocab_size - 1),
                action_labels
            )
            
            # Compute action loss
            action_loss = F.cross_entropy(
                action_logits.reshape(-1, action_vocab_size),
                action_labels.reshape(-1),
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            
            # Add weighted action loss
            total_loss = total_loss + self.action_weight * action_loss
        else:
            # No action tokens, detach to prevent gradient flow
            action_logits = action_logits.detach()
        
        # Log loss components for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Text tokens: {text_token_count}, Action tokens: {action_token_count}")
            logger.debug(f"Total loss: {total_loss:.4f}")
        
        return total_loss
    
    def compute_per_channel_losses(
        self,
        text_logits: torch.Tensor,
        action_logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        channel_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute separate losses for text and action channels.
        
        This method is useful for analysis and debugging.
        
        Returns:
            Tuple of (text_loss, action_loss)
        """
        # Create separate masks
        text_mask = (channel_mask == 0).float() * loss_mask
        action_mask = (channel_mask == 1).float() * loss_mask
        
        # Prepare labels
        text_labels = labels.clone()
        text_labels[text_mask == 0] = self.ignore_index
        
        action_labels = labels.clone()
        action_labels[action_mask == 0] = self.ignore_index
        
        # Ensure action labels are within valid range
        action_vocab_size = action_logits.shape[-1]
        action_labels = torch.where(
            action_labels != self.ignore_index,
            action_labels.clamp(max=action_vocab_size - 1),
            action_labels
        )
        
        # Calculate losses
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.shape[-1]),
            text_labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        action_loss = F.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            action_labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return text_loss, action_loss 