"""
Triple-Head Loss Function for Narrative Engine

This module implements the custom loss function that handles the model's
triple output heads (free-text, action/control tokens, and memory vectors),
correctly routing gradients based on tagged data spans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
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
    
    NOTE: This class is kept for backward compatibility. 
    Use TripleHeadLoss for new implementations.
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


class TripleHeadLoss(nn.Module):
    """
    Custom loss function for triple-head Narrative-LLM.
    
    This loss function:
    - Calculates separate losses for text, control, and memory predictions
    - Uses channel masks to route gradients to the correct head
    - Applies loss masks to ignore irrelevant tokens (e.g., padding, user turns)
    - Combines the losses with configurable weights
    
    The triple-head architecture supports:
    1. Generation Head: Standard language modeling (next-token prediction)
    2. Control Head: Emotional/cognitive control tokens (multi-label classification)
    3. Memory Head: External memory vectors + metadata (Method B)
    """
    
    def __init__(self, 
                 text_weight: float = 1.0, 
                 control_weight: float = 1.0,
                 memory_weight: float = 1.0,
                 memory_embedding_weight: float = 0.7,
                 memory_metadata_weight: float = 0.3,
                 ignore_index: int = -100):
        """
        Initialize the triple-head loss function.
        
        Args:
            text_weight: Weight for text generation loss
            control_weight: Weight for control token loss
            memory_weight: Weight for memory head loss
            memory_embedding_weight: Weight for embedding component within memory loss
            memory_metadata_weight: Weight for metadata component within memory loss
            ignore_index: Label value to ignore in loss calculation
        """
        super().__init__()
        self.text_weight = text_weight
        self.control_weight = control_weight
        self.memory_weight = memory_weight
        self.memory_embedding_weight = memory_embedding_weight
        self.memory_metadata_weight = memory_metadata_weight
        self.ignore_index = ignore_index
        
        # Validate memory component weights sum to 1.0
        total_memory_weight = memory_embedding_weight + memory_metadata_weight
        if abs(total_memory_weight - 1.0) > 1e-6:
            logger.warning(f"Memory component weights sum to {total_memory_weight}, not 1.0. "
                          f"Consider normalizing them.")
        
        logger.info(f"Initialized TripleHeadLoss with text_weight={text_weight}, "
                   f"control_weight={control_weight}, memory_weight={memory_weight}")
    
    def forward(
        self, 
        text_logits: torch.Tensor,
        control_logits: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_metadata: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        control_labels: Optional[torch.Tensor] = None,
        memory_labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss for triple-head outputs.
        
        Args:
            text_logits: Logits from text generation head [batch, seq_len, vocab_size]
            control_logits: Logits from control head [batch, num_control_tokens]
            memory_embedding: Memory embedding vectors [batch, 768]
            memory_metadata: Memory metadata values [batch, 4]
            labels: Ground truth token IDs [batch, seq_len]
            control_labels: Ground truth control tokens [batch, num_control_tokens]
            memory_labels: Ground truth memory vectors [batch, 772] (768 + 4)
            loss_mask: Binary mask for tokens to include in loss [batch, seq_len]
            channel_mask: Channel indicator (0=text, 1=control, 2=memory) [batch, seq_len]
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=text_logits.device, dtype=torch.float32, requires_grad=True)
        
        # 1. Text Generation Loss
        if labels is not None:
            if loss_mask is not None and channel_mask is not None:
                # Use sophisticated masking for text channel only
                text_mask = (channel_mask == 0).float() * loss_mask
                text_token_count = (text_mask > 0).sum()
                
                if text_token_count > 0:
                    # Prepare text labels
                    text_labels = labels.clone()
                    text_labels[text_mask == 0] = self.ignore_index
                    
                    # Compute text loss
                    text_loss = F.cross_entropy(
                        text_logits.reshape(-1, text_logits.shape[-1]),
                        text_labels.reshape(-1),
                        ignore_index=self.ignore_index,
                        reduction='mean'
                    )
                    losses['text_loss'] = text_loss
                    total_loss = total_loss + self.text_weight * text_loss
            else:
                # Standard next-token prediction loss
                text_loss = F.cross_entropy(
                    text_logits.view(-1, text_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.ignore_index,
                    reduction='mean'
                )
                losses['text_loss'] = text_loss
                total_loss = total_loss + self.text_weight * text_loss
        
        # 2. Control Token Loss
        if control_labels is not None:
            # Multi-label binary classification loss
            control_loss = F.binary_cross_entropy(
                control_logits,
                control_labels.float()
            )
            losses['control_loss'] = control_loss
            total_loss = total_loss + self.control_weight * control_loss
        
        # 3. Memory Head Loss (Method B)
        if memory_labels is not None:
            # Split memory labels: [batch, 772] -> [batch, 768] + [batch, 4]
            target_embedding = memory_labels[:, :768]
            target_metadata = memory_labels[:, 768:]
            
            # Embedding loss: cosine similarity loss
            embedding_loss = 1 - F.cosine_similarity(memory_embedding, target_embedding, dim=1).mean()
            
            # Metadata loss: MSE loss
            metadata_loss = F.mse_loss(memory_metadata, target_metadata)
            
            # Combined memory loss with component weights
            memory_loss = (
                self.memory_embedding_weight * embedding_loss + 
                self.memory_metadata_weight * metadata_loss
            )
            
            losses['memory_embedding_loss'] = embedding_loss
            losses['memory_metadata_loss'] = metadata_loss
            losses['memory_loss'] = memory_loss
            total_loss = total_loss + self.memory_weight * memory_loss
        
        # Store total loss
        losses['total_loss'] = total_loss
        
        # Log loss components for debugging
        if logger.isEnabledFor(logging.DEBUG):
            debug_info = []
            if 'text_loss' in losses:
                debug_info.append(f"text={losses['text_loss']:.4f}")
            if 'control_loss' in losses:
                debug_info.append(f"control={losses['control_loss']:.4f}")
            if 'memory_loss' in losses:
                debug_info.append(f"memory={losses['memory_loss']:.4f}")
            logger.debug(f"TripleHeadLoss: {', '.join(debug_info)}, total={total_loss:.4f}")
        
        return losses
    
    def compute_per_head_losses(
        self,
        text_logits: torch.Tensor,
        control_logits: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_metadata: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        control_labels: Optional[torch.Tensor] = None,
        memory_labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for each head separately.
        
        This method is useful for analysis and debugging.
        
        Returns:
            Dictionary with separate losses for each head
        """
        return self.forward(
            text_logits=text_logits,
            control_logits=control_logits,
            memory_embedding=memory_embedding,
            memory_metadata=memory_metadata,
            labels=labels,
            control_labels=control_labels,
            memory_labels=memory_labels,
            loss_mask=loss_mask,
            channel_mask=channel_mask
        )


# Alias for backward compatibility
DualHeadLoss = DualHeadLoss  # Keep original dual-head for backward compatibility


class QuadHeadLoss(nn.Module):
    """
    Custom loss function for quad-head Narrative-LLM.
    
    Extends TripleHeadLoss to include speech generation loss:
    1. Generation Head: Standard language modeling (next-token prediction)
    2. Control Head: Emotional/cognitive control tokens (multi-label classification)
    3. Memory Head: External memory vectors + metadata (Method B)
    4. Speech Head: Mel-spectrogram generation (MSE regression)
    """
    
    def __init__(self, 
                 text_weight: float = 1.0, 
                 control_weight: float = 1.0,
                 memory_weight: float = 1.0,
                 speech_weight: float = 0.5,  # NEW: Speech loss weight
                 memory_embedding_weight: float = 0.7,
                 memory_metadata_weight: float = 0.3,
                 use_spectral_loss: bool = False,  # NEW: Enhanced speech loss
                 ignore_index: int = -100):
        """
        Initialize the quad-head loss function.
        
        Args:
            text_weight: Weight for text generation loss
            control_weight: Weight for control token loss
            memory_weight: Weight for memory head loss
            speech_weight: Weight for speech generation loss (NEW)
            memory_embedding_weight: Weight for embedding component within memory loss
            memory_metadata_weight: Weight for metadata component within memory loss
            use_spectral_loss: Whether to use spectral loss for speech (NEW)
            ignore_index: Label value to ignore in loss calculation
        """
        super().__init__()
        self.text_weight = text_weight
        self.control_weight = control_weight
        self.memory_weight = memory_weight
        self.speech_weight = speech_weight  # NEW
        self.memory_embedding_weight = memory_embedding_weight
        self.memory_metadata_weight = memory_metadata_weight
        self.use_spectral_loss = use_spectral_loss  # NEW
        self.ignore_index = ignore_index
        
        # Validate memory component weights sum to 1.0
        total_memory_weight = memory_embedding_weight + memory_metadata_weight
        if abs(total_memory_weight - 1.0) > 1e-6:
            logger.warning(f"Memory component weights sum to {total_memory_weight}, not 1.0. "
                          f"Consider normalizing them.")
        
        logger.info(f"Initialized QuadHeadLoss with text_weight={text_weight}, "
                   f"control_weight={control_weight}, memory_weight={memory_weight}, "
                   f"speech_weight={speech_weight}")
    
    def forward(
        self, 
        text_logits: torch.Tensor,
        control_logits: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_metadata: torch.Tensor,
        speech_logits: torch.Tensor,  # NEW: Speech head output
        labels: Optional[torch.Tensor] = None,
        control_labels: Optional[torch.Tensor] = None,
        memory_labels: Optional[torch.Tensor] = None,
        speech_labels: Optional[torch.Tensor] = None,  # NEW: Speech targets
        loss_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss for quad-head outputs.
        
        Args:
            text_logits: Logits from text generation head [batch, seq_len, vocab_size]
            control_logits: Logits from control head [batch, num_control_tokens]
            memory_embedding: Memory embedding vectors [batch, 768]
            memory_metadata: Memory metadata values [batch, 4]
            speech_logits: Logits from speech head [batch, seq_len, mel_bins] (NEW)
            labels: Ground truth token IDs [batch, seq_len]
            control_labels: Ground truth control tokens [batch, num_control_tokens]
            memory_labels: Ground truth memory vectors [batch, 772] (768 + 4)
            speech_labels: Ground truth mel-spectrograms [batch, seq_len, mel_bins] (NEW)
            loss_mask: Binary mask for tokens to include in loss [batch, seq_len]
            channel_mask: Channel indicator (0=text, 1=control, 2=memory, 3=speech) [batch, seq_len]
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=text_logits.device, dtype=torch.float32, requires_grad=True)
        
        # 1. Text Generation Loss (inherited from TripleHeadLoss)
        if labels is not None:
            if loss_mask is not None:
                # Apply loss mask to exclude certain tokens
                active_loss = loss_mask.view(-1) == 1
                active_logits = text_logits.view(-1, text_logits.size(-1))[active_loss]
                active_labels = labels.view(-1)[active_loss]
                generation_loss = F.cross_entropy(active_logits, active_labels, ignore_index=self.ignore_index)
            else:
                generation_loss = F.cross_entropy(
                    text_logits.view(-1, text_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.ignore_index
                )
            
            losses['generation_loss'] = generation_loss
            total_loss = total_loss + self.text_weight * generation_loss
        
        # 2. Control Loss (inherited from TripleHeadLoss) 
        if control_labels is not None:
            control_loss = F.binary_cross_entropy(
                control_logits,
                control_labels.float()
            )
            losses['control_loss'] = control_loss
            total_loss = total_loss + self.control_weight * control_loss
        
        # 3. Memory Loss (inherited from TripleHeadLoss)
        if memory_labels is not None:
            target_embedding = memory_labels[:, :768]
            target_metadata = memory_labels[:, 768:]
            
            # Cosine similarity loss for embeddings
            embedding_loss = 1 - F.cosine_similarity(memory_embedding, target_embedding).mean()
            
            # MSE loss for metadata
            metadata_loss = F.mse_loss(memory_metadata, target_metadata)
            
            # Combined memory loss
            memory_loss = (self.memory_embedding_weight * embedding_loss + 
                          self.memory_metadata_weight * metadata_loss)
            losses['memory_loss'] = memory_loss
            total_loss = total_loss + self.memory_weight * memory_loss
        
        # 4. Speech Loss (NEW)
        if speech_labels is not None:
            # Basic MSE loss for mel-spectrogram regression
            speech_mse_loss = F.mse_loss(speech_logits, speech_labels)
            
            speech_loss = speech_mse_loss
            
            # Optional spectral loss for better perceptual quality
            if self.use_spectral_loss:
                # Spectral loss - encourage similar frequency characteristics
                speech_fft = torch.fft.fft(speech_logits, dim=-1).abs()
                target_fft = torch.fft.fft(speech_labels, dim=-1).abs()
                spectral_loss = F.mse_loss(speech_fft, target_fft)
                
                speech_loss = speech_mse_loss + 0.1 * spectral_loss
                losses['speech_spectral_loss'] = spectral_loss
            
            losses['speech_loss'] = speech_loss
            total_loss = total_loss + self.speech_weight * speech_loss
        
        # Combined total loss
        losses['total_loss'] = total_loss
        
        return losses 