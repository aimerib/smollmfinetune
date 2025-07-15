"""
Test suite for Narrative Engine Dual-Head Loss Function
"""

import torch
import torch.nn as nn
import pytest
from backend.app.narrative_engine.loss import DualHeadLoss


class TestDualHeadLoss:
    """Test cases for the dual-head loss function"""
    
    def test_dual_head_loss_calculation(self):
        """Test that DualHeadLoss returns a single scalar tensor"""
        # Create dummy data
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        num_control_tokens = 50
        
        # Text logits for standard generation head
        text_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Action logits for control head
        action_logits = torch.randn(batch_size, seq_len, num_control_tokens)
        
        # Ground truth labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Loss mask (1 for tokens to include in loss, 0 to ignore)
        loss_mask = torch.ones(batch_size, seq_len)
        
        # Channel mask (0 for text, 1 for action)
        channel_mask = torch.zeros(batch_size, seq_len)
        channel_mask[:, 5:] = 1  # Second half is action channel
        
        # Instantiate loss function
        loss_fn = DualHeadLoss()
        
        # Calculate loss
        loss = loss_fn(text_logits, action_logits, labels, loss_mask, channel_mask)
        
        # Assert output is a single scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.dtype == torch.float32
    
    def test_loss_masking_logic(self):
        """Test that loss correctly applies masking for different scenarios"""
        batch_size = 1
        seq_len = 10
        vocab_size = 100
        num_control_tokens = 50
        
        # Initialize loss function
        loss_fn = DualHeadLoss()
        
        # Scenario 1: Action logits perfect, text logits wrong
        # Create perfect action predictions (one-hot matches labels)
        action_labels = torch.randint(0, num_control_tokens, (batch_size, seq_len))
        action_logits = torch.full((batch_size, seq_len, num_control_tokens), -10.0)
        for b in range(batch_size):
            for t in range(seq_len):
                action_logits[b, t, action_labels[b, t]] = 10.0
        
        # Create wrong text predictions
        text_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        text_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # All tokens are included in loss
        loss_mask = torch.ones(batch_size, seq_len)
        
        # First half text, second half action
        channel_mask = torch.zeros(batch_size, seq_len)
        channel_mask[:, seq_len//2:] = 1
        
        # For text tokens, use text labels; for action tokens, use action labels
        combined_labels = text_labels.clone()
        combined_labels[:, seq_len//2:] = action_labels[:, seq_len//2:]
        
        loss1 = loss_fn(text_logits, action_logits, combined_labels, loss_mask, channel_mask)
        assert loss1 > 0, "Loss should be > 0 when text predictions are wrong"
        
        # Scenario 2: Text logits perfect, action logits wrong
        # Create perfect text predictions
        text_logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        for b in range(batch_size):
            for t in range(seq_len):
                text_logits[b, t, text_labels[b, t]] = 10.0
        
        # Create wrong action predictions
        action_logits = torch.randn(batch_size, seq_len, num_control_tokens)
        
        loss2 = loss_fn(text_logits, action_logits, combined_labels, loss_mask, channel_mask)
        assert loss2 > 0, "Loss should be > 0 when action predictions are wrong"
        
        # Scenario 3: Both perfect
        # Perfect text predictions
        text_logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        for b in range(batch_size):
            for t in range(seq_len):
                text_logits[b, t, text_labels[b, t]] = 10.0
        
        # Perfect action predictions
        action_logits = torch.full((batch_size, seq_len, num_control_tokens), -10.0)
        for b in range(batch_size):
            for t in range(seq_len):
                action_logits[b, t, action_labels[b, t]] = 10.0
        
        loss3 = loss_fn(text_logits, action_logits, combined_labels, loss_mask, channel_mask)
        assert loss3 < 0.01, f"Loss should be near 0 when both predictions are perfect, got {loss3}"
    
    def test_loss_mask_exclusion(self):
        """Test that tokens with loss_mask=0 are excluded from loss calculation"""
        batch_size = 1
        seq_len = 10
        vocab_size = 100
        num_control_tokens = 50
        
        # Initialize loss function
        loss_fn = DualHeadLoss()
        
        # Create specific predictions to test masking
        # Make predictions that will have high loss
        text_logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        action_logits = torch.full((batch_size, seq_len, num_control_tokens), -10.0)
        
        # Set specific wrong predictions (not matching labels)
        for i in range(seq_len):
            text_logits[0, i, 0] = 10.0  # Predict class 0 for all positions
        
        # Create labels that don't match predictions
        labels = torch.full((batch_size, seq_len), 1)  # All labels are class 1
        
        # All text channel
        channel_mask = torch.zeros(batch_size, seq_len)
        
        # Test 1: Loss with no masked tokens (all ones)
        loss_mask_none = torch.zeros(batch_size, seq_len)
        loss_none = loss_fn(text_logits, action_logits, labels, loss_mask_none, channel_mask)
        assert loss_none == 0.0, "Loss should be 0 when all tokens are masked"
        
        # Test 2: Loss with some tokens
        loss_mask_some = torch.zeros(batch_size, seq_len)
        loss_mask_some[0, :3] = 1  # Only first 3 tokens contribute
        loss_some = loss_fn(text_logits, action_logits, labels, loss_mask_some, channel_mask)
        assert loss_some > 0, "Loss should be positive with some unmasked tokens"
        
        # Test 3: Loss with all tokens  
        loss_mask_all = torch.ones(batch_size, seq_len)
        loss_all = loss_fn(text_logits, action_logits, labels, loss_mask_all, channel_mask)
        assert loss_all > 0, "Loss should be positive with all tokens unmasked"
        
        # The average loss per token should be similar (since we use mean reduction)
        # but having more tokens means more contributions to gradient
    
    def test_channel_routing(self):
        """Test that channel mask correctly routes gradients to appropriate head"""
        batch_size = 2
        seq_len = 8
        vocab_size = 100
        num_control_tokens = 50
        
        # Initialize loss function with specific weights
        loss_fn = DualHeadLoss(text_weight=1.0, action_weight=2.0)
        
        # Create logits
        text_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        action_logits = torch.randn(batch_size, seq_len, num_control_tokens, requires_grad=True)
        
        # Labels
        labels = torch.randint(0, min(vocab_size, num_control_tokens), (batch_size, seq_len))
        
        # All tokens included
        loss_mask = torch.ones(batch_size, seq_len)
        
        # Test 1: All text channel
        channel_mask = torch.zeros(batch_size, seq_len)
        loss = loss_fn(text_logits, action_logits, labels, loss_mask, channel_mask)
        loss.backward()
        
        # Text logits should have gradients, action logits should not
        assert text_logits.grad is not None and text_logits.grad.abs().sum() > 0
        # Action logits should either have no grad or zero grad (due to detachment)
        assert action_logits.grad is None or action_logits.grad.abs().sum() == 0
        
        # Reset gradients
        if text_logits.grad is not None:
            text_logits.grad.zero_()
        if action_logits.grad is not None:
            action_logits.grad.zero_()
        
        # Test 2: All action channel
        channel_mask = torch.ones(batch_size, seq_len)
        loss = loss_fn(text_logits, action_logits, labels, loss_mask, channel_mask)
        loss.backward()
        
        # Action logits should have gradients, text logits should not
        assert action_logits.grad is not None and action_logits.grad.abs().sum() > 0
        # Text logits should either have no grad or zero grad (due to detachment)
        assert text_logits.grad is None or text_logits.grad.abs().sum() == 0
    
    def test_weighted_loss_combination(self):
        """Test that text and action losses are properly weighted"""
        batch_size = 1
        seq_len = 10
        vocab_size = 100
        num_control_tokens = 50
        
        # Create data
        text_logits = torch.randn(batch_size, seq_len, vocab_size)
        action_logits = torch.randn(batch_size, seq_len, num_control_tokens)
        labels = torch.randint(0, min(vocab_size, num_control_tokens), (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        
        # Half text, half action
        channel_mask = torch.zeros(batch_size, seq_len)
        channel_mask[:, seq_len//2:] = 1
        
        # Test different weight combinations
        loss_fn_equal = DualHeadLoss(text_weight=1.0, action_weight=1.0)
        loss_fn_text_heavy = DualHeadLoss(text_weight=2.0, action_weight=1.0)
        loss_fn_action_heavy = DualHeadLoss(text_weight=1.0, action_weight=2.0)
        
        loss_equal = loss_fn_equal(text_logits, action_logits, labels, loss_mask, channel_mask)
        loss_text = loss_fn_text_heavy(text_logits, action_logits, labels, loss_mask, channel_mask)
        loss_action = loss_fn_action_heavy(text_logits, action_logits, labels, loss_mask, channel_mask)
        
        # Different weights should produce different losses
        assert not torch.allclose(loss_equal, loss_text), "Text-weighted loss should differ"
        assert not torch.allclose(loss_equal, loss_action), "Action-weighted loss should differ"
        assert not torch.allclose(loss_text, loss_action), "Different weights should produce different losses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 