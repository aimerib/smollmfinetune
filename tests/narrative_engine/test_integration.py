"""
Integration test for Narrative Engine components
"""

import torch
from narrative_engine.model import NarrativeLLM, NarrativeLLMConfig
from narrative_engine.data_pipeline import DatasetProcessor
from narrative_engine.data_schema import DatasetSample, Turn
from narrative_engine.loss import DualHeadLoss
from transformers import AutoTokenizer


def test_dual_head_loss_integration():
    """Test that DualHeadLoss integrates properly with model outputs and data pipeline"""
    
    # Create model
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=256
    )
    model = NarrativeLLM(config)
    tokenizer = model.tokenizer
    
    # Create data processor
    processor = DatasetProcessor(tokenizer)
    
    # Create sample data
    sample = DatasetSample(
        session_id="test_session",
        persona_mix={"friendly": 0.7, "curious": 0.3},
        memory_slots=["Remember user likes cats"],
        turns=[
            Turn(sender="user", text="Tell me about cats", channel="text"),
            Turn(sender="assistant", text="Cats are wonderful pets!", channel="text"),
            Turn(sender="user", text="How do I pet a cat?", channel="text"),
            Turn(sender="assistant", text='{"action": "demonstrate_petting"}', channel="action")
        ]
    )
    
    # Process sample
    processed = processor.process_sample(sample)
    
    # Prepare batch
    batch_size = 1
    input_ids = processed["input_ids"].unsqueeze(0)
    attention_mask = processed["attention_mask"].unsqueeze(0)
    loss_mask = processed["loss_mask"].unsqueeze(0)
    channel_mask = processed["channel_mask"].unsqueeze(0)
    
    # Forward pass through model
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids  # For language modeling
    )
    
    # Extract logits
    text_logits = outputs["text_logits"]
    action_logits = outputs["action_logits"]
    
    # Create loss function
    loss_fn = DualHeadLoss(text_weight=1.0, action_weight=1.5)
    
    # Calculate loss
    labels = input_ids.clone()
    loss = loss_fn(
        text_logits=text_logits,
        action_logits=action_logits,
        labels=labels,
        loss_mask=loss_mask,
        channel_mask=channel_mask
    )
    
    # Verify loss is valid
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0  # Should have some loss
    
    print(f"✅ Integration test passed! Loss: {loss.item():.4f}")


def test_model_with_integrated_loss():
    """Test how to integrate DualHeadLoss directly into the model"""
    
    # Create a modified forward method that uses DualHeadLoss
    class NarrativeLLMWithDualHeadLoss(NarrativeLLM):
        def __init__(self, config, control_tokens_path=None):
            super().__init__(config, control_tokens_path)
            # Add our dual-head loss function
            self.dual_head_loss = DualHeadLoss(
                text_weight=1.0,
                action_weight=1.0
            )
        
        def forward(self, input_ids, attention_mask=None, labels=None,
                   loss_mask=None, channel_mask=None, **kwargs):
            # Get base outputs
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            
            # If we have masks, use our dual-head loss
            if labels is not None and loss_mask is not None and channel_mask is not None:
                dual_loss = self.dual_head_loss(
                    text_logits=outputs["text_logits"],
                    action_logits=outputs["action_logits"],
                    labels=labels,
                    loss_mask=loss_mask,
                    channel_mask=channel_mask
                )
                
                # Replace the losses
                outputs["losses"]["dual_head_loss"] = dual_loss
                outputs["losses"]["total_loss"] = dual_loss
            
            return outputs
    
    # Test the integrated model
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    model = NarrativeLLMWithDualHeadLoss(config)
    
    # Create dummy batch
    batch_size = 2
    seq_len = 20
    vocab_size = model.base_model.config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    loss_mask = torch.ones_like(input_ids)
    channel_mask = torch.randint(0, 2, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        channel_mask=channel_mask
    )
    
    # Check outputs
    assert "dual_head_loss" in outputs["losses"]
    assert outputs["losses"]["total_loss"] == outputs["losses"]["dual_head_loss"]
    assert not torch.isnan(outputs["losses"]["total_loss"])
    
    print(f"✅ Model integration test passed! Total loss: {outputs['losses']['total_loss'].item():.4f}")


if __name__ == "__main__":
    print("🧪 Testing Narrative Engine Integration...")
    test_dual_head_loss_integration()
    test_model_with_integrated_loss()
    print("✨ All integration tests passed!") 