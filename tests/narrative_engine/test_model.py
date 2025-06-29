import pytest
import torch
from narrative_engine.model import NarrativeLLM, create_narrative_model
from narrative_engine.config import NarrativeLLMConfig


def test_model_imports():
    """Test that NarrativeLLM and NarrativeLLMConfig can be imported"""
    # This test passes if imports don't raise exceptions
    assert NarrativeLLM is not None
    assert NarrativeLLMConfig is not None


def test_model_instantiation():
    """Test that NarrativeLLM can be instantiated with NarrativeLLMConfig"""
    # Create a minimal config
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=128,  # Smaller for testing
        num_hidden_layers=2,   # Smaller for testing
    )
    
    # Should be able to instantiate the model
    model = NarrativeLLM(config)
    
    # Check that model has the required components
    assert hasattr(model, 'base_model')
    assert hasattr(model, 'control_head')
    assert hasattr(model, 'tokenizer')


def test_forward_pass_shapes():
    """Test that forward pass produces outputs of expected shapes"""
    # Create minimal config for testing
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64,   # Small for testing
        num_hidden_layers=1,   # Minimal for testing
    )
    
    model = NarrativeLLM(config)
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    session_id = torch.tensor([42])  # Single session ID
    
    # Forward pass
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        session_id=session_id,
        external_memory_states=None
    )
    
    # Check output structure matches R4-1 requirements
    assert 'text_logits' in outputs
    assert 'action_logits' in outputs
    
    # Check shapes
    text_logits = outputs['text_logits']
    action_logits = outputs['action_logits']
    
    assert text_logits.shape == (batch_size, seq_len, model.base_model.config.vocab_size)
    assert action_logits.shape == (batch_size, len(model.control_tokens))


def test_factory_function():
    """Test that create_narrative_model factory function works"""
    model = create_narrative_model(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64
    )
    
    assert isinstance(model, NarrativeLLM)
    assert model.config.base_model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert model.config.control_head_dim == 64


def test_config_dataclass():
    """Test that NarrativeLLMConfig has all expected fields from R4-1"""
    config = NarrativeLLMConfig()
    
    # Basic R4-1 requirements
    assert hasattr(config, 'vocab_size')
    assert hasattr(config, 'token_dim')
    assert hasattr(config, 'max_position_embeddings')
    assert hasattr(config, 'session_embedding_dim')
    assert hasattr(config, 'num_hidden_layers')
    assert hasattr(config, 'num_experts')
    
    # Advanced C.L.A.R.A. Loop features we added
    assert hasattr(config, 'control_head_dim')
    assert hasattr(config, 'surprise_threshold')
    assert hasattr(config, 'decay_steps') 