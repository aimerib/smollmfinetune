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


def test_load_adapter_method():
    """Test that NarrativeLLM has load_adapter method"""
    # Create minimal config for testing
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64,
    )
    
    model = NarrativeLLM(config)
    
    # Test that the method exists
    assert hasattr(model, 'load_adapter'), "Model should have load_adapter method"
    assert callable(getattr(model, 'load_adapter')), "load_adapter should be callable"


def test_set_active_adapter_method():
    """Test that NarrativeLLM has set_active_adapter method"""
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64,
    )
    
    model = NarrativeLLM(config)
    
    # Test that the method exists
    assert hasattr(model, 'set_active_adapter'), "Model should have set_active_adapter method"
    assert callable(getattr(model, 'set_active_adapter')), "set_active_adapter should be callable"


def test_adapter_loading_changes_weights():
    """Test that loading an adapter actually changes the model behavior"""
    import tempfile
    import os
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from safetensors.torch import save_file
    
    # Create minimal config
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64,
        lora_r=4,  # Small rank for testing
        lora_alpha=8,
        lora_dropout=0.0,
    )
    
    model = NarrativeLLM(config)
    
    # Create a test input
    test_input = torch.randint(0, 1000, (1, 10))
    
    # Get output before adapter loading
    with torch.no_grad():
        original_output = model.forward(test_input)
        original_logits = original_output['text_logits'].clone()
    
    # Create a dummy adapter using PEFT
    # First, create a PEFT model to generate adapter weights
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Common LoRA targets
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create a temporary adapter
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize PEFT on the base model to create adapter structure
        temp_peft_model = get_peft_model(model.base_model, peft_config)
        
        # Modify the adapter weights to ensure they have an effect
        # Access the LoRA A and B matrices and set them to non-zero values
        for name, module in temp_peft_model.named_modules():
            if hasattr(module, 'lora_A'):
                for key in module.lora_A.keys():
                    # Initialize with small random values to ensure effect
                    torch.nn.init.normal_(module.lora_A[key].weight, mean=0.0, std=0.1)
                    torch.nn.init.normal_(module.lora_B[key].weight, mean=0.0, std=0.1)
        
        # Save the adapter
        adapter_path = os.path.join(temp_dir, "test_adapter")
        temp_peft_model.save_pretrained(adapter_path)
        
        # Load the adapter into our model
        model.load_adapter(adapter_path, "test_adapter")
        
        # Check that model is now a PEFT model
        assert isinstance(model.base_model, PeftModel), \
            "Model should be wrapped with PeftModel after loading adapter"
        
        # Check that the adapter is loaded
        assert "test_adapter" in model.loaded_adapters, \
            "Adapter should be tracked in loaded_adapters"
        
        # Get output after adapter loading
        with torch.no_grad():
            new_output = model.forward(test_input)
            new_logits = new_output['text_logits']
        
        # The outputs should be different after loading adapter
        # LoRA modifies the effective computation, changing the output
        assert not torch.allclose(original_logits, new_logits, rtol=1e-4), \
            "Model output should change after loading adapter with non-zero weights"


def test_adapter_hot_swapping():
    """Test hot-swapping between multiple adapters"""
    import tempfile
    import os
    from copy import deepcopy
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    
    # Create minimal config
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    
    model = NarrativeLLM(config)
    
    # Create test input
    test_input = torch.randint(0, 1000, (1, 10))
    
    # Get baseline output
    with torch.no_grad():
        baseline_output = model.forward(test_input)
        baseline_logits = baseline_output['text_logits'].clone()
    
    # Create two different adapters with different initializations
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create adapter 1
        peft_model_1 = get_peft_model(deepcopy(model.base_model), peft_config)
        for name, module in peft_model_1.named_modules():
            if hasattr(module, 'lora_A'):
                for key in module.lora_A.keys():
                    torch.nn.init.normal_(module.lora_A[key].weight, mean=0.0, std=0.1)
                    torch.nn.init.normal_(module.lora_B[key].weight, mean=0.0, std=0.1)
        adapter_path_1 = os.path.join(temp_dir, "adapter_1")
        peft_model_1.save_pretrained(adapter_path_1)
        
        # Create adapter 2 with different initialization
        peft_model_2 = get_peft_model(deepcopy(model.base_model), peft_config)
        for name, module in peft_model_2.named_modules():
            if hasattr(module, 'lora_A'):
                for key in module.lora_A.keys():
                    torch.nn.init.normal_(module.lora_A[key].weight, mean=0.5, std=0.1)
                    torch.nn.init.normal_(module.lora_B[key].weight, mean=0.5, std=0.1)
        adapter_path_2 = os.path.join(temp_dir, "adapter_2")
        peft_model_2.save_pretrained(adapter_path_2)
        
        # Load both adapters
        model.load_adapter(adapter_path_1, "adapter_1")
        model.load_adapter(adapter_path_2, "adapter_2")
        
        # Check that both adapters are loaded
        assert len(model.loaded_adapters) == 2, "Should have 2 adapters loaded"
        assert "adapter_1" in model.loaded_adapters
        assert "adapter_2" in model.loaded_adapters
        
        # Test swapping to adapter 1
        model.set_active_adapter("adapter_1")
        assert model.active_adapter == "adapter_1"
        
        with torch.no_grad():
            output_1 = model.forward(test_input)
            logits_1 = output_1['text_logits'].clone()
        
        # Test swapping to adapter 2
        model.set_active_adapter("adapter_2")
        assert model.active_adapter == "adapter_2"
        
        with torch.no_grad():
            output_2 = model.forward(test_input)
            logits_2 = output_2['text_logits'].clone()
        
        # Outputs should be different for different adapters
        assert not torch.allclose(logits_1, logits_2, rtol=1e-4), \
            "Different adapters should produce different outputs"
        
        # Both should be different from baseline
        assert not torch.allclose(baseline_logits, logits_1, rtol=1e-4)
        assert not torch.allclose(baseline_logits, logits_2, rtol=1e-4)
        
        # Test swapping back to adapter 1
        model.set_active_adapter("adapter_1")
        with torch.no_grad():
            output_1_again = model.forward(test_input)
            logits_1_again = output_1_again['text_logits']
        
        # Should get same output as before when using same adapter
        assert torch.allclose(logits_1, logits_1_again, rtol=1e-4), \
            "Same adapter should produce consistent outputs"


def test_adapter_combination():
    """Test combining multiple adapters with weights"""
    import tempfile
    import os
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Create minimal config
    config = NarrativeLLMConfig(
        base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        control_head_dim=64,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    
    model = NarrativeLLM(config)
    
    # This test is a placeholder for adapter combination functionality
    # which would be implemented according to the persona_mix specification
    # For now, just verify the model has the necessary infrastructure
    
    assert hasattr(model, 'loaded_adapters'), "Model should track loaded adapters"
    assert hasattr(model, 'active_adapter'), "Model should track active adapter" 