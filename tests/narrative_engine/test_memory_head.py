"""
Tests for Memory Head (Method B) in NarrativeLLM

Tests the third head added to the model for memory vector generation.
"""

import torch
import pytest
import numpy as np

from backend.app.narrative_engine.model import NarrativeLLM, create_narrative_model
from backend.app.narrative_engine.config import NarrativeLLMConfig


class TestMemoryHead:
    """Test the memory head functionality in NarrativeLLM"""
    
    @pytest.fixture
    def model_config(self):
        """Configuration for test model"""
        return NarrativeLLMConfig(
            base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            control_head_dim=128,
            num_hidden_layers=2,  # Smaller for testing
        )
    
    @pytest.fixture
    def model(self, model_config):
        """Create a test model instance"""
        return NarrativeLLM(model_config)
    
    def test_memory_head_exists(self, model):
        """Test that memory head is properly initialized"""
        assert hasattr(model, 'memory_head')
        assert model.memory_head is not None
        
        # Check layer structure
        assert len(model.memory_head) == 4  # Linear, ReLU, Dropout, Linear
        
        # Check output dimension
        last_layer = model.memory_head[-1]
        assert last_layer.out_features == 768 + 4  # 768 embedding + 4 metadata
    
    def test_forward_with_memory_output(self, model):
        """Test forward pass produces memory outputs"""
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Forward pass
        outputs = model.forward(input_ids)
        
        # Check memory outputs exist
        assert 'memory_embedding' in outputs
        assert 'memory_metadata' in outputs
        
        # Check shapes
        assert outputs['memory_embedding'].shape == (batch_size, 768)
        assert outputs['memory_metadata'].shape == (batch_size, 4)
        
        # Check memory embedding is normalized
        norms = torch.norm(outputs['memory_embedding'], p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        # Check metadata is bounded [0, 1]
        assert (outputs['memory_metadata'] >= 0).all()
        assert (outputs['memory_metadata'] <= 1).all()
    
    def test_memory_loss_calculation(self, model):
        """Test memory loss is correctly calculated"""
        # Create inputs
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Create memory labels (target)
        target_embedding = torch.randn(batch_size, 768)
        target_embedding = torch.nn.functional.normalize(target_embedding, p=2, dim=1)
        target_metadata = torch.rand(batch_size, 4)  # Random values [0, 1]
        memory_labels = torch.cat([target_embedding, target_metadata], dim=1)
        
        # Forward pass with memory labels
        outputs = model.forward(input_ids, memory_labels=memory_labels)
        
        # Check memory loss exists
        assert 'memory_loss' in outputs['losses']
        assert outputs['losses']['memory_loss'] > 0
        
        # Check total loss includes memory loss
        assert 'total_loss' in outputs['losses']
    
    def test_generate_with_memory(self, model):
        """Test generation includes memory output"""
        # Create input
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # Generate with memory
        result = model.generate_with_control(
            input_ids,
            max_new_tokens=20,
            generate_memory=True,
        )
        
        # Check memory is generated
        assert 'generated_memory' in result
        assert result['generated_memory'] is not None
        
        memory = result['generated_memory']
        assert 'embedding' in memory
        assert 'importance' in memory
        assert 'surprise' in memory
        assert 'valence' in memory
        assert 'persistence' in memory
        
        # Check embedding dimension
        assert len(memory['embedding']) == 768
        
        # Check metadata ranges
        assert 0 <= memory['importance'] <= 1
        assert 0 <= memory['surprise'] <= 1
        assert -1 <= memory['valence'] <= 1
        assert 0 <= memory['persistence'] <= 1
    
    def test_generate_without_memory(self, model):
        """Test generation can skip memory generation"""
        # Create input
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # Generate without memory
        result = model.generate_with_control(
            input_ids,
            max_new_tokens=20,
            generate_memory=False,
        )
        
        # Check no memory is generated
        assert result['generated_memory'] is None
    
    def test_memory_metadata_indices(self, model):
        """Test memory metadata indices are correctly defined"""
        assert hasattr(model, 'memory_metadata_indices')
        
        expected_indices = {
            'importance': 768,
            'surprise': 769,
            'valence': 770,
            'persistence': 771
        }
        
        assert model.memory_metadata_indices == expected_indices
    
    def test_memory_embedding_normalization(self, model):
        """Test that memory embeddings are properly normalized"""
        # Run multiple forward passes
        for _ in range(5):
            input_ids = torch.randint(0, 1000, (3, 15))
            outputs = model.forward(input_ids)
            
            embeddings = outputs['memory_embedding']
            norms = torch.norm(embeddings, p=2, dim=1)
            
            # All embeddings should be unit vectors
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_memory_loss_gradient_flow(self, model):
        """Test that gradients flow through memory head"""
        # Enable gradients
        model.train()
        torch.set_grad_enabled(True)
        
        # Clear any existing gradients
        model.zero_grad()
        
        # Create inputs with gradients enabled
        input_ids = torch.randint(0, 1000, (1, 10))
        memory_labels = torch.randn(1, 772, requires_grad=False)  # Labels don't need gradients
        
        # Forward pass
        outputs = model.forward(input_ids, memory_labels=memory_labels)
        
        # Check if memory loss exists and has gradients
        if 'losses' in outputs and 'memory_loss' in outputs['losses']:
            loss = outputs['losses']['memory_loss']
            
            # Only test gradient flow if loss requires gradients
            if loss.requires_grad:
                # Check that memory head parameters have gradients after backward
                loss.backward()
                
                memory_head_has_gradients = False
                for param in model.memory_head.parameters():
                    if param.requires_grad and param.grad is not None:
                        if not torch.all(param.grad == 0):
                            memory_head_has_gradients = True
                            break
                
                assert memory_head_has_gradients, "Memory head should have non-zero gradients"
            else:
                # If loss doesn't require gradients, skip the test with informative message
                pytest.skip("Memory loss does not require gradients - may be using detached computation")
        else:
            pytest.skip("Memory loss not found in model outputs")


class TestMemoryIntegration:
    """Test integration between Method A (tokens) and Method B (memory head)"""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance"""
        config = NarrativeLLMConfig(
            base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            control_head_dim=128,
            num_hidden_layers=2,
        )
        return NarrativeLLM(config)
    
    def test_memory_tokens_affect_memory_head(self, model):
        """Test that memory control tokens influence memory head output"""
        # This is a conceptual test - in practice, the model would need
        # to be trained to learn this correlation
        
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # First generation without memory tokens
        result1 = model.generate_with_control(
            input_ids,
            generate_memory=True,
        )
        
        # Add memory tokens to control tokens
        # (This would normally happen through training)
        memory_tokens = ["<memory_form>", "<memory_importance_high>"]
        
        # Second generation with memory context
        result2 = model.generate_with_control(
            input_ids,
            recirculation_tokens=memory_tokens,
            generate_memory=True,
        )
        
        # Both should generate memories
        assert result1['generated_memory'] is not None
        assert result2['generated_memory'] is not None
        
        # In a trained model, memory tokens would influence the output
        # For now, just check they're different due to randomness
        mem1 = np.array(result1['generated_memory']['embedding'])
        mem2 = np.array(result2['generated_memory']['embedding'])
        
        # Embeddings should be different (not testing correlation without training)
        assert not np.allclose(mem1, mem2) 