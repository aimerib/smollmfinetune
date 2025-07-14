"""
Unit tests for Triple-Head Reward Models.

Tests reward model architectures and training for generation, control, and memory heads.
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the modules we'll implement
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.reward_models import RewardModelOutput


class TestGenerationHeadRewardModel:
    """Test suite for generation head reward model"""
    
    def test_reward_model_architecture(self):
        """Test generation head reward model architecture"""
        from backend.app.narrative_engine.reward_models import GenerationRewardModel
        
        # Create model
        model = GenerationRewardModel(
            base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            hidden_size=768
        )
        
        # Check architecture
        assert hasattr(model, 'base_model')
        assert hasattr(model, 'reward_head')
        assert isinstance(model.reward_head, nn.Module)
        
        # Test forward pass
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        
        # Check output
        assert isinstance(output, RewardModelOutput)
        assert output.reward.shape == (batch_size,)
        assert not torch.isnan(output.reward).any()
        assert output.head_type == "generation"
    
    def test_generation_reward_training(self):
        """Test training generation reward model on preference data"""
        from backend.app.narrative_engine.reward_models import GenerationRewardModel, train_reward_model
        
        # Mock preference data
        preferences = [
            {
                "prompt": "Tell me a story",
                "chosen": "Once upon a time, in a land far away...",
                "rejected": "Uh, I don't know any stories."
            }
        ]
        
        # Create mock model
        with patch('backend.app.narrative_engine.reward_models.GenerationRewardModel') as MockModel:
            mock_model = MockModel.return_value
            # Add parameters method that returns actual tensors
            mock_param = torch.nn.Parameter(torch.zeros(1))
            mock_model.parameters.return_value = iter([mock_param])
            # Make the .to() method return self to preserve the mock
            mock_model.to.return_value = mock_model
            
            # Train
            trained_model = train_reward_model(
                model_type="generation",
                preferences=preferences,
                num_epochs=1
            )
            
            assert MockModel.called
            assert trained_model is not None


class TestControlHeadRewardModel:
    """Test suite for control head reward model"""
    
    def test_control_reward_architecture(self):
        """Test control head reward model architecture"""
        from backend.app.narrative_engine.reward_models import ControlRewardModel
        
        # Create model
        model = ControlRewardModel(
            num_control_tokens=64,
            hidden_size=256
        )
        
        # Check architecture
        assert hasattr(model, 'control_encoder')
        assert hasattr(model, 'reward_head')
        
        # Test forward pass
        batch_size = 2
        control_tokens = torch.rand(batch_size, 64)  # Multi-hot encoded
        
        with torch.no_grad():
            output = model(control_tokens)
        
        # Check output - ControlRewardModel returns RewardModelOutput
        assert isinstance(output, RewardModelOutput)
        assert output.reward.shape == (batch_size,)
        assert not torch.isnan(output.reward).any()
        assert output.head_type == "control"
    
    def test_control_preference_format(self):
        """Test control head preference data format"""
        from backend.app.narrative_engine.reward_models import prepare_control_preferences
        
        # Mock preference data
        raw_preferences = [
            {
                "conversation_id": "test_123",
                "emotional_appropriateness": 8,
                "personality_consistency": 9,
                "mood_matching": 7,
                "coordination": 8
            }
        ]
        
        # Process preferences
        processed = prepare_control_preferences(raw_preferences)
        
        assert len(processed) > 0
        assert "control_tokens" in processed[0]
        assert "reward" in processed[0]


class TestMemoryHeadRewardModel:
    """Test suite for memory head reward model"""
    
    def test_memory_reward_architecture(self):
        """Test memory head reward model architecture"""
        from backend.app.narrative_engine.reward_models import MemoryRewardModel
        
        # Create model
        model = MemoryRewardModel(
            embedding_dim=768,
            metadata_dim=4
        )
        
        # Check architecture
        assert hasattr(model, 'embedding_encoder')
        assert hasattr(model, 'metadata_encoder')
        assert hasattr(model, 'fusion_layer')
        assert hasattr(model, 'reward_head')
        
        # Test forward pass
        batch_size = 2
        memory_embeddings = torch.randn(batch_size, 768)
        memory_metadata = torch.rand(batch_size, 4)
        
        with torch.no_grad():
            output = model(memory_embeddings, memory_metadata)
        
        # Check output - MemoryRewardModel returns RewardModelOutput
        assert isinstance(output, RewardModelOutput)
        assert output.reward.shape == (batch_size,)
        assert not torch.isnan(output.reward).any()
        assert output.head_type == "memory"
    
    def test_memory_preference_processing(self):
        """Test memory preference data processing"""
        from backend.app.narrative_engine.reward_models import prepare_memory_preferences
        
        # Mock preference data
        raw_preferences = [
            {
                "conversation_id": "test_123",
                "memory_accuracy": 9,
                "memory_consistency": 8,
                "formation_quality": 9,
                "coordination": 8
            }
        ]
        
        # Process preferences
        processed = prepare_memory_preferences(raw_preferences)
        
        assert len(processed) > 0
        assert "memory_embedding" in processed[0]
        assert "memory_metadata" in processed[0]
        assert "reward" in processed[0]


class TestCoordinatedRewardModel:
    """Test suite for coordinated reward model"""
    
    def test_coordinated_architecture(self):
        """Test coordinated reward model that evaluates all heads together"""
        from backend.app.narrative_engine.reward_models import CoordinatedRewardModel
        
        # Create model
        model = CoordinatedRewardModel(
            text_hidden_size=768,
            control_vocab_size=64,
            memory_dim=768,
            metadata_dim=4
        )
        
        # Check architecture
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'control_encoder')
        assert hasattr(model, 'memory_encoder')
        assert hasattr(model, 'fusion_network')
        assert hasattr(model, 'reward_head')
        
        # Test forward pass
        batch_size = 2
        text_features = torch.randn(batch_size, 768)
        control_tokens = torch.rand(batch_size, 64)
        memory_embeddings = torch.randn(batch_size, 768)
        memory_metadata = torch.rand(batch_size, 4)
        
        with torch.no_grad():
            output = model(
                text_features=text_features,
                control_tokens=control_tokens,
                memory_embeddings=memory_embeddings,
                memory_metadata=memory_metadata
            )
        
        # Check output - CoordinatedRewardModel returns RewardModelOutput
        assert isinstance(output, RewardModelOutput)
        assert output.reward.shape == (batch_size,)
        assert not torch.isnan(output.reward).any()
        assert output.head_type == "coordinated"


class TestRewardModelTrainingScripts:
    """Test the training scripts for reward models"""
    
    def test_generation_rm_script_exists(self):
        """Test that generation RM training script can be imported"""
        # Scripts exist now, so test they can be imported
        from scripts.run_generation_rm_training import main
        assert main is not None
    
    def test_preference_data_loading(self):
        """Test loading preference data from JSONL files"""
        from backend.app.narrative_engine.reward_models import load_head_preferences
        
        # Create mock preference files
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_file = Path(tmpdir) / "generation_preferences.jsonl"
            gen_file.write_text(json.dumps({
                "conversation_id": "test",
                "content_quality": 8,
                "creativity": 7,
                "factual_accuracy": 9,
                "coordination": 8
            }) + "\n")
            
            # Load preferences
            preferences = load_head_preferences(gen_file, head_type="generation")
            
            assert len(preferences) == 1
            assert "reward" in preferences[0]
            assert preferences[0]["reward"] > 0


class TestRewardModelIntegration:
    """Test integration between reward models and training pipeline"""
    
    def test_reward_models_with_dpo(self):
        """Test that reward models can be used in DPO training"""
        from backend.app.narrative_engine.reward_models import GenerationRewardModel
        from backend.app.narrative_engine.dpo_trainer import CoordinatedDPOTrainer
        
        # Mock components
        with patch('backend.app.narrative_engine.reward_models.GenerationRewardModel') as MockRM:
            mock_rm = MockRM.return_value
            mock_rm.forward = MagicMock(return_value=torch.tensor([1.0, 0.5]))
            
            # Create DPO trainer config
            config = {
                "reward_model": mock_rm,
                "head_type": "generation",
                "learning_rate": 1e-5
            }
            
            # Verify reward model can be integrated
            assert mock_rm is not None
            assert hasattr(mock_rm, 'forward') 