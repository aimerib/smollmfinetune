"""
Unit tests for Triple-Head DPO Training.

Tests DPO training pipelines for generation, control, and memory heads.
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datasets import Dataset
import numpy as np

# Import the modules we'll implement
import sys
sys.path.append(str(Path(__file__).parent.parent))


class TestTripleHeadDPOConfig:
    """Test DPO configuration for triple-head architecture"""
    
    def test_dpo_config_creation(self):
        """Test creating DPO config for triple-head training"""
        from backend.app.narrative_engine.dpo_trainer import TripleHeadDPOConfig
        
        config = TripleHeadDPOConfig(
            head_type="generation",
            learning_rate=1e-5,
            beta=0.1,
            head_specific_lr={
                "generation": 1e-5,
                "control": 5e-6,
                "memory": 2e-6
            },
            cross_head_regularization=0.01
        )
        
        assert config.head_type == "generation"
        assert config.learning_rate == 1e-5
        assert config.beta == 0.1
        assert config.head_specific_lr["control"] == 5e-6
        assert config.cross_head_regularization == 0.01
    
    def test_dpo_config_validation(self):
        """Test DPO config validation"""
        from backend.app.narrative_engine.dpo_trainer import TripleHeadDPOConfig
        
        # Should raise error for invalid head type
        with pytest.raises(ValueError, match="Invalid head_type"):
            TripleHeadDPOConfig(head_type="invalid")


class TestGenerationHeadDPO:
    """Test DPO training for generation head"""
    
    def test_generation_dpo_trainer(self):
        """Test DPO trainer for generation head"""
        from backend.app.narrative_engine.dpo_trainer import GenerationDPOTrainer
        
        # Mock model and config
        with patch('backend.app.narrative_engine.model.NarrativeLLM') as MockModel:
            # Create a mock that inherits from nn.Module
            import torch.nn as nn
            
            class MockNarrativeLLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tp_size = None
                    self.is_parallelizable = False
                    self.model_parallel = False
                
                def forward(self, *args, **kwargs):
                    return {}
            
            mock_model = MockNarrativeLLM()
            MockModel.return_value = mock_model
            
            mock_config = MagicMock()
            mock_config.beta = 0.1
            mock_config.head_type = "generation"
            
            # Create actual training arguments to avoid mocking issues
            from transformers import TrainingArguments
            training_args = TrainingArguments(
                output_dir="test_output",
                logging_steps=10,
                eval_strategy="no",  # Disable evaluation to avoid eval dataset requirement
                per_device_train_batch_size=1,
                num_train_epochs=1
            )
            
            trainer = GenerationDPOTrainer(
                model=mock_model,
                config=mock_config,
                args=training_args,
                train_dataset=Dataset.from_list([{"prompt": "test", "chosen": "good", "rejected": "bad"}])
            )
            
            assert trainer is not None
            assert hasattr(trainer, 'compute_dpo_loss')
    
    def test_generation_dpo_loss(self):
        """Test DPO loss calculation for generation head"""
        from backend.app.narrative_engine.dpo_trainer import compute_generation_dpo_loss
        
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        # Mock logits and labels
        policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        ref_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        ref_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_generation_dpo_loss(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            ref_chosen_logits=ref_chosen_logits,
            ref_rejected_logits=ref_rejected_logits,
            labels=labels,
            beta=0.1
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)


class TestControlHeadDPO:
    """Test DPO training for control head"""
    
    def test_control_dpo_loss(self):
        """Test DPO loss for control token distributions"""
        from backend.app.narrative_engine.dpo_trainer import compute_control_dpo_loss
        
        batch_size = 2
        num_control_tokens = 64
        
        # Mock control token probabilities - ensure they're valid probabilities
        policy_chosen_probs = torch.rand(batch_size, num_control_tokens)
        policy_rejected_probs = torch.rand(batch_size, num_control_tokens)
        ref_chosen_probs = torch.rand(batch_size, num_control_tokens)
        ref_rejected_probs = torch.rand(batch_size, num_control_tokens)
        
        # Normalize to ensure they're valid probabilities (sum to 1)
        policy_chosen_probs = policy_chosen_probs / policy_chosen_probs.sum(dim=-1, keepdim=True)
        policy_rejected_probs = policy_rejected_probs / policy_rejected_probs.sum(dim=-1, keepdim=True)
        ref_chosen_probs = ref_chosen_probs / ref_chosen_probs.sum(dim=-1, keepdim=True)
        ref_rejected_probs = ref_rejected_probs / ref_rejected_probs.sum(dim=-1, keepdim=True)
        
        loss = compute_control_dpo_loss(
            policy_chosen_probs=policy_chosen_probs,
            policy_rejected_probs=policy_rejected_probs,
            ref_chosen_probs=ref_chosen_probs,
            ref_rejected_probs=ref_rejected_probs,
            beta=0.1,
            emotional_arc_weight=0.2
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestMemoryHeadDPO:
    """Test DPO training for memory head"""
    
    def test_memory_dpo_loss(self):
        """Test DPO loss for memory embeddings and metadata"""
        from backend.app.narrative_engine.dpo_trainer import compute_memory_dpo_loss
        
        batch_size = 2
        embedding_dim = 768
        metadata_dim = 4
        
        # Mock memory outputs
        policy_chosen_embedding = torch.randn(batch_size, embedding_dim)
        policy_rejected_embedding = torch.randn(batch_size, embedding_dim)
        ref_chosen_embedding = torch.randn(batch_size, embedding_dim)
        ref_rejected_embedding = torch.randn(batch_size, embedding_dim)
        
        policy_chosen_metadata = torch.rand(batch_size, metadata_dim)
        policy_rejected_metadata = torch.rand(batch_size, metadata_dim)
        ref_chosen_metadata = torch.rand(batch_size, metadata_dim)
        ref_rejected_metadata = torch.rand(batch_size, metadata_dim)
        
        loss = compute_memory_dpo_loss(
            policy_chosen_embedding=policy_chosen_embedding,
            policy_rejected_embedding=policy_rejected_embedding,
            ref_chosen_embedding=ref_chosen_embedding,
            ref_rejected_embedding=ref_rejected_embedding,
            policy_chosen_metadata=policy_chosen_metadata,
            policy_rejected_metadata=policy_rejected_metadata,
            ref_chosen_metadata=ref_chosen_metadata,
            ref_rejected_metadata=ref_rejected_metadata,
            beta=0.1,
            embedding_weight=0.7,
            metadata_weight=0.3
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestCoordinatedDPO:
    """Test coordinated DPO training across all heads"""
    
    def test_coordinated_dpo_trainer(self):
        """Test coordinated DPO trainer that optimizes all heads"""
        from backend.app.narrative_engine.dpo_trainer import CoordinatedDPOTrainer
        
        with patch('backend.app.narrative_engine.model.NarrativeLLM') as MockModel:
            # Create a mock that inherits from nn.Module
            import torch.nn as nn
            
            class MockNarrativeLLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tp_size = None
                    self.is_parallelizable = False
                    self.model_parallel = False
                
                def forward(self, *args, **kwargs):
                    return {}
            
            mock_model = MockNarrativeLLM()
            MockModel.return_value = mock_model
            
            mock_config = MagicMock()
            mock_config.generation_weight = 1.0
            mock_config.control_weight = 0.8
            mock_config.memory_weight = 0.6
            mock_config.coordination_weight = 0.4
            
            # Create actual training arguments
            from transformers import TrainingArguments
            training_args = TrainingArguments(
                output_dir="test_output",
                logging_steps=10,
                eval_strategy="no",
                per_device_train_batch_size=1,
                num_train_epochs=1
            )
            
            trainer = CoordinatedDPOTrainer(
                model=mock_model,
                config=mock_config,
                args=training_args
            )
            
            assert trainer is not None
            assert hasattr(trainer, 'generation_weight')
            assert hasattr(trainer, 'control_weight')
            assert hasattr(trainer, 'memory_weight')
    
    def test_cross_head_regularization(self):
        """Test cross-head regularization in coordinated DPO"""
        from backend.app.narrative_engine.dpo_trainer import compute_cross_head_regularization
        
        # Mock head outputs
        generation_loss = torch.tensor(1.0)
        control_loss = torch.tensor(0.8)
        memory_loss = torch.tensor(1.2)
        
        reg_loss = compute_cross_head_regularization(
            generation_loss=generation_loss,
            control_loss=control_loss,
            memory_loss=memory_loss,
            target_balance=1.0
        )
        
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0  # Should be non-negative


class TestDPOTrainingScripts:
    """Test the DPO training scripts"""
    
    def test_generation_dpo_script(self):
        """Test generation head DPO training script structure"""
        # Scripts exist now, so test they can be imported
        from scripts.run_generation_dpo import main
        assert main is not None
    
    def test_coordinated_dpo_script(self):
        """Test coordinated DPO training script"""
        from scripts.run_triple_head_dpo import main
        assert main is not None


class TestDPODataPreparation:
    """Test data preparation for DPO training"""
    
    def test_preference_to_dpo_format(self):
        """Test converting preference data to DPO format"""
        from backend.app.narrative_engine.dpo_trainer import prepare_dpo_dataset
        
        # Mock preference data
        preferences = [
            {
                "prompt": "Hello",
                "chosen": "Hi there! How are you?",
                "rejected": "What?"
            }
        ]
        
        dpo_dataset = prepare_dpo_dataset(
            preferences=preferences,
            tokenizer=None,  # Mock tokenizer
            max_length=512
        )
        
        assert len(dpo_dataset) == len(preferences)
        assert "prompt" in dpo_dataset[0]
        assert "chosen" in dpo_dataset[0]
        assert "rejected" in dpo_dataset[0]
    
    def test_head_specific_preference_loading(self):
        """Test loading head-specific preferences"""
        from backend.app.narrative_engine.dpo_trainer import load_head_specific_preferences
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock preference files
            gen_file = Path(tmpdir) / "generation_preferences.jsonl"
            gen_file.write_text(json.dumps({
                "conversation_id": "test",
                "content_quality": 8,
                "prompt": "test prompt",
                "chosen": "good response",
                "rejected": "bad response"
            }) + "\n")
            
            prefs = load_head_specific_preferences(
                preference_dir=Path(tmpdir),
                head_type="generation"
            )
            
            assert len(prefs) > 0
            assert "prompt" in prefs[0]


class TestDPOIntegration:
    """Test DPO integration with existing infrastructure"""
    
    def test_dpo_with_rlhf_trainer(self):
        """Test that DPO can use existing RLHF infrastructure"""
        from backend.app.services.training.rlhf_trainer import RLHFConfig
        
        # Create DPO-specific config
        config = RLHFConfig(
            algorithm="dpo",  # New algorithm type
            learning_rate=1e-5,
            beta=0.1,
            max_steps=1000
        )
        
        assert config.algorithm == "dpo"
        assert config.learning_rate == 1e-5
    
    def test_dpo_model_loading(self):
        """Test loading triple-head model for DPO"""
        from backend.app.narrative_engine.dpo_trainer import load_model_for_dpo
        
        with patch('backend.app.narrative_engine.model.NarrativeLLM') as MockModel:
            mock_model = MockModel.return_value
            
            model = load_model_for_dpo(
                model_path="test/path",
                head_type="generation"
            )
            
            assert model is not None 