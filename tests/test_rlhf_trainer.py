"""
Unit tests for RLHF (PPO/GRPO) trainer functionality.

Tests the implementation of preference-based fine-tuning using TRL library.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from datasets import Dataset

# Import the modules we'll implement
import sys
sys.path.append(str(Path(__file__).parent.parent / "app"))


class TestRLHFTrainer:
    """Test suite for RLHF trainer functionality"""
    
    def test_rlhf_trainer_import(self):
        """Test that we can import the RLHF trainer module"""
        try:
            from app.utils.rlhf_trainer import run_rlhf, prepare_preference_dataset, RLHFConfig
            assert True
        except ImportError:
            pytest.fail("Failed to import rlhf_trainer module")
    
    def test_rlhf_config_defaults(self):
        """Test RLHF configuration dataclass with default values"""
        from app.utils.rlhf_trainer import RLHFConfig
        
        config = RLHFConfig()
        assert config.algorithm == "grpo"
        assert config.learning_rate == 5e-6
        assert config.beta == 0.0  # No reference model by default
        assert config.num_generations == 6
        assert config.max_steps == 500
        assert config.per_device_train_batch_size == 1
    
    def test_prepare_preference_dataset(self):
        """Test preference dataset preparation from NDJSON logs"""
        from app.utils.rlhf_trainer import prepare_preference_dataset
        
        # Create test preference data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            json.dump({"prompt": "Hello", "chosen": "Hi there!", "rejected": ["Hello.", "Hey!"]}, f)
            f.write('\n')
            json.dump({"prompt": "How are you?", "chosen": "I'm doing well!", "rejected": ["Fine.", "Good."]}, f)
            temp_path = Path(f.name)
        
        try:
            # Test dataset creation
            dataset = prepare_preference_dataset(temp_path)
            
            assert isinstance(dataset, Dataset)
            assert len(dataset) == 2
            assert "prompt" in dataset.column_names
            assert "chosen" in dataset.column_names
            assert "rejected" in dataset.column_names
            
            # Check data integrity
            assert dataset[0]["prompt"] == "Hello"
            assert dataset[0]["chosen"] == "Hi there!"
            assert dataset[0]["rejected"] in ["Hello.", "Hey!"]
            
        finally:
            temp_path.unlink()
    
    @patch('app.utils.rlhf_trainer.load_model_and_tokenizer')
    def test_run_rlhf_grpo(self, mock_load_model):
        """Test running RLHF with GRPO algorithm"""
        # Mock TRL being available
        import app.utils.rlhf_trainer as rlhf_module
        
        # Create mock trainer classes
        mock_grpo_trainer = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_instance.save_model.return_value = "/tmp/saved_model"
        mock_grpo_trainer.return_value = mock_trainer_instance
        
        # Mock TRL classes on the module
        rlhf_module.TRL_AVAILABLE = True
        rlhf_module.GRPOTrainer = mock_grpo_trainer
        rlhf_module.GRPOConfig = Mock()
        
        from app.utils.rlhf_trainer import run_rlhf, RLHFConfig
        
        # Setup model loading mock
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Create test dataset
        dataset = Dataset.from_dict({
            "prompt": ["Test prompt"],
            "chosen": ["Good response"],
            "rejected": ["Bad response"]
        })
        
        # Run RLHF
        config = RLHFConfig(algorithm="grpo", max_steps=10)
        result_path = run_rlhf(
            model_path="test/model",
            pref_dataset=dataset,
            config=config
        )
        
        # Verify trainer was called correctly
        assert mock_grpo_trainer.called
        assert mock_trainer_instance.train.called
        assert mock_trainer_instance.save_model.called
        assert result_path == "rlhf_output/adapter_grpo"
    
    @patch('app.utils.rlhf_trainer.load_model_and_tokenizer')
    def test_run_rlhf_ppo(self, mock_load_model):
        """Test running RLHF with PPO algorithm"""
        # Mock TRL being available
        import app.utils.rlhf_trainer as rlhf_module
        
        # Create mock trainer classes
        mock_ppo_trainer = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_instance.save_model.return_value = "/tmp/saved_model"
        mock_ppo_trainer.return_value = mock_trainer_instance
        
        # Mock TRL classes on the module
        rlhf_module.TRL_AVAILABLE = True
        rlhf_module.PPOTrainer = mock_ppo_trainer
        rlhf_module.PPOConfig = Mock()
        
        from app.utils.rlhf_trainer import run_rlhf, RLHFConfig
        
        # Setup model loading mock
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Create test dataset
        dataset = Dataset.from_dict({
            "prompt": ["Test prompt"],
            "chosen": ["Good response"],
            "rejected": ["Bad response"]
        })
        
        # Run RLHF
        config = RLHFConfig(algorithm="ppo", max_steps=10)
        result_path = run_rlhf(
            model_path="test/model",
            pref_dataset=dataset,
            config=config
        )
        
        # Verify PPO trainer was called
        assert mock_ppo_trainer.called
        assert result_path == "rlhf_output/adapter_ppo"
    
    def test_rlhf_with_wandb_logging(self):
        """Test RLHF integration with WandB logging"""
        from app.utils.rlhf_trainer import RLHFConfig
        
        config = RLHFConfig(report_to="wandb")
        assert config.report_to == "wandb"
        
        # Test with disabled wandb
        config_no_wandb = RLHFConfig(report_to="none")
        assert config_no_wandb.report_to == "none"


class TestPreferenceAggregation:
    """Test suite for preference data aggregation script"""
    
    def test_aggregate_preferences_script_exists(self):
        """Test that aggregate_preferences.py script exists"""
        script_path = Path("scripts/aggregate_preferences.py")
        # Just check if we can import it (it will be created)
        assert True  # Placeholder until script is created
    
    def test_aggregate_multiple_preference_files(self):
        """Test aggregating preferences from multiple character directories"""
        # This will be implemented when we create the script
        pass


class TestTrainingIntegration:
    """Test suite for RLHF integration with TrainingManager"""
    
    @patch('utils.training.Path.exists')
    def test_training_manager_detects_preference_data(self, mock_exists):
        """Test that TrainingManager can detect if preference data exists"""
        from app.utils.training import TrainingManager
        
        # Mock that preference logs exist
        mock_exists.return_value = True
        
        manager = TrainingManager()
        
        # This method will be added to TrainingManager
        # assert manager.has_preference_data("test_character")
        
    def test_training_config_enables_rlhf(self):
        """Test that training config can enable RLHF when sufficient preferences exist"""
        # This will test the UI integration
        pass 