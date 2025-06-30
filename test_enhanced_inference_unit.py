#!/usr/bin/env python3
"""
Unit tests for Enhanced InferenceManager with Adapter Hot-Swapping

This script tests the new functionality without requiring actual trained adapters.
"""

import sys
sys.path.append('app')

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from utils.inference import InferenceManager, NARRATIVE_ENGINE_AVAILABLE


class TestEnhancedInferenceManager(unittest.TestCase):
    """Test the enhanced InferenceManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.inference_manager = InferenceManager()
    
    def test_narrative_engine_available(self):
        """Test that NarrativeLLM imports are working"""
        self.assertTrue(NARRATIVE_ENGINE_AVAILABLE, "NarrativeLLM should be available")
        self.assertTrue(hasattr(self.inference_manager, 'narrative_models'))
        self.assertTrue(hasattr(self.inference_manager, 'character_adapters'))
        self.assertTrue(hasattr(self.inference_manager, 'active_adapters'))
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    def test_find_all_character_adapters(self, mock_iterdir, mock_exists):
        """Test finding character adapters"""
        # Mock file structure
        mock_adapter_dir = Mock()
        mock_safetensors = Mock()
        mock_safetensors.exists.return_value = True
        
        mock_adapter_dir.__truediv__ = Mock(return_value=mock_safetensors)
        mock_exists.return_value = True
        
        # Mock checkpoint directory
        mock_checkpoint = Mock()
        mock_checkpoint.is_dir.return_value = True
        mock_checkpoint.name = "checkpoint-100"
        mock_checkpoint.__truediv__ = Mock(return_value=mock_safetensors)
        
        mock_iterdir.return_value = [mock_checkpoint]
        
        # Test the method
        with patch.object(self.inference_manager.project_dir, '__truediv__', return_value=mock_adapter_dir):
            adapters = self.inference_manager._find_all_character_adapters("test_character")
        
        # Should find SFT and checkpoint adapters
        self.assertIn("SFT", adapters)
        self.assertIn("Checkpoint-checkpoint-100", adapters)
    
    def test_get_preferred_adapter_type(self):
        """Test adapter preference logic"""
        # Test with RLHF preferred
        adapters = {
            "SFT": Path("/path/sft"),
            "RLHF-GRPO": Path("/path/grpo"),
            "Checkpoint-100": Path("/path/checkpoint")
        }
        preferred = self.inference_manager._get_preferred_adapter_type(adapters)
        self.assertEqual(preferred, "RLHF-GRPO")
        
        # Test with only SFT
        adapters = {"SFT": Path("/path/sft")}
        preferred = self.inference_manager._get_preferred_adapter_type(adapters)
        self.assertEqual(preferred, "SFT")
        
        # Test with only checkpoints
        adapters = {"Checkpoint-100": Path("/path/checkpoint")}
        preferred = self.inference_manager._get_preferred_adapter_type(adapters)
        self.assertEqual(preferred, "Checkpoint-100")
    
    @patch('narrative_engine.model.NarrativeLLM')
    @patch('narrative_engine.config.NarrativeLLMConfig')
    def test_load_character_with_hot_swap(self, mock_config, mock_narrative_llm):
        """Test loading character with hot-swapping"""
        # Mock NarrativeLLM instance
        mock_model = Mock()
        mock_narrative_llm.return_value = mock_model
        
        # Mock finding adapters
        mock_adapters = {
            "SFT": Path("/path/sft"),
            "RLHF-GRPO": Path("/path/grpo")
        }
        
        with patch.object(self.inference_manager, '_find_all_character_adapters', return_value=mock_adapters):
            result = self.inference_manager.load_character_with_hot_swap("test_character")
        
        # Verify success
        self.assertIn("✅", result)
        self.assertIn("test_character", self.inference_manager.narrative_models)
        self.assertIn("test_character", self.inference_manager.character_adapters)
        
        # Verify adapters were loaded
        self.assertEqual(mock_model.load_adapter.call_count, 2)
        mock_model.set_active_adapter.assert_called_once()
    
    def test_character_info(self):
        """Test getting character information"""
        # Test with unloaded character
        info = self.inference_manager.get_character_info("nonexistent")
        self.assertFalse(info["loaded"])
        
        # Mock a loaded character
        mock_model = Mock()
        self.inference_manager.narrative_models["test_char"] = mock_model
        self.inference_manager.character_adapters["test_char"] = {"SFT": Path("/path")}
        self.inference_manager.active_adapters["test_char"] = "test_char_SFT"
        
        info = self.inference_manager.get_character_info("test_char")
        self.assertTrue(info["loaded"])
        self.assertEqual(info["character_name"], "test_char")
        self.assertEqual(info["active_adapter"], "SFT")
        self.assertEqual(info["total_adapters"], 1)
    
    def test_switch_character_adapter(self):
        """Test adapter switching"""
        # Setup mock character
        mock_model = Mock()
        self.inference_manager.narrative_models["test_char"] = mock_model
        self.inference_manager.character_adapters["test_char"] = {
            "SFT": Path("/path/sft"),
            "RLHF-GRPO": Path("/path/grpo")
        }
        
        # Test successful switch
        result = self.inference_manager.switch_character_adapter("test_char", "RLHF-GRPO")
        self.assertIn("✅", result)
        mock_model.set_active_adapter.assert_called_with("test_char_RLHF-GRPO")
        
        # Test invalid adapter
        result = self.inference_manager.switch_character_adapter("test_char", "INVALID")
        self.assertIn("❌", result)
    
    @patch('narrative_engine.model.NarrativeLLM')
    def test_generate_with_character(self, mock_narrative_llm):
        """Test generation with character"""
        # Setup mock model and response
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model.tokenizer = mock_tokenizer
        mock_model.base_model.device = "cpu"
        
        mock_tokenizer.return_value = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_model.generate_with_control.return_value = {
            "generated_text": "Hello! I'm doing well, thank you!",
            "control_tokens": ["<mood_happy>"],
            "emotional_state": {"happy": 0.8},
            "surprise_score": 0.1
        }
        
        # Mock tensor movement
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        mock_tokenizer.return_value = {"input_ids": mock_tensor, "attention_mask": mock_tensor}
        
        self.inference_manager.narrative_models["test_char"] = mock_model
        self.inference_manager.active_adapters["test_char"] = "test_char_SFT"
        
        # Test generation
        response = self.inference_manager.generate_with_character(
            "test_char", 
            "Hello! How are you?"
        )
        
        self.assertNotIn("error", response)
        self.assertEqual(response["response"], "Hello! I'm doing well, thank you!")
        self.assertEqual(response["active_adapter"], "SFT")
        self.assertEqual(response["control_tokens"], ["<mood_happy>"])
    
    def test_unload_character(self):
        """Test character unloading"""
        # Setup mock character
        mock_model = Mock()
        self.inference_manager.narrative_models["test_char"] = mock_model
        self.inference_manager.character_adapters["test_char"] = {"SFT": Path("/path")}
        self.inference_manager.active_adapters["test_char"] = "test_char_SFT"
        
        # Test unloading
        result = self.inference_manager.unload_character("test_char")
        self.assertIn("✅", result)
        
        # Verify cleanup
        self.assertNotIn("test_char", self.inference_manager.narrative_models)
        self.assertNotIn("test_char", self.inference_manager.character_adapters)
        self.assertNotIn("test_char", self.inference_manager.active_adapters)
    
    def test_clear_model_cache_includes_narrative_models(self):
        """Test that clearing cache also clears narrative models"""
        # Add some mock data
        self.inference_manager.narrative_models["test"] = Mock()
        self.inference_manager.character_adapters["test"] = {}
        self.inference_manager.active_adapters["test"] = "test_SFT"
        
        # Clear cache
        self.inference_manager.clear_model_cache()
        
        # Verify everything is cleared
        self.assertEqual(len(self.inference_manager.narrative_models), 0)
        self.assertEqual(len(self.inference_manager.character_adapters), 0)
        self.assertEqual(len(self.inference_manager.active_adapters), 0)


def run_tests():
    """Run all the tests"""
    print("🧪 Testing Enhanced InferenceManager...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedInferenceManager)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All tests passed! Enhanced InferenceManager is working correctly.")
        print("\n✨ New capabilities verified:")
        print("  🔍 Character adapter discovery")
        print("  📥 Multi-adapter loading")
        print("  🔄 Adapter hot-swapping")
        print("  💬 Enhanced generation with control tokens")
        print("  📊 Character information tracking")
        print("  🧹 Proper cleanup and memory management")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        for failure in result.failures:
            print(f"  - {failure[0]}")


if __name__ == "__main__":
    run_tests() 