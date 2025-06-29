"""
Unit tests for Inference Observability (R3-4)

Tests the enhanced inference pipeline with observability features:
- Optional capture of intermediate model states
- Storage of observability data with request IDs
- Debug flag functionality
- Proper error handling for observability features
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from pathlib import Path
import torch
from typing import Dict, Any, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent / "app"))

from utils.inference import InferenceManager
from utils.observability import ObservabilityLogger, InferenceObservabilityData


class TestInferenceObservability(unittest.TestCase):
    """Test enhanced inference pipeline with observability features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.inference_manager = InferenceManager()
        self.test_model_path = "Base: test-model"
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generate_response_without_debug_flag(self):
        """Test that normal generation works without capturing observability data"""
        with patch.object(self.inference_manager, '_generate_with_model') as mock_generate:
            mock_generate.return_value = "Test response"
            
            response = self.inference_manager.generate_response(
                model_path=self.test_model_path,
                prompt="Test prompt",
                enable_observability=False
            )
            
            self.assertEqual(response, "Test response")
            # Should not pass observability parameters to model.generate
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1]
            self.assertNotIn('output_attentions', call_args)
            self.assertNotIn('output_hidden_states', call_args)
    
    def test_generate_response_with_debug_flag(self):
        """Test that generation with debug flag captures intermediate states"""
        # Mock outputs for observability mode
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.LongTensor([[1, 2, 3, 4, 5]])
        mock_outputs.attentions = [torch.randn(1, 8, 10, 10)]  # Mock attention weights
        mock_outputs.hidden_states = [torch.randn(1, 10, 512)]  # Mock hidden states
        mock_outputs.scores = [torch.randn(1, 1000)]  # Mock token scores
        
        with patch.object(self.inference_manager, 'load_model') as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            
            # Mock tokenizer properly
            mock_tokenizer.return_value = {
                'input_ids': torch.LongTensor([[1, 2, 3]]),
                'attention_mask': torch.LongTensor([[1, 1, 1]])
            }
            mock_tokenizer.decode.return_value = "Test response"
            mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
            
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Mock the model.generate call to return observability data
            mock_model.generate.return_value = mock_outputs
            mock_model.device = torch.device('cpu')
            
            with patch('app.utils.observability.ObservabilityLogger') as mock_logger_class:
                mock_logger_instance = MagicMock()
                mock_logger_class.return_value = mock_logger_instance
                
                with patch('app.utils.observability.get_observability_logger') as mock_get_logger:
                    mock_get_logger.return_value = mock_logger_instance
                    
                    response = self.inference_manager.generate_response(
                        model_path=self.test_model_path,
                        prompt="Test prompt",
                        enable_observability=True,
                        request_id="test-123"
                    )
                    
                    self.assertEqual(response, "Test response")
                    
                    # Should have called model.generate with observability flags
                    mock_model.generate.assert_called_once()
                    call_kwargs = mock_model.generate.call_args[1]
                    self.assertTrue(call_kwargs.get('output_attentions', False))
                    self.assertTrue(call_kwargs.get('output_hidden_states', False))
                    
                    # Should have logged observability data
                    mock_logger_instance.log_inference_data.assert_called_once()
    
    def test_request_id_generation(self):
        """Test that request IDs are generated when not provided"""
        with patch.object(self.inference_manager, '_generate_with_model') as mock_generate:
            mock_generate.return_value = "Test response"
            
            with patch('app.utils.observability.generate_request_id') as mock_gen_id:
                mock_gen_id.return_value = 'req_generated-request-id'
                
                response = self.inference_manager.generate_response(
                    model_path=self.test_model_path,
                    prompt="Test prompt",
                    enable_observability=True
                )
                
                self.assertEqual(response, "Test response")
                # Should have generated a request ID
                mock_gen_id.assert_called_once()


class TestObservabilityLogger(unittest.TestCase):
    """Test the observability logging system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ObservabilityLogger(log_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_log_inference_data(self):
        """Test logging of inference observability data"""
        test_data = InferenceObservabilityData(
            request_id="test-123",
            prompt="Test prompt",
            response="Test response",
            model_path="test-model",
            generation_config={"temperature": 0.8},
            attention_weights=[torch.randn(1, 8, 10, 10)],
            hidden_states=[torch.randn(1, 10, 512)],
            token_probabilities={"token1": 0.5, "token2": 0.3, "token3": 0.2}
        )
        
        self.logger.log_inference_data(test_data)
        
        # Check that log file was created
        log_file = Path(self.temp_dir) / "test-123.json"
        self.assertTrue(log_file.exists())
        
        # Check log file contents
        with open(log_file, 'r') as f:
            logged_data = json.load(f)
            
        self.assertEqual(logged_data['request_id'], "test-123")
        self.assertEqual(logged_data['prompt'], "Test prompt")
        self.assertEqual(logged_data['response'], "Test response")
        self.assertEqual(logged_data['model_path'], "test-model")
        self.assertIn('timestamp', logged_data)
        self.assertIn('attention_weights_shape', logged_data)
        self.assertIn('hidden_states_shape', logged_data)
        self.assertEqual(logged_data['token_probabilities'], {"token1": 0.5, "token2": 0.3, "token3": 0.2})
    
    def test_get_log_data(self):
        """Test retrieval of logged observability data"""
        # First log some data
        test_data = InferenceObservabilityData(
            request_id="test-456",
            prompt="Test prompt 2",
            response="Test response 2",
            model_path="test-model-2",
            generation_config={"temperature": 0.7},
            attention_weights=[],
            hidden_states=[],
            token_probabilities={}
        )
        
        self.logger.log_inference_data(test_data)
        
        # Now retrieve it
        retrieved_data = self.logger.get_log_data("test-456")
        
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(retrieved_data['request_id'], "test-456")
        self.assertEqual(retrieved_data['prompt'], "Test prompt 2")
        self.assertEqual(retrieved_data['response'], "Test response 2")
    
    def test_get_nonexistent_log_data(self):
        """Test retrieval of non-existent log data returns None"""
        retrieved_data = self.logger.get_log_data("nonexistent-123")
        self.assertIsNone(retrieved_data)
    
    def test_list_available_logs(self):
        """Test listing all available log request IDs"""
        # Log multiple requests
        for i in range(3):
            test_data = InferenceObservabilityData(
                request_id=f"test-{i}",
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                model_path="test-model",
                generation_config={},
                attention_weights=[],
                hidden_states=[],
                token_probabilities={}
            )
            self.logger.log_inference_data(test_data)
        
        available_logs = self.logger.list_available_logs()
        
        self.assertEqual(len(available_logs), 3)
        self.assertIn("test-0", available_logs)
        self.assertIn("test-1", available_logs)
        self.assertIn("test-2", available_logs)


class TestInferenceObservabilityData(unittest.TestCase):
    """Test the observability data structure"""
    
    def test_data_structure_creation(self):
        """Test creation of observability data structure"""
        data = InferenceObservabilityData(
            request_id="test-789",
            prompt="Test prompt",
            response="Test response",
            model_path="test-model",
            generation_config={"temperature": 0.8, "max_tokens": 100},
            attention_weights=[torch.randn(1, 8, 10, 10)],
            hidden_states=[torch.randn(1, 10, 512)],
            token_probabilities={"hello": 0.8, "world": 0.2}
        )
        
        self.assertEqual(data.request_id, "test-789")
        self.assertEqual(data.prompt, "Test prompt")
        self.assertEqual(data.response, "Test response")
        self.assertEqual(data.model_path, "test-model")
        self.assertEqual(data.generation_config["temperature"], 0.8)
        self.assertEqual(len(data.attention_weights), 1)
        self.assertEqual(len(data.hidden_states), 1)
        self.assertEqual(data.token_probabilities["hello"], 0.8)
    
    def test_to_dict_serialization(self):
        """Test conversion to dictionary for JSON serialization"""
        attention_tensor = torch.randn(1, 8, 10, 10)
        hidden_tensor = torch.randn(1, 10, 512)
        
        data = InferenceObservabilityData(
            request_id="test-serialization",
            prompt="Test prompt",
            response="Test response",
            model_path="test-model",
            generation_config={"temperature": 0.9},
            attention_weights=[attention_tensor],
            hidden_states=[hidden_tensor],
            token_probabilities={"test": 0.7, "data": 0.3}
        )
        
        data_dict = data.to_dict()
        
        self.assertEqual(data_dict['request_id'], "test-serialization")
        self.assertEqual(data_dict['prompt'], "Test prompt")
        self.assertEqual(data_dict['response'], "Test response")
        self.assertIn('timestamp', data_dict)
        
        # Check that tensor shapes are preserved but tensors are not serialized directly
        self.assertEqual(data_dict['attention_weights_shape'], [list(attention_tensor.shape)])
        self.assertEqual(data_dict['hidden_states_shape'], [list(hidden_tensor.shape)])
        
        # Token probabilities should be preserved
        self.assertEqual(data_dict['token_probabilities'], {"test": 0.7, "data": 0.3})


if __name__ == '__main__':
    unittest.main() 