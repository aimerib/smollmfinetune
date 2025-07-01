"""
Tests for R4-9 Evaluation Harness & Safety Layer modules
"""

import pytest
import json
import time
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile


class TestJSONCorrectnessEvaluation:
    """Tests for JSON correctness evaluation module"""
    
    def test_json_evaluation_with_valid_json(self):
        """Test that valid JSON outputs get a score of 1.0"""
        from narrative_engine.evaluation.eval_json_correctness import JSONCorrectnessEvaluator
        
        evaluator = JSONCorrectnessEvaluator()
        
        # Test with valid JSON string
        valid_json = '{"action": "move", "target": "door", "confidence": 0.95}'
        score = evaluator.evaluate_single_output(valid_json)
        
        assert score == 1.0
    
    def test_json_evaluation_with_malformed_json(self):
        """Test that malformed JSON outputs get a score of 0.0"""
        from narrative_engine.evaluation.eval_json_correctness import JSONCorrectnessEvaluator
        
        evaluator = JSONCorrectnessEvaluator()
        
        # Test with malformed JSON
        malformed_json = '{"action": "move", "target": door}'  # Missing quotes around door
        score = evaluator.evaluate_single_output(malformed_json)
        
        assert score == 0.0
    
    def test_json_evaluation_with_model(self):
        """Test JSON evaluation with a mock model"""
        from narrative_engine.evaluation.eval_json_correctness import JSONCorrectnessEvaluator
        
        evaluator = JSONCorrectnessEvaluator()
        
        # Mock model that generates JSON responses
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock responses - mix of valid and invalid JSON
        mock_tokenizer.decode.side_effect = [
            '{"action": "speak", "text": "Hello"}',  # Valid
            '{"action": "move", target: "door"}',    # Invalid
            '{"action": "think", "thought": "Hmm"}', # Valid
        ]
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        results = evaluator.evaluate(
            model=mock_model,
            tokenizer=mock_tokenizer,
            test_prompts=[
                "Generate a speak action",
                "Generate a move action", 
                "Generate a think action"
            ]
        )
        
        assert results['json_correctness_rate'] == pytest.approx(2/3)
        assert results['total_prompts'] == 3
        assert results['valid_json_count'] == 2
        assert len(results['individual_scores']) == 3


class TestCoherenceEvaluation:
    """Tests for coherence evaluation using LLM-as-judge"""
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_coherence_evaluation_basic(self):
        """Test basic coherence evaluation setup"""
        from narrative_engine.evaluation.eval_coherence import CoherenceEvaluator
        
        evaluator = CoherenceEvaluator()
        
        # Mock conversation with contradictions
        conversation = [
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "My name is Alice."},
            {"role": "user", "content": "What did you say your name was?"},
            {"role": "assistant", "content": "My name is Bob."}  # Contradiction!
        ]
        
        result = evaluator.evaluate_conversation(conversation)
        
        # Should detect the contradiction and give a low coherence score
        assert 'coherence_score' in result
        assert 0.0 <= result['coherence_score'] <= 1.0
        assert result['coherence_score'] < 0.5  # Should be low due to contradiction
        assert 'contradictions' in result
        assert len(result['contradictions']) > 0  # Should detect the name contradiction
        assert 'reasoning' in result
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_coherence_evaluation_with_long_conversation(self):
        """Test coherence evaluation handles long conversations properly"""
        from narrative_engine.evaluation.eval_coherence import CoherenceEvaluator
        
        evaluator = CoherenceEvaluator()
        
        # Create a long but consistent conversation
        long_conversation = []
        for i in range(10):  # Reduced from 50 to be more reasonable for real LLM calls
            long_conversation.extend([
                {"role": "user", "content": f"Tell me about topic {i}"},
                {"role": "assistant", "content": f"Here's information about topic {i}. It's quite interesting and relates to previous topics we've discussed."}
            ])
        
        result = evaluator.evaluate_conversation(long_conversation)
        
        # Should handle long conversations and maintain consistency
        assert 'coherence_score' in result
        assert 0.0 <= result['coherence_score'] <= 1.0
        # With consistent responses, should have high coherence
        assert result['coherence_score'] > 0.6
        assert 'contradictions' in result
        assert 'reasoning' in result
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation  
    def test_coherence_evaluation_consistent_conversation(self):
        """Test coherence evaluation with a consistent conversation"""
        from narrative_engine.evaluation.eval_coherence import CoherenceEvaluator
        
        evaluator = CoherenceEvaluator()
        
        # Consistent conversation without contradictions
        consistent_conversation = [
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "My name is Alice."},
            {"role": "user", "content": "Nice to meet you Alice. What do you like to do?"},
            {"role": "assistant", "content": "Thank you! I enjoy reading books and having conversations like this one."},
            {"role": "user", "content": "Alice, what was the last book you read?"},
            {"role": "assistant", "content": "As Alice, I should mention that I recently enjoyed a science fiction novel about space exploration."}
        ]
        
        result = evaluator.evaluate_conversation(consistent_conversation)
        
        # Should have high coherence score for consistent conversation
        assert result['coherence_score'] > 0.7
        assert len(result['contradictions']) == 0  # No contradictions expected
        assert 'reasoning' in result


class TestLatencyEvaluation:
    """Tests for latency measurement evaluation"""
    
    def test_latency_measurement_basic(self):
        """Test basic latency measurement functionality"""
        from narrative_engine.evaluation.eval_latency import LatencyEvaluator
        
        evaluator = LatencyEvaluator()
        
        # Mock model with controlled timing
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Simulate generation with delays
        def mock_generate(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms generation time
            return torch.tensor([[1, 2, 3, 4, 5]])
        
        mock_model.generate = mock_generate
        mock_tokenizer.decode.return_value = "Test response"
        
        results = evaluator.evaluate(
            model=mock_model,
            tokenizer=mock_tokenizer,
            num_samples=3
        )
        
        assert 'time_to_first_token_ms' in results
        assert 'avg_tokens_per_second' in results
        assert 'p50_latency_ms' in results
        assert 'p95_latency_ms' in results
        assert results['num_samples'] == 3
    
    def test_latency_statistics(self):
        """Test that latency statistics are computed correctly"""
        from narrative_engine.evaluation.eval_latency import LatencyEvaluator
        
        evaluator = LatencyEvaluator()
        
        # Test statistical calculations
        latencies = [100, 150, 200, 250, 300]  # milliseconds
        stats = evaluator._compute_latency_statistics(latencies)
        
        assert stats['mean'] == 200.0
        assert stats['p50'] == 200.0
        assert stats['p95'] == 290.0  # 95th percentile
        assert stats['min'] == 100.0
        assert stats['max'] == 300.0


class TestMemoryConsistencyEvaluation:
    """Tests for memory head consistency evaluation"""
    
    def test_memory_consistency_basic(self):
        """Test basic memory consistency evaluation"""
        from narrative_engine.evaluation.eval_memory_consistency import MemoryConsistencyEvaluator
        
        evaluator = MemoryConsistencyEvaluator()
        
        # Mock model with memory head outputs
        mock_model = Mock()
        mock_outputs = Mock()
        
        # Valid memory embeddings (normalized)
        memory_embeddings = torch.nn.functional.normalize(
            torch.randn(2, 10, 768),  # batch=2, seq=10, embed_dim=768
            p=2, dim=-1
        )
        mock_outputs.memory_embeddings = memory_embeddings
        
        # Valid memory metadata
        mock_outputs.memory_metadata = {
            'importance_scores': torch.rand(2, 10),  # Between 0 and 1
            'memory_types': torch.randint(0, 4, (2, 10))  # Memory type categories
        }
        
        mock_model.return_value = mock_outputs
        
        results = evaluator.evaluate(mock_model)
        
        assert results['embeddings_normalized'] == True
        assert results['metadata_valid'] == True
        assert results['memory_head_functional'] == True
        assert 'embedding_statistics' in results
        assert 'metadata_statistics' in results
    
    def test_memory_consistency_with_invalid_embeddings(self):
        """Test memory evaluation with non-normalized embeddings"""
        from narrative_engine.evaluation.eval_memory_consistency import MemoryConsistencyEvaluator
        
        evaluator = MemoryConsistencyEvaluator()
        
        mock_model = Mock()
        mock_outputs = Mock()
        
        # Non-normalized embeddings (invalid)
        mock_outputs.memory_embeddings = torch.randn(2, 10, 768) * 10  # Large values
        
        mock_outputs.memory_metadata = {
            'importance_scores': torch.rand(2, 10),
            'memory_types': torch.randint(0, 4, (2, 10))
        }
        
        mock_model.return_value = mock_outputs
        
        results = evaluator.evaluate(mock_model)
        
        assert results['embeddings_normalized'] == False
        assert results['memory_head_functional'] == False
    
    @pytest.mark.slow
    @pytest.mark.llm  
    @pytest.mark.evaluation
    def test_memory_formation_accuracy(self):
        """Test memory formation accuracy evaluation"""
        from narrative_engine.evaluation.eval_memory_consistency import MemoryConsistencyEvaluator
        
        evaluator = MemoryConsistencyEvaluator()
        
        # Test memory formation from conversation
        conversation = [
            {"role": "user", "content": "My name is John and I love pizza."},
            {"role": "assistant", "content": "Nice to meet you, John! What's your favorite pizza topping?"},
            {"role": "user", "content": "I really enjoy pepperoni."},
        ]
        
        # Mock model for this test since it's about the evaluation logic
        mock_model = Mock()
        
        extracted_memories = [
            {"content": "User's name is John", "importance": 0.9, "type": "identity"},
            {"content": "User loves pizza", "importance": 0.7, "type": "preference"},
            {"content": "User's favorite topping is pepperoni", "importance": 0.6, "type": "preference"}
        ]
        
        accuracy = evaluator.evaluate_memory_formation(
            model=mock_model,
            conversation=conversation,
            expected_memories=extracted_memories
        )
        
        assert 'formation_accuracy' in accuracy
        assert 'precision' in accuracy
        assert 'recall' in accuracy
        assert accuracy['formation_accuracy'] >= 0.0


class TestSafetyLayer:
    """Tests for the SafetyLayer content filtering"""
    
    def test_safety_layer_initialization(self):
        """Test SafetyLayer initialization with blocklists"""
        from narrative_engine.evaluation.safety_layer import SafetyLayer
        
        blocklist = ["harmful_word", "dangerous_phrase"]
        safety = SafetyLayer(blocklist=blocklist)
        
        assert safety.blocklist == blocklist
        assert safety.is_enabled is True
    
    def test_safety_layer_input_filtering(self):
        """Test that SafetyLayer blocks harmful inputs"""
        from narrative_engine.evaluation.safety_layer import SafetyLayer
        
        safety = SafetyLayer(blocklist=["violence", "explicit"])
        
        # Test safe input
        safe_input = "Tell me about puppies"
        assert safety.check_input(safe_input) is True
        
        # Test blocked input
        blocked_input = "Tell me about violence"
        assert safety.check_input(blocked_input) is False
    
    def test_safety_layer_output_filtering(self):
        """Test that SafetyLayer filters harmful outputs"""
        from narrative_engine.evaluation.safety_layer import SafetyLayer
        
        safety = SafetyLayer(blocklist=["inappropriate"])
        
        # Test safe output
        safe_output = "Here's a nice story about friendship"
        filtered = safety.filter_output(safe_output)
        assert filtered == safe_output
        
        # Test blocked output
        blocked_output = "This contains inappropriate content"
        filtered = safety.filter_output(blocked_output)
        assert filtered == SafetyLayer.BLOCKED_MESSAGE
    
    def test_safety_layer_wrapped_generation(self):
        """Test SafetyLayer wrapping a model's generation function"""
        from narrative_engine.evaluation.safety_layer import SafetyLayer
        
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = "Safe response content"
        
        # Wrap with safety layer
        safety = SafetyLayer(blocklist=["danger"])
        wrapped_generate = safety.wrap_generation_function(mock_model.generate)
        
        # Test safe generation
        result = wrapped_generate("Safe prompt")
        assert result == "Safe response content"
        
        # Test blocked input
        result = wrapped_generate("This is danger")
        assert result == SafetyLayer.BLOCKED_MESSAGE
        
        # Test blocked output
        mock_model.generate.return_value = "Response with danger"
        result = wrapped_generate("Safe prompt")
        assert result == SafetyLayer.BLOCKED_MESSAGE
    
    def test_safety_layer_configuration(self):
        """Test SafetyLayer configuration loading"""
        from narrative_engine.evaluation.safety_layer import SafetyLayer
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "blocklist": ["word1", "word2"],
                "enabled": True,
                "case_sensitive": False
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            safety = SafetyLayer.from_config(config_path)
            assert safety.blocklist == ["word1", "word2"]
            assert safety.is_enabled is True
            assert safety.case_sensitive is False
        finally:
            Path(config_path).unlink()


class TestEvaluationOrchestration:
    """Tests for the main evaluation orchestration script"""
    
    @patch('narrative_engine.evaluation.eval_json_correctness.JSONCorrectnessEvaluator')
    @patch('narrative_engine.evaluation.eval_coherence.CoherenceEvaluator')
    @patch('narrative_engine.evaluation.eval_latency.LatencyEvaluator')
    @patch('narrative_engine.evaluation.eval_memory_consistency.MemoryConsistencyEvaluator')
    def test_run_r4_9_evaluation_suite(self, mock_memory, mock_latency, mock_coherence, mock_json):
        """Test that all R4-9 evaluations are orchestrated correctly"""
        # This will test the main orchestration script
        # We'll implement this after creating the script
        pass 