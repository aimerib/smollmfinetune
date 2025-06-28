"""
Tests for Lore Adherence Metric (R1-9)

Tests the lore adherence evaluation functionality including:
- LLM-as-a-judge scoring for lore fact incorporation
- Integration with TrainingQualityTracker
- WandB logging capabilities
- Mock-based testing for LLM calls
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import logging

# These imports will fail initially (TDD - red phase)
from app.utils.evaluation.lore_metric import (
    calculate_lore_adherence,
    evaluate_multiple_lore_facts,
    LoreEvaluationResult
)


class TestLoreMetric(unittest.TestCase):
    """Test lore adherence metric functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_response = "The ancient dragon Alduin rules over the northern mountains with great power."
        self.test_lore_fact = "Alduin is an ancient dragon who dominates the northern mountain regions."
        self.contradictory_response = "Alduin is a young wizard who lives in the southern plains."
        self.ignoring_response = "The weather today is quite pleasant for a walk."
    
    def test_calculate_lore_adherence_incorporated(self):
        """Test lore adherence calculation when lore is correctly incorporated"""
        # Mock the OpenAI client response for perfect incorporation
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"lore_score": 1.0, "reasoning": "Response correctly mentions Alduin as an ancient dragon ruling northern mountains"}'
        
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            score = calculate_lore_adherence(self.test_response, self.test_lore_fact)
            
            # Should return perfect score
            self.assertEqual(score, 1.0)
            
            # Verify LLM was called with correct parameters
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            self.assertIn("gpt-4o-mini", str(call_args))
    
    def test_calculate_lore_adherence_contradicted(self):
        """Test lore adherence calculation when lore is contradicted"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"lore_score": 0.0, "reasoning": "Response contradicts lore by calling Alduin a young wizard instead of ancient dragon"}'
        
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            score = calculate_lore_adherence(self.contradictory_response, self.test_lore_fact)
            
            # Should return zero score for contradiction
            self.assertEqual(score, 0.0)
    
    def test_calculate_lore_adherence_ignored(self):
        """Test lore adherence calculation when lore is ignored"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"lore_score": 0.5, "reasoning": "Response ignores the lore fact about Alduin entirely"}'
        
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            score = calculate_lore_adherence(self.ignoring_response, self.test_lore_fact)
            
            # Should return neutral score for ignoring
            self.assertEqual(score, 0.5)
    
    def test_calculate_lore_adherence_invalid_input(self):
        """Test lore adherence with invalid inputs"""
        # Empty response
        with self.assertRaises(ValueError):
            calculate_lore_adherence("", self.test_lore_fact)
        
        # Empty lore fact
        with self.assertRaises(ValueError):
            calculate_lore_adherence(self.test_response, "")
        
        # None inputs
        with self.assertRaises(ValueError):
            calculate_lore_adherence(None, self.test_lore_fact)
    
    def test_calculate_lore_adherence_json_parse_error(self):
        """Test handling of malformed JSON response from LLM"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'Invalid JSON response'
        
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            with patch('app.utils.evaluation.lore_metric.logger') as mock_logger:
                score = calculate_lore_adherence(self.test_response, self.test_lore_fact)
                
                # Should return default score and log warning
                self.assertEqual(score, 0.5)
                mock_logger.warning.assert_called()
    
    def test_calculate_lore_adherence_api_error(self):
        """Test handling of API errors"""
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_get_client.return_value = mock_client
            
            with patch('app.utils.evaluation.lore_metric.logger') as mock_logger:
                score = calculate_lore_adherence(self.test_response, self.test_lore_fact)
                
                # Should return default score and log error
                self.assertEqual(score, 0.5)
                mock_logger.error.assert_called()
    
    def test_evaluate_multiple_lore_facts(self):
        """Test evaluation of multiple lore facts"""
        lore_facts = [
            "Alduin is an ancient dragon who dominates the northern mountain regions.",
            "The kingdom of Skyrim is known for its harsh winters and Nordic culture.",
            "Magic is forbidden in the capital city of Whiterun."
        ]
        
        # Mock responses for each fact
        mock_responses = [
            '{"lore_score": 1.0, "reasoning": "Correctly mentions Alduin"}',
            '{"lore_score": 0.5, "reasoning": "No mention of Skyrim"}',
            '{"lore_score": 0.0, "reasoning": "Contradicts magic ban"}'
        ]
        
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                Mock(choices=[Mock(message=Mock(content=resp))]) for resp in mock_responses
            ]
            mock_get_client.return_value = mock_client
            
            result = evaluate_multiple_lore_facts(self.test_response, lore_facts)
            
            # Check result structure
            self.assertIsInstance(result, LoreEvaluationResult)
            self.assertEqual(result.average_score, 0.5)  # (1.0 + 0.5 + 0.0) / 3
            self.assertEqual(len(result.individual_scores), 3)
            self.assertEqual(result.individual_scores[0], 1.0)
            self.assertEqual(result.individual_scores[1], 0.5)
            self.assertEqual(result.individual_scores[2], 0.0)
    
    def test_lore_evaluation_result_dataclass(self):
        """Test LoreEvaluationResult dataclass structure"""
        result = LoreEvaluationResult(
            average_score=0.75,
            individual_scores=[1.0, 0.8, 0.5],
            lore_facts_evaluated=3,
            evaluation_details=[
                {"fact": "Test fact 1", "score": 1.0, "reasoning": "Perfect"},
                {"fact": "Test fact 2", "score": 0.8, "reasoning": "Good"},
                {"fact": "Test fact 3", "score": 0.5, "reasoning": "Neutral"}
            ]
        )
        
        self.assertEqual(result.average_score, 0.75)
        self.assertEqual(len(result.individual_scores), 3)
        self.assertEqual(result.lore_facts_evaluated, 3)
        self.assertEqual(len(result.evaluation_details), 3)
    
    def test_lore_metric_integration_with_training_quality_tracker(self):
        """Test integration with TrainingQualityTracker"""
        # This will test the integration points once implemented
        mock_tracker = Mock()
        mock_lore_facts = ["Test lore fact"]
        
        # Mock the OpenAI client properly to return expected score
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"lore_score": 0.8, "reasoning": "Response correctly incorporates lore"}'
        
        with patch('app.utils.evaluation.lore_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            # Simulate how the training pipeline would use this
            score = calculate_lore_adherence("Test response", "Test lore fact")
            
            # Verify the score is calculated correctly
            self.assertEqual(score, 0.8)
            # Additional integration testing will be added when TrainingQualityTracker is enhanced


if __name__ == '__main__':
    unittest.main() 