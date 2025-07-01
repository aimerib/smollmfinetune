"""
Tests for Lore Adherence Metric (R1-9)

Tests the lore adherence evaluation functionality including:
- LLM-as-a-judge scoring for lore fact incorporation
- Integration with TrainingQualityTracker
- WandB logging capabilities
- Real LLM integration testing
"""

import unittest
import pytest
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
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_calculate_lore_adherence_incorporated(self):
        """Test lore adherence calculation when lore is correctly incorporated"""
        score = calculate_lore_adherence(self.test_response, self.test_lore_fact)
        
        # Should return high score for good incorporation (0.7-1.0 range)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # With good lore incorporation, expect score > 0.6
        self.assertGreater(score, 0.6)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_calculate_lore_adherence_contradicted(self):
        """Test lore adherence calculation when lore is contradicted"""
        score = calculate_lore_adherence(self.contradictory_response, self.test_lore_fact)
        
        # Should return low score for contradiction (0.0-0.4 range)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # With contradiction, expect score < 0.4
        self.assertLess(score, 0.4)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_calculate_lore_adherence_ignored(self):
        """Test lore adherence calculation when lore is ignored"""
        score = calculate_lore_adherence(self.ignoring_response, self.test_lore_fact)
        
        # Should return neutral score for ignoring (0.4-0.6 range)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # With ignored lore, expect neutral score
        self.assertGreaterEqual(score, 0.3)
        self.assertLessEqual(score, 0.7)
    
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
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_calculate_lore_adherence_json_parse_error_handling(self):
        """Test handling of edge cases that might cause JSON parse errors"""
        # Test with complex punctuation that might confuse JSON parsing
        complex_response = 'The dragon said: "I am ancient!" with quotes and special chars: @#$%'
        score = calculate_lore_adherence(complex_response, self.test_lore_fact)
        
        # Should still return valid score despite complex input
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_evaluate_multiple_lore_facts(self):
        """Test evaluation of multiple lore facts"""
        lore_facts = [
            "Alduin is an ancient dragon who dominates the northern mountain regions.",
            "The kingdom of Skyrim is known for its harsh winters and Nordic culture.",
            "Magic is forbidden in the capital city of Whiterun."
        ]
        
        result = evaluate_multiple_lore_facts(self.test_response, lore_facts)
        
        # Check result structure
        self.assertIsInstance(result, LoreEvaluationResult)
        self.assertIsInstance(result.average_score, float)
        self.assertGreaterEqual(result.average_score, 0.0)
        self.assertLessEqual(result.average_score, 1.0)
        self.assertEqual(len(result.individual_scores), 3)
        self.assertEqual(result.lore_facts_evaluated, 3)
        
        # All individual scores should be valid
        for score in result.individual_scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # First fact (about Alduin) should score highest since response mentions Alduin
        self.assertGreater(result.individual_scores[0], 0.5)
    
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
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_lore_metric_integration_scoring_consistency(self):
        """Test that scoring is consistent for similar inputs"""
        # Test with multiple similar lore-consistent responses
        responses = [
            "The ancient dragon Alduin dominates the northern mountains.",
            "Alduin, an ancient dragon, rules over the northern mountain regions.",
            "In the northern mountains, the ancient dragon Alduin holds dominion."
        ]
        
        scores = []
        for response in responses:
            score = calculate_lore_adherence(response, self.test_lore_fact)
            scores.append(score)
        
        # All scores should be reasonably high and consistent
        for score in scores:
            self.assertGreater(score, 0.6)  # High adherence expected
        
        # Scores should be relatively consistent (within 0.3 range)
        score_range = max(scores) - min(scores)
        self.assertLess(score_range, 0.4)


if __name__ == '__main__':
    unittest.main() 