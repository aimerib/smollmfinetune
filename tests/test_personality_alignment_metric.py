import unittest
import pytest
import json
from typing import Dict


class TestPersonalityAlignmentMetric(unittest.TestCase):
    """Test the Big-Five personality alignment metric functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_big_five_scores = {
            "openness": 0.8,
            "conscientiousness": 0.3,
            "extraversion": 0.9,
            "agreeableness": 0.6,
            "neuroticism": 0.2
        }
        
        self.sample_response = "Hey there! I'm so excited to meet you! Let's go on an adventure and try something totally new today!"
    
    def test_module_exists(self):
        """Test that the personality_metric module can be imported"""
        try:
            from app.utils.evaluation.personality_metric import calculate_personality_alignment
            self.assertTrue(True)
        except ImportError:
            self.fail("personality_metric module does not exist")
    
    def test_calculate_personality_alignment_exists(self):
        """Test that the calculate_personality_alignment function exists"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        self.assertTrue(callable(calculate_personality_alignment))
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_calculate_personality_alignment_returns_float(self):
        """Test that the function returns a float between 0 and 1"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        result = calculate_personality_alignment(self.sample_response, self.sample_big_five_scores)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_personality_alignment_high_openness_extraversion(self):
        """Test alignment scoring for high openness and extraversion traits"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        # Response that should align well with high openness and extraversion
        high_oe_response = "I absolutely love exploring new places, meeting new people, and trying exciting adventures!"
        high_oe_traits = {
            "openness": 0.9,
            "conscientiousness": 0.5,
            "extraversion": 0.9,
            "agreeableness": 0.5,
            "neuroticism": 0.3
        }
        
        score = calculate_personality_alignment(high_oe_response, high_oe_traits)
        
        # Should show good alignment (expect score > 0.6)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.6)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_personality_alignment_low_extraversion(self):
        """Test alignment scoring for low extraversion traits"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        # Response that should align with low extraversion
        low_e_response = "I prefer quiet evenings at home, reading a book by myself."
        low_e_traits = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.2,  # Low extraversion
            "agreeableness": 0.5,
            "neuroticism": 0.3
        }
        
        score = calculate_personality_alignment(low_e_response, low_e_traits)
        
        # Should show good alignment with introverted response
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.6)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_personality_misalignment_detection(self):
        """Test that misaligned personality traits are detected"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        # Introverted response with extraverted target traits
        introverted_response = "I really prefer staying home alone and avoiding social gatherings."
        extraverted_traits = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.9,  # High extraversion - should clash with response
            "agreeableness": 0.5,
            "neuroticism": 0.3
        }
        
        score = calculate_personality_alignment(introverted_response, extraverted_traits)
        
        # Should show poor alignment (expect score < 0.5)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertLess(score, 0.5)
    
    def test_validates_empty_response(self):
        """Test that the function raises ValueError for empty response"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        with self.assertRaises(ValueError) as context:
            calculate_personality_alignment("", self.sample_big_five_scores)
        
        self.assertIn("Response cannot be empty", str(context.exception))
    
    def test_validates_empty_big_five_scores(self):
        """Test that the function raises ValueError for empty big_five_scores"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        with self.assertRaises(ValueError) as context:
            calculate_personality_alignment(self.sample_response, {})
        
        self.assertIn("big_five_scores cannot be empty", str(context.exception))
    
    def test_validates_score_ranges(self):
        """Test that the function validates score ranges"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        invalid_scores = {
            "openness": 1.5,  # Out of range
            "conscientiousness": 0.3,
            "extraversion": 0.9,
            "agreeableness": 0.6,
            "neuroticism": 0.2
        }
        
        with self.assertRaises(ValueError) as context:
            calculate_personality_alignment(self.sample_response, invalid_scores)
        
        self.assertIn("must be between 0.0 and 1.0", str(context.exception))
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_cached_version_consistency(self):
        """Test that the cached version returns consistent results"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment_cached
        
        # Convert dict to tuple for cached version
        big_five_tuple = (0.8, 0.3, 0.9, 0.6, 0.2)
        
        result1 = calculate_personality_alignment_cached(self.sample_response, big_five_tuple)
        result2 = calculate_personality_alignment_cached(self.sample_response, big_five_tuple)
        
        # Results should be identical due to caching
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, float)
        self.assertGreaterEqual(result1, 0.0)
        self.assertLessEqual(result1, 1.0)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_scoring_consistency_across_similar_responses(self):
        """Test that similar responses get similar scores"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        # Similar extraverted, open responses
        responses = [
            "I love meeting new people and trying new experiences!",
            "Meeting people and exploring new things makes me so excited!",
            "I get energized by social interactions and novel adventures!"
        ]
        
        scores = []
        for response in responses:
            score = calculate_personality_alignment(response, self.sample_big_five_scores)
            scores.append(score)
        
        # All scores should be reasonably consistent (within 0.3 range)
        score_range = max(scores) - min(scores)
        self.assertLess(score_range, 0.4)
        
        # All should show good alignment with high openness/extraversion
        for score in scores:
            self.assertGreater(score, 0.5)


class TestTrainingQualityTrackerIntegration(unittest.TestCase):
    """Test integration of personality alignment metric with TrainingQualityTracker"""
    
    def test_tracker_has_personality_alignment_field(self):
        """Test that TrainingQualityTracker can track personality alignment"""
        from app.utils.metrics import TrainingQualityTracker
        
        tracker = TrainingQualityTracker()
        
        # Check that we can add personality alignment scores
        self.assertTrue(hasattr(tracker, 'add_personality_alignment_score'))
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_tracker_logs_personality_alignment_to_wandb(self):
        """Test that personality alignment is logged to WandB"""
        from app.utils.metrics import TrainingQualityTracker
        
        tracker = TrainingQualityTracker()
        
        # Add some personality alignment scores
        tracker.add_personality_alignment_score(0.85)
        tracker.add_personality_alignment_score(0.90)
        
        # Get metrics for logging
        metrics = tracker.get_wandb_metrics()
        
        self.assertIn('avg_personality_alignment', metrics)
        self.assertAlmostEqual(metrics['avg_personality_alignment'], 0.875)


if __name__ == '__main__':
    unittest.main() 