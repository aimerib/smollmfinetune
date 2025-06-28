import unittest
from unittest.mock import Mock, patch
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
    
    def test_calculate_personality_alignment_returns_float(self):
        """Test that the function returns a float between 0 and 1"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        # Mock the OpenAI client
        with patch('app.utils.evaluation.personality_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Mock the chat completion response
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = '{"alignment_score": 0.85}'
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = calculate_personality_alignment(self.sample_response, self.sample_big_five_scores)
            
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    def test_llm_prompt_contains_required_elements(self):
        """Test that the LLM prompt contains all required elements"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        with patch('app.utils.evaluation.personality_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = '{"alignment_score": 0.85}'
            mock_client.chat.completions.create.return_value = mock_completion
            
            calculate_personality_alignment(self.sample_response, self.sample_big_five_scores)
            
            # Check that create was called
            mock_client.chat.completions.create.assert_called_once()
            
            # Get the actual call arguments
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            
            # Check system message exists
            self.assertEqual(messages[0]['role'], 'system')
            self.assertIn('personality psychologist', messages[0]['content'].lower())
            
            # Check user message contains all personality traits
            user_message = messages[1]['content']
            self.assertIn('Openness: 0.8', user_message)
            self.assertIn('Conscientiousness: 0.3', user_message)
            self.assertIn('Extraversion: 0.9', user_message)
            self.assertIn('Agreeableness: 0.6', user_message)
            self.assertIn('Neuroticism: 0.2', user_message)
            
            # Check response is included
            self.assertIn(self.sample_response, user_message)
    
    def test_handles_invalid_json_response(self):
        """Test that the function handles invalid JSON responses gracefully"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        with patch('app.utils.evaluation.personality_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Return invalid JSON
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = 'Not valid JSON'
            mock_client.chat.completions.create.return_value = mock_completion
            
            # Should return a default score (e.g., 0.5) or raise an appropriate exception
            result = calculate_personality_alignment(self.sample_response, self.sample_big_five_scores)
            self.assertEqual(result, 0.5)  # Default score for error cases
    
    def test_handles_missing_alignment_score_key(self):
        """Test that the function handles JSON without alignment_score key"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment
        
        with patch('app.utils.evaluation.personality_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Return JSON without alignment_score
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = '{"score": 0.85}'
            mock_client.chat.completions.create.return_value = mock_completion
            
            result = calculate_personality_alignment(self.sample_response, self.sample_big_five_scores)
            self.assertEqual(result, 0.5)  # Default score for error cases
    
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
    
    def test_cached_version_works(self):
        """Test that the cached version returns the same results"""
        from app.utils.evaluation.personality_metric import calculate_personality_alignment_cached
        
        with patch('app.utils.evaluation.personality_metric.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = '{"alignment_score": 0.75}'
            mock_client.chat.completions.create.return_value = mock_completion
            
            # Convert dict to tuple for cached version
            big_five_tuple = (0.8, 0.3, 0.9, 0.6, 0.2)
            
            result = calculate_personality_alignment_cached(self.sample_response, big_five_tuple)
            self.assertEqual(result, 0.75)
            
            # Second call should use cache (mock shouldn't be called again)
            result2 = calculate_personality_alignment_cached(self.sample_response, big_five_tuple)
            self.assertEqual(result2, 0.75)
            
            # Verify API was only called once due to caching
            mock_client.chat.completions.create.assert_called_once()


class TestTrainingQualityTrackerIntegration(unittest.TestCase):
    """Test integration of personality alignment metric with TrainingQualityTracker"""
    
    def test_tracker_has_personality_alignment_field(self):
        """Test that TrainingQualityTracker can track personality alignment"""
        from app.utils.metrics import TrainingQualityTracker
        
        tracker = TrainingQualityTracker()
        
        # Check that we can add personality alignment scores
        self.assertTrue(hasattr(tracker, 'add_personality_alignment_score'))
    
    def test_tracker_logs_personality_alignment_to_wandb(self):
        """Test that personality alignment is logged to WandB"""
        from app.utils.metrics import TrainingQualityTracker
        
        with patch('app.utils.metrics.wandb') as mock_wandb:
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