"""
Test integration of personality alignment metric with the training pipeline
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import queue
from typing import Dict, Any


class TestPersonalityAlignmentTrainingIntegration(unittest.TestCase):
    """Test that personality alignment metric is properly integrated into training"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.character_with_big_five = {
            'name': 'TestCharacter',
            'personality': 'Creative and introverted artist',
            'big_five_scores': {
                'openness': 0.9,
                'conscientiousness': 0.4, 
                'extraversion': 0.2,
                'agreeableness': 0.7,
                'neuroticism': 0.8
            }
        }
        
        self.character_without_big_five = {
            'name': 'LegacyCharacter',
            'personality': 'Traditional character without Big-Five scores'
        }
        
        self.sample_response = "*nervously fidgets with paintbrush* I... I was just working on something creative. It's probably not very good though..."
    
    def test_training_callback_evaluates_personality_alignment(self):
        """Test that TrainingCallback evaluates personality alignment during training"""
        from app.utils.training import TrainingCallback
        from app.utils.metrics import TrainingQualityTracker
        
        # Set up callback with character that has Big-Five scores
        status_queue = queue.Queue()
        quality_tracker = TrainingQualityTracker()
        
        callback = TrainingCallback(
            status_queue=status_queue,
            character=self.character_with_big_five,
            quality_tracker=quality_tracker,
            log_interval=10
        )
        
        # Mock training state and components
        mock_state = Mock()
        mock_state.global_step = 50  # Trigger evaluation (50 % (10 * 5) == 0)
        
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = f"User: Hello!\nAssistant: {self.sample_response}"
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(return_value={'input_ids': [1, 2, 3]})
        
        mock_dataloader = Mock()
        mock_dataloader.dataset = mock_dataset
        
        # Mock the personality alignment calculation
        with patch('app.utils.evaluation.personality_metric.calculate_personality_alignment') as mock_calc:
            mock_calc.return_value = 0.85
            
            # Call the evaluation method
            callback.on_evaluate(
                args=None,
                state=mock_state,
                control=None,
                model=Mock(),
                tokenizer=mock_tokenizer,
                eval_dataloader=mock_dataloader
            )
        
        # Verify personality alignment was calculated
        mock_calc.assert_called()
        call_args = mock_calc.call_args
        self.assertIn(self.sample_response.strip(), call_args[0][0])  # Response content
        self.assertEqual(call_args[0][1], self.character_with_big_five['big_five_scores'])  # Big-Five scores
        
        # Verify quality tracker received the score
        self.assertEqual(len(quality_tracker.personality_alignment_scores), 1)
        self.assertEqual(quality_tracker.personality_alignment_scores[0], 0.85)
        
        # Verify status queue received the metrics
        status_updates = []
        while not status_queue.empty():
            status_updates.append(status_queue.get_nowait())
        
        consistency_update = [u for u in status_updates if u.get('type') == 'consistency_evaluation'][0]
        self.assertIn('avg_personality_alignment', consistency_update)
        self.assertEqual(consistency_update['avg_personality_alignment'], 0.85)
    
    def test_training_callback_skips_personality_alignment_without_big_five(self):
        """Test that TrainingCallback skips personality alignment for characters without Big-Five scores"""
        from app.utils.training import TrainingCallback
        from app.utils.metrics import TrainingQualityTracker
        
        # Set up callback with character that lacks Big-Five scores
        status_queue = queue.Queue()
        quality_tracker = TrainingQualityTracker()
        
        callback = TrainingCallback(
            status_queue=status_queue,
            character=self.character_without_big_five,
            quality_tracker=quality_tracker,
            log_interval=10
        )
        
        # Mock training state and components
        mock_state = Mock()
        mock_state.global_step = 50
        
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = f"User: Hello!\nAssistant: {self.sample_response}"
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(return_value={'input_ids': [1, 2, 3]})
        
        mock_dataloader = Mock()
        mock_dataloader.dataset = mock_dataset
        
        # Mock the personality alignment calculation
        with patch('app.utils.evaluation.personality_metric.calculate_personality_alignment') as mock_calc:
            # Call the evaluation method
            callback.on_evaluate(
                args=None,
                state=mock_state,
                control=None,
                model=Mock(),
                tokenizer=mock_tokenizer,
                eval_dataloader=mock_dataloader
            )
        
        # Verify personality alignment was NOT calculated
        mock_calc.assert_not_called()
        
        # Verify quality tracker has no personality alignment scores
        self.assertEqual(len(quality_tracker.personality_alignment_scores), 0)
        
        # Verify status queue does not include personality alignment metrics
        status_updates = []
        while not status_queue.empty():
            status_updates.append(status_queue.get_nowait())
        
        consistency_update = [u for u in status_updates if u.get('type') == 'consistency_evaluation'][0]
        self.assertIsNone(consistency_update.get('avg_personality_alignment'))
    
    def test_wandb_metrics_include_personality_alignment(self):
        """Test that WandB metrics include personality alignment when available"""
        from app.utils.metrics import TrainingQualityTracker
        
        tracker = TrainingQualityTracker()
        
        # Add some personality alignment scores
        tracker.add_personality_alignment_score(0.85)
        tracker.add_personality_alignment_score(0.90)
        tracker.add_personality_alignment_score(0.88)
        
        # Get WandB metrics
        metrics = tracker.get_wandb_metrics()
        
        # Verify personality alignment metrics are included
        self.assertIn('avg_personality_alignment', metrics)
        self.assertIn('recent_personality_alignment', metrics)
        self.assertAlmostEqual(metrics['avg_personality_alignment'], 0.877, places=2)
        self.assertAlmostEqual(metrics['recent_personality_alignment'], 0.877, places=2)
    
    def test_training_manager_status_processing_includes_personality_alignment(self):
        """Test that TrainingManager properly processes personality alignment status updates"""
        from app.utils.training import TrainingManager
        
        manager = TrainingManager()
        
        # Simulate a consistency evaluation status with personality alignment
        status = {
            'type': 'consistency_evaluation',
            'step': 100,
            'avg_consistency': 0.75,
            'avg_personality_alignment': 0.88,
            'evaluated_samples': []
        }
        
        # Process the status update
        manager._process_status_update(status)
        
        # Verify personality alignment metrics are in current_metrics
        self.assertIn('avg_personality_alignment', manager.current_metrics)
        self.assertEqual(manager.current_metrics['avg_personality_alignment'], 0.88)
        self.assertIn('personality_alignment_last_eval_step', manager.current_metrics)
        self.assertEqual(manager.current_metrics['personality_alignment_last_eval_step'], 100)


if __name__ == '__main__':
    unittest.main() 