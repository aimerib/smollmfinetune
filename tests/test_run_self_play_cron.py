import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
import os

# Add the scripts directory to the path so we can import the module
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

class TestSelfPlayCron(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.datasets_dir = self.temp_dir / "datasets" / "self_play"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock character data
        self.char_a = {
            "name": "Alice",
            "description": "A friendly mage from the capital",
            "personality": "confident, intelligent, helpful"
        }
        
        self.char_b = {
            "name": "Bob", 
            "description": "A mysterious wanderer",
            "personality": "cautious, wise, philosophical"
        }
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_self_play_harness_arguments(self):
        """Test that harness accepts required command line arguments"""
        # This will fail until we implement the script
        try:
            import run_self_play_cron
            
            # Test argument parsing
            args = run_self_play_cron.parse_arguments([
                "--world", "Default World",
                "--charA", "Alice", 
                "--charB", "Bob",
                "--turns", "10"
            ])
            
            self.assertEqual(args.world, "Default World")
            self.assertEqual(args.charA, "Alice")
            self.assertEqual(args.charB, "Bob")
            self.assertEqual(args.turns, 10)
            
        except ImportError:
            self.fail("run_self_play_cron module should exist")
    
    @patch('run_self_play_cron.InferenceManager')
    @patch('run_self_play_cron.WorldManager')
    @patch('run_self_play_cron.CharacterManager')
    def test_dialogue_generation(self, mock_character_manager, mock_world_manager, mock_inference_manager):
        """Test that dialogue between two characters generates properly formatted output"""
        # Mock the world manager
        mock_world = MagicMock()
        mock_world_manager.return_value = mock_world
        mock_world.get_characters_path.return_value = Path("/fake/characters")
        
        # Mock the character manager
        mock_char_manager = MagicMock()
        mock_character_manager.return_value = mock_char_manager
        
        # Create mock character cores with the expected attributes
        mock_char_a = MagicMock()
        mock_char_a.name = "Alice"
        mock_char_a.description = "A friendly mage from the capital"
        mock_char_a.personality_traits = MagicMock()
        mock_char_a.personality_traits.openness = 0.8
        mock_char_a.personality_traits.conscientiousness = 0.7
        mock_char_a.personality_traits.extraversion = 0.6
        mock_char_a.personality_traits.agreeableness = 0.9
        mock_char_a.personality_traits.neuroticism = 0.3
        mock_char_a.scenario = "In a fantasy world"
        mock_char_a.backstory = "Born in the capital"
        mock_char_a.goals = "Help others with magic"
        
        mock_char_b = MagicMock()
        mock_char_b.name = "Bob"
        mock_char_b.description = "A mysterious wanderer"
        mock_char_b.personality_traits = MagicMock()
        mock_char_b.personality_traits.openness = 0.7
        mock_char_b.personality_traits.conscientiousness = 0.8
        mock_char_b.personality_traits.extraversion = 0.4
        mock_char_b.personality_traits.agreeableness = 0.6
        mock_char_b.personality_traits.neuroticism = 0.5
        mock_char_b.scenario = "Wandering the lands"
        mock_char_b.backstory = "Unknown origins"
        mock_char_b.goals = "Seek wisdom"
        
        mock_char_manager.load_character_core.side_effect = [mock_char_a, mock_char_b]
        
        # Mock the inference manager  
        mock_inference = MagicMock()
        mock_inference_manager.return_value = mock_inference
        mock_inference.generate_response.return_value = "Hello there, fellow traveler!"
        
        # Mock Path.exists() to return True for character folders
        with patch('pathlib.Path.exists', return_value=True):
            # Import and test
            import run_self_play_cron
            
            harness = run_self_play_cron.SelfPlayHarness(
                world_name="Default World",
                char_a_name="Alice",
                char_b_name="Bob"
            )
            
            # Generate dialogue (it's not async!)
            dialogue = harness.generate_dialogue(turns=3)
        
        # Validate output format matches R4 DatasetSample schema
        self.assertIsInstance(dialogue, list)
        self.assertGreater(len(dialogue), 0)
        
        # Check first sample structure
        sample = dialogue[0]
        self.assertIn('session_id', sample)
        self.assertIn('persona_mix', sample)
        self.assertIn('memory_slots', sample)
        self.assertIn('turns', sample)
        
        # Validate turns structure
        turns = sample['turns']
        self.assertGreater(len(turns), 0)
        
        for turn in turns:
            self.assertIn('sender', turn)
            self.assertIn('text', turn)
            self.assertIn('channel', turn)
            self.assertIn(turn['sender'], ['user', 'assistant'])
            self.assertIn(turn['channel'], ['text', 'action'])
    
    @patch('run_self_play_cron.wandb')
    def test_wandb_logging(self, mock_wandb):
        """Test that metrics are logged to WandB self_play_faucet project"""
        import run_self_play_cron
        
        metrics_logger = run_self_play_cron.SelfPlayMetrics()
        
        # Log some test metrics
        metrics_logger.log_generation_metrics(
            tokens_generated=5000,
            good_samples=45,
            avg_quality=7.2,
            dialogue_length=10
        )
        
        # Verify WandB was initialized with correct project (allowing for additional config)
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args
        self.assertEqual(call_args[1]['project'], "self_play_faucet")
        self.assertEqual(call_args[1]['tags'], ["self_play", "data_generation"])
        self.assertIn('config', call_args[1])  # Config is present but we don't care about exact content
        
        # Verify metrics were logged (allowing for additional timestamp field)
        mock_wandb.log.assert_called_once()
        log_call_args = mock_wandb.log.call_args
        logged_metrics = log_call_args[0][0]  # First positional argument
        
        # Check the expected metrics are present
        self.assertEqual(logged_metrics['tokens_generated'], 5000)
        self.assertEqual(logged_metrics['good_samples'], 45)
        self.assertEqual(logged_metrics['avg_quality'], 7.2)
        self.assertEqual(logged_metrics['dialogue_length'], 10)
        self.assertEqual(logged_metrics['acceptance_rate'], 0.9)
        self.assertIn('timestamp', logged_metrics)  # Timestamp is present but we don't care about exact value
    
    def test_r4_schema_validation(self):
        """Test that generated samples are validated against R4 schema"""
        import run_self_play_cron
        
        # Create test sample that should be valid
        test_sample = {
            'session_id': 'test_123',
            'persona_mix': {'Alice': 1.0},
            'memory_slots': [],
            'turns': [
                {
                    'sender': 'user',
                    'text': 'Hello',
                    'channel': 'text'
                }
            ]
        }
        
        validator = run_self_play_cron.SelfPlayValidator()
        result = validator.validate_samples([test_sample])
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['valid_count'], 1)
        self.assertEqual(result['total_count'], 1)
        self.assertEqual(len(result['errors']), 0)
        
        # Test with invalid sample
        invalid_sample = {
            'session_id': 'test_123',
            # Missing required fields
            'turns': []  # Invalid - must have at least 1 turn
        }
        
        result_invalid = validator.validate_samples([invalid_sample])
        self.assertFalse(result_invalid['valid'])
        self.assertEqual(result_invalid['valid_count'], 0)
        self.assertGreater(len(result_invalid['errors']), 0)
    
    def test_quality_filtering(self):
        """Test that quality filters remove samples with poor metrics"""
        import run_self_play_cron
        
        # Create test samples with different quality scores
        samples = [
            {'quality_score': 0.9, 'json_correctness': 0.95, 'length': 6},  # Good
            {'quality_score': 0.5, 'json_correctness': 0.89, 'length': 2},  # Poor quality
            {'quality_score': 0.8, 'json_correctness': 0.85, 'length': 5},  # Poor JSON
            {'quality_score': 0.85, 'json_correctness': 0.92, 'length': 8}  # Good
        ]
        
        filter_engine = run_self_play_cron.QualityFilter()
        filtered = filter_engine.filter_samples(samples)
        
        # Should keep samples with JSON correctness >= 90% and length >= 4 turns
        self.assertEqual(len(filtered), 2)  # Only first and last samples
    
    def test_control_token_injection(self):
        """Test that control tokens are injected every 10 turns"""
        import run_self_play_cron
        
        dialogue_generator = run_self_play_cron.DialogueGenerator()
        
        # Test that scene tokens are added at appropriate intervals
        with patch.object(dialogue_generator, '_inject_scene_token') as mock_inject:
            mock_inject.return_value = "<scene_night>"
            
            # Generate 25 turns - should inject tokens at turns 10 and 20
            for turn in range(25):
                dialogue_generator.maybe_inject_control_token(turn)
            
            # Should be called twice (turn 10 and 20)
            self.assertEqual(mock_inject.call_count, 2)
    
    def test_output_jsonl_format(self):
        """Test that output is saved in JSONL format with correct naming"""
        import run_self_play_cron
        from datetime import datetime
        
        # Mock datetime to get predictable filename
        with patch('run_self_play_cron.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250127"
            
            output_manager = run_self_play_cron.OutputManager(str(self.datasets_dir))
            
            # Test data
            samples = [
                {'session_id': 'test1', 'persona_mix': {}, 'memory_slots': [], 'turns': []},
                {'session_id': 'test2', 'persona_mix': {}, 'memory_slots': [], 'turns': []}
            ]
            
            filename = output_manager.save_samples(samples)
            
            # Check filename format
            expected_file = self.datasets_dir / "nightly_20250127.jsonl"
            self.assertEqual(Path(filename), expected_file)
            
            # Check file exists and has correct format
            self.assertTrue(expected_file.exists())
            
            with open(expected_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
                
                # Each line should be valid JSON
                for line in lines:
                    sample = json.loads(line.strip())
                    self.assertIn('session_id', sample)


class TestSelfPlayIntegration(unittest.TestCase):
    """Integration tests for the complete self-play pipeline"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('run_self_play_cron.CharacterManager')
    @patch('run_self_play_cron.InferenceManager')
    @patch('run_self_play_cron.WorldManager') 
    @patch('run_self_play_cron.wandb')
    def test_complete_pipeline(self, mock_wandb, mock_world_manager, mock_inference_manager, mock_character_manager):
        """Test the complete self-play pipeline from start to finish"""
        # Set up mocks
        mock_world = MagicMock()
        mock_world_manager.return_value = mock_world
        mock_world.get_characters_path.return_value = Path("/fake/characters")
        
        mock_char_manager = MagicMock()
        mock_character_manager.return_value = mock_char_manager
        
        # Create mock character cores (same as above)
        mock_char_a = MagicMock()
        mock_char_a.name = "Alice"
        mock_char_a.description = "Mage"
        mock_char_a.personality_traits = MagicMock()
        mock_char_a.personality_traits.openness = 0.8
        mock_char_a.personality_traits.conscientiousness = 0.7
        mock_char_a.personality_traits.extraversion = 0.6
        mock_char_a.personality_traits.agreeableness = 0.9
        mock_char_a.personality_traits.neuroticism = 0.3
        mock_char_a.scenario = "In a fantasy world"
        mock_char_a.backstory = "A skilled mage"
        mock_char_a.goals = "Protect the realm"
        
        mock_char_b = MagicMock()
        mock_char_b.name = "Bob"
        mock_char_b.description = "Warrior"
        mock_char_b.personality_traits = MagicMock()
        mock_char_b.personality_traits.openness = 0.7
        mock_char_b.personality_traits.conscientiousness = 0.8
        mock_char_b.personality_traits.extraversion = 0.5
        mock_char_b.personality_traits.agreeableness = 0.6
        mock_char_b.personality_traits.neuroticism = 0.4
        mock_char_b.scenario = "In battle"
        mock_char_b.backstory = "A brave warrior"
        mock_char_b.goals = "Defend the innocent"
        
        mock_char_manager.load_character_core.side_effect = [mock_char_a, mock_char_b]
        
        mock_inference = MagicMock()
        mock_inference_manager.return_value = mock_inference
        mock_inference.generate_response.side_effect = [
            "Hello Bob, how are you today?",
            "I'm doing well Alice, thanks for asking!",
            "That's great to hear!",
            "Indeed it is!"
        ]
        
        # Mock Path.exists() for character folders
        with patch('pathlib.Path.exists', return_value=True):
            import run_self_play_cron
            
            # Run complete pipeline (not async!)
            result = run_self_play_cron.run_self_play_session(
                world_name="Default World",
                char_a_name="Alice",
                char_b_name="Bob", 
                turns=4,
                output_dir=str(self.temp_dir)
            )
        
        # Verify success
        self.assertTrue(result['success'])
        self.assertGreater(result['tokens_generated'], 0)
        self.assertGreater(result['samples_generated'], 0)
        
        # Verify output file exists
        output_files = list(self.temp_dir.glob("*.jsonl"))
        self.assertEqual(len(output_files), 1)


if __name__ == '__main__':
    unittest.main() 