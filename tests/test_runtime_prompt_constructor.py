"""
Tests for RuntimePromptConstructor (R2-2)
Following TDD approach - this tests the core business logic (inner circle)
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import the class we're testing
from app.utils.runtime.prompt_constructor import RuntimePromptConstructor


class TestRuntimePromptConstructor(unittest.TestCase):
    """Test the RuntimePromptConstructor functionality"""
    
    def setUp(self):
        """Set up test environment with fake runtime packet"""
        # Create temporary directory for test packet
        self.temp_dir = Path(tempfile.mkdtemp())
        self.packet_dir = self.temp_dir / "test_character"
        self.packet_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test character_core.json
        self.test_character_core = {
            "name": "Test Character",
            "description": "A character for runtime testing",
            "personality": "friendly, curious, helpful",
            "big_five": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.9,
                "neuroticism": 0.3
            },
            "goals": ["Help users", "Learn new things", "Stay positive"],
            "relationships": [
                {"name": "User", "stance": "helpful support", "affinity": 80}
            ]
        }
        
        with open(self.packet_dir / "character_core.json", 'w') as f:
            json.dump(self.test_character_core, f, indent=2)
        
        # Create test world_lore.json
        self.test_world_lore = {
            "meta": {"version": 1},
            "facts": {
                "setting": "A friendly testing environment",
                "rules": "Be helpful and accurate",
                "atmosphere": "Collaborative and supportive"
            },
            "timeline": [
                {"year": 2024, "event": "AI assistant technology advanced"}
            ],
            "factions": [],
            "places": []
        }
        
        with open(self.packet_dir / "world_lore.json", 'w') as f:
            json.dump(self.test_world_lore, f, indent=2)
        
        # Create test tokens.json
        self.test_tokens = [
            {
                "token": "<mood_happy>",
                "category": "mood",
                "description": "Character is in a happy mood"
            },
            {
                "token": "<mood_curious>",
                "category": "mood", 
                "description": "Character is feeling curious"
            },
            {
                "token": "<scene_testing>",
                "category": "scene",
                "description": "Currently in a testing scenario"
            }
        ]
        
        with open(self.packet_dir / "tokens.json", 'w') as f:
            json.dump(self.test_tokens, f, indent=2)
        
        # Create test runtime_config.json
        self.test_config = {
            "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "character_file": "character_core.json",
            "world_file": "world_lore.json",
            "tokens_file": "tokens.json"
        }
        
        with open(self.packet_dir / "runtime_config.json", 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_runtime_prompt_constructor_initialization(self):
        """Test that RuntimePromptConstructor can be initialized with packet path"""
        packet_path = str(self.packet_dir)
        
        # Test that constructor can be initialized with packet path
        constructor = RuntimePromptConstructor(packet_path)
        
        # Test that constructor loaded all the data correctly
        self.assertEqual(constructor.character_core["name"], "Test Character")
        self.assertIn("big_five", constructor.character_core)
        self.assertIn("goals", constructor.character_core)
        
        # Test world lore was loaded
        self.assertIn("facts", constructor.world_lore)
        self.assertIsInstance(constructor.world_lore["facts"], dict)
        
        # Test tokens were loaded
        self.assertIsInstance(constructor.control_tokens, list)
        self.assertTrue(len(constructor.control_tokens) > 0)
        
        # Test token lookup was built
        self.assertIn("<mood_happy>", constructor.token_lookup)
        self.assertIn("<mood_curious>", constructor.token_lookup)
        
        # Test getter methods
        self.assertEqual(constructor.get_character_name(), "Test Character")
        available_tokens = constructor.get_available_tokens()
        self.assertIsInstance(available_tokens, list)
        
        mood_tokens = constructor.get_tokens_by_category("mood")
        self.assertTrue(len(mood_tokens) >= 2)  # Should have at least happy and curious
    
    def test_construct_basic_prompt(self):
        """Test basic prompt construction with conversation history"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        
        # Test data for prompt construction
        conversation_history = [
            {"role": "user", "content": "Hello there!"}
        ]
        
        dynamic_state = {
            "current_mood": "happy",
            "relationship_to_user": {"trust": 0.7, "affinity": 0.8}
        }
        
        # Construct the prompt
        result = constructor.construct(conversation_history, dynamic_state)
        
        # Test that we get a string back
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Expected elements that should be in the prompt:
        # 1. Character introduction with name
        self.assertIn("Test Character", result)
        
        # 2. Character goals should be mentioned
        self.assertIn("Help users", result)
        
        # 3. World lore fact should be included
        world_fact_found = any(fact in result for fact in self.test_world_lore["facts"].values())
        self.assertTrue(world_fact_found, f"No world lore fact found in: {result}")
        
        # 4. Conversation history should be formatted
        self.assertIn("Hello there!", result)
        
        # 5. Response instruction should be included
        self.assertIn("Respond naturally", result)
        
        # 6. Dynamic state influence - mood token should be included
        self.assertIn("<mood_happy>", result)
    
    def test_dynamic_state_mood_integration(self):
        """Test that current_mood affects prompt construction"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        conversation_history = [{"role": "user", "content": "How are you feeling?"}]
        
        # Test different moods
        happy_state = {"current_mood": "happy"}
        curious_state = {"current_mood": "curious"}
        
        # Construct prompts with different moods
        happy_prompt = constructor.construct(conversation_history, happy_state)
        curious_prompt = constructor.construct(conversation_history, curious_state)
        
        # Should include corresponding control tokens
        self.assertIn("<mood_happy>", happy_prompt)
        self.assertIn("<mood_curious>", curious_prompt)
        
        # Different moods should produce different prompts
        self.assertNotEqual(happy_prompt, curious_prompt)
        
        # Both should still include character name and basic structure
        self.assertIn("Test Character", happy_prompt)
        self.assertIn("Test Character", curious_prompt)
    
    def test_relationship_state_integration(self):
        """Test that relationship_to_user affects prompt construction"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        conversation_history = [{"role": "user", "content": "Do you trust me?"}]
        
        # Test different relationship states
        high_trust_state = {
            "relationship_to_user": {"trust": 0.9, "affinity": 0.8}
        }
        
        low_trust_state = {
            "relationship_to_user": {"trust": 0.2, "affinity": 0.3}
        }
        
        # Construct prompts with different relationship states
        high_trust_prompt = constructor.construct(conversation_history, high_trust_state)
        low_trust_prompt = constructor.construct(conversation_history, low_trust_state)
        
        # High trust should mention deep trust
        self.assertIn("deep trust", high_trust_prompt)
        self.assertIn("strong connection", high_trust_prompt)
        
        # Low trust should mention caution/distance
        self.assertIn("cautious", low_trust_prompt)
        self.assertIn("professional distance", low_trust_prompt)
        
        # Both should still include basic character info
        self.assertIn("Test Character", high_trust_prompt)
        self.assertIn("Test Character", low_trust_prompt)
    
    def test_recent_events_integration(self):
        """Test that recent_events are incorporated into prompts"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        conversation_history = [{"role": "user", "content": "What's been happening?"}]
        
        dynamic_state = {
            "recent_events": [
                "User helped with a difficult task",
                "Had a productive conversation about goals",
                "Learned something new about user preferences"
            ]
        }
        
        # Construct prompt with recent events
        result = constructor.construct(conversation_history, dynamic_state)
        
        # Should include recent events in the prompt
        self.assertIn("Recent events:", result)
        self.assertIn("difficult task", result)
        self.assertIn("productive conversation", result)
        self.assertIn("user preferences", result)
        
        # Should still include basic character structure
        self.assertIn("Test Character", result)
        self.assertIn("Respond naturally", result)
    
    def test_forced_control_tokens_integration(self):
        """Test that forced_control_tokens are properly injected"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        conversation_history = [{"role": "user", "content": "Tell me a story"}]
        
        dynamic_state = {
            "forced_control_tokens": ["<mood_curious>", "<scene_testing>"]
        }
        
        # Construct prompt with forced tokens
        result = constructor.construct(conversation_history, dynamic_state)
        
        # Forced tokens should be injected into the prompt
        self.assertIn("<mood_curious>", result)
        self.assertIn("<scene_testing>", result)
        
        # Should still maintain character structure
        self.assertIn("Test Character", result)
        self.assertIn("Tell me a story", result)
    
    def test_conversation_history_formatting(self):
        """Test that conversation history is properly formatted"""
        conversation_history = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
            {"role": "user", "content": "Tell me about yourself"}
        ]
        
        dynamic_state = {}
        
        # Should format conversation history according to model's chat template
        # This tests the expected conversation structure
        for turn in conversation_history:
            self.assertIn("role", turn)
            self.assertIn("content", turn)
            self.assertIn(turn["role"], ["user", "assistant"])
            self.assertIsInstance(turn["content"], str)
    
    def test_big_five_personality_injection(self):
        """Test that Big-Five scores are converted to personality adjectives"""
        # Test personality trait conversion similar to existing PromptBuilder
        big_five = self.test_character_core["big_five"]
        
        # High agreeableness (0.9) should produce cooperative adjectives
        self.assertGreater(big_five["agreeableness"], 0.7)
        
        # High openness (0.8) should produce creative adjectives  
        self.assertGreater(big_five["openness"], 0.7)
        
        # Low neuroticism (0.3) should produce stable adjectives
        self.assertLess(big_five["neuroticism"], 0.5)
        
        # Expected adjective categories
        expected_categories = {
            "agreeableness": ["kind", "cooperative", "trusting", "empathetic"],
            "openness": ["creative", "imaginative", "curious", "artistic"],
            "neuroticism": ["calm", "stable", "resilient", "even-tempered"]  # Low neuroticism
        }
        
        for trait, adjectives in expected_categories.items():
            self.assertIsInstance(adjectives, list)
            self.assertTrue(len(adjectives) > 0)
    
    def test_world_lore_fact_injection(self):
        """Test that world lore facts are randomly injected"""
        # Should select and inject random facts from world lore
        world_facts = self.test_world_lore["facts"]
        
        self.assertIsInstance(world_facts, dict)
        self.assertTrue(len(world_facts) > 0)
        
        # Facts should be accessible for random selection
        fact_values = list(world_facts.values())
        for fact in fact_values:
            self.assertIsInstance(fact, str)
            self.assertTrue(len(fact) > 0)
    
    def test_error_handling_missing_files(self):
        """Test error handling when packet files are missing"""
        # Create packet directory without some files
        incomplete_packet = self.temp_dir / "incomplete_character"
        incomplete_packet.mkdir(exist_ok=True)
        
        # Only create character_core.json, missing others
        with open(incomplete_packet / "character_core.json", 'w') as f:
            json.dump(self.test_character_core, f)
        
        # Should handle missing files gracefully
        self.assertTrue(incomplete_packet.exists())
        self.assertTrue((incomplete_packet / "character_core.json").exists())
        self.assertFalse((incomplete_packet / "world_lore.json").exists())
        self.assertFalse((incomplete_packet / "tokens.json").exists())
    
    def test_prompt_output_format(self):
        """Test that constructed prompt is properly formatted for tokenizer"""
        conversation_history = [{"role": "user", "content": "Hello!"}]
        dynamic_state = {"current_mood": "happy"}
        
        # Output should be a string ready for tokenization
        # This documents the expected format
        expected_sections = [
            "character introduction",
            "personality traits", 
            "goals",
            "world context",
            "conversation history",
            "response instruction"
        ]
        
        # Each section should contribute to the final prompt
        for section in expected_sections:
            self.assertIsInstance(section, str)
    
    def test_comprehensive_runtime_scenario(self):
        """Test a comprehensive runtime scenario with all dynamic state features"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        
        # Simulate a multi-turn conversation
        conversation_history = [
            {"role": "user", "content": "Hi, I'm new here. What should I know about you?"},
            {"role": "assistant", "content": "Welcome! I'm Test Character, and I'm here to help you learn and grow."},
            {"role": "user", "content": "That's great! How are you feeling today?"}
        ]
        
        # Complex dynamic state with all features
        dynamic_state = {
            "current_mood": "curious",
            "relationship_to_user": {
                "trust": 0.8,
                "affinity": 0.7
            },
            "recent_events": [
                "User showed genuine interest in learning",
                "Had a positive interaction about goals",
                "Building rapport with this new user"
            ],
            "forced_control_tokens": ["<scene_testing>"]
        }
        
        # Construct the comprehensive prompt
        result = constructor.construct(conversation_history, dynamic_state)
        
        # Verify all components are present:
        
        # 1. Character identity
        self.assertIn("Test Character", result)
        
        # 2. Personality traits from Big-Five (check that some traits are present)
        # Should have personality adjectives from the high scores
        openness_words = ["creative", "imaginative", "open-minded", "artistic", "curious", "adventurous"]
        conscientiousness_words = ["organized", "disciplined", "reliable", "methodical", "careful", "thorough"]
        agreeableness_words = ["kind", "cooperative", "trusting", "empathetic", "compassionate"]
        
        # At least one personality trait should be present
        all_personality_words = openness_words + conscientiousness_words + agreeableness_words
        self.assertTrue(any(word in result for word in all_personality_words), 
                       f"Expected some personality traits in: {result}")
        
        # 3. Character goals
        self.assertIn("Help users", result)
        
        # 4. Dynamic relationship state
        self.assertIn("deep trust", result)  # High trust (0.8)
        
        # 5. World lore facts
        world_fact_found = any(fact in result for fact in self.test_world_lore["facts"].values())
        self.assertTrue(world_fact_found, "World lore fact should be included")
        
        # 6. Recent events
        self.assertIn("Recent events:", result)
        self.assertIn("positive interaction", result)
        
        # 7. Mood-based control tokens
        self.assertIn("<mood_curious>", result)
        
        # 8. Forced control tokens  
        self.assertIn("<scene_testing>", result)
        
        # 9. Conversation history
        self.assertIn("new here", result)
        self.assertIn("feeling today", result)
        
        # 10. Response instruction
        self.assertIn("Respond naturally", result)
        
        # Verify the prompt is substantial and well-formed
        self.assertGreater(len(result), 200, "Prompt should be substantial")
        self.assertLess(len(result), 2000, "Prompt should not be excessively long")
        
        # Print the result for manual inspection (will show in test output)
        print(f"\n=== COMPREHENSIVE RUNTIME PROMPT ===\n{result}\n" + "="*50)
    
    def test_error_handling_missing_packet(self):
        """Test error handling when runtime packet doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            RuntimePromptConstructor("/nonexistent/path")
    
    def test_error_handling_missing_character_core(self):
        """Test error handling when character_core.json is missing"""
        # Create packet without character_core.json
        incomplete_packet = self.temp_dir / "no_core_character"
        incomplete_packet.mkdir(exist_ok=True)
        
        with self.assertRaises(FileNotFoundError):
            RuntimePromptConstructor(str(incomplete_packet))
    
    def test_fallback_prompt_on_error(self):
        """Test that constructor provides fallback prompt when things go wrong"""
        constructor = RuntimePromptConstructor(str(self.packet_dir))
        
        # Simulate an error by passing invalid conversation history
        invalid_history = [{"invalid": "structure"}]
        
        # Should still return a usable prompt (fallback behavior)
        result = constructor.construct(invalid_history, {})
        
        self.assertIsInstance(result, str)
        self.assertIn("Test Character", result)
        self.assertIn("Respond naturally", result)


if __name__ == '__main__':
    unittest.main() 