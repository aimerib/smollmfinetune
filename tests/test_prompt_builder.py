"""
Unit tests for the Prompt Builder functionality.

Tests the core prompt building features including Big-5 personality injection,
lore facts, NSFW style tags, and multi-turn conversation support.
"""

import pytest
from unittest.mock import Mock, patch
import json

from app.utils.generation.prompt_builder import PromptBuilder, build_prompt, build_conversation_turn
from app.utils.character.models import CharacterCore


class TestPromptBuilder:
    """Test the PromptBuilder class functionality"""
    
    @pytest.fixture
    def sample_character(self):
        """Sample character with Big-5 traits and world context"""
        return {
            'name': 'Cricket',
            'description': 'A struggling debt collection agent',
            'personality': 'anxious, determined, empathetic',
            'big_five': {
                'openness': 0.8,
                'conscientiousness': 0.8, 
                'extraversion': 0.2,
                'agreeableness': 0.9,
                'neuroticism': 0.8
            },
            'goals': ['Pay off debt', 'Keep agency running', 'Help clients'],
            'relationships': [
                {'name': 'Boss', 'type': 'professional', 'stance': 'respectful fear'},
                {'name': 'Clients', 'type': 'work', 'stance': 'empathetic helper'}
            ]
        }
    
    @pytest.fixture
    def sample_world_lore(self):
        """Sample world lore with facts"""
        return {
            'name': 'Modern Urban',
            'facts': [
                'Debt collection agencies operate in legal gray areas',
                'Economic inequality drives desperate measures',
                'Technology makes tracking debtors easier',
                'Personal relationships often complicate business'
            ]
        }
    
    @pytest.fixture
    def prompt_builder(self, sample_world_lore):
        """Create a PromptBuilder instance with world context"""
        return PromptBuilder(world_lore=sample_world_lore)
    
    def test_prompt_builder_initialization(self, sample_world_lore):
        """Test PromptBuilder can be initialized with world lore"""
        builder = PromptBuilder(world_lore=sample_world_lore)
        assert builder.world_lore == sample_world_lore
        assert hasattr(builder, 'big_five_adjectives')
    
    def test_build_prompt_chat_mode(self, prompt_builder, sample_character):
        """Test building a chat mode prompt with personality injection"""
        result = prompt_builder.build_prompt(
            character=sample_character,
            mode="chat",
            base_prompt="Tell me about your work"
        )
        
        # Should contain Big-5 personality adjectives (check for any from the high/low categories)
        openness_words = ['creative', 'imaginative', 'open-minded', 'artistic', 'curious', 'adventurous']
        conscientiousness_words = ['organized', 'disciplined', 'reliable', 'methodical', 'careful', 'thorough']
        extraversion_words = ['introverted', 'reserved', 'quiet', 'thoughtful', 'contemplative']  # Low extraversion
        agreeableness_words = ['kind', 'cooperative', 'trusting', 'empathetic', 'compassionate']
        neuroticism_words = ['anxious', 'worried', 'stressed', 'emotional', 'sensitive']
        
        # Due to randomness, test that at least some personality traits are included
        all_expected_words = openness_words + conscientiousness_words + extraversion_words + agreeableness_words + neuroticism_words
        assert any(adj in result for adj in all_expected_words), f"Expected some personality adjectives in: {result}"
        
        # Test specific high-scoring dimensions (more reliable)
        assert any(adj in result for adj in openness_words) or any(adj in result for adj in conscientiousness_words) or any(adj in result for adj in agreeableness_words)
        
        # Should contain top goals
        assert 'Pay off debt' in result
        assert 'Keep agency running' in result
        
        # Should contain relationship stance
        assert any(stance in result for stance in ['respectful fear', 'empathetic helper'])
        
        # Should contain a random lore fact
        assert any(fact in result for fact in [
            'Debt collection agencies operate in legal gray areas',
            'Economic inequality drives desperate measures',
            'Technology makes tracking debtors easier',
            'Personal relationships often complicate business'
        ])
        
        # Should NOT contain NSFW tags in chat mode
        assert '[NSFW:' not in result
    
    def test_build_prompt_nsfw_mode_soft(self, prompt_builder, sample_character):
        """Test building NSFW prompt with soft style tag"""
        result = prompt_builder.build_prompt(
            character=sample_character,
            mode="nsfw",
            nsfw_style="soft",
            base_prompt="How do you feel about intimacy?"
        )
        
        # Should contain NSFW tag with specified style
        assert '[NSFW:soft]' in result
        
        # Should still contain personality and lore elements
        personality_words = ['anxious', 'worried', 'stressed', 'emotional', 'sensitive', 'creative', 'organized', 'kind', 'reserved']
        assert any(adj in result for adj in personality_words), f"Expected personality adjectives in: {result}"
        assert any(goal in result for goal in sample_character.get('goals', []))
    
    def test_build_prompt_nsfw_mode_explicit(self, prompt_builder, sample_character):
        """Test building NSFW prompt with explicit style tag"""
        result = prompt_builder.build_prompt(
            character=sample_character,
            mode="nsfw", 
            nsfw_style="explicit",
            base_prompt="Tell me your desires"
        )
        
        assert '[NSFW:explicit]' in result
    
    def test_build_prompt_nsfw_mode_kink(self, prompt_builder, sample_character):
        """Test building NSFW prompt with kink style tag"""
        result = prompt_builder.build_prompt(
            character=sample_character,
            mode="nsfw",
            nsfw_style="kink", 
            base_prompt="What are your hidden fantasies?"
        )
        
        assert '[NSFW:kink]' in result
    
    def test_build_prompt_qa_mode(self, prompt_builder, sample_character):
        """Test building QA mode prompt for factual questioning"""
        result = prompt_builder.build_prompt(
            character=sample_character,
            mode="qa",
            base_prompt="What is your occupation?"
        )
        
        # Should contain factual elements but be more structured for QA
        assert sample_character['name'] in result
        assert any(goal in result for goal in sample_character['goals'])
        
        # Should NOT contain NSFW tags in QA mode
        assert '[NSFW:' not in result
    
    def test_build_prompt_missing_elements(self, prompt_builder):
        """Test graceful handling when character is missing elements"""
        minimal_character = {
            'name': 'TestChar',
            'description': 'Basic character'
        }
        
        result = prompt_builder.build_prompt(
            character=minimal_character,
            mode="chat",
            base_prompt="Hello"
        )
        
        # Should not crash and should still build a prompt
        assert isinstance(result, str)
        assert len(result) > 0
        assert minimal_character['name'] in result
    
    def test_build_conversation_turn(self):
        """Test building individual conversation turns with metadata"""
        turn_data = build_conversation_turn(
            turn_idx=1,
            role="user", 
            text="What's your favorite memory?",
            speaker_name="Player",
            emotion="curious"
        )
        
        expected_keys = ['turn_id', 'role', 'content', 'metadata']
        for key in expected_keys:
            assert key in turn_data
        
        assert turn_data['turn_id'] == 1
        assert turn_data['role'] == "user"
        assert turn_data['content'] == "What's your favorite memory?"
        assert turn_data['metadata']['speaker_name'] == "Player"
        assert turn_data['metadata']['emotion'] == "curious"
    
    def test_build_conversation_turn_assistant(self):
        """Test building assistant turns with character context"""
        turn_data = build_conversation_turn(
            turn_idx=2,
            role="assistant",
            text="I remember the day I first helped someone avoid bankruptcy...",
            character_name="Cricket",
            mood="nostalgic"
        )
        
        assert turn_data['turn_id'] == 2
        assert turn_data['role'] == "assistant"
        assert "remember the day" in turn_data['content']
        assert turn_data['metadata']['character_name'] == "Cricket" 
        assert turn_data['metadata']['mood'] == "nostalgic"
    
    def test_big_five_to_adjectives_mapping(self, prompt_builder):
        """Test that Big-5 scores map to appropriate adjectives"""
        # Test that high scores produce adjectives (exact ones may vary due to randomness)
        high_openness_adj = prompt_builder._big_five_to_adjectives({'openness': 0.9})
        assert len(high_openness_adj) > 0, "High openness should produce adjectives"
        
        # Test low scores  
        low_extraversion_adj = prompt_builder._big_five_to_adjectives({'extraversion': 0.1})
        assert len(low_extraversion_adj) > 0, "Low extraversion should produce adjectives"
        
        # Test medium scores
        medium_conscientiousness_adj = prompt_builder._big_five_to_adjectives({'conscientiousness': 0.5})
        assert len(medium_conscientiousness_adj) > 0, "Medium conscientiousness should produce adjectives"
        
        # Test that different scores produce different levels
        high_neuro = prompt_builder._score_to_level(0.9)
        low_neuro = prompt_builder._score_to_level(0.1)
        assert high_neuro != low_neuro, "High and low scores should map to different levels"
    
    def test_random_lore_fact_selection(self, prompt_builder, sample_character):
        """Test that lore facts are randomly selected"""
        facts_selected = set()
        
        # Generate multiple prompts to test randomness
        for _ in range(10):
            result = prompt_builder.build_prompt(
                character=sample_character,
                mode="chat",
                base_prompt="Tell me something interesting"
            )
            
            # Extract which fact was used
            for fact in prompt_builder.world_lore['facts']:
                if fact in result:
                    facts_selected.add(fact)
                    break
        
        # Should have used multiple different facts across runs
        assert len(facts_selected) > 1
    
    def test_prompt_caching(self, prompt_builder, sample_character):
        """Test that prompts can be cached for performance"""
        # This test ensures we can implement caching later
        prompt1 = prompt_builder.build_prompt(
            character=sample_character,
            mode="chat",
            base_prompt="Hello",
            use_cache=True
        )
        
        prompt2 = prompt_builder.build_prompt(
            character=sample_character, 
            mode="chat",
            base_prompt="Hello",
            use_cache=True
        )
        
        # For now, just test that both complete successfully
        # Later we can add actual caching logic
        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)


class TestPromptBuilderIntegration:
    """Integration tests for prompt builder with other components"""
    
    @pytest.fixture
    def character_core(self):
        """CharacterCore object for integration testing"""
        return CharacterCore(
            name="Cricket",
            description="A struggling debt collection agent", 
            personality="anxious, determined, empathetic",
            big_five={
                'openness': 0.8,
                'conscientiousness': 0.8,
                'extraversion': 0.2, 
                'agreeableness': 0.9,
                'neuroticism': 0.8
            },
            goals=["Pay off debt", "Keep agency running"],
            relationships=[{"name": "Boss", "stance": "respectful fear", "affinity": 3}]
        )
    
    def test_integration_with_character_core(self, character_core):
        """Test that prompt builder works with CharacterCore objects"""
        world_lore = {
            'name': 'Test World',
            'facts': ['Debt collection is challenging']
        }
        
        builder = PromptBuilder(world_lore=world_lore)
        result = builder.build_prompt(
            character=character_core,
            mode="chat",
            base_prompt="How are you?"
        )
        
        assert isinstance(result, str)
        assert character_core.name in result
        assert len(result) > 0


# Standalone function tests
class TestStandaloneFunctions:
    """Test standalone prompt building functions"""
    
    def test_build_prompt_function(self):
        """Test the standalone build_prompt function"""
        character = {
            'name': 'TestChar',
            'description': 'Test character',
            'goals': ['Test goal']
        }
        
        world_lore = {
            'facts': ['Test fact']
        }
        
        result = build_prompt(
            character=character,
            world_lore=world_lore,
            mode="chat",
            base_prompt="Hello"
        )
        
        assert isinstance(result, str)
        assert character['name'] in result
        assert 'Test goal' in result
        assert 'Test fact' in result
    
    def test_build_conversation_turn_function(self):
        """Test the standalone build_conversation_turn function"""
        turn = build_conversation_turn(
            turn_idx=1,
            role="user",
            text="Hello world",
            test_metadata="test_value"
        )
        
        assert turn['turn_id'] == 1
        assert turn['role'] == "user"
        assert turn['content'] == "Hello world"
        assert turn['metadata']['test_metadata'] == "test_value" 