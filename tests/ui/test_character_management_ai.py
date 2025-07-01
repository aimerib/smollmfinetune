"""
Tests for AI integration in Character Management Studio
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Add the app directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))

from utils.character.models import CharacterCore, Personality, Relationship


class TestCharacterManagementAI:
    """Test AI integration functions in character management"""
    
    @pytest.fixture
    def sample_character_core(self):
        """Create a sample CharacterCore for testing"""
        return CharacterCore(
            name="Test Character",
            description="A brave warrior with a kind heart",
            scenario="Medieval fantasy setting",
            backstory="Grew up in a small village",
            appearance="Tall with blue eyes",
            personality_traits=Personality(
                openness=0.8,
                conscientiousness=0.7,
                extraversion=0.6,
                agreeableness=0.9,
                neuroticism=0.3
            ),
            goals=["Protect the innocent", "Find their destiny"],
            relationships=[
                Relationship(name="Village Elder", affinity=8),
                Relationship(name="Dark Sorcerer", affinity=-7)
            ],
            tags=["fantasy", "warrior", "hero"]
        )
    
    
    async def test_llm_suggest_description_structured_output(self, sample_character_core):
        """Test description suggestions with structured output"""
        from app.pages.character_management import llm_suggest_description
        
        # Mock the OpenAI client
        mock_response = {
            "suggestions": [
                "A noble warrior with unwavering courage and compassion",
                "A battle-hardened hero who fights for justice and peace",
                "A legendary champion known for their bravery and kindness"
            ],
            "reasoning": "Enhanced descriptions that emphasize the character's heroic nature"
        }
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate.return_value = '{"suggestions": ["A noble warrior with unwavering courage and compassion", "A battle-hardened hero who fights for justice and peace", "A legendary champion known for their bravery and kindness"], "reasoning": "Enhanced descriptions that emphasize the character\'s heroic nature"}'
            mock_get_client.return_value = mock_client
            
            suggestions = await llm_suggest_description(sample_character_core)
            
            assert isinstance(suggestions, list)
            assert len(suggestions) == 3
            assert "noble warrior" in suggestions[0]
            assert "battle-hardened hero" in suggestions[1]
            assert "legendary champion" in suggestions[2]
            
            # Verify the client was called with correct parameters
            mock_client.generate.assert_called_once()
            call_args = mock_client.generate.call_args
            assert "response_format" in call_args.kwargs
            assert call_args.kwargs["response_format"]["type"] == "json_schema"
    
    
    async def test_llm_suggest_description_fallback(self, sample_character_core):
        """Test description suggestions fallback when structured output fails"""
        from app.pages.character_management import llm_suggest_description
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            # First call fails (structured output), second call succeeds (fallback)
            mock_client.generate.side_effect = [
                Exception("Structured output not supported"),
                '{"suggestions": ["Fallback description 1", "Fallback description 2", "Fallback description 3"]}'
            ]
            mock_get_client.return_value = mock_client
            
            suggestions = await llm_suggest_description(sample_character_core)
            
            assert isinstance(suggestions, list)
            assert len(suggestions) == 3
            assert "Fallback description" in suggestions[0]
            
            # Verify fallback was called
            assert mock_client.generate.call_count == 2
    
    
    async def test_llm_suggest_goals_with_personality(self, sample_character_core):
        """Test goal suggestions that consider personality traits"""
        from app.pages.character_management import llm_suggest_goals
        
        mock_response = '{"goals": ["Become a legendary protector", "Build a peaceful kingdom", "Master ancient combat techniques", "Mentor young warriors"], "reasoning": "Goals that align with high agreeableness and openness"}'
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            goals = await llm_suggest_goals(sample_character_core)
            
            assert isinstance(goals, list)
            assert len(goals) >= 3
            assert any("protector" in goal.lower() for goal in goals)
            
            # Check that personality traits were included in the prompt
            call_args = mock_client.generate.call_args
            prompt = call_args.args[0] if call_args.args else call_args.kwargs.get('prompt', '')
            assert "Agreeableness" in prompt or "agreeableness" in prompt.lower()
    
    
    async def test_llm_suggest_scenario_with_context(self, sample_character_core):
        """Test scenario suggestions that use character context"""
        from app.pages.character_management import llm_suggest_scenario
        
        mock_response = '{"scenarios": ["Defending the village from bandits", "Searching for a lost artifact", "Training new recruits"], "reasoning": "Scenarios that fit the medieval fantasy setting"}'
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            scenarios = await llm_suggest_scenario(sample_character_core)
            
            assert isinstance(scenarios, list)
            assert len(scenarios) >= 3
            assert any("village" in scenario.lower() for scenario in scenarios)
            
            # Check that character context was included
            call_args = mock_client.generate.call_args
            prompt = call_args.args[0] if call_args.args else call_args.kwargs.get('prompt', '')
            assert "Test Character" in prompt
            assert "Medieval fantasy" in prompt or "medieval fantasy" in prompt.lower()
    
    
    async def test_llm_suggest_examples_with_character_voice(self, sample_character_core):
        """Test dialogue example suggestions that capture character voice"""
        from app.pages.character_management import llm_suggest_examples
        
        mock_response = '{"examples": ["User: Hello there!\\nTest Character: *bows respectfully* Greetings, friend. How may I assist you?", "User: Are you ready for battle?\\nTest Character: *grips sword firmly* I am always ready to defend those who cannot defend themselves."], "style_notes": "Formal, respectful speech with heroic undertones"}'
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            examples = await llm_suggest_examples(sample_character_core)
            
            assert isinstance(examples, list)
            assert len(examples) >= 2
            assert any("User:" in example and "Test Character:" in example for example in examples)
            assert any("*" in example for example in examples)  # Action descriptions
            
            # Check that personality was considered
            call_args = mock_client.generate.call_args
            prompt = call_args.args[0] if call_args.args else call_args.kwargs.get('prompt', '')
            assert "cooperative and trusting" in prompt.lower() or "agreeableness" in prompt.lower()
    
    
    async def test_error_handling_with_fallback(self, sample_character_core):
        """Test that AI functions gracefully handle errors and provide fallbacks"""
        from app.pages.character_management import llm_suggest_description
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate.side_effect = Exception("Network error")
            mock_get_client.return_value = mock_client
            
            suggestions = await llm_suggest_description(sample_character_core)
            
            # Should still return reasonable fallback suggestions
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0
            assert any("Test Character" in suggestion for suggestion in suggestions)
    
    def test_personality_helper_functions(self, sample_character_core):
        """Test helper functions for personality analysis"""
        from app.pages.character_management import _get_high_traits, _get_low_traits, _describe_personality
        
        personality = sample_character_core.personality_traits
        
        high_traits = _get_high_traits(personality)
        assert "Agreeableness" in high_traits  # 0.9 is high
        assert "Openness" in high_traits  # 0.8 is high
        
        low_traits = _get_low_traits(personality)
        assert "Neuroticism" in low_traits  # 0.3 is low
        
        description = _describe_personality(personality)
        assert "cooperative and trusting" in description  # High agreeableness
        assert "emotionally stable" in description  # Low neuroticism
    
    
    async def test_client_response_format_parameter(self, sample_character_core):
        """Test that the response_format parameter is properly passed to the client"""
        from app.pages.character_management import llm_suggest_description
        
        with patch('app.pages.character_management.get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate.return_value = '{"suggestions": ["Test suggestion"], "reasoning": "Test reasoning"}'
            mock_get_client.return_value = mock_client
            
            await llm_suggest_description(sample_character_core)
            
            # The function tries structured output first, so check the first call
            assert mock_client.generate.call_count >= 1
            first_call_args = mock_client.generate.call_args_list[0]
            assert "response_format" in first_call_args.kwargs
            response_format = first_call_args.kwargs["response_format"]
            assert response_format["type"] == "json_schema"
            assert "json_schema" in response_format
            assert "name" in response_format["json_schema"]
            assert "schema" in response_format["json_schema"] 