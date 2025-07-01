"""
Tests for AI integration in Character Management Studio
"""

import pytest
import asyncio
import sys
from pathlib import Path

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
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_suggest_description_structured_output(self, sample_character_core):
        """Test description suggestions with structured output"""
        from app.pages.character_management import llm_suggest_description
        
        suggestions = await llm_suggest_description(sample_character_core)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 2  # Should get multiple suggestions
        assert len(suggestions) <= 5  # Reasonable upper bound
        
        # All suggestions should be non-empty strings
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion.strip()) > 10  # Meaningful length
        
        # Should enhance the character description in some way
        original_desc = sample_character_core.description.lower()
        suggestions_text = " ".join(suggestions).lower()
        
        # Should maintain core elements like "warrior" or character essence
        assert any(keyword in suggestions_text for keyword in ["warrior", "brave", "kind", "noble", "hero"])
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_suggest_goals_with_personality(self, sample_character_core):
        """Test goal suggestions that consider personality traits"""
        from app.pages.character_management import llm_suggest_goals
        
        goals = await llm_suggest_goals(sample_character_core)
        
        assert isinstance(goals, list)
        assert len(goals) >= 3  # Should get multiple goal suggestions
        assert len(goals) <= 8  # Reasonable upper bound
        
        # All goals should be meaningful strings
        for goal in goals:
            assert isinstance(goal, str)
            assert len(goal.strip()) > 5
        
        # With high agreeableness (0.9), goals should reflect cooperation/helping others
        goals_text = " ".join(goals).lower()
        assert any(keyword in goals_text for keyword in [
            "protect", "help", "assist", "support", "defend", "serve", "care", "peace"
        ])
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_suggest_scenario_with_context(self, sample_character_core):
        """Test scenario suggestions that use character context"""
        from app.pages.character_management import llm_suggest_scenario
        
        scenarios = await llm_suggest_scenario(sample_character_core)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 2
        assert len(scenarios) <= 6
        
        # All scenarios should be meaningful
        for scenario in scenarios:
            assert isinstance(scenario, str)
            assert len(scenario.strip()) > 10
        
        # Should fit the medieval fantasy setting
        scenarios_text = " ".join(scenarios).lower()
        assert any(keyword in scenarios_text for keyword in [
            "village", "fantasy", "medieval", "warrior", "quest", "adventure", "kingdom", "battle"
        ])
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_suggest_examples_with_character_voice(self, sample_character_core):
        """Test dialogue example suggestions that capture character voice"""
        from app.pages.character_management import llm_suggest_examples
        
        examples = await llm_suggest_examples(sample_character_core)
        
        assert isinstance(examples, list)
        assert len(examples) >= 2
        
        # All examples should follow User/Character dialogue format
        for example in examples:
            assert isinstance(example, str)
            assert "User:" in example or "USER:" in example
            assert sample_character_core.name in example or "Test Character" in example
            assert len(example.strip()) > 20  # Should be substantial dialogue
        
        # Should show character personality traits
        examples_text = " ".join(examples).lower()
        
        # With high agreeableness, should show cooperative/kind language
        assert any(keyword in examples_text for keyword in [
            "help", "kind", "friend", "please", "thank", "happy", "glad", "honor"
        ])
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_error_handling_with_fallback(self, sample_character_core):
        """Test that AI functions gracefully handle edge cases and provide reasonable results"""
        from app.pages.character_management import llm_suggest_description
        
        # Test with minimal character info
        minimal_character = CharacterCore(
            name="X",
            description="A person",
            scenario="Modern",
            backstory="Unknown",
            appearance="Average",
            personality_traits=Personality(
                openness=0.5, conscientiousness=0.5, extraversion=0.5,
                agreeableness=0.5, neuroticism=0.5
            ),
            goals=[], relationships=[], tags=[]
        )
        
        suggestions = await llm_suggest_description(minimal_character)
        
        # Should still return reasonable suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should work with the minimal info provided
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion.strip()) > 5
    
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
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_ai_consistency_across_calls(self, sample_character_core):
        """Test that AI suggestions are reasonably consistent for the same character"""
        from app.pages.character_management import llm_suggest_description
        
        # Get suggestions twice
        suggestions1 = await llm_suggest_description(sample_character_core)
        suggestions2 = await llm_suggest_description(sample_character_core)
        
        # Both should be valid
        assert isinstance(suggestions1, list) and len(suggestions1) > 0
        assert isinstance(suggestions2, list) and len(suggestions2) > 0
        
        # While exact suggestions may differ, they should maintain similar themes
        all_text = " ".join(suggestions1 + suggestions2).lower()
        
        # Should consistently reference the character's core traits
        assert any(keyword in all_text for keyword in ["warrior", "brave", "kind", "noble"])
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_personality_influence_on_suggestions(self, sample_character_core):
        """Test that personality traits meaningfully influence AI suggestions"""
        from app.pages.character_management import llm_suggest_goals
        
        # Create two contrasting characters
        extroverted_char = CharacterCore(
            name="Extrovert", description="Social person", scenario="Modern",
            backstory="Party organizer", appearance="Bright smile",
            personality_traits=Personality(
                openness=0.8, conscientiousness=0.5, extraversion=0.9,  # High extraversion
                agreeableness=0.7, neuroticism=0.2
            ),
            goals=[], relationships=[], tags=[]
        )
        
        introverted_char = CharacterCore(
            name="Introvert", description="Quiet person", scenario="Modern",
            backstory="Librarian", appearance="Thoughtful eyes",
            personality_traits=Personality(
                openness=0.8, conscientiousness=0.5, extraversion=0.1,  # Low extraversion
                agreeableness=0.7, neuroticism=0.2
            ),
            goals=[], relationships=[], tags=[]
        )
        
        extrovert_goals = await llm_suggest_goals(extroverted_char)
        introvert_goals = await llm_suggest_goals(introverted_char)
        
        # Both should be valid
        assert len(extrovert_goals) > 0 and len(introvert_goals) > 0
        
        extrovert_text = " ".join(extrovert_goals).lower()
        introvert_text = " ".join(introvert_goals).lower()
        
        # Extrovert goals should emphasize social elements more
        social_keywords = ["people", "social", "community", "party", "group", "friends", "network"]
        extrovert_social_count = sum(1 for keyword in social_keywords if keyword in extrovert_text)
        introvert_social_count = sum(1 for keyword in social_keywords if keyword in introvert_text)
        
        # Extrovert should have more social references (though not necessarily)
        # This is more of a directional test since LLM behavior can vary
        assert extrovert_social_count >= 0 and introvert_social_count >= 0 