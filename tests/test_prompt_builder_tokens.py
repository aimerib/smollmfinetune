"""
Tests for PromptBuilder control token functionality.

This module tests the enhanced PromptBuilder that can detect control tokens
in prompts and add natural language hints for the model to understand.
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.utils.generation.prompt_builder import PromptBuilder


class TestPromptBuilderTokens:
    """Test PromptBuilder control token functionality"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Sample control tokens for testing"""
        return [
            {
                "token": "<mood_happy>",
                "category": "mood",
                "description": "Character speaks in a cheerful, upbeat tone",
                "ui_icon": "😊"
            },
            {
                "token": "<stage_whisper>",
                "category": "action",
                "description": "Character speaks softly or whispers",
                "ui_icon": "🤫"
            },
            {
                "token": "<nsfw_soft>",
                "category": "nsfw",
                "description": "Scene contains soft, romantic content",
                "ui_icon": "🌶️"
            }
        ]
    
    @pytest.fixture
    def sample_character(self):
        """Sample character data for testing"""
        return {
            'name': 'TestChar',
            'description': 'A test character',
            'personality': 'friendly and helpful',
            'big_five': {
                'openness': 0.8,
                'conscientiousness': 0.6,
                'extraversion': 0.7,
                'agreeableness': 0.9,
                'neuroticism': 0.3
            },
            'goals': ['help users', 'be friendly', 'provide good information'],
            'relationships': [
                {'name': 'User', 'stance': 'helpful assistant'}
            ]
        }
    
    def test_prompt_builder_initialization_with_tokens(self, sample_tokens):
        """Test PromptBuilder initializes correctly with control tokens"""
        world_lore = {'facts': ['Test fact']}
        
        builder = PromptBuilder(world_lore=world_lore, control_tokens=sample_tokens)
        
        assert builder.control_tokens == sample_tokens
        assert len(builder.token_lookup) == 3
        assert "<mood_happy>" in builder.token_lookup
        assert builder.token_lookup["<mood_happy>"]["description"] == "Character speaks in a cheerful, upbeat tone"
    
    def test_find_tokens_in_text(self, sample_tokens):
        """Test finding control tokens in text"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        text = "Hello <mood_happy> there! I hope <stage_whisper> you can hear me."
        found_tokens = builder._find_tokens_in_text(text)
        
        assert found_tokens == ["<mood_happy>", "<stage_whisper>"]
    
    def test_find_tokens_in_text_no_tokens(self, sample_tokens):
        """Test finding tokens when none are present"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        text = "This is regular text with no special tokens."
        found_tokens = builder._find_tokens_in_text(text)
        
        assert found_tokens == []
    
    def test_find_tokens_in_text_unknown_tokens(self, sample_tokens):
        """Test finding tokens filters out unknown tokens"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        text = "Hello <mood_happy> and <unknown_token> world!"
        found_tokens = builder._find_tokens_in_text(text)
        
        assert found_tokens == ["<mood_happy>"]
    
    def test_generate_token_hints_single_token(self, sample_tokens):
        """Test generating hints for a single token"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        components = ["You are TestChar.", "You are creative."]
        base_prompt = "Hello <mood_happy> there!"
        
        hints = builder._generate_token_hints(components, base_prompt)
        
        expected = "Token meanings: The token <mood_happy> means Character speaks in a cheerful, upbeat tone."
        assert hints == expected
    
    def test_generate_token_hints_multiple_tokens(self, sample_tokens):
        """Test generating hints for multiple tokens"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        components = ["You are TestChar."]
        base_prompt = "Hello <mood_happy> there! <stage_whisper> Can you hear me?"
        
        hints = builder._generate_token_hints(components, base_prompt)
        
        assert "The token <mood_happy> means Character speaks in a cheerful, upbeat tone" in hints
        assert "The token <stage_whisper> means Character speaks softly or whispers" in hints
        assert hints.startswith("Token meanings:")
        assert hints.endswith(".")
    
    def test_generate_token_hints_no_tokens(self, sample_tokens):
        """Test generating hints when no tokens are present"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        components = ["You are TestChar."]
        base_prompt = "Hello there!"
        
        hints = builder._generate_token_hints(components, base_prompt)
        
        assert hints is None
    
    def test_generate_token_hints_duplicate_tokens(self, sample_tokens):
        """Test that duplicate tokens only appear once in hints"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        components = ["You are <mood_happy> TestChar."]
        base_prompt = "Hello <mood_happy> there!"
        
        hints = builder._generate_token_hints(components, base_prompt)
        
        # Should only appear once despite being in both components and base_prompt
        token_count = hints.count("<mood_happy>")
        assert token_count == 1
    
    def test_build_prompt_with_token_hints_enabled(self, sample_character, sample_tokens):
        """Test building prompt with token hints enabled"""
        world_lore = {'facts': ['Test lore fact']}
        builder = PromptBuilder(world_lore=world_lore, control_tokens=sample_tokens)
        
        result = builder.build_prompt(
            character=sample_character,
            mode="chat",
            base_prompt="Hello <mood_happy> there!",
            add_token_hints=True
        )
        
        # Should contain the token hint
        assert "Token meanings:" in result
        assert "The token <mood_happy> means Character speaks in a cheerful, upbeat tone" in result
        
        # Should also contain regular prompt elements
        assert "TestChar" in result
        assert "Hello <mood_happy> there!" in result
    
    def test_build_prompt_with_token_hints_disabled(self, sample_character, sample_tokens):
        """Test building prompt with token hints disabled"""
        world_lore = {'facts': ['Test lore fact']}
        builder = PromptBuilder(world_lore=world_lore, control_tokens=sample_tokens)
        
        result = builder.build_prompt(
            character=sample_character,
            mode="chat",
            base_prompt="Hello <mood_happy> there!",
            add_token_hints=False
        )
        
        # Should NOT contain token hints
        assert "Token meanings:" not in result
        
        # Should still contain regular prompt elements
        assert "TestChar" in result
        assert "Hello <mood_happy> there!" in result
    
    def test_build_prompt_nsfw_mode_converts_to_token(self, sample_character, sample_tokens):
        """Test that NSFW mode converts style to control token format"""
        world_lore = {'facts': ['Test lore fact']}
        builder = PromptBuilder(world_lore=world_lore, control_tokens=sample_tokens)
        
        result = builder.build_prompt(
            character=sample_character,
            mode="nsfw",
            nsfw_style="soft",
            base_prompt="Tell me something romantic",
            add_token_hints=True
        )
        
        # Should contain the converted NSFW token
        assert "<nsfw_soft>" in result
        
        # Should contain token hint for the NSFW token
        assert "Token meanings:" in result
        assert "The token <nsfw_soft> means Scene contains soft, romantic content" in result
    
    def test_apply_control_tokens(self, sample_tokens):
        """Test applying control tokens to text"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        text = "Hello there!"
        selected_tokens = ["<mood_happy>", "<stage_whisper>"]
        
        result = builder.apply_control_tokens(text, selected_tokens)
        
        assert result == "<mood_happy> <stage_whisper> Hello there!"
    
    def test_apply_control_tokens_empty_list(self, sample_tokens):
        """Test applying empty token list returns original text"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        text = "Hello there!"
        selected_tokens = []
        
        result = builder.apply_control_tokens(text, selected_tokens)
        
        assert result == text
    
    def test_get_available_tokens_by_category(self, sample_tokens):
        """Test getting tokens filtered by category"""
        builder = PromptBuilder(control_tokens=sample_tokens)
        
        # Test getting mood tokens
        mood_tokens = builder.get_available_tokens_by_category("mood")
        assert len(mood_tokens) == 1
        assert mood_tokens[0]["token"] == "<mood_happy>"
        
        # Test getting action tokens
        action_tokens = builder.get_available_tokens_by_category("action")
        assert len(action_tokens) == 1
        assert action_tokens[0]["token"] == "<stage_whisper>"
        
        # Test getting all tokens
        all_tokens = builder.get_available_tokens_by_category(None)
        assert len(all_tokens) == 3
        
        # Test getting non-existent category
        none_tokens = builder.get_available_tokens_by_category("nonexistent")
        assert len(none_tokens) == 0
    
    def test_build_prompt_with_tokens_in_components(self, sample_character, sample_tokens):
        """Test that tokens in character components are also detected"""
        # Modify character to include tokens in goals
        character_with_tokens = sample_character.copy()
        character_with_tokens['goals'] = ['<mood_happy> help users', 'be friendly']
        
        world_lore = {'facts': ['Test lore fact']}
        builder = PromptBuilder(world_lore=world_lore, control_tokens=sample_tokens)
        
        result = builder.build_prompt(
            character=character_with_tokens,
            mode="chat",
            base_prompt="Hello there!",
            add_token_hints=True
        )
        
        # Should detect token in goals and provide hint
        assert "Token meanings:" in result
        assert "The token <mood_happy> means Character speaks in a cheerful, upbeat tone" in result
    
    def test_prompt_builder_backwards_compatibility(self, sample_character):
        """Test that PromptBuilder still works without control tokens (backwards compatibility)"""
        world_lore = {'facts': ['Test lore fact']}
        builder = PromptBuilder(world_lore=world_lore)  # No control_tokens parameter
        
        result = builder.build_prompt(
            character=sample_character,
            mode="chat",
            base_prompt="Hello there!"
        )
        
        # Should work normally without tokens
        assert "TestChar" in result
        assert "Hello there!" in result
        assert "Token meanings:" not in result
    
    def test_standalone_build_prompt_function_with_tokens(self, sample_character, sample_tokens):
        """Test the standalone build_prompt function with control tokens"""
        from app.utils.generation.prompt_builder import build_prompt
        
        world_lore = {'facts': ['Test lore fact']}
        
        result = build_prompt(
            character=sample_character,
            world_lore=world_lore,
            mode="chat",
            base_prompt="Hello <mood_happy> there!",
            control_tokens=sample_tokens
        )
        
        # Should contain token hints
        assert "Token meanings:" in result
        assert "The token <mood_happy> means Character speaks in a cheerful, upbeat tone" in result 