"""
Tests for Prompt Templates

Testing the subtext-enhanced prompt templates that enable
the dual-output Iceberg Model functionality.
"""

import pytest
from backend.app.core.prompt_templates import (
    SUBTEXT_ACTION_PROMPT_SUFFIX,
    build_subtext_prompt
)


class TestPromptTemplates:
    """Test the prompt template functionality"""
    
    def test_subtext_action_prompt_suffix_contains_required_tags(self):
        """Test that the prompt suffix contains both SUBTEXT and ACTION tags"""
        # Snapshot test: assert that required tags are present
        assert "[SUBTEXT]" in SUBTEXT_ACTION_PROMPT_SUFFIX
        assert "[/SUBTEXT]" in SUBTEXT_ACTION_PROMPT_SUFFIX
        assert "[ACTION]" in SUBTEXT_ACTION_PROMPT_SUFFIX
        assert "[/ACTION]" in SUBTEXT_ACTION_PROMPT_SUFFIX
    
    def test_subtext_action_prompt_suffix_contains_instructions(self):
        """Test that the prompt suffix contains clear instructions"""
        suffix = SUBTEXT_ACTION_PROMPT_SUFFIX
        
        # Should contain instruction text
        assert "inner" in suffix.lower() or "private" in suffix.lower()
        assert "thoughts" in suffix.lower() or "monologue" in suffix.lower()
        assert "action" in suffix.lower()
        assert "character" in suffix.lower()
    
    def test_build_subtext_prompt_basic_functionality(self):
        """Test basic functionality of build_subtext_prompt"""
        base_prompt = "You are a test character in a test world."
        
        enhanced_prompt = build_subtext_prompt(base_prompt)
        
        # Should contain the original prompt
        assert base_prompt in enhanced_prompt
        
        # Should contain the subtext/action tags
        assert "[SUBTEXT]" in enhanced_prompt
        assert "[ACTION]" in enhanced_prompt
    
    def test_build_subtext_prompt_preserves_base_content(self):
        """Test that the base prompt content is preserved"""
        base_prompt = """
        You are Clara, a mysterious librarian.
        Current location: Ancient Library
        Goals: Protect the forbidden knowledge
        Personality: Secretive, wise, cautious
        
        A stranger has entered seeking information about the lost artifact.
        What do you do?
        """
        
        enhanced_prompt = build_subtext_prompt(base_prompt)
        
        # All original content should be preserved
        assert "Clara" in enhanced_prompt
        assert "Ancient Library" in enhanced_prompt
        assert "forbidden knowledge" in enhanced_prompt
        assert "lost artifact" in enhanced_prompt
        
        # Enhanced content should be added
        assert "[SUBTEXT]" in enhanced_prompt
        assert "[ACTION]" in enhanced_prompt
    
    def test_build_subtext_prompt_with_empty_base(self):
        """Test behavior with empty base prompt"""
        enhanced_prompt = build_subtext_prompt("")
        
        # Should still contain the tags even with empty base
        assert "[SUBTEXT]" in enhanced_prompt
        assert "[ACTION]" in enhanced_prompt
    
    def test_build_subtext_prompt_with_multiline_base(self):
        """Test with complex multiline base prompt"""
        base_prompt = """
        Complex prompt with:
        - Multiple lines
        - Bullet points
        - Various formatting
        
        And some concluding instructions.
        """
        
        enhanced_prompt = build_subtext_prompt(base_prompt)
        
        # Should preserve formatting
        assert "Multiple lines" in enhanced_prompt
        assert "Bullet points" in enhanced_prompt
        assert "concluding instructions" in enhanced_prompt
        
        # Should add tags at the end
        assert enhanced_prompt.endswith(SUBTEXT_ACTION_PROMPT_SUFFIX)
    
    def test_prompt_suffix_structure_validation(self):
        """Test the structure and format of the prompt suffix"""
        suffix = SUBTEXT_ACTION_PROMPT_SUFFIX
        
        # Check tag order (SUBTEXT should come before ACTION)
        subtext_pos = suffix.find("[SUBTEXT]")
        action_pos = suffix.find("[ACTION]")
        
        assert subtext_pos < action_pos, "SUBTEXT tags should appear before ACTION tags"
        
        # Check that closing tags exist
        assert suffix.find("[/SUBTEXT]") > subtext_pos
        assert suffix.find("[/ACTION]") > action_pos
    
    def test_prompt_suffix_content_between_tags(self):
        """Test that there's explanatory content between the tags"""
        suffix = SUBTEXT_ACTION_PROMPT_SUFFIX
        
        # Extract content between SUBTEXT tags
        subtext_start = suffix.find("[SUBTEXT]") + len("[SUBTEXT]")
        subtext_end = suffix.find("[/SUBTEXT]")
        subtext_content = suffix[subtext_start:subtext_end].strip()
        
        # Should have explanatory content
        assert len(subtext_content) > 0
        assert "character" in subtext_content.lower()
        
        # Extract content between ACTION tags
        action_start = suffix.find("[ACTION]") + len("[ACTION]")
        action_end = suffix.find("[/ACTION]")
        action_content = suffix[action_start:action_end].strip()
        
        # Should have explanatory content
        assert len(action_content) > 0
        assert "action" in action_content.lower()
    
    def test_prompt_template_is_string(self):
        """Test that the prompt suffix is a string"""
        assert isinstance(SUBTEXT_ACTION_PROMPT_SUFFIX, str)
        assert len(SUBTEXT_ACTION_PROMPT_SUFFIX.strip()) > 0
    
    def test_build_subtext_prompt_returns_string(self):
        """Test that build_subtext_prompt always returns a string"""
        result1 = build_subtext_prompt("test")
        result2 = build_subtext_prompt("")
        result3 = build_subtext_prompt("longer test prompt with multiple words")
        
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str) 