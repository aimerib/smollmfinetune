"""
Tests for Subtext Parser

Testing the regex-based parser for extracting SUBTEXT and ACTION blocks
from model output, ensuring proper error handling and edge cases.
"""

import pytest
from backend.app.narrative_engine.subtext_parser import (
    parse_subtext_and_action,
    validate_tagged_output,
    SubtextParseError
)


class TestSubtextParser:
    """Test the subtext parsing functionality"""
    
    def test_parse_valid_subtext_and_action(self):
        """Test parsing valid model output with both tags"""
        model_output = """
        The character considers their options carefully.
        
        [SUBTEXT]
        I can't trust anyone here. Everyone seems to be hiding something.
        [/SUBTEXT]
        
        [ACTION]
        I will move to the Forest to gather my thoughts alone.
        [/ACTION]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        assert subtext == "I can't trust anyone here. Everyone seems to be hiding something."
        assert action == "I will move to the Forest to gather my thoughts alone."
    
    def test_parse_case_insensitive_tags(self):
        """Test that tag parsing is case insensitive"""
        model_output = """
        [subtext]
        This is my inner thought.
        [/subtext]
        
        [action]
        This is my action.
        [/action]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        assert subtext == "This is my inner thought."
        assert action == "This is my action."
    
    def test_parse_multiline_content(self):
        """Test parsing content that spans multiple lines"""
        model_output = """
        [SUBTEXT]
        This is a longer inner monologue
        that spans multiple lines
        and has various thoughts.
        [/SUBTEXT]
        
        [ACTION]
        I will speak to Clara and say:
        "Hello, how are you today?"
        [/ACTION]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        assert "longer inner monologue" in subtext
        assert "multiple lines" in subtext
        assert "speak to Clara" in action
        assert "Hello, how are you today?" in action
    
    def test_parse_missing_subtext_graceful_fallback(self):
        """Test graceful fallback when SUBTEXT tags are missing"""
        model_output = """
        [ACTION]
        I will move to the Village Square.
        [/ACTION]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        assert subtext == ""  # Graceful fallback
        assert action == "I will move to the Village Square."
    
    def test_parse_missing_action_raises_error(self):
        """Test that missing ACTION tags raise SubtextParseError"""
        model_output = """
        [SUBTEXT]
        I'm thinking about what to do.
        [/SUBTEXT]
        
        Just some text without action tags.
        """
        
        with pytest.raises(SubtextParseError, match="No \\[ACTION\\] tags found"):
            parse_subtext_and_action(model_output)
    
    def test_parse_empty_action_content_raises_error(self):
        """Test that empty ACTION content raises SubtextParseError"""
        model_output = """
        [SUBTEXT]
        I'm thinking.
        [/SUBTEXT]
        
        [ACTION]
        
        [/ACTION]
        """
        
        with pytest.raises(SubtextParseError, match="Empty \\[ACTION\\] content found"):
            parse_subtext_and_action(model_output)
    
    def test_parse_empty_input_raises_error(self):
        """Test that empty input raises SubtextParseError"""
        with pytest.raises(SubtextParseError, match="Empty model output provided"):
            parse_subtext_and_action("")
        
        with pytest.raises(SubtextParseError, match="Empty model output provided"):
            parse_subtext_and_action("   \n  \t  ")
    
    def test_parse_non_string_input_raises_error(self):
        """Test that non-string input raises SubtextParseError"""
        with pytest.raises(SubtextParseError, match="Expected string input"):
            parse_subtext_and_action(None)
        
        with pytest.raises(SubtextParseError, match="Expected string input"):
            parse_subtext_and_action(123)
        
        with pytest.raises(SubtextParseError, match="Expected string input"):
            parse_subtext_and_action(['not', 'a', 'string'])
    
    def test_parse_whitespace_handling(self):
        """Test that leading/trailing whitespace is properly handled"""
        model_output = """
        [SUBTEXT]
        
        
        This has extra whitespace.    
        
        
        [/SUBTEXT]
        
        [ACTION]
        
           This also has whitespace.
           
        [/ACTION]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        assert subtext == "This has extra whitespace."
        assert action == "This also has whitespace."


class TestTagValidation:
    """Test the tag validation functionality"""
    
    def test_validate_complete_output(self):
        """Test validation of complete, well-formed output"""
        model_output = """
        [SUBTEXT]
        I'm thinking.
        [/SUBTEXT]
        
        [ACTION]
        I will act.
        [/ACTION]
        """
        
        result = validate_tagged_output(model_output)
        
        assert result["has_subtext_tags"] is True
        assert result["has_action_tags"] is True
        assert result["subtext_content_exists"] is True
        assert result["action_content_exists"] is True
    
    def test_validate_missing_subtext_tags(self):
        """Test validation when subtext tags are missing"""
        model_output = """
        [ACTION]
        I will act.
        [/ACTION]
        """
        
        result = validate_tagged_output(model_output)
        
        assert result["has_subtext_tags"] is False
        assert result["has_action_tags"] is True
        assert result["subtext_content_exists"] is False
        assert result["action_content_exists"] is True
    
    def test_validate_empty_content(self):
        """Test validation when tags exist but content is empty"""
        model_output = """
        [SUBTEXT]
        [/SUBTEXT]
        
        [ACTION]
        Something meaningful here.
        [/ACTION]
        """
        
        result = validate_tagged_output(model_output)
        
        assert result["has_subtext_tags"] is True
        assert result["has_action_tags"] is True
        assert result["subtext_content_exists"] is False
        assert result["action_content_exists"] is True
    
    def test_validate_no_tags(self):
        """Test validation when no tags are present"""
        model_output = "Just some regular text without any tags."
        
        result = validate_tagged_output(model_output)
        
        assert result["has_subtext_tags"] is False
        assert result["has_action_tags"] is False
        assert result["subtext_content_exists"] is False
        assert result["action_content_exists"] is False


class TestEdgeCases:
    """Test edge cases and malformed input"""
    
    def test_nested_tags_handling(self):
        """Test handling of nested or malformed tag structures"""
        model_output = """
        [SUBTEXT]
        I'm thinking about [ACTION] tags inside subtext.
        [/SUBTEXT]
        
        [ACTION]
        I will do something with [SUBTEXT] references.
        [/ACTION]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        assert "[ACTION] tags inside subtext" in subtext
        assert "[SUBTEXT] references" in action
    
    def test_multiple_tag_pairs(self):
        """Test behavior with multiple tag pairs (should use first occurrence)"""
        model_output = """
        [SUBTEXT]
        First subtext.
        [/SUBTEXT]
        
        [ACTION]
        First action.
        [/ACTION]
        
        [SUBTEXT]
        Second subtext.
        [/SUBTEXT]
        
        [ACTION]
        Second action.
        [/ACTION]
        """
        
        subtext, action = parse_subtext_and_action(model_output)
        
        # Should extract the first occurrence
        assert subtext == "First subtext."
        assert action == "First action."
    
    def test_malformed_closing_tags(self):
        """Test behavior with malformed closing tags"""
        model_output = """
        [SUBTEXT]
        Content without proper closing.
        [/WRONG_TAG]
        
        [ACTION]
        This should still work.
        [/ACTION]
        """
        
        # Should fail to find subtext but find action
        subtext, action = parse_subtext_and_action(model_output)
        
        assert subtext == ""  # No proper subtext closing tag
        assert action == "This should still work." 