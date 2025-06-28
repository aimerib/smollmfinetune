"""
Tests for control token tokenizer patching functionality.

This module tests the tokenizer patching script that adds control tokens
to base model tokenizers and caches the results.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.patch_tokenizer import (
    load_control_tokens,
    patch_tokenizer,
    verify_patched_tokenizer
)


class TestLoadControlTokens:
    """Test loading control tokens from JSON file"""
    
    def test_load_control_tokens_success(self):
        """Test successfully loading tokens from a valid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            token_data = [
                {"token": "<mood_happy>", "category": "mood", "description": "cheerful tone"},
                {"token": "<stage_whisper>", "category": "action", "description": "speaks softly"}
            ]
            json.dump(token_data, f)
            temp_path = f.name
        
        try:
            tokens = load_control_tokens(temp_path)
            assert tokens == ["<mood_happy>", "<stage_whisper>"]
        finally:
            os.unlink(temp_path)
    
    def test_load_control_tokens_file_not_found(self):
        """Test handling when tokens file doesn't exist"""
        tokens = load_control_tokens("nonexistent_file.json")
        assert tokens == []
    
    def test_load_control_tokens_invalid_json(self):
        """Test handling of invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            tokens = load_control_tokens(temp_path)
            assert tokens == []
        finally:
            os.unlink(temp_path)
    
    def test_load_control_tokens_missing_token_field(self):
        """Test handling when some items don't have token field"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            token_data = [
                {"token": "<mood_happy>", "category": "mood"},
                {"category": "mood", "description": "missing token field"},
                {"token": "<stage_whisper>", "category": "action"}
            ]
            json.dump(token_data, f)
            temp_path = f.name
        
        try:
            tokens = load_control_tokens(temp_path)
            assert tokens == ["<mood_happy>", "<stage_whisper>"]
        finally:
            os.unlink(temp_path)


class TestPatchTokenizer:
    """Test tokenizer patching functionality"""
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_patch_tokenizer_adds_new_tokens(self, mock_tokenizer_class):
        """Test that new tokens are added to tokenizer"""
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"existing_token": 0}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create temporary tokens file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            token_data = [
                {"token": "<mood_happy>", "category": "mood"},
                {"token": "<existing_token>", "category": "test"}  # Already exists
            ]
            json.dump(token_data, f)
            tokens_file = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result_path = patch_tokenizer(
                    base_model_name="test-model",
                    tokens_file=tokens_file,
                    cache_dir=temp_dir
                )
                
                # Verify tokenizer operations
                mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
                # Should add only tokens not already in vocab
                expected_tokens = ["<mood_happy>"]  # <existing_token> should be filtered out
                mock_tokenizer.add_tokens.assert_called_once()
                actual_call_args = mock_tokenizer.add_tokens.call_args[0][0]
                assert "<mood_happy>" in actual_call_args
                mock_tokenizer.save_pretrained.assert_called_once()
                
                assert result_path.endswith("test_model_patched")
                
            finally:
                os.unlink(tokens_file)
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_patch_tokenizer_no_new_tokens(self, mock_tokenizer_class):
        """Test when all tokens already exist in tokenizer"""
        # Create mock tokenizer with all tokens already present
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            "<mood_happy>": 0,
            "<stage_whisper>": 1
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create temporary tokens file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            token_data = [
                {"token": "<mood_happy>", "category": "mood"},
                {"token": "<stage_whisper>", "category": "action"}
            ]
            json.dump(token_data, f)
            tokens_file = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result_path = patch_tokenizer(
                    base_model_name="test-model",
                    tokens_file=tokens_file,
                    cache_dir=temp_dir
                )
                
                # Should not call add_tokens since all tokens exist
                mock_tokenizer.add_tokens.assert_not_called()
                # But should still save for consistency
                mock_tokenizer.save_pretrained.assert_called_once()
                
            finally:
                os.unlink(tokens_file)
    
    def test_patch_tokenizer_cached_exists(self):
        """Test using cached tokenizer when it already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake cached tokenizer directory
            cached_path = Path(temp_dir) / "test_model_patched"
            cached_path.mkdir()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump([{"token": "<test>", "category": "test"}], f)
                tokens_file = f.name
            
            try:
                result_path = patch_tokenizer(
                    base_model_name="test-model",
                    tokens_file=tokens_file,
                    cache_dir=temp_dir
                )
                
                assert result_path == str(cached_path)
                
            finally:
                os.unlink(tokens_file)
    
    def test_patch_tokenizer_no_tokens_raises_error(self):
        """Test that empty tokens list raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)  # Empty tokens list
            tokens_file = f.name
        
        try:
            with pytest.raises(ValueError, match="No control tokens found"):
                patch_tokenizer("test-model", tokens_file=tokens_file)
        finally:
            os.unlink(tokens_file)


class TestVerifyPatchedTokenizer:
    """Test tokenizer verification functionality"""
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_verify_patched_tokenizer_success(self, mock_tokenizer_class):
        """Test successful verification of patched tokenizer"""
        # Mock tokenizer with all required tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            "<mood_happy>": 0,
            "<stage_whisper>": 1
        }
        # Mock successful encode/decode round trip
        mock_tokenizer.encode.return_value = [0, 1]
        mock_tokenizer.decode.return_value = "<mood_happy> <stage_whisper>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        control_tokens = ["<mood_happy>", "<stage_whisper>"]
        
        result = verify_patched_tokenizer("fake_path", control_tokens)
        assert result is True
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_verify_patched_tokenizer_missing_tokens(self, mock_tokenizer_class):
        """Test verification failure when tokens are missing"""
        # Mock tokenizer missing some tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            "<mood_happy>": 0
            # Missing <stage_whisper>
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        control_tokens = ["<mood_happy>", "<stage_whisper>"]
        
        result = verify_patched_tokenizer("fake_path", control_tokens)
        assert result is False
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_verify_patched_tokenizer_round_trip_failure(self, mock_tokenizer_class):
        """Test verification failure when round-trip encoding fails"""
        # Mock tokenizer with tokens but failed round trip
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            "<mood_happy>": 0,
            "<stage_whisper>": 1
        }
        # Mock failed round trip
        mock_tokenizer.encode.return_value = [0, 1]
        mock_tokenizer.decode.return_value = "corrupted output"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        control_tokens = ["<mood_happy>", "<stage_whisper>"]
        
        result = verify_patched_tokenizer("fake_path", control_tokens)
        assert result is False
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_verify_patched_tokenizer_exception_handling(self, mock_tokenizer_class):
        """Test verification handles exceptions gracefully"""
        # Mock tokenizer that raises exception
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Loading failed")
        
        control_tokens = ["<mood_happy>"]
        
        result = verify_patched_tokenizer("fake_path", control_tokens)
        assert result is False


class TestIntegration:
    """Integration tests for the complete tokenizer patching flow"""
    
    @patch('scripts.patch_tokenizer.AutoTokenizer')
    def test_complete_patching_flow(self, mock_tokenizer_class):
        """Test complete flow from loading tokens to verifying patched tokenizer"""
        # Set up mock tokenizer
        mock_tokenizer = MagicMock()
        initial_vocab = {"existing": 0}
        patched_vocab = {"existing": 0, "<mood_happy>": 1, "<stage_whisper>": 2}
        
        # First call returns initial vocab, second call returns patched vocab for verification
        mock_tokenizer.get_vocab.side_effect = [initial_vocab, patched_vocab, patched_vocab]
        mock_tokenizer.encode.return_value = [1, 2]
        mock_tokenizer.decode.return_value = "<mood_happy> <stage_whisper>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create tokens file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            token_data = [
                {"token": "<mood_happy>", "category": "mood"},
                {"token": "<stage_whisper>", "category": "action"}
            ]
            json.dump(token_data, f)
            tokens_file = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Patch the tokenizer
                result_path = patch_tokenizer(
                    base_model_name="test-model",
                    tokens_file=tokens_file,
                    cache_dir=temp_dir
                )
                
                # Load tokens for verification
                tokens = load_control_tokens(tokens_file)
                
                # Verify the patched tokenizer
                is_valid = verify_patched_tokenizer(result_path, tokens)
                
                assert is_valid is True
                assert mock_tokenizer.add_tokens.called
                assert mock_tokenizer.save_pretrained.called
                
            finally:
                os.unlink(tokens_file) 