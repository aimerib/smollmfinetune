"""
Tests for WorldManager control token functionality.

This module tests the extended WorldManager that handles control tokens
on a per-world basis, including copying default tokens and managing
world-specific token configurations.
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.world.world import WorldManager


class TestWorldManagerTokens:
    """Test WorldManager control token functionality"""
    
    @pytest.fixture
    def sample_tokens(self):
        """Sample control tokens for testing"""
        return [
            {
                "token": "<mood_happy>",
                "category": "mood", 
                "description": "Character speaks in a cheerful tone",
                "ui_icon": "😊"
            },
            {
                "token": "<stage_whisper>",
                "category": "action",
                "description": "Character speaks softly",
                "ui_icon": "🤫"
            }
        ]
    
    @pytest.fixture
    def temp_worlds_dir(self):
        """Create temporary worlds directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def world_manager(self, temp_worlds_dir):
        """Create WorldManager with temporary directory"""
        return WorldManager(worlds_root=temp_worlds_dir)
    
    def test_get_tokens_path(self, world_manager):
        """Test getting path to world's tokens.json file"""
        tokens_path = world_manager.get_tokens_path("TestWorld")
        expected_path = Path(world_manager.worlds_root) / "TestWorld" / "tokens.json"
        assert tokens_path == expected_path
    
    def test_copy_default_tokens_success(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test successfully copying default tokens to new world"""
        # Create default tokens file
        default_tokens_path = Path("core_data/tokens.json")
        
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('shutil.copy2') as mock_copy:
                mock_exists.return_value = True
                
                world_path = Path(temp_worlds_dir) / "TestWorld"
                world_path.mkdir()
                
                result = world_manager._copy_default_tokens(world_path)
                
                assert result is True
                mock_copy.assert_called_once_with(
                    default_tokens_path,
                    world_path / "tokens.json"
                )
    
    def test_copy_default_tokens_file_not_found(self, world_manager, temp_worlds_dir):
        """Test handling when default tokens file doesn't exist"""
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        
        with patch('pathlib.Path.exists', return_value=False):
            with patch('builtins.open', mock_open()) as mock_file:
                result = world_manager._copy_default_tokens(world_path)
                
                assert result is False
                # Should create empty tokens file as fallback
                mock_file.assert_called_once_with(world_path / "tokens.json", 'w')
                mock_file().write.assert_called_once_with("[]")
    
    def test_copy_default_tokens_copy_error(self, world_manager, temp_worlds_dir):
        """Test handling copy error"""
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('shutil.copy2', side_effect=OSError("Permission denied")):
                result = world_manager._copy_default_tokens(world_path)
                
                assert result is False
    
    def test_create_world_copies_tokens(self, world_manager, temp_worlds_dir):
        """Test that creating a world copies default tokens"""
        with patch.object(world_manager, '_copy_default_tokens') as mock_copy:
            mock_copy.return_value = True
            
            result = world_manager.create_world("TestWorld")
            
            assert result is True
            mock_copy.assert_called_once()
            
            # Verify the world directory was created
            world_path = Path(temp_worlds_dir) / "TestWorld"
            assert world_path.exists()
    
    def test_load_world_tokens_success(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test successfully loading world tokens"""
        # Create world with tokens file
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        tokens_path = world_path / "tokens.json"
        
        with open(tokens_path, 'w') as f:
            json.dump(sample_tokens, f)
        
        tokens = world_manager.load_world_tokens("TestWorld")
        
        assert tokens == sample_tokens
    
    def test_load_world_tokens_file_not_found(self, world_manager, temp_worlds_dir):
        """Test loading tokens when file doesn't exist"""
        # Create world without tokens file
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        
        tokens = world_manager.load_world_tokens("TestWorld")
        
        assert tokens == []
    
    def test_load_world_tokens_invalid_json(self, world_manager, temp_worlds_dir):
        """Test loading tokens with invalid JSON"""
        # Create world with invalid tokens file
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        tokens_path = world_path / "tokens.json"
        
        with open(tokens_path, 'w') as f:
            f.write("invalid json")
        
        tokens = world_manager.load_world_tokens("TestWorld")
        
        assert tokens == []
    
    def test_save_world_tokens_success(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test successfully saving world tokens"""
        # Create world directory
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        
        result = world_manager.save_world_tokens("TestWorld", sample_tokens)
        
        assert result is True
        
        # Verify tokens were saved
        tokens_path = world_path / "tokens.json"
        assert tokens_path.exists()
        
        with open(tokens_path, 'r') as f:
            saved_tokens = json.load(f)
        
        assert saved_tokens == sample_tokens
    
    def test_save_world_tokens_updates_current(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test that saving tokens updates current_tokens if it's the current world"""
        # Create world directory
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        
        # Set as current world
        world_manager.current_world = "TestWorld"
        
        result = world_manager.save_world_tokens("TestWorld", sample_tokens)
        
        assert result is True
        assert world_manager.current_tokens == sample_tokens
    
    def test_save_world_tokens_write_error(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test handling write error when saving tokens"""
        # Don't create world directory to cause write error
        
        result = world_manager.save_world_tokens("TestWorld", sample_tokens)
        
        assert result is False
    
    def test_get_current_tokens_with_loaded_tokens(self, world_manager, sample_tokens):
        """Test getting current tokens when they're already loaded"""
        world_manager.current_tokens = sample_tokens
        
        tokens = world_manager.get_current_tokens()
        
        assert tokens == sample_tokens
    
    def test_get_current_tokens_loads_from_current_world(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test getting current tokens loads from current world"""
        # Create world with tokens
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        tokens_path = world_path / "tokens.json"
        
        with open(tokens_path, 'w') as f:
            json.dump(sample_tokens, f)
        
        # Set current world but not current tokens
        world_manager.current_world = "TestWorld"
        world_manager.current_tokens = None
        
        tokens = world_manager.get_current_tokens()
        
        assert tokens == sample_tokens
        assert world_manager.current_tokens == sample_tokens
    
    def test_get_current_tokens_no_current_world(self, world_manager):
        """Test getting current tokens when no world is loaded"""
        world_manager.current_world = None
        world_manager.current_tokens = None
        
        tokens = world_manager.get_current_tokens()
        
        assert tokens == []
    
    def test_get_tokens_by_category_all(self, world_manager, sample_tokens):
        """Test getting all tokens when no category filter"""
        world_manager.current_tokens = sample_tokens
        
        tokens = world_manager.get_tokens_by_category(None)
        
        assert tokens == sample_tokens
    
    def test_get_tokens_by_category_filtered(self, world_manager, sample_tokens):
        """Test getting tokens filtered by category"""
        world_manager.current_tokens = sample_tokens
        
        mood_tokens = world_manager.get_tokens_by_category("mood")
        
        assert len(mood_tokens) == 1
        assert mood_tokens[0]["token"] == "<mood_happy>"
        
        action_tokens = world_manager.get_tokens_by_category("action")
        
        assert len(action_tokens) == 1
        assert action_tokens[0]["token"] == "<stage_whisper>"
    
    def test_get_tokens_by_category_no_matches(self, world_manager, sample_tokens):
        """Test getting tokens for category with no matches"""
        world_manager.current_tokens = sample_tokens
        
        tokens = world_manager.get_tokens_by_category("nonexistent")
        
        assert tokens == []
    
    def test_load_world_loads_tokens(self, world_manager, sample_tokens, temp_worlds_dir):
        """Test that loading a world also loads its tokens"""
        # Create world with lore and tokens
        world_path = Path(temp_worlds_dir) / "TestWorld"
        world_path.mkdir()
        
        # Create world_lore.json
        lore_data = {
            "meta": {"version": 1},
            "facts": {},
            "factions": [],
            "timeline": [],
            "places": []
        }
        with open(world_path / "world_lore.json", 'w') as f:
            json.dump(lore_data, f)
        
        # Create tokens.json
        with open(world_path / "tokens.json", 'w') as f:
            json.dump(sample_tokens, f)
        
        # Load the world
        lore = world_manager.load_world("TestWorld")
        
        assert lore is not None
        assert world_manager.current_world == "TestWorld"
        assert world_manager.current_tokens == sample_tokens
    
    def test_integration_create_and_load_world_with_tokens(self, world_manager, sample_tokens, temp_worlds_dir):
        """Integration test: create world, save tokens, load world"""
        # Create a default tokens file
        core_data_dir = Path("core_data")
        core_data_dir.mkdir(exist_ok=True)
        default_tokens_path = core_data_dir / "tokens.json"
        
        try:
            with open(default_tokens_path, 'w') as f:
                json.dump(sample_tokens, f)
            
            # Create world (should copy default tokens)
            result = world_manager.create_world("TestWorld")
            assert result is True
            
            # Verify tokens were copied
            world_tokens_path = Path(temp_worlds_dir) / "TestWorld" / "tokens.json"
            assert world_tokens_path.exists()
            
            # Load the world
            lore = world_manager.load_world("TestWorld")
            assert lore is not None
            
            # Verify tokens are loaded
            assert world_manager.current_tokens == sample_tokens
            
            # Test getting tokens by category
            mood_tokens = world_manager.get_tokens_by_category("mood")
            assert len(mood_tokens) == 1
            assert mood_tokens[0]["token"] == "<mood_happy>"
            
        finally:
            # Clean up
            if default_tokens_path.exists():
                default_tokens_path.unlink()
            if core_data_dir.exists() and not any(core_data_dir.iterdir()):
                core_data_dir.rmdir() 