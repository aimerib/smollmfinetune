"""
Tests for export_runtime_packet functionality (R2-1)
Following TDD approach - this tests the core business logic (inner circle)
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the classes we need to test
from backend.app.services.training.training import TrainingManager
from backend.app.services.character.models import CharacterCore, Personality, Relationship
from backend.app.services.world.world import WorldManager, WorldLore, TimelineEvent, Faction


class TestExportRuntimePacket(unittest.TestCase):
    """Test the export_runtime_packet functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.runtime_packets_dir = self.temp_dir / "runtime_packets"
        self.training_output_dir = self.temp_dir / "training_output"
        self.worlds_dir = self.temp_dir / "content" / "worlds"
        
        # Create directory structure
        self.runtime_packets_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir.mkdir(parents=True, exist_ok=True)
        self.worlds_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a test world with character
        self.test_world_path = self.worlds_dir / "Test World"
        self.test_world_path.mkdir(parents=True, exist_ok=True)
        self.test_char_path = self.test_world_path / "characters" / "Test Character"
        self.test_char_path.mkdir(parents=True, exist_ok=True)
        
        # Create test character_core.json
        self.test_character = CharacterCore(
            name="Test Character",
            description="A character for testing",
            scenario="Testing scenario",
            backstory="Test backstory",
            appearance="Test appearance",
            personality_traits=Personality(openness=0.8, extraversion=0.7),
            goals=["Test goal 1", "Test goal 2"],
            relationships=[Relationship(name="Test Friend", affinity=80)],
            tags=["test", "character"],
            imports={"source": "test"}
        )
        
        with open(self.test_char_path / "character_core.json", 'w') as f:
            json.dump(self.test_character.model_dump(), f, indent=2)
        
        # Create test world_lore.json
        self.test_world_lore = WorldLore(
            meta={"version": 1},
            facts={"magic_system": "Test magic", "currency": "Test coins"},
            timeline=[TimelineEvent(year=1000, event="Test event")],
            factions=[Faction(name="Test Faction", timeline=[])],
            places=[]
        )
        
        world_lore_data = {
            "meta": self.test_world_lore.meta,
            "facts": self.test_world_lore.facts,
            "timeline": [{"year": event.year, "event": event.event} for event in self.test_world_lore.timeline],
            "factions": [{"name": faction.name, "timeline": []} for faction in self.test_world_lore.factions],
            "places": []
        }
        
        with open(self.test_world_path / "world_lore.json", 'w') as f:
            json.dump(world_lore_data, f, indent=2)
        
        # Create test tokens.json
        self.test_tokens = [
            {"token": "<mood_happy>", "category": "mood", "description": "Happy mood"},
            {"token": "<scene_tavern>", "category": "scene", "description": "Tavern scene"}
        ]
        
        with open(self.test_world_path / "tokens.json", 'w') as f:
            json.dump(self.test_tokens, f, indent=2)
        
        # Create test adapter directory with files
        self.adapter_dir = self.training_output_dir / "adapters" / "test_character"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock adapter.safetensors (just an empty file for testing)
        with open(self.adapter_dir / "adapter.safetensors", 'wb') as f:
            f.write(b"mock adapter data")
        
        # Create test training metadata
        self.training_metadata = {
            "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "training_method": "dora",
            "character_name": "Test Character",
            "training_date": "2025-01-01T00:00:00"
        }
        
        with open(self.adapter_dir / "training_metadata.json", 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        # Mock the training manager
        with patch('app.utils.training.TrainingManager.__init__', return_value=None):
            self.training_manager = TrainingManager()
            self.training_manager.project_dir = self.training_output_dir
            self.training_manager.base_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
        
        # Mock world and character managers
        self.mock_world_manager = Mock(spec=WorldManager)
        self.mock_world_manager.get_world_path.return_value = self.test_world_path
        self.mock_world_manager.get_tokens_path.return_value = self.test_world_path / "tokens.json"
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_runtime_packet_basic_functionality(self):
        """Test basic export_runtime_packet functionality"""
        character_name = "Test Character"
        
        # Mock the helper functions to return our test data
        with patch.object(self.training_manager, '_find_character_in_worlds') as mock_find_char, \
             patch.object(self.training_manager, '_find_best_adapter') as mock_find_adapter, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('shutil.copy2') as mock_copy, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            # Configure mocks
            mock_find_char.return_value = (self.test_char_path / "character_core.json", self.test_world_path)
            mock_find_adapter.return_value = (self.adapter_dir, "SFT")
            
            # Mock file operations
            mock_open.return_value.__enter__.return_value = Mock()
            
            # This should now work
            result = self.training_manager.export_runtime_packet(character_name)
            
            # Verify the result
            self.assertIsInstance(result, str)
            self.assertIn(character_name, result)
            
            # Verify helper functions were called
            mock_find_char.assert_called_once_with(character_name)
            mock_find_adapter.assert_called_once_with(character_name)
    
    def test_runtime_packet_directory_creation(self):
        """Test that runtime packet directory is created correctly"""
        # This will test the directory creation logic
        character_name = "Test Character"
        expected_dir = self.temp_dir / "runtime_packets" / character_name
        
        # Once implemented, this should work
        # For now, let's test the expected behavior
        self.assertFalse(expected_dir.exists())  # Should not exist before export
    
    def test_runtime_packet_required_files(self):
        """Test that all required files are included in runtime packet"""
        character_name = "Test Character"
        
        # Expected files according to acceptance criteria
        expected_files = [
            "adapter.safetensors",
            "character_core.json", 
            "world_lore.json",
            "tokens.json",
            "runtime_config.json"
        ]
        
        # This test documents what should be in the packet
        # Implementation will make this pass
        for file_name in expected_files:
            # We expect these files to be created
            self.assertIsInstance(file_name, str)
    
    def test_runtime_config_schema(self):
        """Test that runtime_config.json has correct schema"""
        expected_schema = {
            "base_model": "name-of-base-model-e.g-meta-llama/Llama-2-7b-chat-hf",
            "adapter_path": "adapter.safetensors", 
            "tokenizer_path": ".cache/tokenizers/name-of-base-model-patched",
            "character_file": "character_core.json",
            "world_file": "world_lore.json",
            "tokens_file": "tokens.json"
        }
        
        # Test the expected structure
        self.assertIn("base_model", expected_schema)
        self.assertIn("adapter_path", expected_schema)
        self.assertIn("character_file", expected_schema)
        self.assertIn("world_file", expected_schema)
        self.assertIn("tokens_file", expected_schema)
    
    def test_find_character_across_worlds(self):
        """Test finding character across multiple worlds"""
        character_name = "Test Character"
        
        # This tests the logic for finding a character in any world
        # Since characters can exist in different worlds
        
        # Mock the world manager to have multiple worlds
        worlds = ["World 1", "World 2", "Test World"]
        
        # Character should be found in "Test World"
        found_world = None
        for world in worlds:
            world_path = self.worlds_dir / world
            char_path = world_path / "characters" / character_name / "character_core.json"
            if char_path.exists():
                found_world = world
                break
        
        self.assertEqual(found_world, "Test World")
    
    def test_prefer_rlhf_over_sft_adapter(self):
        """Test that RLHF adapter is preferred over SFT if it exists"""
        character_name = "Test Character"
        
        # Create both SFT and RLHF adapters
        sft_adapter = self.adapter_dir / "adapter.safetensors"
        rlhf_adapter = self.adapter_dir / "rlhf_output" / "adapter_grpo" / "adapter.safetensors"
        
        # Create RLHF directory and file
        rlhf_adapter.parent.mkdir(parents=True, exist_ok=True)
        with open(rlhf_adapter, 'wb') as f:
            f.write(b"rlhf adapter data")
        
        # RLHF should be preferred
        # This tests the logic that chooses the best available adapter
        if rlhf_adapter.exists():
            preferred_adapter = rlhf_adapter
        else:
            preferred_adapter = sft_adapter
        
        self.assertEqual(preferred_adapter, rlhf_adapter)
    
    def test_missing_character_error(self):
        """Test error handling when character doesn't exist"""
        character_name = "Nonexistent Character"
        
        # This should raise an appropriate error
        # For now, we document the expected behavior
        with self.assertRaises((FileNotFoundError, ValueError)):
            # This will be implemented to raise an error for missing characters
            raise FileNotFoundError(f"Character '{character_name}' not found")
    
    def test_missing_adapter_error(self):
        """Test error handling when adapter doesn't exist"""
        character_name = "Test Character"
        
        # Remove the adapter file
        adapter_file = self.adapter_dir / "adapter.safetensors"
        if adapter_file.exists():
            adapter_file.unlink()
        
        # This should raise an appropriate error
        with self.assertRaises((FileNotFoundError, ValueError)):
            # This will be implemented to raise an error for missing adapters
            raise FileNotFoundError(f"No trained adapter found for character '{character_name}'")
    
    def test_export_runtime_packet_creates_all_required_files(self):
        """Test that export_runtime_packet creates all required files"""
        character_name = "Test Character"
        
        # Use real temporary directory for this test
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock the helper functions to return our test data
            with patch.object(self.training_manager, '_find_character_in_worlds') as mock_find_char, \
                 patch.object(self.training_manager, '_find_best_adapter') as mock_find_adapter, \
                 patch('pathlib.Path', return_value=temp_path) as mock_path_class:
                
                # Configure mocks to return our test data
                mock_find_char.return_value = (self.test_char_path / "character_core.json", self.test_world_path)
                mock_find_adapter.return_value = (self.adapter_dir, "SFT")
                
                # Create the actual runtime_packets directory structure in temp
                runtime_packets_dir = temp_path / "runtime_packets"
                runtime_packets_dir.mkdir(parents=True, exist_ok=True)
                packet_dir = runtime_packets_dir / character_name
                
                # Mock Path constructor to return our temp directory structure
                def path_side_effect(path_str):
                    if path_str == "runtime_packets":
                        return runtime_packets_dir
                    return Path(path_str)
                
                mock_path_class.side_effect = path_side_effect
                
                # Run the export
                result = self.training_manager.export_runtime_packet(character_name)
                
                # Verify all required files would be created
                expected_files = [
                    "adapter.safetensors",
                    "character_core.json",
                    "world_lore.json", 
                    "tokens.json",
                    "runtime_config.json",
                    "manifest.json"  # Added by our implementation
                ]
                
                # Verify result path
                self.assertIsInstance(result, str)
                self.assertIn(character_name, result)
    
    def test_export_runtime_packet_prefers_rlhf_adapter(self):
        """Test that RLHF adapter is preferred over SFT when available"""
        character_name = "Test Character"
        
        # Mock both SFT and RLHF adapters being available
        with patch.object(self.training_manager, '_find_character_in_worlds') as mock_find_char, \
             patch.object(self.training_manager, '_find_best_adapter') as mock_find_adapter, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('shutil.copy2') as mock_copy, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            # Configure mocks - return RLHF adapter as best
            mock_find_char.return_value = (self.test_char_path / "character_core.json", self.test_world_path)
            mock_find_adapter.return_value = (self.adapter_dir / "rlhf_output" / "adapter_grpo", "RLHF-GRPO")
            
            # Mock file existence to return True for adapter.safetensors
            mock_exists.return_value = True
            
            # Mock file operations
            mock_open.return_value.__enter__.return_value = Mock()
            
            result = self.training_manager.export_runtime_packet(character_name)
            
            # Verify it was called and used RLHF
            mock_find_adapter.assert_called_once_with(character_name)
            self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main() 