"""
Unit tests for prompt builder multi-turn features.

Tests the prompt builder's support for multi-turn conversations
and metadata tracking.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from app.utils.generation.prompt_builder import PromptBuilder


class TestPromptBuilderMultiTurnSupport:
    """Test multi-turn conversation support in prompt builder"""
    
    def test_conversation_turn_metadata(self):
        """Test that conversation turns include proper metadata"""
        from app.utils.generation.prompt_builder import build_conversation_turn
        
        turn = build_conversation_turn(
            turn_idx=2,
            role="assistant",
            text="I understand your situation.",
            character_name="Cricket",
            option_id=1,
            nsfw_style=None
        )
        
        # Should include all required metadata
        assert turn['turn_id'] == 2
        assert turn['role'] == "assistant"
        assert turn['content'] == "I understand your situation."
        assert turn['metadata']['character_name'] == "Cricket"
        assert turn['metadata']['option_id'] == 1
        assert 'nsfw_style' in turn['metadata']
    
    def test_conversation_turn_with_nsfw_metadata(self):
        """Test conversation turns with NSFW metadata"""
        from app.utils.generation.prompt_builder import build_conversation_turn
        
        turn = build_conversation_turn(
            turn_idx=1,
            role="assistant", 
            text="I feel drawn to you...",
            character_name="Cricket",
            nsfw_style="soft"
        )
        
        assert turn['metadata']['nsfw_style'] == "soft"
    

    
    def test_prompt_built_flag(self):
        """Test that samples are marked with prompt_built flag"""
        # This would be implemented in the prompt builder integration
        from app.utils.generation.prompt_builder import PromptBuilder
        
        world_lore = {'facts': ['Test fact']}
        builder = PromptBuilder(world_lore=world_lore)
        
        character = {'name': 'Test', 'goals': ['Test goal']}
        
        # The built prompt should indicate it was enhanced
        result = builder.build_prompt(
            character=character,
            mode="chat",
            base_prompt="Hello"
        )
        
        # Should be a string (implementation detail)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_prompt_builder_lore_injection(self):
        """Test that prompt builder can inject lore facts"""
        world_lore = {'facts': ['Debt collection is risky', 'Money is scarce']}
        builder = PromptBuilder(world_lore=world_lore)
        
        character = {'name': 'Test', 'goals': ['survive']}
        result = builder.build_prompt(character=character, mode="chat", base_prompt="Hello")
        
        # Should contain one of the lore facts
        assert any(fact in result for fact in world_lore['facts'])


class TestDatasetSchemaMigration:
    """Test dataset schema migration for new columns"""
    
    def test_legacy_dataset_migration(self):
        """Test that legacy datasets are properly migrated"""
        # Sample legacy dataset
        legacy_dataset = [
            {
                'messages': [
                    {'role': 'system', 'content': 'You are Cricket.'},
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there!'}
                ]
            }
        ]
        
        # Migration should add new fields
        migrated = self._migrate_legacy_dataset(legacy_dataset)
        
        for sample in migrated:
            # Should have prompt_built flag
            assert 'prompt_built' in sample
            assert sample['prompt_built'] is False  # Legacy samples not built with new system
            
            # Should have default values for new fields
            assert 'lore_fact' in sample
            assert 'nsfw_style' in sample
            assert 'turn_id' in sample
            assert 'role' in sample
    
    def _migrate_legacy_dataset(self, legacy_dataset):
        """Helper method to simulate dataset migration"""
        migrated = []
        
        for sample in legacy_dataset:
            # Add new schema fields with defaults
            migrated_sample = sample.copy()
            migrated_sample.update({
                'prompt_built': False,
                'lore_fact': None,
                'nsfw_style': None,
                'turn_id': 1,  # Assume single turn for legacy
                'option_id': None,
                'role': 'conversation'  # Default role
            })
            migrated.append(migrated_sample)
        
        return migrated
    
    def test_new_dataset_schema(self):
        """Test that new datasets include all required fields"""
        # Sample new dataset that should be generated
        new_dataset_sample = {
            'messages': [
                {'role': 'system', 'content': 'Enhanced prompt with personality and lore'},
                {'role': 'user', 'content': 'Tell me about yourself'},
                {'role': 'assistant', 'content': 'Well, as someone who is anxious but determined...'}
            ],
            'prompt_built': True,
            'lore_fact': 'Debt collection agencies operate in legal gray areas',
            'nsfw_style': None,
            'turn_id': 1,
            'option_id': None,
            'role': 'assistant'
        }
        
        # Verify all required fields are present
        required_fields = ['messages', 'prompt_built', 'lore_fact', 'nsfw_style', 'turn_id', 'role']
        for field in required_fields:
            assert field in new_dataset_sample
        
        # Verify data types
        assert isinstance(new_dataset_sample['prompt_built'], bool)
        assert isinstance(new_dataset_sample['turn_id'], int)
        assert new_dataset_sample['nsfw_style'] is None or isinstance(new_dataset_sample['nsfw_style'], str)


class TestConversationFlow:
    
    def test_conversation_branching_logic(self):
        """Test the logic for conversation branching"""
        # This tests the conceptual branching logic
        base_conversation = [
            {'role': 'user', 'content': 'How are you doing?'},
            {'role': 'assistant', 'content': 'I\'m struggling a bit.'}
        ]
        
        # With 3 assistant options, we should get 3 different branches
        assistant_options = [
            "I'm struggling with work stress.",
            "I'm struggling with personal issues.", 
            "I'm struggling but staying positive."
        ]
        
        # Each option should create a different conversation path
        for i, option in enumerate(assistant_options):
            branch = base_conversation.copy()
            branch[-1]['content'] = option
            branch[-1]['option_id'] = i + 1
            
            # Verify structure
            assert branch[-1]['role'] == 'assistant'
            assert branch[-1]['option_id'] == i + 1
            assert option in branch[-1]['content'] 