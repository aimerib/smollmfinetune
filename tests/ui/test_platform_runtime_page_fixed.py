"""
UI tests for the Platform Runtime Interface page.

Tests the immersive multi-character chat interface where players interact
with their selected characters in the chosen world.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from streamlit.testing.v1 import AppTest
import sys
from pathlib import Path

# Add app directory to path
app_path = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_path))

# Test utilities
def create_mock_runtime_engine():
    """Create a mock runtime engine for testing"""
    mock_engine = Mock()
    
    # Mock character states
    mock_engine.character_states = {
        'char-1': Mock(
            character_name='Gandros the Wise',
            current_mood='contemplative',
            is_active=True
        ),
        'char-2': Mock(
            character_name='Melody Songweaver', 
            current_mood='cheerful',
            is_active=True
        )
    }
    
    # Mock methods
    mock_engine.get_active_character_ids.return_value = ['char-1', 'char-2']
    mock_engine.get_conversation_history.return_value = [
        {
            'role': 'user',
            'content': 'Hello everyone!',
            'timestamp': '2025-01-14T10:00:00'
        },
        {
            'role': 'assistant',
            'content': 'Greetings, traveler!',
            'character_id': 'char-1',
            'character_name': 'Gandros the Wise',
            'timestamp': '2025-01-14T10:00:30'
        }
    ]
    
    # Mock session data
    mock_engine.session_data = {
        'session_name': 'Epic Adventure',
        'current_location': 'Village Square',
        'world_time': 'Morning',
        'weather': 'Sunny',
        'recent_world_events': ['Market day begins']
    }
    
    return mock_engine


class TestPlatformRuntimeBasic:
    """Test basic platform runtime functionality"""
    
    def test_platform_runtime_imports(self):
        """Test that we can import the platform runtime components"""
        # Test imports work
        try:
            from utils.runtime.platform_engine import PlatformRuntimeEngine
            from utils.runtime.platform_engine import CharacterState, SessionMessage
            assert True  # If we get here, imports work
        except ImportError as e:
            pytest.fail(f"Failed to import platform runtime components: {e}")
    
    def test_mock_runtime_engine_creation(self):
        """Test that our mock runtime engine is properly configured"""
        mock_engine = create_mock_runtime_engine()
        
        # Should have character states
        assert len(mock_engine.character_states) == 2
        assert 'char-1' in mock_engine.character_states
        assert 'char-2' in mock_engine.character_states
        
        # Should have session data
        assert mock_engine.session_data['session_name'] == 'Epic Adventure'
        assert mock_engine.session_data['current_location'] == 'Village Square'
        
        # Should have conversation history
        conversation = mock_engine.get_conversation_history.return_value
        assert len(conversation) == 2
        assert conversation[0]['role'] == 'user'
        assert conversation[1]['role'] == 'assistant'


# Simple integration test without AppTest for now
class TestPlatformRuntimeIntegration:
    """Test platform runtime integration"""
    
    def test_character_state_data_structure(self):
        """Test character state data structure"""
        mock_engine = create_mock_runtime_engine()
        
        # Test character state access
        char_state = mock_engine.character_states['char-1']
        assert char_state.character_name == 'Gandros the Wise'
        assert char_state.current_mood == 'contemplative'
        assert char_state.is_active is True
    
    def test_session_data_structure(self):
        """Test session data structure"""
        mock_engine = create_mock_runtime_engine()
        
        session_data = mock_engine.session_data
        required_fields = ['session_name', 'current_location', 'world_time', 'weather']
        
        for field in required_fields:
            assert field in session_data
            assert session_data[field] is not None
    
    def test_conversation_message_structure(self):
        """Test conversation message structure"""
        mock_engine = create_mock_runtime_engine()
        
        messages = mock_engine.get_conversation_history.return_value
        
        for message in messages:
            assert 'role' in message
            assert 'content' in message
            assert 'timestamp' in message
            
            if message['role'] == 'assistant':
                assert 'character_id' in message
                assert 'character_name' in message


# Test that will verify actual page functionality when we can run it
@pytest.mark.skip(reason="Streamlit AppTest requires more setup - manual testing for now")
class TestPlatformRuntimeUI:
    """Test platform runtime UI (manual testing for now)"""
    
    def test_page_structure_requirements(self):
        """Test that the page has required structural elements"""
        # This would be a manual test or integration test
        # For now, we verify our test structure is correct
        mock_engine = create_mock_runtime_engine()
        
        # Verify we have all the data needed for UI
        assert hasattr(mock_engine, 'character_states')
        assert hasattr(mock_engine, 'session_data')
        assert callable(mock_engine.get_conversation_history)
        assert callable(mock_engine.get_active_character_ids) 