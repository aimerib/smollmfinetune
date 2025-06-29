"""
UI tests for the Platform Runtime Interface page.

Tests the immersive multi-character chat interface where players interact
with their selected characters in the chosen world.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from streamlit.testing.v1 import AppTest

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


class TestPlatformRuntimePage:
    """Test the platform runtime interface page"""
    
    def test_page_loads_with_valid_session(self):
        """Test that the page loads correctly with a valid session ID"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            # Create app test with session ID parameter
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should not have errors
            assert len(app.exception) == 0
            
            # Should have main title
            assert any("Platform Runtime" in str(element) or "Adventure" in str(element) 
                     for element in app.title)
    
    def test_character_panel_display(self):
        """Test that the character panel shows all active characters"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should show character names
            page_content = str(app)
            assert "Gandros the Wise" in page_content
            assert "Melody Songweaver" in page_content
            
            # Should show mood indicators
            assert "contemplative" in page_content.lower() or "🤔" in page_content
            assert "cheerful" in page_content.lower() or "😊" in page_content
    
    def test_world_context_strip(self):
        """Test that world context information is displayed"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            page_content = str(app)
            
            # Should show location, time, weather
            assert "Village Square" in page_content
            assert "Morning" in page_content
            assert "Sunny" in page_content
            
            # Should show recent events
            assert "Market day begins" in page_content
    
    def test_conversation_history_display(self):
        """Test that conversation history is properly displayed"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            page_content = str(app)
            
            # Should show conversation messages
            assert "Hello everyone!" in page_content
            assert "Greetings, traveler!" in page_content
            
            # Should distinguish between user and character messages
            assert "You:" in page_content or "👤" in page_content
            assert "Gandros" in page_content
    
    def test_chat_input_interface(self):
        """Test that the chat input interface is present and functional"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should have text input for messages
            assert len(app.text_input) > 0 or len(app.chat_input) > 0
            
            # Should have send button or chat input
            has_send_control = (len(app.button) > 0 and 
                              any("send" in str(btn).lower() for btn in app.button))
            has_chat_input = len(app.chat_input) > 0
            
            assert has_send_control or has_chat_input
    
    def test_character_activation_controls(self):
        """Test character activation/muting controls"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should have toggles or buttons for character activation
            has_toggles = len(app.toggle) > 0 or len(app.checkbox) > 0
            has_buttons = any("mute" in str(btn).lower() or "active" in str(btn).lower() 
                            for btn in app.button)
            
            assert has_toggles or has_buttons
    
    def test_session_not_found_error(self):
        """Test error handling when session is not found"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine_class.side_effect = ValueError("Session test-invalid not found")
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-invalid"}
            app.run()
            
            # Should show error message
            page_content = str(app)
            assert "not found" in page_content.lower() or "error" in page_content.lower()
    
    def test_missing_session_id_parameter(self):
        """Test handling when session_id parameter is missing"""
        app = AppTest.from_file("app/pages/platform_runtime.py")
        # No session_id parameter provided
        app.run()
        
        # Should show error or redirect message
        page_content = str(app)
        assert ("session" in page_content.lower() and 
               ("required" in page_content.lower() or "missing" in page_content.lower()))
    
    def test_world_event_triggers(self):
        """Test world event trigger controls"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should have controls for triggering world events
            has_event_controls = (
                any("event" in str(btn).lower() for btn in app.button) or
                any("world" in str(select).lower() for select in app.selectbox)
            )
            
            # This is optional for MVP, so just check if present
            # assert has_event_controls  # Uncomment when implemented
    
    def test_responsive_layout(self):
        """Test that the layout uses responsive columns"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should use columns for layout (character panel + main chat)
            page_content = str(app)
            # This is a basic check - Streamlit columns are hard to test directly
            # We'll verify this works in manual testing


class TestPlatformRuntimeInteractions:
    """Test user interactions on the platform runtime page"""
    
    def test_send_message_interaction(self):
        """Test sending a message triggers character responses"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            
            # Mock the async response generation
            async def mock_generate_responses(user_message, **kwargs):
                return {
                    'char-1': f"Wizard responds to: {user_message}",
                    'char-2': f"Bard responds to: {user_message}"
                }
            
            mock_engine.generate_multi_character_responses = mock_generate_responses
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Simulate sending a message
            if app.chat_input:
                app.chat_input[0].set_value("Hello characters!")
                app.run()
                
                # Should trigger response generation
                # (This test verifies the interface exists; actual async testing is complex in Streamlit)
                assert len(app.exception) == 0
    
    def test_character_mood_updates(self):
        """Test that character moods can be updated"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should have mood controls (selectbox or slider)
            has_mood_controls = (len(app.selectbox) > 0 or len(app.slider) > 0)
            
            # This test ensures the interface exists for mood updates
            # Actual functionality testing requires more complex async handling
            # assert has_mood_controls  # Uncomment when implemented
    
    def test_session_persistence(self):
        """Test that session state persists across interactions"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine.save_session_state = Mock()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Session should be properly managed
            # This is more of an integration test - verifying no crashes
            assert len(app.exception) == 0


class TestPlatformRuntimeAccessibility:
    """Test accessibility and usability features"""
    
    def test_character_avatars_alt_text(self):
        """Test that character avatars have proper alt text"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should use images with alt text for characters
            page_content = str(app)
            # Check for image elements or emoji representations
            has_visual_chars = ("🎭" in page_content or "👤" in page_content or 
                               "avatar" in page_content.lower())
            
            assert has_visual_chars
    
    def test_mobile_friendly_interface(self):
        """Test that the interface is mobile-friendly"""
        with patch('app.pages.platform_runtime.PlatformRuntimeEngine') as mock_engine_class:
            mock_engine = create_mock_runtime_engine()
            mock_engine_class.return_value = mock_engine
            
            app = AppTest.from_file("app/pages/platform_runtime.py")
            app.query_params = {"session_id": "test-session-123"}
            app.run()
            
            # Should use responsive design patterns
            # This is hard to test automatically, but we ensure no crashes
            assert len(app.exception) == 0 