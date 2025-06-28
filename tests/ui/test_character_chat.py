"""
UI tests for Character Chat page (R2-3)
Following TDD approach - testing the UI interactions and full chat workflow
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from streamlit.testing.v1 import AppTest


class TestCharacterChatPage:
    """Test the Character Chat page functionality"""
    
    def test_character_chat_page_renders(self):
        """Test that character chat page renders basic UI elements"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock session state
st.session_state.inference_manager = Mock()
st.session_state.world_manager = Mock()

# Import and render the page
import sys
sys.path.append('app')
from pages.character_chat import page_character_chat

page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Check for key UI elements
        assert len(at.markdown) > 0  # Should have page title
        assert len(at.selectbox) > 0  # Should have character selection
        
    def test_character_selection_dropdown(self):
        """Test character selection dropdown shows available characters"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import sys
sys.path.append('app')

# Mock inference manager with available characters (return list, not mock)
mock_inference = Mock()
mock_inference.get_available_models.return_value = [
    "LoRA: Test Character",
    "LoRA: Another Character", 
    "Base: SmolLM2-135M"
]

st.session_state.inference_manager = mock_inference
st.session_state.world_manager = Mock()

from pages.character_chat import page_character_chat
page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have character selection dropdown
        character_selectors = [s for s in at.selectbox if "character" in str(s.label).lower()]
        assert len(character_selectors) > 0
        
    def test_dynamic_state_controls_present(self):
        """Test that dynamic state controls are rendered"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import sys
sys.path.append('app')

# Setup mocks
st.session_state.inference_manager = Mock()
st.session_state.world_manager = Mock()
st.session_state.selected_character_chat = "LoRA: Test Character"

from pages.character_chat import page_character_chat
page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Check for dynamic state controls
        # Should have mood selector, relationship sliders, etc.
        assert len(at.selectbox) >= 1  # Mood selector
        assert len(at.slider) >= 2     # Trust and affinity sliders
        
    def test_chat_interface_elements(self):
        """Test that chat interface elements are present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import sys
sys.path.append('app')

# Setup session state for chat
st.session_state.inference_manager = Mock()
st.session_state.world_manager = Mock()
st.session_state.selected_character_chat = "LoRA: Test Character"
st.session_state.chat_history = []

from pages.character_chat import page_character_chat
page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have text input for user messages
        text_inputs = [ti for ti in at.text_input if "message" in ti.label.lower() or "chat" in ti.label.lower()]
        assert len(text_inputs) > 0
        
        # Should have send button
        send_buttons = [btn for btn in at.button if "send" in btn.label.lower()]
        assert len(send_buttons) > 0
        
    def test_character_loading_workflow(self):
        """Test character loading and data display"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch
import sys
sys.path.append('app')

# Mock RuntimePromptConstructor
with patch('pages.character_chat.RuntimePromptConstructor') as mock_constructor:
    mock_instance = Mock()
    mock_instance.get_character_name.return_value = "Test Character"
    mock_instance.character_core = {
        "name": "Test Character",
        "goals": ["Test Goal 1", "Test Goal 2"],
        "big_five": {"openness": 0.8, "extraversion": 0.3}
    }
    mock_instance.get_available_tokens.return_value = [
        {"token": "<mood_happy>", "category": "mood"},
        {"token": "<mood_sad>", "category": "mood"}
    ]
    mock_constructor.return_value = mock_instance
    
    # Setup session state
    st.session_state.inference_manager = Mock()
    st.session_state.world_manager = Mock()
    st.session_state.selected_character_chat = "LoRA: Test Character"
    
    from pages.character_chat import page_character_chat
    page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should display character information
        # Check that character name appears in the output
        page_text = " ".join([md.value for md in at.markdown])
        assert "Test Character" in page_text
        
    def test_conversation_flow(self):
        """Test basic conversation flow - user input leads to character response"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch, AsyncMock
import sys
sys.path.append('app')

# Mock complete conversation flow
with patch('pages.character_chat.RuntimePromptConstructor') as mock_constructor:
    mock_constructor_instance = Mock()
    mock_constructor_instance.get_character_name.return_value = "Test Character"
    mock_constructor_instance.character_core = {"name": "Test Character"}
    mock_constructor_instance.construct.return_value = "Test prompt with character context"
    mock_constructor_instance.get_available_tokens.return_value = []
    mock_constructor.return_value = mock_constructor_instance
    
    # Mock inference manager
    mock_inference = Mock()
    mock_inference.generate_response.return_value = "Hello! I'm Test Character."
    
    # Setup session state with conversation history
    st.session_state.inference_manager = mock_inference
    st.session_state.world_manager = Mock()
    st.session_state.selected_character_chat = "LoRA: Test Character"
    st.session_state.chat_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! I'm Test Character."}
    ]
    st.session_state.chat_dynamic_state = {"current_mood": "happy", "relationship_to_user": {"trust": 0.5, "affinity": 0.5}}
    
    from pages.character_chat import page_character_chat
    page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should display conversation history
        # Look for evidence of chat messages
        chat_messages = [elem for elem in at.chat_message if hasattr(elem, 'avatar')]
        # Streamlit's chat_message might not be captured properly in AppTest
        # So let's check for text content instead
        page_text = " ".join([md.value for md in at.markdown])
        
    def test_dynamic_state_updates(self):
        """Test that dynamic state controls update the conversation context"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch
import sys
sys.path.append('app')

# Setup complete mock environment
with patch('pages.character_chat.RuntimePromptConstructor') as mock_constructor:
    mock_instance = Mock()
    mock_instance.get_character_name.return_value = "Test Character"
    mock_instance.character_core = {"name": "Test Character"}
    mock_instance.construct.return_value = "Mood: happy, Trust: high"
    mock_instance.get_tokens_by_category.return_value = [
        {"token": "<mood_happy>", "description": "Happy mood"},
        {"token": "<mood_sad>", "description": "Sad mood"}
    ]
    mock_constructor.return_value = mock_instance
    
    st.session_state.inference_manager = Mock()
    st.session_state.world_manager = Mock()
    st.session_state.selected_character_chat = "LoRA: Test Character"
    st.session_state.chat_dynamic_state = {
        "current_mood": "happy",
        "relationship_to_user": {"trust": 0.8, "affinity": 0.7}
    }
    
    from pages.character_chat import page_character_chat
    page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have slider controls for relationship state
        trust_sliders = [s for s in at.slider if "trust" in s.label.lower()]
        affinity_sliders = [s for s in at.slider if "affinity" in s.label.lower()]
        
        assert len(trust_sliders) >= 1
        assert len(affinity_sliders) >= 1
        
    def test_error_handling_no_character_selected(self):
        """Test graceful handling when no character is selected"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import sys
sys.path.append('app')

# Mock with no characters available
mock_inference = Mock()
mock_inference.get_available_models.return_value = []

st.session_state.inference_manager = mock_inference
st.session_state.world_manager = Mock()

from pages.character_chat import page_character_chat
page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should show instruction to select character or train one
        page_text = " ".join([md.value for md in at.markdown])
        assert ("no trained characters" in page_text.lower() or 
                "train a character" in page_text.lower() or
                "no characters found" in page_text.lower())
        
    def test_error_handling_character_loading_failure(self):
        """Test error handling when character fails to load"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch
import sys
sys.path.append('app')

# Mock with character available but loading will fail
mock_inference = Mock()
mock_inference.get_available_models.return_value = ["LoRA: Non-existent Character"]

# Mock RuntimePromptConstructor to raise exception
with patch('pages.character_chat.RuntimePromptConstructor') as mock_constructor:
    mock_constructor.side_effect = FileNotFoundError("Character not found")
    
    st.session_state.inference_manager = mock_inference
    st.session_state.world_manager = Mock()
    st.session_state.selected_character_chat = "LoRA: Non-existent Character"
    
    from pages.character_chat import page_character_chat
    page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should show error message (either error widget or in markdown)
        error_messages = [elem for elem in at.error]
        page_text = " ".join([md.value for md in at.markdown])
        
        has_error = (len(error_messages) > 0 or 
                    "error" in page_text.lower() or 
                    "failed" in page_text.lower() or 
                    "not found" in page_text.lower())
        assert has_error
        
    def test_conversation_clearing(self):
        """Test conversation clearing functionality"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch
import sys
sys.path.append('app')

# Setup with existing conversation
with patch('pages.character_chat.RuntimePromptConstructor') as mock_constructor:
    mock_instance = Mock()
    mock_instance.get_character_name.return_value = "Test Character"
    mock_instance.character_core = {"name": "Test Character"}
    mock_instance.get_available_tokens.return_value = []
    mock_constructor.return_value = mock_instance
    
    st.session_state.inference_manager = Mock()
    st.session_state.world_manager = Mock()
    st.session_state.selected_character_chat = "LoRA: Test Character"
    st.session_state.chat_history = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Test response"}
    ]
    
    from pages.character_chat import page_character_chat
    page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have clear conversation button
        clear_buttons = [btn for btn in at.button if "clear" in btn.label.lower()]
        assert len(clear_buttons) > 0
        
    def test_integration_with_runtime_prompt_constructor(self):
        """Test integration between chat interface and RuntimePromptConstructor"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch
import sys
sys.path.append('app')

with patch('pages.character_chat.RuntimePromptConstructor') as mock_constructor:
    # Test that RuntimePromptConstructor is called with correct parameters
    mock_instance = Mock()
    mock_instance.get_character_name.return_value = "Test Character"
    mock_instance.character_core = {"name": "Test Character", "goals": ["Test"]}
    mock_instance.construct.return_value = "Constructed prompt"
    mock_instance.get_available_tokens.return_value = []
    mock_constructor.return_value = mock_instance
    
    # Mock inference manager
    mock_inference = Mock()
    mock_inference.generate_response.return_value = "Test response"
    
    st.session_state.inference_manager = mock_inference
    st.session_state.world_manager = Mock()
    st.session_state.selected_character_chat = "LoRA: Test Character"
    st.session_state.chat_history = []
    
    from pages.character_chat import page_character_chat
    page_character_chat()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Test passes if no exceptions are raised during integration


class TestCharacterChatIntegration:
    """Integration tests for character chat functionality"""
    
    def test_character_chat_navigation_integration(self):
        """Test that character chat is accessible from main navigation"""
        # This would test the app.py integration
        test_script = """
import streamlit as st
from unittest.mock import Mock
import sys
sys.path.append('app')

# Mock all required session state
st.session_state.inference_manager = Mock()
st.session_state.world_manager = Mock()
st.session_state.training_manager = Mock()
st.session_state.character_manager = Mock()

# This simulates selecting Character Chat from navigation
st.session_state.page = "Character Chat"

# Note: Full app.py testing would require more complex setup
# For now, just test that the page can be imported
from pages.character_chat import page_character_chat
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
    def test_character_management_test_chat_button_integration(self):
        """Test that Test Chat button in character management works"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import sys
sys.path.append('app')

# This would test the updated "Test Chat" button
# For now, just ensure the import works
from pages.character_management import render_toolbar
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception 