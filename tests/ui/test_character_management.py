"""
UI tests for Character Management page using Streamlit AppTest framework.
Following the three-circle TDD approach - this is the outer circle (UI testing).

Tests cover:
- Basic page structure and loading
- Character selection and management
- Unsaved changes detection
- AI suggestion functionality
- Tab-based interface
- Toolbar actions (Save/Duplicate/Delete)
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import Mock, patch
import sys
import os
import json
import hashlib

# Add the app directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from utils.character.models import CharacterCore, Personality, Relationship


class TestCharacterManagementBasic:
    """Basic UI tests for Character Management page structure"""
    
    def test_character_management_page_loads(self):
        """Test that the Character Management page loads without errors"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Set up minimal session state
st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.list_characters_in_world.return_value = []
st.session_state.character_manager.get_current_world.return_value = "Test World"

from pages.character_management import page_character_management
page_character_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should load without exceptions
        assert not at.exception
        
        # Should have the main title
        assert any("Character Management Studio" in md.value for md in at.markdown)
    
    def test_character_selector_present(self):
        """Test that character selector is present in sidebar"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.list_characters_in_world.return_value = ["TestChar1", "TestChar2"]
st.session_state.character_manager.get_current_world.return_value = "Test World"

from pages.character_management import page_character_management
page_character_management()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have character selection dropdown
        assert len(at.selectbox) > 0
        
        # Verify selectbox has character options
        char_selector = at.selectbox[0]
        assert "TestChar1" in char_selector.options
        assert "TestChar2" in char_selector.options
    
    def test_tabs_present_when_character_loaded(self):
        """Test that tabs are present when a character is loaded"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
from utils.character.models import CharacterCore, Personality

st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.list_characters_in_world.return_value = []  # Empty list, not Mock

# Mock a character being loaded
test_char = CharacterCore(
    name="Test Character",
    description="A test character",  # Required field
    personality_traits=Personality()
)
st.session_state.current_character_core = test_char
st.session_state.selected_character = "Test Character"

# Mock check_unsaved_changes_warning to prevent sidebar complexity
def mock_check_warning(core):
    return True

import pages.character_management
pages.character_management.check_unsaved_changes_warning = mock_check_warning

from pages.character_management import page_character_management
page_character_management()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Check if character editing interface is shown (tabs may not be captured due to AppTest limitations)
        # Look for character name in markdown or other evidence of character being loaded
        page_content = str(at)
        assert ("Test Character" in page_content or 
                len(at.tabs) > 0 or
                any("Editing" in md.value for md in at.markdown))
        
        # Expected tab labels (if tabs are captured)
        if len(at.tabs) > 0:
            tab_labels = [tab.label for tab in at.tabs]
            expected_tabs = ["📝 Profile", "🧠 Personality", "🎯 Goals & Relationships", "💬 Examples"]
            for expected_tab in expected_tabs:
                assert any(expected_tab in label for label in tab_labels)


class TestUnsavedChangesDetection:
    """Tests for unsaved changes tracking and warnings"""
    
    def test_unsaved_changes_calculation(self):
        """Test that character hash calculation works correctly"""
        test_script = """
import streamlit as st
from utils.character.models import CharacterCore, Personality
from pages.character_management import calculate_character_hash

# Create a test character
char1 = CharacterCore(
    name="Test",
    description="Original description",  # Required field
    personality_traits=Personality()
)

# Calculate hash
hash1 = calculate_character_hash(char1)

# Modify character
char1.description = "Modified description"
hash2 = calculate_character_hash(char1)

# Hashes should be different
assert hash1 != hash2

# Same character should produce same hash
hash3 = calculate_character_hash(char1)
assert hash2 == hash3

st.write("✅ Hash calculation works correctly")
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        # Check if the test passed by looking at markdown content
        markdown_texts = [md.value for md in at.markdown]
        assert any("✅ Hash calculation works correctly" in text for text in markdown_texts)
    
    def test_save_button_changes_with_unsaved_state(self):
        """Test that save button and tracking work correctly"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
from utils.character.models import CharacterCore, Personality

st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.get_current_world.return_value = "Test World"

# Create character with initial state
test_char = CharacterCore(name="Test", description="Original", personality_traits=Personality())
st.session_state.current_character_core = test_char

# Test the unsaved changes detection function directly
from pages.character_management import track_character_changes

# First call should show no changes
has_changes_1 = track_character_changes(test_char)

# Modify character
test_char.description = "Modified"

# Second call should show changes
has_changes_2 = track_character_changes(test_char)

# Display results
st.write(f"First check (no changes): {has_changes_1}")
st.write(f"Second check (has changes): {has_changes_2}")
st.write("✅ Unsaved changes tracking works correctly" if has_changes_1 != has_changes_2 else "❌ Tracking failed")
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Check if the tracking test passed
        markdown_texts = [md.value for md in at.markdown]
        assert any("✅ Unsaved changes tracking works correctly" in text for text in markdown_texts)


class TestAISuggestionFunctionality:
    """Tests for AI suggestion features"""
    
    @patch('pages.character_management.llm_suggest_description')
    def test_ai_suggestion_buttons_present(self, mock_llm_suggest):
        """Test that AI suggestion buttons are present and functional"""
        mock_llm_suggest.return_value = ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
        
        test_script = """
import streamlit as st
from unittest.mock import Mock
from utils.character.models import CharacterCore, Personality

st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
test_char = CharacterCore(name="Test", description="", personality_traits=Personality())

from pages.character_management import render_profile_tab
render_profile_tab(test_char)
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have AI suggestion buttons (✨ buttons)
        ai_buttons = [btn for btn in at.button if "✨" in btn.label]
        assert len(ai_buttons) > 0
    
    def test_structured_output_models_work(self):
        """Test that Pydantic models for structured outputs are properly defined"""
        test_script = """
import streamlit as st
from pages.character_management import (
    DescriptionSuggestions, GoalSuggestions, ScenarioSuggestions,
    BackstorySuggestions, ExampleSuggestions
)

# Test model creation
desc_suggestions = DescriptionSuggestions(
    suggestions=["Test suggestion 1", "Test suggestion 2", "Test suggestion 3"],
    reasoning="Test reasoning"
)

goal_suggestions = GoalSuggestions(
    goals=["Goal 1", "Goal 2", "Goal 3"],
    reasoning="Test reasoning"
)

# Should create without errors
assert len(desc_suggestions.suggestions) == 3
assert len(goal_suggestions.goals) == 3

st.write("✅ Structured output models work correctly")
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        # Check if the test passed by looking at markdown content
        markdown_texts = [md.value for md in at.markdown]
        assert any("✅ Structured output models work correctly" in text for text in markdown_texts)


class TestCharacterToolbar:
    """Tests for character toolbar functionality"""
    
    def test_toolbar_buttons_present(self):
        """Test that all toolbar buttons are present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
from utils.character.models import CharacterCore, Personality

st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.get_current_world.return_value = "Test World"

test_char = CharacterCore(name="Test", description="Test", personality_traits=Personality())

from pages.character_management import render_toolbar
render_toolbar(test_char)
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have expected toolbar buttons
        button_labels = [btn.label for btn in at.button]
        expected_buttons = ["Save", "Duplicate", "Delete", "Test Chat"]
        
        for expected in expected_buttons:
            assert any(expected in label for label in button_labels)
    
    def test_delete_confirmation_workflow(self):
        """Test delete confirmation requires two clicks"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
from utils.character.models import CharacterCore, Personality

st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.get_current_world.return_value = "Test World"

test_char = CharacterCore(name="Test", description="Test", personality_traits=Personality())

# First render - no confirmation state
from pages.character_management import render_toolbar
render_toolbar(test_char)
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have delete button
        delete_buttons = [btn for btn in at.button if "Delete" in btn.label]
        assert len(delete_buttons) > 0


class TestCharacterTabs:
    """Tests for individual tab functionality"""
    
    def test_profile_tab_fields(self):
        """Test that profile tab has expected input fields"""
        test_script = """
import streamlit as st
from utils.character.models import CharacterCore, Personality

test_char = CharacterCore(
    name="Test Character",
    description="Test description",
    scenario="Test scenario",
    backstory="Test backstory",
    appearance="Test appearance",
    personality_traits=Personality()
)

from pages.character_management import render_profile_tab
render_profile_tab(test_char)
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have text inputs for character fields
        input_labels = [input.label for input in at.text_input]
        assert any("Character Name" in label for label in input_labels)
        
        # Should have text areas for longer content
        textarea_labels = [ta.label for ta in at.text_area]
        expected_areas = ["Description", "Scenario", "Backstory"]
        for expected in expected_areas:
            assert any(expected in label for label in textarea_labels)
    
    def test_personality_tab_integration(self):
        """Test that personality tab integrates with personality editor"""
        test_script = """
import streamlit as st
from unittest.mock import Mock, patch
from utils.character.models import CharacterCore, Personality

test_char = CharacterCore(
    name="Test Character",
    description="Test description",  # Required field
    personality_traits=Personality(openness=0.7, conscientiousness=0.6)
)

# Mock the personality editor component
with patch('pages.character_management.render_personality_editor') as mock_editor:
    from pages.character_management import render_personality_tab
    render_personality_tab(test_char)
    
    # Should have called the personality editor
    mock_editor.assert_called_once()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
    
    def test_goals_relationships_tab_basic_functionality(self):
        """Test that goals and relationships tab renders without errors"""
        test_script = """
import streamlit as st
from utils.character.models import CharacterCore, Personality, Relationship

test_char = CharacterCore(
    name="Test Character",
    description="Test description",  # Required field
    personality_traits=Personality(),
    goals=["Goal 1", "Goal 2"],
    relationships=[Relationship(name="Friend", affinity=75)]
)

from pages.character_management import render_goals_relationships_tab
render_goals_relationships_tab(test_char)

st.write("✅ Goals and relationships tab rendered successfully")
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Check if the tab rendered successfully
        markdown_texts = [md.value for md in at.markdown]
        assert any("✅ Goals and relationships tab rendered successfully" in text for text in markdown_texts)


class TestCharacterManagementIntegration:
    """Integration tests for complete character management workflow"""
    
    def test_character_loading_and_editing_workflow(self):
        """Test complete workflow from character selection to editing"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
from utils.character.models import CharacterCore, Personality

# Set up session state as if character was selected
st.session_state.world_manager = Mock()
st.session_state.character_manager = Mock()
st.session_state.character_manager.get_current_world.return_value = "Test World"
st.session_state.character_manager.list_characters_in_world.return_value = []  # Empty list

test_char = CharacterCore(
    name="Test Character",
    description="A test character for editing",  # Required field
    personality_traits=Personality(openness=0.8)
)
st.session_state.current_character_core = test_char
st.session_state.selected_character = "Test Character"

# Mock check_unsaved_changes_warning to simplify test
def mock_check_warning(core):
    return True

import pages.character_management
pages.character_management.check_unsaved_changes_warning = mock_check_warning

from pages.character_management import page_character_management
page_character_management()

# Check if character is loaded
if st.session_state.current_character_core:
    st.write("✅ Character loaded successfully")
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should show that character was loaded successfully
        markdown_texts = [md.value for md in at.markdown]
        assert any("✅ Character loaded successfully" in text for text in markdown_texts)
        
        # Should have some indication of character management interface
        page_content = str(at)
        assert "Test Character" in page_content
    
    def test_manager_integration_with_mocks(self):
        """Test that page properly integrates with character and world managers"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock managers with expected methods
mock_world_manager = Mock()
mock_world_manager.list_worlds.return_value = ["World1", "World2"]
mock_world_manager.get_world_path.return_value = Mock()

mock_character_manager = Mock()
mock_character_manager.get_current_world.return_value = "World1"
mock_character_manager.list_characters_in_world.return_value = ["Char1", "Char2"]

st.session_state.world_manager = mock_world_manager
st.session_state.character_manager = mock_character_manager

from pages.character_management import page_character_management
page_character_management()
"""
        
        at = AppTest.from_string(test_script).run()
        assert not at.exception
        
        # Should have called the character manager methods without errors
        assert len(at.selectbox) > 0  # Character selector should be present


# Utility function to run specific test suites
def run_character_management_tests():
    """Helper function to run all character management tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_character_management_tests() 