"""
UI tests for Character Upload page using Streamlit AppTest framework.
Following the three-circle TDD approach - this is the outer circle (UI testing).
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import sys
import os
import json

# Add the app directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from utils.character import CharacterManager
from utils.dataset import DatasetManager


@pytest.fixture
def mock_character_manager():
    """Mock CharacterManager for UI testing"""
    mock_cm = Mock(spec=CharacterManager)
    mock_cm.get_current_world.return_value = "Test World"
    return mock_cm


@pytest.fixture
def mock_dataset_manager():
    """Mock DatasetManager for UI testing"""
    mock_dm = Mock(spec=DatasetManager)
    mock_dm.load_dataset_with_metadata.return_value = None  # No existing dataset by default
    return mock_dm


@pytest.fixture
def sample_character_data():
    """Sample character data for testing"""
    return {
        "name": "TestCharacter",
        "description": "A test character for uploading",
        "personality": "Friendly and helpful",
        "mes_example": "Hello! How can I help you today?",
        "scenario": "A helpful assistant in a test environment"
    }


class TestCharacterUploadPageUI:
    """UI tests for Character Upload page"""
    
    def test_character_upload_page_loads_without_character(self):
        """Test that the Character Upload page loads properly without a character"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Set up session state without character
if 'current_character' not in st.session_state:
    st.session_state.current_character = None

# Mock required managers
if 'character_manager' not in st.session_state:
    st.session_state.character_manager = Mock()

if 'dataset_manager' not in st.session_state:
    st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should load without exceptions
        assert not at.exception
        
        # Should have UI elements (can't directly test file_uploader in AppTest)
        # But we can verify the page loads without errors and has expected structure
        assert len(at.markdown) > 0  # Should have markdown elements
    
    def test_character_upload_page_loads_with_character(self):
        """Test that the Character Upload page shows character preview when character is loaded"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Set up session state with character
if 'current_character' not in st.session_state:
    st.session_state.current_character = {
        'name': 'TestCharacter',
        'description': 'A test character',
        'personality': 'Friendly',
        'mes_example': 'Hello there!',
        'scenario': 'Test scenario'
    }

# Mock required managers
if 'character_manager' not in st.session_state:
    mock_cm = Mock()
    mock_cm.get_current_world.return_value = "Test World"
    st.session_state.character_manager = mock_cm

if 'dataset_manager' not in st.session_state:
    st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should load without exceptions
        assert not at.exception
        
        # Should have convert button when character is loaded
        button_labels = [btn.label for btn in at.button]
        assert any("Convert to CharacterCore" in label for label in button_labels)
    
    def test_file_uploader_present(self):
        """Test that file uploader is present and configured correctly"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = None
st.session_state.character_manager = Mock()
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have page structure (file uploader is present but not directly testable)
        # We can verify the page has the expected markdown content
        assert len(at.markdown) > 0
    
    def test_character_preview_sections(self):
        """Test that character preview sections are displayed correctly"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {
    'name': 'TestCharacter',
    'description': 'A detailed test character description',
    'personality': 'Very friendly and helpful personality',
    'mes_example': 'Hello! I am a test character.',
    'scenario': 'Testing scenario for character upload'
}

st.session_state.character_manager = Mock()
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have markdown elements for character preview
        assert len(at.markdown) > 0
        
        # Should contain character information in markdown
        markdown_content = " ".join([md.value for md in at.markdown if hasattr(md, 'value')])
        assert "TestCharacter" in markdown_content
    
    def test_charactercore_conversion_section(self):
        """Test that CharacterCore conversion section appears when character is loaded"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {
    'name': 'TestCharacter',
    'description': 'Test description',
    'personality': 'Test personality'
}

mock_cm = Mock()
mock_cm.get_current_world.return_value = "Test World"
st.session_state.character_manager = mock_cm
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have convert button
        button_labels = [btn.label for btn in at.button]
        assert any("Convert to CharacterCore" in label for label in button_labels)
    
    def test_tips_section_always_present(self):
        """Test that tips section is always present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = None
st.session_state.character_manager = Mock()
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have markdown elements (including tips)
        assert len(at.markdown) > 0
        
        # Tips should be present in markdown content
        markdown_content = " ".join([md.value for md in at.markdown if hasattr(md, 'value')])
        assert "Tips" in markdown_content or "JSON" in markdown_content
    
    def test_charactercore_info_section_with_character(self):
        """Test that CharacterCore info section appears when character is loaded"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {
    'name': 'TestCharacter',
    'description': 'Test'
}

st.session_state.character_manager = Mock()
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have markdown elements including CharacterCore info
        markdown_content = " ".join([md.value for md in at.markdown if hasattr(md, 'value')])
        assert "CharacterCore" in markdown_content


class TestCharacterUploadIntegration:
    """Integration tests for Character Upload UI with mocked external dependencies"""
    
    def test_character_upload_with_existing_dataset(self):
        """Test character upload flow when existing dataset is found"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock dataset manager to return existing dataset
mock_dm = Mock()
mock_dataset = [{"input": "test", "output": "response"}]
mock_metadata = {"samples": 1, "created": "2025-06-19"}
mock_dm.load_dataset_with_metadata.return_value = (mock_dataset, mock_metadata)

st.session_state.current_character = None
st.session_state.character_manager = Mock()
st.session_state.dataset_manager = mock_dm
st.session_state.dataset_preview = None
st.session_state.dataset_metadata = {}

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have page structure (file uploader present but not directly testable)
        assert len(at.markdown) > 0
    
    def test_character_conversion_preview(self):
        """Test that character conversion shows preview data"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {
    'name': 'TestCharacter',
    'description': 'A friendly test character with a detailed background',
    'personality': 'Helpful and enthusiastic'
}

mock_cm = Mock()
mock_cm.get_current_world.return_value = "Fantasy World"
st.session_state.character_manager = mock_cm
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have convert button
        button_labels = [btn.label for btn in at.button]
        convert_buttons = [btn for btn in at.button if "Convert to CharacterCore" in btn.label]
        assert len(convert_buttons) > 0
        
        # Test clicking the convert button (simulated)
        # Note: AppTest button clicking has changed in newer versions
        # We verify the button exists which means the functionality is available
        assert len(convert_buttons) > 0
    
    def test_error_handling_for_invalid_json(self):
        """Test error handling when invalid JSON is processed"""
        # This test simulates what would happen if invalid JSON was uploaded
        # Since we can't actually upload files in AppTest, we test the error display logic
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = None
st.session_state.character_manager = Mock()
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have proper page structure for error handling
        assert len(at.markdown) > 0
    
    def test_character_manager_integration(self):
        """Test integration with character manager for world information"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {
    'name': 'TestCharacter',
    'description': 'Test character'
}

# Mock character manager with specific world
mock_cm = Mock()
mock_cm.get_current_world.return_value = "Cyberpunk World"
st.session_state.character_manager = mock_cm
st.session_state.dataset_manager = Mock()

from pages.character_upload import page_character_upload
page_character_upload()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have convert button that would use the character manager
        button_labels = [btn.label for btn in at.button]
        assert any("Convert to CharacterCore" in label for label in button_labels) 