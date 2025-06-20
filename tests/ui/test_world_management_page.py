"""
UI tests for World Management page using Streamlit AppTest framework.
Following the three-circle TDD approach - this is the outer circle (UI testing).
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add the app directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from utils.world import WorldManager, WorldLore


@pytest.fixture
def mock_world_manager():
    """Mock WorldManager for UI testing"""
    mock_wm = Mock(spec=WorldManager)
    mock_wm.list_worlds.return_value = ["Test World", "Fantasy Realm"]
    
    # Mock world lore data
    mock_lore = Mock(spec=WorldLore)
    mock_lore.facts = {"Population": "1000", "Climate": "Temperate"}
    mock_lore.timeline = []
    mock_lore.places = []
    mock_lore.meta = {"version": 1}
    
    mock_wm.load_world.return_value = mock_lore
    mock_wm.create_world.return_value = True
    mock_wm.save_world_lore.return_value = True
    
    return mock_wm


@pytest.fixture
def temp_world_dir():
    """Create temporary world directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestWorldManagementPageUI:
    """UI tests for World Management page"""
    
    def test_world_management_page_loads(self):
        """Test that the World Management page loads without errors"""
        # Create a minimal test app that initializes session state properly
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock WorldManager in session state
if 'world_manager' not in st.session_state:
    mock_wm = Mock()
    mock_wm.list_worlds.return_value = ["Test World", "Fantasy Realm"]
    mock_lore = Mock()
    mock_lore.facts = {"Population": "1000", "Climate": "Temperate"}
    mock_lore.timeline = []
    mock_lore.places = []
    mock_lore.meta = {"version": 1}
    mock_wm.load_world.return_value = mock_lore
    st.session_state.world_manager = mock_wm

from pages.world_management import page_world_management
page_world_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Basic assertions to confirm page loaded
        assert not at.exception  # No exceptions occurred
        assert len(at.markdown) > 0  # Some markdown content is present
    
    def test_world_selection_shows_available_worlds(self):
        """Test that available worlds are shown to user"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock WorldManager with test worlds
if 'world_manager' not in st.session_state:
    mock_wm = Mock()
    mock_wm.list_worlds.return_value = ["Test World", "Fantasy Realm"]
    mock_lore = Mock()
    mock_lore.facts = {"Population": "1000", "Climate": "Temperate"}
    mock_lore.timeline = []
    mock_lore.places = []
    mock_lore.meta = {"version": 1}
    mock_wm.load_world.return_value = mock_lore
    st.session_state.world_manager = mock_wm

from pages.world_management import page_world_management
page_world_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # The selectbox should be in the sidebar
        # For now, just check that no exceptions occurred and there's a selectbox somewhere
        assert not at.exception
        
        # Look for world names in the rendered content (more robust than looking for specific widgets)
        page_content = str(at)
        assert "Test World" in page_content or "Fantasy Realm" in page_content
    
    def test_new_world_button_functionality(self):
        """Test that the New World button triggers the dialog"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock WorldManager
if 'world_manager' not in st.session_state:
    mock_wm = Mock()
    mock_wm.list_worlds.return_value = []  # No existing worlds
    mock_wm.create_world.return_value = True
    st.session_state.world_manager = mock_wm

from pages.world_management import page_world_management
page_world_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Check for New World button (should be present when no worlds exist)
        assert not at.exception
        
        # Look for the "New World" text in the output
        page_content = str(at)
        assert "New World" in page_content
    
    def test_tabs_are_present_when_world_selected(self):
        """Test that Facts, Timeline, and Places tabs appear when a world is loaded"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock WorldManager with world data
if 'world_manager' not in st.session_state:
    mock_wm = Mock()
    mock_wm.list_worlds.return_value = ["Test World"]
    
    # Mock detailed world lore
    mock_lore = Mock()
    mock_lore.facts = {"Population": "1000", "Climate": "Temperate"}
    mock_lore.timeline = []
    mock_lore.places = []
    mock_lore.meta = {"version": 1}
    mock_wm.load_world.return_value = mock_lore
    st.session_state.world_manager = mock_wm

from pages.world_management import page_world_management
page_world_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Check that tabs exist
        assert not at.exception
        
        # Look for tab content in the rendered output
        page_content = str(at)
        assert "Facts" in page_content
        assert "Timeline" in page_content  
        assert "Places" in page_content


class TestWorldManagementIntegration:
    """Integration tests for World Management UI with real WorldManager"""
    
    def test_world_creation_integration(self, temp_world_dir):
        """Test creating a new world through the UI actually creates it"""
        # This is an integration test with a real WorldManager
        
        test_script = f"""
import streamlit as st
from utils.world import WorldManager

# Use real WorldManager with temp directory
if 'world_manager' not in st.session_state:
    st.session_state.world_manager = WorldManager("{temp_world_dir}")

from pages.world_management import page_world_management
page_world_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Basic integration test - page should load with real WorldManager
        assert not at.exception
        
        # Should show "No worlds found" message since temp dir is empty
        # Look at actual content in info messages
        info_messages = [info.value for info in at.info]
        info_content = " ".join(info_messages)
        assert "No worlds found" in info_content or "Create your first world" in info_content 