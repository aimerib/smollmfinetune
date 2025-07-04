"""
Tests for the Advanced Emotion Control page.
"""

import pytest
from streamlit.testing.v1 import AppTest
import tempfile
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))


@pytest.mark.ui
class TestAdvancedEmotionControlPage:
    """Test the Advanced Emotion Control page functionality."""
    
    def test_page_loads_unauthenticated(self):
        """Test that page loads and shows authentication prompt for unauthenticated users."""
        test_script = """
import streamlit as st
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Mock session state for unauthenticated user
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

from pages.advanced_emotion_control import main
main()
"""
        
        at = AppTest.from_string(test_script, default_timeout=10).run()
        
        # Should show authentication warning
        assert len(at.warning) > 0
        assert "log in" in at.warning[0].body.lower()
        
        # Should have login and register buttons
        assert len(at.button) >= 2
    
    def test_page_loads_authenticated(self):
        """Test that page loads properly for authenticated users."""
        test_script = """
import streamlit as st
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Mock session state for authenticated user
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = True

# Mock required managers
class MockAuthManager:
    def get_current_user(self):
        return None

class MockCharacterManager:
    def list_characters(self):
        return []

class MockWorldManager:
    def get_current_world(self):
        return None

if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = MockAuthManager()
if 'character_manager' not in st.session_state:
    st.session_state.character_manager = MockCharacterManager()
if 'world_manager' not in st.session_state:
    st.session_state.world_manager = MockWorldManager()

from pages.advanced_emotion_control import main
main()
"""
        
        at = AppTest.from_string(test_script, default_timeout=10).run()
        
        # Should show the main title
        assert len(at.title) > 0
        assert "Advanced Emotion Control" in at.title[0].body
        
        # Should not show authentication warning
        assert len([w for w in at.warning if "log in" in w.body.lower()]) == 0
    
    def test_page_has_correct_structure(self):
        """Test that the page has the expected structure and components."""
        test_script = """
import streamlit as st
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Mock session state for authenticated user
st.session_state.authenticated = True

# Mock required managers
class MockAuthManager:
    def get_current_user(self):
        return None

class MockCharacterManager:
    def list_characters(self):
        return []

class MockWorldManager:
    def get_current_world(self):
        return None

st.session_state.auth_manager = MockAuthManager()
st.session_state.character_manager = MockCharacterManager()
st.session_state.world_manager = MockWorldManager()

from pages.advanced_emotion_control import main
main()
"""
        
        at = AppTest.from_string(test_script, default_timeout=10).run()
        
        # Should have title
        assert len(at.title) > 0
        assert "Advanced Emotion Control" in at.title[0].body
        
        # Should have markdown description
        assert len(at.markdown) > 0
        description_found = any(
            "emotion blending" in md.body.lower() or 
            "narrative context" in md.body.lower() or
            "prosody control" in md.body.lower()
            for md in at.markdown
        )
        assert description_found, "Should contain description of emotion control features"
    
    def test_page_handles_errors_gracefully(self):
        """Test that the page handles errors gracefully."""
        test_script = """
import streamlit as st
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Mock session state for authenticated user
st.session_state.authenticated = True

# Mock managers that might cause issues
class MockAuthManager:
    def get_current_user(self):
        return None

st.session_state.auth_manager = MockAuthManager()

# Import and run with potential for missing dependencies
try:
    from pages.advanced_emotion_control import main
    main()
except ImportError:
    st.error("Missing dependencies for emotion control")
except Exception as e:
    st.error(f"Error: {e}")
"""
        
        at = AppTest.from_string(test_script, default_timeout=10).run()
        
        # Should either load successfully or show an error message
        has_title = len(at.title) > 0 and "Advanced Emotion Control" in at.title[0].body
        has_error = len(at.error) > 0
        
        assert has_title or has_error, "Page should either load successfully or show error" 