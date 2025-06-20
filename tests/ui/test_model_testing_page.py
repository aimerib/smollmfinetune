"""
UI tests for Model Testing page using Streamlit AppTest framework.
Following the three-circle TDD approach - this is the outer circle (UI testing).
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import Mock
import sys
import os

# Add the app directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from utils.inference import InferenceManager


@pytest.fixture
def mock_inference_manager():
    """Mock InferenceManager for UI testing"""
    mock_im = Mock(spec=InferenceManager)
    mock_im.get_available_models.return_value = ["LoRA: TestCharacter", "Base: SmolLM2-360M"]
    mock_im.get_model_metadata.return_value = {
        'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'training_method': 'lora',
        'use_dora': False,
        'use_rslora': False
    }
    mock_im.base_model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    
    return mock_im


@pytest.fixture
def mock_dataset_manager():
    """Mock DatasetManager for UI testing"""
    mock_dm = Mock()
    mock_client = Mock()
    mock_client.chat_complete = Mock()
    mock_dm.client = mock_client
    
    return mock_dm


@pytest.fixture
def mock_character():
    """Mock character data for testing"""
    return {
        'name': 'TestCharacter',
        'description': 'A test character for model testing',
        'personality': 'Friendly and helpful'
    }


class TestModelTestingPageUI:
    """UI tests for Model Testing page"""
    
    def test_model_testing_page_loads_without_character(self):
        """Test that the Model Testing page shows warning when no character is loaded"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Set up session state without character
if 'current_character' not in st.session_state:
    st.session_state.current_character = None

# Mock other required managers
if 'inference_manager' not in st.session_state:
    mock_im = Mock()
    mock_im.get_available_models.return_value = []
    st.session_state.inference_manager = mock_im

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should show warning about missing character
        assert not at.exception
        assert len(at.warning) > 0  # Should have warning elements
    
    def test_model_testing_page_loads_with_character(self):
        """Test that the Model Testing page loads properly with character and models"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Set up session state with character
if 'current_character' not in st.session_state:
    st.session_state.current_character = {
        'name': 'TestCharacter',
        'description': 'A test character'
    }

# Mock inference manager with models
if 'inference_manager' not in st.session_state:
    mock_im = Mock()
    mock_im.get_available_models.return_value = ["LoRA: TestCharacter", "Base: SmolLM2-360M"]
    mock_im.get_model_metadata.return_value = {
        'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'training_method': 'lora'
    }
    mock_im.base_model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    st.session_state.inference_manager = mock_im

# Mock dataset manager
if 'dataset_manager' not in st.session_state:
    mock_dm = Mock()
    st.session_state.dataset_manager = mock_dm

# Mock dataset metadata
if 'dataset_metadata' not in st.session_state:
    st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should load without exceptions
        assert not at.exception
        
        # Should have UI elements
        assert len(at.markdown) > 0  # Should have markdown elements
        assert len(at.selectbox) > 0  # Should have model selection
    
    def test_model_selection_shows_available_models(self):
        """Test that available models are shown in the selectbox"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Set up session state
st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter", "Base: SmolLM2-360M"]
mock_im.get_model_metadata.return_value = {
    'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'training_method': 'lora'
}
mock_im.base_model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have a selectbox for model selection
        assert len(at.selectbox) > 0
        
        # Check if the selectbox has the correct options
        model_selectbox = at.selectbox[0]  # First selectbox should be model selection
        assert model_selectbox.options == ["LoRA: TestCharacter", "Base: SmolLM2-360M"]
    
    def test_no_models_available_message(self):
        """Test that appropriate message is shown when no models are available"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = []  # No models available
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        # Should have info message about no models
        assert len(at.info) > 0
    
    def test_system_prompt_options_displayed(self):
        """Test that system prompt strategy options are displayed"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {'base_model': 'test-model'}
mock_im.base_model = 'test-model'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have radio button for system prompt selection
        assert len(at.radio) > 0
        
        # Check that radio has appropriate options
        system_prompt_radio = at.radio[0]
        assert len(system_prompt_radio.options) > 0
        assert any("Default" in option for option in system_prompt_radio.options)
    
    def test_quick_test_buttons_present(self):
        """Test that quick test buttons are present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {'base_model': 'test-model'}
mock_im.base_model = 'test-model'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have multiple buttons (quick test buttons)
        assert len(at.button) > 6  # At least 6 quick test buttons plus generate button
        
        # Check for specific quick test button labels
        button_labels = [btn.label for btn in at.button]
        assert any("Who are you?" in label for label in button_labels)
    
    def test_generation_settings_expandable(self):
        """Test that generation settings are available in an expander"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {'base_model': 'test-model'}
mock_im.base_model = 'test-model'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have sliders for generation settings
        assert len(at.slider) > 0  # Should have temperature, top_p, etc.
    
    def test_text_input_and_area_present(self):
        """Test that test prompt text area is present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {'base_model': 'test-model'}
mock_im.base_model = 'test-model'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have text area for test prompt
        assert len(at.text_area) > 0


class TestModelTestingIntegration:
    """Integration tests for Model Testing UI with mocked external dependencies"""
    
    def test_model_testing_with_dataset_metadata(self):
        """Test that dataset metadata affects system prompt options"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {'base_model': 'test-model'}
mock_im.base_model = 'test-model'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()

# Set dataset metadata with custom system prompt
st.session_state.dataset_metadata = {
    'system_prompt_config': {
        'type': 'custom',
        'prompt': 'You are a test character'
    }
}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have radio with dataset system prompt option
        assert len(at.radio) > 0
        system_prompt_radio = at.radio[0]
        assert any("Dataset System Prompt" in option for option in system_prompt_radio.options)
    
    def test_model_metadata_display(self):
        """Test that model metadata is properly displayed"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {
    'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'training_method': 'lora',
    'use_dora': True,
    'use_rslora': False
}
mock_im.base_model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
st.session_state.inference_manager = mock_im

st.session_state.dataset_manager = Mock()
st.session_state.dataset_metadata = {}

from pages.model_testing import page_model_testing
page_model_testing()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have info messages about model metadata
        assert len(at.info) > 0 