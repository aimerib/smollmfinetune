"""
UI tests for Model Management page using Streamlit AppTest framework.
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

# Add the app directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from utils.inference import InferenceManager
from utils.training import TrainingManager


@pytest.fixture
def mock_inference_manager():
    """Mock InferenceManager for UI testing"""
    mock_im = Mock(spec=InferenceManager)
    mock_im.get_available_models.return_value = ["LoRA: TestCharacter", "Base: SmolLM2-360M"]
    mock_im.get_model_metadata.return_value = {
        'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'training_method': 'lora',
        'use_dora': False,
        'use_rslora': False,
        'training_date': '2025-06-19',
        'dataset_size': 100,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'total_steps': 1000
    }
    mock_im.get_model_metrics.return_value = {
        'current_loss': 0.5,
        'character_consistency': 0.85,
        'elapsed_time': 600  # 10 minutes
    }
    
    return mock_im


@pytest.fixture
def mock_training_manager():
    """Mock TrainingManager for UI testing"""
    mock_tm = Mock(spec=TrainingManager)
    mock_tm.get_available_checkpoints.return_value = ["checkpoint-100", "checkpoint-200"]
    mock_tm.merge_and_export_model.return_value = Path("/tmp/merged_model.safetensors")
    mock_tm.clear_training_assets.return_value = True
    mock_tm.export_lora.return_value = Path("/tmp/lora_export.zip")
    mock_tm.export_latest_checkpoint.return_value = Path("/tmp/checkpoint_export.zip")
    mock_tm.add_metadata_to_existing_model.return_value = True
    mock_tm.add_metadata_to_checkpoint.return_value = True
    
    return mock_tm


@pytest.fixture
def mock_character():
    """Mock character data for testing"""
    return {
        'name': 'TestCharacter',
        'description': 'A test character for model management',
        'personality': 'Friendly and helpful'
    }


class TestModelManagementPageUI:
    """UI tests for Model Management page"""
    
    def test_model_management_page_loads_without_character(self):
        """Test that the Model Management page shows warning when no character is loaded"""
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

if 'training_manager' not in st.session_state:
    st.session_state.training_manager = Mock()

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should show warning about missing character
        assert not at.exception
        assert len(at.warning) > 0  # Should have warning elements
    
    def test_model_management_page_loads_with_character(self):
        """Test that the Model Management page loads properly with character and models"""
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
        'training_method': 'lora',
        'training_date': '2025-06-19'
    }
    mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
    st.session_state.inference_manager = mock_im

# Mock training manager
if 'training_manager' not in st.session_state:
    mock_tm = Mock()
    mock_tm.get_available_checkpoints.return_value = []
    st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should load without exceptions
        assert not at.exception
        
        # Should have UI elements - tabs
        assert len(at.tabs) > 0  # Should have tab container
    
    def test_model_merging_tab_functionality(self):
        """Test that model merging tab shows appropriate UI elements"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {
    'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'use_dora': True,
    'training_date': '2025-06-19'
}
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
mock_tm.get_available_checkpoints.return_value = ["checkpoint-100"]
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have selectbox for model selection
        assert len(at.selectbox) > 0
        
        # Should have checkbox for checkpoint selection
        assert len(at.checkbox) > 0
        
        # Should have merge button
        button_labels = [btn.label for btn in at.button]
        assert any("Merge Model" in label for label in button_labels)
    
    def test_no_models_available_message(self):
        """Test that appropriate message is shown when no models are available"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["Base: SmolLM2-360M"]  # Only base models
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        # Should have info message about no models
        assert len(at.info) > 0
    
    def test_model_overview_tab_expanders(self):
        """Test that model overview tab shows expandable model information"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {
    'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'training_method': 'lora',
    'lora_r': 32
}
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have expanders for model details
        assert len(at.expander) > 0
    
    def test_model_assets_tab_buttons(self):
        """Test that model assets tab shows management buttons"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {'base_model': 'test-model'}
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have multiple buttons for asset management
        button_labels = [btn.label for btn in at.button]
        assert any("Clear All Training Assets" in label for label in button_labels)
        assert any("Export LoRA" in label for label in button_labels)
        assert any("Export Latest Checkpoint" in label for label in button_labels)
    
    def test_disk_usage_metrics_displayed(self):
        """Test that disk usage metrics are displayed"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have metrics for disk usage
        assert len(at.metric) > 0
    
    def test_legacy_model_fix_interface(self):
        """Test that legacy model fix interface is present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = None  # No metadata (legacy model)
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have selectboxes for legacy model fixing
        selectbox_options = []
        for sb in at.selectbox:
            if hasattr(sb, 'options') and sb.options:
                selectbox_options.extend(sb.options)
        
        # Should include base model options
        assert any("SmolLM2" in str(option) for option in selectbox_options)


class TestModelManagementIntegration:
    """Integration tests for Model Management UI with mocked external dependencies"""
    
    def test_model_management_with_dora_models(self):
        """Test that DoRA models show specific messaging"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
mock_im.get_available_models.return_value = ["LoRA: TestCharacter"]
mock_im.get_model_metadata.return_value = {
    'base_model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'use_dora': True,  # DoRA model
    'training_method': 'dora'
}
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
mock_tm.get_available_checkpoints.return_value = []
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have info messages about DoRA benefits
        assert len(at.info) > 0
    
    def test_checkpoint_metadata_fixing(self):
        """Test checkpoint metadata fixing functionality"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

st.session_state.current_character = {'name': 'TestCharacter'}

mock_im = Mock()
# Include checkpoints without metadata
mock_im.get_available_models.return_value = [
    "LoRA: TestCharacter", 
    "Checkpoint: TestCharacter/checkpoint-100"
]
# Return None for checkpoint (no metadata)
def get_metadata_side_effect(model):
    if "Checkpoint:" in model:
        return None
    return {'base_model': 'test-model', 'training_method': 'dora'}

mock_im.get_model_metadata.side_effect = get_metadata_side_effect
mock_im.get_model_metrics.return_value = {'current_loss': 0.5}
st.session_state.inference_manager = mock_im

mock_tm = Mock()
st.session_state.training_manager = mock_tm

from pages.model_management import page_model_management
page_model_management()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have warning about missing metadata
        assert len(at.warning) > 0
        
        # Should have button to fix metadata
        button_labels = [btn.label for btn in at.button]
        assert any("Fix All Checkpoint Metadata" in label for label in button_labels) 