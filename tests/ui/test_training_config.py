"""
Tests for the Training Configuration page.

Validates that the extracted training config page maintains all functionality
including form handling, profile management, and training configuration validation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the app directory to Python path so we can import from pages
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))

class MockSessionState(dict):
    """Mock session state that supports both dict access and attribute access"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None
    
    def __setattr__(self, name, value):
        self[name] = value

class MockContextManager:
    """Mock context manager for streamlit elements like columns, expanders, etc."""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

def create_comprehensive_streamlit_mock():
    """Create a comprehensive streamlit mock with all necessary methods"""
    mock = Mock()
    
    # Add all the streamlit methods that need to be mocked
    mock.markdown = Mock()
    mock.error = Mock()
    mock.warning = Mock()
    mock.info = Mock()
    mock.success = Mock()
    mock.toast = Mock()
    mock.rerun = Mock()
    mock.checkbox = Mock(return_value=True)
    mock.selectbox = Mock(return_value="HuggingFaceTB/SmolLM2-135M-Instruct")
    mock.radio = Mock(return_value="lora")
    mock.slider = Mock(return_value=5)
    mock.select_slider = Mock(return_value=2e-4)
    mock.multiselect = Mock(return_value=["q_proj", "k_proj", "v_proj", "o_proj"])
    mock.number_input = Mock(return_value=0)
    mock.form_submit_button = Mock(return_value=False)
    mock.file_uploader = Mock(return_value=None)
    mock.button = Mock(return_value=False)
    mock.text_input = Mock(return_value="")
    mock.text_area = Mock(return_value="")
    mock.metric = Mock()
    mock.progress = Mock()
    mock.popover = Mock(return_value=MockContextManager())
    mock.tabs = Mock(return_value=[MockContextManager(), MockContextManager()])
    mock.plotly_chart = Mock()
    
    return mock

def test_training_config_page_initialization():
    """Test that the training config page initializes correctly with required managers."""
    
    # Setup mock session state - start with empty state to test error handling
    mock_session_state = MockSessionState()
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.error') as mock_error:
        
        from pages.training_config import page_training_config
        
        # Should handle missing managers gracefully
        page_training_config()
        
        # Verify error message was called for missing managers
        assert mock_error.called, "Expected 'error' to have been called for missing managers"

def test_training_config_with_valid_setup():
    """Test training config page with all required components set up."""
    
    # Mock session state with required components
    mock_training_manager = Mock()
    mock_inference_manager = Mock()
    mock_character = {"name": "Test Character"}
    mock_dataset = [{"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]}]
    
    # Configure training manager methods to return proper types
    mock_training_manager.get_available_checkpoints.return_value = []  # Return empty list instead of Mock
    mock_training_manager.base_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    mock_training_manager.set_base_model = Mock()
    
    # Configure inference manager methods
    mock_inference_manager.set_base_model = Mock()
    
    mock_session_state = MockSessionState({
        'training_manager': mock_training_manager,
        'inference_manager': mock_inference_manager,
        'current_character': mock_character,
        'dataset_preview': mock_dataset,
        'training_status': 'idle'
    })
    
    # Mock context managers
    mock_col1 = MockContextManager()
    mock_col2 = MockContextManager()
    mock_expander = MockContextManager()
    mock_form = MockContextManager()
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.columns', return_value=[mock_col1, mock_col2]), \
         patch('streamlit.expander', return_value=mock_expander), \
         patch('streamlit.form', return_value=mock_form), \
         patch('streamlit.slider', side_effect=lambda label, *args, **kwargs: {
             "Epochs": 5,
             "Warmup Steps": 10,
             "Max Gradient Norm": 1.0,
             "Number of Samples": 100,
             "Rank (r)": 16,
             "Alpha": 16,
             "Dropout": 0.1,
             "Save Every N Steps": 50,
             "Log Every N Steps": 5,
             "Evaluation Steps": 10,
             "Early Stopping Patience": 3
         }.get(label, 5)), \
         patch('streamlit.selectbox', side_effect=lambda label, options, *args, **kwargs: {
             "Select base model": "HuggingFaceTB/SmolLM2-135M-Instruct",
             "Batch Size": 2,
             "Gradient Accumulation Steps": 2
         }.get(label, options[0] if options else "default")), \
         patch('streamlit.select_slider', return_value=2e-4), \
         patch('streamlit.number_input', return_value=0), \
         patch('streamlit.checkbox', return_value=True), \
         patch('streamlit.radio', return_value="LoRA"), \
         patch('streamlit.text_input', return_value=""), \
         patch('streamlit.multiselect', return_value=["q_proj", "k_proj", "v_proj", "o_proj"]), \
         patch('streamlit.form_submit_button', return_value=False), \
         patch('streamlit.toast'):
        
        from pages.training_config import page_training_config
        
        # Should run without errors and display the main UI
        page_training_config()
        
        # Verify main UI elements were rendered
        assert mock_markdown.called, "Expected 'markdown' to have been called for main UI rendering"

def test_training_config_imports_successfully():
    """Test that the training config module can be imported without errors."""
    
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        # Should be able to import the module
        from pages.training_config import page_training_config
        
        # Function should exist and be callable
        assert callable(page_training_config)

def test_training_config_function_exists():
    """Test that the page_training_config function exists and has proper docstring."""
    
    with patch.dict('sys.modules', {'streamlit': Mock()}):
        from pages.training_config import page_training_config
        
        # Function should exist
        assert page_training_config is not None
        
        # Should have a docstring
        assert page_training_config.__doc__ is not None
        assert "Enhanced training configuration page" in page_training_config.__doc__

if __name__ == "__main__":
    pytest.main([__file__]) 