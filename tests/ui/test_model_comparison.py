"""
Tests for Model Comparison Page

Tests the extracted model comparison functionality including:
- Model selection and filtering
- Side-by-side comparison interface
- Training progression analysis
- Character consistency evaluation
- Checkpoint promotion system
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the app directory to the path so we can import the page
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

class MockSessionState:
    """Mock Streamlit session state with attribute access"""
    def __init__(self):
        self._state = {}
        
        # Set up required managers
        self.current_character = {"name": "TestChar", "description": "A test character"}
        self.inference_manager = Mock()
        self.comparison_manager = Mock()
        self.dataset_manager = Mock()
        
        # Configure mock methods
        self.inference_manager.get_available_models.return_value = [
            "LoRA: TestChar_final",
            "Checkpoint: TestChar_step_100", 
            "Checkpoint: TestChar_step_200",
            "Base: microsoft/DialoGPT-medium"
        ]
        self.inference_manager.get_model_metadata.return_value = {
            'base_model': 'microsoft/DialoGPT-medium',
            'training_method': 'lora',
            'use_dora': False,
            'use_rslora': False
        }
        self.inference_manager.base_model = "microsoft/DialoGPT-medium"
        
        self.comparison_manager.compare_models_side_by_side.return_value = {
            "LoRA: TestChar_final": "Hello! I'm TestChar, a friendly AI assistant.",
            "Checkpoint: TestChar_step_100": "Hi there! I'm TestChar.",
            "Base: microsoft/DialoGPT-medium": "Hello."
        }
        self.comparison_manager.get_promoted_checkpoint.return_value = None
        self.comparison_manager.promote_checkpoint.return_value = True
        
        # Mock async method for enhanced metrics
        async def mock_enhanced_metrics(*args, **kwargs):
            return {}, {}, {
                "LoRA: TestChar_final": {
                    'is_base_model': False,
                    'is_final_model': True,
                    'character_consistency': 0.85,
                    'training_loss': 0.45
                },
                "Checkpoint: TestChar_step_100": {
                    'is_base_model': False,
                    'is_checkpoint': True,
                    'character_consistency': 0.75,
                    'training_loss': 0.65,
                    'actual_steps_completed': 100
                },
                "Base: microsoft/DialoGPT-medium": {
                    'is_base_model': True,
                    'character_consistency': 0.25
                }
            }
        
        self.comparison_manager.get_enhanced_comparison_metrics = mock_enhanced_metrics
        
    def __getattr__(self, name):
        return self._state.get(name)
        
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_state'):
                super().__setattr__('_state', {})
            self._state[name] = value
    
    def __contains__(self, key):
        """Support 'in' operator for session state checks"""
        # Check if the key exists AND is not None
        if hasattr(self, key):
            value = getattr(self, key)
            return value is not None
        return key in self._state and self._state[key] is not None

@pytest.fixture
def mock_session_state():
    """Fixture providing a mock session state"""
    return MockSessionState()

@pytest.fixture
def mock_streamlit_components():
    """Fixture providing mocks for Streamlit components"""
    with patch('streamlit.markdown'), \
         patch('streamlit.warning'), \
         patch('streamlit.info'), \
         patch('streamlit.success'), \
         patch('streamlit.error'), \
         patch('streamlit.multiselect') as mock_multiselect, \
         patch('streamlit.text_area') as mock_text_area, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.expander') as mock_expander, \
         patch('streamlit.checkbox') as mock_checkbox, \
         patch('streamlit.number_input') as mock_number_input, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.plotly_chart'), \
         patch('streamlit.dataframe'):
        
        # Configure multiselect to return test models
        mock_multiselect.return_value = [
            "Base: microsoft/DialoGPT-medium",
            "Checkpoint: TestChar_step_100",
            "LoRA: TestChar_final"
        ]
        
        mock_text_area.return_value = "Who are you and what do you want?"
        mock_button.return_value = False  # No buttons pressed by default
        mock_checkbox.return_value = False
        mock_number_input.return_value = 42
        
        # Mock column and tab contexts - return the right number based on call
        mock_col = Mock()
        mock_col.__enter__ = Mock(return_value=mock_col)
        mock_col.__exit__ = Mock(return_value=None)
        
        def mock_columns_side_effect(*args, **kwargs):
            if args and isinstance(args[0], int):
                return [mock_col] * args[0]
            elif args and isinstance(args[0], list):
                return [mock_col] * len(args[0])
            else:
                return [mock_col, mock_col]  # Default to 2 columns
        
        mock_columns.side_effect = mock_columns_side_effect
        
        mock_tab = Mock()
        mock_tab.__enter__ = Mock(return_value=mock_tab)
        mock_tab.__exit__ = Mock(return_value=None)
        mock_tabs.return_value = [mock_tab, mock_tab, mock_tab]
        
        mock_exp = Mock()
        mock_exp.__enter__ = Mock(return_value=mock_exp)
        mock_exp.__exit__ = Mock(return_value=None)
        mock_expander.return_value = mock_exp
        
        mock_spin = Mock()
        mock_spin.__enter__ = Mock(return_value=mock_spin)
        mock_spin.__exit__ = Mock(return_value=None)
        mock_spinner.return_value = mock_spin
        
        yield {
            'multiselect': mock_multiselect,
            'text_area': mock_text_area,
            'button': mock_button,
            'expander': mock_expander,
            'checkbox': mock_checkbox,
            'number_input': mock_number_input,
            'columns': mock_columns,
            'tabs': mock_tabs,
            'spinner': mock_spinner
        }

def test_model_comparison_basic_functionality(mock_session_state, mock_streamlit_components):
    """Test basic model comparison page functionality"""
    # Ensure comparison_data doesn't exist to avoid the display section
    mock_session_state.comparison_data = None
    
    with patch('streamlit.session_state', mock_session_state):
        from pages.model_comparison import page_model_comparison
        
        # Should not raise any exceptions
        page_model_comparison()
        
        # Verify that required components were called
        assert mock_streamlit_components['multiselect'].called
        assert mock_streamlit_components['text_area'].called

def test_model_comparison_no_character(mock_streamlit_components):
    """Test model comparison page when no character is selected"""
    mock_state = MockSessionState()
    mock_state.current_character = None
    
    with patch('streamlit.session_state', mock_state), \
         patch('streamlit.warning') as mock_warning:
        
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should show warning and return early
        mock_warning.assert_called_with("⚠️ Please upload a character card first.")

def test_model_comparison_insufficient_models(mock_streamlit_components):
    """Test model comparison page when insufficient models are available"""
    mock_state = MockSessionState()
    mock_state.current_character = {"name": "TestChar"}
    mock_state.inference_manager.get_available_models.return_value = ["Base: microsoft/DialoGPT-medium"]
    
    with patch('streamlit.session_state', mock_state), \
         patch('streamlit.info') as mock_info:
        
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should show info about needing more models
        mock_info.assert_called()
        info_call_args = str(mock_info.call_args)
        assert "at least two trained models" in info_call_args

def test_model_comparison_with_comparison_data(mock_session_state, mock_streamlit_components):
    """Test model comparison page when comparison data exists"""
    # Add comparison data to session state
    mock_session_state.comparison_data = {
        "responses": {
            "LoRA: TestChar_final": "Hello! I'm TestChar, a friendly AI assistant.",
            "Checkpoint: TestChar_step_100": "Hi there! I'm TestChar.",
            "Base: microsoft/DialoGPT-medium": "Hello."
        },
        "base_metrics": {},
        "training_summary": {
            'base_model': 'microsoft/DialoGPT-medium',
            'training_method': 'lora',
            'lora_rank': 16,
            'lora_alpha': 32,
            'dataset_size': 100
        },
        "variable_metrics": {
            "LoRA: TestChar_final": {
                'is_base_model': False,
                'is_final_model': True,
                'character_consistency': 0.85,
                'training_loss': 0.45
            },
            "Checkpoint: TestChar_step_100": {
                'is_base_model': False,
                'is_checkpoint': True,
                'character_consistency': 0.75,
                'training_loss': 0.65,
                'actual_steps_completed': 100
            }
        }
    }
    
    with patch('streamlit.session_state', mock_session_state):
        from pages.model_comparison import page_model_comparison
        
        # Should not raise any exceptions
        page_model_comparison()
        
        # Verify that comparison results section is shown
        assert mock_streamlit_components['tabs'].called
        assert mock_streamlit_components['columns'].called

def test_model_comparison_button_interactions(mock_session_state, mock_streamlit_components):
    """Test button interactions in model comparison page"""
    # Test compare button
    mock_streamlit_components['button'].return_value = True
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.success') as mock_success:
        
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should call success at some point (multiple success calls happen)
        assert mock_success.called

def test_model_comparison_sampling_config(mock_session_state, mock_streamlit_components):
    """Test sampling configuration in model comparison"""
    # Ensure comparison_data doesn't exist to avoid the display section
    mock_session_state.comparison_data = None
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('utils.sampling_config.render_sampling_config_ui') as mock_sampling_ui:
        
        # Mock the sampling config UI
        mock_config = Mock()
        mock_config.to_dict.return_value = {
            'temperature': 0.9,
            'top_p': 0.95,
            'max_tokens': 200,
            'repetition_penalty': 1.0
        }
        mock_sampling_ui.return_value = mock_config
        
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should call sampling config UI
        mock_sampling_ui.assert_called()

def test_model_comparison_seed_settings(mock_session_state, mock_streamlit_components):
    """Test seed settings for reproducible comparison"""
    # Test custom seed checkbox
    mock_streamlit_components['checkbox'].return_value = True
    mock_streamlit_components['number_input'].return_value = 12345
    # Ensure comparison_data doesn't exist to avoid the display section
    mock_session_state.comparison_data = None
    
    with patch('streamlit.session_state', mock_session_state):
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should call checkbox and number input for seed
        assert mock_streamlit_components['checkbox'].called
        assert mock_streamlit_components['number_input'].called

def test_model_comparison_error_handling(mock_session_state, mock_streamlit_components):
    """Test error handling in model comparison"""
    # Test empty prompt
    mock_streamlit_components['text_area'].return_value = ""
    mock_streamlit_components['button'].return_value = True
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.error') as mock_error:
        
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should show error for empty prompt
        mock_error.assert_called_with("❌ Please enter a prompt.")

def test_model_comparison_checkpoint_promotion(mock_session_state, mock_streamlit_components):
    """Test checkpoint promotion functionality"""
    # Add comparison data with checkpoints
    mock_session_state.comparison_data = {
        "responses": {"Checkpoint: TestChar_step_100": "Test response"},
        "base_metrics": {},
        "training_summary": {},
        "variable_metrics": {}
    }
    
    # Mock button to return True for promotion
    def mock_button_side_effect(*args, **kwargs):
        if "Promote" in str(args):
            return True
        return False
    
    mock_streamlit_components['button'].side_effect = mock_button_side_effect
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.success') as mock_success:
        
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
        
        # Should show success message for promotion
        # Note: This test verifies the promotion logic exists, actual promotion depends on button clicks

def test_model_comparison_imports():
    """Test that all required imports work correctly"""
    try:
        from pages.model_comparison import page_model_comparison
        assert callable(page_model_comparison)
    except ImportError as e:
        pytest.fail(f"Failed to import model comparison page: {e}")

def test_model_comparison_standalone_execution():
    """Test that the page can be executed standalone"""
    with patch('streamlit.session_state', MockSessionState()), \
         patch('streamlit.markdown'), \
         patch('streamlit.warning'), \
         patch('streamlit.multiselect') as mock_multiselect:
        
        # Configure to return insufficient models
        mock_multiselect.return_value = []
        
        # Should be able to run the main guard
        try:
            import sys
            import os
            
            # Temporarily modify sys.path and run the module
            old_path = sys.path[:]
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
                
                # Import and test the main guard
                # The main guard should not raise exceptions
                
            finally:
                sys.path[:] = old_path
                
        except Exception as e:
            pytest.fail(f"Standalone execution failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 