"""
Tests for the Training Dashboard page.

Validates that the extracted training dashboard page maintains all functionality
including real-time monitoring, training controls, and chart rendering.
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
    
    def get(self, key, default=None):
        return super().get(key, default)

class MockContextManager:
    """Mock context manager for streamlit components"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def container(self):
        return MockContextManager()
    def columns(self, n):
        return [MockContextManager() for _ in range(n)]
    def expander(self, title, expanded=False):
        return MockContextManager()
    def popover(self, label, help=None):
        return MockContextManager()

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
    mock.button = Mock(return_value=False)
    mock.metric = Mock()
    mock.progress = Mock()
    mock.plotly_chart = Mock()
    mock.code = Mock()
    mock.write = Mock()
    mock.caption = Mock()
    
    # Mock containers and layout - FIXED: columns should return tuple-like objects
    mock.empty = Mock(return_value=MockContextManager())
    
    def mock_columns(spec):
        """Mock columns function that returns unpacked context managers"""
        if isinstance(spec, int):
            return tuple(MockContextManager() for _ in range(spec))
        elif isinstance(spec, list):
            return tuple(MockContextManager() for _ in range(len(spec)))
        else:
            return tuple(MockContextManager() for _ in range(2))  # default
    
    mock.columns = Mock(side_effect=mock_columns)
    mock.expander = Mock(return_value=MockContextManager())
    mock.popover = Mock(return_value=MockContextManager())
    
    return mock

def create_dynamic_columns_mock():
    """Create a mock for st.columns that returns the correct number of context managers"""
    def mock_columns(spec):
        if isinstance(spec, int):
            num_cols = spec
        elif isinstance(spec, list):
            num_cols = len(spec)
        else:
            num_cols = 2  # Default fallback
        
        return [MockContextManager() for _ in range(num_cols)]
    
    return mock_columns

def test_training_dashboard_page_initialization():
    """Test that the training dashboard page initializes correctly with required managers."""
    
    # Create mock session state with required components
    mock_session_state = MockSessionState({
        'training_manager': Mock(),
        'training_status': 'idle'
    })
    
    # Configure training manager
    mock_session_state.training_manager.get_metrics.return_value = {}
    mock_session_state.training_manager.get_training_status.return_value = 'idle'
    
    # Create streamlit component mocks
    with patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.info') as mock_info, \
         patch('streamlit.session_state', mock_session_state):
        
        from pages.training_dashboard import page_training_dashboard
        
        # Should run without errors
        page_training_dashboard()
        
        # Verify basic UI elements were called
        assert mock_markdown.called, "Expected 'markdown' to have been called"
        assert mock_info.called, "Expected 'info' to have been called"

def test_training_dashboard_with_active_training():
    """Test training dashboard with active training session."""
    
    # Mock session state with active training
    mock_training_manager = Mock()
    mock_metrics = {
        'current_loss': 1.5,
        'loss_delta': -0.1,
        'current_step': 50,
        'total_steps': 100,
        'learning_rate': 2e-4,
        'elapsed_time': 300,
        'training_health_status': 'healthy',
        'health_warnings': [],
        'loss_history': [2.0, 1.8, 1.6, 1.5]
    }
    
    mock_session_state = MockSessionState({
        'training_manager': mock_training_manager,
        'training_status': 'training',
        'advanced_training_config': {
            'enable_wandb': False,
            'enable_tensorboard': False
        },
        'active_training_config': {
            'logging_steps': 10,
            'learning_rate': 2e-4,
            'finetune_method': 'lora',
            'lora_r': 16,
            'batch_size': 2,
            'save_steps': 50,
            'max_samples': 'All',
            'fp16': False
        }
    })
    
    # Configure manager methods
    mock_training_manager.get_metrics.return_value = mock_metrics
    mock_training_manager.get_training_status.return_value = 'training'
    mock_training_manager.pause_training = Mock()
    mock_training_manager.resume_training = Mock()
    mock_training_manager.stop_training = Mock()
    
    # Mock pandas DataFrame for charts
    mock_pandas = Mock()
    mock_df = Mock()
    mock_pandas.DataFrame.return_value = mock_df
    
    # Mock plotly
    mock_px = Mock()
    mock_fig = Mock()
    mock_px.line.return_value = mock_fig
    mock_fig.update_layout = Mock()
    mock_fig.add_scatter = Mock()
    mock_fig.add_vline = Mock()
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.button', return_value=False) as mock_button, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.columns', side_effect=create_dynamic_columns_mock()), \
         patch('streamlit.plotly_chart') as mock_plotly_chart, \
         patch('pages.training_dashboard.pd', mock_pandas), \
         patch('pages.training_dashboard.px', mock_px), \
         patch('pages.training_dashboard.time'):
        
        from pages.training_dashboard import page_training_dashboard
        
        # Should run without errors
        page_training_dashboard()
        
        # Verify training controls were rendered
        assert mock_button.call_count >= 4  # Pause, Resume, Test, Stop buttons
        
        # Verify metrics were displayed
        mock_metric.assert_called()
        
        # Verify chart was created
        mock_pandas.DataFrame.assert_called()
        mock_px.line.assert_called()

def test_training_dashboard_with_validation_metrics():
    """Test training dashboard with validation metrics."""
    
    # Mock session state with validation metrics
    mock_training_manager = Mock()
    mock_metrics = {
        'current_loss': 1.2,
        'loss_delta': -0.05,
        'current_step': 150,
        'total_steps': 200,
        'learning_rate': 1.8e-4,
        'elapsed_time': 900,
        'training_health_status': 'healthy',
        'health_warnings': [],
        'loss_history': [2.0, 1.8, 1.6, 1.4, 1.2],
        'validation_loss': 1.3,
        'validation_loss_history': [2.1, 1.9, 1.7, 1.5, 1.3],
        'validation_perplexity': 3.67,
        'validation_perplexity_history': [8.2, 6.7, 5.4, 4.5, 3.67]
    }
    
    mock_session_state = MockSessionState({
        'training_manager': mock_training_manager,
        'training_status': 'training',
        'advanced_training_config': {
            'enable_wandb': False,
            'enable_tensorboard': True
        },
        'active_training_config': {
            'logging_steps': 10,
            'learning_rate': 2e-4,
            'finetune_method': 'lora',
            'lora_r': 16,
            'batch_size': 2,
            'save_steps': 50,
            'max_samples': 'All',
            'fp16': False,
            'eval_steps': 25
        }
    })
    
    # Configure manager methods
    mock_training_manager.get_metrics.return_value = mock_metrics
    mock_training_manager.get_training_status.return_value = 'training'
    mock_training_manager.pause_training = Mock()
    mock_training_manager.resume_training = Mock()
    mock_training_manager.stop_training = Mock()
    
    # Mock pandas and plotly
    mock_pandas = Mock()
    mock_df = Mock()
    mock_pandas.DataFrame.return_value = mock_df
    
    mock_px = Mock()
    mock_fig = Mock()
    mock_px.line.return_value = mock_fig
    mock_fig.update_layout = Mock()
    mock_fig.add_scatter = Mock()
    mock_fig.add_vline = Mock()
    
    mock_go = Mock()
    mock_go_fig = Mock()
    mock_go.Figure.return_value = mock_go_fig
    mock_go_fig.add_trace = Mock()
    mock_go_fig.update_layout = Mock()
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.columns', side_effect=create_dynamic_columns_mock()), \
         patch('streamlit.plotly_chart') as mock_plotly_chart, \
         patch('pages.training_dashboard.pd', mock_pandas), \
         patch('pages.training_dashboard.px', mock_px), \
         patch('pages.training_dashboard.go', mock_go), \
         patch('pages.training_dashboard.time'):
        
        from pages.training_dashboard import page_training_dashboard
        
        # Should run without errors
        page_training_dashboard()
        
        # Verify validation metrics were displayed
        mock_metric.assert_called()
        
        # Verify charts were created
        mock_pandas.DataFrame.assert_called()

def test_training_dashboard_health_warnings():
    """Test training dashboard with health warnings."""
    
    # Mock session state with health warnings
    mock_training_manager = Mock()
    mock_metrics = {
        'current_loss': 0.8,
        'loss_delta': 0.02,  # Positive delta indicates increasing loss
        'current_step': 200,
        'total_steps': 300,
        'learning_rate': 2e-4,
        'elapsed_time': 1200,
        'training_health_status': 'warning',
        'health_warnings': ['Loss has been increasing for 10 consecutive steps', 'Learning rate might be too high'],
        'loss_history': [1.0, 0.9, 0.85, 0.82, 0.8]
    }
    
    mock_session_state = MockSessionState({
        'training_manager': mock_training_manager,
        'training_status': 'training',
        'advanced_training_config': {
            'enable_wandb': False,
            'enable_tensorboard': False
        },
        'active_training_config': {
            'logging_steps': 10,
            'learning_rate': 2e-4,
            'finetune_method': 'lora',
            'lora_r': 16,
            'batch_size': 2,
            'save_steps': 50,
            'max_samples': 'All',
            'fp16': False
        }
    })
    
    # Configure manager methods
    mock_training_manager.get_metrics.return_value = mock_metrics
    mock_training_manager.get_training_status.return_value = 'training'
    
    # Mock pandas and plotly
    mock_pandas = Mock()
    mock_df = Mock()
    mock_pandas.DataFrame.return_value = mock_df
    
    mock_px = Mock()
    mock_fig = Mock()
    mock_px.line.return_value = mock_fig
    mock_fig.update_layout = Mock()
    
    with patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.columns', side_effect=create_dynamic_columns_mock()), \
         patch('streamlit.plotly_chart') as mock_plotly_chart, \
         patch('pages.training_dashboard.pd', mock_pandas), \
         patch('pages.training_dashboard.px', mock_px), \
         patch('pages.training_dashboard.time'):
        
        from pages.training_dashboard import page_training_dashboard
        
        # Should run without errors
        page_training_dashboard()
        
        # Verify warning messages were displayed
        mock_warning.assert_called()

def test_render_consistency_deep_dive():
    """Test the render_consistency_deep_dive function."""
    
    with patch('streamlit.expander') as mock_expander:
        # Configure expander to return a context manager
        mock_exp_ctx = MockContextManager()
        mock_expander.return_value = mock_exp_ctx
        
        from pages.training_dashboard import render_consistency_deep_dive
        
        # Test with sample metrics
        metrics = {
            'loss_history': [2.0, 1.8, 1.6, 1.4, 1.2, 1.0],
            'validation_loss_history': [2.1, 1.9, 1.7, 1.5, 1.3, 1.1],
            'validation_perplexity_history': [8.2, 6.7, 5.4, 4.5, 3.67, 2.8]
        }
        
        # Should run without errors
        render_consistency_deep_dive(metrics)
        
        # Verify expander was called
        assert mock_expander.called, "Expected 'expander' to have been called"

def test_render_healthy_run_example():
    """Test the render_healthy_run_example function."""
    
    with patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.code') as mock_code:
        
        from pages.training_dashboard import render_healthy_run_example
        
        # Should run without errors
        render_healthy_run_example()
        
        # Verify markdown was called to render the example
        assert mock_markdown.called, "Expected 'markdown' to have been called"

if __name__ == "__main__":
    pytest.main([__file__]) 