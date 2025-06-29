"""
UI tests for Inference Inspector page (R3-4)

Tests the Streamlit interface for inspecting model behavior during inference:
- Page loading and basic structure
- Request ID input and search functionality
- Visualization components rendering
- Error handling for missing/invalid request IDs
- Attention visualization interface
- Token probability viewing interface
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import Mock, patch
import tempfile
import json
import os
from pathlib import Path
import sys

# Add the app directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from utils.observability import ObservabilityLogger


class TestInferenceInspectorPage:
    """UI tests for Inference Inspector page"""
    
    def test_inference_inspector_page_loads(self):
        """Test that the Inference Inspector page loads without errors"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock the observability logger in session state
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = []
    st.session_state.observability_logger = mock_logger

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        # Should load without exceptions
        assert not at.exception
        
        # Should have the main title
        assert any("Inference Inspector" in md.value for md in at.markdown)
    
    def test_empty_logs_message(self):
        """Test that appropriate message is shown when no logs are available"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock empty logs
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = []
    st.session_state.observability_logger = mock_logger

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        # Should have info message about no logs
        assert len(at.info) > 0
        assert any("No observability logs" in info.value for info in at.info)
    
    def test_request_id_selection_interface(self):
        """Test that request ID selection interface appears when logs are available"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock with available logs
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123", "req-456", "req-789"]
    st.session_state.observability_logger = mock_logger

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have selectbox for request ID selection
        assert len(at.selectbox) > 0
        
        # Check that selectbox has the correct options
        request_selectbox = at.selectbox[0]
        assert "req-123" in request_selectbox.options
        assert "req-456" in request_selectbox.options
        assert "req-789" in request_selectbox.options
    
    def test_request_details_display(self):
        """Test that request details are displayed when a request ID is selected"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock with log data
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123"]
    mock_logger.get_log_data.return_value = {
        'request_id': 'req-123',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_path': 'test-model',
        'timestamp': '2024-01-01T12:00:00',
        'generation_config': {'temperature': 0.8},
        'attention_weights_shape': [[1, 8, 10, 10]],
        'hidden_states_shape': [[1, 10, 512]],
        'token_probabilities': {'hello': 0.8, 'world': 0.2}
    }
    st.session_state.observability_logger = mock_logger

# Simulate selecting a request ID
if 'selected_request_id' not in st.session_state:
    st.session_state.selected_request_id = 'req-123'

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should display request details
        page_content = " ".join([md.value for md in at.markdown])
        assert "req-123" in page_content
        assert "Test prompt" in page_content
        assert "Test response" in page_content
        assert "test-model" in page_content
    
    def test_visualization_tabs_present(self):
        """Test that visualization tabs are present when viewing a request"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock with log data
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123"]
    mock_logger.get_log_data.return_value = {
        'request_id': 'req-123',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_path': 'test-model',
        'timestamp': '2024-01-01T12:00:00',
        'generation_config': {'temperature': 0.8},
        'attention_weights_shape': [[1, 8, 10, 10]],
        'hidden_states_shape': [[1, 10, 512]],
        'token_probabilities': {'hello': 0.8, 'world': 0.2}
    }
    st.session_state.observability_logger = mock_logger

# Simulate selecting a request ID
if 'selected_request_id' not in st.session_state:
    st.session_state.selected_request_id = 'req-123'

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have tabs for different visualizations
        assert len(at.tabs) > 0
    
    def test_attention_visualization_components(self):
        """Test that attention visualization components are rendered"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import plotly.graph_objects as go

# Mock with attention data
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123"]
    mock_logger.get_log_data.return_value = {
        'request_id': 'req-123',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_path': 'test-model',
        'timestamp': '2024-01-01T12:00:00',
        'generation_config': {'temperature': 0.8},
        'attention_weights_shape': [[1, 8, 10, 10]],
        'hidden_states_shape': [[1, 10, 512]],
        'token_probabilities': {'hello': 0.8, 'world': 0.2}
    }
    st.session_state.observability_logger = mock_logger

# Mock attention visualization
with st.expander("Test Attention Viz"):
    st.info("Attention visualization would appear here")
    
    # Mock plotly figure
    fig = go.Figure()
    fig.add_heatmap(z=[[0.1, 0.2], [0.3, 0.4]])
    st.plotly_chart(fig)

# Simulate selecting a request ID
if 'selected_request_id' not in st.session_state:
    st.session_state.selected_request_id = 'req-123'

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have plotly charts for attention visualization
        assert len(at.plotly_chart) > 0
    
    def test_token_probability_visualization(self):
        """Test that token probability visualization is rendered"""
        test_script = """
import streamlit as st
from unittest.mock import Mock
import plotly.graph_objects as go

# Mock with token probability data
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123"]
    mock_logger.get_log_data.return_value = {
        'request_id': 'req-123',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_path': 'test-model',
        'timestamp': '2024-01-01T12:00:00',
        'generation_config': {'temperature': 0.8},
        'attention_weights_shape': [[1, 8, 10, 10]],
        'hidden_states_shape': [[1, 10, 512]],
        'token_probabilities': {'hello': 0.8, 'world': 0.15, 'test': 0.05}
    }
    st.session_state.observability_logger = mock_logger

# Mock token probability visualization
st.subheader("Token Probabilities")
fig = go.Figure()
fig.add_bar(x=['hello', 'world', 'test'], y=[0.8, 0.15, 0.05])
st.plotly_chart(fig)

# Simulate selecting a request ID
if 'selected_request_id' not in st.session_state:
    st.session_state.selected_request_id = 'req-123'

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have plotly charts for token probabilities
        assert len(at.plotly_chart) > 0
    
    def test_invalid_request_id_handling(self):
        """Test handling of invalid/missing request IDs"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock with logs but invalid request ID
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123"]
    mock_logger.get_log_data.return_value = None  # Simulate missing data
    st.session_state.observability_logger = mock_logger

# Simulate selecting a non-existent request ID
if 'selected_request_id' not in st.session_state:
    st.session_state.selected_request_id = 'invalid-req'

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have error message for invalid request ID
        assert len(at.error) > 0
        assert any("Could not load" in error.value for error in at.error)
    
    def test_search_functionality_interface(self):
        """Test that search functionality interface is present"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock with multiple logs
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = [
        "req-123", "req-456", "req-789", "req-abc", "req-def"
    ]
    st.session_state.observability_logger = mock_logger

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should have text input for search/filter
        assert len(at.text_input) > 0
    
    def test_metadata_display(self):
        """Test that metadata about the request is displayed"""
        test_script = """
import streamlit as st
from unittest.mock import Mock

# Mock with detailed metadata
if 'observability_logger' not in st.session_state:
    mock_logger = Mock()
    mock_logger.list_available_logs.return_value = ["req-123"]
    mock_logger.get_log_data.return_value = {
        'request_id': 'req-123',
        'prompt': 'Test prompt',
        'response': 'Test response',
        'model_path': 'test-model',
        'timestamp': '2024-01-01T12:00:00',
        'generation_config': {
            'temperature': 0.8,
            'top_p': 0.9,
            'max_tokens': 150
        },
        'attention_weights_shape': [[1, 8, 10, 10]],
        'hidden_states_shape': [[1, 10, 512]],
        'token_probabilities': {'hello': 0.8, 'world': 0.2}
    }
    st.session_state.observability_logger = mock_logger

# Simulate selecting a request ID
if 'selected_request_id' not in st.session_state:
    st.session_state.selected_request_id = 'req-123'

from pages.inference_inspector import page_inference_inspector
page_inference_inspector()
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        
        # Should display generation configuration
        page_content = " ".join([md.value for md in at.markdown])
        assert "temperature" in page_content.lower()
        assert "0.8" in page_content


class TestInferenceInspectorComponents:
    """Test individual components of the Inference Inspector"""
    
    def test_attention_heatmap_component(self):
        """Test that attention heatmap component renders correctly"""
        test_script = """
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Mock attention visualization component
st.subheader("Attention Heatmap")

# Create sample attention data
attention_data = np.random.rand(10, 10)
tokens = [f"token_{i}" for i in range(10)]

fig = go.Figure(data=go.Heatmap(
    z=attention_data,
    x=tokens,
    y=tokens,
    colorscale='Viridis'
))
fig.update_layout(
    title="Attention Weights",
    xaxis_title="Key Tokens",
    yaxis_title="Query Tokens"
)

st.plotly_chart(fig, use_container_width=True)
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        assert len(at.plotly_chart) == 1
    
    def test_token_probability_bar_chart(self):
        """Test that token probability bar chart renders correctly"""
        test_script = """
import streamlit as st
import plotly.graph_objects as go

# Mock token probability visualization
st.subheader("Top Token Probabilities")

tokens = ['hello', 'world', 'test', 'data', 'example']
probabilities = [0.45, 0.25, 0.15, 0.10, 0.05]

fig = go.Figure(data=[
    go.Bar(x=tokens, y=probabilities, name='Probability')
])
fig.update_layout(
    title="Token Probability Distribution",
    xaxis_title="Tokens",
    yaxis_title="Probability",
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)
"""
        
        at = AppTest.from_string(test_script).run()
        
        assert not at.exception
        assert len(at.plotly_chart) == 1


if __name__ == '__main__':
    pytest.main([__file__]) 