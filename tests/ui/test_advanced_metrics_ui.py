"""
Tests for Advanced Metrics UI (R1-9)

Tests the UI enhancements for displaying advanced model metrics including:
- Enhanced training dashboard with personality alignment and lore adherence
- Model comparison personality drift analysis with radar charts
- Enhanced deep dive functionality with personality comparison
- Beautiful visualizations and user interactions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from streamlit.testing.v1 import AppTest
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Any

# These imports will be created during implementation
from app.utils.evaluation.lore_metric import LoreEvaluationResult
from app.components.personality_drift_analyzer import (
    PersonalityDriftAnalyzer,
    PersonalityDriftResult,
    render_personality_drift_chart
)


class MockSessionState:
    """Mock session state for testing"""
    def __init__(self, initial_state=None):
        self._state = initial_state or {}
    
    def __getattr__(self, name):
        return self._state.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._state[name] = value
    
    def get(self, key, default=None):
        return self._state.get(key, default)


class TestAdvancedMetricsUI(unittest.TestCase):
    """Test advanced metrics UI functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_character = {
            'name': 'Test Character',
            'personality_traits': {
                'openness': 0.8,
                'conscientiousness': 0.6,
                'extraversion': 0.4,
                'agreeableness': 0.7,
                'neuroticism': 0.3
            }
        }
        
        self.mock_training_metrics = {
            'current_loss': 1.2,
            'current_step': 100,
            'total_steps': 200,
            'avg_personality_alignment': 0.75,
            'avg_lore_adherence': 0.68,
            'character_consistency': 0.82,
            'training_health_status': 'healthy'
        }
        
        self.mock_model_list = [
            'Base: llama2-7b',
            'Checkpoint: test_character_step_50',
            'Checkpoint: test_character_step_100', 
            'Final: test_character_final'
        ]
    
    def test_training_dashboard_enhanced_metrics_display(self):
        """Test enhanced training dashboard displays new metrics"""
        # Mock session state with enhanced metrics
        mock_session_state = MockSessionState({
            'training_manager': Mock(),
            'training_status': 'training',
            'current_character_core': self.mock_character
        })
        
        # Mock training manager to return enhanced metrics
        mock_session_state.training_manager.get_metrics.return_value = self.mock_training_metrics
        mock_session_state.training_manager.get_training_status.return_value = 'training'
        
        # Create proper context manager mocks for columns
        def create_column_mock():
            """Create a mock that supports context manager protocol"""
            column_mock = Mock()
            column_mock.__enter__ = Mock(return_value=column_mock)
            column_mock.__exit__ = Mock(return_value=None)
            return column_mock
        
        with patch('streamlit.session_state', mock_session_state), \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.columns', return_value=[create_column_mock() for _ in range(4)]) as mock_columns, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.button', return_value=False) as mock_button, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.empty') as mock_empty, \
             patch('time.sleep') as mock_sleep:
            
            # Mock the expander context manager
            mock_expander_context = Mock()
            mock_expander_context.__enter__ = Mock(return_value=mock_expander_context)
            mock_expander_context.__exit__ = Mock(return_value=None)
            mock_expander.return_value = mock_expander_context
            
            # Mock the empty containers
            mock_container = Mock()
            mock_container.container.return_value.__enter__ = Mock(return_value=mock_container)
            mock_container.container.return_value.__exit__ = Mock(return_value=None)
            mock_empty.return_value = mock_container
            
            # Import and test the enhanced training dashboard
            from app.pages.training_dashboard import page_training_dashboard
            
            # This should run without errors and display enhanced metrics
            page_training_dashboard()
            
            # Verify that new metrics are displayed
            metric_calls = [call[0] if call[0] else [] for call in mock_metric.call_args_list]
            
            # Check that personality alignment and lore adherence metrics are shown
            personality_alignment_displayed = any(
                'Personality Alignment' in str(call) or 'avg_personality_alignment' in str(call)
                for call in metric_calls
            )
            lore_adherence_displayed = any(
                'Lore Adherence' in str(call) or 'avg_lore_adherence' in str(call)
                for call in metric_calls
            )
            
            # These should now pass since we've implemented the features
            self.assertTrue(personality_alignment_displayed, "Personality Alignment metric should be displayed")
            self.assertTrue(lore_adherence_displayed, "Lore Adherence metric should be displayed")


@dataclass
class PersonalityDriftResult:
    """Data class for personality drift analysis results (will be implemented)"""
    authored_personality: Dict[str, float]
    generated_personality: Dict[str, float]
    drift_magnitude: float
    samples_analyzed: int
    confidence_score: float


class PersonalityDriftAnalyzer:
    """Mock class for personality drift analyzer (will be implemented)"""
    def __init__(self, model_id: str, character: Dict[str, Any]):
        self.model_id = model_id
        self.character = character
    
    def analyze_personality_drift(self, num_samples: int = 50) -> PersonalityDriftResult:
        """Analyze personality drift between authored and generated personalities"""
        pass


def render_personality_drift_chart(drift_result: PersonalityDriftResult) -> go.Figure:
    """Render personality drift comparison chart (will be implemented)"""
    pass


if __name__ == '__main__':
    unittest.main() 