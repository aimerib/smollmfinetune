"""
UI tests for the Advanced Emotion Control component.
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, Mock
from datetime import datetime

# Add app directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))


@pytest.mark.ui
class TestAdvancedEmotionControlUI:
    """Tests for the Advanced Emotion Control UI component."""

    @patch('utils.openai_client.get_client')
    def test_emotion_control_page_loads(self, mock_get_client):
        """Test that the emotion control component loads without errors."""
        mock_get_client.return_value = Mock()
        
        test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.advanced_emotion_control import render_advanced_emotion_control

render_advanced_emotion_control()
"""
        
        at = AppTest.from_string(test_script, default_timeout=30).run()
        
        assert not at.exception
        assert len(at.title) > 0
        assert "Advanced Emotion Control" in at.title[0].value

    @patch('utils.openai_client.get_client')
    def test_emotion_control_tabs_present(self, mock_get_client):
        """Test that all required tabs are present."""
        mock_get_client.return_value = Mock()
        
        test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.advanced_emotion_control import render_advanced_emotion_control

render_advanced_emotion_control()
"""
        
        at = AppTest.from_string(test_script, default_timeout=30).run()
        
        assert not at.exception
        assert len(at.tabs) > 0
        
        # Check for expected tab labels
        tab_labels = [tab.label for tab in at.tabs]
        expected_tabs = ["🎭 Emotion Control", "📖 Narrative Context", "📈 Emotional Timeline", "🔧 Live Integration"]
        
        for expected_tab in expected_tabs:
            assert any(expected_tab in label for label in tab_labels)

    @patch('utils.openai_client.get_client')
    def test_emotion_control_sidebar_controls(self, mock_get_client):
        """Test that sidebar controls are present."""
        mock_get_client.return_value = Mock()
        
        test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.advanced_emotion_control import render_advanced_emotion_control

render_advanced_emotion_control()
"""
        
        at = AppTest.from_string(test_script, default_timeout=30).run()
        
        assert not at.exception
        
        # Check for configuration sliders
        slider_labels = [slider.label for slider in at.slider]
        expected_sliders = ["Adaptation Speed", "Memory Influence", "Story Influence", "Control Influence"]
        
        for expected_slider in expected_sliders:
            assert any(expected_slider in label for label in slider_labels)

    @patch('utils.openai_client.get_client')
    def test_emotion_blend_controls_render(self, mock_get_client):
        """Test that emotion blend controls render correctly."""
        mock_get_client.return_value = Mock()
        
        test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.advanced_emotion_control import render_emotion_blend_controls
from utils.narrative_context import EmotionBlendingService

service = EmotionBlendingService()
emotion_blend = render_emotion_blend_controls(service)
"""
        
        at = AppTest.from_string(test_script, default_timeout=30).run()
        
        assert not at.exception
        
        # Check for primary emotion selectbox
        selectbox_labels = [sb.label for sb in at.selectbox]
        assert any("Primary Emotion" in label for label in selectbox_labels)

    @patch('utils.openai_client.get_client')
    def test_narrative_context_form_render(self, mock_get_client):
        """Test that narrative context form renders correctly."""
        mock_get_client.return_value = Mock()
        
        test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.advanced_emotion_control import render_narrative_context_form

narrative_context = render_narrative_context_form()
"""
        
        at = AppTest.from_string(test_script, default_timeout=30).run()
        
        assert not at.exception
        
        # Check for narrative context controls
        selectbox_labels = [sb.label for sb in at.selectbox]
        expected_selectboxes = ["Character Arc Stage", "Scene Atmosphere", "Dialogue Context"]
        
        for expected_sb in expected_selectboxes:
            assert any(expected_sb in label for label in selectbox_labels)

    @patch('utils.openai_client.get_client')
    def test_prosody_controls_render(self, mock_get_client):
        """Test that prosody controls render correctly."""
        mock_get_client.return_value = Mock()
        
        test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.advanced_emotion_control import render_prosody_controls
from utils.narrative_context import EmotionBlend, NarrativeContext, EmotionBlendingService

emotion_blend = EmotionBlend(
    primary_emotion="joy",
    secondary_emotions={"excitement": 0.3},
    overall_intensity=0.8
)

narrative_context = NarrativeContext(
    narrative_tension=0.6,
    character_arc_stage="rising_action",
    scene_atmosphere="energetic",
    dialogue_context="celebration",
    primary_emotion="joy"
)

service = EmotionBlendingService()
prosody_control = render_prosody_controls(emotion_blend, narrative_context, service)
"""
        
        at = AppTest.from_string(test_script, default_timeout=30).run()
        
        assert not at.exception
        
        # Check for prosody sliders
        slider_labels = [slider.label for slider in at.slider]
        expected_sliders = ["Speaking Rate", "Pitch Variation", "Pause Duration"]
        
        for expected_slider in expected_sliders:
            assert any(expected_slider in label for label in slider_labels) 