"""
Unit tests for the Narrative Context Service.
"""

import pytest
from unittest.mock import patch, Mock
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Add app directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))

# Delayed import to allow for patching
# from backend.app.core.narrative_context import NarrativeContext, NarrativeContextService


class TestNarrativeContextModels:
    """Tests for the Pydantic models used in narrative context."""

    def test_narrative_context_model(self):
        """Test the creation and validation of the NarrativeContext model."""
        from backend.app.core.narrative_context import NarrativeContext
        context_data = {
            "narrative_tension": 0.8,
            "character_arc_stage": "rising_action",
            "scene_atmosphere": "tense",
            "dialogue_context": "confrontation",
            "primary_emotion": "determination",
            "secondary_emotions": ["anxiety", "hope"]
        }
        context = NarrativeContext(**context_data)
        assert context.narrative_tension == 0.8
        assert context.primary_emotion == "determination"
        assert "hope" in context.secondary_emotions

    def test_emotion_blend_model(self):
        """Test the EmotionBlend model creation and validation."""
        from backend.app.core.narrative_context import EmotionBlend
        
        blend_data = {
            "primary_emotion": "determination",
            "secondary_emotions": {"anxiety": 0.3, "hope": 0.2},
            "overall_intensity": 0.8
        }
        blend = EmotionBlend(**blend_data)
        assert blend.primary_emotion == "determination"
        assert blend.secondary_emotions["anxiety"] == 0.3
        assert blend.overall_intensity == 0.8

    def test_emotion_blend_validation(self):
        """Test that EmotionBlend validates emotion weights correctly."""
        from backend.app.core.narrative_context import EmotionBlend
        
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            EmotionBlend(
                primary_emotion="joy",
                secondary_emotions={"sadness": 1.5},  # Invalid weight
                overall_intensity=0.5
            )

    def test_prosody_control_model(self):
        """Test the ProsodyControl model creation and validation."""
        from backend.app.core.narrative_context import ProsodyControl
        
        prosody_data = {
            "speaking_rate": 1.2,
            "pause_duration": 0.3,
            "emphasis_words": ["important", "critical"],
            "pitch_variation": 1.1
        }
        prosody = ProsodyControl(**prosody_data)
        assert prosody.speaking_rate == 1.2
        assert prosody.pause_duration == 0.3
        assert "important" in prosody.emphasis_words
        assert prosody.pitch_variation == 1.1

    def test_prosody_control_validation(self):
        """Test that ProsodyControl validates rate ranges correctly."""
        from backend.app.core.narrative_context import ProsodyControl
        
        with pytest.raises(ValueError, match="Rate must be between 0.5 and 2.0"):
            ProsodyControl(speaking_rate=3.0)  # Invalid rate

    def test_emotional_state_model(self):
        """Test the EmotionalState model creation."""
        from backend.app.core.narrative_context import EmotionalState
        
        state_data = {
            "primary_emotion": "joy",
            "secondary_emotions": {"excitement": 0.4, "relief": 0.2},
            "intensity": 0.8,
            "narrative_tension": 0.6,
            "scene_id": "scene_1",
            "trigger_event": "good_news_received"
        }
        state = EmotionalState(**state_data)
        assert state.primary_emotion == "joy"
        assert state.secondary_emotions["excitement"] == 0.4
        assert state.intensity == 0.8
        assert state.scene_id == "scene_1"
        assert state.trigger_event == "good_news_received"

    def test_emotional_transition_model(self):
        """Test the EmotionalTransition model creation."""
        from backend.app.core.narrative_context import EmotionalState, EmotionalTransition
        
        from_state = EmotionalState(
            primary_emotion="calm",
            secondary_emotions={},
            intensity=0.3,
            narrative_tension=0.2
        )
        
        to_state = EmotionalState(
            primary_emotion="excitement",
            secondary_emotions={"joy": 0.3},
            intensity=0.8,
            narrative_tension=0.7
        )
        
        transition = EmotionalTransition(
            from_state=from_state,
            to_state=to_state,
            transition_type="gradual",
            smoothing_factor=0.7,
            duration_seconds=3.0
        )
        
        assert transition.from_state.primary_emotion == "calm"
        assert transition.to_state.primary_emotion == "excitement"
        assert transition.transition_type == "gradual"
        assert transition.smoothing_factor == 0.7


class TestTemporalConsistencyTracker:
    """Tests for the TemporalConsistencyTracker."""

    def test_tracker_initialization(self):
        """Test basic tracker initialization."""
        from backend.app.core.narrative_context import TemporalConsistencyTracker
        
        tracker = TemporalConsistencyTracker("character_1", max_history_length=10)
        assert tracker.character_id == "character_1"
        assert tracker.max_history_length == 10
        assert len(tracker.emotional_history) == 0
        assert tracker.current_scene_id is None

    def test_add_emotional_state(self):
        """Test adding emotional states to the tracker."""
        from backend.app.core.narrative_context import TemporalConsistencyTracker, EmotionalState
        
        tracker = TemporalConsistencyTracker("character_1")
        
        state1 = EmotionalState(
            primary_emotion="calm",
            secondary_emotions={},
            intensity=0.3,
            narrative_tension=0.2,
            scene_id="scene_1"
        )
        
        tracker.add_emotional_state(state1)
        
        assert len(tracker.emotional_history) == 1
        assert tracker.current_scene_id == "scene_1"
        assert tracker.get_current_state().primary_emotion == "calm"

    def test_history_length_limit(self):
        """Test that history length is properly limited."""
        from backend.app.core.narrative_context import TemporalConsistencyTracker, EmotionalState
        
        tracker = TemporalConsistencyTracker("character_1", max_history_length=3)
        
        # Add 5 states
        for i in range(5):
            state = EmotionalState(
                primary_emotion=f"emotion_{i}",
                secondary_emotions={},
                intensity=0.5,
                narrative_tension=0.5
            )
            tracker.add_emotional_state(state)
        
        # Should only keep the last 3
        assert len(tracker.emotional_history) == 3
        assert tracker.emotional_history[0].primary_emotion == "emotion_2"
        assert tracker.emotional_history[-1].primary_emotion == "emotion_4"

    def test_get_emotional_arc(self):
        """Test getting emotional arc within a time window."""
        from backend.app.core.narrative_context import TemporalConsistencyTracker, EmotionalState
        
        tracker = TemporalConsistencyTracker("character_1")
        
        # Add states with different timestamps
        now = datetime.now()
        
        # Old state (45 minutes ago)
        old_state = EmotionalState(
            primary_emotion="old_emotion",
            secondary_emotions={},
            intensity=0.3,
            narrative_tension=0.2,
            timestamp=now - timedelta(minutes=45)
        )
        
        # Recent state (15 minutes ago)
        recent_state = EmotionalState(
            primary_emotion="recent_emotion",
            secondary_emotions={},
            intensity=0.7,
            narrative_tension=0.6,
            timestamp=now - timedelta(minutes=15)
        )
        
        tracker.add_emotional_state(old_state)
        tracker.add_emotional_state(recent_state)
        
        # Get arc for last 30 minutes
        arc = tracker.get_emotional_arc(lookback_minutes=30)
        
        assert len(arc) == 1
        assert arc[0].primary_emotion == "recent_emotion"

    def test_calculate_emotional_momentum(self):
        """Test emotional momentum calculation."""
        from backend.app.core.narrative_context import TemporalConsistencyTracker, EmotionalState
        
        tracker = TemporalConsistencyTracker("character_1")
        
        # Add states showing emotional progression
        states = [
            EmotionalState(primary_emotion="calm", secondary_emotions={}, intensity=0.3, narrative_tension=0.2),
            EmotionalState(primary_emotion="calm", secondary_emotions={}, intensity=0.4, narrative_tension=0.3),
            EmotionalState(primary_emotion="anxiety", secondary_emotions={}, intensity=0.6, narrative_tension=0.5),
            EmotionalState(primary_emotion="fear", secondary_emotions={}, intensity=0.8, narrative_tension=0.7),
        ]
        
        for state in states:
            tracker.add_emotional_state(state)
        
        momentum = tracker.calculate_emotional_momentum()
        
        assert "_intensity_trend" in momentum
        assert momentum["_intensity_trend"] > 0  # Intensity is increasing

    def test_predict_next_emotional_state_no_history(self):
        """Test predicting next state when no history exists."""
        from backend.app.core.narrative_context import (
            TemporalConsistencyTracker, EmotionBlendingService, 
            NarrativeContext
        )
        
        tracker = TemporalConsistencyTracker("character_1")
        service = EmotionBlendingService()
        
        context = NarrativeContext(
            narrative_tension=0.6,
            character_arc_stage="rising_action",
            scene_atmosphere="tense",
            dialogue_context="confrontation",
            primary_emotion="determination",
            secondary_emotions=["anxiety"]
        )
        
        predicted_state = tracker.predict_next_emotional_state(context, service)
        
        assert predicted_state.primary_emotion == "determination"
        assert "anxiety" in predicted_state.secondary_emotions
        assert predicted_state.intensity == 0.6  # Should match narrative tension

    def test_predict_next_emotional_state_with_history(self):
        """Test predicting next state with existing history."""
        from backend.app.core.narrative_context import (
            TemporalConsistencyTracker, EmotionBlendingService,
            NarrativeContext, EmotionalState
        )
        
        tracker = TemporalConsistencyTracker("character_1")
        service = EmotionBlendingService()
        
        # Add current state
        current_state = EmotionalState(
            primary_emotion="calm",
            secondary_emotions={"contentment": 0.3},
            intensity=0.4,
            narrative_tension=0.3
        )
        tracker.add_emotional_state(current_state)
        
        # New context with same emotion
        context = NarrativeContext(
            narrative_tension=0.6,
            character_arc_stage="rising_action",
            scene_atmosphere="tense",
            dialogue_context="confrontation",
            primary_emotion="calm",  # Same as current
            secondary_emotions=["focus"]
        )
        
        predicted_state = tracker.predict_next_emotional_state(context, service)
        
        assert predicted_state.primary_emotion == "calm"
        assert predicted_state.intensity > current_state.intensity  # Should increase due to tension
        assert "focus" in predicted_state.secondary_emotions

    def test_create_smooth_transition(self):
        """Test creating smooth transitions between emotional states."""
        from backend.app.core.narrative_context import (
            TemporalConsistencyTracker, EmotionalState
        )
        
        tracker = TemporalConsistencyTracker("character_1")
        
        # Add current state
        current_state = EmotionalState(
            primary_emotion="calm",
            secondary_emotions={},
            intensity=0.3,
            narrative_tension=0.2
        )
        tracker.add_emotional_state(current_state)
        
        # Target state
        target_state = EmotionalState(
            primary_emotion="excitement",
            secondary_emotions={"joy": 0.4},
            intensity=0.8,
            narrative_tension=0.7
        )
        
        transition = tracker.create_smooth_transition(target_state, transition_duration=3.0)
        
        assert transition.from_state.primary_emotion == "calm"
        assert transition.to_state.primary_emotion == "excitement"
        assert transition.transition_type in ["sudden", "gradual", "subtle"]
        assert transition.duration_seconds == 3.0

    def test_create_smooth_transition_no_history(self):
        """Test creating transition when no history exists."""
        from backend.app.core.narrative_context import (
            TemporalConsistencyTracker, EmotionalState
        )
        
        tracker = TemporalConsistencyTracker("character_1")
        
        target_state = EmotionalState(
            primary_emotion="joy",
            secondary_emotions={},
            intensity=0.7,
            narrative_tension=0.5
        )
        
        transition = tracker.create_smooth_transition(target_state)
        
        assert transition.transition_type == "immediate"
        assert transition.duration_seconds == 0.0


class TestEmotionBlendingService:
    """Tests for the EmotionBlendingService."""

    def test_create_emotion_blend_basic(self):
        """Test basic emotion blend creation."""
        from backend.app.core.narrative_context import EmotionBlendingService
        
        service = EmotionBlendingService()
        blend = service.create_emotion_blend(
            primary_emotion="determination",
            secondary_emotions=["anxiety", "hope"],
            narrative_tension=0.7
        )
        
        assert blend.primary_emotion == "determination"
        assert "anxiety" in blend.secondary_emotions
        assert "hope" in blend.secondary_emotions
        assert blend.overall_intensity > 0.6  # Should be influenced by tension

    def test_create_emotion_blend_tension_scaling(self):
        """Test that narrative tension affects emotion blend intensity."""
        from backend.app.core.narrative_context import EmotionBlendingService
        
        service = EmotionBlendingService()
        
        # Low tension blend
        low_tension_blend = service.create_emotion_blend(
            primary_emotion="calm",
            secondary_emotions=["contentment"],
            narrative_tension=0.2
        )
        
        # High tension blend
        high_tension_blend = service.create_emotion_blend(
            primary_emotion="fear",
            secondary_emotions=["panic"],
            narrative_tension=0.9
        )
        
        assert high_tension_blend.overall_intensity > low_tension_blend.overall_intensity
        assert high_tension_blend.secondary_emotions["panic"] > low_tension_blend.secondary_emotions["contentment"]

    def test_create_emotion_blend_from_state(self):
        """Test creating emotion blend from emotional state."""
        from backend.app.core.narrative_context import EmotionBlendingService, EmotionalState
        
        service = EmotionBlendingService()
        
        emotional_state = EmotionalState(
            primary_emotion="joy",
            secondary_emotions={"excitement": 0.4, "relief": 0.2},
            intensity=0.8,
            narrative_tension=0.6
        )
        
        blend = service.create_emotion_blend_from_state(emotional_state)
        
        assert blend.primary_emotion == "joy"
        assert blend.secondary_emotions["excitement"] == 0.4
        assert blend.overall_intensity == 0.8

    def test_calculate_prosody_from_context(self):
        """Test prosody calculation from emotion blend and narrative context."""
        from backend.app.core.narrative_context import EmotionBlendingService, EmotionBlend, NarrativeContext
        
        service = EmotionBlendingService()
        
        emotion_blend = EmotionBlend(
            primary_emotion="excitement",
            secondary_emotions={"joy": 0.3},
            overall_intensity=0.8
        )
        
        narrative_context = NarrativeContext(
            narrative_tension=0.6,
            character_arc_stage="rising_action",
            scene_atmosphere="energetic",
            dialogue_context="celebration",
            primary_emotion="excitement",
            secondary_emotions=["joy"]
        )
        
        prosody = service.calculate_prosody_from_context(emotion_blend, narrative_context)
        
        assert prosody.speaking_rate > 1.0  # Excitement should increase speaking rate
        assert prosody.pitch_variation > 1.0  # Excitement should increase pitch variation
        assert 0.5 <= prosody.speaking_rate <= 2.0  # Within valid range

    def test_calculate_prosody_different_emotions(self):
        """Test prosody calculation for different emotion types."""
        from backend.app.core.narrative_context import EmotionBlendingService, EmotionBlend, NarrativeContext
        
        service = EmotionBlendingService()
        
        # Test sadness
        sadness_blend = EmotionBlend(
            primary_emotion="sadness",
            secondary_emotions={},
            overall_intensity=0.7
        )
        
        context = NarrativeContext(
            narrative_tension=0.3,
            character_arc_stage="falling_action",
            scene_atmosphere="somber",
            dialogue_context="grief",
            primary_emotion="sadness"
        )
        
        sadness_prosody = service.calculate_prosody_from_context(sadness_blend, context)
        
        assert sadness_prosody.speaking_rate < 1.0  # Sadness should slow speech
        assert sadness_prosody.pause_duration > 0.0  # Sadness should add pauses

    def test_generate_emotion_tag_simple(self):
        """Test emotion tag generation for simple emotions."""
        from backend.app.core.narrative_context import EmotionBlendingService, EmotionBlend
        
        service = EmotionBlendingService()
        
        simple_blend = EmotionBlend(
            primary_emotion="joy",
            secondary_emotions={},
            overall_intensity=0.8
        )
        
        tag = service.generate_emotion_tag(simple_blend)
        assert tag == "<joy>"

    def test_generate_emotion_tag_complex(self):
        """Test emotion tag generation for complex emotion blends."""
        from backend.app.core.narrative_context import EmotionBlendingService, EmotionBlend
        
        service = EmotionBlendingService()
        
        complex_blend = EmotionBlend(
            primary_emotion="determination",
            secondary_emotions={"anxiety": 0.4, "hope": 0.3},
            overall_intensity=0.9
        )
        
        tag = service.generate_emotion_tag(complex_blend)
        assert "determination" in tag
        assert "anxiety" in tag
        assert "hope" in tag
        assert tag.startswith("<determination with")

    def test_generate_emotion_tag_filters_weak_emotions(self):
        """Test that weak secondary emotions are filtered out of tags."""
        from backend.app.core.narrative_context import EmotionBlendingService, EmotionBlend
        
        service = EmotionBlendingService()
        
        blend_with_weak_secondary = EmotionBlend(
            primary_emotion="confidence",
            secondary_emotions={"doubt": 0.1, "pride": 0.3},  # doubt is too weak
            overall_intensity=0.8
        )
        
        tag = service.generate_emotion_tag(blend_with_weak_secondary)
        assert "confidence" in tag
        assert "pride" in tag
        assert "doubt" not in tag  # Should be filtered out


class TestNarrativeContextService:
    """Tests for the NarrativeContextService."""

    @pytest.fixture
    def mock_llm_client(self):
        """Fixture for a mocked LLM client."""
        mock_client = Mock()
        # Simulate a JSON response from the LLM
        mock_response = {
            "narrative_tension": 0.7,
            "character_arc_stage": "climax",
            "scene_atmosphere": "chaotic",
            "dialogue_context": "desperate_plea",
            "primary_emotion": "determination",
            "secondary_emotions": ["fear", "resolve"]
        }
        # The client's generate method is async and returns a JSON string
        async def mock_generate(*args, **kwargs):
            import json
            return json.dumps(mock_response)
        
        mock_client.generate = Mock(side_effect=mock_generate)
        return mock_client

    @pytest.mark.asyncio
    async def test_analyze_context_from_dialogue(self, mock_llm_client):
        """Test that the service can analyze a piece of dialogue and return a structured context."""
        from backend.app.core.narrative_context import NarrativeContext, NarrativeContextService

        service = NarrativeContextService(client=mock_llm_client)
        
        dialogue_history = [
            {"role": "user", "content": "What are you doing?"},
            {"role": "assistant", "content": "I am preparing for the final battle."},
            {"role": "user", "content": "You can't win. It's hopeless!"},
            {"role": "assistant", "content": "I have to try! For everyone's sake!"}
        ]
        
        narrative_context = await service.analyze_context(dialogue_history)
        
        assert isinstance(narrative_context, NarrativeContext)
        assert narrative_context.primary_emotion == "determination"
        assert narrative_context.narrative_tension == 0.7
        assert "resolve" in narrative_context.secondary_emotions
        
        # Verify that the LLM was called with the correct prompt structure
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        prompt = call_args.kwargs['prompt']
        assert "You are a narrative analysis engine." in prompt
        assert "I have to try! For everyone's sake!" in prompt 