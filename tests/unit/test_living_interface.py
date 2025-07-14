"""
Unit tests for the Living Interface Integration Layer.
"""

import pytest
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Add app directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))


class TestLivingInterfaceModels:
    """Tests for the living interface data models."""

    def test_head_output_creation(self):
        """Test HeadOutput dataclass creation."""
        from backend.app.core.living_interface import HeadOutput, HeadType
        
        output = HeadOutput(
            head_type=HeadType.MEMORY,
            content={"test": "data"},
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"source": "test"}
        )
        
        assert output.head_type == HeadType.MEMORY
        assert output.content["test"] == "data"
        assert output.confidence == 0.85
        assert output.metadata["source"] == "test"

    def test_user_interaction_creation(self):
        """Test UserInteraction dataclass creation."""
        from backend.app.core.living_interface import UserInteraction, InteractionType
        
        interaction = UserInteraction(
            interaction_type=InteractionType.DIALOGUE,
            content="Hello, how are you?",
            timestamp=datetime.now(),
            context={"scene": "greeting"}
        )
        
        assert interaction.interaction_type == InteractionType.DIALOGUE
        assert interaction.content == "Hello, how are you?"
        assert interaction.context["scene"] == "greeting"

    def test_voice_adaptation_config(self):
        """Test VoiceAdaptationConfig model."""
        from backend.app.core.living_interface import VoiceAdaptationConfig
        
        config = VoiceAdaptationConfig(
            adaptation_speed=0.8,
            memory_influence=0.4,
            story_influence=0.6,
            control_influence=0.3
        )
        
        assert config.adaptation_speed == 0.8
        assert config.memory_influence == 0.4
        assert config.story_influence == 0.6
        assert config.control_influence == 0.3


class TestTriHeadInterface:
    """Tests for the TriHeadInterface."""

    def test_interface_initialization(self):
        """Test TriHeadInterface initialization."""
        from backend.app.core.living_interface import TriHeadInterface
        
        interface = TriHeadInterface("test_character")
        
        assert interface.character_id == "test_character"
        assert interface.memory_head_active is True
        assert interface.story_head_active is True
        assert interface.control_head_active is True

    @pytest.mark.asyncio
    async def test_query_memory_head(self):
        """Test querying the memory head."""
        from backend.app.core.living_interface import TriHeadInterface, HeadType
        
        interface = TriHeadInterface("test_character")
        
        output = await interface.query_memory_head(
            "What do you remember about our last conversation?",
            {"narrative_context": {"tension": 0.5}}
        )
        
        assert output.head_type == HeadType.MEMORY
        assert "relevant_memories" in output.content
        assert "character_growth" in output.content
        assert output.confidence > 0.0
        assert output.metadata["query"] == "What do you remember about our last conversation?"

    @pytest.mark.asyncio
    async def test_query_story_head(self):
        """Test querying the story generation head."""
        from backend.app.core.living_interface import TriHeadInterface, HeadType
        from backend.app.core.narrative_context import NarrativeContext
        
        interface = TriHeadInterface("test_character")
        
        context = NarrativeContext(
            narrative_tension=0.7,
            character_arc_stage="rising_action",
            scene_atmosphere="tense",
            dialogue_context="confrontation",
            primary_emotion="determination"
        )
        
        output = await interface.query_story_head(context)
        
        assert output.head_type == HeadType.STORY_GENERATION
        assert "narrative_predictions" in output.content
        assert "story_momentum" in output.content
        assert output.content["story_momentum"]["intensity"] == 0.7
        assert output.confidence > 0.0

    @pytest.mark.asyncio
    async def test_query_control_head(self):
        """Test querying the control head."""
        from backend.app.core.living_interface import TriHeadInterface, HeadType
        
        interface = TriHeadInterface("test_character")
        
        output = await interface.query_control_head(
            "maintain_character_consistency",
            {"emotional_state": {"intensity": 0.8}}
        )
        
        assert output.head_type == HeadType.CONTROL
        assert "generation_parameters" in output.content
        assert "content_filters" in output.content
        assert "response_guidelines" in output.content
        assert output.confidence > 0.0


class TestLivingInterfaceOrchestrator:
    """Tests for the LivingInterfaceOrchestrator."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client for testing."""
        mock_client = Mock()
        
        async def mock_generate(*args, **kwargs):
            import json
            return json.dumps({
                "narrative_tension": 0.6,
                "character_arc_stage": "rising_action",
                "scene_atmosphere": "tense",
                "dialogue_context": "confrontation",
                "primary_emotion": "determination",
                "secondary_emotions": ["anxiety"]
            })
        
        mock_client.generate = Mock(side_effect=mock_generate)
        return mock_client

    def test_orchestrator_initialization(self, mock_client):
        """Test LivingInterfaceOrchestrator initialization."""
        from backend.app.core.living_interface import LivingInterfaceOrchestrator, VoiceAdaptationConfig
        
        config = VoiceAdaptationConfig(adaptation_speed=0.9)
        orchestrator = LivingInterfaceOrchestrator(
            "test_character",
            client=mock_client,
            config=config
        )
        
        assert orchestrator.character_id == "test_character"
        assert orchestrator.config.adaptation_speed == 0.9
        assert len(orchestrator.interaction_history) == 0

    @pytest.mark.asyncio
    async def test_process_user_interaction(self, mock_client):
        """Test processing a user interaction."""
        from backend.app.core.living_interface import (
            LivingInterfaceOrchestrator, UserInteraction, InteractionType
        )
        
        orchestrator = LivingInterfaceOrchestrator("test_character", client=mock_client)
        
        interaction = UserInteraction(
            interaction_type=InteractionType.DIALOGUE,
            content="I'm feeling worried about tomorrow.",
            timestamp=datetime.now()
        )
        
        dialogue_history = [
            {"role": "user", "content": "I'm feeling worried about tomorrow."},
            {"role": "assistant", "content": "I understand your concern. Let's talk about it."}
        ]
        
        response_params = await orchestrator.process_user_interaction(
            interaction,
            dialogue_history
        )
        
        # Verify response structure
        assert "narrative_context" in response_params
        assert "emotional_state" in response_params
        assert "emotion_blend" in response_params
        assert "prosody_control" in response_params
        assert "emotion_tag" in response_params
        assert "tri_head_outputs" in response_params
        assert "generation_parameters" in response_params
        assert "adaptation_metadata" in response_params
        
        # Verify tri-head outputs
        tri_head = response_params["tri_head_outputs"]
        assert "memory" in tri_head
        assert "story" in tri_head
        assert "control" in tri_head
        
        # Verify interaction was stored
        assert len(orchestrator.interaction_history) == 1
        assert orchestrator.interaction_history[0].content == "I'm feeling worried about tomorrow."

    @pytest.mark.asyncio
    async def test_apply_tri_head_influences(self, mock_client):
        """Test applying tri-head influences to emotional state."""
        from backend.app.core.living_interface import (
            LivingInterfaceOrchestrator, HeadOutput, HeadType
        )
        from backend.app.core.narrative_context import EmotionalState
        
        orchestrator = LivingInterfaceOrchestrator("test_character", client=mock_client)
        
        base_state = EmotionalState(
            primary_emotion="calm",
            secondary_emotions={"contentment": 0.3},
            intensity=0.4,
            narrative_tension=0.3
        )
        
        memory_output = HeadOutput(
            head_type=HeadType.MEMORY,
            content={
                "character_growth": {
                    "confidence_level": 0.8,
                    "relationship_depth": 0.6
                }
            },
            confidence=0.85,
            timestamp=datetime.now()
        )
        
        story_output = HeadOutput(
            head_type=HeadType.STORY_GENERATION,
            content={
                "suggested_emotional_tone": "determined_with_underlying_vulnerability"
            },
            confidence=0.78,
            timestamp=datetime.now()
        )
        
        control_output = HeadOutput(
            head_type=HeadType.CONTROL,
            content={
                "content_filters": {
                    "emotional_intensity_cap": 0.7
                }
            },
            confidence=0.92,
            timestamp=datetime.now()
        )
        
        influenced_state = await orchestrator._apply_tri_head_influences(
            base_state,
            memory_output,
            story_output,
            control_output
        )
        
        # Memory influence should increase intensity due to high confidence
        assert influenced_state.intensity > base_state.intensity
        
        # Story influence should add determination and vulnerability
        assert "determination" in influenced_state.secondary_emotions or "vulnerability" in influenced_state.secondary_emotions

    def test_get_character_emotional_summary(self, mock_client):
        """Test getting character emotional summary."""
        from backend.app.core.living_interface import LivingInterfaceOrchestrator
        from backend.app.core.narrative_context import EmotionalState
        
        orchestrator = LivingInterfaceOrchestrator("test_character", client=mock_client)
        
        # Add some emotional state
        state = EmotionalState(
            primary_emotion="joy",
            secondary_emotions={"excitement": 0.4},
            intensity=0.7,
            narrative_tension=0.5
        )
        orchestrator.temporal_tracker.add_emotional_state(state)
        
        summary = orchestrator.get_character_emotional_summary()
        
        assert summary["character_id"] == "test_character"
        assert summary["current_state"]["primary_emotion"] == "joy"
        assert summary["interaction_count"] == 0
        assert "last_updated" in summary

    @pytest.mark.asyncio
    async def test_adapt_voice_parameters(self, mock_client):
        """Test adapting voice parameters based on emotional state."""
        from backend.app.core.living_interface import LivingInterfaceOrchestrator
        from backend.app.core.narrative_context import EmotionalState, NarrativeContext
        
        orchestrator = LivingInterfaceOrchestrator("test_character", client=mock_client)
        
        # Add emotional state
        state = EmotionalState(
            primary_emotion="excitement",
            secondary_emotions={"joy": 0.3},
            intensity=0.8,
            narrative_tension=0.6
        )
        orchestrator.temporal_tracker.add_emotional_state(state)
        
        base_parameters = {
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        context = NarrativeContext(
            narrative_tension=0.6,
            character_arc_stage="rising_action",
            scene_atmosphere="energetic",
            dialogue_context="celebration",
            primary_emotion="excitement"
        )
        
        adapted_params = await orchestrator.adapt_voice_parameters(
            base_parameters,
            context
        )
        
        # Should have increased temperature due to high intensity
        assert adapted_params["temperature"] > base_parameters["temperature"]
        
        # Should have prosody control and emotion tag
        assert "prosody_control" in adapted_params
        assert "emotion_tag" in adapted_params
        assert "<excitement" in adapted_params["emotion_tag"]
        assert "joy" in adapted_params["emotion_tag"]

    @pytest.mark.asyncio
    async def test_adapt_voice_parameters_no_state(self, mock_client):
        """Test adapting voice parameters when no emotional state exists."""
        from backend.app.core.living_interface import LivingInterfaceOrchestrator
        from backend.app.core.narrative_context import NarrativeContext
        
        orchestrator = LivingInterfaceOrchestrator("test_character", client=mock_client)
        
        base_parameters = {
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        context = NarrativeContext(
            narrative_tension=0.5,
            character_arc_stage="exposition",
            scene_atmosphere="neutral",
            dialogue_context="introduction",
            primary_emotion="calm"
        )
        
        adapted_params = await orchestrator.adapt_voice_parameters(
            base_parameters,
            context
        )
        
        # Should return unchanged parameters when no state exists
        assert adapted_params == base_parameters

    @pytest.mark.asyncio
    async def test_emotional_state_progression(self, mock_client):
        """Test that emotional states progress naturally through interactions."""
        from backend.app.core.living_interface import (
            LivingInterfaceOrchestrator, UserInteraction, InteractionType
        )
        
        orchestrator = LivingInterfaceOrchestrator("test_character", client=mock_client)
        
        # First interaction - establish baseline
        interaction1 = UserInteraction(
            interaction_type=InteractionType.DIALOGUE,
            content="Hello, nice to meet you!",
            timestamp=datetime.now()
        )
        
        dialogue1 = [
            {"role": "user", "content": "Hello, nice to meet you!"},
            {"role": "assistant", "content": "Hello! It's wonderful to meet you too."}
        ]
        
        response1 = await orchestrator.process_user_interaction(interaction1, dialogue1)
        initial_state = response1["emotional_state"]
        
        # Second interaction - more intense
        interaction2 = UserInteraction(
            interaction_type=InteractionType.DIALOGUE,
            content="I'm really excited about this adventure!",
            timestamp=datetime.now()
        )
        
        dialogue2 = dialogue1 + [
            {"role": "user", "content": "I'm really excited about this adventure!"},
            {"role": "assistant", "content": "Yes! I can feel the excitement building too!"}
        ]
        
        response2 = await orchestrator.process_user_interaction(interaction2, dialogue2)
        second_state = response2["emotional_state"]
        
        # Verify progression
        assert len(orchestrator.interaction_history) == 2
        
        # The emotional state should have some continuity or progression
        # (exact values depend on the mock LLM responses, but we can verify structure)
        assert "primary_emotion" in second_state
        assert "intensity" in second_state
        assert isinstance(second_state["intensity"], (int, float))

    def test_voice_adaptation_config_influences(self, mock_client):
        """Test that VoiceAdaptationConfig properly influences the system."""
        from backend.app.core.living_interface import LivingInterfaceOrchestrator, VoiceAdaptationConfig
        
        # High memory influence config
        high_memory_config = VoiceAdaptationConfig(
            memory_influence=0.8,
            story_influence=0.1,
            control_influence=0.1
        )
        
        orchestrator = LivingInterfaceOrchestrator(
            "test_character",
            client=mock_client,
            config=high_memory_config
        )
        
        assert orchestrator.config.memory_influence == 0.8
        assert orchestrator.config.story_influence == 0.1
        assert orchestrator.config.control_influence == 0.1 