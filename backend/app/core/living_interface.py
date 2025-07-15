"""
Living Interface Integration Layer

This module provides the integration layer between the advanced emotion control system
and the tri-head architecture (memory, story generation, control heads). It enables
dynamic voice adaptation based on real-time user interaction and narrative context.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from .narrative_context import (
    NarrativeContext, NarrativeContextService, EmotionBlendingService,
    TemporalConsistencyTracker, EmotionalState, EmotionBlend, ProsodyControl
)
from .openai_client import get_client, OpenAIClient

logger = logging.getLogger(__name__)


class HeadType(Enum):
    """Types of heads in the tri-head architecture."""
    MEMORY = "memory"
    STORY_GENERATION = "story_generation"
    CONTROL = "control"


class InteractionType(Enum):
    """Types of user interactions."""
    DIALOGUE = "dialogue"
    ACTION = "action"
    EMOTIONAL_RESPONSE = "emotional_response"
    SCENE_TRANSITION = "scene_transition"


@dataclass
class HeadOutput:
    """Output from a specific head in the tri-head architecture."""
    head_type: HeadType
    content: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserInteraction:
    """Represents a user interaction with the system."""
    interaction_type: InteractionType
    content: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


class VoiceAdaptationConfig(BaseModel):
    """Configuration for dynamic voice adaptation."""
    adaptation_speed: float = Field(
        default=0.7,
        description="How quickly voice adapts to changes (0.0-1.0)"
    )
    memory_influence: float = Field(
        default=0.3,
        description="How much character memory influences voice (0.0-1.0)"
    )
    story_influence: float = Field(
        default=0.5,
        description="How much story context influences voice (0.0-1.0)"
    )
    control_influence: float = Field(
        default=0.2,
        description="How much control head influences voice (0.0-1.0)"
    )
    emotional_momentum_weight: float = Field(
        default=0.4,
        description="Weight of emotional momentum in adaptation (0.0-1.0)"
    )


class TriHeadInterface:
    """
    Interface for communicating with the tri-head architecture.
    
    This is a mock interface for the actual tri-head system that would be
    implemented in the narrative engine.
    """
    
    def __init__(self, character_id: str):
        self.character_id = character_id
        self.memory_head_active = True
        self.story_head_active = True
        self.control_head_active = True
    
    async def query_memory_head(self, query: str, context: Dict[str, Any]) -> HeadOutput:
        """
        Query the memory head for character memories and experiences.
        
        Args:
            query: The query to send to the memory head
            context: Current context information
            
        Returns:
            HeadOutput from the memory head
        """
        # Mock implementation - in reality this would interface with the memory system
        memory_content = {
            "relevant_memories": [
                {"memory": "First meeting with the user", "emotional_weight": 0.7},
                {"memory": "Previous conversation about goals", "emotional_weight": 0.5}
            ],
            "emotional_associations": {
                "user": "trust and curiosity",
                "current_topic": "determination and focus"
            },
            "character_growth": {
                "confidence_level": 0.8,
                "relationship_depth": 0.6
            }
        }
        
        return HeadOutput(
            head_type=HeadType.MEMORY,
            content=memory_content,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"query": query}
        )
    
    async def query_story_head(self, current_context: NarrativeContext) -> HeadOutput:
        """
        Query the story generation head for narrative predictions.
        
        Args:
            current_context: Current narrative context
            
        Returns:
            HeadOutput from the story generation head
        """
        # Mock implementation - in reality this would interface with the story system
        story_content = {
            "narrative_predictions": [
                {"event": "emotional_revelation", "probability": 0.7},
                {"event": "conflict_resolution", "probability": 0.4}
            ],
            "story_momentum": {
                "direction": "rising_tension",
                "intensity": current_context.narrative_tension
            },
            "character_arc_position": current_context.character_arc_stage,
            "suggested_emotional_tone": "determined_with_underlying_vulnerability"
        }
        
        return HeadOutput(
            head_type=HeadType.STORY_GENERATION,
            content=story_content,
            confidence=0.78,
            timestamp=datetime.now(),
            metadata={"narrative_tension": current_context.narrative_tension}
        )
    
    async def query_control_head(self, desired_outcome: str, constraints: Dict[str, Any]) -> HeadOutput:
        """
        Query the control head for generation parameters.
        
        Args:
            desired_outcome: The desired outcome for the interaction
            constraints: Any constraints to consider
            
        Returns:
            HeadOutput from the control head
        """
        # Mock implementation - in reality this would interface with the control system
        control_content = {
            "generation_parameters": {
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "content_filters": {
                "emotional_intensity_cap": 0.9,
                "narrative_coherence_weight": 0.8
            },
            "response_guidelines": {
                "max_length": 150,
                "tone_consistency": True,
                "character_voice_strength": 0.85
            }
        }
        
        return HeadOutput(
            head_type=HeadType.CONTROL,
            content=control_content,
            confidence=0.92,
            timestamp=datetime.now(),
            metadata={"desired_outcome": desired_outcome}
        )


class LivingInterfaceOrchestrator:
    """
    Main orchestrator for the living interface system.
    
    Coordinates between narrative context analysis, emotion blending, temporal consistency,
    and the tri-head architecture to provide dynamic voice adaptation.
    """
    
    def __init__(
        self,
        character_id: str,
        client: Optional[OpenAIClient] = None,
        config: Optional[VoiceAdaptationConfig] = None
    ):
        self.character_id = character_id
        self.config = config or VoiceAdaptationConfig()
        
        # Initialize core services
        self.narrative_service = NarrativeContextService(client)
        self.emotion_service = EmotionBlendingService(client)
        self.temporal_tracker = TemporalConsistencyTracker(character_id)
        self.tri_head_interface = TriHeadInterface(character_id)
        
        # Interaction history
        self.interaction_history: List[UserInteraction] = []
        
    async def process_user_interaction(
        self,
        interaction: UserInteraction,
        dialogue_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Process a user interaction and generate comprehensive response parameters.
        
        Args:
            interaction: The user interaction to process
            dialogue_history: Recent dialogue history
            
        Returns:
            Dictionary containing all response parameters
        """
        # Store interaction
        self.interaction_history.append(interaction)
        
        # 1. Analyze narrative context
        narrative_context = await self.narrative_service.analyze_context(dialogue_history)
        
        # 2. Query tri-head architecture
        memory_output = await self.tri_head_interface.query_memory_head(
            interaction.content,
            {"narrative_context": narrative_context.model_dump()}
        )
        
        story_output = await self.tri_head_interface.query_story_head(narrative_context)
        
        control_output = await self.tri_head_interface.query_control_head(
            "maintain_character_consistency",
            {"emotional_state": self.temporal_tracker.get_current_state()}
        )
        
        # 3. Predict next emotional state
        predicted_state = self.temporal_tracker.predict_next_emotional_state(
            narrative_context,
            self.emotion_service
        )
        
        # 4. Apply tri-head influences to emotional state
        influenced_state = await self._apply_tri_head_influences(
            predicted_state,
            memory_output,
            story_output,
            control_output
        )
        
        # 5. Create emotion blend and prosody controls
        emotion_blend = self.emotion_service.create_emotion_blend_from_state(influenced_state)
        prosody_control = self.emotion_service.calculate_prosody_from_context(
            emotion_blend,
            narrative_context
        )
        
        # 6. Generate emotion tag
        emotion_tag = self.emotion_service.generate_emotion_tag(emotion_blend)
        
        # 7. Create smooth transition if needed
        transition = self.temporal_tracker.create_smooth_transition(influenced_state)
        
        # 8. Update temporal tracker
        self.temporal_tracker.add_emotional_state(influenced_state)
        
        # 9. Compile response parameters
        response_params = {
            "narrative_context": narrative_context.model_dump(),
            "emotional_state": influenced_state.model_dump(),
            "emotion_blend": emotion_blend.model_dump(),
            "prosody_control": prosody_control.model_dump(),
            "emotion_tag": emotion_tag,
            "transition": transition.__dict__,
            "tri_head_outputs": {
                "memory": memory_output.__dict__,
                "story": story_output.__dict__,
                "control": control_output.__dict__
            },
            "generation_parameters": control_output.content.get("generation_parameters", {}),
            "adaptation_metadata": {
                "character_id": self.character_id,
                "timestamp": datetime.now().isoformat(),
                "interaction_count": len(self.interaction_history),
                "emotional_momentum": self.temporal_tracker.calculate_emotional_momentum()
            }
        }
        
        return response_params
    
    async def _apply_tri_head_influences(
        self,
        base_state: EmotionalState,
        memory_output: HeadOutput,
        story_output: HeadOutput,
        control_output: HeadOutput
    ) -> EmotionalState:
        """
        Apply influences from the tri-head architecture to the emotional state.
        
        Args:
            base_state: Base emotional state from temporal prediction
            memory_output: Output from memory head
            story_output: Output from story head
            control_output: Output from control head
            
        Returns:
            Modified emotional state with tri-head influences
        """
        # Start with base state
        influenced_state = EmotionalState(
            primary_emotion=base_state.primary_emotion,
            secondary_emotions=dict(base_state.secondary_emotions),
            intensity=base_state.intensity,
            narrative_tension=base_state.narrative_tension,
            scene_id=base_state.scene_id,
            trigger_event=base_state.trigger_event,
            timestamp=datetime.now()
        )
        
        # Apply memory influence
        if memory_output.content.get("character_growth"):
            growth = memory_output.content["character_growth"]
            confidence_modifier = growth.get("confidence_level", 0.5)
            
            # Adjust intensity based on character confidence
            intensity_adjustment = (confidence_modifier - 0.5) * self.config.memory_influence
            influenced_state.intensity = max(0.0, min(1.0, 
                influenced_state.intensity + intensity_adjustment
            ))
        
        # Apply story influence
        if story_output.content.get("suggested_emotional_tone"):
            suggested_tone = story_output.content["suggested_emotional_tone"]
            
            # Parse suggested tone (e.g., "determined_with_underlying_vulnerability")
            if "determined" in suggested_tone and influenced_state.primary_emotion != "determination":
                # Story suggests determination - blend it in
                if "determination" not in influenced_state.secondary_emotions:
                    influenced_state.secondary_emotions["determination"] = 0.3 * self.config.story_influence
            
            if "vulnerability" in suggested_tone:
                influenced_state.secondary_emotions["vulnerability"] = 0.2 * self.config.story_influence
        
        # Apply control influence
        if control_output.content.get("content_filters"):
            filters = control_output.content["content_filters"]
            intensity_cap = filters.get("emotional_intensity_cap", 1.0)
            
            # Cap intensity if control head suggests it
            if influenced_state.intensity > intensity_cap:
                influenced_state.intensity = intensity_cap * (1.0 - self.config.control_influence) + \
                                           influenced_state.intensity * self.config.control_influence
        
        # Clean up weak secondary emotions
        influenced_state.secondary_emotions = {
            k: v for k, v in influenced_state.secondary_emotions.items() 
            if v > 0.1
        }
        
        return influenced_state
    
    def get_character_emotional_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the character's emotional state and progression.
        
        Returns:
            Dictionary containing emotional summary
        """
        current_state = self.temporal_tracker.get_current_state()
        momentum = self.temporal_tracker.calculate_emotional_momentum()
        recent_arc = self.temporal_tracker.get_emotional_arc(lookback_minutes=15)
        
        return {
            "current_state": current_state.model_dump() if current_state else None,
            "emotional_momentum": momentum,
            "recent_emotional_arc": [state.model_dump() for state in recent_arc],
            "interaction_count": len(self.interaction_history),
            "character_id": self.character_id,
            "last_updated": datetime.now().isoformat()
        }
    
    async def adapt_voice_parameters(
        self,
        base_parameters: Dict[str, Any],
        current_context: NarrativeContext
    ) -> Dict[str, Any]:
        """
        Adapt voice parameters based on current emotional state and context.
        
        Args:
            base_parameters: Base voice generation parameters
            current_context: Current narrative context
            
        Returns:
            Adapted voice parameters
        """
        current_state = self.temporal_tracker.get_current_state()
        if not current_state:
            return base_parameters
        
        # Create emotion blend from current state
        emotion_blend = self.emotion_service.create_emotion_blend_from_state(current_state)
        
        # Calculate prosody
        prosody = self.emotion_service.calculate_prosody_from_context(
            emotion_blend,
            current_context
        )
        
        # Adapt parameters
        adapted_params = dict(base_parameters)
        
        # Adjust generation parameters based on emotional state
        if current_state.intensity > 0.7:
            adapted_params["temperature"] = min(1.0, adapted_params.get("temperature", 0.8) + 0.1)
        elif current_state.intensity < 0.3:
            adapted_params["temperature"] = max(0.1, adapted_params.get("temperature", 0.8) - 0.1)
        
        # Add prosody controls
        adapted_params["prosody_control"] = prosody.model_dump()
        adapted_params["emotion_tag"] = self.emotion_service.generate_emotion_tag(emotion_blend)
        
        return adapted_params 