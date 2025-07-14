"""
This module defines the data structures and service for analyzing narrative context
to inform emotion and prosody control in the TTS system.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, conlist, field_validator

from utils.openai_client import get_client, OpenAIClient

logger = logging.getLogger(__name__)


class EmotionalState(BaseModel):
    """
    Represents a character's emotional state at a specific point in time.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    primary_emotion: str = Field(..., description="The dominant emotion")
    secondary_emotions: Dict[str, float] = Field(
        default_factory=dict,
        description="Secondary emotions with intensity weights"
    )
    intensity: float = Field(..., description="Overall emotional intensity (0.0-1.0)")
    narrative_tension: float = Field(..., description="Narrative tension at this moment")
    scene_id: Optional[str] = Field(None, description="Identifier for the current scene")
    trigger_event: Optional[str] = Field(None, description="Event that triggered this emotional state")


class EmotionalTransition(BaseModel):
    """
    Represents a transition between two emotional states.
    """
    from_state: EmotionalState
    to_state: EmotionalState
    transition_type: str = Field(..., description="Type of transition (gradual, sudden, triggered)")
    smoothing_factor: float = Field(
        default=0.7,
        description="How smoothly to blend between states (0.0-1.0)"
    )
    duration_seconds: float = Field(
        default=2.0,
        description="Duration of the transition in seconds"
    )


class TemporalConsistencyTracker:
    """
    Tracks character emotional progression through the story and maintains consistency.
    """
    
    def __init__(self, character_id: str, max_history_length: int = 50):
        self.character_id = character_id
        self.max_history_length = max_history_length
        self.emotional_history: List[EmotionalState] = []
        self.current_scene_id: Optional[str] = None
        
    def add_emotional_state(self, state: EmotionalState) -> None:
        """
        Adds a new emotional state to the history.
        
        Args:
            state: The new emotional state to add
        """
        self.emotional_history.append(state)
        
        # Keep history within bounds
        if len(self.emotional_history) > self.max_history_length:
            self.emotional_history.pop(0)
            
        # Update current scene if provided
        if state.scene_id:
            self.current_scene_id = state.scene_id
    
    def get_current_state(self) -> Optional[EmotionalState]:
        """
        Gets the most recent emotional state.
        
        Returns:
            The current emotional state or None if no history exists
        """
        if not self.emotional_history:
            return None
        return self.emotional_history[-1]
    
    def get_emotional_arc(self, lookback_minutes: int = 30) -> List[EmotionalState]:
        """
        Gets the emotional arc over a specified time period.
        
        Args:
            lookback_minutes: How far back to look in minutes
            
        Returns:
            List of emotional states within the time window
        """
        if not self.emotional_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        return [
            state for state in self.emotional_history
            if state.timestamp >= cutoff_time
        ]
    
    def calculate_emotional_momentum(self) -> Dict[str, float]:
        """
        Calculates the emotional momentum - how emotions are trending.
        
        Returns:
            Dictionary mapping emotions to their momentum scores (-1.0 to 1.0)
        """
        if len(self.emotional_history) < 2:
            return {}
        
        # Look at recent states (last 5 or all if fewer)
        recent_states = self.emotional_history[-5:]
        
        momentum = {}
        
        # Track primary emotion changes
        primary_emotions = [state.primary_emotion for state in recent_states]
        if len(set(primary_emotions)) == 1:
            # Stable primary emotion
            momentum[primary_emotions[0]] = 0.0
        else:
            # Calculate trend for each emotion
            for emotion in set(primary_emotions):
                positions = [i for i, e in enumerate(primary_emotions) if e == emotion]
                if positions:
                    # Simple trend calculation based on position
                    avg_position = sum(positions) / len(positions)
                    normalized_position = (avg_position / (len(primary_emotions) - 1)) * 2 - 1
                    momentum[emotion] = normalized_position
        
        # Track intensity momentum
        intensities = [state.intensity for state in recent_states]
        if len(intensities) >= 2:
            intensity_trend = (intensities[-1] - intensities[0]) / (len(intensities) - 1)
            momentum['_intensity_trend'] = intensity_trend
        
        return momentum
    
    def predict_next_emotional_state(
        self, 
        narrative_context: 'NarrativeContext',
        emotion_blending_service: 'EmotionBlendingService'
    ) -> EmotionalState:
        """
        Predicts the next emotional state based on history and current context.
        
        Args:
            narrative_context: Current narrative context
            emotion_blending_service: Service for creating emotion blends
            
        Returns:
            Predicted emotional state
        """
        current_state = self.get_current_state()
        momentum = self.calculate_emotional_momentum()
        
        # If no history, use context directly
        if not current_state:
            return EmotionalState(
                primary_emotion=narrative_context.primary_emotion,
                secondary_emotions={
                    emotion: 0.3 for emotion in narrative_context.secondary_emotions or []
                },
                intensity=narrative_context.narrative_tension,
                narrative_tension=narrative_context.narrative_tension,
                scene_id=self.current_scene_id
            )
        
        # Calculate emotional inertia - tendency to maintain current emotion
        inertia_factor = 0.6  # How much the current state influences the next
        
        # Blend current state with new context
        if current_state.primary_emotion == narrative_context.primary_emotion:
            # Same primary emotion - maintain with slight adjustment
            new_intensity = (
                current_state.intensity * inertia_factor +
                narrative_context.narrative_tension * (1 - inertia_factor)
            )
            
            # Merge secondary emotions
            new_secondary = dict(current_state.secondary_emotions)
            for emotion in narrative_context.secondary_emotions or []:
                if emotion in new_secondary:
                    new_secondary[emotion] = min(1.0, new_secondary[emotion] + 0.1)
                else:
                    new_secondary[emotion] = 0.2
                    
        else:
            # Different primary emotion - transition
            transition_strength = 1.0 - inertia_factor
            
            # Check if this is a dramatic shift
            dramatic_emotions = {'fear', 'anger', 'desperation', 'joy'}
            if (narrative_context.primary_emotion in dramatic_emotions or
                current_state.primary_emotion in dramatic_emotions):
                transition_strength = 0.8  # Faster transition for dramatic emotions
            
            new_intensity = (
                current_state.intensity * (1 - transition_strength) +
                narrative_context.narrative_tension * transition_strength
            )
            
            # Gradual transition of secondary emotions
            new_secondary = {}
            for emotion, weight in current_state.secondary_emotions.items():
                new_secondary[emotion] = weight * (1 - transition_strength)
            
            for emotion in narrative_context.secondary_emotions or []:
                if emotion in new_secondary:
                    new_secondary[emotion] += 0.3 * transition_strength
                else:
                    new_secondary[emotion] = 0.3 * transition_strength
        
        # Clean up weak secondary emotions
        new_secondary = {k: v for k, v in new_secondary.items() if v > 0.1}
        
        return EmotionalState(
            primary_emotion=narrative_context.primary_emotion,
            secondary_emotions=new_secondary,
            intensity=max(0.0, min(1.0, new_intensity)),
            narrative_tension=narrative_context.narrative_tension,
            scene_id=self.current_scene_id
        )
    
    def create_smooth_transition(
        self, 
        target_state: EmotionalState,
        transition_duration: float = 2.0
    ) -> EmotionalTransition:
        """
        Creates a smooth transition to a target emotional state.
        
        Args:
            target_state: The target emotional state
            transition_duration: Duration of transition in seconds
            
        Returns:
            EmotionalTransition object
        """
        current_state = self.get_current_state()
        
        if not current_state:
            # No current state, immediate transition
            return EmotionalTransition(
                from_state=target_state,  # Use target as both from and to
                to_state=target_state,
                transition_type="immediate",
                smoothing_factor=1.0,
                duration_seconds=0.0
            )
        
        # Determine transition type
        emotion_distance = 0.0 if current_state.primary_emotion == target_state.primary_emotion else 1.0
        intensity_distance = abs(current_state.intensity - target_state.intensity)
        
        if emotion_distance > 0.5 or intensity_distance > 0.4:
            transition_type = "sudden"
            smoothing_factor = 0.3
        elif emotion_distance > 0.2 or intensity_distance > 0.2:
            transition_type = "gradual"
            smoothing_factor = 0.7
        else:
            transition_type = "subtle"
            smoothing_factor = 0.9
        
        return EmotionalTransition(
            from_state=current_state,
            to_state=target_state,
            transition_type=transition_type,
            smoothing_factor=smoothing_factor,
            duration_seconds=transition_duration
        )


class EmotionBlend(BaseModel):
    """
    Represents a blended emotion with multiple components and intensities.
    """
    primary_emotion: str = Field(..., description="The dominant emotion")
    secondary_emotions: Dict[str, float] = Field(
        default_factory=dict,
        description="Secondary emotions with their intensity weights (0.0-1.0)"
    )
    overall_intensity: float = Field(
        default=1.0,
        description="Overall emotional intensity (0.0-1.0)"
    )
    
    @field_validator('secondary_emotions')
    @classmethod
    def validate_emotion_weights(cls, v):
        """Ensure all emotion weights are between 0.0 and 1.0."""
        for emotion, weight in v.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Emotion weight for '{emotion}' must be between 0.0 and 1.0")
        return v


class ProsodyControl(BaseModel):
    """
    Controls for advanced prosody manipulation.
    """
    speaking_rate: float = Field(
        default=1.0,
        description="Speaking rate multiplier (0.5-2.0, 1.0 is normal)"
    )
    pause_duration: float = Field(
        default=0.0,
        description="Additional pause duration in seconds"
    )
    emphasis_words: List[str] = Field(
        default_factory=list,
        description="Words to emphasize in speech"
    )
    pitch_variation: float = Field(
        default=1.0,
        description="Pitch variation multiplier (0.5-2.0, 1.0 is normal)"
    )
    
    @field_validator('speaking_rate', 'pitch_variation')
    @classmethod
    def validate_rate_range(cls, v):
        """Ensure rates are within reasonable bounds."""
        if not 0.5 <= v <= 2.0:
            raise ValueError("Rate must be between 0.5 and 2.0")
        return v


class NarrativeContext(BaseModel):
    """
    A structured representation of the current narrative and emotional context.
    """
    narrative_tension: float = Field(
        ...,
        description="A score from 0.0 (calm) to 1.0 (intense climax) representing the story's tension."
    )
    character_arc_stage: str = Field(
        ...,
        description="The character's current stage in their narrative arc (e.g., 'inciting_incident', 'rising_action', 'climax', 'falling_action', 'resolution')."
    )
    scene_atmosphere: str = Field(
        ...,
        description="The overall atmosphere or mood of the current scene (e.g., 'tense', 'comedic', 'somber', 'chaotic')."
    )
    dialogue_context: str = Field(
        ...,
        description="The immediate context of the conversation (e.g., 'friendly_banter', 'heated_argument', 'solemn_confession', 'desperate_plea')."
    )
    primary_emotion: str = Field(
        ...,
        description="The dominant emotion the character should be feeling."
    )
    secondary_emotions: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of one or two secondary emotions that add complexity."
    )


class EmotionBlendingService:
    """
    Service for creating complex emotion blends and prosody controls.
    """
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        self._client = client

    @property
    def client(self) -> OpenAIClient:
        """Lazily gets the OpenAI client."""
        if self._client is None:
            self._client = get_client()
        return self._client

    def create_emotion_blend(
        self, 
        primary_emotion: str, 
        secondary_emotions: List[str], 
        narrative_tension: float = 0.5
    ) -> EmotionBlend:
        """
        Creates a blended emotion from primary and secondary emotions.
        
        Args:
            primary_emotion: The dominant emotion
            secondary_emotions: List of secondary emotions
            narrative_tension: Current narrative tension (affects intensity)
            
        Returns:
            EmotionBlend object with calculated weights
        """
        # Calculate secondary emotion weights based on narrative tension
        secondary_weights = {}
        base_weight = 0.3  # Base weight for secondary emotions
        
        for i, emotion in enumerate(secondary_emotions):
            # Decrease weight for each additional secondary emotion
            weight = base_weight * (0.8 ** i)
            # Adjust based on narrative tension
            weight *= (0.5 + narrative_tension * 0.5)
            secondary_weights[emotion] = min(weight, 1.0)
        
        # Overall intensity influenced by narrative tension
        overall_intensity = 0.6 + (narrative_tension * 0.4)
        
        return EmotionBlend(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_weights,
            overall_intensity=overall_intensity
        )

    def create_emotion_blend_from_state(self, emotional_state: EmotionalState) -> EmotionBlend:
        """
        Creates an emotion blend from an emotional state.
        
        Args:
            emotional_state: The emotional state to convert
            
        Returns:
            EmotionBlend object
        """
        return EmotionBlend(
            primary_emotion=emotional_state.primary_emotion,
            secondary_emotions=emotional_state.secondary_emotions,
            overall_intensity=emotional_state.intensity
        )

    def calculate_prosody_from_context(
        self, 
        emotion_blend: EmotionBlend, 
        narrative_context: NarrativeContext
    ) -> ProsodyControl:
        """
        Calculates prosody controls based on emotion blend and narrative context.
        
        Args:
            emotion_blend: The blended emotion
            narrative_context: Current narrative context
            
        Returns:
            ProsodyControl object with calculated parameters
        """
        # Base prosody settings
        speaking_rate = 1.0
        pause_duration = 0.0
        pitch_variation = 1.0
        
        # Adjust based on primary emotion
        emotion_adjustments = {
            'excitement': {'speaking_rate': 1.3, 'pitch_variation': 1.4},
            'fear': {'speaking_rate': 1.2, 'pitch_variation': 1.3, 'pause_duration': 0.2},
            'sadness': {'speaking_rate': 0.8, 'pitch_variation': 0.7, 'pause_duration': 0.3},
            'anger': {'speaking_rate': 1.1, 'pitch_variation': 1.2},
            'determination': {'speaking_rate': 0.9, 'pitch_variation': 1.1},
            'desperation': {'speaking_rate': 1.4, 'pitch_variation': 1.5, 'pause_duration': 0.1},
            'calm': {'speaking_rate': 0.9, 'pitch_variation': 0.8},
            'confusion': {'speaking_rate': 0.8, 'pitch_variation': 1.2, 'pause_duration': 0.4}
        }
        
        if emotion_blend.primary_emotion in emotion_adjustments:
            adjustments = emotion_adjustments[emotion_blend.primary_emotion]
            speaking_rate = adjustments.get('speaking_rate', 1.0)
            pitch_variation = adjustments.get('pitch_variation', 1.0)
            pause_duration = adjustments.get('pause_duration', 0.0)
        
        # Apply intensity scaling
        intensity_factor = emotion_blend.overall_intensity
        speaking_rate = 1.0 + (speaking_rate - 1.0) * intensity_factor
        pitch_variation = 1.0 + (pitch_variation - 1.0) * intensity_factor
        pause_duration *= intensity_factor
        
        # Adjust based on narrative tension
        tension_factor = narrative_context.narrative_tension
        if tension_factor > 0.7:  # High tension
            speaking_rate *= 1.1
            pitch_variation *= 1.1
            pause_duration *= 0.7
        elif tension_factor < 0.3:  # Low tension
            speaking_rate *= 0.95
            pitch_variation *= 0.9
            pause_duration *= 1.2
        
        # Clamp values to valid ranges
        speaking_rate = max(0.5, min(2.0, speaking_rate))
        pitch_variation = max(0.5, min(2.0, pitch_variation))
        pause_duration = max(0.0, min(2.0, pause_duration))
        
        return ProsodyControl(
            speaking_rate=speaking_rate,
            pause_duration=pause_duration,
            pitch_variation=pitch_variation
        )

    def generate_emotion_tag(self, emotion_blend: EmotionBlend) -> str:
        """
        Generates a formatted emotion tag for TTS systems.
        
        Args:
            emotion_blend: The blended emotion
            
        Returns:
            Formatted emotion tag string
        """
        if not emotion_blend.secondary_emotions:
            return f"<{emotion_blend.primary_emotion}>"
        
        # Create complex emotion tag
        secondary_parts = []
        for emotion, weight in emotion_blend.secondary_emotions.items():
            if weight > 0.2:  # Only include significant secondary emotions
                secondary_parts.append(f"{emotion} ({weight:.1f})")
        
        if secondary_parts:
            secondary_str = ", ".join(secondary_parts)
            return f"<{emotion_blend.primary_emotion} with {secondary_str}>"
        else:
            return f"<{emotion_blend.primary_emotion}>"


class NarrativeContextService:
    """
    A service for analyzing dialogue and story state to produce a structured narrative context.
    """

    def __init__(self, client: Optional[OpenAIClient] = None):
        """
        Initializes the service.
        
        Args:
            client: An OpenAIClient instance. If not provided, one will be retrieved globally.
        """
        self._client = client

    @property
    def client(self) -> OpenAIClient:
        """Lazily gets the OpenAI client."""
        if self._client is None:
            self._client = get_client()
        return self._client

    async def analyze_context(self, dialogue_history: List[Dict[str, str]]) -> NarrativeContext:
        """
        Analyzes a snippet of dialogue to extract narrative and emotional context.

        Args:
            dialogue_history: A list of message dicts, e.g., [{"role": "user", "content": "..."}].

        Returns:
            A populated NarrativeContext object.
        """
        formatted_dialogue = "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in dialogue_history)

        prompt = f"""
You are a narrative analysis engine. Your task is to analyze the following dialogue snippet and determine the underlying narrative and emotional context. Provide your analysis in a structured JSON format.

Dialogue:
---
{formatted_dialogue}
---

Analyze the dialogue and determine the following:
- narrative_tension: A score from 0.0 to 1.0 for the scene's tension.
- character_arc_stage: The character's current arc stage.
- scene_atmosphere: The overall mood of the scene.
- dialogue_context: The immediate purpose of the conversation.
- primary_emotion: The dominant emotion of the last speaker ("Assistant").
- secondary_emotions: One or two other emotions present.

Return ONLY the JSON object that conforms to the specified schema.
"""
        
        try:
            response_text = await self.client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "narrative_context_analysis",
                        "schema": NarrativeContext.model_json_schema()
                    }
                }
            )
            
            response_data = json.loads(response_text)
            context = NarrativeContext(**response_data)
            return context
            
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM response for narrative context.")
            # In a real scenario, we might have more robust fallback mechanisms
            raise ValueError("LLM response was not valid JSON.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during narrative context analysis: {e}")
            raise 