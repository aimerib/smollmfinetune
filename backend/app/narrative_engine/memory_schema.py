"""
Memory Schema Definitions for Narrative Engine

Pydantic models for memory formation, storage, and retrieval in the 
dual-layer memory architecture.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Literal, Optional, Union, Any
from datetime import datetime, timezone
import numpy as np


class Turn(BaseModel):
    """Single conversation turn for context"""
    sender: Literal["user", "assistant", "system"]
    text: str
    timestamp: Optional[datetime] = None
    

class MemoryAnnotation(BaseModel):
    """
    Core memory annotation that supports both Method A (tokens) and Method B (vectors).
    Used for training data generation and evaluation.
    """
    # Context window that led to this memory
    conversation_window: List[Turn] = Field(
        description="Last N turns of conversation that contextualize this memory"
    )
    
    # Core memory content
    memory_content: str = Field(
        description="The actual memory to be stored (1-3 sentences)"
    )
    
    # Memory characteristics
    surprise_score: float = Field(
        ge=0, le=1,
        description="How unexpected/surprising this memory is (0=expected, 1=very surprising)"
    )
    
    emotional_valence: float = Field(
        ge=-1, le=1,
        description="Emotional tone (-1=negative, 0=neutral, 1=positive)"
    )
    
    importance: float = Field(
        ge=0, le=1,
        description="Overall importance/salience of this memory"
    )
    
    memory_type: Literal["episodic", "semantic", "emotional", "procedural"] = Field(
        description="Category of memory for retrieval optimization"
    )
    
    # Character-specific context
    character_id: str = Field(
        description="ID of the character forming this memory"
    )
    
    familiarity_score: float = Field(
        ge=-1, le=1, default=0,
        description="How familiar the character is with the interaction partner"
    )
    
    # Method A: Token approach
    method_a_tokens: List[str] = Field(
        default_factory=list,
        description="Control tokens for Method A (e.g., ['<memory_form>', '<importance_high>'])"
    )
    
    # Method B: Vector approach
    method_b_vector: Optional[List[float]] = Field(
        default=None,
        description="Dense vector representation for Method B memory head"
    )
    
    method_b_metadata: Optional[Dict[str, float]] = Field(
        default=None,
        description="Additional metadata for Method B (decay_rate, persistence_factor, etc.)"
    )
    
    # Persistence and decay
    decay_rate: float = Field(
        ge=0, le=1, default=0.1,
        description="How quickly this memory fades (adjusted by surprise)"
    )
    
    formation_strength: float = Field(
        ge=0, le=1, default=1.0,
        description="Initial strength of memory formation"
    )
    
    @field_validator('method_b_vector')
    def validate_vector_dimension(cls, v):
        if v is not None and len(v) != 768:  # Assuming 768-dim embeddings
            raise ValueError(f"Memory vector must be 768-dimensional, got {len(v)}")
        return v
    
    @field_validator('method_a_tokens')
    def validate_token_format(cls, v):
        for token in v:
            if not (token.startswith('<') and token.endswith('>')):
                raise ValueError(f"Control token must be in format <token>, got {token}")
        return v

    model_config = ConfigDict(extra='forbid')


class MemoryFormationEvent(BaseModel):
    """
    Real-time event emitted when a memory is formed during inference.
    Used for Director's View visualization.
    """
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    character_id: str
    memory: MemoryAnnotation
    
    # Visualization hints
    bubble_color: str = Field(
        default="",
        description="HSL color string for visualization"
    )
    bubble_size: float = Field(
        ge=0, le=100,
        description="Size of memory bubble in pixels"
    )
    animation_duration: float = Field(
        default=2.0,
        description="Seconds for formation animation"
    )
    
    @field_validator('bubble_color', mode='after')
    def generate_color_from_valence(cls, v, info):
        # Auto-generate if not provided
        if not v:
            memory = info.data.get('memory')
            if memory:
                valence = memory.emotional_valence
                # Map valence to hue: -1=0 (red), 0=60 (yellow), 1=120 (green)
                hue = 60 * (valence + 1)
                return f"hsl({hue}, 70%, 50%)"
        return v


class MemoryQuery(BaseModel):
    """Query structure for memory retrieval"""
    session_id: str
    query_text: str
    k: int = Field(default=5, description="Number of memories to retrieve")
    memory_types: Optional[List[Literal["episodic", "semantic", "emotional", "procedural"]]] = None
    recency_weight: float = Field(
        ge=0, le=1, default=0.3,
        description="How much to weight recent memories vs relevant ones"
    )
    importance_threshold: float = Field(
        ge=0, le=1, default=0.1,
        description="Minimum importance to consider"
    )


class MemoryRetrievalResult(BaseModel):
    """Result from memory retrieval operation"""
    memories: List[MemoryAnnotation]
    relevance_scores: List[float]
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debug info like query embedding, search time, etc."
    )


class MemoryTrainingBatch(BaseModel):
    """
    Batch of memories for training data augmentation.
    Includes both Method A and Method B labels.
    """
    conversation_id: str
    memories: List[MemoryAnnotation]
    
    # Method-specific training targets
    method_a_sequence: List[Dict[str, Union[str, List[str]]]] = Field(
        description="Sequence of tokens to insert in conversation"
    )
    
    method_b_targets: List[Dict[str, Union[List[float], Dict[str, float]]]] = Field(
        description="Target vectors and metadata for memory head"
    )
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to format expected by training pipeline"""
        return {
            'conversation_id': self.conversation_id,
            'memory_count': len(self.memories),
            'method_a_tokens': [m.method_a_tokens for m in self.memories],
            'method_b_vectors': [m.method_b_vector for m in self.memories],
            'importance_scores': [m.importance for m in self.memories],
            'surprise_scores': [m.surprise_score for m in self.memories],
        }


class EmotionalMomentumState(BaseModel):
    """
    Current emotional state with momentum tracking.
    Used for recirculation and Director's View.
    """
    active_emotions: Dict[str, float] = Field(
        description="Token -> strength mapping"
    )
    
    decay_states: Dict[str, Dict[str, float]] = Field(
        description="Token -> {strength, decay_rate, turns_remaining}"
    )
    
    recirculation_tokens: List[str] = Field(
        description="Tokens to inject in next turn"
    )
    
    surprise_history: List[float] = Field(
        default_factory=list,
        max_length=10,
        description="Recent surprise scores"
    )
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Format for Director's View"""
        return {
            'emotions': [
                {'token': k, 'strength': v, 'color': self._emotion_to_color(k)}
                for k, v in self.active_emotions.items()
            ],
            'average_surprise': np.mean(self.surprise_history) if self.surprise_history else 0,
            'emotional_volatility': np.std(self.surprise_history) if len(self.surprise_history) > 1 else 0,
        }
    
    @staticmethod
    def _emotion_to_color(token: str) -> str:
        """Map emotion tokens to colors for visualization"""
        color_map = {
            'happy': 'hsl(45, 100%, 50%)',    # Yellow
            'sad': 'hsl(210, 50%, 50%)',      # Blue
            'angry': 'hsl(0, 100%, 50%)',     # Red
            'nervous': 'hsl(270, 50%, 50%)',  # Purple
            'love': 'hsl(330, 100%, 50%)',    # Pink
            'curious': 'hsl(30, 100%, 50%)',  # Orange
        }
        
        # Extract base emotion from token like <mood_happy_1>
        for emotion, color in color_map.items():
            if emotion in token.lower():
                return color
        
        return 'hsl(0, 0%, 50%)'  # Gray for unknown


# Convenience types for API contracts
MemoryFormationRequest = MemoryAnnotation
MemoryUpdateRequest = MemoryAnnotation 