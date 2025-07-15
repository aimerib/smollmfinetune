"""
Quality Models for Adaptive Voice Generation

Defines models for quality levels, narrative contexts, and quality decisions
that drive adaptive voice generation based on narrative importance.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class QualityLevel(Enum):
    """Voice generation quality levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ULTRA = 4
    
    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class NarrativeContext:
    """Context information for determining voice generation quality"""
    
    tension_level: float  # 0.0 to 1.0
    emotional_intensity: float  # 0.0 to 1.0  
    narrative_importance: float  # 0.0 to 1.0
    dialogue_type: str  # "casual", "dramatic", "urgent", "emotional", etc.
    scene_type: str  # "conversation", "action", "climax", "intimate", etc.
    character_focus: str  # "protagonist", "background", "two_characters", etc.
    
    def is_high_priority(self) -> bool:
        """Determine if this context warrants high quality generation"""
        # High priority if multiple factors are elevated
        high_factors = sum([
            self.tension_level > 0.7,
            self.emotional_intensity > 0.7,
            self.narrative_importance > 0.6,
            self.dialogue_type in ["dramatic", "urgent", "emotional"],
            self.scene_type in ["climax", "crisis", "dramatic_revelation"],
            self.character_focus in ["protagonist", "main_character"]
        ])
        
        return high_factors >= 3


@dataclass
class QualityDecision:
    """Decision result for voice generation quality"""
    
    quality_level: QualityLevel
    reasoning: str
    confidence_score: float  # 0.0 to 1.0
    performance_impact: float  # 0.0 to 1.0 (expected resource usage)
    estimated_latency_ms: int
    cached_decision: bool = False
    performance_consideration_applied: bool = False
    
    def should_use_cache(self) -> bool:
        """Determine if this decision should be cached"""
        return not self.cached_decision and self.confidence_score > 0.7
    
    def is_acceptable_latency(self, max_latency_ms: int) -> bool:
        """Check if estimated latency is within acceptable bounds"""
        return self.estimated_latency_ms <= max_latency_ms


@dataclass
class OptimizedTTSParameters:
    """Optimized TTS parameters based on quality level"""
    
    quality_level: QualityLevel
    sample_rate: int
    inference_steps: int
    temperature: float
    use_advanced_features: bool
    batch_size: int = 1
    use_gpu_optimization: bool = True
    memory_limit_mb: Optional[int] = None
    
    @classmethod
    def from_quality_level(cls, quality_level: QualityLevel, base_config: Dict[str, Any]) -> 'OptimizedTTSParameters':
        """Create optimized parameters based on quality level"""
        quality_mappings = {
            QualityLevel.LOW: {
                "sample_rate": 22050,
                "inference_steps": 20,
                "temperature": 0.9,
                "use_advanced_features": False
            },
            QualityLevel.MEDIUM: {
                "sample_rate": 44100,
                "inference_steps": 35,
                "temperature": 0.8,
                "use_advanced_features": False
            },
            QualityLevel.HIGH: {
                "sample_rate": 44100,
                "inference_steps": 50,
                "temperature": 0.7,
                "use_advanced_features": True
            },
            QualityLevel.ULTRA: {
                "sample_rate": 48000,
                "inference_steps": 80,
                "temperature": 0.6,
                "use_advanced_features": True
            }
        }
        
        params = quality_mappings[quality_level]
        
        return cls(
            quality_level=quality_level,
            sample_rate=params["sample_rate"],
            inference_steps=params["inference_steps"],
            temperature=params["temperature"],
            use_advanced_features=params["use_advanced_features"]
        )


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics for quality adaptation"""
    
    current_load: float  # 0.0 to 1.0
    average_latency_ms: float
    memory_usage_mb: float
    gpu_utilization: float
    active_requests: int
    queue_depth: int
    
    def is_under_load(self, threshold: float = 0.8) -> bool:
        """Check if system is under heavy load"""
        return self.current_load > threshold or self.gpu_utilization > threshold


class CharacterImportance(Enum):
    """Character importance levels for quality decisions"""
    BACKGROUND = 1
    SUPPORTING = 2  
    MAIN = 3
    PROTAGONIST = 4
    
    @classmethod
    def from_character_id(cls, character_id: str) -> 'CharacterImportance':
        """Determine character importance from ID"""
        if "protagonist" in character_id.lower() or "main" in character_id.lower():
            return cls.PROTAGONIST
        elif "background" in character_id.lower() or "npc" in character_id.lower():
            return cls.BACKGROUND
        elif "guard" in character_id.lower() or "crowd" in character_id.lower():
            return cls.BACKGROUND
        else:
            return cls.SUPPORTING  # Default for unknown characters 