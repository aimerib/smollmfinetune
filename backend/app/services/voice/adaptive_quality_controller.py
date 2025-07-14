"""
Adaptive Quality Controller

Main controller for adaptive voice generation quality. Makes intelligent
quality decisions based on narrative context, character importance, and
system performance metrics.
"""

import asyncio
import time
import hashlib
from typing import Dict, Any, Optional
from backend.app.services.voice.quality_models import (
    QualityLevel, NarrativeContext, QualityDecision, OptimizedTTSParameters,
    PerformanceMetrics, CharacterImportance
)
from backend.app.services.voice.narrative_quality_analyzer import NarrativeQualityAnalyzer


class PerformanceTracker:
    """Simple performance tracker for quality decisions"""
    
    def __init__(self):
        self.current_load = 0.0
        self.average_latency_ms = 100.0
        self.active_requests = 0
        
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return PerformanceMetrics(
            current_load=self.current_load,
            average_latency_ms=self.average_latency_ms,
            memory_usage_mb=512.0,  # Mock value
            gpu_utilization=self.current_load,  # Approximate
            active_requests=self.active_requests,
            queue_depth=max(0, self.active_requests - 2)
        )


class AdaptiveQualityController:
    """Adaptive quality controller for voice generation"""
    
    def __init__(
        self,
        default_quality: QualityLevel = QualityLevel.MEDIUM,
        adaptation_threshold: float = 0.1,
        performance_weight: float = 0.3
    ):
        """
        Initialize the adaptive quality controller
        
        Args:
            default_quality: Default quality level when no context is available
            adaptation_threshold: Minimum change required to adapt quality
            performance_weight: Weight of performance considerations (0.0 to 1.0)
        """
        self.default_quality = default_quality
        self.adaptation_threshold = adaptation_threshold
        self.performance_weight = performance_weight
        
        # Initialize components
        self.narrative_analyzer = NarrativeQualityAnalyzer()
        self.performance_tracker = PerformanceTracker()
        
        # Caching for similar contexts
        self.decision_cache: Dict[str, QualityDecision] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
    
    async def determine_quality_level(
        self,
        narrative_context: NarrativeContext,
        character_id: str,
        current_load: Optional[float] = None
    ) -> QualityDecision:
        """
        Determine appropriate quality level for voice generation
        
        Args:
            narrative_context: Current narrative context
            character_id: ID of the character speaking
            current_load: Current system load (0.0 to 1.0)
            
        Returns:
            QualityDecision with reasoning and parameters
        """
        # Update performance tracker if load provided
        if current_load is not None:
            self.performance_tracker.current_load = current_load
        
        # Check cache first
        cache_key = self._generate_cache_key(narrative_context, character_id, current_load)
        cached_decision = self._get_cached_decision(cache_key)
        if cached_decision:
            cached_decision.cached_decision = True
            return cached_decision
        
        # Calculate base quality from narrative context
        importance_score = await self.narrative_analyzer.calculate_importance_score(narrative_context)
        character_importance = CharacterImportance.from_character_id(character_id)
        
        # Determine base quality level
        base_quality = self._calculate_base_quality(importance_score, character_importance, narrative_context)
        
        # Apply performance considerations
        final_quality, performance_applied = self._apply_performance_adjustments(
            base_quality, current_load or 0.0
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            narrative_context, character_importance, importance_score, performance_applied
        )
        
        # Calculate confidence and performance impact
        confidence_score = self._calculate_confidence(narrative_context, character_importance)
        performance_impact = self._estimate_performance_impact(final_quality)
        estimated_latency = self._estimate_latency(final_quality)
        
        # Create decision
        decision = QualityDecision(
            quality_level=final_quality,
            reasoning=reasoning,
            confidence_score=confidence_score,
            performance_impact=performance_impact,
            estimated_latency_ms=estimated_latency,
            cached_decision=False,
            performance_consideration_applied=performance_applied
        )
        
        # Cache the decision
        self._cache_decision(cache_key, decision)
        
        return decision
    
    async def get_optimized_tts_parameters(
        self,
        narrative_context: NarrativeContext,
        character_id: str,
        base_voice_config: Dict[str, Any]
    ) -> OptimizedTTSParameters:
        """
        Get optimized TTS parameters based on quality decision
        
        Args:
            narrative_context: Current narrative context
            character_id: ID of the character speaking
            base_voice_config: Base voice configuration
            
        Returns:
            OptimizedTTSParameters for TTS generation
        """
        quality_decision = await self.determine_quality_level(
            narrative_context, character_id
        )
        
        return OptimizedTTSParameters.from_quality_level(
            quality_decision.quality_level, base_voice_config
        )
    
    def _calculate_base_quality(
        self,
        importance_score: float,
        character_importance: CharacterImportance,
        context: NarrativeContext
    ) -> QualityLevel:
        """Calculate base quality level from context and character importance"""
        
        # Character importance bonus
        character_bonus = {
            CharacterImportance.BACKGROUND: 0.0,
            CharacterImportance.SUPPORTING: 0.1,
            CharacterImportance.MAIN: 0.2,
            CharacterImportance.PROTAGONIST: 0.3
        }
        
        adjusted_score = importance_score + character_bonus[character_importance]
        
        # High priority contexts get boosted
        if context.is_high_priority():
            adjusted_score += 0.2
        
        # Map to quality levels
        if adjusted_score >= 0.8:
            return QualityLevel.HIGH
        elif adjusted_score >= 0.6:
            return QualityLevel.MEDIUM
        elif adjusted_score >= 0.3:
            return QualityLevel.MEDIUM
        else:
            return QualityLevel.LOW
    
    def _apply_performance_adjustments(
        self,
        base_quality: QualityLevel,
        current_load: float
    ) -> tuple[QualityLevel, bool]:
        """Apply performance-based adjustments to quality level"""
        
        # No adjustment needed if load is reasonable
        if current_load < 0.7:
            return base_quality, False
        
        # High load - reduce quality
        if current_load > 0.9:
            # Severe load - drop to low quality
            if base_quality == QualityLevel.ULTRA:
                return QualityLevel.MEDIUM, True
            elif base_quality == QualityLevel.HIGH:
                return QualityLevel.MEDIUM, True
            elif base_quality == QualityLevel.MEDIUM:
                return QualityLevel.LOW, True
            # LOW stays LOW
            return base_quality, True
        
        elif current_load > 0.8:
            # Moderate load - reduce by one level
            if base_quality == QualityLevel.ULTRA:
                return QualityLevel.HIGH, True
            elif base_quality == QualityLevel.HIGH:
                return QualityLevel.MEDIUM, True
            # MEDIUM and LOW stay the same
            return base_quality, True
        
        else:
            # Light load - minor reduction
            if base_quality == QualityLevel.ULTRA:
                return QualityLevel.HIGH, True
            return base_quality, True
    
    def _generate_reasoning(
        self,
        context: NarrativeContext,
        character_importance: CharacterImportance,
        importance_score: float,
        performance_applied: bool
    ) -> str:
        """Generate human-readable reasoning for the quality decision"""
        
        reasons = []
        
        # Narrative factors
        if context.narrative_importance > 0.7:
            reasons.append("High narrative importance")
        elif context.narrative_importance < 0.3:
            reasons.append("Low narrative importance")
        
        if context.emotional_intensity > 0.7:
            reasons.append("high emotional intensity")
        
        if context.tension_level > 0.7:
            reasons.append("high tension")
        
        # Character factors
        if character_importance == CharacterImportance.PROTAGONIST:
            reasons.append("protagonist character")
        elif character_importance == CharacterImportance.BACKGROUND:
            reasons.append("background character")
        
        # Performance factors
        if performance_applied:
            reasons.append("adjusted for performance load")
        
        # Combine reasons
        if reasons:
            if len(reasons) == 1:
                return reasons[0].capitalize()
            elif len(reasons) == 2:
                return f"{reasons[0].capitalize()} and {reasons[1]}"
            else:
                return f"{', '.join(reasons[:-1]).capitalize()}, and {reasons[-1]}"
        
        return "Standard quality assessment"
    
    def _calculate_confidence(
        self,
        context: NarrativeContext,
        character_importance: CharacterImportance
    ) -> float:
        """Calculate confidence score for the quality decision"""
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence for clear indicators
        if context.is_high_priority():
            confidence += 0.3
        
        if character_importance in [CharacterImportance.PROTAGONIST, CharacterImportance.MAIN]:
            confidence += 0.2
        
        if context.dialogue_type in ["dramatic", "urgent", "emotional"]:
            confidence += 0.2
        
        if context.scene_type in ["climax", "crisis", "action"]:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _estimate_performance_impact(self, quality_level: QualityLevel) -> float:
        """Estimate performance impact of the quality level"""
        impact_map = {
            QualityLevel.LOW: 0.2,
            QualityLevel.MEDIUM: 0.5,
            QualityLevel.HIGH: 0.8,
            QualityLevel.ULTRA: 1.0
        }
        return impact_map[quality_level]
    
    def _estimate_latency(self, quality_level: QualityLevel) -> int:
        """Estimate latency in milliseconds for the quality level"""
        latency_map = {
            QualityLevel.LOW: 300,
            QualityLevel.MEDIUM: 600,
            QualityLevel.HIGH: 1200,
            QualityLevel.ULTRA: 2000
        }
        return latency_map[quality_level]
    
    def _generate_cache_key(
        self,
        context: NarrativeContext,
        character_id: str,
        current_load: Optional[float]
    ) -> str:
        """Generate cache key for similar contexts"""
        # Create a normalized representation
        context_str = f"{context.tension_level:.1f}:{context.emotional_intensity:.1f}:{context.narrative_importance:.1f}:{context.dialogue_type}:{context.scene_type}:{context.character_focus}"
        load_str = f"{current_load:.1f}" if current_load else "0.0"
        full_key = f"{character_id}:{context_str}:{load_str}"
        
        return hashlib.md5(full_key.encode()).hexdigest()[:16]
    
    def _get_cached_decision(self, cache_key: str) -> Optional[QualityDecision]:
        """Get cached decision if valid and recent"""
        if cache_key not in self.decision_cache:
            return None
        
        # Check if cache entry is still valid
        timestamp = self.cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self.cache_ttl:
            # Expired - remove from cache
            del self.decision_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        return self.decision_cache[cache_key]
    
    def _cache_decision(self, cache_key: str, decision: QualityDecision):
        """Cache a quality decision"""
        if decision.should_use_cache():
            self.decision_cache[cache_key] = decision
            self.cache_timestamps[cache_key] = time.time()
            
            # Basic cache cleanup - remove oldest entries if cache gets too large
            if len(self.decision_cache) > 100:
                oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
                del self.decision_cache[oldest_key]
                del self.cache_timestamps[oldest_key] 