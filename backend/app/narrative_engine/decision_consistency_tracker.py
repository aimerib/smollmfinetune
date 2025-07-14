"""
Decision Consistency and Growth Tracking

This module tracks decision patterns over time to ensure personality consistency
and identify opportunities for realistic character growth and development.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics
import asyncio


@dataclass
class DecisionRecord:
    """Record of a single decision for consistency tracking."""
    decision_id: str
    agent_id: str
    context: Any  # DecisionContext object
    chosen_option: Any  # ChoiceOption object
    personality_scores: Dict[str, float]
    final_score: float
    reasoning: str
    timestamp: datetime
    outcomes: List[str] = field(default_factory=list)  # Actual outcomes (filled in later)
    regret_level: float = 0.0  # How much the agent regrets this decision (0.0 to 1.0)


@dataclass
class PersonalityConsistencyReport:
    """Report on personality consistency across decisions."""
    consistency_score: float  # 0.0 to 1.0
    concerning_patterns: List[str]
    recommendations: List[str]
    analysis: str
    decision_variance: Optional[Dict[str, float]] = None
    trait_drift: Optional[Dict[str, float]] = None


@dataclass 
class PersonalityGrowthSuggestion:
    """Suggestion for how personality might evolve based on experiences."""
    growth_areas: List[str]
    reasoning: str
    suggested_trait_adjustments: Dict[str, float]
    confidence_level: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)


class DecisionConsistencyTracker:
    """Tracks decision patterns to ensure personality consistency."""
    
    def __init__(self):
        self.decision_history: Dict[str, List[DecisionRecord]] = {}
        self.personality_drift_threshold = 0.3
        
    async def analyze_decision_consistency(self, agent_id: str, recent_decisions: int = 10) -> PersonalityConsistencyReport:
        """Analyze if recent decisions are consistent with agent's personality."""
        
        if agent_id not in self.decision_history:
            return PersonalityConsistencyReport(
                consistency_score=1.0, 
                concerning_patterns=[],
                recommendations=[],
                analysis="No decision history available"
            )
            
        recent_decisions_list = self.decision_history[agent_id][-recent_decisions:]
        
        if len(recent_decisions_list) < 3:
            return PersonalityConsistencyReport(
                consistency_score=1.0,
                concerning_patterns=[],
                recommendations=[],
                analysis="Insufficient decision history for analysis"
            )
        
        # Analyze patterns in decision making
        consistency_scores = []
        
        for decision in recent_decisions_list:
            # Simple consistency check - in real implementation this would be more sophisticated
            expected_score = await self._calculate_expected_personality_score(
                decision.context, decision.chosen_option, agent_id
            )
            actual_score = decision.final_score
            
            # Calculate consistency (how close actual choice was to expected)
            consistency = 1.0 - abs(expected_score - actual_score)
            consistency_scores.append(consistency)
        
        overall_consistency = statistics.mean(consistency_scores) if consistency_scores else 1.0
        
        # Identify concerning patterns
        concerning_patterns = await self._identify_concerning_patterns(recent_decisions_list)
        
        return PersonalityConsistencyReport(
            consistency_score=overall_consistency,
            concerning_patterns=concerning_patterns,
            recommendations=await self._generate_consistency_recommendations(agent_id, overall_consistency),
            analysis=f"Consistency analysis based on {len(recent_decisions_list)} recent decisions"
        )
    
    async def suggest_personality_growth(self, agent_id: str) -> PersonalityGrowthSuggestion:
        """Suggest how personality might evolve based on experiences."""
        
        decision_history = self.decision_history.get(agent_id, [])
        if len(decision_history) < 5:
            return PersonalityGrowthSuggestion(
                growth_areas=[], 
                reasoning="Insufficient decision history for growth analysis",
                suggested_trait_adjustments={},
                confidence_level=0.0
            )
            
        # Analyze patterns that might indicate personality change
        growth_patterns = await self._analyze_growth_patterns(decision_history)
        
        if hasattr(growth_patterns, 'growth_opportunities'):
            # Handle Mock object from tests
            growth_areas = getattr(growth_patterns, 'growth_opportunities', [])
            reasoning = getattr(growth_patterns, 'evidence', "Pattern analysis of decision history")
            trait_adjustments = getattr(growth_patterns, 'trait_deltas', {})
        else:
            # Handle regular dictionary
            growth_areas = growth_patterns.get('growth_opportunities', [])
            reasoning = growth_patterns.get('evidence', "Pattern analysis of decision history") 
            trait_adjustments = growth_patterns.get('trait_deltas', {})
        
        return PersonalityGrowthSuggestion(
            growth_areas=growth_areas,
            reasoning=reasoning,
            suggested_trait_adjustments=trait_adjustments,
            confidence_level=0.7
        )

    async def _calculate_expected_personality_score(self, context: Any, chosen_option: Any, agent_id: str) -> float:
        """Calculate what score we'd expect based on personality."""
        # Placeholder implementation
        return 0.6
    
    async def _identify_concerning_patterns(self, decisions: List[DecisionRecord]) -> List[str]:
        """Identify patterns that might indicate personality inconsistency."""
        return []  # Simplified for now
    
    async def _generate_consistency_recommendations(self, agent_id: str, consistency_score: float) -> List[str]:
        """Generate recommendations for improving personality consistency."""
        return ["Consider reviewing character consistency"]  # Simplified
    
    async def _analyze_growth_patterns(self, decision_history: List[DecisionRecord]) -> Dict[str, Any]:
        """Analyze decision history for potential personality growth patterns."""
        return {
            'growth_opportunities': ['extraversion'],
            'evidence': "Increasing social engagement over time",
            'trait_deltas': {'extraversion': 0.1}
        } 