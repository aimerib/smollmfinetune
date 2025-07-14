"""
Personality-Driven Decision Making Engine

This module implements a sophisticated decision-making system that uses Big Five 
personality traits, emotional states, relationship contexts, and group dynamics 
to generate authentic character choices that feel psychologically consistent.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from .relationship_manager import RelationshipManager


@dataclass
class ChoiceOption:
    """Represents a single choice option with all its psychological attributes."""
    choice_id: str
    description: str
    predicted_outcomes: List[str]
    personality_alignment: Dict[str, float]  # Big Five alignment scores
    emotional_cost: Dict[str, float]
    relationship_impact: Dict[str, float]  # Impact on relationships
    moral_weight: Dict[str, float]  # Moral dimensions alignment
    group_acceptance: float = 0.5  # How group would view this choice


@dataclass 
class DecisionContext:
    """Complete context for a personality-driven decision."""
    agent_id: str
    available_choices: List[ChoiceOption]
    situation_description: str
    emotional_state: Dict[str, float]
    relationship_context: Dict[str, Any]  # Relevant relationships
    group_context: Optional[Any]  # Social group context
    time_pressure: float = 0.5  # 0.0 to 1.0
    stakes_level: float = 0.5   # 0.0 to 1.0 
    moral_dimensions: List[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    """Result of a personality-driven decision."""
    chosen_option: ChoiceOption
    confidence_score: float
    reasoning: str
    personality_factors: Dict[str, float]
    emotional_influences: Optional[Dict[str, float]] = None
    relationship_influences: Optional[Dict[str, float]] = None
    group_influences: Optional[Dict[str, float]] = None


@dataclass
class PersonalityWeights:
    """How each Big Five trait influences different decision types."""
    openness_weights: Dict[str, float] = field(default_factory=lambda: {
        'exploration': 0.8, 'creativity': 0.7, 'change_acceptance': 0.9,
        'novelty_seeking': 0.8, 'conventional_resistance': -0.6
    })
    conscientiousness_weights: Dict[str, float] = field(default_factory=lambda: {
        'duty': 0.9, 'planning': 0.8, 'persistence': 0.7,
        'responsibility': 0.9, 'impulsivity': -0.8
    })
    extraversion_weights: Dict[str, float] = field(default_factory=lambda: {
        'social': 0.8, 'assertive': 0.7, 'excitement_seeking': 0.6,
        'leadership': 0.7, 'solitude': -0.8
    })
    agreeableness_weights: Dict[str, float] = field(default_factory=lambda: {
        'cooperation': 0.9, 'trust': 0.8, 'altruism': 0.8,
        'compassion': 0.7, 'competition': -0.6
    })
    neuroticism_weights: Dict[str, float] = field(default_factory=lambda: {
        'anxiety': 0.8, 'stress_sensitivity': 0.7, 'emotional_volatility': 0.8,
        'risk_aversion': 0.6, 'stability': -0.9
    })


class PersonalityDecisionEngine:
    """Core engine for personality-driven decision making."""
    
    def __init__(self, triple_head_model, relationship_manager: RelationshipManager):
        self.model = triple_head_model
        self.relationship_manager = relationship_manager
        self.personality_weights = PersonalityWeights()
        self.decision_history: Dict[str, List] = {}
        
        # Import other components
        from .emotional_decision_modifiers import EmotionalDecisionModifier
        from .relationship_decision_influence import RelationshipDecisionInfluence
        from .decision_consistency_tracker import DecisionConsistencyTracker
        
        self.emotional_modifier = EmotionalDecisionModifier()
        self.relationship_influence = RelationshipDecisionInfluence(relationship_manager)
        self.consistency_tracker = DecisionConsistencyTracker()
        
    async def make_personality_driven_decision(self, context: DecisionContext) -> DecisionResult:
        """Generate a decision based on personality psychology."""
        
        # 1. Analyze agent's personality profile
        agent_personality = await self._get_agent_personality(context.agent_id)
        
        # 2. Score each choice against personality
        choice_scores = await self._score_choices_by_personality(
            context.available_choices, 
            agent_personality,
            context
        )
        
        # 3. Apply emotional modifiers
        choice_scores = await self._apply_emotional_influence(
            choice_scores, 
            context.emotional_state,
            agent_personality
        )
        
        # 4. Factor in relationship considerations
        choice_scores = await self._apply_relationship_influence(
            choice_scores,
            context.relationship_context,
            agent_personality
        )
        
        # 5. Account for group pressure/influence
        if context.group_context:
            choice_scores = await self._apply_group_influence(
                choice_scores,
                context.group_context,
                agent_personality
            )
        
        # 6. Generate final decision with reasoning
        decision = await self._generate_final_decision(
            choice_scores, 
            context,
            agent_personality
        )
        
        # 7. Record decision for personality consistency tracking
        await self._record_decision(context.agent_id, decision, context)
        
        return decision

    async def _get_agent_personality(self, agent_id: str) -> Dict[str, float]:
        """Get agent's Big Five personality profile."""
        # For now, return a default personality profile
        # In real implementation, this would fetch from character data
        return {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }

    async def _score_choices_by_personality(self, choices: List[ChoiceOption], 
                                          personality: Dict[str, float], 
                                          context: DecisionContext) -> Dict[str, float]:
        """Score each choice based on personality alignment."""
        
        scores = {}
        
        for choice in choices:
            personality_score = 0.0
            
            # Openness to Experience
            openness_level = personality.get('openness', 0.5)
            if openness_level > 0.6:
                # High openness: prefers novel, creative choices
                personality_score += choice.personality_alignment.get('openness_appeal', 0) * openness_level * 0.85
            else:
                # Low openness: prefers familiar, conventional choices
                personality_score += choice.personality_alignment.get('conventional_appeal', 0) * (1.0 - openness_level) * 0.8
                
            # Conscientiousness
            conscientiousness_level = personality.get('conscientiousness', 0.5)
            if conscientiousness_level > 0.6:
                # High conscientiousness: values duty, planning, responsibility
                personality_score += choice.personality_alignment.get('duty_alignment', 0) * conscientiousness_level * 0.4
                personality_score += choice.personality_alignment.get('long_term_benefit', 0) * conscientiousness_level * 0.3
            else:
                # Low conscientiousness: more impulsive, present-focused
                personality_score += choice.personality_alignment.get('immediate_gratification', 0) * (1.0 - conscientiousness_level) * 0.4
                
            # Extraversion
            extraversion_level = personality.get('extraversion', 0.5)
            if extraversion_level > 0.6:
                # High extraversion: social, assertive choices
                personality_score += choice.personality_alignment.get('social_engagement', 0) * extraversion_level * 0.4
                personality_score += choice.personality_alignment.get('assertiveness', 0) * extraversion_level * 0.3
            else:
                # Low extraversion (introversion): private, reflective choices
                personality_score += choice.personality_alignment.get('privacy_preference', 0) * (1.0 - extraversion_level) * 0.4
                
            # Agreeableness  
            agreeableness_level = personality.get('agreeableness', 0.5)
            if agreeableness_level > 0.6:
                # High agreeableness: cooperative, trusting, altruistic
                personality_score += choice.personality_alignment.get('cooperation', 0) * 0.2
                personality_score += choice.personality_alignment.get('altruism', 0) * 0.15
            else:
                # Low agreeableness: competitive, skeptical, self-interested
                personality_score += choice.personality_alignment.get('self_interest', 0) * 0.2
                personality_score += choice.personality_alignment.get('competitive_advantage', 0) * 0.15
                
            # Neuroticism
            neuroticism_level = personality.get('neuroticism', 0.5)
            if neuroticism_level > 0.6:
                # High neuroticism: anxiety-driven, risk-averse choices
                anxiety_cost = choice.emotional_cost.get('anxiety', 0)
                personality_score -= anxiety_cost * 0.3
                risk_level = choice.personality_alignment.get('risk_level', 0)
                personality_score -= risk_level * 0.2
            else:
                # Low neuroticism (emotional stability): calm, risk-tolerant
                risk_tolerance = choice.personality_alignment.get('risk_tolerance', 0)
                personality_score += risk_tolerance * 0.1
                
            scores[choice.choice_id] = max(0.0, min(1.0, personality_score))  # Clamp to [0,1]
            
        return scores

    async def _apply_group_influence(self, choice_scores: Dict[str, float], 
                                   group_context: Any, 
                                   personality: Dict[str, float]) -> Dict[str, float]:
        """Apply group pressure influence to choice scores."""
        # Simple implementation for now
        modified_scores = choice_scores.copy()
        
        # Group influence strength based on personality
        # High agreeableness = more susceptible to group pressure
        group_susceptibility = personality.get('agreeableness', 0.5) * 0.3
        
        for choice_id, score in choice_scores.items():
            # Find the choice option to get group_acceptance
            choice_option = None
            # Note: This is a simplified approach; in real implementation 
            # we'd need to pass the choice options or store them differently
            group_modifier = group_susceptibility * 0.2  # Placeholder
            modified_scores[choice_id] = max(0.0, min(1.0, score + group_modifier))
        
        return modified_scores

    async def _apply_emotional_influence(self, choice_scores: Dict[str, float], 
                                       emotional_state: Dict[str, float], 
                                       personality: Dict[str, float]) -> Dict[str, float]:
        """Apply emotional influence to choice scores."""
        return await self.emotional_modifier.apply_emotional_influence(
            choice_scores, emotional_state, personality
        )

    async def _apply_relationship_influence(self, choice_scores: Dict[str, float], 
                                          relationship_context: Dict[str, Any], 
                                          personality: Dict[str, float]) -> Dict[str, float]:
        """Apply relationship influence to choice scores."""
        return await self.relationship_influence.apply_relationship_influence(
            choice_scores, relationship_context, personality
        )

    async def _generate_final_decision(self, choice_scores: Dict[str, float], 
                                     context: DecisionContext,
                                     personality: Dict[str, float]) -> DecisionResult:
        """Generate final decision with reasoning."""
        
        # Find the highest scoring choice
        best_choice_id = max(choice_scores.keys(), key=lambda x: choice_scores[x])
        best_choice = next(c for c in context.available_choices if c.choice_id == best_choice_id)
        
        confidence = choice_scores[best_choice_id]
        
        # Generate reasoning based on dominant personality factors
        reasoning_parts = []
        
        # Identify dominant personality traits
        dominant_traits = [trait for trait, value in personality.items() if value > 0.6]
        
        if 'openness' in dominant_traits:
            reasoning_parts.append("high openness drives exploration and novelty-seeking")
        if 'conscientiousness' in dominant_traits:
            reasoning_parts.append("conscientiousness prioritizes duty and long-term planning")
        if 'extraversion' in dominant_traits:
            reasoning_parts.append("extraversion favors social engagement and assertive action")
        if 'agreeableness' in dominant_traits:
            reasoning_parts.append("agreeableness emphasizes cooperation and relationship harmony")
        if 'neuroticism' in dominant_traits:
            reasoning_parts.append("emotional sensitivity increases caution and risk awareness")
        
        if not reasoning_parts:
            reasoning_parts.append("balanced personality weighs multiple factors")
            
        reasoning = f"Decision driven by {', '.join(reasoning_parts)}"
        
        return DecisionResult(
            chosen_option=best_choice,
            confidence_score=confidence,
            reasoning=reasoning,
            personality_factors=personality
        )

    async def _record_decision(self, agent_id: str, decision: DecisionResult, context: DecisionContext):
        """Record decision for consistency tracking."""
        from .decision_consistency_tracker import DecisionRecord
        
        record = DecisionRecord(
            decision_id=f"{agent_id}_{datetime.now().isoformat()}",
            agent_id=agent_id,
            context=context,
            chosen_option=decision.chosen_option,
            personality_scores=decision.personality_factors,
            final_score=decision.confidence_score,
            reasoning=decision.reasoning,
            timestamp=datetime.now(),
            outcomes=[],  # Will be filled in later
            regret_level=0.0  # Will be calculated later
        )
        
        if agent_id not in self.decision_history:
            self.decision_history[agent_id] = []
        
        self.decision_history[agent_id].append(record) 