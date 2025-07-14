"""
Relationship-Influenced Decisions

This module factors relationship considerations into character decision making.
Relationships can strongly influence choices based on affinity, history, and
the character's personality traits like agreeableness.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

from .relationship_manager import RelationshipManager


@dataclass
class EnhancedRelationship:
    """Enhanced relationship data for decision making."""
    agent_id: str
    other_id: str
    status: str
    affinity: float
    emotional_history: List[str]
    memory_significance: float
    recent_interactions: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedRelationship':
        """Create from dictionary data."""
        return cls(
            agent_id=data.get('agent_id', ''),
            other_id=data.get('other_id', ''),
            status=data.get('status', 'Stranger'),
            affinity=data.get('affinity', 0.0),
            emotional_history=data.get('emotional_history', []),
            memory_significance=data.get('memory_significance', 0.0),
            recent_interactions=data.get('recent_interactions', [])
        )


class RelationshipDecisionInfluence:
    """Modifies decisions based on relationship considerations."""
    
    def __init__(self, relationship_manager: RelationshipManager):
        self.relationship_manager = relationship_manager
        
        # Relationship importance weights by status
        self.status_weights = {
            "Family": 0.9,
            "Best Friend": 0.8, 
            "Romantic": 0.85,
            "Close Friend": 0.7,
            "Friend": 0.5,
            "Acquaintance": 0.2,
            "Rival": 0.6,  # Rivals still influence decisions significantly
            "Enemy": 0.4,
            "Stranger": 0.1
        }
        
    async def apply_relationship_influence(self, choice_scores: Dict[str, float], 
                                         relationship_context: Dict[str, Any], 
                                         personality: Dict[str, float]) -> Dict[str, float]:
        """Modify choice scores based on relationship implications."""
        
        modified_scores = choice_scores.copy()
        
        for choice_id, base_score in choice_scores.items():
            relationship_modifier = 0.0
            
            # Analyze impact on each relevant relationship
            for other_agent_id, relationship_data in relationship_context.items():
                if isinstance(relationship_data, dict):
                    relationship = EnhancedRelationship.from_dict(relationship_data)
                    
                    # Choice impact on this relationship
                    choice_impact = await self._analyze_choice_relationship_impact(
                        choice_id, other_agent_id, relationship, personality
                    )
                    
                    # Weight by relationship importance
                    relationship_importance = self._calculate_relationship_importance(
                        relationship, personality
                    )
                    
                    relationship_modifier += choice_impact * relationship_importance
                    
            modified_scores[choice_id] = base_score + relationship_modifier
            
        # Ensure scores remain in valid range
        for choice_id in modified_scores:
            modified_scores[choice_id] = max(0.0, min(1.0, modified_scores[choice_id]))
            
        return modified_scores
    
    async def _analyze_choice_relationship_impact(self, choice_id: str, other_agent_id: str, 
                                                relationship: EnhancedRelationship, 
                                                personality: Dict[str, float]) -> float:
        """Analyze how a choice would affect a specific relationship."""
        
        # Base impact assessment using simple heuristics
        # In a real implementation, this might use LLM analysis
        
        impact_score = 0.0
        
        # Analyze choice keywords for relationship implications
        if await self._choice_helps_person(choice_id, other_agent_id):
            if relationship.affinity > 0.3:
                impact_score += 0.4  # Helping someone you like is good
            else:
                impact_score += 0.1  # Helping someone you dislike might still be positive
                
        elif await self._choice_harms_person(choice_id, other_agent_id):
            if relationship.affinity > 0.3:
                impact_score -= 0.6  # Harming someone you like is very bad
            else:
                impact_score -= 0.2  # Harming someone you dislike is less bad
                
        elif await self._choice_ignores_person(choice_id, other_agent_id):
            if relationship.status in ["Family", "Best Friend", "Romantic"]:
                impact_score -= 0.3  # Ignoring close relationships is bad
            else:
                impact_score -= 0.1  # Ignoring others is mildly negative
        
        # Consider relationship history
        if 'supportive' in relationship.emotional_history:
            impact_score += 0.1  # Past support makes positive choices more valuable
        elif 'hostile' in relationship.emotional_history:
            impact_score -= 0.1  # Past hostility makes negative choices worse
            
        # Consider personality alignment with relationship maintenance
        agreeableness = personality.get('agreeableness', 0.5)
        impact_score *= (0.5 + agreeableness * 0.5)  # More agreeable = more relationship concern
        
        return max(-1.0, min(1.0, impact_score))
    
    def _calculate_relationship_importance(self, relationship: EnhancedRelationship, 
                                         personality: Dict[str, float]) -> float:
        """Calculate how much this relationship should influence decisions."""
        
        importance = 0.0
        
        # Base importance on relationship strength and type
        importance += abs(relationship.affinity) * 0.4
        importance += relationship.memory_significance * 0.3
        
        # Personality modifiers
        agreeableness = personality.get('agreeableness', 0.5)
        importance += agreeableness * 0.3  # Agreeable people weight relationships more
        
        # Relationship status modifiers
        importance *= self.status_weights.get(relationship.status, 0.3)
        
        return min(importance, 1.0)
    
    async def _choice_helps_person(self, choice_id: str, person_id: str) -> bool:
        """Determine if choice helps a specific person."""
        # Simple keyword-based heuristic
        help_keywords = ['help', 'assist', 'support', 'aid', 'save', 'protect']
        choice_text = choice_id.lower()
        
        return any(keyword in choice_text for keyword in help_keywords) and person_id.lower() in choice_text
    
    async def _choice_harms_person(self, choice_id: str, person_id: str) -> bool:
        """Determine if choice harms a specific person."""
        # Simple keyword-based heuristic
        harm_keywords = ['hurt', 'harm', 'betray', 'abandon', 'attack', 'insult']
        choice_text = choice_id.lower()
        
        return any(keyword in choice_text for keyword in harm_keywords) and person_id.lower() in choice_text
    
    async def _choice_ignores_person(self, choice_id: str, person_id: str) -> bool:
        """Determine if choice ignores a specific person."""
        # Simple keyword-based heuristic
        ignore_keywords = ['ignore', 'avoid', 'dismiss', 'reject', 'leave']
        choice_text = choice_id.lower()
        
        return any(keyword in choice_text for keyword in ignore_keywords) and person_id.lower() in choice_text


class RelationshipConflictResolver:
    """Helps resolve conflicts between multiple relationship considerations."""
    
    def __init__(self):
        self.conflict_resolution_strategies = {
            'prioritize_closest': self._prioritize_closest_relationship,
            'minimize_damage': self._minimize_relationship_damage,
            'balance_all': self._balance_all_relationships,
            'personality_driven': self._personality_driven_resolution
        }
    
    async def resolve_relationship_conflicts(self, choice_scores: Dict[str, float],
                                           relationship_impacts: Dict[str, Dict[str, float]],
                                           personality: Dict[str, float]) -> Dict[str, float]:
        """Resolve conflicts when choices affect multiple relationships differently."""
        
        # Determine resolution strategy based on personality
        strategy = self._select_resolution_strategy(personality)
        
        # Apply the selected strategy
        resolved_scores = await self.conflict_resolution_strategies[strategy](
            choice_scores, relationship_impacts, personality
        )
        
        return resolved_scores
    
    def _select_resolution_strategy(self, personality: Dict[str, float]) -> str:
        """Select conflict resolution strategy based on personality."""
        
        agreeableness = personality.get('agreeableness', 0.5)
        conscientiousness = personality.get('conscientiousness', 0.5)
        neuroticism = personality.get('neuroticism', 0.5)
        
        if agreeableness > 0.7:
            return 'minimize_damage'  # Highly agreeable people avoid hurting anyone
        elif conscientiousness > 0.7:
            return 'personality_driven'  # Conscientious people stick to principles
        elif neuroticism > 0.7:
            return 'prioritize_closest'  # Anxious people focus on closest relationships
        else:
            return 'balance_all'  # Balanced approach
    
    async def _prioritize_closest_relationship(self, choice_scores: Dict[str, float],
                                             relationship_impacts: Dict[str, Dict[str, float]],
                                             personality: Dict[str, float]) -> Dict[str, float]:
        """Prioritize the closest/most important relationship."""
        # Find the most important relationship and weight its impact heavily
        return choice_scores  # Simplified implementation
    
    async def _minimize_relationship_damage(self, choice_scores: Dict[str, float],
                                          relationship_impacts: Dict[str, Dict[str, float]],
                                          personality: Dict[str, float]) -> Dict[str, float]:
        """Choose options that minimize damage to any relationship."""
        # Heavily penalize choices that damage any relationship
        return choice_scores  # Simplified implementation
    
    async def _balance_all_relationships(self, choice_scores: Dict[str, float],
                                       relationship_impacts: Dict[str, Dict[str, float]],
                                       personality: Dict[str, float]) -> Dict[str, float]:
        """Try to balance the needs of all relationships."""
        # Weight all relationships equally
        return choice_scores  # Simplified implementation
    
    async def _personality_driven_resolution(self, choice_scores: Dict[str, float],
                                           relationship_impacts: Dict[str, Dict[str, float]],
                                           personality: Dict[str, float]) -> Dict[str, float]:
        """Let personality traits drive the resolution."""
        # Use core personality values to guide decisions regardless of relationships
        return choice_scores  # Simplified implementation


class RelationshipMemoryInfluence:
    """Considers relationship memories and history in decision making."""
    
    def __init__(self):
        self.memory_decay_factor = 0.9  # How much older memories fade
        
    def calculate_historical_influence(self, relationship: EnhancedRelationship,
                                     choice_id: str) -> float:
        """Calculate influence based on relationship history and memories."""
        
        historical_influence = 0.0
        
        # Weight recent interactions more heavily
        for i, interaction in enumerate(relationship.recent_interactions[-5:]):  # Last 5 interactions
            interaction_relevance = self._assess_interaction_relevance(interaction, choice_id)
            decay_weight = (self.memory_decay_factor ** (len(relationship.recent_interactions) - i - 1))
            historical_influence += interaction_relevance * decay_weight
            
        # Consider emotional trajectory of the relationship
        if len(relationship.emotional_history) >= 2:
            recent_trend = self._analyze_emotional_trend(relationship.emotional_history)
            historical_influence += recent_trend * 0.2
            
        return historical_influence
    
    def _assess_interaction_relevance(self, interaction: Dict[str, Any], choice_id: str) -> float:
        """Assess how relevant a past interaction is to the current choice."""
        # Simple similarity-based relevance
        interaction_keywords = set(interaction.get('description', '').lower().split())
        choice_keywords = set(choice_id.lower().split('_'))
        
        overlap = len(interaction_keywords.intersection(choice_keywords))
        total_keywords = len(interaction_keywords.union(choice_keywords))
        
        if total_keywords == 0:
            return 0.0
            
        similarity = overlap / total_keywords
        
        # Weight by interaction outcome
        outcome_weight = 1.0
        if interaction.get('outcome') == 'positive':
            outcome_weight = 1.2
        elif interaction.get('outcome') == 'negative':
            outcome_weight = 0.8
            
        return similarity * outcome_weight
    
    def _analyze_emotional_trend(self, emotional_history: List[str]) -> float:
        """Analyze the emotional trend of the relationship."""
        if len(emotional_history) < 2:
            return 0.0
            
        # Simple trend analysis
        positive_emotions = ['happy', 'supportive', 'loving', 'caring', 'grateful']
        negative_emotions = ['angry', 'hostile', 'disappointed', 'hurt', 'betrayed']
        
        recent_score = 0.0
        older_score = 0.0
        
        # Score recent emotions (last 3)
        for emotion in emotional_history[-3:]:
            if emotion in positive_emotions:
                recent_score += 1
            elif emotion in negative_emotions:
                recent_score -= 1
                
        # Score older emotions (previous 3)
        for emotion in emotional_history[-6:-3]:
            if emotion in positive_emotions:
                older_score += 1
            elif emotion in negative_emotions:
                older_score -= 1
                
        # Return trend (positive = improving, negative = deteriorating)
        return (recent_score - older_score) / 6.0  # Normalize to [-1, 1] 