"""
Emotional Decision Modifiers

This module modifies personality-based decisions using current emotional states.
Emotions can temporarily override or enhance personality traits, making characters
more realistic by showing how feelings influence rational decision-making.
"""

from typing import Dict, List, Any
import random


class EmotionalDecisionModifier:
    """Modifies personality-based decisions using current emotional state."""
    
    def __init__(self):
        # Mapping of emotions to their behavioral effects
        self.emotion_effects = {
            'angry': {
                'confrontation': 0.4,  # Increases confrontational choices
                'cooperation': -0.3,   # Decreases cooperative choices
                'risk_tolerance': 0.2, # Slightly increases risk-taking
                'patience': -0.5       # Decreases patience/planning
            },
            'fearful': {
                'risk_aversion': 0.6,  # Strongly increases risk aversion
                'social_withdrawal': 0.3, # Increases preference for isolation
                'conservative_choice': 0.4, # Prefers familiar/safe options
                'trust': -0.2          # Decreases trust in others
            },
            'happy': {
                'social_engagement': 0.3, # Increases social openness
                'optimism': 0.4,          # More optimistic choices
                'risk_tolerance': 0.2,    # Slight increase in risk-taking
                'generosity': 0.3         # More generous/altruistic
            },
            'sad': {
                'energy': -0.4,           # Decreases high-energy choices
                'social_withdrawal': 0.2, # Slight preference for solitude
                'comfort_seeking': 0.5,   # Seeks comfort/support
                'pessimism': 0.3          # More pessimistic outlook
            },
            'excited': {
                'impulsivity': 0.4,       # Increases impulsive choices
                'social_engagement': 0.4, # Increases social activity
                'novelty_seeking': 0.3,   # Seeks new experiences
                'energy': 0.5             # Prefers high-energy options
            },
            'anxious': {
                'overthinking': 0.3,      # Increases analysis paralysis
                'risk_aversion': 0.4,     # Increases risk aversion
                'comfort_seeking': 0.3,   # Seeks familiar options
                'social_caution': 0.2     # More cautious socially
            }
        }
    
    async def apply_emotional_influence(self, choice_scores: Dict[str, float], 
                                      emotional_state: Dict[str, float], 
                                      personality: Dict[str, float]) -> Dict[str, float]:
        """Modify choice scores based on current emotions."""
        
        modified_scores = choice_scores.copy()
        
        for emotion, intensity in emotional_state.items():
            if intensity > 0.3:  # Only apply significant emotions
                modified_scores = await self._apply_specific_emotion(
                    modified_scores, emotion, intensity, personality
                )
        
        # Ensure scores remain in valid range
        for choice_id in modified_scores:
            modified_scores[choice_id] = max(0.0, min(1.0, modified_scores[choice_id]))
        
        return modified_scores
    
    async def _apply_specific_emotion(self, choice_scores: Dict[str, float], 
                                    emotion: str, intensity: float,
                                    personality: Dict[str, float]) -> Dict[str, float]:
        """Apply the effects of a specific emotion to choice scores."""
        
        if emotion not in self.emotion_effects:
            return choice_scores  # Unknown emotion, no modification
        
        effects = self.emotion_effects[emotion]
        
        # Anger increases impulsivity, decreases agreeableness temporarily
        if emotion == 'angry':
            for choice_id, score in choice_scores.items():
                if await self._choice_involves_confrontation(choice_id):
                    choice_scores[choice_id] += intensity * effects['confrontation']
                if await self._choice_involves_cooperation(choice_id):
                    choice_scores[choice_id] += intensity * effects['cooperation']
                    
        # Fear increases risk aversion across all personality types
        elif emotion == 'fearful':
            for choice_id, score in choice_scores.items():
                risk_level = await self._assess_choice_risk(choice_id)
                choice_scores[choice_id] -= intensity * risk_level * effects['risk_aversion']
                
        # Joy increases openness and extraversion temporarily
        elif emotion == 'happy':
            for choice_id, score in choice_scores.items():
                if await self._choice_involves_social_interaction(choice_id):
                    choice_scores[choice_id] += intensity * effects['social_engagement']
                if await self._choice_involves_novelty(choice_id):
                    choice_scores[choice_id] += intensity * effects['optimism']
                    
        # Sadness decreases energy, increases need for comfort/support
        elif emotion == 'sad':
            for choice_id, score in choice_scores.items():
                if await self._choice_provides_comfort(choice_id):
                    choice_scores[choice_id] += intensity * effects['comfort_seeking']
                if await self._choice_requires_high_energy(choice_id):
                    choice_scores[choice_id] += intensity * effects['energy']  # Negative value
        
        # Excitement increases impulsivity and social engagement
        elif emotion == 'excited':
            for choice_id, score in choice_scores.items():
                if await self._choice_involves_social_interaction(choice_id):
                    choice_scores[choice_id] += intensity * effects['social_engagement']
                if await self._choice_involves_novelty(choice_id):
                    choice_scores[choice_id] += intensity * effects['novelty_seeking']
                if await self._choice_requires_immediate_action(choice_id):
                    choice_scores[choice_id] += intensity * effects['impulsivity']
        
        # Anxiety increases caution and overthinking
        elif emotion == 'anxious':
            for choice_id, score in choice_scores.items():
                risk_level = await self._assess_choice_risk(choice_id)
                choice_scores[choice_id] -= intensity * risk_level * effects['risk_aversion']
                if await self._choice_provides_comfort(choice_id):
                    choice_scores[choice_id] += intensity * effects['comfort_seeking']
        
        return choice_scores
    
    async def _choice_involves_confrontation(self, choice_id: str) -> bool:
        """Determine if choice involves confrontational behavior."""
        # Simple heuristic based on choice_id keywords
        confrontational_keywords = ['fight', 'argue', 'confront', 'challenge', 'oppose', 'attack']
        return any(keyword in choice_id.lower() for keyword in confrontational_keywords)
    
    async def _choice_involves_cooperation(self, choice_id: str) -> bool:
        """Determine if choice involves cooperative behavior."""
        cooperative_keywords = ['help', 'cooperate', 'collaborate', 'support', 'assist', 'share']
        return any(keyword in choice_id.lower() for keyword in cooperative_keywords)
    
    async def _choice_involves_social_interaction(self, choice_id: str) -> bool:
        """Determine if choice involves social interaction."""
        social_keywords = ['social', 'talk', 'meet', 'party', 'group', 'friend', 'gather']
        return any(keyword in choice_id.lower() for keyword in social_keywords)
    
    async def _choice_involves_novelty(self, choice_id: str) -> bool:
        """Determine if choice involves novel/new experiences."""
        novelty_keywords = ['new', 'explore', 'adventure', 'discover', 'unknown', 'mysterious']
        return any(keyword in choice_id.lower() for keyword in novelty_keywords)
    
    async def _choice_provides_comfort(self, choice_id: str) -> bool:
        """Determine if choice provides comfort or support."""
        comfort_keywords = ['home', 'safe', 'comfort', 'rest', 'familiar', 'peaceful']
        return any(keyword in choice_id.lower() for keyword in comfort_keywords)
    
    async def _choice_requires_high_energy(self, choice_id: str) -> bool:
        """Determine if choice requires high energy or effort."""
        energy_keywords = ['run', 'fight', 'climb', 'work', 'effort', 'strenuous', 'active']
        return any(keyword in choice_id.lower() for keyword in energy_keywords)
    
    async def _choice_requires_immediate_action(self, choice_id: str) -> bool:
        """Determine if choice requires immediate action."""
        immediate_keywords = ['now', 'quick', 'immediate', 'urgent', 'fast', 'rush']
        return any(keyword in choice_id.lower() for keyword in immediate_keywords)
    
    async def _assess_choice_risk(self, choice_id: str) -> float:
        """Assess risk level of choice (0.0 to 1.0)."""
        # Simple heuristic based on choice_id keywords
        high_risk_keywords = ['danger', 'risk', 'uncertain', 'unknown', 'fight', 'gamble']
        low_risk_keywords = ['safe', 'certain', 'familiar', 'proven', 'secure', 'stable']
        
        risk_score = 0.5  # Default medium risk
        
        for keyword in high_risk_keywords:
            if keyword in choice_id.lower():
                risk_score += 0.2
        
        for keyword in low_risk_keywords:
            if keyword in choice_id.lower():
                risk_score -= 0.2
        
        return max(0.0, min(1.0, risk_score))


class EmotionalStateAnalyzer:
    """Analyzes and tracks emotional states for decision making."""
    
    def __init__(self):
        self.emotion_history = {}
    
    def analyze_emotional_trajectory(self, agent_id: str, current_emotions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze how emotions are changing over time."""
        if agent_id not in self.emotion_history:
            self.emotion_history[agent_id] = []
        
        self.emotion_history[agent_id].append(current_emotions)
        
        # Keep only recent history (last 10 states)
        if len(self.emotion_history[agent_id]) > 10:
            self.emotion_history[agent_id] = self.emotion_history[agent_id][-10:]
        
        if len(self.emotion_history[agent_id]) < 2:
            return {'trajectory': 'insufficient_data'}
        
        # Calculate emotional trajectory
        recent_emotions = self.emotion_history[agent_id][-2:]
        prev_emotions = recent_emotions[0]
        curr_emotions = recent_emotions[1]
        
        trajectory = {}
        for emotion in curr_emotions:
            if emotion in prev_emotions:
                change = curr_emotions[emotion] - prev_emotions[emotion]
                trajectory[emotion] = {
                    'change': change,
                    'direction': 'increasing' if change > 0.1 else 'decreasing' if change < -0.1 else 'stable'
                }
        
        return {'trajectory': trajectory}
    
    def predict_emotional_volatility(self, agent_id: str) -> float:
        """Predict how emotionally volatile the agent is being."""
        if agent_id not in self.emotion_history or len(self.emotion_history[agent_id]) < 3:
            return 0.5  # Default moderate volatility
        
        recent_states = self.emotion_history[agent_id][-5:]  # Last 5 states
        
        # Calculate average change between consecutive states
        total_change = 0.0
        comparisons = 0
        
        for i in range(1, len(recent_states)):
            prev_state = recent_states[i-1]
            curr_state = recent_states[i]
            
            for emotion in curr_state:
                if emotion in prev_state:
                    total_change += abs(curr_state[emotion] - prev_state[emotion])
                    comparisons += 1
        
        if comparisons == 0:
            return 0.5
        
        avg_change = total_change / comparisons
        
        # Normalize to 0-1 scale (assume max change of 2.0 per emotion)
        volatility = min(1.0, avg_change / 0.5)
        
        return volatility 