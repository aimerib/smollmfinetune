"""
Fallback prompt constructor for characters without runtime packets.

This module provides a basic prompt constructor that uses character data directly 
from the database when runtime packets are not available.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class FallbackPromptConstructor:
    """
    Fallback prompt constructor for characters without runtime packets.
    
    Uses character data directly from the database to construct basic prompts
    without requiring the full runtime packet infrastructure.
    """
    
    def __init__(self, character_id: str, character_name: str, character_data: Dict[str, Any]):
        """
        Initialize fallback prompt constructor.
        
        Args:
            character_id: Unique identifier for the character
            character_name: Display name of the character
            character_data: Character data from database including Big Five traits
        """
        self.character_id = character_id
        self.character_name = character_name
        self.character_data = character_data
        
        # Extract Big Five personality traits if available
        self.big_five = character_data.get('big_five', {})
        
        logger.info(f"Fallback prompt constructor created for {character_name}")
    
    def construct(self, conversation_history: List[Dict], dynamic_state: Optional[Dict] = None) -> str:
        """
        Construct a basic prompt using character data.
        
        Args:
            conversation_history: List of conversation turns
            dynamic_state: Dynamic character state (mood, relationships, events)
            
        Returns:
            Constructed prompt string
        """
        try:
            components = self._build_prompt_components(conversation_history, dynamic_state)
            return " ".join(components)
            
        except Exception as e:
            logger.error(f"Error in fallback prompt construction for {self.character_name}: {e}")
            return self._get_minimal_fallback_prompt()
    
    def _build_prompt_components(self, conversation_history: List[Dict], 
                                dynamic_state: Optional[Dict] = None) -> List[str]:
        """Build all prompt components systematically."""
        components = []
        
        # 1. Basic character introduction
        components.append(f"You are {self.character_name}.")
        
        # 2. Character description
        description = self.character_data.get('description')
        if description:
            components.append(f"Description: {description}")
        
        # 3. Personality traits from Big Five
        personality_traits = self._get_personality_traits()
        if personality_traits:
            components.append(f"You are {', '.join(personality_traits)}.")
        
        # 4. Relationship context
        relationship_context = self._build_relationship_context(dynamic_state)
        if relationship_context:
            components.append(relationship_context)
        
        # 5. Mood context
        mood_context = self._build_mood_context(dynamic_state)
        if mood_context:
            components.append(mood_context)
        
        # 6. Recent events
        events_context = self._build_events_context(dynamic_state)
        if events_context:
            components.append(events_context)
        
        # 7. Conversation history
        history_context = self._format_conversation_history(conversation_history)
        if history_context:
            components.append(history_context)
        
        # 8. Response instruction
        components.append("Respond naturally as this character.")
        
        return components
    
    def _get_personality_traits(self) -> List[str]:
        """
        Convert Big Five scores to descriptive personality traits.
        
        Returns:
            List of personality trait adjectives
        """
        if not self.big_five:
            return []
        
        traits = []
        trait_mappings = self._get_trait_mappings()
        
        for trait_name, score in self.big_five.items():
            if trait_name in trait_mappings and isinstance(score, (int, float)):
                trait_adjectives = self._get_adjectives_for_score(trait_mappings[trait_name], score)
                traits.extend(trait_adjectives)
        
        return traits[:4]  # Limit to avoid overwhelming the prompt
    
    def _get_trait_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Get the Big Five trait to adjective mappings."""
        return {
            'openness': {
                'high': ['creative', 'imaginative'],
                'low': ['practical', 'conventional']
            },
            'conscientiousness': {
                'high': ['organized', 'disciplined'],
                'low': ['spontaneous', 'flexible']
            },
            'extraversion': {
                'high': ['outgoing', 'energetic'],
                'low': ['introverted', 'reserved']
            },
            'agreeableness': {
                'high': ['kind', 'cooperative'],
                'low': ['competitive', 'direct']
            },
            'neuroticism': {
                'high': ['anxious', 'emotional'],
                'low': ['calm', 'stable']
            }
        }
    
    def _get_adjectives_for_score(self, trait_mapping: Dict[str, List[str]], score: float) -> List[str]:
        """Get adjectives based on trait score."""
        if score >= 0.7:
            return trait_mapping.get('high', [])
        elif score <= 0.3:
            return trait_mapping.get('low', [])
        return []
    
    def _build_relationship_context(self, dynamic_state: Optional[Dict]) -> Optional[str]:
        """Build relationship context from dynamic state."""
        if not dynamic_state or 'relationship_to_user' not in dynamic_state:
            return None
        
        relationship = dynamic_state['relationship_to_user']
        if not relationship:
            return None
        
        context_parts = []
        trust = relationship.get('trust', 0.5)
        affinity = relationship.get('affinity', 0.5)
        
        if trust > 0.7:
            context_parts.append("You trust the user.")
        elif trust < 0.3:
            context_parts.append("You are wary of the user.")
        
        if affinity > 0.7:
            context_parts.append("You like the user.")
        elif affinity < 0.3:
            context_parts.append("You dislike the user.")
        
        return " ".join(context_parts) if context_parts else None
    
    def _build_mood_context(self, dynamic_state: Optional[Dict]) -> Optional[str]:
        """Build mood context from dynamic state."""
        if not dynamic_state or 'current_mood' not in dynamic_state:
            return None
        
        mood = dynamic_state['current_mood']
        if mood and mood != 'neutral':
            return f"You are feeling {mood}."
        
        return None
    
    def _build_events_context(self, dynamic_state: Optional[Dict]) -> Optional[str]:
        """Build recent events context from dynamic state."""
        if not dynamic_state or 'recent_events' not in dynamic_state:
            return None
        
        recent_events = dynamic_state['recent_events']
        if recent_events:
            # Only include the most recent event to avoid overwhelming the prompt
            latest_event = recent_events[-1]
            return f"Recently: {latest_event}"
        
        return None
    
    def _format_conversation_history(self, history: List[Dict]) -> Optional[str]:
        """
        Format conversation history for prompt inclusion.
        
        Args:
            history: List of conversation turns
            
        Returns:
            Formatted conversation string or None if empty
        """
        if not history:
            return None
        
        formatted_turns = []
        # Only include recent messages to avoid prompt length issues
        recent_history = history[-5:] if len(history) > 5 else history
        
        for turn in recent_history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted_turns.append(f"User: {content}")
            elif role == 'assistant':
                character_name = turn.get('character_name', 'Assistant')
                formatted_turns.append(f"{character_name}: {content}")
        
        return "\n".join(formatted_turns) if formatted_turns else None
    
    def _get_minimal_fallback_prompt(self) -> str:
        """Get ultra-minimal fallback prompt for error cases."""
        return f"You are {self.character_name}. Respond naturally as this character."
    
    # Interface compatibility methods
    def get_character_name(self) -> str:
        """Get the character's name."""
        return self.character_name
    
    def get_available_tokens(self) -> List[Dict[str, Any]]:
        """Get available control tokens (empty for fallback)."""
        return []
    
    def get_tokens_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tokens by category (empty for fallback)."""
        return [] 