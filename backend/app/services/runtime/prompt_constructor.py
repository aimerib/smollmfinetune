"""
Runtime Prompt Constructor for character cartridges.

This module provides the RuntimePromptConstructor class that loads all assets
from a character runtime packet and constructs prompts with dynamic state
for real-time character interactions.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RuntimePromptConstructor:
    """
    Runtime prompt constructor for character cartridges.
    
    Loads character assets from a runtime packet and constructs prompts
    incorporating dynamic turn-by-turn state like mood, relationships,
    and recent events.
    """
    
    def __init__(self, packet_path: str):
        """
        Initialize the runtime prompt constructor by loading all assets.
        
        Args:
            packet_path: Path to the character runtime packet directory
            
        Raises:
            FileNotFoundError: If required packet files are missing
            ValueError: If packet files contain invalid data
        """
        self.packet_path = Path(packet_path)
        
        if not self.packet_path.exists():
            raise FileNotFoundError(f"Runtime packet not found: {packet_path}")
        
        # Load all packet assets
        self.character_core = self._load_character_core()
        self.world_lore = self._load_world_lore()
        self.control_tokens = self._load_control_tokens()
        self.runtime_config = self._load_runtime_config()
        
        # Build token lookup for fast access
        self.token_lookup = {token.get("token", ""): token for token in self.control_tokens}
        
        # Big Five personality trait mappings (mirrored from PromptBuilder)
        self.big_five_adjectives = {
            'openness': {
                'high': ['creative', 'imaginative', 'open-minded', 'artistic', 'curious', 'adventurous'],
                'medium': ['thoughtful', 'reflective', 'interested', 'moderate'],
                'low': ['practical', 'conventional', 'traditional', 'down-to-earth']
            },
            'conscientiousness': {
                'high': ['organized', 'disciplined', 'reliable', 'methodical', 'careful', 'thorough'],
                'medium': ['reasonably organized', 'generally reliable', 'fairly disciplined'],
                'low': ['spontaneous', 'flexible', 'relaxed', 'easygoing']
            },
            'extraversion': {
                'high': ['outgoing', 'energetic', 'sociable', 'talkative', 'assertive'],
                'medium': ['moderately social', 'balanced', 'selectively social'],
                'low': ['introverted', 'reserved', 'quiet', 'thoughtful', 'contemplative']
            },
            'agreeableness': {
                'high': ['kind', 'cooperative', 'trusting', 'empathetic', 'compassionate'],
                'medium': ['friendly', 'reasonably trusting', 'cooperative when needed'],
                'low': ['skeptical', 'competitive', 'direct', 'challenging']
            },
            'neuroticism': {
                'high': ['anxious', 'worried', 'stressed', 'emotional', 'sensitive'],
                'medium': ['occasionally stressed', 'moderately emotional', 'balanced'],
                'low': ['calm', 'stable', 'resilient', 'even-tempered']
            }
        }
        
        logger.info(f"Runtime prompt constructor initialized for '{self.character_core.get('name', 'Unknown')}'")
    
    def construct(self, conversation_history: List[Dict], dynamic_state: Optional[Dict] = None) -> str:
        """
        Construct a prompt with conversation history and dynamic state.
        
        Args:
            conversation_history: List of conversation turns with 'role' and 'content'
            dynamic_state: Optional dynamic state dictionary with keys:
                - current_mood: String mood affecting token selection
                - relationship_to_user: Dict with trust/affinity scores
                - recent_events: List of recent event summaries
                - forced_control_tokens: List of tokens to inject
                
        Returns:
            Complete prompt string ready for tokenization
        """
        if dynamic_state is None:
            dynamic_state = {}
        
        try:
            # Build prompt components
            components = []
            
            # 1. Base character introduction
            char_name = self.character_core.get('name', 'Character')
            components.append(f"You are {char_name}.")
            
            # 2. Inject Big-Five personality adjectives
            personality_traits = self._inject_big_five_traits()
            if personality_traits:
                components.append(f"You are {', '.join(personality_traits)}.")
            
            # 3. Add character goals
            goals = self.character_core.get('goals', [])
            if goals:
                top_goals = goals[:3]  # Top 3 goals
                components.append(f"Your main goals are: {', '.join(top_goals)}.")
            
            # 4. Add relationship context with dynamic state influence
            relationship_context = self._build_relationship_context(dynamic_state)
            if relationship_context:
                components.append(relationship_context)
            
            # 5. Inject random world lore fact
            lore_fact = self._get_random_lore_fact()
            if lore_fact:
                components.append(f"Remember: {lore_fact}")
            
            # 6. Add recent events if provided
            recent_events = dynamic_state.get('recent_events', [])
            if recent_events:
                events_summary = self._format_recent_events(recent_events)
                if events_summary:
                    components.append(events_summary)
            
            # 7. Add mood-based control tokens
            mood_tokens = self._get_mood_tokens(dynamic_state)
            if mood_tokens:
                components.extend(mood_tokens)
            
            # 8. Add forced control tokens
            forced_tokens = dynamic_state.get('forced_control_tokens', [])
            if forced_tokens:
                components.extend(forced_tokens)
            
            # 9. Format conversation history
            history_text = self._format_conversation_history(conversation_history)
            if history_text:
                components.append(history_text)
            
            # 10. Add response instruction
            components.append("Respond naturally as this character.")
            
            return " ".join(components)
            
        except Exception as e:
            logger.error(f"Error constructing runtime prompt: {e}")
            # Fallback to minimal prompt
            char_name = self.character_core.get('name', 'Character')
            last_user_msg = ""
            if conversation_history:
                for turn in reversed(conversation_history):
                    if turn.get('role') == 'user':
                        last_user_msg = turn.get('content', '')
                        break
            return f"You are {char_name}. {last_user_msg} Respond naturally as this character."
    
    def _load_character_core(self) -> Dict[str, Any]:
        """Load character_core.json from the packet"""
        core_path = self.packet_path / "character_core.json"
        if not core_path.exists():
            raise FileNotFoundError(f"character_core.json not found in packet: {core_path}")
        
        try:
            with open(core_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid character_core.json: {e}")
    
    def _load_world_lore(self) -> Dict[str, Any]:
        """Load world_lore.json from the packet"""
        lore_path = self.packet_path / "world_lore.json"
        if not lore_path.exists():
            logger.warning("world_lore.json not found, using empty lore")
            return {"facts": {}, "timeline": [], "factions": [], "places": []}
        
        try:
            with open(lore_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Invalid world_lore.json: {e}, using empty lore")
            return {"facts": {}, "timeline": [], "factions": [], "places": []}
    
    def _load_control_tokens(self) -> List[Dict[str, Any]]:
        """Load tokens.json from the packet"""
        tokens_path = self.packet_path / "tokens.json"
        if not tokens_path.exists():
            logger.warning("tokens.json not found, using no control tokens")
            return []
        
        try:
            with open(tokens_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Invalid tokens.json: {e}, using no control tokens")
            return []
    
    def _load_runtime_config(self) -> Dict[str, Any]:
        """Load runtime_config.json from the packet"""
        config_path = self.packet_path / "runtime_config.json"
        if not config_path.exists():
            logger.warning("runtime_config.json not found, using default config")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Invalid runtime_config.json: {e}, using default config")
            return {}
    
    def _inject_big_five_traits(self) -> List[str]:
        """Convert Big-Five scores to descriptive adjectives"""
        big_five = self.character_core.get('big_five', {})
        if not big_five:
            return []
        
        traits = []
        for dimension, score in big_five.items():
            if dimension in self.big_five_adjectives:
                level = self._score_to_level(score)
                adjectives = self.big_five_adjectives[dimension][level]
                # Pick 1-2 random adjectives per dimension
                selected = random.sample(adjectives, min(2, len(adjectives)))
                traits.extend(selected)
        
        return traits[:5]  # Limit to 5 total traits
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to level (high/medium/low)"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _build_relationship_context(self, dynamic_state: Dict[str, Any]) -> str:
        """Build relationship context with dynamic state influence"""
        # Get base relationships from character
        relationships = self.character_core.get('relationships', [])
        
        # Get dynamic relationship state
        relationship_to_user = dynamic_state.get('relationship_to_user', {})
        
        context_parts = []
        
        # Add base relationship stances
        if relationships:
            stances = []
            for rel in relationships[:2]:  # Top 2 relationships
                name = rel.get('name', 'someone')
                stance = rel.get('stance', 'neutral')
                stances.append(f"you have {stance} toward {name}")
            
            if stances:
                context_parts.append(f"In relationships, {' and '.join(stances)}.")
        
        # Add dynamic relationship state
        if relationship_to_user:
            trust = relationship_to_user.get('trust', 0.5)
            affinity = relationship_to_user.get('affinity', 0.5)
            
            if trust >= 0.8:
                context_parts.append("You feel a deep trust with the user.")
            elif trust >= 0.6:
                context_parts.append("You have growing trust with the user.")
            elif trust <= 0.3:
                context_parts.append("You remain cautious with the user.")
            
            if affinity >= 0.8:
                context_parts.append("You feel a strong connection to the user.")
            elif affinity <= 0.3:
                context_parts.append("You maintain professional distance from the user.")
        
        return " ".join(context_parts)
    
    def _get_random_lore_fact(self) -> Optional[str]:
        """Get a random fact from world lore"""
        facts = self.world_lore.get('facts', {})
        if facts:
            # Convert dict values to list and pick randomly
            fact_values = list(facts.values())
            return random.choice(fact_values)
        return None
    
    def _format_recent_events(self, events: List[str]) -> str:
        """Format recent events for inclusion in prompt"""
        if not events:
            return ""
        
        # Limit to most recent 3 events
        recent = events[-3:]
        return f"Recent events: {'; '.join(recent)}."
    
    def _get_mood_tokens(self, dynamic_state: Dict[str, Any]) -> List[str]:
        """Get control tokens based on current mood"""
        current_mood = dynamic_state.get('current_mood', '')
        if not current_mood:
            return []
        
        # Look for mood-specific control tokens
        mood_token = f"<mood_{current_mood}>"
        if mood_token in self.token_lookup:
            return [mood_token]
        
        return []
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for inclusion in prompt"""
        if not history:
            return ""
        
        formatted_turns = []
        for turn in history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted_turns.append(f"User: {content}")
            elif role == 'assistant':
                formatted_turns.append(f"Assistant: {content}")
        
        return "\n".join(formatted_turns)
    
    def get_character_name(self) -> str:
        """Get the character's name"""
        return self.character_core.get('name', 'Unknown')
    
    def get_available_tokens(self) -> List[Dict[str, Any]]:
        """Get all available control tokens"""
        return self.control_tokens.copy()
    
    def get_tokens_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get control tokens filtered by category"""
        return [token for token in self.control_tokens if token.get('category') == category] 