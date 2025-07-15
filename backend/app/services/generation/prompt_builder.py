"""
Advanced prompt builder for character-aware dataset generation.

This module provides sophisticated prompt building capabilities that inject
Big-Five personality traits, character goals, relationship stances, and world lore
into conversation prompts. Supports NSFW styling and multi-turn conversations.
"""

import random
import re
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Advanced prompt builder with personality and lore injection"""
    
    def __init__(self, world_lore: Optional[Dict[str, Any]] = None, control_tokens: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the prompt builder with world context and control tokens.
        
        Args:
            world_lore: Dictionary containing world facts and context
            control_tokens: List of control token definitions
        """
        self.world_lore = world_lore or {'facts': []}
        self.control_tokens = control_tokens or []
        
        # Build token lookup for fast access
        self.token_lookup = {token.get("token", ""): token for token in self.control_tokens}
        
        # Big Five personality trait mappings
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
    
    def build_prompt(self, character: Union[Dict[str, Any], Any], mode: str = "chat", 
                    base_prompt: str = "", nsfw_style: Optional[str] = None,
                    use_cache: bool = False, add_token_hints: bool = True, **opts) -> str:
        """
        Build an enhanced prompt with personality, goals, relationships, and lore.
        
        Args:
            character: Character dict or CharacterCore object
            mode: Generation mode ("chat", "nsfw", "qa")
            base_prompt: Base prompt text to enhance
            nsfw_style: NSFW style tag ("soft", "explicit", "kink")
            use_cache: Whether to use caching (placeholder for future implementation)
            add_token_hints: Whether to add natural language hints for control tokens
            **opts: Additional options
            
        Returns:
            Enhanced prompt string with personality and lore injected
        """
        try:
            # Extract character data (handle both dict and object formats)
            char_data = self._extract_character_data(character)
            
            # Build prompt components
            components = []
            
            # 1. Base character introduction
            components.append(f"You are {char_data['name']}.")
            
            # 2. Inject Big-Five personality adjectives
            personality_traits = self._inject_big_five_traits(char_data)
            if personality_traits:
                components.append(f"You are {', '.join(personality_traits)}.")
            
            # 3. Add character goals
            goals = char_data.get('goals', [])
            if goals:
                top_goals = goals[:3]  # Top 3 goals
                components.append(f"Your main goals are: {', '.join(top_goals)}.")
            
            # 4. Add relationship context
            relationships = char_data.get('relationships', [])
            if relationships:
                rel_context = self._build_relationship_context(relationships)
                if rel_context:
                    components.append(rel_context)
            
            # 5. Inject random lore fact
            lore_fact = self._get_random_lore_fact()
            if lore_fact:
                components.append(f"Remember: {lore_fact}")
            
            # 6. Add NSFW style tag if needed (convert to control token format)
            if mode == "nsfw" and nsfw_style:
                nsfw_token = f"<nsfw_{nsfw_style}>"
                components.append(nsfw_token)
            
            # 7. Add control token hints if enabled and tokens are present
            if add_token_hints:
                token_hints = self._generate_token_hints(components, base_prompt)
                if token_hints:
                    components.insert(-len(components), token_hints)  # Insert after character info but before content
            
            # 8. Add the base prompt
            if base_prompt:
                components.append(f"\nUser: {base_prompt}")
            
            # 9. Mode-specific formatting
            if mode == "qa":
                components.append("Answer factually and stay in character.")
            elif mode == "chat":
                components.append("Respond naturally as this character.")
            
            return " ".join(components)
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Fallback to basic prompt
            char_name = getattr(character, 'name', None) or character.get('name', 'Character')
            return f"You are {char_name}. {base_prompt}"
    
    def _extract_character_data(self, character: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Extract character data from either dict or object format"""
        if hasattr(character, 'name'):
            # CharacterCore object - convert to dict
            return {
                'name': character.name,
                'description': getattr(character, 'description', ''),
                'personality': getattr(character, 'personality', ''),
                'big_five': getattr(character, 'big_five', {}),
                'goals': getattr(character, 'goals', []),
                'relationships': getattr(character, 'relationships', [])
            }
        else:
            # Already a dict
            return character
    
    def _inject_big_five_traits(self, char_data: Dict[str, Any]) -> List[str]:
        """Convert Big-Five scores to descriptive adjectives"""
        big_five = char_data.get('big_five', {})
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
    
    def _big_five_to_adjectives(self, big_five: Dict[str, float]) -> List[str]:
        """Public method for testing - convert Big-Five scores to adjectives"""
        return self._inject_big_five_traits({'big_five': big_five})
    
    def _build_relationship_context(self, relationships: List[Dict[str, Any]]) -> str:
        """Build relationship context string"""
        if not relationships:
            return ""
        
        # Extract relationship stances
        stances = []
        for rel in relationships[:2]:  # Top 2 relationships
            name = rel.get('name', 'someone')
            stance = rel.get('stance', 'neutral')
            stances.append(f"you have {stance} toward {name}")
        
        if stances:
            return f"In relationships, {' and '.join(stances)}."
        
        return ""
    
    def _get_random_lore_fact(self) -> Optional[str]:
        """Get a random fact from world lore"""
        facts = self.world_lore.get('facts', [])
        if facts:
            return random.choice(facts)
        return None
    
    def _find_tokens_in_text(self, text: str) -> List[str]:
        """Find control tokens in text using regex"""
        token_pattern = r'<[^>]+>'
        found_tokens = re.findall(token_pattern, text)
        
        # Filter to only known control tokens
        valid_tokens = []
        for token in found_tokens:
            if token in self.token_lookup:
                valid_tokens.append(token)
        
        return valid_tokens
    
    def _generate_token_hints(self, components: List[str], base_prompt: str = "") -> Optional[str]:
        """Generate natural language hints for control tokens found in the prompt"""
        # Combine all text to search for tokens
        full_text = " ".join(components) + " " + base_prompt
        
        # Find all control tokens
        found_tokens = self._find_tokens_in_text(full_text)
        
        if not found_tokens:
            return None
        
        # Remove duplicates while preserving order
        unique_tokens = list(dict.fromkeys(found_tokens))
        
        # Generate hints
        hints = []
        for token in unique_tokens:
            token_info = self.token_lookup.get(token)
            if token_info:
                description = token_info.get("description", "")
                if description:
                    hints.append(f"The token {token} means {description}")
        
        if hints:
            hint_text = "Token meanings: " + "; ".join(hints) + "."
            return hint_text
        
        return None
    
    def apply_control_tokens(self, text: str, selected_tokens: List[str]) -> str:
        """Apply control tokens to text by inserting them at the beginning"""
        if not selected_tokens:
            return text
        
        # Insert tokens at the beginning
        token_string = " ".join(selected_tokens)
        return f"{token_string} {text}"
    
    def get_available_tokens_by_category(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available control tokens, optionally filtered by category"""
        if category is None:
            return self.control_tokens
        
        return [token for token in self.control_tokens if token.get("category") == category]


def build_prompt(character: Union[Dict[str, Any], Any], world_lore: Dict[str, Any],
                mode: str = "chat", base_prompt: str = "", control_tokens: Optional[List[Dict[str, Any]]] = None, **opts) -> str:
    """
    Standalone function to build a prompt.
    
    Args:
        character: Character data
        world_lore: World lore context
        mode: Generation mode
        base_prompt: Base prompt text
        control_tokens: List of control token definitions
        **opts: Additional options
        
    Returns:
        Enhanced prompt string
    """
    builder = PromptBuilder(world_lore=world_lore, control_tokens=control_tokens)
    return builder.build_prompt(character=character, mode=mode, base_prompt=base_prompt, **opts)


def build_conversation_turn(turn_idx: int, role: str, text: str, **metadata) -> Dict[str, Any]:
    """
    Build a conversation turn with metadata for multi-turn conversations.
    
    Args:
        turn_idx: Turn index in conversation
        role: Role ("user" or "assistant")
        text: Turn content text
        **metadata: Additional metadata
        
    Returns:
        Turn data with structure and metadata
    """
    return {
        'turn_id': turn_idx,
        'role': role,
        'content': text,
        'metadata': dict(metadata)
    } 