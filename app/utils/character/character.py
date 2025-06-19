import re
import orjson
from pathlib import Path
from typing import Dict, Any, Optional
from ..world import WorldManager


class CharacterManager:
    """Manages character card loading and processing"""
    
    KEEP_FIELDS = {
        "name",
        "description",
        "personality",
        "mes_example",
        "scenario",
    }
    
    def __init__(self, world_manager: Optional[WorldManager] = None):
        self.current_character: Optional[Dict[str, Any]] = None
        self.world_manager = world_manager or WorldManager()
        self.current_world: Optional[str] = None
        
        # Ensure default world exists
        self._ensure_default_world()
    
    def _ensure_default_world(self):
        """Ensure a default world exists if no worlds are available"""
        worlds = self.world_manager.list_worlds()
        if not worlds:
            # Create default world
            self.world_manager.create_world("Default World")
            self.current_world = "Default World"
        else:
            # Use first available world as current if none set
            if not self.current_world:
                self.current_world = worlds[0]
    
    def set_current_world(self, world_name: str) -> bool:
        """Set the current active world"""
        worlds = self.world_manager.list_worlds()
        if world_name not in worlds:
            return False
        
        self.current_world = world_name
        return True
    
    def get_current_world(self) -> Optional[str]:
        """Get the current active world name"""
        return self.current_world
    
    def get_world_lore(self) -> Optional[Dict[str, Any]]:
        """Get the current world's lore for context"""
        if not self.current_world:
            return None
        
        lore = self.world_manager.load_world(self.current_world)
        if not lore:
            return None
        
        # Return a simplified dict representation for use in prompts
        return {
            "facts": lore.facts,
            "timeline": [{"year": event.year, "event": event.event} for event in lore.timeline],
            "factions": [
                {
                    "name": faction.name,
                    "timeline": [{"year": event.year, "event": event.event} for event in faction.timeline]
                }
                for faction in lore.factions
            ],
            "places": [
                {
                    "name": place.name,
                    "description": place.description,
                    "npcs": [{"name": npc.name, "description": npc.description} for npc in place.npcs]
                }
                for place in lore.places
            ]
        }
    
    def load_character_card(self, card_data: Dict[str, Any]) -> Dict[str, str]:
        """Load and process a character card"""
        # Keep only whitelisted fields and ensure they're strings
        card = {
            k: v for k, v in card_data.items() 
            if k in self.KEEP_FIELDS and isinstance(v, str)
        }
        
        # Normalize whitespace
        for k, v in card.items():
            card[k] = re.sub(r"\s+", " ", v).strip()
        
        self.current_character = card
        return card
    
    def make_card_block(self, card: Optional[Dict[str, str]] = None, include_world_context: bool = True) -> str:
        """Generate the canonical <CHAR_CARD> block for system prompts"""
        if card is None:
            card = self.current_character
        
        if not card:
            return ""
        
        lines = ["### <CHAR_CARD>"]
        lines.append(f"Name: {card.get('name', 'Unknown')}")

        # Main description fields
        for field in ["description", "scenario", "personality"]:
            if field in card:
                pretty = card[field].replace("\n", " ")
                lines.append(f"{field.capitalize()}: {pretty}")
        
        # Add example
        if "mes_example" in card:
            lines.append(f"Example: {card['mes_example']}")
        
        lines.append("<|endofcard|>")
        
        # Add world context if requested and available
        if include_world_context and self.current_world:
            world_lore = self.get_world_lore()
            if world_lore:
                lines.append("\n### <WORLD_CONTEXT>")
                lines.append(f"World: {self.current_world}")
                
                # Add key facts
                if world_lore["facts"]:
                    lines.append("World Facts:")
                    for key, value in world_lore["facts"].items():
                        lines.append(f"- {key}: {value}")
                
                # Add major timeline events (limit to most recent/important)
                if world_lore["timeline"]:
                    lines.append("Key Historical Events:")
                    for event in world_lore["timeline"][-3:]:  # Last 3 events
                        lines.append(f"- Year {event['year']}: {event['event']}")
                
                lines.append("<|endofworld|>")
        
        return "\n".join(lines)
    
    def validate_character_card(self, card: Dict[str, Any]) -> tuple[bool, str]:
        """Validate a character card and return (is_valid, error_message)"""
        if not isinstance(card, dict):
            return False, "Character card must be a JSON object"
        
        if 'name' not in card:
            return False, "Character card must have a 'name' field"
        
        if not isinstance(card['name'], str) or not card['name'].strip():
            return False, "Character name must be a non-empty string"
        
        # Check for minimum required content
        has_description = 'description' in card and isinstance(card['description'], str)
        has_personality = 'personality' in card and isinstance(card['personality'], str)
        
        if not has_description and not has_personality:
            return False, "Character card must have either 'description' or 'personality' field"
        
        return True, ""
    
    def get_character_summary(self, card: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get a summary of character statistics"""
        if card is None:
            card = self.current_character
        
        if not card:
            return {}
        
        summary = {
            'name': card.get('name', 'Unknown'),
            'has_description': bool(card.get('description')),
            'has_personality': bool(card.get('personality')),
            'has_scenario': bool(card.get('scenario')),
            'description_length': len(card.get('description', '')),
            'personality_length': len(card.get('personality', '')),
            'total_content_length': sum(len(card.get(field, '')) for field in self.KEEP_FIELDS),
            'current_world': self.current_world
        }
        
        return summary 