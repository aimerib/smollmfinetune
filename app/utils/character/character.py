import re
import orjson
from pathlib import Path
from typing import Dict, Any, Optional


class CharacterManager:
    """Manages character card loading and processing"""
    
    KEEP_FIELDS = {
        "name",
        "description",
        "personality",
        "mes_example",
        "scenario",
    }
    
    def __init__(self):
        self.current_character: Optional[Dict[str, Any]] = None
    
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
    
    def make_card_block(self, card: Optional[Dict[str, str]] = None) -> str:
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
            'total_content_length': sum(len(card.get(field, '')) for field in self.KEEP_FIELDS)
        }
        
        return summary 