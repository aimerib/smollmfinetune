import re
import orjson
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from backend.app.core.openai_client import get_client

# from .openai_client import OpenAIClient

from backend.app.services.world.world import WorldManager
from .models import CharacterCore, Personality, Relationship, llm_estimate_big5

logger = logging.getLogger(__name__)


class CharacterManager:
    """Manages character card loading and processing"""
    
    KEEP_FIELDS = {
        "name",
        "description",
        "personality",
        "mes_example",
        "scenario",
    }
    
    def __init__(self, world_manager: Optional[WorldManager] = None, client = None):
        self.current_character: Optional[Dict[str, Any]] = None
        self.current_character_core: Optional[CharacterCore] = None
        self.world_manager = world_manager or WorldManager()
        self.current_world: Optional[str] = None
        self.client = client or get_client()
        
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
    
    async def import_sillytavern_card(self, card_data: Dict[str, Any]) -> CharacterCore:
        """
        Import a SillyTavern character card and convert to CharacterCore format
        
        Args:
            card_data: Raw SillyTavern card data
            
        Returns:
            CharacterCore object with converted data
        """
        # Basic field extraction
        name = card_data.get('name', 'Unknown Character')
        description = card_data.get('description', '')
        scenario = card_data.get('scenario', '')
        mes_example = card_data.get('mes_example', '')
        personality_text = card_data.get('personality', '')
        
        # Use existing analysis tools to extract richer information
        from ..dataset.character_analysis import extract_character_knowledge
        knowledge = await extract_character_knowledge(client=self.client, character=card_data)
        
        # Extract appearance from description if structured
        appearance = ""
        if 'appearance' in knowledge and knowledge['appearance']:
            appearance = ", ".join(knowledge['appearance'])
        
        # Extract backstory elements
        backstory = ""
        if knowledge.get('backstory_elements'):
            backstory = ". ".join(knowledge['backstory_elements'][:3])  # Top 3 backstory elements
        
        # Estimate Big Five personality traits using LLM
        try:
            personality_traits = await llm_estimate_big5(description + " " + personality_text, mes_example)
        except Exception as e:
            logger.warning(f"Failed to estimate personality traits: {e}")
            personality_traits = Personality()
        
        # Extract goals
        goals = knowledge.get('goals', [])
        
        # Build relationships from knowledge
        relationships = []
        relationship_data = knowledge.get('relationships', [])
        for rel in relationship_data[:5]:  # Limit to 5 relationships
            if isinstance(rel, str):
                # Simple relationship, assume neutral affinity
                relationships.append(Relationship(name=rel, affinity=0))
        
        # Generate tags from analysis
        tags = []
        if knowledge.get('species'):
            tags.append(knowledge['species'])
        if knowledge.get('occupation'):
            tags.append(knowledge['occupation'])
        tags.extend(knowledge.get('tags', [])[:5])  # Limit tags
        
        # Create CharacterCore object
        character_core = CharacterCore(
            name=name,
            description=description,
            scenario=scenario,
            backstory=backstory,
            appearance=appearance,
            personality_traits=personality_traits,
            goals=goals,
            relationships=relationships,
            tags=list(set(tags)),  # Remove duplicates
            imports={
                "source": "sillytavern",
                "raw_fields": list(card_data.keys()),
                "original_personality": personality_text,
                "original_mes_example": mes_example,
                "has_mes_example": bool(mes_example)
            }
        )
        
        self.current_character_core = character_core
        return character_core
    
    def save_character(self, core: CharacterCore, world_path: Optional[Path] = None) -> bool:
        """
        Save character in the new folder structure
        
        Args:
            core: CharacterCore object to save
            world_path: Path to world directory (uses current world if None)
            
        Returns:
            True if successful, False otherwise
        """
        if world_path is None:
            if not self.current_world:
                logger.error("No current world set for saving character")
                return False
            world_path = self.world_manager.get_world_path(self.current_world)
        
        # Create character folder structure
        char_folder = world_path / "characters" / core.name
        char_folder.mkdir(parents=True, exist_ok=True)
        
        # Create assets folder
        assets_folder = char_folder / "assets"
        assets_folder.mkdir(exist_ok=True)
        
        try:
            # Save character_core.json
            core_path = char_folder / "character_core.json"
            with open(core_path, 'wb') as f:
                f.write(orjson.dumps(core.model_dump(), option=orjson.OPT_INDENT_2))
            
            # Save mes_example.txt if available from imports
            if 'original_mes_example' in core.imports or hasattr(self, '_temp_mes_example'):
                mes_example_text = core.imports.get('original_mes_example', getattr(self, '_temp_mes_example', ''))
                if mes_example_text:
                    mes_example_path = char_folder / "mes_example.txt"
                    with open(mes_example_path, 'w', encoding='utf-8') as f:
                        f.write(mes_example_text)
            
            logger.info(f"Successfully saved character '{core.name}' to {char_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save character '{core.name}': {e}")
            return False
    
    def load_character_core(self, char_folder: Path) -> Optional[CharacterCore]:
        """
        Load a character from the new folder structure
        
        Args:
            char_folder: Path to character folder
            
        Returns:
            CharacterCore object if successful, None otherwise
        """
        core_path = char_folder / "character_core.json"
        if not core_path.exists():
            logger.error(f"character_core.json not found in {char_folder}")
            return None
        
        try:
            with open(core_path, 'rb') as f:
                data = orjson.loads(f.read())
            
            character_core = CharacterCore(**data)
            self.current_character_core = character_core
            
            logger.info(f"Successfully loaded character '{character_core.name}' from {char_folder}")
            return character_core
            
        except Exception as e:
            logger.error(f"Failed to load character from {char_folder}: {e}")
            return None
    
    def list_characters_in_world(self, world_name: Optional[str] = None) -> List[str]:
        """
        List all characters in a world
        
        Args:
            world_name: Name of world (uses current world if None)
            
        Returns:
            List of character names
        """
        if world_name is None:
            world_name = self.current_world
        
        if not world_name:
            return []
        
        chars_path = self.world_manager.get_characters_path(world_name)
        if not chars_path.exists():
            return []
        
        characters = []
        for char_folder in chars_path.iterdir():
            if char_folder.is_dir() and (char_folder / "character_core.json").exists():
                characters.append(char_folder.name)
        
        return sorted(characters)
    
    # Legacy methods for backward compatibility
    def load_character_card(self, card_data: Dict[str, Any]) -> Dict[str, str]:
        """Load and process a character card (legacy format)"""
        # Keep only whitelisted fields and ensure they're strings
        card = {
            k: v for k, v in card_data.items() 
            if k in self.KEEP_FIELDS and isinstance(v, str)
        }
        
        # Normalize whitespace
        for k, v in card.items():
            card[k] = re.sub(r"\s+", " ", v).strip()
        
        self.current_character = card
        
        # Store mes_example temporarily for potential save operation
        if 'mes_example' in card_data:
            self._temp_mes_example = card_data['mes_example']
        
        return card
    
    def make_card_block(self, card: Optional[Dict[str, str]] = None, include_world_context: bool = True, 
                       character_core: Optional[CharacterCore] = None) -> str:
        """Generate the canonical <CHAR_CARD> block for system prompts"""
        
        # Use CharacterCore if available for enhanced prompts
        if character_core or self.current_character_core:
            core = character_core or self.current_character_core
            return self._make_core_card_block(core, include_world_context)
        
        # Fall back to legacy format
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
    
    def _make_core_card_block(self, core: CharacterCore, include_world_context: bool = True) -> str:
        """Generate card block from CharacterCore (enhanced format)"""
        lines = ["### <CHAR_CARD>"]
        lines.append(f"Name: {core.name}")
        
        # Core description
        lines.append(f"Description: {core.description}")
        
        # Add appearance if available (as requested in task)
        if core.appearance:
            lines.append(f"Appearance: {core.appearance}")
        
        # Scenario
        if core.scenario:
            lines.append(f"Scenario: {core.scenario}")
        
        # Backstory
        if core.backstory:
            lines.append(f"Backstory: {core.backstory}")
        
        # Big Five personality traits summary
        traits = core.personality_traits
        trait_summary = (f"Openness: {traits.openness:.1f}, "
                        f"Conscientiousness: {traits.conscientiousness:.1f}, "
                        f"Extraversion: {traits.extraversion:.1f}, "
                        f"Agreeableness: {traits.agreeableness:.1f}, "
                        f"Neuroticism: {traits.neuroticism:.1f}")
        lines.append(f"Personality (Big-5): {trait_summary}")
        
        # Goals
        if core.goals:
            lines.append(f"Goals: {', '.join(core.goals)}")
        
        # Key relationships
        if core.relationships:
            rel_strs = [f"{rel.name} ({rel.affinity:+d})" for rel in core.relationships[:3]]
            lines.append(f"Key Relationships: {', '.join(rel_strs)}")
        
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
    
    def get_character_summary(self, card: Optional[Dict[str, str]] = None, 
                             character_core: Optional[CharacterCore] = None) -> Dict[str, Any]:
        """Get a summary of character statistics"""
        
        # Use CharacterCore if available
        if character_core or self.current_character_core:
            core = character_core or self.current_character_core
            return {
                'name': core.name,
                'has_description': bool(core.description),
                'has_personality': True,  # Always has Big Five
                'has_scenario': bool(core.scenario),
                'has_appearance': bool(core.appearance),
                'has_backstory': bool(core.backstory),
                'description_length': len(core.description),
                'backstory_length': len(core.backstory),
                'appearance_length': len(core.appearance),
                'goals_count': len(core.goals),
                'relationships_count': len(core.relationships),
                'tags_count': len(core.tags),
                'current_world': self.current_world,
                'format': 'character_core'
            }
        
        # Fall back to legacy format
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
            'current_world': self.current_world,
            'format': 'legacy'
        }
        
        return summary 