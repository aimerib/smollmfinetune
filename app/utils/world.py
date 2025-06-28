import os
import orjson
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """Represents a single timeline event"""
    year: int
    event: str


@dataclass 
class NPC:
    """Represents a non-player character in a place"""
    name: str
    description: str


@dataclass
class PlaceEvent:
    """Represents an event that can happen at a place"""
    name: str
    description: str
    random: str = "false"  # String to match JSON schema


@dataclass
class Place:
    """Represents a place of interest in the world"""
    name: str
    description: str
    npcs: List[NPC] = None
    events: List[PlaceEvent] = None
    
    def __post_init__(self):
        if self.npcs is None:
            self.npcs = []
        if self.events is None:
            self.events = []


@dataclass
class Faction:
    """Represents a faction with its own timeline"""
    name: str
    timeline: List[TimelineEvent] = None
    
    def __post_init__(self):
        if self.timeline is None:
            self.timeline = []


@dataclass
class WorldLore:
    """Complete world lore data structure"""
    meta: Dict[str, Any]
    facts: Dict[str, str] = None
    factions: List[Faction] = None
    timeline: List[TimelineEvent] = None
    places: List[Place] = None
    
    def __post_init__(self):
        if self.facts is None:
            self.facts = {}
        if self.factions is None:
            self.factions = []
        if self.timeline is None:
            self.timeline = []
        if self.places is None:
            self.places = []


class WorldManager:
    """Manages world directories, lore, character organization, and control tokens"""
    
    def __init__(self, worlds_root: Optional[str] = None):
        """Initialize WorldManager with configurable root path"""
        self.worlds_root = Path(worlds_root or os.environ.get('WORLDS_PATH', 'content/worlds'))
        self.current_world: Optional[str] = None
        self.current_lore: Optional[WorldLore] = None
        self.current_tokens: Optional[List[Dict[str, Any]]] = None
        
        # Ensure worlds directory exists
        self.worlds_root.mkdir(parents=True, exist_ok=True)
    
    def list_worlds(self) -> List[str]:
        """List all available world names"""
        if not self.worlds_root.exists():
            return []
        
        worlds = []
        for path in self.worlds_root.iterdir():
            if path.is_dir() and (path / "world_lore.json").exists():
                worlds.append(path.name)
        
        return sorted(worlds)
    
    def create_world(self, world_name: str) -> bool:
        """Create a new world directory structure"""
        if not world_name or not world_name.strip():
            raise ValueError("World name cannot be empty")
        
        # Sanitize world name for filesystem
        safe_name = "".join(c for c in world_name if c.isalnum() or c in "._- ").strip()
        if not safe_name:
            raise ValueError("World name contains only invalid characters")
        
        world_path = self.worlds_root / safe_name
        
        if world_path.exists():
            logger.warning(f"World '{safe_name}' already exists")
            return False
        
        # Create world directory structure
        world_path.mkdir(parents=True, exist_ok=True)
        characters_path = world_path / "characters"
        characters_path.mkdir(exist_ok=True)
        
        # Create initial world lore
        initial_lore = WorldLore(
            meta={"version": 1},
            facts={},
            factions=[],
            timeline=[],
            places=[]
        )
        
        self._write_lore_file(world_path / "world_lore.json", initial_lore)
        
        # Copy default control tokens to the new world
        self._copy_default_tokens(world_path)
        
        logger.info(f"Created new world: {safe_name}")
        return True
    
    def load_world(self, world_name: str) -> Optional[WorldLore]:
        """Load world lore from disk"""
        world_path = self.worlds_root / world_name
        lore_path = world_path / "world_lore.json"
        
        if not lore_path.exists():
            logger.error(f"World lore file not found: {lore_path}")
            return None
        
        try:
            with open(lore_path, 'rb') as f:
                data = orjson.loads(f.read())
            
            lore = self._deserialize_lore(data)
            self.current_world = world_name
            self.current_lore = lore
            
            # Load world-specific tokens
            self.current_tokens = self.load_world_tokens(world_name)
            
            logger.info(f"Loaded world: {world_name}")
            return lore
            
        except Exception as e:
            logger.error(f"Failed to load world {world_name}: {e}")
            return None
    
    def save_world_lore(self, world_name: Optional[str] = None, lore: Optional[WorldLore] = None) -> bool:
        """Save world lore to disk, auto-incrementing version"""
        target_world = world_name or self.current_world
        target_lore = lore or self.current_lore
        
        if not target_world or not target_lore:
            logger.error("No world or lore specified for saving")
            return False
        
        world_path = self.worlds_root / target_world
        if not world_path.exists():
            logger.error(f"World directory does not exist: {world_path}")
            return False
        
        # Auto-increment version
        target_lore.meta["version"] = target_lore.meta.get("version", 0) + 1
        
        lore_path = world_path / "world_lore.json"
        
        try:
            self._write_lore_file(lore_path, target_lore)
            self.current_lore = target_lore
            logger.info(f"Saved world lore for {target_world}, version {target_lore.meta['version']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save world lore: {e}")
            return False
    
    def get_world_path(self, world_name: str) -> Path:
        """Get the full path to a world directory"""
        return self.worlds_root / world_name
    
    def get_characters_path(self, world_name: str) -> Path:
        """Get the path to a world's characters directory"""
        return self.get_world_path(world_name) / "characters"
    
    def get_tokens_path(self, world_name: str) -> Path:
        """Get the path to a world's tokens.json file"""
        return self.get_world_path(world_name) / "tokens.json"
    
    def _copy_default_tokens(self, world_path: Path) -> bool:
        """Copy default tokens.json to a new world directory"""
        default_tokens_path = Path("core_data/tokens.json")
        world_tokens_path = world_path / "tokens.json"
        
        try:
            if default_tokens_path.exists():
                shutil.copy2(default_tokens_path, world_tokens_path)
                logger.info(f"Copied default tokens to {world_tokens_path}")
                return True
            else:
                # Create empty tokens file as fallback
                with open(world_tokens_path, 'w') as f:
                    f.write("[]")
                logger.warning(f"Default tokens not found, created empty tokens file: {world_tokens_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to copy default tokens: {e}")
            return False
    
    def load_world_tokens(self, world_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load control tokens for a specific world"""
        tokens_path = self.get_tokens_path(world_name)
        
        if not tokens_path.exists():
            logger.warning(f"Tokens file not found for world {world_name}: {tokens_path}")
            return []
        
        try:
            with open(tokens_path, 'r') as f:
                tokens_data = orjson.loads(f.read())
            
            logger.info(f"Loaded {len(tokens_data)} tokens for world {world_name}")
            return tokens_data
            
        except Exception as e:
            logger.error(f"Failed to load tokens for world {world_name}: {e}")
            return []
    
    def save_world_tokens(self, world_name: str, tokens: List[Dict[str, Any]]) -> bool:
        """Save control tokens for a specific world"""
        tokens_path = self.get_tokens_path(world_name)
        
        try:
            with open(tokens_path, 'wb') as f:
                f.write(orjson.dumps(tokens, option=orjson.OPT_INDENT_2))
            
            logger.info(f"Saved {len(tokens)} tokens for world {world_name}")
            
            # Update current tokens if this is the current world
            if world_name == self.current_world:
                self.current_tokens = tokens
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tokens for world {world_name}: {e}")
            return False
    
    def get_current_tokens(self) -> List[Dict[str, Any]]:
        """Get control tokens for the current world"""
        if self.current_tokens is not None:
            return self.current_tokens
        
        if self.current_world:
            self.current_tokens = self.load_world_tokens(self.current_world)
            return self.current_tokens or []
        
        return []
    
    def get_tokens_by_category(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tokens filtered by category"""
        tokens = self.get_current_tokens()
        
        if category is None:
            return tokens
        
        return [token for token in tokens if token.get("category") == category]
    
    def _write_lore_file(self, path: Path, lore: WorldLore) -> None:
        """Write world lore to JSON file"""
        # Convert dataclasses to dict, handling nested structures
        data = self._serialize_lore(lore)
        
        with open(path, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    
    def _serialize_lore(self, lore: WorldLore) -> Dict[str, Any]:
        """Convert WorldLore dataclass to JSON-serializable dict"""
        data = {
            "meta": lore.meta,
            "facts": lore.facts,
            "factions": [],
            "timeline": [],
            "places": []
        }
        
        # Serialize factions
        for faction in lore.factions:
            faction_data = {
                "name": faction.name,
                "timeline": [{"year": event.year, "event": event.event} for event in faction.timeline]
            }
            data["factions"].append(faction_data)
        
        # Serialize timeline
        for event in lore.timeline:
            data["timeline"].append({"year": event.year, "event": event.event})
        
        # Serialize places
        for place in lore.places:
            place_data = {
                "name": place.name,
                "description": place.description,
                "npcs": [{"name": npc.name, "description": npc.description} for npc in place.npcs],
                "events": [{"name": event.name, "description": event.description, "random": event.random} for event in place.events]
            }
            data["places"].append(place_data)
        
        return data
    
    def _deserialize_lore(self, data: Dict[str, Any]) -> WorldLore:
        """Convert JSON dict to WorldLore dataclass"""
        # Deserialize factions
        factions = []
        for faction_data in data.get("factions", []):
            timeline_events = [
                TimelineEvent(year=event["year"], event=event["event"])
                for event in faction_data.get("timeline", [])
            ]
            factions.append(Faction(name=faction_data["name"], timeline=timeline_events))
        
        # Deserialize main timeline
        timeline = [
            TimelineEvent(year=event["year"], event=event["event"])
            for event in data.get("timeline", [])
        ]
        
        # Deserialize places
        places = []
        for place_data in data.get("places", []):
            npcs = [
                NPC(name=npc["name"], description=npc["description"])
                for npc in place_data.get("npcs", [])
            ]
            events = [
                PlaceEvent(
                    name=event["name"], 
                    description=event["description"],
                    random=event.get("random", "false")
                )
                for event in place_data.get("events", [])
            ]
            places.append(Place(
                name=place_data["name"],
                description=place_data["description"],
                npcs=npcs,
                events=events
            ))
        
        return WorldLore(
            meta=data.get("meta", {"version": 1}),
            facts=data.get("facts", {}),
            factions=factions,
            timeline=timeline,
            places=places
        ) 