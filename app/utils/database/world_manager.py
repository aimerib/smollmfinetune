"""
Database-powered World Manager for the Character Creation Platform.

Replaces the file-based WorldManager with SQLAlchemy database operations
while maintaining the same interface for backward compatibility.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import (
    World, WorldFact, WorldTimeline, WorldFaction, 
    WorldPlace, WorldNPC, WorldEvent, User
)
from .session import session_scope, transaction_scope
from ..world import WorldLore, TimelineEvent, Faction, Place, NPC, PlaceEvent

logger = logging.getLogger(__name__)


class DatabaseWorldManager:
    """
    Database-powered world manager using SQLAlchemy models.
    
    Provides the same interface as the file-based WorldManager but
    stores all data in the database for better consistency and querying.
    """
    
    def __init__(self, user_id: int = 1):
        """
        Initialize the database world manager.
        
        Args:
            user_id: ID of the user who owns the worlds (default: 1)
        """
        self.user_id = user_id
        self.current_world_id: Optional[int] = None
        self.current_lore: Optional[WorldLore] = None
        self.current_tokens: Optional[List[Dict[str, Any]]] = None
        
        logger.info(f"Database world manager initialized for user {user_id}")
    
    def list_worlds(self) -> List[str]:
        """List all available world names for the current user"""
        with session_scope() as session:
            worlds = session.query(World).filter_by(owner_id=self.user_id).order_by(World.name).all()
            return [world.name for world in worlds]
    
    def create_world(self, world_name: str) -> bool:
        """
        Create a new world in the database.
        
        Args:
            world_name: Name of the world to create
            
        Returns:
            True if successful, False if world already exists
        """
        if not world_name or not world_name.strip():
            raise ValueError("World name cannot be empty")
        
        # Sanitize world name
        safe_name = "".join(c for c in world_name if c.isalnum() or c in "._- ").strip()
        if not safe_name:
            raise ValueError("World name contains only invalid characters")
        
        try:
            with transaction_scope() as session:
                # Check if world already exists
                existing_world = session.query(World).filter_by(
                    name=safe_name, owner_id=self.user_id
                ).first()
                
                if existing_world:
                    logger.warning(f"World '{safe_name}' already exists")
                    return False
                
                # Create new world
                world = World(
                    name=safe_name,
                    owner_id=self.user_id,
                    description=f"World: {safe_name}",
                    version=1,
                    meta_data={"version": 1},
                    control_tokens=self._get_default_tokens()
                )
                session.add(world)
                session.flush()  # Get the world ID
                
                # Add default facts
                default_fact = WorldFact(
                    world_id=world.id,
                    key="world_name",
                    value=safe_name
                )
                session.add(default_fact)
                
                logger.info(f"Created new world: {safe_name}")
                return True
                
        except IntegrityError as e:
            logger.error(f"Failed to create world '{safe_name}': {e}")
            return False
    
    def load_world(self, world_name: str) -> Optional[WorldLore]:
        """
        Load world lore from database.
        
        Args:
            world_name: Name of the world to load
            
        Returns:
            WorldLore object if successful, None otherwise
        """
        try:
            with session_scope() as session:
                # Find the world
                world = session.query(World).filter_by(
                    name=world_name, owner_id=self.user_id
                ).first()
                
                if not world:
                    logger.error(f"World '{world_name}' not found")
                    return None
                
                # Load world data and convert to WorldLore format
                world_lore = self._convert_db_to_lore(session, world)
                
                self.current_world_id = world.id
                self.current_lore = world_lore
                self.current_tokens = world.control_tokens or []
                
                logger.info(f"Loaded world: {world_name}")
                return world_lore
                
        except Exception as e:
            logger.error(f"Failed to load world {world_name}: {e}")
            return None
    
    def save_world_lore(self, world_name: Optional[str] = None, lore: Optional[WorldLore] = None) -> bool:
        """
        Save world lore to database, auto-incrementing version.
        
        Args:
            world_name: Name of world to save (uses current if None)
            lore: WorldLore object to save (uses current if None)
            
        Returns:
            True if successful, False otherwise
        """
        target_lore = lore or self.current_lore
        
        if not target_lore:
            logger.error("No world lore specified for saving")
            return False
        
        try:
            with transaction_scope() as session:
                # Find the world
                if world_name:
                    world = session.query(World).filter_by(
                        name=world_name, owner_id=self.user_id
                    ).first()
                elif self.current_world_id:
                    world = session.query(World).filter_by(id=self.current_world_id).first()
                else:
                    logger.error("No world specified for saving")
                    return False
                
                if not world:
                    logger.error(f"World not found for saving")
                    return False
                
                # Update world version and metadata
                world.version += 1
                world.meta_data = target_lore.meta
                world.updated_at = datetime.now(timezone.utc)
                
                # Clear existing data
                session.query(WorldFact).filter_by(world_id=world.id).delete()
                session.query(WorldTimeline).filter_by(world_id=world.id).delete()
                session.query(WorldFaction).filter_by(world_id=world.id).delete()
                session.query(WorldPlace).filter_by(world_id=world.id).delete()
                
                # Save facts
                for key, value in target_lore.facts.items():
                    fact = WorldFact(world_id=world.id, key=key, value=value)
                    session.add(fact)
                
                # Save timeline events
                for event in target_lore.timeline:
                    timeline_event = WorldTimeline(
                        world_id=world.id,
                        year=event.year,
                        event=event.event
                    )
                    session.add(timeline_event)
                
                # Save factions
                for faction in target_lore.factions:
                    faction_record = WorldFaction(
                        world_id=world.id,
                        name=faction.name,
                        timeline_events=[
                            {"year": event.year, "event": event.event}
                            for event in faction.timeline
                        ]
                    )
                    session.add(faction_record)
                
                # Save places
                for place in target_lore.places:
                    place_record = WorldPlace(
                        world_id=world.id,
                        name=place.name,
                        description=place.description
                    )
                    session.add(place_record)
                    session.flush()  # Get place ID
                    
                    # Save NPCs
                    for npc in place.npcs:
                        npc_record = WorldNPC(
                            place_id=place_record.id,
                            name=npc.name,
                            description=npc.description
                        )
                        session.add(npc_record)
                    
                    # Save events
                    for event in place.events:
                        event_record = WorldEvent(
                            place_id=place_record.id,
                            name=event.name,
                            description=event.description,
                            is_random=(event.random.lower() == 'true')
                        )
                        session.add(event_record)
                
                self.current_lore = target_lore
                logger.info(f"Saved world lore for {world.name}, version {world.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save world lore: {e}")
            return False
    
    def get_world_path(self, world_name: str) -> Path:
        """
        Get the path to a world directory (for backward compatibility).
        
        Note: This returns a virtual path since data is now in database.
        """
        return Path("database") / "worlds" / world_name
    
    def get_characters_path(self, world_name: str) -> Path:
        """
        Get the path to a world's characters directory (for backward compatibility).
        
        Note: This returns a virtual path since data is now in database.
        """
        return self.get_world_path(world_name) / "characters"
    
    def get_tokens_path(self, world_name: str) -> Path:
        """
        Get the path to a world's tokens.json file (for backward compatibility).
        
        Note: This returns a virtual path since data is now in database.
        """
        return self.get_world_path(world_name) / "tokens.json"
    
    def load_world_tokens(self, world_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load control tokens for a specific world from database"""
        try:
            with session_scope() as session:
                world = session.query(World).filter_by(
                    name=world_name, owner_id=self.user_id
                ).first()
                
                if not world:
                    logger.warning(f"World '{world_name}' not found")
                    return []
                
                tokens = world.control_tokens or []
                logger.info(f"Loaded {len(tokens)} tokens for world {world_name}")
                return tokens
                
        except Exception as e:
            logger.error(f"Failed to load tokens for world {world_name}: {e}")
            return []
    
    def save_world_tokens(self, world_name: str, tokens: List[Dict[str, Any]]) -> bool:
        """Save control tokens for a specific world to database"""
        try:
            with transaction_scope() as session:
                world = session.query(World).filter_by(
                    name=world_name, owner_id=self.user_id
                ).first()
                
                if not world:
                    logger.error(f"World '{world_name}' not found")
                    return False
                
                world.control_tokens = tokens
                world.updated_at = datetime.now(timezone.utc)
                
                logger.info(f"Saved {len(tokens)} tokens for world {world_name}")
                
                # Update current tokens if this is the current world
                if world.id == self.current_world_id:
                    self.current_tokens = tokens
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save tokens for world {world_name}: {e}")
            return False
    
    def get_current_tokens(self) -> List[Dict[str, Any]]:
        """Get control tokens for the current world"""
        if self.current_tokens is not None:
            return self.current_tokens
        
        if self.current_world_id:
            try:
                with session_scope() as session:
                    world = session.query(World).filter_by(id=self.current_world_id).first()
                    if world:
                        self.current_tokens = world.control_tokens or []
                        return self.current_tokens
            except Exception as e:
                logger.error(f"Failed to get current tokens: {e}")
        
        return []
    
    def get_tokens_by_category(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tokens filtered by category"""
        tokens = self.get_current_tokens()
        
        if category is None:
            return tokens
        
        return [token for token in tokens if token.get("category") == category]
    
    def _convert_db_to_lore(self, session: Session, world: World) -> WorldLore:
        """Convert database world model to WorldLore format"""
        # Load facts
        facts = {}
        for fact in session.query(WorldFact).filter_by(world_id=world.id).all():
            facts[fact.key] = fact.value
        
        # Load timeline events
        timeline = []
        for event in session.query(WorldTimeline).filter_by(world_id=world.id).order_by(WorldTimeline.year).all():
            timeline.append(TimelineEvent(year=event.year, event=event.event))
        
        # Load factions
        factions = []
        for faction_record in session.query(WorldFaction).filter_by(world_id=world.id).all():
            faction_timeline = []
            for event_data in faction_record.timeline_events or []:
                faction_timeline.append(TimelineEvent(
                    year=event_data["year"],
                    event=event_data["event"]
                ))
            
            factions.append(Faction(name=faction_record.name, timeline=faction_timeline))
        
        # Load places
        places = []
        for place_record in session.query(WorldPlace).filter_by(world_id=world.id).all():
            # Load NPCs for this place
            npcs = []
            for npc_record in session.query(WorldNPC).filter_by(place_id=place_record.id).all():
                npcs.append(NPC(name=npc_record.name, description=npc_record.description))
            
            # Load events for this place
            events = []
            for event_record in session.query(WorldEvent).filter_by(place_id=place_record.id).all():
                events.append(PlaceEvent(
                    name=event_record.name,
                    description=event_record.description,
                    random="true" if event_record.is_random else "false"
                ))
            
            places.append(Place(
                name=place_record.name,
                description=place_record.description,
                npcs=npcs,
                events=events
            ))
        
        return WorldLore(
            meta=world.meta_data or {"version": world.version},
            facts=facts,
            factions=factions,
            timeline=timeline,
            places=places
        )
    
    def _get_default_tokens(self) -> List[Dict[str, Any]]:
        """Get default control tokens for new worlds"""
        return [
            {"type": "action", "token": "<ACT>", "description": "Action token"},
            {"type": "emotion", "token": "<EMO>", "description": "Emotion token"},
            {"type": "thought", "token": "<THINK>", "description": "Thought token"}
        ] 