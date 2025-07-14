"""
Runtime State Manager for Narrative Engine

Manages dynamic world state with support for atomic transactions,
event logging, and complex queries. Integrates with our memory system
to persist character memories and emotional states.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum
import threading
import logging
import asyncio
from copy import deepcopy
import json

logger = logging.getLogger(__name__)


class StateManagerError(Exception):
    """Base exception for state manager errors"""
    pass


class TransactionError(StateManagerError):
    """Raised when a transaction fails"""
    pass


class BackendType(Enum):
    """Supported backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    MONGODB = "mongodb"


@dataclass
class EntityState:
    """
    Represents the complete state of an entity (character, item, location).
    """
    entity_id: str
    entity_type: str  # character, item, location, etc.
    location: Optional[str] = None
    inventory: List[str] = field(default_factory=list)
    status_effects: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    memories: List[Dict[str, Any]] = field(default_factory=list)  # Integration with memory system!
    custom_data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime to ISO format
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityState':
        """Create from dictionary"""
        # Convert ISO datetime string back to datetime
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)
    
    def apply_update(self, changes: Dict[str, Any]) -> None:
        """Apply changes to this entity state"""
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store unknown fields in custom_data
                self.custom_data[key] = value
        self.last_updated = datetime.now()


@dataclass
class StateUpdate:
    """Represents an update to an entity's state"""
    entity_id: str
    changes: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StateTransaction:
    """Represents an atomic transaction of multiple state updates"""
    updates: List[StateUpdate]
    transaction_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EventLog:
    """Represents a logged state change event"""
    event_id: str
    event_type: str  # entity_created, entity_updated, entity_deleted, SpeakToAction, etc.
    entity_id: str
    changes: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None  # Additional event details
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateQuery:
    """Query parameters for finding entities"""
    entity_type: Optional[str] = None
    location: Optional[str] = None
    custom_filters: Optional[Callable[[EntityState], bool]] = None
    limit: Optional[int] = None


@dataclass
class SubtextEntry:
    """Represents a character's internal monologue entry"""
    agent_id: str
    subtext: str
    timestamp: datetime = field(default_factory=datetime.now)
    entry_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "subtext": self.subtext,
            "timestamp": self.timestamp.isoformat(),
            "entry_id": self.entry_id
        }


class InMemoryBackend:
    """In-memory backend for development and testing"""
    
    def __init__(self):
        self.entities: Dict[str, EntityState] = {}
        self.events: List[EventLog] = []
        self.subtext_log: List[SubtextEntry] = []  # Store subtext entries
        self.lock = threading.RLock()
        self._event_counter = 0
        self._subtext_counter = 0
    
    def create_entity(self, entity: EntityState) -> None:
        """Create a new entity"""
        with self.lock:
            if entity.entity_id in self.entities:
                raise StateManagerError(f"Entity {entity.entity_id} already exists")
            
            self.entities[entity.entity_id] = deepcopy(entity)
            self._log_event("entity_created", entity.entity_id)
    
    def get_entity(self, entity_id: str) -> Optional[EntityState]:
        """Get an entity by ID"""
        with self.lock:
            entity = self.entities.get(entity_id)
            return deepcopy(entity) if entity else None
    
    def update_entity(self, entity_id: str, changes: Dict[str, Any]) -> None:
        """Update an entity"""
        with self.lock:
            if entity_id not in self.entities:
                raise StateManagerError(f"Entity {entity_id} not found")
            
            entity = self.entities[entity_id]
            old_state = deepcopy(entity.to_dict())
            entity.apply_update(changes)
            
            self._log_event("entity_updated", entity_id, changes)
    
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity"""
        with self.lock:
            if entity_id not in self.entities:
                raise StateManagerError(f"Entity {entity_id} not found")
            
            del self.entities[entity_id]
            self._log_event("entity_deleted", entity_id)
    
    def query_entities(self, query: StateQuery) -> List[EntityState]:
        """Query entities based on criteria"""
        with self.lock:
            results = []
            
            for entity in self.entities.values():
                # Check entity type
                if query.entity_type and entity.entity_type != query.entity_type:
                    continue
                
                # Check location
                if query.location and entity.location != query.location:
                    continue
                
                # Apply custom filters
                if query.custom_filters and not query.custom_filters(entity):
                    continue
                
                results.append(deepcopy(entity))
                
                # Check limit
                if query.limit and len(results) >= query.limit:
                    break
            
            return results
    
    def get_events(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[EventLog]:
        """Get event logs"""
        with self.lock:
            events = self.events
            
            if since:
                events = [e for e in events if e.timestamp >= since]
            
            if limit:
                events = events[-limit:]
            
            return deepcopy(events)
    
    def _log_event(self, event_type: str, entity_id: str, changes: Optional[Dict[str, Any]] = None):
        """Log an event"""
        self._event_counter += 1
        event = EventLog(
            event_id=f"event_{self._event_counter}",
            event_type=event_type,
            entity_id=entity_id,
            changes=changes
        )
        self.events.append(event)
    
    def add_subtext(self, agent_id: str, subtext: str, timestamp: Optional[datetime] = None) -> str:
        """Add a subtext entry to the log"""
        with self.lock:
            self._subtext_counter += 1
            entry_id = f"subtext_{self._subtext_counter}"
            
            entry = SubtextEntry(
                agent_id=agent_id,
                subtext=subtext,
                timestamp=timestamp or datetime.now(),
                entry_id=entry_id
            )
            
            self.subtext_log.append(entry)
            return entry_id
    
    def get_subtext(self, agent_id: Optional[str] = None, since: Optional[datetime] = None, 
                   limit: Optional[int] = None) -> List[SubtextEntry]:
        """Get subtext entries, optionally filtered"""
        with self.lock:
            entries = self.subtext_log
            
            # Filter by agent
            if agent_id:
                entries = [e for e in entries if e.agent_id == agent_id]
            
            # Filter by time
            if since:
                entries = [e for e in entries if e.timestamp >= since]
            
            # Apply limit
            if limit:
                entries = entries[-limit:]
            
            return deepcopy(entries)


class StateManager:
    """
    Main state manager that handles entity state, transactions, and queries.
    """
    
    def __init__(self, backend: Union[str, BackendType] = BackendType.MEMORY):
        """
        Initialize state manager with specified backend.
        
        Args:
            backend: Backend type (memory, redis, mongodb)
        """
        self.backend_type = BackendType(backend) if isinstance(backend, str) else backend
        
        # Initialize backend
        if self.backend_type == BackendType.MEMORY:
            self.backend = InMemoryBackend()
        else:
            raise NotImplementedError(f"Backend {self.backend_type} not yet implemented")
        
        # Event subscription system
        self._event_subscribers: List[Callable[[EventLog], None]] = []
        
        logger.info(f"Initialized StateManager with {self.backend_type.value} backend")
    
    def create_entity(self, entity: EntityState) -> None:
        """Create a new entity"""
        self.backend.create_entity(entity)
        logger.debug(f"Created entity {entity.entity_id} of type {entity.entity_type}")
    
    def get_entity(self, entity_id: str) -> Optional[EntityState]:
        """Get an entity by ID"""
        return self.backend.get_entity(entity_id)
    
    def update_entity(self, update: StateUpdate) -> None:
        """Update an entity's state"""
        self.backend.update_entity(update.entity_id, update.changes)
        logger.debug(f"Updated entity {update.entity_id}")
    
    def atomic_update(self, entity_id: str, update_func: Callable[[EntityState], Dict[str, Any]]) -> None:
        """
        Perform an atomic read-modify-write operation.
        
        Args:
            entity_id: The entity to update
            update_func: Function that takes current state and returns changes
        """
        if hasattr(self.backend, 'atomic_update'):
            self.backend.atomic_update(entity_id, update_func)
        else:
            # Fallback for backends without atomic update
            with self.backend.lock:
                entity = self.backend.entities.get(entity_id)
                if not entity:
                    raise StateManagerError(f"Entity {entity_id} not found")
                
                changes = update_func(deepcopy(entity))
                entity.apply_update(changes)
                self.backend._log_event("entity_updated", entity_id, changes)
    
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity"""
        self.backend.delete_entity(entity_id)
        logger.debug(f"Deleted entity {entity_id}")
    
    def query_by_location(self, location: str) -> List[EntityState]:
        """Query all entities at a specific location"""
        query = StateQuery(location=location)
        return self.backend.query_entities(query)
    
    def query(self, query: StateQuery) -> List[EntityState]:
        """Execute a complex query"""
        return self.backend.query_entities(query)
    
    def execute_transaction(self, transaction: StateTransaction) -> None:
        """
        Execute an atomic transaction.
        All updates succeed or all fail.
        """
        # Validate all entities exist
        for update in transaction.updates:
            if not self.get_entity(update.entity_id):
                raise TransactionError(f"Entity {update.entity_id} not found")
        
        # Save current states for rollback
        original_states = {}
        for update in transaction.updates:
            entity = self.get_entity(update.entity_id)
            original_states[update.entity_id] = entity.to_dict() if entity else None
        
        try:
            # Apply all updates
            for update in transaction.updates:
                self.update_entity(update)
                
            logger.info(f"Transaction completed with {len(transaction.updates)} updates")
            
        except Exception as e:
            # Rollback on any failure
            logger.error(f"Transaction failed, rolling back: {e}")
            
            for entity_id, original_state in original_states.items():
                if original_state:
                    # Restore original state
                    entity = EntityState.from_dict(original_state)
                    # Direct backend update to avoid additional events
                    self.backend.entities[entity_id] = entity
            
            raise TransactionError(f"Transaction failed: {e}") from e
    
    def get_recent_events(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[EventLog]:
        """Get recent events from the event log"""
        return self.backend.get_events(since=since, limit=limit)
    
    def load_world_state(self, world_data: Dict[str, Any]) -> None:
        """
        Load initial state from WorldManager data.
        Creates entities for NPCs and places.
        """
        places = world_data.get("places", [])
        
        for place in places:
            place_name = place["name"]
            
            # Create NPCs as entities
            for npc in place.get("npcs", []):
                entity = EntityState(
                    entity_id=f"npc_{npc['name'].lower().replace(' ', '_')}",
                    entity_type="character",
                    location=place_name,
                    custom_data={
                        "name": npc["name"],
                        "description": npc.get("description", ""),
                        "is_npc": True
                    }
                )
                
                try:
                    self.create_entity(entity)
                except StateManagerError:
                    # Entity might already exist
                    logger.warning(f"NPC {npc['name']} already exists")
        
        logger.info(f"Loaded world state with {len(places)} places")
    
    def add_memory_to_character(self, character_id: str, memory: Dict[str, Any]) -> None:
        """
        Add a memory to a character's memory list.
        This integrates with our memory system!
        """
        entity = self.get_entity(character_id)
        if not entity:
            raise StateManagerError(f"Character {character_id} not found")
        
        if entity.entity_type != "character":
            raise StateManagerError(f"Entity {character_id} is not a character")
        
        # Get current memories
        memories = entity.memories.copy()
        memories.append(memory)
        
        # Update entity
        update = StateUpdate(
            entity_id=character_id,
            changes={"memories": memories}
        )
        self.update_entity(update)
        
        logger.info(f"Added memory to character {character_id}: {memory.get('content', 'No content')}")
    
    def get_character_memories(self, character_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get memories for a character, optionally limited to most recent"""
        entity = self.get_entity(character_id)
        if not entity:
            return []
        
        memories = entity.memories
        if limit:
            memories = memories[-limit:]
        
        return memories 
    
    def add_subtext(self, agent_id: str, subtext: str, timestamp: Optional[datetime] = None) -> str:
        """
        Add a subtext entry for a character's internal monologue.
        
        Args:
            agent_id: The ID of the agent/character
            subtext: The internal monologue text
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Entry ID for the subtext entry
        """
        return self.backend.add_subtext(agent_id, subtext, timestamp)
    
    def get_subtext(self, agent_id: Optional[str] = None, since: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[SubtextEntry]:
        """
        Get subtext entries, optionally filtered.
        
        Args:
            agent_id: Filter by specific agent (optional)
            since: Filter entries since this timestamp (optional)
            limit: Maximum number of entries to return (optional)
            
        Returns:
            List of SubtextEntry objects
        """
        return self.backend.get_subtext(agent_id, since, limit)
    
    def get_recent_subtext(self, agent_id: str, limit: int = 10) -> List[SubtextEntry]:
        """
        Get the most recent subtext entries for a specific agent.
        
        Args:
            agent_id: The agent to get subtext for
            limit: Maximum number of entries (default 10)
            
        Returns:
            List of recent SubtextEntry objects
        """
        return self.get_subtext(agent_id=agent_id, limit=limit)
    
    def subscribe_to_events(self, callback: Callable[[EventLog], None]) -> None:
        """
        Subscribe to state events.
        
        Args:
            callback: Function to call when events occur
        """
        self._event_subscribers.append(callback)
        logger.debug(f"Added event subscriber, total: {len(self._event_subscribers)}")
    
    def unsubscribe_from_events(self, callback: Callable[[EventLog], None]) -> None:
        """
        Unsubscribe from state events.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)
            logger.debug(f"Removed event subscriber, total: {len(self._event_subscribers)}")
    
    def log_event(self, event_type: str, entity_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an event and notify subscribers.
        
        Args:
            event_type: Type of event (e.g., "SpeakToAction")
            entity_id: ID of the entity involved
            details: Additional event details
        """
        # Create event log
        event_log = EventLog(
            event_id=f"event_{datetime.now().isoformat()}",
            event_type=event_type,
            entity_id=entity_id,
            details=details,
            metadata=details or {}
        )
        
        # Add to backend events
        self.backend.events.append(event_log)
        
        # Notify subscribers
        for subscriber in self._event_subscribers:
            try:
                # Call subscribers asynchronously if they're async
                if asyncio.iscoroutinefunction(subscriber):
                    asyncio.create_task(subscriber(event_log))
                else:
                    subscriber(event_log)
            except Exception as e:
                logger.error(f"Error notifying event subscriber: {e}")
        
        logger.debug(f"Logged event {event_type} for entity {entity_id}")