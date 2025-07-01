"""
Session State Management for persistent character interactions.

Manages session lifecycle, character states, and cross-request persistence
for the production inference engine.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

logger = logging.getLogger(__name__)


@dataclass
class CharacterState:
    """State for a character within a session"""
    character_id: str
    mood: str = "neutral"
    location: str = "unknown"
    talking_to: List[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    custom_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """Complete state for a play session"""
    session_id: str
    user_id: str
    character_ids: List[str]
    world_id: Optional[str] = None
    status: str = "active"  # active, paused, ended
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    conversation_turn: int = 0
    current_location: str = "default"
    relationship_scores: Dict[str, float] = field(default_factory=dict)
    session_metadata: Dict[str, Any] = field(default_factory=dict)


class SessionNotFoundError(Exception):
    """Raised when session is not found"""
    pass


class SessionStateManager:
    """
    Manages session state across inference requests.
    
    Features:
    - Persistent session state
    - Character state coordination
    - Memory management per session
    - Analytics and monitoring
    - Clean lifecycle management
    """
    
    def __init__(self, storage_backend: str = "memory"):
        """
        Initialize session manager.
        
        Args:
            storage_backend: Storage backend (memory, redis, dynamodb)
        """
        self.storage_backend = storage_backend
        
        # In-memory storage
        self.sessions: Dict[str, SessionState] = {}
        self.character_states: Dict[str, Dict[str, CharacterState]] = {}  # session_id -> char_id -> state
        self.session_memories: Dict[str, List[str]] = {}  # session_id -> memory_ids
        self.session_analytics: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> events
        
        logger.info(f"SessionStateManager initialized with {storage_backend} backend")
    
    async def create_session(self, user_id: str, character_ids: List[str],
                           world_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_id: User identifier
            character_ids: List of character IDs in session
            world_id: Optional world identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        # Create session state
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            character_ids=character_ids,
            world_id=world_id
        )
        
        self.sessions[session_id] = session
        
        # Initialize character states
        self.character_states[session_id] = {}
        for char_id in character_ids:
            self.character_states[session_id][char_id] = CharacterState(
                character_id=char_id
            )
        
        # Initialize tracking
        self.session_memories[session_id] = []
        self.session_analytics[session_id] = []
        
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session_id
    
    async def get_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state dictionary
            
        Raises:
            SessionNotFoundError: If session not found
        """
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "character_ids": session.character_ids,
            "world_id": session.world_id,
            "status": session.status,
            "conversation_turn": session.conversation_turn,
            "current_location": session.current_location,
            "relationship_scores": session.relationship_scores,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "metadata": session.session_metadata
        }
    
    async def update_state(self, session_id: str, updates: Dict[str, Any]):
        """
        Update session state.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates
        """
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                session.session_metadata[key] = value
        
        # Update activity timestamp
        session.last_activity = time.time()
        
        logger.debug(f"Updated session {session_id} state")
    
    async def get_character_state(self, session_id: str, character_id: str) -> Dict[str, Any]:
        """Get character state within session"""
        if session_id not in self.character_states:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        if character_id not in self.character_states[session_id]:
            raise ValueError(f"Character {character_id} not in session")
        
        char_state = self.character_states[session_id][character_id]
        
        return {
            "character_id": char_state.character_id,
            "mood": char_state.mood,
            "location": char_state.location,
            "talking_to": char_state.talking_to,
            "last_active": char_state.last_active,
            "custom_state": char_state.custom_state
        }
    
    async def set_character_state(self, session_id: str, character_id: str,
                                state_updates: Dict[str, Any]):
        """Update character state within session"""
        if session_id not in self.character_states:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        if character_id not in self.character_states[session_id]:
            self.character_states[session_id][character_id] = CharacterState(
                character_id=character_id
            )
        
        char_state = self.character_states[session_id][character_id]
        
        # Update fields
        for key, value in state_updates.items():
            if hasattr(char_state, key):
                setattr(char_state, key, value)
            else:
                char_state.custom_state[key] = value
        
        char_state.last_active = time.time()
    
    async def get_scene_state(self, session_id: str) -> Dict[str, Any]:
        """Get coordinated scene state for all characters"""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        char_states = self.character_states.get(session_id, {})
        
        # Determine active location (where most characters are)
        location_counts = {}
        characters_present = []
        
        for char_id, char_state in char_states.items():
            loc = char_state.location
            location_counts[loc] = location_counts.get(loc, 0) + 1
            if time.time() - char_state.last_active < 300:  # Active in last 5 min
                characters_present.append(char_id)
        
        current_location = max(location_counts.items(), key=lambda x: x[1])[0] if location_counts else "unknown"
        
        # Find active conversations
        active_conversations = []
        processed_pairs = set()
        
        for char_id, char_state in char_states.items():
            for other_char in char_state.talking_to:
                pair = tuple(sorted([char_id, other_char]))
                if pair not in processed_pairs:
                    active_conversations.append(list(pair))
                    processed_pairs.add(pair)
        
        return {
            "session_id": session_id,
            "location": current_location,
            "characters_present": characters_present,
            "active_conversations": active_conversations,
            "conversation_turn": session.conversation_turn
        }
    
    async def add_session_memory(self, session_id: str, character_id: str,
                               content: str, embedding: List[float],
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory to session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
        
        memory_id = str(uuid.uuid4())
        
        # Track memory ID (actual storage handled by MemoryService)
        self.session_memories[session_id].append(memory_id)
        
        # Log analytics event
        await self.log_interaction(session_id, {
            "type": "memory_created",
            "character_id": character_id,
            "memory_id": memory_id,
            "content_length": len(content)
        })
        
        return memory_id
    
    async def get_session_memories(self, session_id: str, limit: int = 50) -> List[str]:
        """Get memory IDs for session"""
        if session_id not in self.session_memories:
            return []
        
        memories = self.session_memories[session_id]
        return memories[-limit:] if len(memories) > limit else memories
    
    async def clear_old_memories(self, session_id: str, keep_recent: int = 10) -> int:
        """Clear old memories from session"""
        if session_id not in self.session_memories:
            return 0
        
        memories = self.session_memories[session_id]
        if len(memories) <= keep_recent:
            return 0
        
        to_remove = len(memories) - keep_recent
        self.session_memories[session_id] = memories[-keep_recent:]
        
        return to_remove
    
    async def log_interaction(self, session_id: str, event: Dict[str, Any]):
        """Log an interaction event for analytics"""
        if session_id not in self.session_analytics:
            self.session_analytics[session_id] = []
        
        event["timestamp"] = time.time()
        self.session_analytics[session_id].append(event)
        
        # Update session activity
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = time.time()
            
            # Increment turn counter for messages
            if event.get("type") == "message":
                self.sessions[session_id].conversation_turn += 1
    
    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for session"""
        if session_id not in self.session_analytics:
            return {}
        
        events = self.session_analytics[session_id]
        if not events:
            return {
                "total_interactions": 0,
                "avg_response_time_ms": 0,
                "total_tokens_generated": 0
            }
        
        # Calculate metrics
        total_interactions = len(events)
        message_events = [e for e in events if e.get("type") == "message"]
        
        response_times = [e.get("response_time_ms", 0) for e in message_events]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        total_tokens = sum(e.get("tokens_generated", 0) for e in message_events)
        
        # Find peak activity
        if events:
            # Group by hour
            hour_counts = {}
            for event in events:
                hour = int(event["timestamp"] // 3600)
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
            peak_time = datetime.fromtimestamp(peak_hour * 3600).isoformat()
        else:
            peak_time = None
        
        return {
            "total_interactions": total_interactions,
            "avg_response_time_ms": avg_response_time,
            "total_tokens_generated": total_tokens,
            "peak_activity_time": peak_time,
            "user_messages": sum(1 for e in events if e.get("type") == "message" and e.get("role") == "user"),
            "assistant_messages": sum(1 for e in events if e.get("type") == "message" and e.get("role") == "assistant")
        }
    
    async def is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        if session_id not in self.sessions:
            return False
        
        return self.sessions[session_id].status == "active"
    
    async def pause_session(self, session_id: str):
        """Pause a session"""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        self.sessions[session_id].status = "paused"
        await self.log_interaction(session_id, {"type": "session_paused"})
    
    async def resume_session(self, session_id: str):
        """Resume a paused session"""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        self.sessions[session_id].status = "active"
        await self.log_interaction(session_id, {"type": "session_resumed"})
    
    async def end_session(self, session_id: str):
        """End a session and cleanup"""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        # Mark as ended
        self.sessions[session_id].status = "ended"
        await self.log_interaction(session_id, {"type": "session_ended"})
        
        # Clean up after a delay (to allow final operations)
        await asyncio.sleep(5)
        
        # Remove from active storage
        del self.sessions[session_id]
        if session_id in self.character_states:
            del self.character_states[session_id]
        if session_id in self.session_memories:
            del self.session_memories[session_id]
        if session_id in self.session_analytics:
            del self.session_analytics[session_id]
    
    async def get_session_memory_usage(self, session_id: str) -> Dict[str, float]:
        """Estimate memory usage for session"""
        if session_id not in self.sessions:
            return {"total_mb": 0}
        
        # Estimate sizes
        base_size = 1  # Session state ~1MB
        char_states_size = len(self.character_states.get(session_id, {})) * 0.5  # 0.5MB per character
        
        # Memory embeddings (768 floats * 4 bytes per float)
        num_memories = len(self.session_memories.get(session_id, []))
        embeddings_size = (num_memories * 768 * 4) / (1024 * 1024)  # Convert to MB
        
        # Analytics events
        num_events = len(self.session_analytics.get(session_id, []))
        analytics_size = (num_events * 0.001)  # ~1KB per event
        
        total = base_size + char_states_size + embeddings_size + analytics_size
        
        return {
            "total_mb": total,
            "breakdown": {
                "base_mb": base_size,
                "character_states_mb": char_states_size,
                "embeddings_mb": embeddings_size,
                "analytics_mb": analytics_size,
                "metadata_mb": total - (base_size + char_states_size + embeddings_size + analytics_size)
            }
        } 