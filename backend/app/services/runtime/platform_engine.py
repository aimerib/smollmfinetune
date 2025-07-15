"""
Platform Runtime Engine for multi-character session management

This module provides the core runtime engine that manages active play sessions
with multiple characters, handling concurrent response generation, character
state tracking, and session persistence.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .prompt_constructor import RuntimePromptConstructor
from .fallback_prompt_constructor import FallbackPromptConstructor
from .session_database_service import SessionDatabaseService
from ..database.session import SessionManager
from ..inference import InferenceManager

logger = logging.getLogger(__name__)


@dataclass
class CharacterState:
    """Tracks the dynamic state of a character during gameplay"""
    
    character_id: str
    character_name: str
    character_data: Dict[str, Any]
    current_mood: str = 'neutral'
    relationships: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recent_events: List[str] = field(default_factory=list)
    is_active: bool = True
    memory_context: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize character state to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterState':
        """Deserialize character state from dictionary"""
        return cls(**data)


@dataclass
class SessionMessage:
    """Represents a message in the conversation history"""
    
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate message ID if not provided"""
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary"""
        data = asdict(self)
        # Convert datetime to ISO string for JSON serialization 
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMessage':
        """Deserialize message from dictionary"""
        # Convert ISO string back to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class PlatformRuntimeEngine:
    """
    Core runtime engine for managing multi-character play sessions
    
    Handles character state management, concurrent response generation,
    conversation history, and session persistence.
    """
    
    def __init__(self, session_id: str, session_db_service: Optional[SessionDatabaseService] = None):
        """
        Initialize the platform runtime engine
        
        Args:
            session_id: Unique identifier for the play session
            session_db_service: Database service for session operations (creates new if not provided)
        """
        self.session_id = session_id
        self.db_service = session_db_service or SessionDatabaseService()
        self.inference_manager = InferenceManager()
        
        # Load session data from database
        self.session_data = self.db_service.load_session_data(session_id)
        if not self.session_data:
            raise ValueError(f"Session {session_id} not found")
        
        self.world_id = self.session_data['world_id']
        self.user_id = self.session_data['user_id']
        
        # Load and initialize character states
        self.character_states: Dict[str, CharacterState] = {}
        self.prompt_constructors: Dict[str, RuntimePromptConstructor] = {}
        self._initialize_characters()
        
        # Conversation history
        self.conversation_history: List[SessionMessage] = []
        self._load_conversation_history()
        
        logger.info(f"Platform runtime engine initialized for session {session_id}")
    
    def _initialize_characters(self):
        """Load and initialize character states and prompt constructors"""
        session_characters = self.db_service.load_session_characters(self.session_id)
        
        for char_data in session_characters:
            character_id = char_data['character_id']
            character_name = char_data['character_name']
            character_info = char_data['character_data']
            
            # Initialize character state
            self.character_states[character_id] = CharacterState(
                character_id=character_id,
                character_name=character_name,
                character_data=character_info,
                is_active=char_data.get('is_active', True)
            )
            
            # Initialize prompt constructor 
            packet_path = f"runtime_packets/{character_name}"
            if Path(packet_path).exists():
                self.prompt_constructors[character_id] = RuntimePromptConstructor(packet_path)
            else:
                # Fallback: create a basic prompt constructor with character data
                logger.warning(f"Runtime packet not found for {character_name}, using fallback")
                self.prompt_constructors[character_id] = self._create_fallback_prompt_constructor(
                    character_id, character_name, character_info
                )
    
    def _load_conversation_history(self):
        """Load existing conversation history from database"""
        conversation_data = self.db_service.load_conversation_history(self.session_id)
        
        # Convert loaded data to SessionMessage objects
        self.conversation_history = []
        for msg_data in conversation_data:
            # Convert timestamp if it's a string
            timestamp = msg_data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            message = SessionMessage(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=timestamp,
                character_id=msg_data.get('character_id'),
                character_name=msg_data.get('character_name'),
                message_id=msg_data.get('message_id'),
                metadata=msg_data.get('metadata', {})
            )
            self.conversation_history.append(message)
    
    def get_character_state(self, character_id: str) -> CharacterState:
        """Get the current state of a character"""
        if character_id not in self.character_states:
            raise ValueError(f"Character {character_id} not found in session")
        return self.character_states[character_id]
    
    def update_character_mood(self, character_id: str, mood: str):
        """Update a character's current mood"""
        if character_id in self.character_states:
            self.character_states[character_id].current_mood = mood
    
    def update_character_relationship(self, character_id: str, target_id: str, relationship_data: Dict[str, float]):
        """Update relationship between characters or with user"""
        if character_id in self.character_states:
            self.character_states[character_id].relationships[target_id] = relationship_data
    
    def is_character_active(self, character_id: str) -> bool:
        """Check if a character is currently active in the session"""
        if character_id in self.character_states:
            return self.character_states[character_id].is_active
        return False
    
    def set_character_active(self, character_id: str, active: bool):
        """Set character active/inactive status"""
        if character_id in self.character_states:
            self.character_states[character_id].is_active = active
    
    def get_active_character_ids(self) -> List[str]:
        """Get list of currently active character IDs"""
        return [
            char_id for char_id, state in self.character_states.items()
            if state.is_active
        ]
    
    def add_message_to_history(self, message_data: Dict[str, Any]):
        """Add a message to the conversation history"""
        if 'timestamp' not in message_data:
            message_data['timestamp'] = datetime.now()
        
        message = SessionMessage(**message_data)
        self.conversation_history.append(message)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history as dictionaries"""
        return [msg.to_dict() for msg in self.conversation_history]
    
    def get_recent_conversation_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for prompt construction"""
        recent_messages = self.conversation_history[-max_messages:] if self.conversation_history else []
        return [msg.to_dict() for msg in recent_messages]
    
    async def generate_character_response(self, character_id: str, user_message: str, 
                                        conversation_history: List[Dict] = None) -> str:
        """Generate a response from a single character"""
        if character_id not in self.character_states:
            raise ValueError(f"Character {character_id} not found")
        
        if character_id not in self.prompt_constructors:
            raise ValueError(f"Prompt constructor not available for character {character_id}")
        
        # Use provided history or get recent context
        if conversation_history is None:
            conversation_history = self.get_recent_conversation_context()
        
        # Add the current user message to conversation history for prompt construction
        full_conversation_history = conversation_history.copy()
        full_conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Get character state for dynamic context
        character_state = self.character_states[character_id]
        dynamic_state = {
            'current_mood': character_state.current_mood,
            'relationship_to_user': character_state.relationships.get(self.user_id, {}),
            'recent_events': character_state.recent_events[-3:],  # Last 3 events
        }
        
        # Construct prompt
        prompt_constructor = self.prompt_constructors[character_id]
        prompt = prompt_constructor.construct(full_conversation_history, dynamic_state)
        
        # Generate response using inference manager
        model_path = f"character-{character_id}"  # Placeholder model path
        response = await self.inference_manager.generate_response(
            model_path=model_path,
            prompt=prompt,
            max_tokens=200,
            temperature=0.8
        )
        
        return response
    
    async def generate_multi_character_responses(self, user_message: str, 
                                               conversation_history: List[Dict] = None,
                                               responding_characters: List[str] = None) -> Dict[str, str]:
        """Generate concurrent responses from multiple characters"""
        if responding_characters is None:
            responding_characters = self.get_active_character_ids()
        
        # Create concurrent tasks for each character
        tasks = []
        for character_id in responding_characters:
            if character_id in self.character_states and self.is_character_active(character_id):
                task = self.generate_character_response(character_id, user_message, conversation_history)
                tasks.append((character_id, task))
        
        # Execute all tasks concurrently
        responses = {}
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (character_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating response for {character_id}: {result}")
                    responses[character_id] = f"[Error: {character_id} could not respond]"
                else:
                    responses[character_id] = result
        
        return responses
    
    async def generate_character_interaction_response(self, responding_character_id: str,
                                                    triggering_character_id: str,
                                                    triggering_message: str,
                                                    conversation_history: List[Dict] = None) -> str:
        """Generate a character response to another character's message"""
        # Add the triggering message to conversation context
        if conversation_history is None:
            conversation_history = self.get_recent_conversation_context()
        
        # Add the triggering message as context
        interaction_history = conversation_history + [{
            'role': 'assistant',
            'content': triggering_message,
            'character_id': triggering_character_id
        }]
        
        # Generate response with interaction context
        return await self.generate_character_response(
            responding_character_id, 
            triggering_message,  # The "user message" is the character's statement
            interaction_history
        )
    
    def determine_responding_characters(self, message: str, message_type: str = "user",
                                     speaking_character_id: str = None,
                                     conversation_history: List[Dict] = None) -> List[str]:
        """Determine which characters should respond to a message"""
        active_characters = self.get_active_character_ids()
        
        if message_type == "user":
            # For user messages, typically 1-2 characters respond
            # Simple logic: return up to 2 active characters
            return active_characters[:2]
        
        elif message_type == "character":
            # For character messages, other characters might respond
            # Don't include the speaking character in the response list
            potential_responders = [
                char_id for char_id in active_characters 
                if char_id != speaking_character_id
            ]
            # Simple logic: 50% chance each character responds
            import random
            return [char_id for char_id in potential_responders if random.random() > 0.5]
        
        return []
    
    def process_world_event(self, world_event: Dict[str, Any]):
        """Process a world event that affects character states"""
        event_description = world_event.get('description', 'Unknown event occurred')
        effects = world_event.get('effects', {})
        
        # Apply effects to all characters if specified
        if effects.get('affects_characters') == 'all':
            for character_state in self.character_states.values():
                # Add to recent events
                character_state.recent_events.append(event_description)
                
                # Limit recent events to last 5
                if len(character_state.recent_events) > 5:
                    character_state.recent_events = character_state.recent_events[-5:]
                
                # Apply mood modifiers if specified
                mood_modifier = effects.get('mood_modifier', 0)
                if mood_modifier != 0:
                    # Simple mood modification logic (placeholder)
                    current_mood = character_state.current_mood
                    # In real implementation, would have more sophisticated mood system
    
    def save_session_state(self):
        """Save current session state to database"""
        # Save conversation history
        conversation_data = self.get_conversation_history()
        self.db_service.save_conversation_messages(self.session_id, conversation_data)
        
        # Save character states
        character_states_data = [
            {
                'character_id': char_id,
                **state.to_dict()
            }
            for char_id, state in self.character_states.items()
        ]
        self.db_service.save_character_states(self.session_id, character_states_data)
    
    def _create_fallback_prompt_constructor(self, character_id: str, character_name: str, character_data: Dict[str, Any]):
        """Create a fallback prompt constructor when runtime packet is not available"""
        return FallbackPromptConstructor(character_id, character_name, character_data) 