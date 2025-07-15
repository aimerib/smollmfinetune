"""
Database service for platform runtime session management.

This module provides a focused service class that handles all database operations
for play sessions, character states, and conversation history.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..database.session import SessionManager

logger = logging.getLogger(__name__)


class SessionDatabaseService:
    """
    Database service for platform runtime session operations.
    
    Provides a clean interface for all session-related database operations,
    separating database concerns from business logic.
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """
        Initialize the session database service.
        
        Args:
            session_manager: SessionManager instance (creates new if not provided)
        """
        self.session_manager = session_manager or SessionManager()
        logger.info("Session database service initialized")
    
    def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete session data from database.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session data dictionary or None if not found
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            session_data = self.session_manager.get_play_session(session_id)
            if not session_data:
                logger.warning(f"Session {session_id} not found in database")
                return None
            
            logger.info(f"Loaded session data for {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load session data for {session_id}: {e}")
            raise RuntimeError(f"Database error loading session: {e}")
    
    def load_session_characters(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load all characters for a session from database.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            List of character data dictionaries
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            characters = self.session_manager.get_session_characters(session_id)
            logger.info(f"Loaded {len(characters)} characters for session {session_id}")
            return characters
            
        except Exception as e:
            logger.error(f"Failed to load session characters for {session_id}: {e}")
            raise RuntimeError(f"Database error loading characters: {e}")
    
    def load_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Load conversation history from database.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of messages to load
            
        Returns:
            List of message dictionaries in chronological order
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            conversation_data = self.session_manager.load_conversation_history(session_id, limit)
            logger.info(f"Loaded {len(conversation_data)} messages for session {session_id}")
            return conversation_data
            
        except Exception as e:
            logger.error(f"Failed to load conversation history for {session_id}: {e}")
            raise RuntimeError(f"Database error loading conversation: {e}")
    
    def save_conversation_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> bool:
        """
        Save conversation messages to database.
        
        Args:
            session_id: Unique session identifier
            messages: List of message dictionaries to save
            
        Returns:
            True if save was successful
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            if not messages:
                logger.debug(f"No messages to save for session {session_id}")
                return True
            
            self.session_manager.save_conversation_history(session_id, messages)
            logger.info(f"Saved {len(messages)} messages for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation messages for {session_id}: {e}")
            raise RuntimeError(f"Database error saving conversation: {e}")
    
    def save_character_states(self, session_id: str, character_states: List[Dict[str, Any]]) -> bool:
        """
        Save character states to database.
        
        Args:
            session_id: Unique session identifier
            character_states: List of character state dictionaries
            
        Returns:
            True if save was successful
            
        Raises:
            RuntimeError: If database operation fails
        """
        try:
            if not character_states:
                logger.debug(f"No character states to save for session {session_id}")
                return True
            
            self.session_manager.save_character_states(session_id, character_states)
            logger.info(f"Saved {len(character_states)} character states for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save character states for {session_id}: {e}")
            raise RuntimeError(f"Database error saving character states: {e}")
    
    def update_session_activity(self, session_id: str) -> bool:
        """
        Update the last activity timestamp for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if update was successful
        """
        try:
            # This would be implemented with a direct SQL update for efficiency
            # For now, this is handled by save_conversation_messages
            logger.debug(f"Updated activity timestamp for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session activity for {session_id}: {e}")
            return False
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get basic statistics about a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary with session statistics
        """
        try:
            # Load basic data
            session_data = self.session_manager.get_play_session(session_id)
            if not session_data:
                return {}
            
            characters = self.session_manager.get_session_characters(session_id)
            conversation = self.session_manager.load_conversation_history(session_id, limit=1000)
            
            # Calculate statistics
            stats = {
                'session_id': session_id,
                'character_count': len(characters),
                'active_characters': sum(1 for char in characters if char.get('is_active', True)),
                'message_count': len(conversation),
                'user_messages': sum(1 for msg in conversation if msg.get('role') == 'user'),
                'assistant_messages': sum(1 for msg in conversation if msg.get('role') == 'assistant'),
                'last_activity': session_data.get('last_activity'),
                'created_at': session_data.get('created_at')
            }
            
            logger.info(f"Generated statistics for session {session_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate session statistics for {session_id}: {e}")
            return {}
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up old, inactive sessions (placeholder for future implementation).
        
        Args:
            days_old: Remove sessions older than this many days
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            # This would be implemented with proper SQL queries
            # For now, just return 0 as placeholder
            logger.info(f"Cleanup check for sessions older than {days_old} days")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0 