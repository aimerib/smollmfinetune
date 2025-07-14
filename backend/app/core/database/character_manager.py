"""
Database-powered Character Manager for the Character Creation Platform.

Replaces the file-based CharacterManager with SQLAlchemy database operations
while maintaining the same interface for backward compatibility.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import (
    User, World, Character, CharacterRelationship, 
    CharacterGoal, CharacterTag
)
from .session import session_scope, transaction_scope
from ..character.models import CharacterCore, Personality, Relationship

logger = logging.getLogger(__name__)


class DatabaseCharacterManager:
    """
    Database-powered character manager using SQLAlchemy models.
    
    Provides the same interface as the file-based CharacterManager but
    stores all data in the database for better consistency and querying.
    """
    
    def __init__(self, user_id: int = 1):
        """
        Initialize the database character manager.
        
        Args:
            user_id: ID of the user who owns the characters (default: 1)
        """
        self.user_id = user_id
        self.current_character_id: Optional[int] = None
        self.current_character_core: Optional[CharacterCore] = None
        self.current_world_id: Optional[int] = None
        
        logger.info(f"Database character manager initialized for user {user_id}")
