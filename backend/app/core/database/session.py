"""
Database session management for the Character Creation Platform.

Provides SQLAlchemy session management, connection pooling, and transaction handling
with context managers and utility functions for database operations.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Any, Generator
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, event, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Database session manager for SQLAlchemy operations.
    
    Handles database connections, session lifecycle, and provides
    convenient context managers for database operations.
    """
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """
        Initialize the session manager.
        
        Args:
            database_url: Database connection URL (defaults to SQLite)
            echo: Whether to echo SQL queries (for debugging)
        """
        self.database_url = database_url or self._get_default_database_url()
        self.echo = echo
        
        # Create engine with appropriate configuration
        self.engine = self._create_engine()
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database session manager initialized: {self.database_url}")
    
    def _get_default_database_url(self) -> str:
        """Get the default database URL from environment or fallback to SQLite."""
        # Check for environment variable first
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        
        # Default to SQLite file in project root
        db_path = Path("platform.db")
        return f"sqlite:///{db_path.absolute()}"
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with appropriate configuration."""
        if self.database_url.startswith('sqlite'):
            # SQLite-specific configuration
            engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,  # Allow SQLite to be used across threads
                    "timeout": 20,  # 20 second timeout for database locks
                },
                pool_pre_ping=True,  # Validate connections before use
            )
            
            # Enable foreign key constraints for SQLite
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
                cursor.close()
        
        else:
            # PostgreSQL or other database configuration
            engine = create_engine(
                self.database_url,
                echo=self.echo,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
        
        return engine
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy session instance
        """
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic cleanup.
        
        Usage:
            with session_manager.session_scope() as session:
                user = session.query(User).first()
                
        Yields:
            SQLAlchemy session instance
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @contextmanager
    def transaction_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database transactions with explicit commit/rollback.
        
        Usage:
            with session_manager.transaction_scope() as session:
                user = User(username="test")
                session.add(user)
                # Automatically commits on success, rolls back on exception
                
        Yields:
            SQLAlchemy session instance
        """
        session = self.get_session()
        transaction = session.begin()
        try:
            yield session
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """Close the database engine and clean up resources."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database engine closed")
    
    # High-level session management methods for PlatformRuntimeEngine
    def get_play_session(self, session_id: str) -> Optional[dict]:
        """Get session data by ID"""
        from .models import PlaySession
        
        with self.session_scope() as session:
            play_session = session.query(PlaySession).filter(
                PlaySession.session_id == session_id
            ).first()
            
            if not play_session:
                return None
            
            return {
                'id': play_session.session_id,
                'world_id': play_session.world_id,
                'user_id': play_session.user_id,
                'session_name': play_session.session_name,
                'status': play_session.status,
                'current_location': play_session.current_location,
                'world_time': play_session.world_time,
                'weather': play_session.weather,
                'recent_world_events': play_session.recent_world_events or [],
                'created_at': play_session.created_at,
                'last_activity': play_session.last_activity
            }
    
    def get_session_characters(self, session_id: str) -> list:
        """Get characters in a session"""
        from .models import PlaySession, SessionCharacter, Character
        
        with self.session_scope() as session:
            # Get the play session first
            play_session = session.query(PlaySession).filter(
                PlaySession.session_id == session_id
            ).first()
            
            if not play_session:
                return []
            
            # Get session characters with character data
            session_characters = session.query(SessionCharacter, Character).join(
                Character, SessionCharacter.character_id == Character.id
            ).filter(
                SessionCharacter.session_id == play_session.id
            ).all()
            
            result = []
            for session_char, character in session_characters:
                character_data = {
                    'id': character.id,
                    'name': character.name,
                    'description': character.description,
                    'world_id': character.world_id,
                    'big_five': {
                        'openness': character.openness,
                        'conscientiousness': character.conscientiousness,
                        'extraversion': character.extraversion,
                        'agreeableness': character.agreeableness,
                        'neuroticism': character.neuroticism
                    },
                    'core_data': character.core_data_json or {}
                }
                
                result.append({
                    'character_id': str(character.id),
                    'character_name': character.name,
                    'character_data': character_data,
                    'is_active': session_char.is_active,
                    'current_mood': session_char.current_mood,
                    'relationships': session_char.relationships_json or {},
                    'recent_events': session_char.recent_events or [],
                    'memory_context': session_char.memory_context or [],
                    'session_character_id': session_char.id
                })
            
            return result
    
    def save_conversation_history(self, session_id: str, conversation_data: list):
        """Save conversation history to database"""
        from .models import PlaySession, ConversationMessage
        
        with self.transaction_scope() as session:
            # Get the play session
            play_session = session.query(PlaySession).filter(
                PlaySession.session_id == session_id
            ).first()
            
            if not play_session:
                logger.error(f"Session {session_id} not found for conversation history save")
                return
            
            # Get current max sequence number
            max_sequence = session.query(func.max(ConversationMessage.sequence_number)).filter(
                ConversationMessage.session_id == play_session.id
            ).scalar() or 0
            
            # Add new messages
            for i, message_data in enumerate(conversation_data):
                # Skip if message already exists
                existing = session.query(ConversationMessage).filter(
                    ConversationMessage.message_id == message_data.get('message_id')
                ).first()
                
                if existing:
                    continue
                
                message = ConversationMessage(
                    session_id=play_session.id,
                    message_id=message_data['message_id'],
                    role=message_data['role'],
                    content=message_data['content'],
                    character_id=message_data.get('character_id'),
                    character_name=message_data.get('character_name'),
                    metadata_json=message_data.get('metadata', {}),
                    sequence_number=max_sequence + i + 1,
                    created_at=message_data.get('timestamp', datetime.now())
                )
                session.add(message)
            
            # Update session last activity
            play_session.last_activity = datetime.now()
            
            logger.info(f"Saved {len(conversation_data)} messages for session {session_id}")
    
    def save_character_states(self, session_id: str, character_states_data: list):
        """Save character states to database"""
        from .models import PlaySession, SessionCharacter
        
        with self.transaction_scope() as session:
            # Get the play session
            play_session = session.query(PlaySession).filter(
                PlaySession.session_id == session_id
            ).first()
            
            if not play_session:
                logger.error(f"Session {session_id} not found for character states save")
                return
            
            # Update character states
            for char_state_data in character_states_data:
                character_id = char_state_data['character_id']
                
                # Find the session character
                session_char = session.query(SessionCharacter).filter(
                    SessionCharacter.session_id == play_session.id,
                    SessionCharacter.character_id == int(character_id)
                ).first()
                
                if session_char:
                    # Update existing character state
                    session_char.is_active = char_state_data.get('is_active', True)
                    session_char.current_mood = char_state_data.get('current_mood', 'neutral')
                    session_char.relationships_json = char_state_data.get('relationships', {})
                    session_char.recent_events = char_state_data.get('recent_events', [])
                    session_char.memory_context = char_state_data.get('memory_context', [])
                    session_char.last_response_at = datetime.now()
                
            logger.info(f"Updated character states for session {session_id}: {len(character_states_data)} characters")
    
    def load_conversation_history(self, session_id: str, limit: int = 50) -> list:
        """Load conversation history from database"""
        from .models import PlaySession, ConversationMessage
        
        with self.session_scope() as session:
            # Get the play session
            play_session = session.query(PlaySession).filter(
                PlaySession.session_id == session_id
            ).first()
            
            if not play_session:
                return []
            
            # Get conversation messages
            messages = session.query(ConversationMessage).filter(
                ConversationMessage.session_id == play_session.id
            ).order_by(ConversationMessage.sequence_number.desc()).limit(limit).all()
            
            # Convert to list of dicts (reverse to get chronological order)
            result = []
            for message in reversed(messages):
                result.append({
                    'message_id': message.message_id,
                    'role': message.role,
                    'content': message.content,
                    'character_id': str(message.character_id) if message.character_id else None,
                    'character_name': message.character_name,
                    'timestamp': message.created_at,
                    'metadata': message.metadata_json or {}
                })
            
            return result


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(database_url: Optional[str] = None, echo: bool = False) -> SessionManager:
    """
    Get or create the global session manager instance.
    
    Args:
        database_url: Database connection URL (only used on first call)
        echo: Whether to echo SQL queries (only used on first call)
        
    Returns:
        SessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        _session_manager = SessionManager(database_url=database_url, echo=echo)
    
    return _session_manager


def get_session() -> Session:
    """
    Get a new database session from the global session manager.
    
    Returns:
        SQLAlchemy session instance
    """
    return get_session_manager().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions using the global session manager.
    
    Usage:
        from backend.app.core.database import session_scope
        
        with session_scope() as session:
            user = session.query(User).first()
            
    Yields:
        SQLAlchemy session instance
    """
    with get_session_manager().session_scope() as session:
        yield session


@contextmanager
def transaction_scope() -> Generator[Session, None, None]:
    """
    Context manager for database transactions using the global session manager.
    
    Usage:
        from backend.app.core.database import transaction_scope
        
        with transaction_scope() as session:
            user = User(username="test")
            session.add(user)
            
    Yields:
        SQLAlchemy session instance
    """
    with get_session_manager().transaction_scope() as session:
        yield session


def init_database(database_url: Optional[str] = None, echo: bool = False, create_tables: bool = True) -> SessionManager:
    """
    Initialize the database with tables and return the session manager.
    
    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL queries
        create_tables: Whether to create tables automatically
        
    Returns:
        Initialized SessionManager instance
    """
    session_manager = get_session_manager(database_url=database_url, echo=echo)
    
    if create_tables:
        session_manager.create_tables()
    
    return session_manager 