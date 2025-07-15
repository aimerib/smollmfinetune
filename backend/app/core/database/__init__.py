"""
Database package for the Character Creation Platform

Provides SQLAlchemy models, session management, and migration utilities
for transitioning from file-based to database-driven storage.
"""

from .models import *
from .session import SessionManager, get_session, session_scope, transaction_scope, init_database
from .migration import DatabaseMigrator
from .world_manager import DatabaseWorldManager

__all__ = [
    # Models
    'User', 'World', 'Character', 'TrainingRun', 'PreferencePair',
    'WorldFact', 'WorldTimeline', 'WorldFaction', 'WorldPlace', 'WorldNPC', 'WorldEvent',
    'CharacterRelationship', 'CharacterGoal', 'CharacterTag',
    
    # Session management
    'SessionManager', 'get_session', 'session_scope', 'transaction_scope', 'init_database',
    
    # Migration utilities
    'DatabaseMigrator',
    
    # Database managers
    'DatabaseWorldManager'
] 