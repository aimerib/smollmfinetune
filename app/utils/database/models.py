"""
SQLAlchemy models for the Character Creation Platform database.

Defines all database tables and relationships for users, worlds, characters,
training runs, preferences, and related entities.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, DateTime, JSON,
    ForeignKey, UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.orm import declarative_base, relationship, backref
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    """User model - extends existing auth system"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default='player')  # admin, creator, player
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    profile_json = Column(JSON, nullable=True)  # User profile data including data collection consent
    
    # Relationships
    worlds = relationship("World", back_populates="owner", cascade="all, delete-orphan")
    characters = relationship("Character", back_populates="owner", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRun", back_populates="owner", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class World(Base):
    """World model - migrated from world_lore.json files"""
    __tablename__ = 'worlds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Metadata
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # JSON data for complex structures
    meta_data = Column(JSON, nullable=True)  # Additional metadata
    control_tokens = Column(JSON, nullable=True)  # Control tokens from tokens.json
    
    # Relationships
    owner = relationship("User", back_populates="worlds")
    characters = relationship("Character", back_populates="world", cascade="all, delete-orphan")
    facts = relationship("WorldFact", back_populates="world", cascade="all, delete-orphan")
    timeline_events = relationship("WorldTimeline", back_populates="world", cascade="all, delete-orphan")
    factions = relationship("WorldFaction", back_populates="world", cascade="all, delete-orphan")
    places = relationship("WorldPlace", back_populates="world", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'owner_id', name='uq_world_name_owner'),
        Index('idx_world_owner_name', 'owner_id', 'name'),
    )
    
    def __repr__(self):
        return f"<World(id={self.id}, name='{self.name}', owner_id={self.owner_id})>"


class WorldFact(Base):
    """World facts - migrated from world_lore.json facts section"""
    __tablename__ = 'world_facts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    world_id = Column(Integer, ForeignKey('worlds.id'), nullable=False, index=True)
    key = Column(String(200), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    world = relationship("World", back_populates="facts")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('world_id', 'key', name='uq_world_fact_key'),
    )
    
    def __repr__(self):
        return f"<WorldFact(id={self.id}, world_id={self.world_id}, key='{self.key}')>"


class WorldTimeline(Base):
    """World timeline events - migrated from world_lore.json timeline section"""
    __tablename__ = 'world_timeline'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    world_id = Column(Integer, ForeignKey('worlds.id'), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    event = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    world = relationship("World", back_populates="timeline_events")
    
    # Constraints
    __table_args__ = (
        Index('idx_world_timeline_year', 'world_id', 'year'),
    )
    
    def __repr__(self):
        return f"<WorldTimeline(id={self.id}, world_id={self.world_id}, year={self.year})>"


class WorldFaction(Base):
    """World factions - migrated from world_lore.json factions section"""
    __tablename__ = 'world_factions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    world_id = Column(Integer, ForeignKey('worlds.id'), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    timeline_events = Column(JSON, nullable=True)  # Array of {year, event} objects
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    world = relationship("World", back_populates="factions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('world_id', 'name', name='uq_world_faction_name'),
    )
    
    def __repr__(self):
        return f"<WorldFaction(id={self.id}, world_id={self.world_id}, name='{self.name}')>"


class WorldPlace(Base):
    """World places - migrated from world_lore.json places section"""
    __tablename__ = 'world_places'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    world_id = Column(Integer, ForeignKey('worlds.id'), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    world = relationship("World", back_populates="places")
    npcs = relationship("WorldNPC", back_populates="place", cascade="all, delete-orphan")
    events = relationship("WorldEvent", back_populates="place", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('world_id', 'name', name='uq_world_place_name'),
    )
    
    def __repr__(self):
        return f"<WorldPlace(id={self.id}, world_id={self.world_id}, name='{self.name}')>"


class WorldNPC(Base):
    """World NPCs - migrated from world_lore.json places.npcs section"""
    __tablename__ = 'world_npcs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    place_id = Column(Integer, ForeignKey('world_places.id'), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    place = relationship("WorldPlace", back_populates="npcs")
    
    def __repr__(self):
        return f"<WorldNPC(id={self.id}, place_id={self.place_id}, name='{self.name}')>"


class WorldEvent(Base):
    """World events - migrated from world_lore.json places.events section"""
    __tablename__ = 'world_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    place_id = Column(Integer, ForeignKey('world_places.id'), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    is_random = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    place = relationship("WorldPlace", back_populates="events")
    
    def __repr__(self):
        return f"<WorldEvent(id={self.id}, place_id={self.place_id}, name='{self.name}')>"


class Character(Base):
    """Character model - migrated from character_core.json files"""
    __tablename__ = 'characters'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    world_id = Column(Integer, ForeignKey('worlds.id'), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Core character data
    description = Column(Text, nullable=True)
    scenario = Column(Text, nullable=True)
    backstory = Column(Text, nullable=True)
    appearance = Column(Text, nullable=True)
    first_message = Column(Text, nullable=True)  # Character's first message/greeting
    
    # Personality traits (Big Five)
    openness = Column(Float, default=0.5, nullable=False)
    conscientiousness = Column(Float, default=0.5, nullable=False)
    extraversion = Column(Float, default=0.5, nullable=False)
    agreeableness = Column(Float, default=0.5, nullable=False)
    neuroticism = Column(Float, default=0.5, nullable=False)
    personality_json = Column(JSON, nullable=True)  # Additional personality data and traits
    
    # Metadata and imports
    core_data_json = Column(JSON, nullable=True)  # Full CharacterCore data
    imports_data = Column(JSON, nullable=True)  # Import metadata
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    world = relationship("World", back_populates="characters")
    owner = relationship("User", back_populates="characters")
    training_runs = relationship("TrainingRun", back_populates="character", cascade="all, delete-orphan")
    preference_pairs = relationship("PreferencePair", back_populates="character", cascade="all, delete-orphan")
    relationships = relationship("CharacterRelationship", back_populates="character", cascade="all, delete-orphan")
    goals = relationship("CharacterGoal", back_populates="character", cascade="all, delete-orphan")
    tags = relationship("CharacterTag", back_populates="character", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'world_id', name='uq_character_name_world'),
        Index('idx_character_world_owner', 'world_id', 'owner_id'),
        CheckConstraint('openness >= 0 AND openness <= 1', name='ck_openness_range'),
        CheckConstraint('conscientiousness >= 0 AND conscientiousness <= 1', name='ck_conscientiousness_range'),
        CheckConstraint('extraversion >= 0 AND extraversion <= 1', name='ck_extraversion_range'),
        CheckConstraint('agreeableness >= 0 AND agreeableness <= 1', name='ck_agreeableness_range'),
        CheckConstraint('neuroticism >= 0 AND neuroticism <= 1', name='ck_neuroticism_range'),
    )
    
    def __repr__(self):
        return f"<Character(id={self.id}, name='{self.name}', world_id={self.world_id})>"


class CharacterRelationship(Base):
    """Character relationships - migrated from CharacterCore relationships"""
    __tablename__ = 'character_relationships'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    affinity = Column(Integer, default=0, nullable=False)  # -10 to +10 range
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    character = relationship("Character", back_populates="relationships")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('affinity >= -10 AND affinity <= 10', name='ck_affinity_range'),
        UniqueConstraint('character_id', 'name', name='uq_character_relationship_name'),
    )
    
    def __repr__(self):
        return f"<CharacterRelationship(id={self.id}, character_id={self.character_id}, name='{self.name}', affinity={self.affinity})>"


class CharacterGoal(Base):
    """Character goals - migrated from CharacterCore goals"""
    __tablename__ = 'character_goals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    goal = Column(Text, nullable=False)
    priority = Column(Integer, default=1, nullable=False)  # 1 = highest priority
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    character = relationship("Character", back_populates="goals")
    
    def __repr__(self):
        return f"<CharacterGoal(id={self.id}, character_id={self.character_id}, priority={self.priority})>"


class CharacterTag(Base):
    """Character tags - migrated from CharacterCore tags"""
    __tablename__ = 'character_tags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    tag = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    character = relationship("Character", back_populates="tags")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('character_id', 'tag', name='uq_character_tag'),
        Index('idx_character_tag_value', 'tag'),
    )
    
    def __repr__(self):
        return f"<CharacterTag(id={self.id}, character_id={self.character_id}, tag='{self.tag}')>"


class TrainingRun(Base):
    """Training run model - migrated from training_output metadata"""
    __tablename__ = 'training_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Training configuration
    base_model = Column(String(200), nullable=False)
    training_method = Column(String(50), nullable=False)  # lora, rslora, dora
    status = Column(String(50), nullable=False, default='pending')  # pending, running, completed, failed
    
    # Paths and artifacts
    sft_adapter_path = Column(String(500), nullable=True)
    rlhf_adapter_path = Column(String(500), nullable=True)
    output_directory = Column(String(500), nullable=False)
    
    # Training metrics and metadata
    metrics_json = Column(JSON, nullable=True)  # Training metrics, loss curves, etc.
    config_json = Column(JSON, nullable=True)   # Training configuration
    
    # Training parameters
    total_steps = Column(Integer, nullable=True)
    dataset_size = Column(Integer, nullable=True)
    final_loss = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    character = relationship("Character", back_populates="training_runs")
    owner = relationship("User", back_populates="training_runs")
    
    # Constraints
    __table_args__ = (
        Index('idx_training_run_status', 'status'),
        Index('idx_training_run_character_status', 'character_id', 'status'),
    )
    
    def __repr__(self):
        return f"<TrainingRun(id={self.id}, character_id={self.character_id}, status='{self.status}')>"


class PreferencePair(Base):
    """Preference pairs - migrated from preference_logs.ndjson files"""
    __tablename__ = 'preference_pairs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    
    # Preference data
    prompt_text = Column(Text, nullable=False)
    chosen_text = Column(Text, nullable=False)
    rejected_json = Column(JSON, nullable=False)  # Array of rejected options
    
    # Context and metadata
    context_json = Column(JSON, nullable=True)  # Additional context data
    source = Column(String(100), nullable=True)  # Where this preference was collected
    quality_score = Column(Float, nullable=True)  # Optional quality rating
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    character = relationship("Character", back_populates="preference_pairs")
    
    # Constraints
    __table_args__ = (
        Index('idx_preference_character_created', 'character_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<PreferencePair(id={self.id}, character_id={self.character_id})>"


# Additional utility tables for system management

class DatabaseVersion(Base):
    """Track database schema version for migrations"""
    __tablename__ = 'database_version'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True)
    applied_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    description = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<DatabaseVersion(version='{self.version}', applied_at={self.applied_at})>"


class PlaySession(Base):
    """Play sessions for platform runtime"""
    __tablename__ = 'play_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    world_id = Column(Integer, ForeignKey('worlds.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Session metadata
    session_name = Column(String(200), nullable=False)
    status = Column(String(50), nullable=False, default='setup')  # setup, active, paused, completed
    privacy_setting = Column(String(20), nullable=False, default='private')  # private, friends, public
    
    # Session state
    current_location = Column(String(200), nullable=True)
    world_time = Column(String(100), nullable=True)
    weather = Column(String(100), nullable=True)
    recent_world_events = Column(JSON, nullable=True)  # Array of recent events
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    world = relationship("World")
    user = relationship("User")
    session_characters = relationship("SessionCharacter", back_populates="session", cascade="all, delete-orphan")
    conversation_messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('idx_session_user_status', 'user_id', 'status'),
        Index('idx_session_world_active', 'world_id', 'status'),
    )
    
    def __repr__(self):
        return f"<PlaySession(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"


class SessionCharacter(Base):
    """Characters active in a play session"""
    __tablename__ = 'session_characters'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('play_sessions.id'), nullable=False, index=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    
    # Character state in session
    is_active = Column(Boolean, default=True, nullable=False)
    current_mood = Column(String(50), nullable=False, default='neutral')
    relationships_json = Column(JSON, nullable=True)  # Character relationships with user/other chars
    recent_events = Column(JSON, nullable=True)  # Array of recent events affecting this character
    memory_context = Column(JSON, nullable=True)  # Character's memory context
    
    # Settings
    personality_overrides = Column(JSON, nullable=True)  # Temporary personality adjustments
    generation_settings = Column(JSON, nullable=True)  # Per-character generation settings
    
    # Timestamps
    added_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_response_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    session = relationship("PlaySession", back_populates="session_characters")
    character = relationship("Character")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('session_id', 'character_id', name='uq_session_character'),
        Index('idx_session_character_active', 'session_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<SessionCharacter(id={self.id}, session_id={self.session_id}, character_id={self.character_id})>"


class ConversationMessage(Base):
    """Messages in play session conversations"""
    __tablename__ = 'conversation_messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('play_sessions.id'), nullable=False, index=True)
    message_id = Column(String(100), nullable=False, unique=True, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=True, index=True)  # NULL for user messages
    character_name = Column(String(200), nullable=True)  # Denormalized for performance
    
    # Message metadata
    metadata_json = Column(JSON, nullable=True)  # Additional message metadata
    generation_settings = Column(JSON, nullable=True)  # Settings used for this message
    response_time_ms = Column(Integer, nullable=True)  # Time taken to generate response
    
    # Message ordering and threading
    sequence_number = Column(Integer, nullable=False)  # Order within session
    parent_message_id = Column(String(100), nullable=True)  # For threaded conversations
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session = relationship("PlaySession", back_populates="conversation_messages")
    character = relationship("Character")
    
    # Constraints
    __table_args__ = (
        Index('idx_conversation_session_sequence', 'session_id', 'sequence_number'),
        Index('idx_conversation_session_created', 'session_id', 'created_at'),
        UniqueConstraint('session_id', 'sequence_number', name='uq_session_sequence'),
    )
    
    def __repr__(self):
        return f"<ConversationMessage(id={self.id}, session_id={self.session_id}, role='{self.role}')>"


class ConversationLog(Base):
    """Conversation logs for data collection and training (R3-2.5)"""
    __tablename__ = 'conversation_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    character_id = Column(Integer, ForeignKey('characters.id'), nullable=False, index=True)
    
    # Conversation data
    conversation_data = Column(JSON, nullable=False)  # Full conversation data (PII scrubbed)
    quality_score = Column(Float, nullable=False, default=3.0)  # 0-5 quality rating
    
    # Processing flags
    processed_for_training = Column(Boolean, default=False, nullable=False)
    nsfw_flagged = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    conversation_length = Column(Integer, nullable=True)  # Number of messages
    user_satisfaction = Column(Integer, nullable=True)  # Optional user rating 1-5
    source = Column(String(50), nullable=False, default='platform')  # platform, api, import
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User")
    character = relationship("Character")
    
    # Constraints
    __table_args__ = (
        Index('idx_conversation_log_quality', 'quality_score'),
        Index('idx_conversation_log_processed', 'processed_for_training'),
        Index('idx_conversation_log_character_date', 'character_id', 'created_at'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 5', name='ck_quality_score_range'),
    )
    
    def __repr__(self):
        return f"<ConversationLog(id={self.id}, user_id={self.user_id}, character_id={self.character_id}, quality={self.quality_score})>"


class SystemConfig(Base):
    """System configuration key-value store"""
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(200), nullable=False, unique=True, index=True)
    value = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<SystemConfig(key='{self.key}')>" 