from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    
class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="active")  # active, archived
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    worlds = relationship("World", back_populates="project", cascade="all, delete-orphan")
    
class World(Base):
    __tablename__ = "worlds"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    
    # World data
    setting = Column(Text)  # Time period, location, etc.
    rules = Column(JSON)    # Physical laws, magic systems, etc.
    history = Column(Text)  # Major events, timeline
    cultures = Column(JSON) # Different societies, customs
    locations = Column(JSON) # Important places
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="worlds")
    characters = relationship("Character", back_populates="world", cascade="all, delete-orphan")
    
class Character(Base):
    __tablename__ = "characters"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    world_id = Column(String, ForeignKey("worlds.id"), nullable=False)
    
    # Personality (Big Five)
    openness = Column(Float, default=0.5)
    conscientiousness = Column(Float, default=0.5)
    extraversion = Column(Float, default=0.5)
    agreeableness = Column(Float, default=0.5)
    neuroticism = Column(Float, default=0.5)
    
    # Character details
    backstory = Column(Text)
    goals = Column(Text)
    relationships = Column(Text)
    traits = Column(JSON)  # Additional personality traits
    voice_style = Column(Text)  # How they speak
    
    # Training status
    is_trained = Column(Boolean, default=False)
    adapter_path = Column(String)  # Path to LoRA/DoRA adapter
    model_version = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    world = relationship("World", back_populates="characters")
    datasets = relationship("Dataset", back_populates="character", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="character", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="character", cascade="all, delete-orphan")
    
class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Dataset info
    conversation_count = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    file_path = Column(String)  # Path to dataset file
    format = Column(String, default="jsonl")  # jsonl, parquet, etc.
    
    # Generation parameters
    generation_params = Column(JSON)
    
    # Status
    status = Column(String, default="pending")  # pending, generating, completed, failed
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    character = relationship("Character", back_populates="datasets")
    
class MultimodalDataset(Base):
    __tablename__ = "multimodal_datasets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    config = Column(JSON)  # Generation configuration
    status = Column(String, default="pending")  # pending, generating, completed, failed, cancelled
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    current_step = Column(String)
    samples_generated = Column(Integer, default=0)
    total_samples = Column(Integer)
    output_path = Column(String)
    error_message = Column(Text)
    
    # Celery task management
    celery_task_id = Column(String)
    
    # TTS and multimodal specific
    tts_provider = Column(String)  # orpheus, kokoro, xtts, bark
    character_count = Column(Integer)
    narrative_types = Column(JSON)  # List of narrative types
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    
    # Job info
    job_type = Column(String, nullable=False)  # sft, dpo, grpo
    status = Column(String, default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)  # 0-100
    
    # Training parameters
    training_params = Column(JSON)
    
    # Results
    adapter_path = Column(String)
    training_loss = Column(Float)
    validation_loss = Column(Float)
    metrics = Column(JSON)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Error handling
    error_message = Column(Text)
    
    # Celery task ID
    celery_task_id = Column(String)
    
    # Relationships
    character = relationship("Character", back_populates="training_jobs")
    
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    character_id = Column(String, ForeignKey("characters.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Session info
    title = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Messages stored as JSON array
    messages = Column(JSON, default=list)
    message_count = Column(Integer, default=0)
    
    # Metadata
    metadata = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    character = relationship("Character", back_populates="chat_sessions")
    user = relationship("User") 