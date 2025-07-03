from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# Base schemas
class TimestampMixin(BaseModel):
    created_at: datetime
    updated_at: datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)

class UserResponse(UserBase, TimestampMixin):
    id: str
    
    class Config:
        from_attributes = True

# Auth schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None

class LoginRequest(BaseModel):
    username: str  # Can be username or email
    password: str

# Project schemas
class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    status: str = "active"

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(active|archived)$")

class ProjectResponse(ProjectBase):
    id: str
    owner_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# World schemas
class WorldBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    setting: Optional[str] = None
    rules: Optional[Dict[str, Any]] = None
    history: Optional[str] = None
    cultures: Optional[Dict[str, Any]] = None
    locations: Optional[Dict[str, Any]] = None

class WorldCreate(WorldBase):
    project_id: str

class WorldUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    setting: Optional[str] = None
    rules: Optional[Dict[str, Any]] = None
    history: Optional[str] = None
    cultures: Optional[Dict[str, Any]] = None
    locations: Optional[Dict[str, Any]] = None

class WorldResponse(WorldBase):
    id: str
    project_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class WorldListResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    character_count: int
    created_at: datetime
    updated_at: datetime

# Character schemas
class PersonalityTraits(BaseModel):
    openness: float = Field(0.5, ge=0, le=1)
    conscientiousness: float = Field(0.5, ge=0, le=1)
    extraversion: float = Field(0.5, ge=0, le=1)
    agreeableness: float = Field(0.5, ge=0, le=1)
    neuroticism: float = Field(0.5, ge=0, le=1)

class CharacterBase(BaseModel):
    name: str
    description: Optional[str] = None
    backstory: Optional[str] = None
    goals: Optional[str] = None
    relationships: Optional[str] = None
    traits: Optional[Dict[str, Any]] = None
    voice_style: Optional[str] = None

class CharacterCreate(CharacterBase):
    world_id: str
    personality: PersonalityTraits = PersonalityTraits()

class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    personality: Optional[PersonalityTraits] = None
    backstory: Optional[str] = None
    goals: Optional[str] = None
    relationships: Optional[str] = None
    traits: Optional[Dict[str, Any]] = None
    voice_style: Optional[str] = None

class CharacterResponse(CharacterBase, TimestampMixin):
    id: str
    world_id: str
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    is_trained: bool
    adapter_path: Optional[str] = None
    model_version: Optional[str] = None
    
    class Config:
        from_attributes = True

# Dataset schemas
class DatasetGenerationParams(BaseModel):
    target_samples: int = Field(100, gt=0, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    batch_size: int = Field(10, gt=0, le=50)
    topics: Optional[List[str]] = None
    quality_mode: str = Field("iterative", pattern="^(fast|balanced|iterative)$")

class DatasetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None

class DatasetCreate(DatasetBase):
    character_id: str
    generation_params: Optional[DatasetGenerationParams] = None

class DatasetResponse(DatasetBase):
    id: str
    character_id: str
    conversation_count: int
    total_messages: int
    file_path: Optional[str]
    format: str
    generation_params: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class DatasetListResponse(BaseModel):
    id: str
    name: str
    conversation_count: int
    status: str
    created_at: datetime

# Generation progress schema
class GenerationProgress(BaseModel):
    current: int
    total: int
    percentage: float
    status: str
    current_topic: Optional[str]

# Training job schemas
class TrainingJobBase(BaseModel):
    job_type: str = Field(..., pattern="^(sft|dpo|grpo)$")
    training_params: Dict[str, Any]

class TrainingJobCreate(TrainingJobBase):
    character_id: str
    dataset_id: Optional[str] = None

class TrainingJobResponse(TrainingJobBase, TimestampMixin):
    id: str
    character_id: str
    dataset_id: Optional[str] = None
    status: str
    progress: float
    adapter_path: Optional[str] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    
    class Config:
        from_attributes = True

# Chat session schemas
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSessionCreate(BaseModel):
    character_id: str
    title: Optional[str] = None

class ChatSessionResponse(BaseModel):
    id: str
    character_id: str
    user_id: str
    title: Optional[str] = None
    is_active: bool
    message_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ChatSessionWithMessages(ChatSessionResponse):
    messages: List[ChatMessage] = []

# WebSocket message schemas
class WSMessage(BaseModel):
    type: str
    data: Dict[str, Any]

class TrainingStatusUpdate(BaseModel):
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

# Multimodal dataset schemas
class MultimodalGenerationConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    sampleCount: int = Field(100, gt=0, le=100000)
    characterCount: int = Field(5, gt=0, le=100)
    narrativeTypes: List[str] = Field(..., min_items=1)
    useMockTTS: bool = True
    ttsProvider: str = Field("orpheus", pattern="^(orpheus|kokoro|xtts|bark)$")
    outputDir: str = Field("multimodal_output")
    batchSize: int = Field(100, gt=0, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)

    @validator('narrativeTypes')
    def validate_narrative_types(cls, v):
        valid_types = {"dialogue", "monologue", "action_scene", "emotional_moment", "memory_recall", "world_description"}
        invalid_types = set(v) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid narrative types: {invalid_types}")
        return v

class MultimodalDatasetResponse(BaseModel):
    id: str
    name: str
    status: str  # pending, generating, completed, failed, cancelled
    progress: float  # 0.0 to 1.0
    config: Dict[str, Any]
    samplesGenerated: int = Field(alias="samples_generated")
    totalSamples: int = Field(alias="total_samples")
    currentStep: Optional[str] = Field(alias="current_step")
    outputPath: Optional[str] = Field(alias="output_path")
    errorMessage: Optional[str] = Field(alias="error_message")
    createdAt: datetime = Field(alias="created_at")
    updatedAt: datetime = Field(alias="updated_at")
    warnings: Optional[List[str]] = None

    class Config:
        from_attributes = True
        allow_population_by_field_name = True

class MultimodalProgressUpdate(BaseModel):
    progress: float
    currentStep: str = Field(alias="current_step")
    samplesGenerated: int = Field(alias="samples_generated")
    totalSamples: int = Field(alias="total_samples")

    class Config:
        allow_population_by_field_name = True 