"""
Narrative Engine - Dual-Head Architecture for Bilingual AI

This module implements the innovative dual-head architecture that enables
models to seamlessly switch between natural language generation and
structured action/tool use. Now enhanced with memory formation capabilities.
"""

from .model import NarrativeLLM, create_narrative_model, NarrativeLLMConfig
from .data_pipeline import DatasetProcessor, NarrativeDataset
from .data_schema import DatasetSample, Turn
from .loss import DualHeadLoss
from .config import NarrativeLLMConfig
from .memory_schema import (
    MemoryAnnotation,
    MemoryFormationEvent,
    MemoryQuery,
    MemoryRetrievalResult,
    MemoryTrainingBatch,
    EmotionalMomentumState,
)
from .memory_generator import (
    MemoryGenerator,
    MemoryGenerationRequest,
    generate_memories_for_dataset,
)

__all__ = [
    # Core model
    'NarrativeLLM',
    'create_narrative_model',
    'NarrativeLLMConfig',
    
    # Data processing
    'DatasetProcessor',
    'NarrativeDataset', 
    'DatasetSample',
    'Turn',
    
    # Loss function
    'DualHeadLoss',
    
    # Memory schemas
    'MemoryAnnotation',
    'MemoryFormationEvent', 
    'MemoryQuery',
    'MemoryRetrievalResult',
    'MemoryTrainingBatch',
    'EmotionalMomentumState',
    
    # Memory generation
    'MemoryGenerator',
    'MemoryGenerationRequest',
    'generate_memories_for_dataset',
]

__version__ = '0.1.0' 