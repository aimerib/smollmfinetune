"""
Narrative Engine - Advanced model architecture for character AI

This module implements:
- Dual-head model architecture for narrative + actions
- Memory system with control tokens and neural memory head
- Runtime state management for persistent world simulation
- Custom loss functions and training loops
"""

from .model import NarrativeLLM, EmotionalMomentumTracker, SurpriseDetector
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
from .memory_generator import MemoryGenerator
from .state_manager import (
    StateManager,
    EntityState,
    StateUpdate,
    StateTransaction,
    StateQuery,
    EventLog,
    StateManagerError,
    TransactionError,
)

__all__ = [
    # Model components
    "NarrativeLLM",
    "EmotionalMomentumTracker", 
    "SurpriseDetector",
    "DualHeadLoss",
    "NarrativeLLMConfig",
    
    # Memory system
    "MemoryAnnotation",
    "MemoryFormationEvent",
    "MemoryQuery",
    "MemoryRetrievalResult", 
    "MemoryTrainingBatch",
    "EmotionalMomentumState",
    "MemoryGenerator",
    
    # State management
    "StateManager",
    "EntityState",
    "StateUpdate",
    "StateTransaction",
    "StateQuery",
    "EventLog",
    "StateManagerError",
    "TransactionError",
]

__version__ = '0.1.0' 