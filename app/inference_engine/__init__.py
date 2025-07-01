"""
Production Inference Engine for Narrative-LLM (R4-10)

High-performance inference server with triple-head outputs,
hot-swappable adapters, and memory integration.
"""

from .core import (
    ProductionInferenceEngine,
    InferenceRequest,
    InferenceResponse,
    TripleHeadOutput,
    ControlToken,
    MemoryMetadata
)

from .adapter_manager import (
    AdapterManager,
    AdapterMetadata,
    AdapterVersion
)

from .memory_service import (
    MemoryService,
    MemoryVector,
    MemorySearchResult
)

from .session_manager import (
    SessionStateManager,
    SessionState,
    CharacterState
)

from .control_processor import (
    ControlTokenProcessor,
    ControlAction,
    UICommand
)

__all__ = [
    # Core
    'ProductionInferenceEngine',
    'InferenceRequest',
    'InferenceResponse',
    'TripleHeadOutput',
    'ControlToken',
    'MemoryMetadata',
    
    # Adapter Management
    'AdapterManager',
    'AdapterMetadata',
    'AdapterVersion',
    
    # Memory Service
    'MemoryService',
    'MemoryVector',
    'MemorySearchResult',
    
    # Session Management
    'SessionStateManager',
    'SessionState',
    'CharacterState',
    
    # Control Processing
    'ControlTokenProcessor',
    'ControlAction',
    'UICommand'
] 