"""
Narrative Engine

A sophisticated narrative language model with triple-head architecture:
- Generation head for natural language
- Control head for emotional/cognitive tokens  
- Memory head for external memory vectors

Key components:
- Triple-head model architecture for narrative + actions + memory
- External memory integration (Method A & B)
- Emotional momentum tracking and surprise detection
- Contamination isolation via MoE routing
- Training pipelines and evaluation harnesses
"""

from .config import NarrativeLLMConfig
from .model import NarrativeLLM, create_narrative_model
from .loss import DualHeadLoss, TripleHeadLoss
from .data_pipeline import DatasetProcessor
from .data_schema import Turn, DatasetSample
try:
    from .clara_trainer import CLARALoopTrainingManager, CLARATrainer, create_clara_training_manager
except ImportError:
    pass  # Optional components
try:
    from .contamination_moe_trainer import ContaminationMoETrainingManager
except ImportError:
    pass
try:
    from .contamination_moe import ContaminationIsolationMoE
except ImportError:
    pass
try:
    from .state_manager import StateManager
except ImportError:
    pass
try:
    from .memory_generator import MemoryGenerator
except ImportError:
    pass
try:
    from .memory_schema import MemorySchema, MemoryBlock
except ImportError:
    pass
try:
    from .evaluation import (
        run_evaluation_suite,
        eval_basic_generation,
        eval_triple_head_sanity,
        eval_training_progress
    )
except ImportError:
    pass  # Evaluation components are optional

# Version info
__version__ = "0.4.0"

# Main exports
__all__ = [
    # Core Model
    "NarrativeLLM",
    "NarrativeLLMConfig", 
    "create_narrative_model",
    
    # Loss Functions
    "DualHeadLoss",     # Backward compatibility
    "TripleHeadLoss",   # New triple-head loss
    
    # Data Processing
    "DatasetProcessor",
    "Turn",
    "DatasetSample",
] 