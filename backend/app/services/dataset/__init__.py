"""Dataset generation and management utilities.

This package provides comprehensive tools for generating, evaluating, and managing
synthetic datasets for character-based AI training.
"""

from .manager import DatasetManager
from .models import GenerationConfig, QualityLevel
from .quality import ProgressiveRefiner, EnhancedQualityFilter
from .factual_qa import QAPair

# Import main functions from modules
from . import character_analysis
from . import prompt_generators
from . import content_evaluation
from . import factual_qa
from . import io_manager
from . import quality_curation

__all__ = [
    # Main classes
    'DatasetManager',
    'GenerationConfig',
    'QualityLevel',
    'ProgressiveRefiner',
    'EnhancedQualityFilter',
    'QAPair',
    
    # Modules
    'character_analysis',
    'prompt_generators',
    'content_evaluation',
    'factual_qa',
    'io_manager',
    'quality_curation',
]

# Version info
__version__ = '1.0.0'
__author__ = 'SmolLM Finetune Team'
__description__ = 'Synthetic dataset generation and management for character-based AI training'