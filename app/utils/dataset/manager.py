"""DatasetManager façade that maintains backward compatibility.

This module provides a DatasetManager class that inherits from NSFWGenerationManager
to maintain backward compatibility while utilizing the new generation package structure.
"""

import asyncio
import logging
import os
import random
import time
import re
import traceback
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datasets import Dataset

from ..generation import NSFWGenerationManager
from .models import GenerationConfig, QualityLevel
from . import character_analysis
from . import prompt_generators
from . import content_evaluation
from . import factual_qa
from . import io_manager
from . import quality_curation

logger = logging.getLogger(__name__)


class DatasetManager(NSFWGenerationManager):
    """Manages synthetic dataset generation and processing using OpenAI API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize DatasetManager with enhanced client and configuration

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OpenAI, but can be changed for compatible endpoints)
            generation_config: Enhanced generation configuration
        """
        # Initialize parent NSFWGenerationManager which handles client setup
        super().__init__(api_key=api_key, base_url=base_url, generation_config=generation_config)
        logger.info(f"DatasetManager façade initialized with model: {os.getenv('MODEL_NAME')}")

    # DatasetManager-specific methods that are not in the base classes

    def _make_card_block(self, card: Dict[str, str]) -> str:
        """Create a formatted character card block for few-shot examples"""
        return f"""Character Card:
Name: {card.get('name', 'Unknown')}
Description: {card.get('description', '')}
Personality: {card.get('personality', '')}
Scenario: {card.get('scenario', '')}"""
