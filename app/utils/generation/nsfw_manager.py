"""NSFW generation manager with specialized NSFW content generation capabilities.

This module contains the NSFWGenerationManager class that extends BaseGenerationManager
with NSFW-specific generation methods and quality evaluations.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_manager import BaseGenerationManager
from ..dataset import content_evaluation

logger = logging.getLogger(__name__)


class NSFWGenerationManager(BaseGenerationManager):
    """NSFW-specialized generation manager that extends base functionality"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_config: Optional[Any] = None,
    ):
        """Initialize NSFWGenerationManager with NSFW-specific enhancements"""
        super().__init__(api_key=api_key, base_url=base_url, generation_config=generation_config)
        logger.info("NSFWGenerationManager initialized with NSFW capabilities")

    async def is_nsfw_content(self, content: str) -> bool:
        """Check if content contains NSFW content"""
        return await content_evaluation.is_nsfw_content(self.client, content)

    async def categorize_nsfw_style(self, content: str) -> str:
        """Categorize the style of NSFW content"""
        return await content_evaluation.categorize_nsfw_style(self.client, content)

    async def evaluate_nsfw_quality(
        self, response: str, character: Dict[str, Any], prompt: str
    ) -> Dict[str, float]:
        """Evaluate quality of NSFW content"""
        return await content_evaluation.evaluate_nsfw_quality(
            self.client, response, character, prompt
        )

    def analyze_character_intimacy_style(
        self, character: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze character's intimacy and NSFW style preferences"""
        return content_evaluation.analyze_character_intimacy_style(character)

    def extract_intimate_speech_patterns(
        self, character: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract intimate speech patterns from character definition"""
        return content_evaluation.extract_intimate_speech_patterns(character)

    async def generate_nsfw_dataset(
        self,
        character: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate NSFW dataset with specialized NSFW content generation.
        
        This method will be enhanced with NSFW-specific generation logic in future iterations.
        For now, it calls the base dataset generation method.
        """
        logger.info("Calling base generate_dataset from NSFW manager.")
        # The base manager now has the implementation.
        return await super().generate_dataset(character=character, **kwargs)

    async def generate_interactive_batch(
        self,
        character: Dict[str, Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate interactive batch with NSFW-aware content generation.
        
        This method will be enhanced with NSFW-specific interactive generation logic in future iterations.
        For now, it calls the base interactive batch generation method.
        """
        logger.info("Calling base generate_interactive_batch from NSFW manager.")
        # The base manager now has the implementation.
        return await super().generate_interactive_batch(character=character, **kwargs) 