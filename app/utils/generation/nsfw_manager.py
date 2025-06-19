"""NSFW generation manager with specialized NSFW content generation capabilities.

This module contains the NSFWGenerationManager class that extends BaseGenerationManager
with NSFW-specific generation methods and quality evaluations.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

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

    def is_nsfw_content(self, sample: Dict[str, Any]) -> bool:
        """Check if sample contains NSFW content"""
        return content_evaluation.is_nsfw_content(sample)

    def categorize_nsfw_style(self, sample: Dict[str, Any]) -> str:
        """Categorize the style of NSFW content"""
        return content_evaluation.categorize_nsfw_style(sample)

    async def evaluate_nsfw_quality(
        self, response: str, character: Dict[str, Any], prompt: str
    ) -> Dict[str, float]:
        """Evaluate quality of NSFW content"""
        return await content_evaluation.evaluate_nsfw_quality(
            response, character, prompt, self._generate_single_response
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
        num_samples: int = 80,
        max_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        progress_callback: Optional[Callable] = None,
        append_to_existing: bool = True,
        custom_system_prompt: Optional[str] = None,
        extra_quality: bool = False,
        quality_level: Optional[Any] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        negative_patterns: Optional[List[str]] = None,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate NSFW dataset with specialized NSFW content generation.
        
        This is a stub method that currently delegates to the existing implementation.
        Will be enhanced with NSFW-specific generation logic in future iterations.
        """
        # TODO: Implement NSFW-specific generation logic
        # For now, delegate to the existing implementation from the original DatasetManager
        
        # Import here to avoid circular imports
        from ..dataset.manager import DatasetManager
        
        # Create a temporary DatasetManager to delegate to existing implementation
        temp_manager = DatasetManager(
            api_key=getattr(self.client, 'api_key', None),
            base_url=getattr(self.client, 'base_url', None),
            generation_config=self.generation_config
        )
        
        return await temp_manager.generate_dataset(
            character=character,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            progress_callback=progress_callback,
            append_to_existing=append_to_existing,
            custom_system_prompt=custom_system_prompt,
            extra_quality=extra_quality,
            quality_level=quality_level,
            few_shot_examples=few_shot_examples,
            negative_patterns=negative_patterns,
            **sampling_kwargs
        )

    async def generate_interactive_batch(
        self,
        character: Dict[str, Any],
        num_samples: int = 20,
        max_tokens: Optional[int] = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        progress_callback: Optional[Callable] = None,
        extra_quality: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        negative_patterns: Optional[List[str]] = None,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate interactive batch with NSFW-aware content generation.
        
        This is a stub method that currently delegates to the existing implementation.
        Will be enhanced with NSFW-specific interactive generation logic in future iterations.
        """
        # TODO: Implement NSFW-specific interactive generation logic
        # For now, delegate to the existing implementation from the original DatasetManager
        
        # Import here to avoid circular imports
        from ..dataset.manager import DatasetManager
        
        # Create a temporary DatasetManager to delegate to existing implementation
        temp_manager = DatasetManager(
            api_key=getattr(self.client, 'api_key', None),
            base_url=getattr(self.client, 'base_url', None),
            generation_config=self.generation_config
        )
        
        return await temp_manager.generate_interactive_batch(
            character=character,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            progress_callback=progress_callback,
            extra_quality=extra_quality,
            few_shot_examples=few_shot_examples,
            negative_patterns=negative_patterns,
            **sampling_kwargs
        ) 