"""Generation package for managing dataset creation and generation workflows.

This package provides a clean separation between base generation functionality
and specialized generation managers (like NSFW) for better code organization.
"""

from .base_manager import BaseGenerationManager
from .nsfw_manager import NSFWGenerationManager

__all__ = [
    'BaseGenerationManager',
    'NSFWGenerationManager',
] 