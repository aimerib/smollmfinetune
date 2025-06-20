"""
Pages package for the Character AI Training Studio.

This package contains extracted page functions from the main app.py file
for better maintainability and organization.

Available pages:
- character_upload: Character card upload and management
- world_management: World creation and lore management  
- training_config: Training configuration and profile management
- model_testing: Model testing and inference
- model_management: Model file management and organization
"""

# Import all page functions for easy access
try:
    from .character_upload import page_character_upload
except ImportError:
    pass

try:
    from .world_management import page_world_management
except ImportError:
    pass

try:
    from .training_config import page_training_config
except ImportError:
    pass

try:
    from .training_dashboard import page_training_dashboard
except ImportError:
    pass

try:
    from .model_testing import page_model_testing
except ImportError:
    pass

try:
    from .model_management import page_model_management
except ImportError:
    pass

__all__ = [
    'page_character_upload',
    'page_world_management', 
    'page_training_config',
    'page_training_dashboard',
    'page_model_testing',
    'page_model_management'
] 