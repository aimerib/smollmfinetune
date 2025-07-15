"""
Subtext Parser for Iceberg Model

Parses model output to extract both SUBTEXT and ACTION blocks,
enabling the dual-output iceberg model functionality.
"""

import re
import logging
from typing import Tuple, Dict, Any
from .types import SubtextParseError

logger = logging.getLogger(__name__)

# Regex patterns for extracting tagged content
SUBTEXT_PATTERN = re.compile(r'\[SUBTEXT\](.*?)\[/SUBTEXT\]', re.DOTALL | re.IGNORECASE)
ACTION_PATTERN = re.compile(r'\[ACTION\](.*?)\[/ACTION\]', re.DOTALL | re.IGNORECASE)


def parse_subtext_and_action(model_output: str) -> Tuple[str, str]:
    """
    Parse model output to extract subtext and action content.
    
    Args:
        model_output: Raw text output from the narrative model
        
    Returns:
        Tuple of (subtext, action_text)
        
    Raises:
        SubtextParseError: If parsing fails or required tags are missing
    """
    if not isinstance(model_output, str):
        raise SubtextParseError(f"Expected string input, got {type(model_output)}")
    
    if not model_output.strip():
        raise SubtextParseError("Empty model output provided")
    
    # Extract subtext
    subtext_match = SUBTEXT_PATTERN.search(model_output)
    if not subtext_match:
        logger.warning("No [SUBTEXT] tags found in model output")
        subtext = ""  # Graceful fallback
    else:
        subtext = subtext_match.group(1).strip()
    
    # Extract action
    action_match = ACTION_PATTERN.search(model_output)
    if not action_match:
        raise SubtextParseError("No [ACTION] tags found in model output")
    
    action_text = action_match.group(1).strip()
    
    if not action_text:
        raise SubtextParseError("Empty [ACTION] content found")
    
    logger.debug(f"Parsed subtext: '{subtext[:50]}...' and action: '{action_text[:50]}...'")
    
    return subtext, action_text


def validate_tagged_output(model_output: str) -> Dict[str, bool]:
    """
    Validate that the model output contains the expected tags.
    
    Args:
        model_output: Raw text output from the narrative model
        
    Returns:
        Dictionary with validation results for each tag type
    """
    result = {
        "has_subtext_tags": bool(SUBTEXT_PATTERN.search(model_output)),
        "has_action_tags": bool(ACTION_PATTERN.search(model_output)),
        "subtext_content_exists": False,
        "action_content_exists": False
    }
    
    # Check content existence
    subtext_match = SUBTEXT_PATTERN.search(model_output)
    if subtext_match:
        result["subtext_content_exists"] = bool(subtext_match.group(1).strip())
    
    action_match = ACTION_PATTERN.search(model_output)
    if action_match:
        result["action_content_exists"] = bool(action_match.group(1).strip())
    
    return result 