"""
Safety Layer

Content filtering layer that can wrap model generation functions.
Provides input and output filtering based on configurable blocklists.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class SafetyLayer:
    """Safety layer for content filtering"""
    
    BLOCKED_MESSAGE = "[Content blocked by safety filter]"
    
    def __init__(
        self,
        blocklist: Optional[List[str]] = None,
        case_sensitive: bool = False,
        is_enabled: bool = True
    ):
        """
        Initialize safety layer.
        
        Args:
            blocklist: List of words/phrases to block
            case_sensitive: Whether to use case-sensitive matching
            is_enabled: Whether the safety layer is active
        """
        self.blocklist = blocklist or []
        self.case_sensitive = case_sensitive
        self.is_enabled = is_enabled
        
        # Pre-process blocklist for efficiency
        if not self.case_sensitive:
            self.blocklist = [word.lower() for word in self.blocklist]
    
    @classmethod
    def from_config(cls, config_path: str) -> 'SafetyLayer':
        """
        Create SafetyLayer from configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configured SafetyLayer instance
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return cls(
                blocklist=config.get('blocklist', []),
                case_sensitive=config.get('case_sensitive', False),
                is_enabled=config.get('enabled', True)
            )
        except Exception as e:
            logger.error(f"Failed to load safety config from {config_path}: {e}")
            # Return default safety layer on error
            return cls()
    
    def check_input(self, text: str) -> bool:
        """
        Check if input text is safe.
        
        Args:
            text: Input text to check
            
        Returns:
            True if safe, False if blocked
        """
        if not self.is_enabled:
            return True
        
        check_text = text if self.case_sensitive else text.lower()
        
        for blocked_word in self.blocklist:
            if blocked_word in check_text:
                logger.warning(f"Blocked input containing: {blocked_word}")
                return False
        
        return True
    
    def filter_output(self, text: str) -> str:
        """
        Filter output text for safety.
        
        Args:
            text: Output text to filter
            
        Returns:
            Filtered text or blocked message
        """
        if not self.is_enabled:
            return text
        
        check_text = text if self.case_sensitive else text.lower()
        
        for blocked_word in self.blocklist:
            if blocked_word in check_text:
                logger.warning(f"Blocked output containing: {blocked_word}")
                return self.BLOCKED_MESSAGE
        
        return text
    
    def wrap_generation_function(self, generate_fn: Callable) -> Callable:
        """
        Wrap a generation function with safety filtering.
        
        Args:
            generate_fn: Original generation function
            
        Returns:
            Wrapped function with safety filtering
        """
        def safe_generate(*args, **kwargs):
            # Check input (assume first arg is the prompt)
            if args:
                prompt = str(args[0])
                if not self.check_input(prompt):
                    logger.warning("Input blocked by safety filter")
                    return self.BLOCKED_MESSAGE
            
            # Call original function
            try:
                result = generate_fn(*args, **kwargs)
                
                # Filter output
                if isinstance(result, str):
                    return self.filter_output(result)
                else:
                    # If not string, return as-is
                    return result
                    
            except Exception as e:
                logger.error(f"Error in wrapped generation: {e}")
                raise
        
        return safe_generate
    
    def add_to_blocklist(self, words: List[str]) -> None:
        """Add words to the blocklist"""
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        
        self.blocklist.extend(words)
        self.blocklist = list(set(self.blocklist))  # Remove duplicates
    
    def remove_from_blocklist(self, words: List[str]) -> None:
        """Remove words from the blocklist"""
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        
        self.blocklist = [w for w in self.blocklist if w not in words]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety layer statistics"""
        return {
            'enabled': self.is_enabled,
            'blocklist_size': len(self.blocklist),
            'case_sensitive': self.case_sensitive
        }
    
    def disable(self) -> None:
        """Disable the safety layer"""
        self.is_enabled = False
        logger.info("Safety layer disabled")
    
    def enable(self) -> None:
        """Enable the safety layer"""
        self.is_enabled = True
        logger.info("Safety layer enabled") 