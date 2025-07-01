"""
Type definitions for the Narrative Engine

This module contains shared dataclasses and types used throughout the narrative engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Action


@dataclass
class ThinkResult:
    """
    Result of agent thinking process, containing both action and internal subtext.
    
    This enables the "Iceberg Model" where characters have visible actions
    and hidden internal thoughts that create dramatic irony.
    """
    action: 'Action'
    subtext: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> bool:
        """Validate that the think result is well-formed"""
        return (
            self.action is not None and 
            self.action.validate() and
            isinstance(self.subtext, str)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format"""
        return {
            "action": self.action.to_dict(),
            "subtext": self.subtext,
            "timestamp": self.timestamp.isoformat()
        }


class SubtextParseError(Exception):
    """Exception raised when parsing subtext from model output fails"""
    pass 