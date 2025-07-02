"""
Type definitions for the Narrative Engine

This module contains shared dataclasses and types used throughout the narrative engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Action


@dataclass
class ThinkResult:
    """
    Enhanced result of agent thinking process for triple-head architecture.
    
    Contains:
    - action: The decided action (from generation head)
    - subtext: Internal monologue (enables "Iceberg Model")
    - emotional_state: Current emotional state (from control head)
    - memory_formation: Memory to be formed (from memory head)
    - next_recirculation: Emotional tokens for next turn
    """
    action: 'Action'
    subtext: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Enhanced fields for triple-head architecture
    emotional_state: Optional[Dict[str, float]] = None
    memory_formation: Optional[Dict[str, Any]] = None
    next_recirculation: Optional[List[str]] = None
    
    def validate(self) -> bool:
        """Validate that the think result is well-formed"""
        return (
            self.action is not None and 
            self.action.validate() and
            isinstance(self.subtext, str)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format"""
        result = {
            "action": self.action.to_dict(),
            "subtext": self.subtext,
            "timestamp": self.timestamp.isoformat()
        }
        
        # Add enhanced fields if present
        if self.emotional_state:
            result["emotional_state"] = self.emotional_state
        if self.memory_formation:
            result["memory_formation"] = self.memory_formation
        if self.next_recirculation:
            result["next_recirculation"] = self.next_recirculation
            
        return result


class SubtextParseError(Exception):
    """Exception raised when parsing subtext from model output fails"""
    pass 