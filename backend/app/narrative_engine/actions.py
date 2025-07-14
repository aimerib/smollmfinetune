"""
Actions module for narrative engine

Defines various action types that agents can perform.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Action:
    """Base class for all agent actions"""
    action_type: str
    agent_id: str
    timestamp: Optional[str] = None
    emotional_modifiers: List[str] = field(default_factory=list)
    
    # For testing group dynamics
    personal_benefit: float = 0.0
    group_harm: float = 0.0
    target: Optional[str] = None


class SpeakToAction(Action):
    """Action representing speech or communication"""
    def __init__(self, agent_id: str, target_id: str, message: str, emotional_tone: str = "neutral", **kwargs):
        super().__init__(action_type="speak", agent_id=agent_id, **kwargs)
        self.target_id = target_id
        self.message = message
        self.emotional_tone = emotional_tone


class MoveToAction(Action):
    """Action representing movement"""
    def __init__(self, agent_id: str, destination: str, **kwargs):
        super().__init__(action_type="move", agent_id=agent_id, **kwargs)
        self.destination = destination


class InteractWithAction(Action):
    """Action representing interaction with an object or entity"""
    def __init__(self, agent_id: str, target: str, interaction_type: str, **kwargs):
        super().__init__(action_type="interact", agent_id=agent_id, **kwargs)
        self.target = target
        self.interaction_type = interaction_type 