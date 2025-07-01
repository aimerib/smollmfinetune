"""
Control Token Processing for narrative and UI control.

Processes control tokens from the model's control head to trigger
actions, UI updates, and narrative flow changes.
"""

import asyncio
import re
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ControlActionType(Enum):
    """Types of control actions"""
    EMOTION = "emotion"
    ACTION = "action"
    UI_COMMAND = "ui_command"
    SCENE_CHANGE = "scene_change"
    MOOD_SHIFT = "mood_shift"
    NARRATIVE_FLOW = "narrative_flow"
    CUSTOM = "custom"


@dataclass
class ControlAction:
    """Parsed control action"""
    action_type: ControlActionType
    token: str
    probability: float
    parsed_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "token": self.token,
            "probability": self.probability,
            "data": self.parsed_data
        }


@dataclass
class UICommand:
    """UI manipulation command"""
    action: str  # highlight, show_image, fade_out, etc.
    target: Optional[str] = None
    resource: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class ControlTokenProcessor:
    """
    Processes control tokens from model outputs.
    
    Features:
    - Token parsing and classification
    - Action handler registration
    - UI command processing
    - Narrative flow control
    - Async event dispatching
    """
    
    def __init__(self):
        """Initialize control token processor"""
        # Handler registrations
        self.action_handlers: Dict[str, List[Callable]] = {}
        self.ui_handlers: List[Callable] = []
        self.flow_handlers: List[Callable] = []
        
        # Token patterns
        self.token_patterns = {
            ControlActionType.EMOTION: re.compile(r'<emotion_(\w+)>'),
            ControlActionType.ACTION: re.compile(r'<action_(\w+)>'),
            ControlActionType.UI_COMMAND: re.compile(r'<ui_(\w+)>(.*?)</ui_\w+>'),
            ControlActionType.SCENE_CHANGE: re.compile(r'<scene_change>(.*?)</scene_change>'),
            ControlActionType.MOOD_SHIFT: re.compile(r'<mood_shift>(.*?)</mood_shift>'),
            ControlActionType.NARRATIVE_FLOW: re.compile(r'<narrative_(\w+)>(.*?)</narrative_\w+>')
        }
        
        logger.info("ControlTokenProcessor initialized")
    
    def extract_control_tokens(self, output: Any) -> List[Dict[str, Any]]:
        """
        Extract control tokens from model output.
        
        Args:
            output: Triple-head model output
            
        Returns:
            List of extracted tokens with metadata
        """
        tokens = []
        
        # Get tokens from control head
        if hasattr(output, 'control_tokens'):
            for token_data in output.control_tokens:
                token = token_data.get("token", "")
                probability = token_data.get("probability", 0.0)
                
                # Classify token type
                action_type = self._classify_token(token)
                
                tokens.append({
                    "token": token,
                    "probability": probability,
                    "type": action_type.value if action_type else "unknown"
                })
        
        # Also extract from generation text
        if hasattr(output, 'generation_text'):
            text_tokens = self._extract_from_text(output.generation_text)
            tokens.extend(text_tokens)
        
        return tokens
    
    async def process_tokens(self, tokens: List[Dict[str, Any]], 
                           context: Optional[Dict[str, Any]] = None):
        """
        Process extracted tokens and trigger handlers.
        
        Args:
            tokens: List of token dictionaries
            context: Optional context for handlers
        """
        if context is None:
            context = {}
        
        # Parse and classify tokens
        actions = []
        for token_data in tokens:
            action = self._parse_token(token_data)
            if action:
                actions.append(action)
        
        # Process each action
        tasks = []
        for action in actions:
            # Dispatch to appropriate handlers
            if action.action_type == ControlActionType.UI_COMMAND:
                tasks.extend([
                    handler(self._create_ui_event(action, context))
                    for handler in self.ui_handlers
                ])
            
            elif action.action_type in [ControlActionType.SCENE_CHANGE, 
                                      ControlActionType.MOOD_SHIFT,
                                      ControlActionType.NARRATIVE_FLOW]:
                tasks.extend([
                    handler(self._create_flow_event(action, context))
                    for handler in self.flow_handlers
                ])
            
            # Check for specific action handlers
            if action.token in self.action_handlers:
                event = {"token": action.token, "action": action, "context": context}
                tasks.extend([
                    handler(event)
                    for handler in self.action_handlers[action.token]
                ])
        
        # Execute all handlers concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def register_action_handler(self, token: str, handler: Callable):
        """
        Register a handler for specific token.
        
        Args:
            token: Token to handle
            handler: Async function to call
        """
        if token not in self.action_handlers:
            self.action_handlers[token] = []
        self.action_handlers[token].append(handler)
        logger.debug(f"Registered handler for token: {token}")
    
    def register_ui_handler(self, handler: Callable):
        """Register a UI event handler"""
        self.ui_handlers.append(handler)
        logger.debug("Registered UI handler")
    
    def register_flow_handler(self, handler: Callable):
        """Register a narrative flow handler"""
        self.flow_handlers.append(handler)
        logger.debug("Registered flow handler")
    
    def _classify_token(self, token: str) -> Optional[ControlActionType]:
        """Classify token type based on pattern"""
        for action_type, pattern in self.token_patterns.items():
            if pattern.match(token):
                return action_type
        return None
    
    def _parse_token(self, token_data: Dict[str, Any]) -> Optional[ControlAction]:
        """Parse token into control action"""
        token = token_data.get("token", "")
        probability = token_data.get("probability", 0.0)
        
        # Try each pattern
        for action_type, pattern in self.token_patterns.items():
            match = pattern.match(token)
            if match:
                parsed_data = self._extract_token_data(action_type, match)
                return ControlAction(
                    action_type=action_type,
                    token=token,
                    probability=probability,
                    parsed_data=parsed_data
                )
        
        return None
    
    def _extract_token_data(self, action_type: ControlActionType, 
                           match: re.Match) -> Dict[str, Any]:
        """Extract data from token match"""
        if action_type == ControlActionType.EMOTION:
            return {"emotion": match.group(1)}
        
        elif action_type == ControlActionType.ACTION:
            return {"action": match.group(1)}
        
        elif action_type == ControlActionType.UI_COMMAND:
            command = match.group(1)
            content = match.group(2) if match.lastindex >= 2 else ""
            
            # Parse UI command specifics
            if command == "highlight":
                return {"command": "highlight", "target": content}
            elif command == "show_image":
                return {"command": "show_image", "resource": content}
            else:
                return {"command": command, "content": content}
        
        elif action_type == ControlActionType.SCENE_CHANGE:
            return {"new_scene": match.group(1)}
        
        elif action_type == ControlActionType.MOOD_SHIFT:
            return {"new_mood": match.group(1)}
        
        elif action_type == ControlActionType.NARRATIVE_FLOW:
            flow_type = match.group(1)
            content = match.group(2) if match.lastindex >= 2 else ""
            return {"flow_type": flow_type, "content": content}
        
        return {}
    
    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract control tokens embedded in generation text"""
        tokens = []
        
        for action_type, pattern in self.token_patterns.items():
            for match in pattern.finditer(text):
                tokens.append({
                    "token": match.group(0),
                    "probability": 1.0,  # Embedded tokens have full confidence
                    "type": action_type.value
                })
        
        return tokens
    
    def _create_ui_event(self, action: ControlAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI event from action"""
        data = action.parsed_data
        
        if data.get("command") == "highlight":
            return {
                "action": "highlight",
                "target": data.get("target", ""),
                "context": context
            }
        
        elif data.get("command") == "show_image":
            return {
                "action": "show_image",
                "resource": data.get("resource", ""),
                "context": context
            }
        
        else:
            return {
                "action": data.get("command", "unknown"),
                "data": data,
                "context": context
            }
    
    def _create_flow_event(self, action: ControlAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create narrative flow event from action"""
        if action.action_type == ControlActionType.SCENE_CHANGE:
            return {
                "type": "scene_change",
                "target": action.parsed_data.get("new_scene", ""),
                "context": context
            }
        
        elif action.action_type == ControlActionType.MOOD_SHIFT:
            return {
                "type": "mood_shift",
                "mood": action.parsed_data.get("new_mood", ""),
                "context": context
            }
        
        elif action.action_type == ControlActionType.NARRATIVE_FLOW:
            return {
                "type": action.parsed_data.get("flow_type", "unknown"),
                "content": action.parsed_data.get("content", ""),
                "context": context
            }
        
        return {"type": "unknown", "action": action, "context": context} 