"""
N-Script (Narrative Scripting Engine) with Triple-Head Support

Provides a YAML-based DSL for creating narrative scripts that can control
all three heads of the narrative model: generation, control, and memory.
"""

import asyncio
import logging
import json
import yaml
import jsonschema
import re
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from .state_manager import StateManager, EntityState, EventLog, StateUpdate
from .agent import Action, ActionResult

logger = logging.getLogger(__name__)


# === Enums ===

class TriggerType(str, Enum):
    """Types of triggers supported by N-Script."""
    ON_ENTER_LOCATION = "ON_ENTER_LOCATION"
    ON_MEMORY_FORMATION = "ON_MEMORY_FORMATION"
    ON_EMOTIONAL_STATE = "ON_EMOTIONAL_STATE"
    ON_GENERATION_QUALITY = "ON_GENERATION_QUALITY"
    ON_HEAD_COORDINATION = "ON_HEAD_COORDINATION"


class ActionType(str, Enum):
    """Types of actions supported by N-Script."""
    TRIPLE_HEAD_ACTION = "TRIPLE_HEAD_ACTION"
    GENERATION_OVERRIDE = "GENERATION_OVERRIDE"
    CONTROL_INJECTION = "CONTROL_INJECTION"
    MEMORY_FORMATION = "MEMORY_FORMATION"
    HEAD_SYNCHRONIZATION = "HEAD_SYNCHRONIZATION"


# === Pydantic Models ===

class GenerationParams(BaseModel):
    """Parameters for generation head control."""
    style: Optional[str] = None
    tone: Optional[str] = None
    narrative_focus: Optional[str] = None
    override_duration: Optional[int] = Field(None, ge=1)


class ControlParams(BaseModel):
    """Parameters for control head management."""
    emotions: List[str] = Field(default_factory=list)
    mood_shift: Optional[str] = None
    control_tokens: List[str] = Field(default_factory=list)


class MemoryParams(BaseModel):
    """Parameters for memory head operations."""
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotional_impact: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotional_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    content: Optional[str] = None


class CoordinationParams(BaseModel):
    """Parameters for head coordination."""
    sync_emotional_state: bool = False
    align_memory_focus: bool = False
    harmonize_narrative_tone: bool = False


class BaseActionData(BaseModel):
    """Base action data for N-Script actions."""
    type: str
    message: Optional[str] = None
    target_location: Optional[str] = None
    item_id: Optional[str] = None
    new_goal_description: Optional[str] = None


class NScriptTrigger(BaseModel):
    """Represents a script trigger with conditions."""
    type: TriggerType
    location_id: Optional[str] = None
    actor_filter: Optional[str] = None
    target_agent_id: Optional[str] = None
    once: bool = False
    conditions: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='before')
    @classmethod
    def location_required_for_location_trigger(cls, values):
        if isinstance(values, dict):
            if values.get('type') == TriggerType.ON_ENTER_LOCATION and not values.get('location_id'):
                raise ValueError("location_id is required for ON_ENTER_LOCATION triggers")
        return values


class NScriptAction(BaseModel):
    """Represents a script action with head-specific parameters."""
    type: ActionType
    target_agent_id: str
    generation_params: Optional[GenerationParams] = None
    control_params: Optional[ControlParams] = None
    memory_params: Optional[MemoryParams] = None
    coordination_params: Optional[CoordinationParams] = None
    action: Optional[BaseActionData] = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_action_params(cls, values):
        if isinstance(values, dict):
            action_type = values.get('type')
            
            if action_type == ActionType.TRIPLE_HEAD_ACTION:
                # Require at least one head parameter
                has_params = any([
                    values.get('generation_params'),
                    values.get('control_params'),
                    values.get('memory_params')
                ])
                if not has_params:
                    raise ValueError("TRIPLE_HEAD_ACTION requires at least one head parameter")
            
            elif action_type == ActionType.GENERATION_OVERRIDE:
                if not values.get('generation_params'):
                    raise ValueError("GENERATION_OVERRIDE requires generation_params")
            
            elif action_type == ActionType.CONTROL_INJECTION:
                if not values.get('control_params'):
                    raise ValueError("CONTROL_INJECTION requires control_params")
            
            elif action_type == ActionType.MEMORY_FORMATION:
                if not values.get('memory_params'):
                    raise ValueError("MEMORY_FORMATION requires memory_params")
            
            elif action_type == ActionType.HEAD_SYNCHRONIZATION:
                if not values.get('coordination_params'):
                    raise ValueError("HEAD_SYNCHRONIZATION requires coordination_params")
        
        return values


class Script(BaseModel):
    """Represents a complete N-Script."""
    script_id: str = Field(..., min_length=1)
    trigger: NScriptTrigger
    actions: List[NScriptAction] = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Script':
        """Create Script from dictionary with validation."""
        return cls(**data)


# === Core Components ===

class ScriptManager:
    """Manages loading, validation, and storage of N-Scripts."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize ScriptManager with JSON schema."""
        self.scripts: Dict[str, Script] = {}
        self.schema_path = schema_path or Path(__file__).parent / "nscript_schema.json"
        self.load_schema()
        
        logger.info("Initialized ScriptManager")
    
    def load_schema(self) -> None:
        """Load JSON schema for validation."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
            logger.debug("Loaded N-Script JSON schema")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self.schema = None
    
    def load_script_from_yaml(self, yaml_content: str) -> Script:
        """Load a script from YAML content."""
        try:
            data = yaml.safe_load(yaml_content)
            script = Script.from_dict(data)
            logger.debug(f"Loaded script: {script.script_id}")
            return script
        except Exception as e:
            logger.error(f"Failed to load script from YAML: {e}")
            raise
    
    def load_script_from_file(self, file_path: Path) -> Script:
        """Load a script from a file."""
        with open(file_path, 'r') as f:
            content = f.read()
        return self.load_script_from_yaml(content)
    
    def validate_script(self, script_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate script data against JSON schema."""
        errors = []
        
        # Basic validation
        if not script_data.get('script_id'):
            errors.append("script_id cannot be empty")
        
        if not script_data.get('trigger'):
            errors.append("trigger is required")
        
        if not script_data.get('actions'):
            errors.append("actions cannot be empty")
        
        # JSON Schema validation if available
        if self.schema:
            try:
                jsonschema.validate(script_data, self.schema)
            except jsonschema.ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")
        
        return len(errors) == 0, errors
    
    def add_script(self, script: Script) -> None:
        """Add or update a script."""
        self.scripts[script.script_id] = script
        logger.info(f"Added script: {script.script_id}")
    
    def get_script(self, script_id: str) -> Optional[Script]:
        """Get a script by ID."""
        return self.scripts.get(script_id)
    
    def get_all_scripts(self) -> List[Script]:
        """Get all loaded scripts."""
        return list(self.scripts.values())
    
    def get_scripts_by_trigger_type(self, trigger_type: str) -> List[Script]:
        """Get scripts filtered by trigger type."""
        return [
            script for script in self.scripts.values()
            if script.trigger.type == trigger_type
        ]
    
    def remove_script(self, script_id: str) -> bool:
        """Remove a script."""
        if script_id in self.scripts:
            del self.scripts[script_id]
            logger.info(f"Removed script: {script_id}")
            return True
        return False


class TriggerMonitor:
    """Monitors events and triggers N-Scripts based on conditions."""
    
    def __init__(self, state_manager: StateManager, script_manager: ScriptManager):
        """Initialize TriggerMonitor."""
        self.state_manager = state_manager
        self.script_manager = script_manager
        self.activated_scripts: List[Script] = []
        self.triggered_once: set[str] = set()  # Track 'once' triggers
        self.monitoring = False
        
        logger.info("Initialized TriggerMonitor")
    
    def start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        self.monitoring = True
        logger.info("Started N-Script trigger monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitoring = False
        logger.info("Stopped N-Script trigger monitoring")
    
    def get_activated_scripts(self) -> List[Script]:
        """Get list of activated scripts."""
        return self.activated_scripts.copy()
    
    def clear_activated_scripts(self) -> None:
        """Clear the activated scripts list."""
        self.activated_scripts.clear()
    
    def check_location_triggers(self, entity_id: str, new_location: str) -> None:
        """Check for location-based triggers."""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_ENTER_LOCATION")
        
        for script in scripts:
            trigger = script.trigger
            
            # Check if already triggered (for 'once' triggers)
            if trigger.once and script.script_id in self.triggered_once:
                continue
            
            # Check location match
            if trigger.location_id != new_location:
                continue
            
            # Check actor filter
            if trigger.actor_filter and trigger.actor_filter != entity_id:
                continue
            
            # Trigger activated!
            self.activated_scripts.append(script)
            
            if trigger.once:
                self.triggered_once.add(script.script_id)
            
            logger.info(f"Location trigger activated: {script.script_id}")
    
    async def handle_memory_formation_event(self, agent_id: str, memory_data: Dict[str, Any]) -> None:
        """Handle memory formation events."""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_MEMORY_FORMATION")
        
        for script in scripts:
            if await self._check_memory_conditions(script.trigger, memory_data):
                self.activated_scripts.append(script)
                logger.info(f"Memory formation trigger activated: {script.script_id}")
    
    async def handle_emotional_state_event(self, emotional_data: Dict[str, Any]) -> None:
        """Handle emotional state change events."""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_EMOTIONAL_STATE")
        
        for script in scripts:
            if await self._check_emotional_conditions(script.trigger, emotional_data):
                self.activated_scripts.append(script)
                logger.info(f"Emotional state trigger activated: {script.script_id}")
    
    async def _check_memory_conditions(self, trigger: NScriptTrigger, memory_data: Dict[str, Any]) -> bool:
        """Check if memory data meets trigger conditions."""
        conditions = trigger.conditions
        
        for condition_key, condition_value in conditions.items():
            memory_value = memory_data.get(condition_key)
            
            if not self._evaluate_condition(memory_value, condition_value):
                return False
        
        return True
    
    async def _check_emotional_conditions(self, trigger: NScriptTrigger, emotional_data: Dict[str, Any]) -> bool:
        """Check if emotional data meets trigger conditions."""
        # Check target agent filter
        if trigger.target_agent_id:
            if emotional_data.get('agent_id') != trigger.target_agent_id:
                return False
        
        conditions = trigger.conditions
        
        for condition_key, condition_value in conditions.items():
            if condition_key == "emotional_state":
                # Special handling for emotional state patterns
                current_state = emotional_data.get('emotional_state', '')
                if not self._match_emotional_state(current_state, condition_value):
                    return False
            else:
                data_value = emotional_data.get(condition_key)
                if not self._evaluate_condition(data_value, condition_value):
                    return False
        
        return True
    
    def _evaluate_condition(self, actual_value: Any, condition: str) -> bool:
        """Evaluate a condition string against an actual value."""
        if isinstance(condition, str) and condition.startswith('>'):
            threshold = float(condition[1:].strip())
            return float(actual_value or 0) > threshold
        elif isinstance(condition, str) and condition.startswith('<'):
            threshold = float(condition[1:].strip())
            return float(actual_value or 0) < threshold
        else:
            return actual_value == condition
    
    def _match_emotional_state(self, current_state: str, pattern: str) -> bool:
        """Match emotional state against pattern (e.g., 'angry|frustrated')."""
        if '|' in pattern:
            states = pattern.split('|')
            return current_state.lower() in [s.strip().lower() for s in states]
        else:
            return current_state.lower() == pattern.lower()


class ActionExecutor:
    """Executes N-Script actions with triple-head model integration."""
    
    def __init__(self, state_manager: StateManager, narrative_model=None):
        """Initialize ActionExecutor."""
        self.state_manager = state_manager
        self.narrative_model = narrative_model
        
        logger.info("Initialized ActionExecutor")
    
    async def execute_action(self, action: NScriptAction) -> ActionResult:
        """Execute an N-Script action."""
        try:
            # Check if target agent exists
            target_entity = self.state_manager.get_entity(action.target_agent_id)
            if not target_entity:
                return ActionResult(
                    success=False,
                    message=f"Target agent {action.target_agent_id} not found"
                )
            
            # Route to appropriate handler
            if action.type == ActionType.TRIPLE_HEAD_ACTION:
                return await self._execute_triple_head_action(action)
            elif action.type == ActionType.GENERATION_OVERRIDE:
                return await self._execute_generation_override(action)
            elif action.type == ActionType.CONTROL_INJECTION:
                return await self._execute_control_injection(action)
            elif action.type == ActionType.MEMORY_FORMATION:
                return await self._execute_memory_formation(action)
            elif action.type == ActionType.HEAD_SYNCHRONIZATION:
                return await self._execute_head_synchronization(action)
            else:
                return ActionResult(
                    success=False,
                    message=f"Unknown action type: {action.type}"
                )
        
        except Exception as e:
            logger.error(f"Error executing action {action.type}: {e}")
            return ActionResult(
                success=False,
                message=f"Action execution failed: {str(e)}"
            )
    
    async def _execute_triple_head_action(self, action: NScriptAction) -> ActionResult:
        """Execute a coordinated triple-head action."""
        if not self.narrative_model:
            # Fallback behavior without model
            return ActionResult(
                success=True,
                message="Triple-head action executed (fallback mode)",
                side_effects=["No AI model available - using fallback behavior"]
            )
        
        try:
            # Call the narrative model with all head parameters
            result = await self.narrative_model.generate_with_triple_head_control(
                generation_params=action.generation_params.model_dump() if action.generation_params else {},
                control_params=action.control_params.model_dump() if action.control_params else {},
                memory_params=action.memory_params.model_dump() if action.memory_params else {},
                target_agent_id=action.target_agent_id
            )
            
            return ActionResult(
                success=True,
                message="Triple-head action executed successfully",
                state_changes={"last_nscript_action": "triple_head"},
                side_effects=[f"Model response generated for {action.target_agent_id}"]
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Triple-head action failed: {str(e)}"
            )
    
    async def _execute_generation_override(self, action: NScriptAction) -> ActionResult:
        """Execute generation parameter override."""
        try:
            if self.narrative_model and hasattr(self.narrative_model, 'override_generation_params'):
                result = await self.narrative_model.override_generation_params(
                    agent_id=action.target_agent_id,
                    params=action.generation_params.model_dump()
                )
            
            return ActionResult(
                success=True,
                message="Generation override applied",
                state_changes={"generation_override_active": True}
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Generation override failed: {str(e)}"
            )
    
    async def _execute_control_injection(self, action: NScriptAction) -> ActionResult:
        """Execute control head injection."""
        try:
            if self.narrative_model and hasattr(self.narrative_model, 'inject_control_state'):
                result = await self.narrative_model.inject_control_state(
                    agent_id=action.target_agent_id,
                    control_params=action.control_params.model_dump()
                )
            
            return ActionResult(
                success=True,
                message="Control injection applied",
                state_changes={"control_injection_active": True},
                side_effects=[f"Emotional state modified for {action.target_agent_id}"]
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Control injection failed: {str(e)}"
            )
    
    async def _execute_memory_formation(self, action: NScriptAction) -> ActionResult:
        """Execute explicit memory formation."""
        try:
            memory_data = action.memory_params.model_dump()
            
            # Add memory to state manager
            self.state_manager.add_memory_to_character(
                action.target_agent_id,
                memory_data
            )
            
            # Also call narrative model if available
            if self.narrative_model and hasattr(self.narrative_model, 'form_explicit_memory'):
                result = await self.narrative_model.form_explicit_memory(
                    agent_id=action.target_agent_id,
                    memory_data=memory_data
                )
            
            return ActionResult(
                success=True,
                message="Memory formation completed",
                state_changes={"memories_count": "+1"},
                side_effects=[f"Memory added to {action.target_agent_id}"]
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Memory formation failed: {str(e)}"
            )
    
    async def _execute_head_synchronization(self, action: NScriptAction) -> ActionResult:
        """Execute head synchronization."""
        try:
            if self.narrative_model and hasattr(self.narrative_model, 'synchronize_heads'):
                result = await self.narrative_model.synchronize_heads(
                    agent_id=action.target_agent_id,
                    coordination_params=action.coordination_params.model_dump()
                )
            
            return ActionResult(
                success=True,
                message="Head synchronization completed",
                state_changes={"heads_synchronized": True},
                side_effects=[f"All heads synchronized for {action.target_agent_id}"]
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Head synchronization failed: {str(e)}"
            )


# === Integration with State Manager ===

class NScriptStateIntegration:
    """Integrates N-Script with StateManager for automatic trigger monitoring."""
    
    def __init__(self, state_manager: StateManager, trigger_monitor: TriggerMonitor):
        """Initialize integration."""
        self.state_manager = state_manager
        self.trigger_monitor = trigger_monitor
        self.monitoring = False
        
        logger.info("Initialized N-Script state integration")
    
    def start_integration(self) -> None:
        """Start monitoring state changes for script triggers."""
        self.monitoring = True
        self.trigger_monitor.start_monitoring()
        
        # Hook into state manager events (simplified approach)
        # In a full implementation, this would use proper event subscription
        logger.info("Started N-Script state integration")
    
    def stop_integration(self) -> None:
        """Stop integration."""
        self.monitoring = False
        self.trigger_monitor.stop_monitoring()
        logger.info("Stopped N-Script state integration")
    
    def handle_entity_update(self, entity_id: str, changes: Dict[str, Any]) -> None:
        """Handle entity updates from StateManager."""
        if not self.monitoring:
            return
        
        # Check for location changes
        if 'location' in changes:
            new_location = changes['location']
            self.trigger_monitor.check_location_triggers(entity_id, new_location) 