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
    # R5-9: New relationship-based triggers
    ON_RELATIONSHIP_CHANGE = "ON_RELATIONSHIP_CHANGE"
    ON_EMOTIONAL_PATTERN = "ON_EMOTIONAL_PATTERN"
    ON_MEMORY_SIGNIFICANCE = "ON_MEMORY_SIGNIFICANCE"
    ON_AFFINITY_THRESHOLD = "ON_AFFINITY_THRESHOLD"
    ON_SOCIAL_GROUP_CHANGE = "ON_SOCIAL_GROUP_CHANGE"


class ActionType(str, Enum):
    """Types of actions supported by N-Script."""
    TRIPLE_HEAD_ACTION = "TRIPLE_HEAD_ACTION"
    GENERATION_OVERRIDE = "GENERATION_OVERRIDE"
    CONTROL_INJECTION = "CONTROL_INJECTION"
    MEMORY_FORMATION = "MEMORY_FORMATION"
    HEAD_SYNCHRONIZATION = "HEAD_SYNCHRONIZATION"
    # R5-9: New visual action types for enhanced features
    TRIGGER_CHAIN = "TRIGGER_CHAIN"           # Chain reactions
    PROBABILITY_BRANCH = "PROBABILITY_BRANCH" # Probability-based branching  
    MULTI_CHARACTER_ORCHESTRATION = "MULTI_CHARACTER_ORCHESTRATION" # Conduct multiple characters
    DYNAMIC_VARIABLE_UPDATE = "DYNAMIC_VARIABLE_UPDATE" # Update story variables
    RELATIONSHIP_MODIFY = "RELATIONSHIP_MODIFY" # Direct relationship manipulation


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
    
    def __init__(self, state_manager: StateManager, script_manager: ScriptManager, relationship_manager=None):
        """Initialize TriggerMonitor."""
        self.state_manager = state_manager
        self.script_manager = script_manager
        self.relationship_manager = relationship_manager
        self.activated_scripts: List[Script] = []
        self.triggered_once: set[str] = set()  # Track 'once' triggers
        self.monitoring = False
        
        # R5-9: Initialize relationship query engine
        self.relationship_query_engine = RelationshipQueryEngine(state_manager, relationship_manager)
        self.relationship_condition_evaluator = RelationshipConditionEvaluator(self.relationship_query_engine)
        
        logger.info("Initialized TriggerMonitor with relationship support")
    
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
    
    # R5-9: New relationship-based trigger handlers
    async def handle_relationship_change_event(self, relationship_data: Dict[str, Any]) -> None:
        """Handle relationship change events"""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_RELATIONSHIP_CHANGE")
        
        for script in scripts:
            if await self._check_relationship_change_conditions(script.trigger, relationship_data):
                self.activated_scripts.append(script)
                
                if script.trigger.once:
                    self.triggered_once.add(script.script_id)
                
                logger.info(f"Relationship change trigger activated: {script.script_id}")
    
    async def handle_affinity_threshold_event(self, affinity_data: Dict[str, Any]) -> None:
        """Handle affinity threshold crossing events"""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_AFFINITY_THRESHOLD")
        
        for script in scripts:
            if await self._check_affinity_threshold_conditions(script.trigger, affinity_data):
                self.activated_scripts.append(script)
                
                if script.trigger.once:
                    self.triggered_once.add(script.script_id)
                
                logger.info(f"Affinity threshold trigger activated: {script.script_id}")
    
    async def handle_emotional_pattern_event(self, pattern_data: Dict[str, Any]) -> None:
        """Handle emotional pattern detection events"""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_EMOTIONAL_PATTERN")
        
        for script in scripts:
            if await self._check_emotional_pattern_conditions(script.trigger, pattern_data):
                self.activated_scripts.append(script)
                
                if script.trigger.once:
                    self.triggered_once.add(script.script_id)
                
                logger.info(f"Emotional pattern trigger activated: {script.script_id}")
    
    async def handle_memory_significance_event(self, memory_data: Dict[str, Any]) -> None:
        """Handle memory significance threshold events"""
        if not self.monitoring:
            return
        
        scripts = self.script_manager.get_scripts_by_trigger_type("ON_MEMORY_SIGNIFICANCE")
        
        for script in scripts:
            if await self._check_memory_significance_conditions(script.trigger, memory_data):
                self.activated_scripts.append(script)
                
                if script.trigger.once:
                    self.triggered_once.add(script.script_id)
                
                logger.info(f"Memory significance trigger activated: {script.script_id}")
    
    # R5-9: New relationship condition evaluation methods
    async def _check_relationship_change_conditions(self, trigger: NScriptTrigger, relationship_data: Dict[str, Any]) -> bool:
        """Check if relationship change data meets trigger conditions"""
        if trigger.once and trigger in self.triggered_once:
            return False
        
        conditions = trigger.conditions
        
        # Use relationship condition evaluator
        for condition_key, condition_value in conditions.items():
            if condition_key == "relationship_pair":
                # Check if this trigger applies to these agents
                trigger_pair = condition_value
                event_pair = [relationship_data.get("agent1"), relationship_data.get("agent2")]
                if set(trigger_pair) != set(event_pair):
                    return False
            elif condition_key == "affinity_threshold":
                # Check affinity change magnitude
                affinity_change = relationship_data.get("affinity_change", 0.0)
                if not self._evaluate_condition(abs(affinity_change), condition_value):
                    return False
            else:
                # Use general condition evaluation
                data_value = relationship_data.get(condition_key)
                if not self._evaluate_condition(data_value, condition_value):
                    return False
        
        return True
    
    async def _check_affinity_threshold_conditions(self, trigger: NScriptTrigger, affinity_data: Dict[str, Any]) -> bool:
        """Check if affinity data meets threshold conditions"""
        if trigger.once and trigger in self.triggered_once:
            return False
        
        # Use the relationship condition evaluator
        return await self.relationship_condition_evaluator.evaluate_affinity_condition(
            {**trigger.conditions, **affinity_data}
        )
    
    async def _check_emotional_pattern_conditions(self, trigger: NScriptTrigger, pattern_data: Dict[str, Any]) -> bool:
        """Check if emotional pattern data meets conditions"""
        if trigger.once and trigger in self.triggered_once:
            return False
        
        return await self.relationship_condition_evaluator.evaluate_emotional_pattern(
            {**trigger.conditions, **pattern_data}
        )
    
    async def _check_memory_significance_conditions(self, trigger: NScriptTrigger, memory_data: Dict[str, Any]) -> bool:
        """Check if memory significance data meets conditions"""
        if trigger.once and trigger in self.triggered_once:
            return False
        
        return await self.relationship_condition_evaluator.evaluate_memory_significance(
            {**trigger.conditions, **memory_data}
        )
    
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

@dataclass
class RelationshipQuery:
    """Query structure for relationship conditions"""
    relationship_pair: List[str]  # [agent_1, agent_2]
    condition_type: str  # "affinity_threshold", "emotional_pattern", "memory_significance"
    threshold_value: Optional[float] = None
    threshold_direction: str = "above"  # "above", "below", "equal"
    emotional_sequence: Optional[List[str]] = None
    within_interactions: Optional[int] = None
    memory_count: Optional[int] = None


class RelationshipQueryEngine:
    """Sophisticated relationship query system for N-Script triggers"""
    
    def __init__(self, state_manager: StateManager, relationship_manager=None):
        self.state_manager = state_manager
        self.relationship_manager = relationship_manager
        self._relationship_cache = {}
        logger.info("Initialized RelationshipQueryEngine")
    
    async def query_relationship(self, query: RelationshipQuery) -> bool:
        """Evaluate complex relationship conditions"""
        if not query.relationship_pair or len(query.relationship_pair) != 2:
            return False
            
        agent1, agent2 = query.relationship_pair
        
        # Get relationship data
        relationship_data = await self._get_relationship_data(agent1, agent2)
        if not relationship_data:
            return False
            
        # Evaluate based on condition type
        if query.condition_type == "affinity_threshold":
            return self._evaluate_affinity_threshold(
                relationship_data, query.threshold_value, query.threshold_direction
            )
        elif query.condition_type == "emotional_pattern":
            return await self._evaluate_emotional_pattern(
                relationship_data, query.emotional_sequence, query.within_interactions
            )
        elif query.condition_type == "memory_significance":
            return self._evaluate_memory_significance(
                relationship_data, query.threshold_value, query.memory_count
            )
        
        return False
    
    async def _get_relationship_data(self, agent1: str, agent2: str) -> Optional[Dict[str, Any]]:
        """Get relationship data between two agents"""
        cache_key = f"{min(agent1, agent2)}_{max(agent1, agent2)}"
        
        if cache_key in self._relationship_cache:
            return self._relationship_cache[cache_key]
        
        # Try to get from state manager first
        entity1 = self.state_manager.get_entity(agent1)
        if entity1 and agent2 in entity1.relationships:
            relationship_data = entity1.relationships[agent2]
            self._relationship_cache[cache_key] = relationship_data
            return relationship_data
        
        # If relationship manager is available, try that
        if self.relationship_manager:
            relationship = self.relationship_manager.get_relationship(agent1, agent2)
            if relationship:
                relationship_data = relationship.to_dict()
                self._relationship_cache[cache_key] = relationship_data
                return relationship_data
        
        return None
    
    def _evaluate_affinity_threshold(
        self, 
        relationship_data: Dict[str, Any], 
        threshold: float, 
        direction: str
    ) -> bool:
        """Evaluate affinity threshold condition"""
        affinity = relationship_data.get("affinity", 0.0)
        
        if direction == "above":
            return affinity > threshold
        elif direction == "below":
            return affinity < threshold
        elif direction == "equal":
            return abs(affinity - threshold) < 0.1  # Small tolerance for float comparison
        
        return False
    
    async def _evaluate_emotional_pattern(
        self, 
        relationship_data: Dict[str, Any], 
        emotional_sequence: List[str], 
        within_interactions: int
    ) -> bool:
        """Evaluate emotional pattern condition"""
        if not emotional_sequence:
            return False
            
        emotional_history = relationship_data.get("emotional_history", [])
        if not emotional_history:
            return False
        
        # Check recent emotional history
        recent_emotions = emotional_history[-within_interactions:] if within_interactions else emotional_history
        
        # Check if the sequence appears in recent emotions
        sequence_str = "->".join(emotional_sequence)
        recent_str = "->".join(recent_emotions)
        
        return sequence_str in recent_str
    
    def _evaluate_memory_significance(
        self, 
        relationship_data: Dict[str, Any], 
        significance_threshold: float, 
        memory_count: int
    ) -> bool:
        """Evaluate memory significance condition"""
        memory_significance = relationship_data.get("memory_significance", 0.0)
        interaction_count = relationship_data.get("interaction_count", 0)
        
        # Both conditions must be met
        return (memory_significance >= significance_threshold and 
                interaction_count >= memory_count)
    
    def get_emotional_patterns(self, agent_id: str, pattern_length: int = 3) -> List[str]:
        """Extract recent emotional patterns for an agent"""
        patterns = []
        
        # Get all relationships for this agent
        entity = self.state_manager.get_entity(agent_id)
        if not entity or not entity.relationships:
            return patterns
        
        for relationship_data in entity.relationships.values():
            if isinstance(relationship_data, dict):
                emotional_history = relationship_data.get("emotional_history", [])
                if len(emotional_history) >= pattern_length:
                    # Extract recent pattern
                    recent_pattern = emotional_history[-pattern_length:]
                    patterns.append("->".join(recent_pattern))
        
        return patterns
    
    def calculate_relationship_momentum(self, agent1: str, agent2: str) -> float:
        """Calculate relationship change velocity"""
        # This would analyze the rate of affinity change over recent interactions
        # For now, return a placeholder
        return 0.0
    
    def find_relationship_clusters(self) -> Dict[str, List[str]]:
        """Identify social groups and cliques"""
        # This would analyze the relationship network to find clusters
        # For now, return empty dict
        return {}


class RelationshipConditionEvaluator:
    """Specialized condition evaluators for relationship contexts"""
    
    def __init__(self, query_engine: RelationshipQueryEngine):
        self.query_engine = query_engine
    
    async def evaluate_affinity_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate affinity-based condition"""
        query = RelationshipQuery(
            relationship_pair=condition.get("relationship_pair", []),
            condition_type="affinity_threshold",
            threshold_value=condition.get("affinity_threshold"),
            threshold_direction=condition.get("direction", "above")
        )
        return await self.query_engine.query_relationship(query)
    
    async def evaluate_emotional_pattern(self, condition: Dict[str, Any]) -> bool:
        """Evaluate emotional pattern condition"""
        query = RelationshipQuery(
            relationship_pair=condition.get("relationship_pair", []),
            condition_type="emotional_pattern",
            emotional_sequence=condition.get("emotion_sequence", []),
            within_interactions=condition.get("within_interactions", 5)
        )
        return await self.query_engine.query_relationship(query)
    
    async def evaluate_memory_significance(self, condition: Dict[str, Any]) -> bool:
        """Evaluate memory significance condition"""
        query = RelationshipQuery(
            relationship_pair=condition.get("relationship_pair", []),
            condition_type="memory_significance",
            threshold_value=condition.get("memory_significance_above", 0.8),
            memory_count=condition.get("memory_count", 3)
        )
        return await self.query_engine.query_relationship(query)
    
    async def evaluate_social_dynamics(self, condition: Dict[str, Any]) -> bool:
        """Evaluate social dynamics condition"""
        # Placeholder for more complex social dynamics evaluation
        return True


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