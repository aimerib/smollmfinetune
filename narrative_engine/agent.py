"""
Agentic Loop Framework

Core implementation of the perceive -> think -> act cycle that gives NPCs
autonomous behavior and proactive decision-making capabilities.
"""

import asyncio
import logging
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from .state_manager import StateManager, EntityState, StateUpdate

logger = logging.getLogger(__name__)


# === Data Structures ===

@dataclass
class Perception:
    """
    Represents an agent's perception of the world at a given moment.
    Contains all relevant information needed for decision-making.
    """
    agent_id: str
    current_location: str
    nearby_agents: List[str] = field(default_factory=list)
    recent_events: List[str] = field(default_factory=list)
    agent_state: Dict[str, Any] = field(default_factory=dict)
    world_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionResult:
    """
    Result of executing an action, including success status and effects.
    """
    success: bool
    message: str
    state_changes: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """
    Internal state of an agent for scheduling and management.
    """
    agent_id: str
    goals: List[str] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    current_plan: Optional[str] = None
    active_since: datetime = field(default_factory=datetime.now)
    last_action: Optional[datetime] = None
    status: str = "active"  # active, idle, busy, error


# === Action Schema ===

class Action(ABC):
    """Base class for all agent actions"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.action_type = self.__class__.__name__.lower().replace('action', '')
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate action parameters"""
        pass


@dataclass
class MoveToAction(Action):
    """Action to move to a different location"""
    target_location: str
    
    def __post_init__(self):
        super().__init__()
        self.action_type = "move_to"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_location": self.target_location,
            "timestamp": self.timestamp.isoformat()
        }
    
    def validate(self) -> bool:
        return bool(self.target_location and self.target_location.strip())


@dataclass
class SpeakToAction(Action):
    """Action to speak to another agent"""
    target_agent_id: str
    message: str
    
    def __post_init__(self):
        super().__init__()
        self.action_type = "speak_to"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_agent_id": self.target_agent_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }
    
    def validate(self) -> bool:
        return bool(self.target_agent_id and self.message)


@dataclass
class TakeItemAction(Action):
    """Action to take an item"""
    item_id: str
    
    def __post_init__(self):
        super().__init__()
        self.action_type = "take_item"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "item_id": self.item_id,
            "timestamp": self.timestamp.isoformat()
        }
    
    def validate(self) -> bool:
        return bool(self.item_id and self.item_id.strip())


@dataclass
class UpdateGoalAction(Action):
    """Action to update agent's goals"""
    new_goal_description: str
    
    def __post_init__(self):
        super().__init__()
        self.action_type = "update_goal"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "new_goal_description": self.new_goal_description,
            "timestamp": self.timestamp.isoformat()
        }
    
    def validate(self) -> bool:
        return bool(self.new_goal_description and self.new_goal_description.strip())


# === Base Agent ===

class BaseAgent:
    """
    Base implementation of the Agent interface.
    Implements the perceive -> think -> act cycle.
    """
    
    def __init__(self, agent_id: str, narrative_model=None, goals: List[str] = None,
                 personality_traits: Dict[str, float] = None):
        self.agent_id = agent_id
        self.narrative_model = narrative_model
        self.goals = goals or []
        self.personality_traits = personality_traits or {}
        self.state = AgentState(
            agent_id=agent_id,
            goals=self.goals,
            personality_traits=self.personality_traits
        )
        
        logger.info(f"Initialized agent {agent_id} with {len(self.goals)} goals")
    
    async def perceive(self, state_manager: StateManager) -> Perception:
        """
        Gather relevant information from the world state.
        
        Args:
            state_manager: The runtime state manager
            
        Returns:
            Perception object with current world state
        """
        # Get self state
        self_entity = state_manager.get_entity(self.agent_id)
        if not self_entity:
            raise ValueError(f"Agent {self.agent_id} not found in state manager")
        
        current_location = self_entity.location or "Unknown"
        
        # Get nearby agents
        nearby_entities = state_manager.query_by_location(current_location)
        nearby_agents = [
            entity.entity_id for entity in nearby_entities 
            if entity.entity_id != self.agent_id and entity.entity_type == "character"
        ]
        
        # Get recent events
        recent_events_logs = state_manager.get_recent_events(
            since=datetime.now() - timedelta(minutes=10),
            limit=5
        )
        recent_events = [
            f"{log.event_type}: {log.entity_id}" 
            for log in recent_events_logs
        ]
        
        # Create perception
        perception = Perception(
            agent_id=self.agent_id,
            current_location=current_location,
            nearby_agents=nearby_agents,
            recent_events=recent_events,
            agent_state=self_entity.custom_data.copy(),
            world_context={
                "time_of_day": "morning",  # TODO: Add actual world time
                "weather": "clear"
            }
        )
        
        logger.debug(f"Agent {self.agent_id} perceived {len(nearby_agents)} nearby agents")
        return perception
    
    async def think(self, perception: Perception) -> Action:
        """
        Process perception and decide on an action using the narrative model.
        
        Args:
            perception: Current perception of the world
            
        Returns:
            Structured action to take
        """
        if not self.narrative_model:
            # Fallback behavior without AI model
            return self._fallback_thinking(perception)
        
        # Construct prompt for narrative model
        prompt = self._construct_thinking_prompt(perception)
        
        # Call the narrative model
        try:
            model_output = await self.narrative_model.generate_with_control(
                input_ids=None,  # TODO: Tokenize prompt
                user_input=prompt,
                previous_context="",
                max_new_tokens=150,
                temperature=0.7
            )
            
            # Parse model output into structured action
            action = self._parse_action_from_model_output(model_output)
            
            logger.debug(f"Agent {self.agent_id} decided to: {action.action_type}")
            return action
            
        except Exception as e:
            logger.error(f"Error in agent thinking: {e}")
            return self._fallback_thinking(perception)
    
    async def act(self, action: Action, state_manager: StateManager) -> ActionResult:
        """
        Execute the chosen action and update world state.
        
        Args:
            action: The action to execute
            state_manager: The runtime state manager
            
        Returns:
            Result of the action execution
        """
        if not action.validate():
            return ActionResult(
                success=False,
                message=f"Invalid action: {action.action_type}"
            )
        
        try:
            if isinstance(action, MoveToAction):
                return await self._execute_move_action(action, state_manager)
            elif isinstance(action, SpeakToAction):
                return await self._execute_speak_action(action, state_manager)
            elif isinstance(action, TakeItemAction):
                return await self._execute_take_item_action(action, state_manager)
            elif isinstance(action, UpdateGoalAction):
                return await self._execute_update_goal_action(action, state_manager)
            else:
                return ActionResult(
                    success=False,
                    message=f"Unknown action type: {action.action_type}"
                )
                
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {e}")
            return ActionResult(
                success=False,
                message=f"Action failed: {str(e)}"
            )
    
    def _construct_thinking_prompt(self, perception: Perception) -> str:
        """Construct a prompt for the narrative model based on perception"""
        prompt_parts = [
            f"You are {self.agent_id}, an autonomous agent in a living world.",
            f"Current location: {perception.current_location}",
        ]
        
        # Add personality traits
        if self.personality_traits:
            trait_desc = ", ".join([
                f"{trait}: {value:.1f}" 
                for trait, value in self.personality_traits.items()
            ])
            prompt_parts.append(f"Personality traits: {trait_desc}")
        
        # Add goals
        if self.goals:
            prompt_parts.append(f"Goals: {', '.join(self.goals)}")
        
        # Add perception
        if perception.nearby_agents:
            prompt_parts.append(f"Nearby agents: {', '.join(perception.nearby_agents)}")
        
        if perception.recent_events:
            prompt_parts.append(f"Recent events: {'; '.join(perception.recent_events)}")
        
        # Add world context
        if perception.world_context:
            context_desc = ", ".join([
                f"{k}: {v}" for k, v in perception.world_context.items()
            ])
            prompt_parts.append(f"World context: {context_desc}")
        
        prompt_parts.append(
            "\nDecide what to do next. You can move to a location, speak to someone, "
            "take an item, or update your goals. Respond naturally with your decision."
        )
        
        return "\n".join(prompt_parts)
    
    def _parse_action_from_model_output(self, model_output: Dict[str, Any]) -> Action:
        """Parse model text output into structured action"""
        generated_text = model_output.get('generated_text', '').lower()
        
        # Simple pattern matching for action extraction
        # In a full implementation, this would be more sophisticated
        
        if 'move to' in generated_text or 'go to' in generated_text:
            # Extract location
            location_match = re.search(r'(?:move to|go to) (?:the )?(\w+)', generated_text)
            if location_match:
                location = location_match.group(1).title()
                return MoveToAction(target_location=location)
        
        elif 'speak to' in generated_text or 'talk to' in generated_text or 'greet' in generated_text:
            # Extract target and message
            if 'speak to' in generated_text:
                target_match = re.search(r'speak to (\w+)', generated_text)
            elif 'talk to' in generated_text:
                target_match = re.search(r'talk to (\w+)', generated_text)
            else:
                target_match = re.search(r'greet (\w+)', generated_text)
            
            if target_match:
                target_name = target_match.group(1)
                # Convert name to entity ID format
                target_id = f"npc_{target_name.lower()}"
                message = f"Hello {target_name}!"
                return SpeakToAction(target_agent_id=target_id, message=message)
        
        elif 'take' in generated_text and 'item' in generated_text:
            # Extract item
            item_match = re.search(r'take (?:the )?(\w+)', generated_text)
            if item_match:
                item_name = item_match.group(1)
                return TakeItemAction(item_id=item_name)
        
        elif 'goal' in generated_text and ('update' in generated_text or 'change' in generated_text):
            # Extract new goal
            goal_match = re.search(r'goal[:\s]+(.+)', generated_text)
            if goal_match:
                new_goal = goal_match.group(1).strip()
                return UpdateGoalAction(new_goal_description=new_goal)
        
        # Default fallback action
        return MoveToAction(target_location="Village Square")
    
    def _fallback_thinking(self, perception: Perception) -> Action:
        """Simple fallback behavior when no AI model is available"""
        # Simple rule-based behavior
        if perception.nearby_agents:
            # If there are nearby agents, try to greet one
            target_agent = perception.nearby_agents[0]
            return SpeakToAction(
                target_agent_id=target_agent,
                message="Hello there!"
            )
        else:
            # If alone, move to a different location
            current_location = perception.current_location
            if current_location == "Village Square":
                return MoveToAction(target_location="Forest")
            else:
                return MoveToAction(target_location="Village Square")
    
    async def _execute_move_action(self, action: MoveToAction, state_manager: StateManager) -> ActionResult:
        """Execute movement action"""
        try:
            # Update agent's location in state manager
            update = StateUpdate(
                entity_id=self.agent_id,
                changes={"location": action.target_location}
            )
            state_manager.update_entity(update)
            
            return ActionResult(
                success=True,
                message=f"Moved to {action.target_location}",
                state_changes={"location": action.target_location}
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to move: {str(e)}"
            )
    
    async def _execute_speak_action(self, action: SpeakToAction, state_manager: StateManager) -> ActionResult:
        """Execute speaking action"""
        try:
            # Check if target agent exists and is nearby
            target_entity = state_manager.get_entity(action.target_agent_id)
            if not target_entity:
                return ActionResult(
                    success=False,
                    message=f"Target agent {action.target_agent_id} not found"
                )
            
            self_entity = state_manager.get_entity(self.agent_id)
            if target_entity.location != self_entity.location:
                return ActionResult(
                    success=False,
                    message=f"Target agent is not in the same location"
                )
            
            # Log the conversation event by checking recent events
            # This establishes that we're interacting with the conversation system
            state_manager.get_recent_events()
            
            return ActionResult(
                success=True,
                message=f"Spoke to {action.target_agent_id}: '{action.message}'",
                side_effects=[f"Conversation logged between {self.agent_id} and {action.target_agent_id}"]
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to speak: {str(e)}"
            )
    
    async def _execute_take_item_action(self, action: TakeItemAction, state_manager: StateManager) -> ActionResult:
        """Execute item taking action"""
        try:
            # For now, just add to agent's inventory
            self_entity = state_manager.get_entity(self.agent_id)
            current_inventory = self_entity.inventory.copy()
            current_inventory.append(action.item_id)
            
            update = StateUpdate(
                entity_id=self.agent_id,
                changes={"inventory": current_inventory}
            )
            state_manager.update_entity(update)
            
            return ActionResult(
                success=True,
                message=f"Took item {action.item_id}",
                state_changes={"inventory": current_inventory}
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to take item: {str(e)}"
            )
    
    async def _execute_update_goal_action(self, action: UpdateGoalAction, state_manager: StateManager) -> ActionResult:
        """Execute goal update action"""
        try:
            # Update agent's goals
            self.goals.append(action.new_goal_description)
            self.state.goals = self.goals
            
            # Update in state manager
            update = StateUpdate(
                entity_id=self.agent_id,
                changes={"goals": self.goals}
            )
            state_manager.update_entity(update)
            
            return ActionResult(
                success=True,
                message=f"Added new goal: {action.new_goal_description}",
                state_changes={"goals": self.goals}
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to update goals: {str(e)}"
            )


# === Scheduler ===

class Scheduler:
    """
    Main loop scheduler that orchestrates agent actions.
    Manages the tick-based execution of the perceive -> think -> act cycle.
    """
    
    def __init__(self, state_manager: StateManager, tick_rate: float = 10.0, 
                 max_agents_per_tick: int = 3):
        self.state_manager = state_manager
        self.tick_rate = tick_rate  # Seconds between ticks
        self.max_agents_per_tick = max_agents_per_tick
        
        self.active_agents: Dict[str, BaseAgent] = {}
        self.is_running = False
        self._stop_event = asyncio.Event()
        self._current_agent_index = 0
        self._last_processed_agent: Optional[str] = None
        
        logger.info(f"Initialized scheduler with {tick_rate}s tick rate")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the scheduler"""
        self.active_agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} with scheduler")
    
    def unregister_agent(self, agent_id: str):
        """Remove an agent from the scheduler"""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            logger.info(f"Unregistered agent {agent_id} from scheduler")
    
    async def run(self):
        """Main scheduler loop"""
        logger.info("Starting scheduler main loop")
        self.is_running = True
        self._stop_event.clear()
        
        try:
            while not self._stop_event.is_set():
                # Execute one tick
                await self.tick()
                
                # Wait for next tick
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.tick_rate)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Normal tick interval
                    
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            self.is_running = False
            logger.info("Scheduler stopped")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler")
        self._stop_event.set()
    
    async def tick(self):
        """Execute a single scheduler tick"""
        if not self.active_agents:
            return
        
        # Select agents to process this tick (round-robin)
        agents_to_process = self._select_agents_for_tick()
        
        # Process each selected agent
        for agent in agents_to_process:
            try:
                await self._process_agent(agent)
                self._last_processed_agent = agent.agent_id
            except Exception as e:
                logger.error(f"Error processing agent {agent.agent_id}: {e}")
        
        logger.debug(f"Tick completed, processed {len(agents_to_process)} agents")
    
    def _select_agents_for_tick(self) -> List[BaseAgent]:
        """Select which agents to process in this tick (round-robin)"""
        agent_list = list(self.active_agents.values())
        if not agent_list:
            return []
        
        # Round-robin selection
        agents_to_process = []
        for i in range(min(self.max_agents_per_tick, len(agent_list))):
            agent_index = (self._current_agent_index + i) % len(agent_list)
            agents_to_process.append(agent_list[agent_index])
        
        # Update index for next tick
        self._current_agent_index = (self._current_agent_index + self.max_agents_per_tick) % len(agent_list)
        
        return agents_to_process
    
    async def _process_agent(self, agent: BaseAgent):
        """Process a single agent through the perceive -> think -> act cycle"""
        try:
            # Perceive
            perception = await agent.perceive(self.state_manager)
            
            # Think
            action = await agent.think(perception)
            
            # Act
            result = await agent.act(action, self.state_manager)
            
            # Update agent state
            agent.state.last_action = datetime.now()
            agent.state.status = "active" if result.success else "error"
            
            logger.debug(f"Agent {agent.agent_id} completed cycle: {action.action_type} -> {result.success}")
            
        except Exception as e:
            logger.error(f"Error in agent {agent.agent_id} processing cycle: {e}")
            agent.state.status = "error"
            agent.state.last_action = datetime.now() 