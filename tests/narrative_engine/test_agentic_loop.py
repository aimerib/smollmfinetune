"""
Unit tests for Agentic Loop Framework

Tests the core Agent interface, Action schemas, and Scheduler that orchestrate
the perceive -> think -> act cycle for autonomous NPCs.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from narrative_engine.agent import (
    BaseAgent, Perception, Action, ActionResult,
    MoveToAction, SpeakToAction, TakeItemAction, UpdateGoalAction,
    Scheduler, AgentState
)
from narrative_engine.state_manager import StateManager, EntityState


class TestActionSchemas:
    """Test structured action data classes"""
    
    def test_move_to_action_creation(self):
        """Test MoveToAction schema"""
        action = MoveToAction(target_location="Village Square")
        assert action.target_location == "Village Square"
        assert action.action_type == "move_to"
        assert isinstance(action.timestamp, datetime)
    
    def test_speak_to_action_creation(self):
        """Test SpeakToAction schema"""
        action = SpeakToAction(target_agent_id="npc_clara", message="Good morning!")
        assert action.target_agent_id == "npc_clara"
        assert action.message == "Good morning!"
        assert action.action_type == "speak_to"
    
    def test_take_item_action_creation(self):
        """Test TakeItemAction schema"""
        action = TakeItemAction(item_id="sword_01")
        assert action.item_id == "sword_01"
        assert action.action_type == "take_item"
    
    def test_update_goal_action_creation(self):
        """Test UpdateGoalAction schema"""
        action = UpdateGoalAction(new_goal_description="Find the ancient artifact")
        assert action.new_goal_description == "Find the ancient artifact"
        assert action.action_type == "update_goal"
    
    def test_action_to_dict(self):
        """Test action serialization"""
        action = MoveToAction(target_location="Forest")
        action_dict = action.to_dict()
        
        assert action_dict["action_type"] == "move_to"
        assert action_dict["target_location"] == "Forest"
        assert "timestamp" in action_dict


class TestPerceptionAndResults:
    """Test Perception and ActionResult data structures"""
    
    def test_perception_creation(self):
        """Test Perception contains relevant world state"""
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=["npc_clara", "player_001"],
            recent_events=["npc_clara moved to Village Square"],
            agent_state={"health": 100, "mood": "curious"},
            world_context={"time_of_day": "morning", "weather": "sunny"}
        )
        
        assert perception.agent_id == "npc_tom"
        assert perception.current_location == "Village Square"
        assert "npc_clara" in perception.nearby_agents
        assert len(perception.recent_events) == 1
        assert perception.agent_state["mood"] == "curious"
    
    def test_action_result_creation(self):
        """Test ActionResult captures execution outcome"""
        result = ActionResult(
            success=True,
            message="Successfully moved to Forest",
            state_changes={"location": "Forest"},
            side_effects=["Discovered a hidden path"]
        )
        
        assert result.success is True
        assert result.message == "Successfully moved to Forest"
        assert result.state_changes["location"] == "Forest"
        assert "Discovered a hidden path" in result.side_effects


class TestBaseAgent:
    """Test the core Agent interface"""
    
    @pytest.fixture
    def mock_state_manager(self):
        """Mock StateManager for testing"""
        state_manager = Mock(spec=StateManager)
        
        # Mock entity states
        tom_entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="Village Square",
            custom_data={"mood": "curious", "health": 100}
        )
        
        clara_entity = EntityState(
            entity_id="npc_clara",
            entity_type="character", 
            location="Village Square",
            custom_data={"mood": "friendly", "health": 100}
        )
        
        state_manager.get_entity.side_effect = lambda entity_id: {
            "npc_tom": tom_entity,
            "npc_clara": clara_entity
        }.get(entity_id)
        
        state_manager.query_by_location.return_value = [tom_entity, clara_entity]
        state_manager.get_recent_events.return_value = []
        
        return state_manager
    
    @pytest.fixture
    def mock_narrative_model(self):
        """Mock NarrativeLLM for testing"""
        model = AsyncMock()
        
        # Mock model response for thinking
        model.generate_with_control.return_value = {
            'generated_text': 'I should greet Clara and ask about her day.',
            'control_tokens': ['<friendly>', '<curious>'],
            'emotional_state': {'friendly': 0.8, 'curious': 0.6}
        }
        
        return model
    
    def test_agent_state_initialization(self):
        """Test AgentState data structure"""
        agent_state = AgentState(
            agent_id="npc_tom",
            goals=["Explore the village", "Make friends"],
            personality_traits={"openness": 0.8, "extraversion": 0.7},
            current_plan="Greet nearby NPCs",
            active_since=datetime.now() - timedelta(hours=2)
        )
        
        assert agent_state.agent_id == "npc_tom"
        assert len(agent_state.goals) == 2
        assert agent_state.personality_traits["openness"] == 0.8
        assert agent_state.current_plan == "Greet nearby NPCs"
        assert agent_state.active_since < datetime.now()
    
    
    async def test_base_agent_perceive(self, mock_state_manager):
        """Test agent perception gathering"""
        agent = BaseAgent(
            agent_id="npc_tom",
            narrative_model=None,  # Not needed for perception test
            goals=["Explore the village"]
        )
        
        perception = await agent.perceive(mock_state_manager)
        
        assert perception.agent_id == "npc_tom"
        assert perception.current_location == "Village Square"
        assert "npc_clara" in perception.nearby_agents
        
        # Verify StateManager calls
        mock_state_manager.get_entity.assert_called_with("npc_tom")
        mock_state_manager.query_by_location.assert_called_with("Village Square")
    
    
    async def test_base_agent_think(self, mock_narrative_model):
        """Test agent decision making"""
        agent = BaseAgent(
            agent_id="npc_tom",
            narrative_model=mock_narrative_model,
            goals=["Make friends"]
        )
        
        # Create test perception
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=["npc_clara"],
            recent_events=[],
            agent_state={"mood": "curious"},
            world_context={"time_of_day": "morning"}
        )
        
        action = await agent.think(perception)
        
        assert isinstance(action, Action)
        # The think method should parse the model output into a structured action
        mock_narrative_model.generate_with_control.assert_called_once()
    
    
    async def test_base_agent_act_move_action(self, mock_state_manager):
        """Test agent action execution - movement"""
        agent = BaseAgent(
            agent_id="npc_tom",
            narrative_model=None,
            goals=["Explore the village"]
        )
        
        action = MoveToAction(target_location="Forest")
        result = await agent.act(action, mock_state_manager)
        
        assert isinstance(result, ActionResult)
        assert result.success is True
        
        # Verify state manager was called to update location
        mock_state_manager.update_entity.assert_called()
    
    
    async def test_base_agent_act_speak_action(self, mock_state_manager):
        """Test agent action execution - speaking"""
        agent = BaseAgent(
            agent_id="npc_tom",
            narrative_model=None,
            goals=["Make friends"]
        )
        
        action = SpeakToAction(target_agent_id="npc_clara", message="Hello Clara!")
        result = await agent.act(action, mock_state_manager)
        
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert "clara" in result.message.lower()
        
        # Should log the conversation event
        mock_state_manager.get_recent_events.assert_called()


class TestScheduler:
    """Test the main loop scheduler"""
    
    @pytest.fixture
    def mock_state_manager(self):
        """Mock StateManager with test agents"""
        state_manager = Mock(spec=StateManager)
        
        # Mock active agents
        tom_entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="Village Square",
            custom_data={"active": True, "last_action": datetime.now() - timedelta(minutes=30)}
        )
        
        clara_entity = EntityState(
            entity_id="npc_clara", 
            entity_type="character",
            location="Forest",
            custom_data={"active": True, "last_action": datetime.now() - timedelta(minutes=45)}
        )
        
        state_manager.query.return_value = [tom_entity, clara_entity]
        return state_manager
    
    @pytest.fixture
    def mock_agents(self):
        """Mock agent instances"""
        tom_agent = AsyncMock(spec=BaseAgent)
        tom_agent.agent_id = "npc_tom"
        tom_agent.state = AgentState(agent_id="npc_tom")  # Add mock state
        tom_agent.perceive.return_value = Mock(spec=Perception)
        tom_agent.think.return_value = MoveToAction(target_location="Forest")
        tom_agent.act.return_value = ActionResult(success=True, message="Moved to Forest")
        
        clara_agent = AsyncMock(spec=BaseAgent)
        clara_agent.agent_id = "npc_clara"
        clara_agent.state = AgentState(agent_id="npc_clara")  # Add mock state
        clara_agent.perceive.return_value = Mock(spec=Perception)
        clara_agent.think.return_value = SpeakToAction(target_agent_id="npc_tom", message="Hi Tom!")
        clara_agent.act.return_value = ActionResult(success=True, message="Spoke to Tom")
        
        return {"npc_tom": tom_agent, "npc_clara": clara_agent}
    
    def test_scheduler_initialization(self, mock_state_manager):
        """Test scheduler setup"""
        scheduler = Scheduler(
            state_manager=mock_state_manager,
            tick_rate=10.0,  # 10 seconds per tick
            max_agents_per_tick=2
        )
        
        assert scheduler.state_manager == mock_state_manager
        assert scheduler.tick_rate == 10.0
        assert scheduler.max_agents_per_tick == 2
        assert scheduler.is_running is False
        assert len(scheduler.active_agents) == 0
    
    
    async def test_scheduler_register_agent(self, mock_state_manager, mock_agents):
        """Test agent registration"""
        scheduler = Scheduler(state_manager=mock_state_manager)
        
        tom_agent = mock_agents["npc_tom"]
        scheduler.register_agent(tom_agent)
        
        assert "npc_tom" in scheduler.active_agents
        assert scheduler.active_agents["npc_tom"] == tom_agent
    
    
    async def test_scheduler_single_tick(self, mock_state_manager, mock_agents):
        """Test a single scheduler tick"""
        scheduler = Scheduler(
            state_manager=mock_state_manager,
            tick_rate=1.0,
            max_agents_per_tick=1
        )
        
        # Register agents
        for agent in mock_agents.values():
            scheduler.register_agent(agent)
        
        # Run single tick
        await scheduler.tick()
        
        # Should have processed at least one agent
        assert any(agent.perceive.called for agent in mock_agents.values())
        assert any(agent.think.called for agent in mock_agents.values()) 
        assert any(agent.act.called for agent in mock_agents.values())
    
    
    async def test_scheduler_agent_selection_round_robin(self, mock_state_manager, mock_agents):
        """Test round-robin agent selection"""
        scheduler = Scheduler(
            state_manager=mock_state_manager,
            max_agents_per_tick=1  # Only one agent per tick
        )
        
        for agent in mock_agents.values():
            scheduler.register_agent(agent)
        
        # Run two ticks
        await scheduler.tick()
        first_tick_agent = scheduler._last_processed_agent
        
        await scheduler.tick()
        second_tick_agent = scheduler._last_processed_agent
        
        # Should be different agents (round-robin)
        assert first_tick_agent != second_tick_agent
    
    
    async def test_scheduler_error_handling(self, mock_state_manager, mock_agents):
        """Test scheduler handles agent errors gracefully"""
        scheduler = Scheduler(state_manager=mock_state_manager)
        
        # Make one agent throw an error
        tom_agent = mock_agents["npc_tom"]
        tom_agent.think.side_effect = Exception("AI model timeout")
        
        scheduler.register_agent(tom_agent)
        scheduler.register_agent(mock_agents["npc_clara"])
        
        # Tick should complete despite error
        await scheduler.tick()
        
        # Error should be logged but not crash scheduler
        assert scheduler.is_running is False  # Not started yet, but shouldn't crash
    
     
    async def test_scheduler_run_integration(self, mock_state_manager, mock_agents):
        """Test full scheduler run loop (brief test)"""
        scheduler = Scheduler(
            state_manager=mock_state_manager,
            tick_rate=0.1  # Very fast for testing
        )
        
        for agent in mock_agents.values():
            scheduler.register_agent(agent)
        
        # Start scheduler
        scheduler_task = asyncio.create_task(scheduler.run())
        
        # Let it run briefly
        await asyncio.sleep(0.3)  # Let it run a few ticks
        
        # Stop scheduler
        scheduler.stop()
        await scheduler_task
        
        # Verify agents were processed
        assert any(agent.perceive.called for agent in mock_agents.values())


class TestNarrativeEngineIntegration:
    """Test integration with the R4 Narrative Engine"""
    
    
    async def test_action_parsing_from_model_output(self):
        """Test parsing model text output into structured actions"""
        # Mock model output
        model_output = {
            'generated_text': 'I will move to the Forest to look for herbs.',
            'control_tokens': ['<curious>', '<determined>'],
            'emotional_state': {'curious': 0.7}
        }
        
        agent = BaseAgent(agent_id="npc_herbalist", narrative_model=None, goals=["Gather herbs"])
        
        # Test the action parsing logic
        action = agent._parse_action_from_model_output(model_output)
        
        assert isinstance(action, MoveToAction)
        assert action.target_location == "Forest"
    
    
    async def test_prompt_construction_for_thinking(self):
        """Test that agent constructs proper prompts for the R4 model"""
        mock_model = AsyncMock()
        mock_model.generate_with_control.return_value = {
            'generated_text': 'I should greet the newcomer.',
            'control_tokens': ['<friendly>'],
            'emotional_state': {'friendly': 0.8}
        }
        
        agent = BaseAgent(
            agent_id="npc_greeter",
            narrative_model=mock_model,
            goals=["Welcome new visitors"],
            personality_traits={"extraversion": 0.9, "agreeableness": 0.8}
        )
        
        perception = Perception(
            agent_id="npc_greeter",
            current_location="Village Entrance", 
            nearby_agents=["player_001"],
            recent_events=["player_001 entered Village Entrance"],
            agent_state={"mood": "welcoming"},
            world_context={"time_of_day": "afternoon"}
        )
        
        await agent.think(perception)
        
        # Verify the model was called with a proper prompt
        mock_model.generate_with_control.assert_called_once()
        call_args = mock_model.generate_with_control.call_args
        
        # The prompt should include personality, goals, and perception
        assert "extraversion" in str(call_args) or "personality" in str(call_args)
        assert "Welcome new visitors" in str(call_args)
        assert "Village Entrance" in str(call_args)


# Integration test that brings it all together
class TestAgenticLoopIntegration:
    """Integration test for the complete agentic loop"""
    
    
    async def test_complete_agent_lifecycle(self):
        """Test a complete perceive -> think -> act cycle"""
        # Set up real StateManager with test data
        state_manager = StateManager()
        
        # Create test world state
        tom_entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="Village Square",
            custom_data={"mood": "curious", "goals": ["Explore the village"]}
        )
        
        clara_entity = EntityState(
            entity_id="npc_clara",
            entity_type="character",
            location="Village Square", 
            custom_data={"mood": "friendly"}
        )
        
        state_manager.create_entity(tom_entity)
        state_manager.create_entity(clara_entity)
        
        # Mock narrative model
        mock_model = AsyncMock()
        mock_model.generate_with_control.return_value = {
            'generated_text': 'I will approach Clara and say hello.',
            'control_tokens': ['<friendly>', '<curious>'],
            'emotional_state': {'friendly': 0.7, 'curious': 0.6}
        }
        
        # Create agent
        agent = BaseAgent(
            agent_id="npc_tom",
            narrative_model=mock_model,
            goals=["Make friends in the village"],
            personality_traits={"extraversion": 0.8, "agreeableness": 0.7}
        )
        
        # Execute complete cycle
        perception = await agent.perceive(state_manager)
        action = await agent.think(perception)
        result = await agent.act(action, state_manager)
        
        # Verify the cycle worked
        assert perception.agent_id == "npc_tom"
        assert "npc_clara" in perception.nearby_agents
        assert isinstance(action, Action)
        assert result.success is True
        
        # Verify state was updated
        updated_tom = state_manager.get_entity("npc_tom")
        assert updated_tom is not None 