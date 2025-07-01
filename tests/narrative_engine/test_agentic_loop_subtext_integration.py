"""
Integration Tests for Agentic Loop with Subtext Logging

Testing the complete agent cycle with subtext logging integration
to verify the Iceberg Model works end-to-end.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta

from narrative_engine.agent import BaseAgent, Scheduler, Perception
from narrative_engine.state_manager import StateManager, EntityState
from narrative_engine.types import ThinkResult


class TestAgenticLoopSubtextIntegration:
    """Integration tests for subtext logging in the agent loop"""
    
    @pytest.fixture
    def state_manager(self):
        """Create a real state manager for integration testing"""
        return StateManager(backend="memory")
    
    @pytest.fixture
    def mock_narrative_model(self):
        """Create a mock narrative model that returns subtext + action"""
        model = AsyncMock()
        model.generate_with_control.return_value = {
            'generated_text': """
            The character carefully considers their next move.
            
            [SUBTEXT]
            I don't trust this situation. Something feels off about everyone here.
            [/SUBTEXT]
            
            [ACTION]
            I will move to the Forest to think things through.
            [/ACTION]
            """
        }
        return model
    
    @pytest.fixture
    def test_agent(self, mock_narrative_model):
        """Create a test agent with subtext-enabled model"""
        return BaseAgent(
            agent_id="npc_alice",
            narrative_model=mock_narrative_model,
            goals=["Find the truth", "Stay safe"],
            personality_traits={"neuroticism": 0.7, "openness": 0.6}
        )
    
    @pytest.fixture
    def populated_state_manager(self, state_manager):
        """Create a state manager with test entities"""
        # Create the test agent entity
        alice_entity = EntityState(
            entity_id="npc_alice",
            entity_type="character",
            location="Village Square",
            custom_data={"mood": "suspicious", "health": 100}
        )
        
        # Create another character entity
        bob_entity = EntityState(
            entity_id="npc_bob",
            entity_type="character",
            location="Village Square",
            custom_data={"mood": "friendly", "health": 100}
        )
        
        state_manager.create_entity(alice_entity)
        state_manager.create_entity(bob_entity)
        
        return state_manager
    
    
    async def test_agent_think_logs_subtext(self, test_agent, populated_state_manager, mock_narrative_model):
        """Test that agent.think() result contains subtext that can be logged"""
        # Agent perceives the world
        perception = await test_agent.perceive(populated_state_manager)
        
        # Agent thinks and generates subtext + action
        think_result = await test_agent.think(perception)
        
        # Verify ThinkResult structure
        assert isinstance(think_result, ThinkResult)
        assert think_result.subtext == "I don't trust this situation. Something feels off about everyone here."
        assert think_result.action.action_type == "move_to"
        assert think_result.action.target_location == "Forest"
    
    
    async def test_scheduler_logs_subtext_during_processing(self, test_agent, populated_state_manager):
        """Test that the scheduler logs subtext when processing agents"""
        # Create scheduler
        scheduler = Scheduler(
            state_manager=populated_state_manager,
            tick_rate=1.0,
            max_agents_per_tick=1
        )
        
        # Register the agent
        scheduler.register_agent(test_agent)
        
        # Process one agent cycle manually
        await scheduler._process_agent(test_agent)
        
        # Verify subtext was logged to state manager
        subtext_entries = populated_state_manager.get_subtext(agent_id="npc_alice")
        
        assert len(subtext_entries) == 1
        assert subtext_entries[0].agent_id == "npc_alice"
        assert "don't trust this situation" in subtext_entries[0].subtext
        assert isinstance(subtext_entries[0].timestamp, datetime)
    
    
    async def test_scheduler_tick_complete_integration(self, test_agent, populated_state_manager):
        """Test a complete scheduler tick with subtext logging"""
        scheduler = Scheduler(
            state_manager=populated_state_manager,
            tick_rate=1.0,
            max_agents_per_tick=1
        )
        
        scheduler.register_agent(test_agent)
        
        # Execute one tick
        await scheduler.tick()
        
        # Verify state changes
        # 1. Agent should have moved to Forest
        alice_entity = populated_state_manager.get_entity("npc_alice")
        assert alice_entity.location == "Forest"
        
        # 2. Subtext should be logged
        subtext_entries = populated_state_manager.get_subtext(agent_id="npc_alice")
        assert len(subtext_entries) == 1
        assert "Something feels off" in subtext_entries[0].subtext
        
        # 3. Agent state should be updated
        assert test_agent.state.status == "active"
        assert test_agent.state.last_action is not None
    
    
    async def test_multiple_agents_subtext_separation(self, populated_state_manager, mock_narrative_model):
        """Test that subtext is properly separated between different agents"""
        # Create two agents with different models
        alice_model = AsyncMock()
        alice_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            Alice's inner thoughts about the mystery.
            [/SUBTEXT]
            [ACTION]
            I will move to the Forest.
            [/ACTION]
            """
        }
        
        bob_model = AsyncMock()
        bob_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            Bob's different perspective on things.
            [/SUBTEXT]
            [ACTION]
            I will move to the Tavern.
            [/ACTION]
            """
        }
        
        alice = BaseAgent(agent_id="npc_alice", narrative_model=alice_model)
        bob = BaseAgent(agent_id="npc_bob", narrative_model=bob_model)
        
        scheduler = Scheduler(populated_state_manager, tick_rate=1.0, max_agents_per_tick=2)
        scheduler.register_agent(alice)
        scheduler.register_agent(bob)
        
        # Process both agents
        await scheduler.tick()
        
        # Verify separate subtext logging
        alice_subtext = populated_state_manager.get_subtext(agent_id="npc_alice")
        bob_subtext = populated_state_manager.get_subtext(agent_id="npc_bob")
        
        assert len(alice_subtext) == 1
        assert len(bob_subtext) == 1
        
        assert "Alice's inner thoughts" in alice_subtext[0].subtext
        assert "Bob's different perspective" in bob_subtext[0].subtext
        
        # Verify they're distinct entries
        assert alice_subtext[0].agent_id != bob_subtext[0].agent_id
    
    
    async def test_subtext_retrieval_api(self, test_agent, populated_state_manager):
        """Test the subtext retrieval API functionality"""
        scheduler = Scheduler(populated_state_manager, tick_rate=1.0)
        scheduler.register_agent(test_agent)
        
        # Process agent multiple times to create history
        await scheduler.tick()
        await scheduler.tick()
        
        # Test various retrieval methods
        all_subtext = populated_state_manager.get_subtext()
        alice_subtext = populated_state_manager.get_subtext(agent_id="npc_alice")
        recent_subtext = populated_state_manager.get_recent_subtext("npc_alice", limit=1)
        
        assert len(all_subtext) >= 2
        assert len(alice_subtext) >= 2
        assert len(recent_subtext) == 1
        
        # All should be for the same agent
        for entry in alice_subtext:
            assert entry.agent_id == "npc_alice"
        
        # Recent should be the latest
        assert recent_subtext[0].timestamp >= alice_subtext[0].timestamp
    
    
    async def test_subtext_with_model_fallback(self, populated_state_manager):
        """Test subtext logging when model falls back to simple action"""
        # Create agent without narrative model (triggers fallback)
        fallback_agent = BaseAgent(
            agent_id="npc_alice",
            narrative_model=None,  # No model - triggers fallback
            goals=["Test goal"]
        )
        
        scheduler = Scheduler(populated_state_manager, tick_rate=1.0)
        scheduler.register_agent(fallback_agent)
        
        await scheduler.tick()
        
        # Should still log subtext (empty in this case)
        subtext_entries = populated_state_manager.get_subtext(agent_id="npc_alice")
        
        # Fallback should produce empty subtext but still log the entry
        assert len(subtext_entries) == 1
        assert subtext_entries[0].subtext == ""
        assert subtext_entries[0].agent_id == "npc_alice"
    
    
    async def test_subtext_timestamp_filtering(self, test_agent, populated_state_manager):
        """Test filtering subtext by timestamp"""
        scheduler = Scheduler(populated_state_manager, tick_rate=1.0)
        scheduler.register_agent(test_agent)
        
        # Record start time
        start_time = datetime.now()
        
        # Process agent
        await scheduler.tick()
        
        # Test timestamp filtering
        recent_entries = populated_state_manager.get_subtext(
            agent_id="npc_alice",
            since=start_time
        )
        old_entries = populated_state_manager.get_subtext(
            agent_id="npc_alice", 
            since=start_time + timedelta(hours=1)  # Future time
        )
        
        assert len(recent_entries) == 1
        assert len(old_entries) == 0
        assert recent_entries[0].timestamp >= start_time
    
    
    async def test_error_handling_continues_processing(self, populated_state_manager, mock_narrative_model):
        """Test that subtext parsing errors don't break the agent loop"""
        # Create model that returns malformed output
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': "Malformed output without proper tags"
        }
        
        agent = BaseAgent(
            agent_id="npc_alice",
            narrative_model=mock_narrative_model
        )
        
        scheduler = Scheduler(populated_state_manager, tick_rate=1.0)
        scheduler.register_agent(agent)
        
        # Should not raise exception
        await scheduler.tick()
        
        # Agent should still be active despite parsing error
        assert agent.state.status in ["active", "error"]  # Either is acceptable
        
        # Should still log subtext (empty due to parsing failure)
        subtext_entries = populated_state_manager.get_subtext(agent_id="npc_alice")
        assert len(subtext_entries) == 1
        assert subtext_entries[0].subtext == ""  # Fallback to empty 