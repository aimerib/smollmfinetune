"""
Unit tests for ProactiveAgent Core

Tests the enhanced agent implementation that uses the triple-head architecture
for generation, emotional control, and memory formation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from narrative_engine.agent import (
    ProactiveAgent, BaseAgent, Perception, Action, ActionResult,
    MoveToAction, SpeakToAction, TakeItemAction, UpdateGoalAction,
    AgentState
)
from narrative_engine.state_manager import StateManager, EntityState, StateUpdate
from narrative_engine.types import ThinkResult


@pytest.fixture
def mock_triple_head_model():
    """Mock NarrativeLLM with triple-head outputs"""
    model = AsyncMock()
    
    # Mock comprehensive triple-head response
    model.generate_with_control.return_value = {
        'generated_text': 'I should approach Clara to discuss the village situation.',
        'control_tokens': ['<thoughtful>', '<concerned>', '<social>'],
        'emotional_state': {'thoughtful': 0.8, 'concerned': 0.6, 'social': 0.7},
        'next_recirculation': ['<thoughtful>', '<concerned>'],
        'surprise_score': 0.3,
        'generated_memory': {
            'embedding': [0.1] * 768,  # 768-dim vector
            'importance': 0.7,
            'surprise': 0.3,
            'valence': 0.2,  # Slightly positive
            'persistence': 0.8
        }
    }
    
    return model

@pytest.fixture
def mock_state_manager_with_memory():
    """Mock StateManager with memory and emotional state support"""
    state_manager = Mock(spec=StateManager)
    
    # Mock entity with memory and emotional state
    tom_entity = EntityState(
        entity_id="npc_tom",
        entity_type="character",
        location="Village Square",
        custom_data={
            "mood": "curious", 
            "health": 100,
            "emotional_state": {"friendly": 0.5, "curious": 0.8},
            "emotional_momentum": ["<curious>", "<friendly>"]
        },
        memories=[
            {
                "content": "Met Clara yesterday near the fountain",
                "importance": 0.6,
                "timestamp": "2025-01-15T10:30:00",
                "emotional_valence": 0.7
            }
        ]
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
    state_manager.get_character_memories.return_value = tom_entity.memories
    state_manager.add_memory_to_character = Mock()
    state_manager.add_subtext = Mock(return_value="subtext_123")
    
    return state_manager


class TestProactiveAgentCore:
    """Test the ProactiveAgent class with triple-head integration"""
    
    def test_proactive_agent_initialization(self):
        """Test ProactiveAgent initialization with character data"""
        character_data = {
            "personality": {"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.7},
            "goals": ["Protect the village", "Find the ancient artifact"],
            "memories": [],
            "emotional_state": {}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=None
        )
        
        assert agent.agent_id == "npc_tom"
        assert agent.personality_traits["openness"] == 0.8
        assert len(agent.goals) == 2
        assert "Protect the village" in agent.goals
        assert agent.emotional_state == {}
        assert agent.memory_context == []
    
    async def test_proactive_agent_enhanced_perceive(self, mock_state_manager_with_memory):
        """Test enhanced perception that includes memory and emotional context"""
        character_data = {
            "personality": {"openness": 0.8, "extraversion": 0.7},
            "goals": ["Make friends in the village"],
            "memories": [],
            "emotional_state": {"curious": 0.6}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=None
        )
        
        perception = await agent.perceive(mock_state_manager_with_memory)
        
        # Standard perception elements
        assert perception.agent_id == "npc_tom"
        assert perception.current_location == "Village Square"
        assert "npc_clara" in perception.nearby_agents
        
        # Enhanced perception elements
        assert "memory_context" in perception.agent_state
        assert "emotional_state" in perception.agent_state
        assert "emotional_momentum" in perception.agent_state
        
        # Verify memory retrieval was called
        mock_state_manager_with_memory.get_character_memories.assert_called_with("npc_tom", limit=5)
    
    async def test_proactive_agent_triple_head_thinking(self, mock_triple_head_model, mock_state_manager_with_memory):
        """Test thinking with triple-head architecture processing"""
        character_data = {
            "personality": {"openness": 0.8, "conscientiousness": 0.6},
            "goals": ["Investigate strange occurrences"],
            "memories": [],
            "emotional_state": {"curious": 0.7}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        # Create enhanced perception
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=["npc_clara"],
            recent_events=["Strange sounds heard near the forest"],
            agent_state={
                "memory_context": ["Met Clara yesterday"],
                "emotional_state": {"curious": 0.7},
                "emotional_momentum": ["<curious>"]
            }
        )
        
        result = await agent.think(perception)
        
        # Verify enhanced ThinkResult
        assert isinstance(result, ThinkResult)
        assert hasattr(result, 'emotional_state')
        assert hasattr(result, 'memory_formation')
        assert hasattr(result, 'subtext')
        
        # Verify emotional state processing
        assert result.emotional_state['thoughtful'] == 0.8
        assert result.emotional_state['concerned'] == 0.6
        
        # Verify memory formation
        assert result.memory_formation is not None
        assert result.memory_formation['importance'] == 0.7
        assert result.memory_formation['surprise'] == 0.3
        
        # Verify recirculation tokens for next turn
        assert hasattr(result, 'next_recirculation')
        assert '<thoughtful>' in result.next_recirculation
        
        # Verify triple-head model was called correctly
        mock_triple_head_model.generate_with_control.assert_called_once()
        call_kwargs = mock_triple_head_model.generate_with_control.call_args.kwargs
        assert 'recirculation_tokens' in call_kwargs
        assert 'generate_memory' in call_kwargs
        assert call_kwargs['generate_memory'] is True
    
    async def test_proactive_agent_enhanced_acting(self, mock_state_manager_with_memory):
        """Test action execution with memory formation and emotional state updates"""
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Make friends"],
            "memories": [],
            "emotional_state": {"friendly": 0.6}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=None
        )
        
        # Create enhanced ThinkResult with memory formation
        think_result = ThinkResult(
            action=SpeakToAction(target_agent_id="npc_clara", message="Hello Clara!"),
            subtext="I wonder if Clara noticed the strange sounds too",
            emotional_state={"friendly": 0.8, "curious": 0.7},
            memory_formation={
                'content': 'Approached Clara about village concerns',
                'importance': 0.6,
                'surprise': 0.2,
                'valence': 0.5,
                'persistence': 0.7
            },
            next_recirculation=["<friendly>", "<curious>"]
        )
        
        result = await agent.act(think_result, mock_state_manager_with_memory)
        
        # Verify basic action execution
        assert result.success is True
        
        # Verify memory formation was processed
        mock_state_manager_with_memory.add_memory_to_character.assert_called_once()
        memory_call = mock_state_manager_with_memory.add_memory_to_character.call_args
        assert memory_call[0][0] == "npc_tom"  # character_id
        stored_memory = memory_call[0][1]  # memory dict
        assert stored_memory['content'] == 'Approached Clara about village concerns'
        assert stored_memory['importance'] == 0.6
        
        # Verify emotional state was updated
        assert agent.emotional_state['friendly'] == 0.8
        assert agent.emotional_state['curious'] == 0.7
        
        # Verify emotional momentum for next turn
        assert agent.emotional_momentum == ["<friendly>", "<curious>"]
        
        # Verify subtext was logged
        mock_state_manager_with_memory.add_subtext.assert_called_with(
            "npc_tom", "I wonder if Clara noticed the strange sounds too"
        )
    
    async def test_memory_context_integration(self, mock_state_manager_with_memory):
        """Test that memory context is properly integrated into perception"""
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Investigate mysteries"],
            "memories": [],
            "emotional_state": {}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=None
        )
        
        perception = await agent.perceive(mock_state_manager_with_memory)
        
        # Verify memory context is included
        memory_context = perception.agent_state.get("memory_context", [])
        assert len(memory_context) > 0
        assert "Met Clara yesterday near the fountain" in str(memory_context)
        
        # Verify emotional state is included
        emotional_state = perception.agent_state.get("emotional_state", {})
        assert "friendly" in emotional_state
        assert "curious" in emotional_state
    
    async def test_emotional_persistence_across_turns(self, mock_triple_head_model, mock_state_manager_with_memory):
        """Test that emotional state persists and influences future thinking"""
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Stay alert"],
            "memories": [],
            "emotional_state": {"concerned": 0.6}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        # Set emotional momentum from previous turn
        agent.emotional_momentum = ["<concerned>", "<alert>"]
        
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=[],
            recent_events=["All quiet"],
            agent_state={}
        )
        
        result = await agent.think(perception)
        
        # Verify emotional momentum was passed to the model
        call_kwargs = mock_triple_head_model.generate_with_control.call_args.kwargs
        assert 'recirculation_tokens' in call_kwargs
        assert "<concerned>" in call_kwargs['recirculation_tokens']
        assert "<alert>" in call_kwargs['recirculation_tokens']


class TestTripleHeadIntegration:
    """Test integration with the triple-head NarrativeLLM architecture"""
    
    async def test_generation_head_processing(self, mock_triple_head_model):
        """Test that generation head output is properly processed"""
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Explore"],
            "memories": [],
            "emotional_state": {}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=["npc_clara"]
        )
        
        result = await agent.think(perception)
        
        # Verify action was parsed from generation head
        assert isinstance(result.action, Action)
        assert result.action.action_type in ['speak_to', 'move_to', 'take_item', 'update_goal']
    
    async def test_control_head_processing(self, mock_triple_head_model):
        """Test that control head output provides emotional state"""
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Make connections"],
            "memories": [],
            "emotional_state": {}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=["npc_clara"]
        )
        
        result = await agent.think(perception)
        
        # Verify emotional state from control head
        assert result.emotional_state is not None
        assert 'thoughtful' in result.emotional_state
        assert 'concerned' in result.emotional_state
        assert 'social' in result.emotional_state
        
        # Verify recirculation tokens for emotional persistence
        assert result.next_recirculation is not None
        assert len(result.next_recirculation) > 0
    
    async def test_memory_head_processing(self, mock_triple_head_model):
        """Test that memory head output creates proper memory formation"""
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Learn about the world"],
            "memories": [],
            "emotional_state": {}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        perception = Perception(
            agent_id="npc_tom",
            current_location="Village Square",
            nearby_agents=["npc_clara"],
            recent_events=["Clara mentioned a strange artifact"]
        )
        
        result = await agent.think(perception)
        
        # Verify memory formation from memory head
        assert result.memory_formation is not None
        assert 'embedding' in result.memory_formation
        assert 'importance' in result.memory_formation
        assert 'surprise' in result.memory_formation
        assert 'valence' in result.memory_formation
        assert 'persistence' in result.memory_formation
        
        # Verify embedding dimension
        assert len(result.memory_formation['embedding']) == 768
        
        # Verify metadata ranges
        assert 0 <= result.memory_formation['importance'] <= 1
        assert 0 <= result.memory_formation['surprise'] <= 1
        assert -1 <= result.memory_formation['valence'] <= 1
        assert 0 <= result.memory_formation['persistence'] <= 1


class TestProactiveAgentIntegration:
    """Integration tests for the complete ProactiveAgent lifecycle"""
    
    async def test_complete_enhanced_lifecycle(self, mock_triple_head_model):
        """Test a complete perceive -> think -> act cycle with all enhancements"""
        # Set up real StateManager
        state_manager = StateManager()
        
        # Create test entities
        tom_entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="Village Square",
            custom_data={
                "mood": "curious",
                "emotional_state": {"curious": 0.6}
            },
            memories=[
                {
                    "content": "Strange sounds heard from the forest",
                    "importance": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        )
        
        clara_entity = EntityState(
            entity_id="npc_clara",
            entity_type="character",
            location="Village Square",
            custom_data={"mood": "friendly"}
        )
        
        state_manager.create_entity(tom_entity)
        state_manager.create_entity(clara_entity)
        
        # Create ProactiveAgent
        character_data = {
            "personality": {"openness": 0.8, "conscientiousness": 0.7},
            "goals": ["Investigate mysteries", "Protect villagers"],
            "memories": [],
            "emotional_state": {"curious": 0.6, "protective": 0.5}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        # Execute complete cycle
        perception = await agent.perceive(state_manager)
        think_result = await agent.think(perception)
        action_result = await agent.act(think_result, state_manager)
        
        # Verify enhanced perception
        assert "memory_context" in perception.agent_state
        assert "emotional_state" in perception.agent_state
        
        # Verify enhanced thinking
        assert think_result.emotional_state is not None
        assert think_result.memory_formation is not None
        assert think_result.next_recirculation is not None
        
        # Verify enhanced acting
        assert action_result.success is True
        
        # Verify state was updated with memories and emotional state
        updated_tom = state_manager.get_entity("npc_tom")
        assert len(updated_tom.memories) > len(tom_entity.memories)  # New memory added
        
        # Verify subtext was logged
        subtext_entries = state_manager.get_subtext(agent_id="npc_tom")
        assert len(subtext_entries) > 0
    
    async def test_memory_formation_and_retrieval_cycle(self, mock_triple_head_model):
        """Test that formed memories are retrieved in future perceptions"""
        state_manager = StateManager()
        
        # Create initial entity
        tom_entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="Village Square",
            custom_data={"mood": "curious"},
            memories=[]
        )
        
        state_manager.create_entity(tom_entity)
        
        character_data = {
            "personality": {"openness": 0.8},
            "goals": ["Learn about the world"],
            "memories": [],
            "emotional_state": {}
        }
        
        agent = ProactiveAgent(
            agent_id="npc_tom",
            character_data=character_data,
            narrative_model=mock_triple_head_model
        )
        
        # First cycle - form a memory
        perception1 = await agent.perceive(state_manager)
        think_result1 = await agent.think(perception1)
        await agent.act(think_result1, state_manager)
        
        # Second cycle - memory should be available in perception
        perception2 = await agent.perceive(state_manager)
        
        # Verify memory context includes the formed memory
        memory_context = perception2.agent_state.get("memory_context", [])
        assert len(memory_context) > 0
        # The memory formed in first cycle should be available
        updated_tom = state_manager.get_entity("npc_tom")
        assert len(updated_tom.memories) > 0 