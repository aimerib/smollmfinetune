"""
Tests for RelationshipManager (Digital Ecology Engine)

Tests the enhanced relationship system that uses triple-head architecture
to create dynamic social interactions and relationship evolution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from narrative_engine.ecology import RelationshipManager, EnhancedRelationship
from narrative_engine.state_manager import StateManager, EntityState, StateUpdate, EventLog
from narrative_engine.agent import Action, SpeakToAction, ProactiveAgent


@pytest.fixture
def mock_triple_head_model():
    """Mock NarrativeLLM with comprehensive triple-head relationship analysis"""
    model = AsyncMock()
    
    # Mock triple-head relationship analysis response
    model.analyze_interaction.return_value = {
        'generation_analysis': {
            'semantic_content': 'Critical feedback delivered constructively',
            'social_intent': 'helping_improve',
            'affinity_change': -0.1,
            'status_change': 'respected_colleague'
        },
        'control_analysis': {
            'emotional_tokens': ['<mood_concerned>', '<tone_helpful>', '<relationship_professional>'],
            'emotional_state': {'concerned': 0.6, 'helpful': 0.8},
            'mood_indicators': ['thoughtful', 'constructive']
        },
        'memory_analysis': {
            'significance_score': 0.7,
            'importance': 0.6,
            'emotional_impact': 0.5,
            'memory_formation': {
                'content': 'Tom gave me thoughtful feedback about my project approach',
                'tags': ['feedback', 'professional_growth'],
                'persistence': 0.7
            }
        }
    }
    
    return model


@pytest.fixture
def state_manager_with_agents():
    """StateManager with test agents for relationship testing"""
    state_manager = StateManager()
    
    # Create Tom entity
    tom_entity = EntityState(
        entity_id="npc_tom",
        entity_type="character",
        location="Village Square",
        custom_data={
            "personality": {"openness": 0.8, "conscientiousness": 0.7, "agreeableness": 0.3}
        },
        relationships={},
        memories=[]
    )
    
    # Create Clara entity  
    clara_entity = EntityState(
        entity_id="npc_clara",
        entity_type="character",
        location="Village Square", 
        custom_data={
            "personality": {"openness": 0.6, "conscientiousness": 0.5, "agreeableness": 0.8}
        },
        relationships={},
        memories=[]
    )
    
    state_manager.create_entity(tom_entity)
    state_manager.create_entity(clara_entity)
    
    return state_manager


class TestEnhancedRelationshipSchema:
    """Test the enhanced relationship data model"""
    
    def test_enhanced_relationship_creation(self):
        """Test creating enhanced relationship with emotional history and memory significance"""
        relationship = EnhancedRelationship(
            affinity=0.75,
            status="Friendly",
            emotional_history=["happy", "curious", "fond"],
            memory_significance=0.8,
            last_interaction=datetime.now(),
            interaction_count=15
        )
        
        assert relationship.affinity == 0.75
        assert relationship.status == "Friendly"
        assert "curious" in relationship.emotional_history
        assert relationship.memory_significance == 0.8
        assert relationship.interaction_count == 15
    
    def test_relationship_serialization(self):
        """Test that enhanced relationships can be serialized to/from dict"""
        relationship = EnhancedRelationship(
            affinity=0.5,
            status="Neutral", 
            emotional_history=["nervous", "hopeful"],
            memory_significance=0.6,
            last_interaction=datetime(2024, 1, 15, 10, 30),
            interaction_count=3
        )
        
        # Convert to dict
        data = relationship.to_dict()
        assert data["affinity"] == 0.5
        assert data["emotional_history"] == ["nervous", "hopeful"]
        
        # Recreate from dict
        relationship2 = EnhancedRelationship.from_dict(data)
        assert relationship2.affinity == relationship.affinity
        assert relationship2.emotional_history == relationship.emotional_history


class TestTripleHeadInteractionAnalysis:
    """Test triple-head analysis of agent interactions"""
    
    async def test_triple_head_analysis_prompt_generation(self, mock_triple_head_model):
        """Test that RelationshipManager generates correct analysis prompts for all three heads"""
        relationship_manager = RelationshipManager(
            state_manager=Mock(),
            narrative_model=mock_triple_head_model
        )
        
        # Mock interaction event
        interaction_event = {
            'speaker_id': 'npc_tom',
            'target_id': 'npc_clara',
            'message': 'That approach won\'t work. You need to consider the risks.',
            'speaker_personality': {'openness': 0.8, 'agreeableness': 0.3},
            'target_personality': {'openness': 0.6, 'agreeableness': 0.8}
        }
        
        # This should fail because RelationshipManager doesn't exist yet
        analysis = await relationship_manager.analyze_interaction_triple_head(interaction_event)
        
        # Verify all three heads were called with proper prompts
        assert analysis['generation_analysis']['semantic_content'] is not None
        assert analysis['control_analysis']['emotional_tokens'] is not None
        assert analysis['memory_analysis']['significance_score'] is not None
    
    async def test_memory_integration_in_analysis(self, mock_triple_head_model):
        """Test that memory head analysis affects relationship persistence"""
        relationship_manager = RelationshipManager(
            state_manager=Mock(),
            narrative_model=mock_triple_head_model
        )
        
        # Mock high-significance interaction
        mock_triple_head_model.analyze_interaction.return_value = {
            'generation_analysis': {'affinity_change': 0.2},
            'control_analysis': {'emotional_state': {'grateful': 0.9}},
            'memory_analysis': {
                'significance_score': 0.95,  # Very high significance
                'importance': 0.9,
                'persistence': 0.9
            }
        }
        
        interaction_event = {
            'speaker_id': 'npc_tom', 
            'target_id': 'npc_clara',
            'message': 'You saved my life!',
            'speaker_personality': {'openness': 0.8},
            'target_personality': {'agreeableness': 0.8}
        }
        
        analysis = await relationship_manager.analyze_interaction_triple_head(interaction_event)
        
        # High memory significance should lead to enhanced relationship persistence
        assert analysis['memory_analysis']['significance_score'] == 0.95
        assert analysis['memory_analysis']['persistence'] == 0.9
    
    async def test_emotional_context_integration(self, mock_triple_head_model):
        """Test that control head emotional states influence relationship dynamics"""
        relationship_manager = RelationshipManager(
            state_manager=Mock(),
            narrative_model=mock_triple_head_model
        )
        
        # Mock emotional interaction
        mock_triple_head_model.analyze_interaction.return_value = {
            'generation_analysis': {'affinity_change': -0.3},
            'control_analysis': {
                'emotional_tokens': ['<mood_angry>', '<tone_harsh>', '<relationship_tension>'],
                'emotional_state': {'angry': 0.8, 'hurt': 0.6},
                'mood_indicators': ['confrontational', 'defensive']
            },
            'memory_analysis': {'significance_score': 0.8}
        }
        
        interaction_event = {
            'speaker_id': 'npc_tom',
            'target_id': 'npc_clara', 
            'message': 'You betrayed my trust!',
            'speaker_personality': {'neuroticism': 0.7},
            'target_personality': {'agreeableness': 0.8}
        }
        
        analysis = await relationship_manager.analyze_interaction_triple_head(interaction_event)
        
        # Verify emotional context is captured
        assert 'angry' in analysis['control_analysis']['emotional_state']
        assert '<relationship_tension>' in analysis['control_analysis']['emotional_tokens']


class TestRelationshipManagerService:
    """Test the RelationshipManager event subscription and state updates"""
    
    async def test_relationship_manager_event_subscription(self, state_manager_with_agents, mock_triple_head_model):
        """Test that RelationshipManager subscribes to StateManager events"""
        relationship_manager = RelationshipManager(
            state_manager=state_manager_with_agents,
            narrative_model=mock_triple_head_model
        )
        
        # This should fail because RelationshipManager doesn't exist yet
        await relationship_manager.start_monitoring()
        
        # Verify event subscription was established
        assert relationship_manager.is_monitoring == True
        assert len(state_manager_with_agents._event_subscribers) > 0
    
    async def test_speak_action_triggers_analysis(self, state_manager_with_agents, mock_triple_head_model):
        """Test that SpeakToAction events trigger relationship analysis"""
        relationship_manager = RelationshipManager(
            state_manager=state_manager_with_agents,
            narrative_model=mock_triple_head_model
        )
        
        await relationship_manager.start_monitoring()
        
        # Simulate a SpeakToAction event
        speak_action = SpeakToAction(
            target_agent_id="npc_clara",
            message="That was a foolish move."
        )
        
        # Log the action in StateManager (should trigger relationship analysis)
        state_manager_with_agents.log_event(
            event_type="SpeakToAction",
            entity_id="npc_tom", 
            details={
                "action": speak_action,
                "target": "npc_clara",
                "message": "That was a foolish move."
            }
        )
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        # Verify that relationship analysis was triggered
        mock_triple_head_model.analyze_interaction.assert_called_once()
    
    async def test_relationship_state_update(self, state_manager_with_agents, mock_triple_head_model):
        """Test that relationship analysis results update entity relationships"""
        relationship_manager = RelationshipManager(
            state_manager=state_manager_with_agents,
            narrative_model=mock_triple_head_model
        )
        
        await relationship_manager.start_monitoring()
        
        # Simulate interaction and processing
        interaction_event = {
            'speaker_id': 'npc_tom',
            'target_id': 'npc_clara',
            'message': 'Great job on the project!',
            'speaker_personality': {'agreeableness': 0.7},
            'target_personality': {'agreeableness': 0.8}
        }
        
        # Mock positive analysis result
        mock_triple_head_model.analyze_interaction.return_value = {
            'generation_analysis': {
                'affinity_change': 0.2,
                'status_change': 'respected_colleague'
            },
            'control_analysis': {
                'emotional_state': {'pleased': 0.7, 'grateful': 0.6}
            },
            'memory_analysis': {
                'significance_score': 0.6,
                'memory_formation': {
                    'content': 'Tom complimented my work',
                    'tags': ['positive_feedback', 'professional']
                }
            }
        }
        
        await relationship_manager.process_interaction(interaction_event)
        
        # Verify relationship was updated in StateManager
        tom_entity = state_manager_with_agents.get_entity("npc_tom")
        clara_entity = state_manager_with_agents.get_entity("npc_clara")
        
        # Check that relationships were updated with enhanced data
        assert "npc_clara" in tom_entity.relationships
        tom_clara_relationship = tom_entity.relationships["npc_clara"]
        
        assert tom_clara_relationship["affinity"] > 0  # Should be positive
        assert "pleased" in tom_clara_relationship["emotional_history"]
        assert tom_clara_relationship["memory_significance"] == 0.6


class TestRelationshipEvolution:
    """Test relationship patterns and evolution over multiple interactions"""
    
    async def test_multiple_interaction_relationship_evolution(self, state_manager_with_agents, mock_triple_head_model):
        """Test that relationships evolve correctly over multiple interactions"""
        relationship_manager = RelationshipManager(
            state_manager=state_manager_with_agents,
            narrative_model=mock_triple_head_model
        )
        
        # Mock sequence of interactions with evolving emotional context
        interactions = [
            {
                'speaker_id': 'npc_tom', 'target_id': 'npc_clara',
                'message': 'Hello, nice to meet you',
                'expected_affinity': 0.1, 'expected_emotions': ['polite', 'curious']
            },
            {
                'speaker_id': 'npc_clara', 'target_id': 'npc_tom', 
                'message': 'Thanks for helping me earlier',
                'expected_affinity': 0.3, 'expected_emotions': ['grateful', 'friendly']
            },
            {
                'speaker_id': 'npc_tom', 'target_id': 'npc_clara',
                'message': 'Would you like to work together on this?',
                'expected_affinity': 0.5, 'expected_emotions': ['collaborative', 'trusting']
            }
        ]
        
        for interaction in interactions:
            # Mock analysis based on expected evolution
            mock_triple_head_model.analyze_interaction.return_value = {
                'generation_analysis': {
                    'affinity_change': interaction['expected_affinity'] / len(interactions)
                },
                'control_analysis': {
                    'emotional_state': {emotion: 0.7 for emotion in interaction['expected_emotions']}
                },
                'memory_analysis': {'significance_score': 0.5}
            }
            
            await relationship_manager.process_interaction(interaction)
        
        # Verify relationship evolved properly
        tom_entity = state_manager_with_agents.get_entity("npc_tom")
        relationship = tom_entity.relationships["npc_clara"]
        
        # Should show progression through emotional states
        # Each interaction has 2 emotions, so total should be 3 * 2 = 6
        assert len(relationship["emotional_history"]) == len(interactions) * 2
        assert relationship["interaction_count"] == len(interactions)
        # Affinity calculation: 
        # Interaction 1 (Tom→Clara): 0.1/3 = 0.033
        # Interaction 2 (Clara→Tom, reverse): 0.3/3 * 0.5 = 0.05  
        # Interaction 3 (Tom→Clara): 0.5/3 = 0.166
        # Total: ~0.25
        assert relationship["affinity"] > 0.2  # Should have grown from interactions 