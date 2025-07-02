"""
Tests for N-Script Relationship Triggers & Conditions (R5-9)

Tests the enhanced N-Script system with relationship-based triggers:
- RelationshipQueryEngine functionality
- New trigger types (affinity, emotional patterns, memory significance)
- Enhanced TriggerMonitor with relationship support
- Visual N-Script builder integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from narrative_engine.nscript import (
    RelationshipQueryEngine, RelationshipQuery, RelationshipConditionEvaluator,
    TriggerMonitor, ScriptManager, Script, NScriptTrigger, NScriptAction,
    TriggerType, ActionType, GenerationParams, ControlParams, MemoryParams
)
from narrative_engine.state_manager import StateManager, EntityState, StateUpdate
from narrative_engine.relationship_manager import RelationshipManager, EnhancedRelationship


class TestRelationshipQueryEngine:
    """Test the RelationshipQueryEngine for complex relationship queries"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.state_manager = StateManager()
        self.relationship_manager = Mock(spec=RelationshipManager)
        self.query_engine = RelationshipQueryEngine(
            self.state_manager, 
            self.relationship_manager
        )
        
        # Create test entities with relationships
        self._setup_test_entities()
    
    def _setup_test_entities(self):
        """Set up test entities with relationship data"""
        tom_entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="village",
            relationships={
                "npc_clara": {
                    "affinity": 0.7,
                    "status": "Friend",
                    "emotional_history": ["happy", "grateful", "trusting"],
                    "memory_significance": 0.8,
                    "interaction_count": 5
                }
            }
        )
        
        clara_entity = EntityState(
            entity_id="npc_clara",
            entity_type="character",
            location="village",
            relationships={
                "npc_tom": {
                    "affinity": 0.6,
                    "status": "Friend", 
                    "emotional_history": ["pleased", "fond", "grateful"],
                    "memory_significance": 0.8,
                    "interaction_count": 5
                }
            }
        )
        
        self.state_manager.create_entity(tom_entity)
        self.state_manager.create_entity(clara_entity)
    
    @pytest.mark.asyncio
    async def test_affinity_threshold_query_above(self):
        """Test affinity threshold query - above threshold"""
        query = RelationshipQuery(
            relationship_pair=["npc_tom", "npc_clara"],
            condition_type="affinity_threshold",
            threshold_value=0.5,
            threshold_direction="above"
        )
        
        result = await self.query_engine.query_relationship(query)
        assert result == True  # Tom's affinity for Clara is 0.7 > 0.5
    
    @pytest.mark.asyncio
    async def test_affinity_threshold_query_below(self):
        """Test affinity threshold query - below threshold"""
        query = RelationshipQuery(
            relationship_pair=["npc_tom", "npc_clara"],
            condition_type="affinity_threshold",
            threshold_value=0.8,
            threshold_direction="below"
        )
        
        result = await self.query_engine.query_relationship(query)
        assert result == True  # Tom's affinity for Clara is 0.7 < 0.8
    
    @pytest.mark.asyncio
    async def test_emotional_pattern_query_match(self):
        """Test emotional pattern detection - pattern found"""
        query = RelationshipQuery(
            relationship_pair=["npc_tom", "npc_clara"],
            condition_type="emotional_pattern",
            emotional_sequence=["happy", "grateful"],
            within_interactions=3
        )
        
        result = await self.query_engine.query_relationship(query)
        assert result == True  # Pattern exists in emotional history
    
    @pytest.mark.asyncio
    async def test_emotional_pattern_query_no_match(self):
        """Test emotional pattern detection - pattern not found"""
        query = RelationshipQuery(
            relationship_pair=["npc_tom", "npc_clara"],
            condition_type="emotional_pattern",
            emotional_sequence=["angry", "betrayed", "hurt"],
            within_interactions=3
        )
        
        result = await self.query_engine.query_relationship(query)
        assert result == False  # Pattern doesn't exist in emotional history
    
    @pytest.mark.asyncio
    async def test_memory_significance_query_meets_threshold(self):
        """Test memory significance query - meets criteria"""
        query = RelationshipQuery(
            relationship_pair=["npc_tom", "npc_clara"],
            condition_type="memory_significance",
            threshold_value=0.7,
            memory_count=4
        )
        
        result = await self.query_engine.query_relationship(query)
        assert result == True  # Memory significance 0.8 > 0.7 and interaction count 5 > 4
    
    @pytest.mark.asyncio
    async def test_memory_significance_query_fails_threshold(self):
        """Test memory significance query - doesn't meet criteria"""
        query = RelationshipQuery(
            relationship_pair=["npc_tom", "npc_clara"],
            condition_type="memory_significance",
            threshold_value=0.9,
            memory_count=3
        )
        
        result = await self.query_engine.query_relationship(query)
        assert result == False  # Memory significance 0.8 < 0.9
    
    @pytest.mark.asyncio
    async def test_get_relationship_data_from_state_manager(self):
        """Test relationship data retrieval from StateManager"""
        relationship_data = await self.query_engine._get_relationship_data("npc_tom", "npc_clara")
        
        assert relationship_data is not None
        assert relationship_data["affinity"] == 0.7
        assert relationship_data["status"] == "Friend"
    
    @pytest.mark.asyncio
    async def test_get_relationship_data_nonexistent(self):
        """Test relationship data retrieval for non-existent relationship"""
        # Configure mock to return None for non-existent relationship
        self.relationship_manager.get_relationship.return_value = None
        
        relationship_data = await self.query_engine._get_relationship_data("npc_tom", "npc_unknown")
        
        assert relationship_data is None
    
    def test_get_emotional_patterns(self):
        """Test emotional pattern extraction"""
        patterns = self.query_engine.get_emotional_patterns("npc_tom", pattern_length=2)
        
        assert len(patterns) > 0
        assert "grateful->trusting" in patterns[0]


class TestRelationshipConditionEvaluator:
    """Test the RelationshipConditionEvaluator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.state_manager = StateManager()
        self.query_engine = RelationshipQueryEngine(self.state_manager)
        self.evaluator = RelationshipConditionEvaluator(self.query_engine)
        
        # Setup test data
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Set up test relationship data"""
        entity = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            relationships={
                "npc_clara": {
                    "affinity": 0.6,
                    "emotional_history": ["happy", "trusting", "grateful"],
                    "memory_significance": 0.8,
                    "interaction_count": 5
                }
            }
        )
        self.state_manager.create_entity(entity)
    
    @pytest.mark.asyncio
    async def test_evaluate_affinity_condition_success(self):
        """Test successful affinity condition evaluation"""
        condition = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "affinity_threshold": 0.5,
            "direction": "above"
        }
        
        result = await self.evaluator.evaluate_affinity_condition(condition)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_evaluate_emotional_pattern_success(self):
        """Test successful emotional pattern evaluation"""
        condition = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "emotion_sequence": ["happy", "trusting"],
            "within_interactions": 5
        }
        
        result = await self.evaluator.evaluate_emotional_pattern(condition)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_evaluate_memory_significance_success(self):
        """Test successful memory significance evaluation"""
        condition = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "memory_significance_above": 0.7,
            "memory_count": 4
        }
        
        result = await self.evaluator.evaluate_memory_significance(condition)
        assert result == True


class TestEnhancedTriggerMonitor:
    """Test the enhanced TriggerMonitor with relationship support"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.state_manager = StateManager()
        self.script_manager = ScriptManager()
        self.relationship_manager = Mock(spec=RelationshipManager)
        self.trigger_monitor = TriggerMonitor(
            self.state_manager, 
            self.script_manager,
            self.relationship_manager
        )
        
        # Setup test entities and scripts
        self._setup_test_entities()
        self._setup_test_scripts()
    
    def _setup_test_entities(self):
        """Set up test entities"""
        entities = ["npc_tom", "npc_clara", "player"]
        for entity_id in entities:
            entity = EntityState(
                entity_id=entity_id,
                entity_type="character",
                location="village"
            )
            self.state_manager.create_entity(entity)
    
    def _setup_test_scripts(self):
        """Set up test scripts with relationship triggers"""
        # Affinity threshold script
        affinity_script = Script(
            script_id="affinity_test",
            trigger=NScriptTrigger(
                type=TriggerType.ON_AFFINITY_THRESHOLD,
                conditions={
                    "relationship_pair": ["npc_tom", "npc_clara"],
                    "affinity_threshold": 0.5,
                    "direction": "above"
                }
            ),
            actions=[
                NScriptAction(
                    type=ActionType.TRIPLE_HEAD_ACTION,
                    target_agent_id="npc_tom",
                    generation_params=GenerationParams(style="friendly"),
                    control_params=ControlParams(emotions=["happy"]),
                    memory_params=MemoryParams(importance=0.8)
                )
            ]
        )
        
        # Emotional pattern script
        pattern_script = Script(
            script_id="pattern_test",
            trigger=NScriptTrigger(
                type=TriggerType.ON_EMOTIONAL_PATTERN,
                conditions={
                    "relationship_pair": ["npc_tom", "npc_clara"],
                    "emotion_sequence": ["betrayed", "angry", "hurt"],
                    "within_interactions": 3
                }
            ),
            actions=[
                NScriptAction(
                    type=ActionType.RELATIONSHIP_MODIFY,
                    target_agent_id="npc_clara"
                )
            ]
        )
        
        # Memory significance script
        memory_script = Script(
            script_id="memory_test",
            trigger=NScriptTrigger(
                type=TriggerType.ON_MEMORY_SIGNIFICANCE,
                conditions={
                    "relationship_pair": ["npc_tom", "npc_clara"],
                    "memory_significance_above": 0.8,
                    "memory_count": 5
                }
            ),
            actions=[
                NScriptAction(
                    type=ActionType.TRIGGER_CHAIN,
                    target_agent_id="narrator"
                )
            ]
        )
        
        # Add scripts to manager
        for script in [affinity_script, pattern_script, memory_script]:
            self.script_manager.add_script(script)
    
    @pytest.mark.asyncio
    async def test_handle_relationship_change_event(self):
        """Test relationship change event handling"""
        relationship_data = {
            "agent1": "npc_tom",
            "agent2": "npc_clara", 
            "affinity_change": 0.6,
            "previous_affinity": 0.2,
            "new_affinity": 0.8
        }
        
        self.trigger_monitor.start_monitoring()
        await self.trigger_monitor.handle_relationship_change_event(relationship_data)
        
        # No scripts should activate for basic relationship change
        # (we'd need specific ON_RELATIONSHIP_CHANGE scripts)
        activated = self.trigger_monitor.get_activated_scripts()
        assert len(activated) == 0
    
    @pytest.mark.asyncio
    async def test_handle_affinity_threshold_event(self):
        """Test affinity threshold event handling"""
        affinity_data = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "affinity": 0.7,
            "threshold": 0.5,
            "direction": "above"
        }
        
        self.trigger_monitor.start_monitoring()
        
        # Mock the condition evaluator to return True
        with patch.object(
            self.trigger_monitor.relationship_condition_evaluator,
            'evaluate_affinity_condition',
            return_value=True
        ):
            await self.trigger_monitor.handle_affinity_threshold_event(affinity_data)
        
        activated = self.trigger_monitor.get_activated_scripts()
        assert len(activated) == 1
        assert activated[0].script_id == "affinity_test"
    
    @pytest.mark.asyncio
    async def test_handle_emotional_pattern_event(self):
        """Test emotional pattern event handling"""
        pattern_data = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "emotion_sequence": ["betrayed", "angry", "hurt"],
            "within_interactions": 3
        }
        
        self.trigger_monitor.start_monitoring()
        
        # Mock the condition evaluator to return True
        with patch.object(
            self.trigger_monitor.relationship_condition_evaluator,
            'evaluate_emotional_pattern',
            return_value=True
        ):
            await self.trigger_monitor.handle_emotional_pattern_event(pattern_data)
        
        activated = self.trigger_monitor.get_activated_scripts()
        assert len(activated) == 1
        assert activated[0].script_id == "pattern_test"
    
    @pytest.mark.asyncio 
    async def test_handle_memory_significance_event(self):
        """Test memory significance event handling"""
        memory_data = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "memory_significance": 0.9,
            "memory_count": 6
        }
        
        self.trigger_monitor.start_monitoring()
        
        # Mock the condition evaluator to return True
        with patch.object(
            self.trigger_monitor.relationship_condition_evaluator,
            'evaluate_memory_significance',
            return_value=True
        ):
            await self.trigger_monitor.handle_memory_significance_event(memory_data)
        
        activated = self.trigger_monitor.get_activated_scripts()
        assert len(activated) == 1
        assert activated[0].script_id == "memory_test"
    
    def test_trigger_monitor_initialization_with_relationship_support(self):
        """Test that TriggerMonitor initializes with relationship components"""
        assert self.trigger_monitor.relationship_manager is not None
        assert self.trigger_monitor.relationship_query_engine is not None
        assert self.trigger_monitor.relationship_condition_evaluator is not None


class TestRelationshipScriptIntegration:
    """Integration tests for complete relationship-driven narrative scenarios"""
    
    def setup_method(self):
        """Set up integration test environment"""
        self.state_manager = StateManager()
        self.script_manager = ScriptManager()
        self.relationship_manager = RelationshipManager()
        self.trigger_monitor = TriggerMonitor(
            self.state_manager,
            self.script_manager, 
            self.relationship_manager
        )
        
        # Setup complete test scenario
        self._setup_scenario()
    
    def _setup_scenario(self):
        """Set up a complete narrative scenario"""
        # Create characters
        tom = EntityState(
            entity_id="npc_tom",
            entity_type="character",
            location="village_square",
            custom_data={
                "personality": {"agreeableness": 0.3, "openness": 0.8}
            },
            relationships={}
        )
        
        clara = EntityState(
            entity_id="npc_clara", 
            entity_type="character",
            location="village_square",
            custom_data={
                "personality": {"agreeableness": 0.8, "openness": 0.6}
            },
            relationships={}
        )
        
        self.state_manager.create_entity(tom)
        self.state_manager.create_entity(clara)
    
    @pytest.mark.asyncio
    async def test_betrayal_scenario_complete_workflow(self):
        """Test complete betrayal scenario from relationship change to script execution"""
        from narrative_engine.nscript import GenerationParams, ControlParams, MemoryParams
        
        # Create betrayal script
        betrayal_script = Script(
            script_id="betrayal_response",
            trigger=NScriptTrigger(
                type=TriggerType.ON_EMOTIONAL_PATTERN,
                conditions={
                    "relationship_pair": ["npc_tom", "npc_clara"],
                    "emotion_sequence": ["trusted", "betrayed", "hurt"],
                    "within_interactions": 2
                }
            ),
            actions=[
                NScriptAction(
                    type=ActionType.TRIPLE_HEAD_ACTION,
                    target_agent_id="npc_clara",
                    generation_params=GenerationParams(style="hurt"),
                    control_params=ControlParams(emotions=["betrayed"]),
                    memory_params=MemoryParams(importance=0.9)
                ),
                NScriptAction(
                    type=ActionType.RELATIONSHIP_MODIFY,
                    target_agent_id="npc_tom"
                )
            ]
        )
        
        self.script_manager.add_script(betrayal_script)
        self.trigger_monitor.start_monitoring()
        
        # Simulate betrayal emotional sequence
        pattern_data = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "emotion_sequence": ["trusted", "betrayed", "hurt"],
            "emotional_history": ["friendly", "trusted", "betrayed", "hurt"],
            "within_interactions": 2
        }
        
        # Mock condition evaluation to return True (betrayal pattern detected)
        with patch.object(
            self.trigger_monitor.relationship_condition_evaluator,
            'evaluate_emotional_pattern',
            return_value=True
        ):
            await self.trigger_monitor.handle_emotional_pattern_event(pattern_data)
        
        # Verify script was activated
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        assert activated_scripts[0].script_id == "betrayal_response"
        assert len(activated_scripts[0].actions) == 2
    
    @pytest.mark.asyncio
    async def test_friendship_milestone_unlocks_content(self):
        """Test friendship progression triggering new content"""
        # Create friendship milestone script
        friendship_script = Script(
            script_id="friendship_milestone",
            trigger=NScriptTrigger(
                type=TriggerType.ON_MEMORY_SIGNIFICANCE,
                conditions={
                    "relationship_pair": ["npc_tom", "npc_clara"],
                    "memory_significance_above": 0.8,
                    "memory_count": 5
                }
            ),
            actions=[
                NScriptAction(
                    type=ActionType.TRIGGER_CHAIN,
                    target_agent_id="narrator"
                )
            ]
        )
        
        self.script_manager.add_script(friendship_script)
        self.trigger_monitor.start_monitoring()
        
        # Simulate friendship milestone
        memory_data = {
            "relationship_pair": ["npc_tom", "npc_clara"],
            "memory_significance": 0.9,
            "memory_count": 6,
            "shared_memories": [
                {"content": "Helped each other", "significance": 0.8},
                {"content": "Shared secrets", "significance": 0.9},
                {"content": "Supported in crisis", "significance": 0.95}
            ]
        }
        
        # Mock condition evaluation
        with patch.object(
            self.trigger_monitor.relationship_condition_evaluator,
            'evaluate_memory_significance',
            return_value=True
        ):
            await self.trigger_monitor.handle_memory_significance_event(memory_data)
        
        # Verify milestone triggered
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        assert activated_scripts[0].script_id == "friendship_milestone"
    
    @pytest.mark.asyncio
    async def test_multiple_relationship_triggers_coexist(self):
        """Test that multiple relationship trigger types can coexist and fire independently"""
        from narrative_engine.nscript import GenerationParams, ControlParams, MemoryParams
        
        # Create multiple scripts with different trigger types
        scripts = [
            Script(
                script_id="affinity_script",
                trigger=NScriptTrigger(type=TriggerType.ON_AFFINITY_THRESHOLD),
                actions=[NScriptAction(
                    type=ActionType.TRIPLE_HEAD_ACTION, 
                    target_agent_id="npc_tom",
                    generation_params=GenerationParams(style="friendly"),
                    control_params=ControlParams(emotions=["happy"]),
                    memory_params=MemoryParams(importance=0.5)
                )]
            ),
            Script(
                script_id="pattern_script",
                trigger=NScriptTrigger(type=TriggerType.ON_EMOTIONAL_PATTERN),
                actions=[NScriptAction(type=ActionType.RELATIONSHIP_MODIFY, target_agent_id="npc_clara")]
            ),
            Script(
                script_id="memory_script",
                trigger=NScriptTrigger(type=TriggerType.ON_MEMORY_SIGNIFICANCE),
                actions=[NScriptAction(type=ActionType.TRIGGER_CHAIN, target_agent_id="narrator")]
            )
        ]
        
        for script in scripts:
            self.script_manager.add_script(script)
        
        self.trigger_monitor.start_monitoring()
        
        # Verify scripts are properly categorized
        affinity_scripts = self.script_manager.get_scripts_by_trigger_type("ON_AFFINITY_THRESHOLD")
        pattern_scripts = self.script_manager.get_scripts_by_trigger_type("ON_EMOTIONAL_PATTERN") 
        memory_scripts = self.script_manager.get_scripts_by_trigger_type("ON_MEMORY_SIGNIFICANCE")
        
        assert len(affinity_scripts) == 1
        assert len(pattern_scripts) == 1
        assert len(memory_scripts) == 1
        
        assert affinity_scripts[0].script_id == "affinity_script"
        assert pattern_scripts[0].script_id == "pattern_script"
        assert memory_scripts[0].script_id == "memory_script"


@pytest.mark.slow
@pytest.mark.integration
class TestNScriptRelationshipPerformance:
    """Performance tests for relationship-based N-Script system"""
    
    def setup_method(self):
        """Set up performance test environment"""
        self.state_manager = StateManager()
        self.relationship_manager = RelationshipManager()
        self.query_engine = RelationshipQueryEngine(
            self.state_manager,
            self.relationship_manager
        )
        
        # Create many entities for performance testing
        self._setup_large_scenario()
    
    def _setup_large_scenario(self):
        """Set up scenario with many characters and relationships"""
        character_count = 20
        
        for i in range(character_count):
            entity = EntityState(
                entity_id=f"npc_{i}",
                entity_type="character",
                relationships={}
            )
            self.state_manager.create_entity(entity)
            
            # Add some relationships
            for j in range(min(5, character_count)):
                if i != j:
                    entity.relationships[f"npc_{j}"] = {
                        "affinity": 0.5,
                        "emotional_history": ["neutral"] * 10,
                        "memory_significance": 0.5,
                        "interaction_count": 10
                    }
    
    @pytest.mark.asyncio
    async def test_query_engine_performance_many_relationships(self):
        """Test query engine performance with many relationships"""
        import time
        
        # Time multiple queries
        start_time = time.time()
        
        for i in range(10):
            for j in range(min(5, 20)):
                if i != j:
                    query = RelationshipQuery(
                        relationship_pair=[f"npc_{i}", f"npc_{j}"],
                        condition_type="affinity_threshold",
                        threshold_value=0.4,
                        threshold_direction="above"
                    )
                    
                    await self.query_engine.query_relationship(query)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete 50 queries in reasonable time (< 1 second)
        assert execution_time < 1.0, f"Query execution took {execution_time:.2f}s, expected < 1.0s"
    
    def test_memory_usage_scales_reasonably(self):
        """Test that memory usage scales reasonably with relationship count"""
        import sys
        
        initial_size = sys.getsizeof(self.query_engine._relationship_cache)
        
        # Add more cached relationships
        for i in range(100):
            cache_key = f"agent_{i}_agent_{i+1}"
            self.query_engine._relationship_cache[cache_key] = {
                "affinity": 0.5,
                "emotional_history": ["neutral"] * 5
            }
        
        final_size = sys.getsizeof(self.query_engine._relationship_cache)
        size_increase = final_size - initial_size
        
        # Memory increase should be reasonable (< 10KB for 100 relationships)
        assert size_increase < 10000, f"Memory increase {size_increase} bytes too large" 