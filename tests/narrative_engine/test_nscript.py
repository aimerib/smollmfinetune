"""
Tests for N-Script (Narrative Scripting Engine) with triple-head support.

Tests the enhanced scripting system that allows designers to control all three
heads of the model: generation, control, and memory.
"""

import pytest
import json
import yaml
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path

from backend.app.narrative_engine.nscript import (
    ScriptManager, TriggerMonitor, ActionExecutor, 
    NScriptTrigger, NScriptAction, Script,
    GenerationParams, ControlParams, MemoryParams, CoordinationParams
)
from backend.app.narrative_engine.state_manager import StateManager, EntityState, EventLog, StateUpdate
from backend.app.narrative_engine.agent import BaseAgent


class TestNScriptParsing:
    """Test parsing of N-Script YAML files with triple-head support."""
    
    def test_parse_basic_script(self):
        """Test parsing a basic N-Script with triple-head actions."""
        script_yaml = """
        script_id: test_script
        trigger:
          type: ON_ENTER_LOCATION
          location_id: "dragon_lair"
          actor_filter: "player"
          once: true
        actions:
          - type: TRIPLE_HEAD_ACTION
            target_agent_id: "dragon_boss"
            generation_params:
              style: "menacing_monologue"
              tone: "ancient_wisdom"
            control_params:
              emotions: ["intimidating", "mysterious"]
              mood_shift: "hostile"
              control_tokens: ["<presence_overwhelming>", "<voice_booming>"]
            memory_params:
              importance: 0.9
              emotional_impact: 0.8
              tags: ["first_encounter", "dragon_lair"]
            action:
              type: SpeakToAction
              message: "Frail mortal, you have entered my domain..."
        """
        
        script_data = yaml.safe_load(script_yaml)
        script = Script.from_dict(script_data)
        
        assert script.script_id == "test_script"
        assert script.trigger.type == "ON_ENTER_LOCATION"
        assert script.trigger.location_id == "dragon_lair"
        assert script.trigger.once is True
        
        action = script.actions[0]
        assert action.type == "TRIPLE_HEAD_ACTION"
        assert action.target_agent_id == "dragon_boss"
        assert action.generation_params.style == "menacing_monologue"
        assert "intimidating" in action.control_params.emotions
        assert action.memory_params.importance == 0.9
    
    def test_parse_memory_trigger_script(self):
        """Test parsing script with memory formation trigger."""
        script_yaml = """
        script_id: memory_formation_script
        trigger:
          type: ON_MEMORY_FORMATION
          conditions:
            memory_significance: "> 0.8"
            emotional_impact: "> 0.7"
        actions:
          - type: MEMORY_FORMATION
            target_agent_id: "narrator"
            memory_params:
              importance: 0.95
              tags: ["pivotal_moment", "character_development"]
              emotional_weight: 0.85
        """
        
        script_data = yaml.safe_load(script_yaml)
        script = Script.from_dict(script_data)
        
        assert script.trigger.type == "ON_MEMORY_FORMATION"
        assert script.trigger.conditions["memory_significance"] == "> 0.8"
        
        action = script.actions[0]
        assert action.type == "MEMORY_FORMATION"
        assert action.memory_params.importance == 0.95
        assert "pivotal_moment" in action.memory_params.tags
    
    def test_parse_emotional_state_trigger(self):
        """Test parsing script with emotional state trigger."""
        script_yaml = """
        script_id: emotional_response_script
        trigger:
          type: ON_EMOTIONAL_STATE
          target_agent_id: "hero"
          conditions:
            emotional_state: "angry|frustrated"
            mood_intensity: "> 0.6"
        actions:
          - type: CONTROL_INJECTION
            target_agent_id: "companion"
            control_params:
              emotions: ["comforting", "supportive"]
              mood_shift: "calming"
              control_tokens: ["<voice_gentle>", "<gesture_reassuring>"]
        """
        
        script_data = yaml.safe_load(script_yaml)
        script = Script.from_dict(script_data)
        
        assert script.trigger.type == "ON_EMOTIONAL_STATE"
        assert script.trigger.target_agent_id == "hero"
        assert "angry|frustrated" in script.trigger.conditions["emotional_state"]
        
        action = script.actions[0]
        assert action.type == "CONTROL_INJECTION"
        assert "comforting" in action.control_params.emotions
    
    def test_script_validation_fails_on_invalid_data(self):
        """Test that script validation fails on invalid data."""
        invalid_script = {
            "script_id": "invalid",
            "trigger": {
                "type": "INVALID_TRIGGER_TYPE"
            },
            "actions": []
        }
        
        with pytest.raises(Exception):  # Catch any validation error
            Script.from_dict(invalid_script)
    
    def test_triple_head_coordination_script(self):
        """Test parsing script that coordinates all three heads."""
        script_yaml = """
        script_id: coordination_test
        trigger:
          type: ON_HEAD_COORDINATION
          conditions:
            generation_quality: "> 0.8"
            control_consistency: "> 0.7"
            memory_relevance: "> 0.6"
        actions:
          - type: HEAD_SYNCHRONIZATION
            target_agent_id: "main_character"
            coordination_params:
              sync_emotional_state: true
              align_memory_focus: true
              harmonize_narrative_tone: true
        """
        
        script_data = yaml.safe_load(script_yaml)
        script = Script.from_dict(script_data)
        
        assert script.trigger.type == "ON_HEAD_COORDINATION"
        action = script.actions[0]
        assert action.type == "HEAD_SYNCHRONIZATION"
        assert action.coordination_params.sync_emotional_state is True


class TestScriptManager:
    """Test the ScriptManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.script_manager = ScriptManager()
    
    def test_create_script_manager(self):
        """Test that ScriptManager can be created."""
        assert self.script_manager is not None
        assert hasattr(self.script_manager, 'load_script_from_yaml')
    
    def test_load_script_from_yaml(self):
        """Test loading a script from YAML content."""
        yaml_content = """
        script_id: test_load
        trigger:
          type: ON_ENTER_LOCATION
          location_id: "test_room"
        actions:
          - type: TRIPLE_HEAD_ACTION
            target_agent_id: "test_npc"
            generation_params:
              style: "friendly"
            control_params:
              emotions: ["welcoming"]
            memory_params:
              importance: 0.5
            action:
              type: SpeakToAction
              message: "Welcome!"
        """
        
        script = self.script_manager.load_script_from_yaml(yaml_content)
        assert script.script_id == "test_load"
        assert len(script.actions) == 1
    
    def test_validate_script_with_valid_data(self):
        """Test script validation with valid triple-head data."""
        script_data = {
            "script_id": "valid_script",
            "trigger": {
                "type": "ON_ENTER_LOCATION",
                "location_id": "valid_location"
            },
            "actions": [{
                "type": "TRIPLE_HEAD_ACTION",
                "target_agent_id": "valid_agent",
                "generation_params": {"style": "normal"},
                "control_params": {"emotions": ["neutral"]},
                "memory_params": {"importance": 0.5},
                "action": {
                    "type": "SpeakToAction",
                    "message": "Hello"
                }
            }]
        }
        
        is_valid, errors = self.script_manager.validate_script(script_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_script_with_invalid_data(self):
        """Test script validation with invalid data."""
        invalid_script = {
            "script_id": "",  # Empty ID
            "trigger": {
                "type": "INVALID_TYPE"  # Invalid trigger type
            },
            "actions": []  # No actions
        }
        
        is_valid, errors = self.script_manager.validate_script(invalid_script)
        assert not is_valid
        assert len(errors) > 0
        assert any("script_id cannot be empty" in error for error in errors)
    
    def test_get_scripts_by_trigger_type(self):
        """Test filtering scripts by trigger type."""
        # Load multiple scripts with different trigger types
        scripts = [
            Script(
                script_id="location_script",
                trigger=NScriptTrigger(type="ON_ENTER_LOCATION", location_id="room1"),
                actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.5))]
            ),
            Script(
                script_id="memory_script", 
                trigger=NScriptTrigger(type="ON_MEMORY_FORMATION"),
                actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.5))]
            ),
            Script(
                script_id="emotional_script",
                trigger=NScriptTrigger(type="ON_EMOTIONAL_STATE"),
                actions=[NScriptAction(type="CONTROL_INJECTION", target_agent_id="test", control_params=ControlParams(emotions=["calm"]))]
            )
        ]
        
        for script in scripts:
            self.script_manager.add_script(script)
        
        location_scripts = self.script_manager.get_scripts_by_trigger_type("ON_ENTER_LOCATION")
        assert len(location_scripts) == 1
        assert location_scripts[0].script_id == "location_script"
        
        memory_scripts = self.script_manager.get_scripts_by_trigger_type("ON_MEMORY_FORMATION")
        assert len(memory_scripts) == 1
        assert memory_scripts[0].script_id == "memory_script"


class TestTriggerMonitor:
    """Test the enhanced TriggerMonitor with triple-head support."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_manager = StateManager()
        self.script_manager = ScriptManager()
        self.trigger_monitor = TriggerMonitor(self.state_manager, self.script_manager)
        
        # Create test entities
        player_entity = EntityState(
            entity_id="player",
            entity_type="character",
            location="starting_room"
        )
        npc_entity = EntityState(
            entity_id="test_npc",
            entity_type="character", 
            location="test_room"
        )
        
        self.state_manager.create_entity(player_entity)
        self.state_manager.create_entity(npc_entity)
    
    def test_location_trigger_monitoring(self):
        """Test monitoring of location-based triggers."""
        # Create a location trigger script
        script = Script(
            script_id="location_test",
            trigger=NScriptTrigger(
                type="ON_ENTER_LOCATION",
                location_id="test_room",
                actor_filter="player",
                once=True
            ),
            actions=[
                NScriptAction(
                    type="TRIPLE_HEAD_ACTION",
                    target_agent_id="test_npc",
                    generation_params=GenerationParams(style="friendly")
                )
            ]
        )
        
        self.script_manager.add_script(script)
        self.trigger_monitor.start_monitoring()
        
        # Simulate player entering the room
        update = StateUpdate(
            entity_id="player",
            changes={"location": "test_room"}
        )
        self.state_manager.update_entity(update)
        
        # Manually trigger the location check
        self.trigger_monitor.check_location_triggers("player", "test_room")
        
        # Check if trigger was activated
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        assert activated_scripts[0].script_id == "location_test"
    
    
    async def test_memory_formation_trigger(self):
        """Test triggers based on memory formation events."""
        # Mock memory formation event
        memory_data = {
            "content": "The dragon's roar echoed through the cavern",
            "memory_significance": 0.9,  # Match the condition key
            "emotional_impact": 0.8,
            "tags": ["dramatic_moment"]
        }
        
        script = Script(
            script_id="memory_trigger_test",
            trigger=NScriptTrigger(
                type="ON_MEMORY_FORMATION",
                conditions={
                    "memory_significance": "> 0.8",
                    "emotional_impact": "> 0.7"
                }
            ),
            actions=[
                NScriptAction(
                    type="MEMORY_FORMATION",
                    target_agent_id="narrator",
                    memory_params=MemoryParams(importance=0.8)
                )
            ]
        )
        
        self.script_manager.add_script(script)
        self.trigger_monitor.start_monitoring()
        
        # Simulate memory formation
        await self.trigger_monitor.handle_memory_formation_event("test_character", memory_data)
        
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        assert activated_scripts[0].script_id == "memory_trigger_test"
    
    
    async def test_emotional_state_trigger(self):
        """Test triggers based on emotional state changes."""
        emotional_data = {
            "agent_id": "hero",
            "emotional_state": "angry",
            "mood_intensity": 0.8,
            "control_tokens": ["<mood_hostile>", "<expression_fierce>"]
        }
        
        script = Script(
            script_id="emotional_trigger_test",
            trigger=NScriptTrigger(
                type="ON_EMOTIONAL_STATE",
                target_agent_id="hero",
                conditions={
                    "emotional_state": "angry|frustrated",
                    "mood_intensity": "> 0.6"
                }
            ),
            actions=[
                NScriptAction(
                    type="CONTROL_INJECTION",
                    target_agent_id="companion",
                    control_params=ControlParams(emotions=["comforting"])
                )
            ]
        )
        
        self.script_manager.add_script(script)
        self.trigger_monitor.start_monitoring()
        
        # Simulate emotional state change
        await self.trigger_monitor.handle_emotional_state_event(emotional_data)
        
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        assert activated_scripts[0].script_id == "emotional_trigger_test"
    
    def test_once_trigger_prevents_reactivation(self):
        """Test that 'once' triggers don't reactivate."""
        script = Script(
            script_id="once_test",
            trigger=NScriptTrigger(
                type="ON_ENTER_LOCATION",
                location_id="test_room",
                actor_filter="player",
                once=True
            ),
            actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.5))]
        )
        
        self.script_manager.add_script(script)
        self.trigger_monitor.start_monitoring()
        
        # First activation
        update = StateUpdate(entity_id="player", changes={"location": "test_room"})
        self.state_manager.update_entity(update)
        self.trigger_monitor.check_location_triggers("player", "test_room")
        
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        
        # Clear activated scripts
        self.trigger_monitor.clear_activated_scripts()
        
        # Second attempt - should not activate again
        update = StateUpdate(entity_id="player", changes={"location": "test_room"})  
        self.state_manager.update_entity(update)
        self.trigger_monitor.check_location_triggers("player", "test_room")
        
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 0  # Should be empty due to 'once' flag


class TestActionExecutor:
    """Test the enhanced ActionExecutor with triple-head actions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_manager = StateManager()
        self.mock_narrative_model = Mock()
        self.action_executor = ActionExecutor(
            self.state_manager, 
            narrative_model=self.mock_narrative_model
        )
        
        # Create test entities
        test_agent = EntityState(
            entity_id="test_agent",
            entity_type="character",
            location="test_location"
        )
        self.state_manager.create_entity(test_agent)
    
    
    async def test_execute_triple_head_action(self):
        """Test execution of triple-head coordinated actions."""
        action = NScriptAction(
            type="TRIPLE_HEAD_ACTION",
            target_agent_id="test_agent",
            generation_params={
                "style": "dramatic",
                "tone": "mysterious"
            },
            control_params={
                "emotions": ["mysterious", "intriguing"],
                "mood_shift": "contemplative",
                "control_tokens": ["<voice_whisper>", "<gesture_thoughtful>"]
            },
            memory_params={
                "importance": 0.8,
                "emotional_impact": 0.7,
                "tags": ["revelation", "mystery"]
            },
            action={
                "type": "SpeakToAction",
                "message": "The truth you seek lies beyond the veil..."
            }
        )
        
        # Mock the narrative model response
        self.mock_narrative_model.generate_with_triple_head_control = AsyncMock(
            return_value={
                "generation_output": "The truth you seek lies beyond the veil, mortal.",
                "control_output": {
                    "emotions": ["mysterious", "intriguing"],
                    "control_tokens": ["<voice_whisper>", "<gesture_thoughtful>"]
                },
                "memory_output": {
                    "formed_memory": {
                        "content": "Revealed mysterious truth to seeker",
                        "importance": 0.8,
                        "tags": ["revelation", "mystery"]
                    }
                }
            }
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert result.success
        assert "triple-head action executed" in result.message.lower()
        assert self.mock_narrative_model.generate_with_triple_head_control.called
    
    
    async def test_execute_control_injection(self):
        """Test execution of control injection actions."""
        action = NScriptAction(
            type="CONTROL_INJECTION",
            target_agent_id="test_agent",
            control_params={
                "emotions": ["calming", "reassuring"],
                "mood_shift": "peaceful",
                "control_tokens": ["<voice_gentle>", "<presence_warm>"]
            }
        )
        
        self.mock_narrative_model.inject_control_state = AsyncMock(
            return_value={
                "success": True,
                "applied_emotions": ["calming", "reassuring"],
                "mood_change": "peaceful"
            }
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert result.success
        assert "control injection" in result.message.lower()
        assert self.mock_narrative_model.inject_control_state.called
    
    
    async def test_execute_memory_formation(self):
        """Test execution of memory formation actions."""
        action = NScriptAction(
            type="MEMORY_FORMATION",
            target_agent_id="test_agent",
            memory_params={
                "importance": 0.9,
                "emotional_impact": 0.8,
                "tags": ["critical_moment", "character_growth"],
                "content": "A moment of profound realization"
            }
        )
        
        self.mock_narrative_model.form_explicit_memory = AsyncMock(
            return_value={
                "memory_id": "mem_12345",
                "success": True,
                "memory_data": {
                    "content": "A moment of profound realization",
                    "importance": 0.9,
                    "tags": ["critical_moment", "character_growth"]
                }
            }
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert result.success
        assert "memory formation" in result.message.lower()
        assert self.mock_narrative_model.form_explicit_memory.called
    
    
    async def test_execute_head_synchronization(self):
        """Test execution of head synchronization actions."""
        action = NScriptAction(
            type="HEAD_SYNCHRONIZATION",
            target_agent_id="test_agent",
            coordination_params={
                "sync_emotional_state": True,
                "align_memory_focus": True,
                "harmonize_narrative_tone": True
            }
        )
        
        self.mock_narrative_model.synchronize_heads = AsyncMock(
            return_value={
                "success": True,
                "synchronization_report": {
                    "emotional_alignment": 0.92,
                    "memory_coherence": 0.88,
                    "narrative_harmony": 0.95
                }
            }
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert result.success
        assert "synchronization" in result.message.lower()
        assert self.mock_narrative_model.synchronize_heads.called
    
      
    async def test_execute_generation_override(self):
        """Test execution of generation override actions."""
        action = NScriptAction(
            type="GENERATION_OVERRIDE",
            target_agent_id="test_agent",
            generation_params={
                "style": "poetic",
                "tone": "melancholic",
                "narrative_focus": "inner_thoughts",
                "override_duration": 5  # 5 turns
            }
        )
        
        self.mock_narrative_model.override_generation_params = AsyncMock(
            return_value={
                "success": True,
                "override_id": "gen_override_123",
                "active_until": "turn_25"
            }
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert result.success
        assert "generation override" in result.message.lower()
        assert self.mock_narrative_model.override_generation_params.called
    
    
    async def test_action_execution_with_invalid_target(self):
        """Test action execution fails gracefully with invalid target."""
        action = NScriptAction(
            type="TRIPLE_HEAD_ACTION",
            target_agent_id="nonexistent_agent",
            generation_params=GenerationParams(style="test")
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert not result.success
        assert "not found" in result.message.lower()
    
    
    async def test_action_execution_with_model_error(self):
        """Test action execution handles model errors gracefully."""
        action = NScriptAction(
            type="TRIPLE_HEAD_ACTION",
            target_agent_id="test_agent",
            generation_params=GenerationParams(style="test")
        )
        
        self.mock_narrative_model.generate_with_triple_head_control = AsyncMock(
            side_effect=Exception("Model error")
        )
        
        result = await self.action_executor.execute_action(action)
        
        assert not result.success
        assert "error" in result.message.lower()


class TestNScriptIntegration:
    """Integration tests for the complete N-Script system."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.state_manager = StateManager()
        self.script_manager = ScriptManager()
        self.trigger_monitor = TriggerMonitor(self.state_manager, self.script_manager)
        self.action_executor = ActionExecutor(self.state_manager)
        
        # Create test world
        player = EntityState(
            entity_id="player",
            entity_type="character",
            location="village_square"
        )
        dragon = EntityState(
            entity_id="dragon_boss",
            entity_type="character",
            location="dragon_lair"
        )
        
        self.state_manager.create_entity(player)
        self.state_manager.create_entity(dragon)
    
    
    async def test_complete_script_execution_flow(self):
        """Test complete flow from trigger to action execution."""
        # Load a complete script
        script_yaml = """
        script_id: dragon_encounter
        trigger:
          type: ON_ENTER_LOCATION
          location_id: "dragon_lair"
          actor_filter: "player"
          once: true
        actions:
          - type: TRIPLE_HEAD_ACTION
            target_agent_id: "dragon_boss"
            generation_params:
              style: "menacing_monologue"
              tone: "ancient_wisdom"
            control_params:
              emotions: ["intimidating", "mysterious"]
              mood_shift: "hostile"
            memory_params:
              importance: 0.9
              emotional_impact: 0.8
              tags: ["first_encounter", "dragon_lair"]
            action:
              type: SpeakToAction
              message: "So... another mortal dares enter my domain."
        """
        
        script = self.script_manager.load_script_from_yaml(script_yaml)
        self.script_manager.add_script(script)
        self.trigger_monitor.start_monitoring()
        
        # Simulate player entering dragon lair
        update = StateUpdate(
            entity_id="player",
            changes={"location": "dragon_lair"}
        )
        self.state_manager.update_entity(update)
        
        # Manually trigger the location check (in real implementation this would be automatic)
        self.trigger_monitor.check_location_triggers("player", "dragon_lair")
        
        # Check trigger activated
        activated_scripts = self.trigger_monitor.get_activated_scripts()
        assert len(activated_scripts) == 1
        
        # Execute the actions
        script_to_execute = activated_scripts[0]
        for action in script_to_execute.actions:
            result = await self.action_executor.execute_action(action)
            # Should succeed even without real model (fallback behavior)
            assert result is not None
    
    def test_script_hot_reload(self):
        """Test hot-reloading of scripts during runtime."""
        # Initial script
        original_script = Script(
            script_id="test_reload",
            trigger=NScriptTrigger(type="ON_ENTER_LOCATION", location_id="room1"),
            actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.5))]
        )
        
        self.script_manager.add_script(original_script)
        assert len(self.script_manager.get_all_scripts()) == 1
        
        # Updated script
        updated_script = Script(
            script_id="test_reload",  # Same ID
            trigger=NScriptTrigger(type="ON_ENTER_LOCATION", location_id="room2"),  # Different location
            actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.7))]
        )
        
        self.script_manager.add_script(updated_script)  # Should replace
        
        scripts = self.script_manager.get_all_scripts()
        assert len(scripts) == 1
        assert scripts[0].trigger.location_id == "room2"
    
    def test_multiple_trigger_types_coexist(self):
        """Test that multiple trigger types can coexist."""
        scripts = [
            Script(
                script_id="location_script",
                trigger=NScriptTrigger(type="ON_ENTER_LOCATION", location_id="room1"),
                actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.5))]
            ),
            Script(
                script_id="memory_script",
                trigger=NScriptTrigger(type="ON_MEMORY_FORMATION"),
                actions=[NScriptAction(type="MEMORY_FORMATION", target_agent_id="test", memory_params=MemoryParams(importance=0.5))]
            ),
            Script(
                script_id="emotional_script",
                trigger=NScriptTrigger(type="ON_EMOTIONAL_STATE"),
                actions=[NScriptAction(type="CONTROL_INJECTION", target_agent_id="test", control_params=ControlParams(emotions=["calm"]))]
            ),
            Script(
                script_id="coordination_script",
                trigger=NScriptTrigger(type="ON_HEAD_COORDINATION"),
                actions=[NScriptAction(type="HEAD_SYNCHRONIZATION", target_agent_id="test", coordination_params=CoordinationParams(sync_emotional_state=True))]
            )
        ]
        
        for script in scripts:
            self.script_manager.add_script(script)
        
        assert len(self.script_manager.get_all_scripts()) == 4
        
        # Check that each trigger type has its scripts
        assert len(self.script_manager.get_scripts_by_trigger_type("ON_ENTER_LOCATION")) == 1
        assert len(self.script_manager.get_scripts_by_trigger_type("ON_MEMORY_FORMATION")) == 1
        assert len(self.script_manager.get_scripts_by_trigger_type("ON_EMOTIONAL_STATE")) == 1
        assert len(self.script_manager.get_scripts_by_trigger_type("ON_HEAD_COORDINATION")) == 1 