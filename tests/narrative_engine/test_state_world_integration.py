"""
Integration test demonstrating the full narrative engine working together:
- WorldManager loads static world data
- StateManager tracks dynamic state
- Memory system records character experiences
"""

import pytest
from datetime import datetime
from pathlib import Path

from backend.app.services.world.world import WorldManager, WorldLore, Place, NPC
from backend.app.narrative_engine.state_manager import StateManager, EntityState, StateUpdate
from backend.app.narrative_engine.memory_schema import MemoryAnnotation


class TestFullIntegration:
    """Test the complete integration of world, state, and memory systems"""
    
    @pytest.fixture
    def world_manager(self, tmp_path):
        """Create a test world manager with temporary directory"""
        return WorldManager(worlds_root=str(tmp_path))
    
    @pytest.fixture
    def state_manager(self):
        """Create a test state manager"""
        return StateManager(backend="memory")
    
    def test_clara_explores_world(self, world_manager, state_manager):
        """
        Simulate Clara exploring a world, forming memories, and building relationships.
        This tests the full narrative loop!
        """
        # Step 1: Create a world with interesting places
        world_manager.create_world("Ethereal Valley")
        
        # Add some lore
        lore = WorldLore(
            meta={"version": 1},
            facts={
                "magic_level": "high",
                "primary_species": "mixed humanoid and fae"
            },
            places=[
                Place(
                    name="Crystal Grove",
                    description="A mystical grove where crystals sing in the wind",
                    npcs=[
                        NPC("Elder Tree", "An ancient sentient tree that whispers secrets")
                    ]
                ),
                Place(
                    name="Village Square", 
                    description="The bustling heart of the village",
                    npcs=[
                        NPC("Merchant Maya", "A friendly merchant with stories from afar")
                    ]
                )
            ]
        )
        
        world_manager.current_world = "Ethereal Valley"
        world_manager.save_world_lore(lore=lore)
        
        # Step 2: Load world into state manager
        state_manager.load_world_state({
            "places": [
                {
                    "name": "Crystal Grove",
                    "npcs": [{"name": "Elder Tree", "description": "An ancient sentient tree"}]
                },
                {
                    "name": "Village Square",
                    "npcs": [{"name": "Merchant Maya", "description": "A friendly merchant"}]
                }
            ]
        })
        
        # Step 3: Create Clara as a character
        clara = EntityState(
            entity_id="clara_001",
            entity_type="character",
            location="Village Square",
            inventory=["journal", "pen"],
            memories=[],
            relationships={},
            custom_data={
                "name": "Clara",
                "personality": {
                    "openness": 0.9,
                    "conscientiousness": 0.7,
                    "extraversion": 0.6,
                    "agreeableness": 0.8,
                    "neuroticism": 0.3
                },
                "mood": "curious"
            }
        )
        state_manager.create_entity(clara)
        
        # Step 4: Clara meets Maya and forms a memory
        maya = state_manager.query_by_location("Village Square")[0]  # Find Maya
        
        # Clara forms a positive memory of meeting Maya
        meeting_memory = {
            "id": "mem_001",
            "content": "Met Maya at the village square. She told me about the Crystal Grove!",
            "timestamp": datetime.now().isoformat(),
            "importance": 0.7,
            "surprise": 0.5,
            "emotional_valence": 0.8,
            "persistence_weight": 0.75,
            "context": {
                "location": "Village Square",
                "participants": ["Merchant Maya"],
                "tokens_used": ["<memory_form>", "<importance_medium>", "<valence_positive>"]
            }
        }
        
        state_manager.add_memory_to_character("clara_001", meeting_memory)
        
        # Update Clara's relationships
        state_manager.update_entity(StateUpdate(
            entity_id="clara_001",
            changes={
                "relationships": {
                    maya.entity_id: {
                        "type": "acquaintance", 
                        "affinity": 40,
                        "first_meeting": datetime.now().isoformat()
                    }
                }
            }
        ))
        
        # Step 5: Clara travels to Crystal Grove
        state_manager.update_entity(StateUpdate(
            entity_id="clara_001",
            changes={"location": "Crystal Grove"}
        ))
        
        # Step 6: Clara has a profound experience at the grove
        profound_memory = {
            "id": "mem_002",
            "content": "The Elder Tree spoke to me! It knew my name before I said it. The crystals resonated with my thoughts.",
            "timestamp": datetime.now().isoformat(),
            "importance": 1.0,  # Maximum importance!
            "surprise": 0.95,   # Very surprising
            "emotional_valence": 0.9,  # Deeply positive
            "persistence_weight": 0.98,  # Will remember forever
            "context": {
                "location": "Crystal Grove",
                "participants": ["Elder Tree"],
                "tokens_used": ["<memory_form>", "<importance_high>", "<type_emotional>", "<valence_positive>"]
            }
        }
        
        state_manager.add_memory_to_character("clara_001", profound_memory)
        
        # Update mood based on experience
        def update_mood(entity):
            return {
                "custom_data": {
                    **entity.custom_data,
                    "mood": "awestruck"
                }
            }
        
        state_manager.atomic_update("clara_001", update_mood)
        
        # Step 7: Verify Clara's state
        clara_final = state_manager.get_entity("clara_001")
        
        # Check location
        assert clara_final.location == "Crystal Grove"
        
        # Check memories
        assert len(clara_final.memories) == 2
        assert clara_final.memories[1]["importance"] == 1.0
        assert clara_final.memories[1]["surprise"] == 0.95
        
        # Check relationships
        assert maya.entity_id in clara_final.relationships
        assert clara_final.relationships[maya.entity_id]["type"] == "acquaintance"
        
        # Check mood change
        assert clara_final.custom_data["mood"] == "awestruck"
        
        # Step 8: Query recent events
        events = state_manager.get_recent_events(limit=10)
        
        # Should have events for: Clara created, memories added, location change, etc.
        event_types = [e.event_type for e in events]
        assert "entity_created" in event_types
        assert "entity_updated" in event_types
        
        # Step 9: Test memory retrieval by importance
        important_memories = [m for m in clara_final.memories if m["importance"] > 0.8]
        assert len(important_memories) == 1
        assert "Elder Tree" in important_memories[0]["content"]
    
    def test_world_persistence_simulation(self, state_manager):
        """Test that the world persists across time with multiple characters"""
        # Create a simple world state
        entities = [
            EntityState("tree_001", "object", location="forest", 
                       custom_data={"type": "oak", "age": 100}),
            EntityState("rock_001", "object", location="forest",
                       custom_data={"type": "granite", "moveable": False}),
            EntityState("clara", "character", location="forest"),
            EntityState("user", "character", location="village")
        ]
        
        for entity in entities:
            state_manager.create_entity(entity)
        
        # Simulate Clara picking up a stick (creating new entity)
        stick = EntityState("stick_001", "item", location="clara",  # In Clara's inventory
                          custom_data={"type": "branch", "from": "tree_001"})
        state_manager.create_entity(stick)
        
        # Update Clara's inventory
        state_manager.update_entity(StateUpdate(
            entity_id="clara",
            changes={"inventory": ["stick_001"]}
        ))
        
        # User travels to forest
        state_manager.update_entity(StateUpdate(
            entity_id="user",
            changes={"location": "forest"}
        ))
        
        # Query who's in the forest now
        forest_entities = state_manager.query_by_location("forest")
        forest_ids = [e.entity_id for e in forest_entities]
        
        assert "clara" in forest_ids
        assert "user" in forest_ids
        assert "tree_001" in forest_ids
        assert "rock_001" in forest_ids
        assert "stick_001" not in forest_ids  # It's in Clara's inventory!
        
        # Verify Clara has the stick
        clara = state_manager.get_entity("clara")
        assert "stick_001" in clara.inventory 