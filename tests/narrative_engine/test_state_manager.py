"""
Tests for Runtime State Manager

Following TDD principles to build a robust state management system
for tracking dynamic world state.
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List
import asyncio

from narrative_engine.state_manager import (
    StateManager,
    EntityState,
    StateUpdate,
    StateTransaction,
    StateQuery,
    EventLog,
    StateManagerError,
    TransactionError,
)


@pytest.fixture
def state_manager():
    """Create a test state manager with in-memory backend"""
    return StateManager(backend="memory")


class TestEntityState:
    """Test the EntityState data model"""
    
    def test_create_character_entity(self):
        """Test creating a character entity state"""
        entity = EntityState(
            entity_id="clara_001",
            entity_type="character",
            location="town_square",
            inventory=["diary", "pen"],
            status_effects={},
            relationships={"user_001": {"type": "friend", "affinity": 50}},
            memories=[],  # Our new memory system!
            custom_data={"mood": "curious"},
        )
        
        assert entity.entity_id == "clara_001"
        assert entity.entity_type == "character"
        assert entity.location == "town_square"
        assert "diary" in entity.inventory
        assert entity.relationships["user_001"]["affinity"] == 50
    
    def test_entity_serialization(self):
        """Test that entities can be serialized to/from dict"""
        entity = EntityState(
            entity_id="item_001",
            entity_type="item",
            location="ground",
            custom_data={"name": "Ancient Tome", "weight": 2}
        )
        
        # Convert to dict
        data = entity.to_dict()
        assert data["entity_id"] == "item_001"
        assert data["custom_data"]["name"] == "Ancient Tome"
        
        # Recreate from dict
        entity2 = EntityState.from_dict(data)
        assert entity2.entity_id == entity.entity_id
        assert entity2.custom_data["weight"] == 2


class TestStateManager:
    """Test the core StateManager functionality"""
    
    def test_create_and_retrieve_entity(self, state_manager):
        """Test basic CRUD operations"""
        # Create entity
        entity = EntityState(
            entity_id="npc_001",
            entity_type="character",
            location="forest",
            custom_data={"name": "Forest Guardian"}
        )
        
        # Save it
        state_manager.create_entity(entity)
        
        # Retrieve it
        retrieved = state_manager.get_entity("npc_001")
        assert retrieved is not None
        assert retrieved.location == "forest"
        assert retrieved.custom_data["name"] == "Forest Guardian"
    
    def test_update_entity_state(self, state_manager):
        """Test updating entity state"""
        # Create initial entity
        entity = EntityState(
            entity_id="clara_001",
            entity_type="character",
            location="home",
            inventory=["book"],
        )
        state_manager.create_entity(entity)
        
        # Update location and inventory
        update = StateUpdate(
            entity_id="clara_001",
            changes={
                "location": "library",
                "inventory": ["book", "library_card"]
            }
        )
        
        state_manager.update_entity(update)
        
        # Verify changes
        updated = state_manager.get_entity("clara_001")
        assert updated.location == "library"
        assert "library_card" in updated.inventory
        assert len(updated.inventory) == 2
    
    def test_query_by_location(self, state_manager):
        """Test querying entities by location"""
        # Create multiple entities
        entities = [
            EntityState("char_1", "character", location="town_square"),
            EntityState("char_2", "character", location="town_square"),
            EntityState("char_3", "character", location="forest"),
            EntityState("item_1", "item", location="town_square"),
        ]
        
        for entity in entities:
            state_manager.create_entity(entity)
        
        # Query town square
        town_entities = state_manager.query_by_location("town_square")
        assert len(town_entities) == 3
        assert all(e.location == "town_square" for e in town_entities)
        
        # Query forest
        forest_entities = state_manager.query_by_location("forest")
        assert len(forest_entities) == 1
        assert forest_entities[0].entity_id == "char_3"
    
    def test_atomic_transaction(self, state_manager):
        """Test atomic transaction handling"""
        # Create two characters with items
        char1 = EntityState("char_1", "character", inventory=["gold_coin"])
        char2 = EntityState("char_2", "character", inventory=[])
        state_manager.create_entity(char1)
        state_manager.create_entity(char2)
        
        # Create transaction to transfer item
        transaction = StateTransaction([
            StateUpdate("char_1", {"inventory": []}),
            StateUpdate("char_2", {"inventory": ["gold_coin"]}),
        ])
        
        # Execute transaction
        state_manager.execute_transaction(transaction)
        
        # Verify both updates succeeded
        updated_char1 = state_manager.get_entity("char_1")
        updated_char2 = state_manager.get_entity("char_2")
        assert len(updated_char1.inventory) == 0
        assert "gold_coin" in updated_char2.inventory
    
    def test_failed_transaction_rollback(self, state_manager):
        """Test that failed transactions are rolled back"""
        # Create entity
        entity = EntityState("char_1", "character", location="start")
        state_manager.create_entity(entity)
        
        # Create transaction with invalid update
        transaction = StateTransaction([
            StateUpdate("char_1", {"location": "middle"}),
            StateUpdate("nonexistent_entity", {"location": "end"}),  # This will fail
        ])
        
        # Execute should raise error
        with pytest.raises(TransactionError):
            state_manager.execute_transaction(transaction)
        
        # Verify original state is preserved
        char = state_manager.get_entity("char_1")
        assert char.location == "start"  # Should not have changed
    
    def test_relationship_updates(self, state_manager):
        """Test updating character relationships"""
        # Create character with relationships
        char = EntityState(
            entity_id="clara_001",
            entity_type="character",
            relationships={
                "user_001": {"type": "acquaintance", "affinity": 20}
            }
        )
        state_manager.create_entity(char)
        
        # Update relationship
        update = StateUpdate(
            entity_id="clara_001",
            changes={
                "relationships": {
                    "user_001": {"type": "friend", "affinity": 75}
                }
            }
        )
        state_manager.update_entity(update)
        
        # Verify
        updated = state_manager.get_entity("clara_001")
        assert updated.relationships["user_001"]["type"] == "friend"
        assert updated.relationships["user_001"]["affinity"] == 75
    
    def test_memory_integration(self, state_manager):
        """Test that memories can be stored in entity state"""
        # Create character
        char = EntityState(
            entity_id="clara_001",
            entity_type="character",
            memories=[]
        )
        state_manager.create_entity(char)
        
        # Add a memory (from our memory system!)
        memory = {
            "id": "mem_001",
            "content": "User complimented my name",
            "importance": 0.8,
            "emotional_valence": 0.7,
            "timestamp": datetime.now().isoformat()
        }
        
        update = StateUpdate(
            entity_id="clara_001",
            changes={
                "memories": [memory]
            }
        )
        state_manager.update_entity(update)
        
        # Verify
        updated = state_manager.get_entity("clara_001")
        assert len(updated.memories) == 1
        assert updated.memories[0]["content"] == "User complimented my name"


class TestEventLog:
    """Test event logging functionality"""
    
    def test_event_logging(self, state_manager):
        """Test that state changes are logged as events"""
        # Create entity
        entity = EntityState("char_1", "character", location="start")
        state_manager.create_entity(entity)
        
        # Update location
        update = StateUpdate("char_1", {"location": "end"})
        state_manager.update_entity(update)
        
        # Get recent events
        events = state_manager.get_recent_events(limit=10)
        assert len(events) >= 2  # Create + update
        
        # Check update event
        update_event = next(e for e in events if e.event_type == "entity_updated")
        assert update_event.entity_id == "char_1"
        assert update_event.changes["location"] == "end"
    
    def test_event_timestamp_filtering(self, state_manager):
        """Test filtering events by timestamp"""
        # Create entity
        entity = EntityState("char_1", "character")
        state_manager.create_entity(entity)
        
        # Get timestamp
        checkpoint = datetime.now()
        
        # Make some updates
        for i in range(3):
            update = StateUpdate("char_1", {"custom_data": {"count": i}})
            state_manager.update_entity(update)
        
        # Get events since checkpoint
        events = state_manager.get_recent_events(since=checkpoint)
        assert len(events) == 3  # Only the updates, not the create


class TestStateQuery:
    """Test advanced querying capabilities"""
    
    def test_query_by_type(self, state_manager):
        """Test querying entities by type"""
        # Create mixed entities
        for i in range(3):
            state_manager.create_entity(
                EntityState(f"char_{i}", "character")
            )
        for i in range(2):
            state_manager.create_entity(
                EntityState(f"item_{i}", "item")
            )
        
        # Query characters
        characters = state_manager.query(
            StateQuery(entity_type="character")
        )
        assert len(characters) == 3
        
        # Query items
        items = state_manager.query(
            StateQuery(entity_type="item")
        )
        assert len(items) == 2
    
    def test_complex_query(self, state_manager):
        """Test complex queries with multiple filters"""
        # Create entities with various properties
        entities = [
            EntityState("char_1", "character", location="town", 
                       custom_data={"level": 5}),
            EntityState("char_2", "character", location="town",
                       custom_data={"level": 10}),
            EntityState("char_3", "character", location="forest",
                       custom_data={"level": 7}),
        ]
        
        for entity in entities:
            state_manager.create_entity(entity)
        
        # Query: characters in town with level > 7
        results = state_manager.query(
            StateQuery(
                entity_type="character",
                location="town",
                custom_filters=lambda e: e.custom_data.get("level", 0) > 7
            )
        )
        
        assert len(results) == 1
        assert results[0].entity_id == "char_2"


class TestConcurrency:
    """Test concurrent access patterns"""
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, state_manager):
        """Test that concurrent updates don't cause race conditions"""
        # Create entity
        entity = EntityState("counter", "item", custom_data={"count": 0})
        state_manager.create_entity(entity)
        
        # Define update function
        async def increment_counter():
            # Use atomic update to avoid race conditions
            def update_func(entity):
                return {"custom_data": {"count": entity.custom_data["count"] + 1}}
            
            # Simulate some async work before the update
            await asyncio.sleep(0.001)
            state_manager.atomic_update("counter", update_func)
        
        # Run multiple concurrent updates
        tasks = [increment_counter() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # With proper locking, count should be 10
        final = state_manager.get_entity("counter")
        assert final.custom_data["count"] == 10


class TestWorldIntegration:
    """Test integration with WorldManager"""
    
    def test_load_initial_state_from_world(self, state_manager):
        """Test loading initial world state"""
        # Mock world data
        world_data = {
            "places": [
                {"name": "Town Square", "npcs": [
                    {"name": "Guard", "description": "A vigilant guard"}
                ]},
                {"name": "Forest", "npcs": []}
            ]
        }
        
        # Load world state
        state_manager.load_world_state(world_data)
        
        # Verify NPCs are created as entities
        guard = state_manager.query(
            StateQuery(custom_filters=lambda e: 
                      e.custom_data.get("name") == "Guard")
        )
        assert len(guard) == 1
        assert guard[0].location == "Town Square" 