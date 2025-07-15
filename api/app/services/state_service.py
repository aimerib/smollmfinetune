"""
State Service

Integrates with the existing StateManager to provide world state access
and real-time updates for the Director's View.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import structlog

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from narrative_engine.state_manager import StateManager, EntityState, StateQuery, StateUpdate
from app.services.event_bus import event_bus, StateUpdateEvent

logger = structlog.get_logger()


class StateService:
    """Service for managing world state and entity tracking"""
    
    def __init__(self):
        self.state_manager: Optional[StateManager] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the state service"""
        try:
            # Create state manager instance
            self.state_manager = StateManager(backend="memory")
            
            # Create demo entities for testing
            await self._create_demo_entities()
            
            self._initialized = True
            logger.info("State service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize state service", error=str(e))
            raise
    
    async def _create_demo_entities(self):
        """Create demo entities for testing the Director's View"""
        # Demo locations
        locations = [
            {"id": "village_square", "name": "Village Square", "type": "location"},
            {"id": "forest_path", "name": "Forest Path", "type": "location"},
            {"id": "elder_tree", "name": "The Elder Tree", "type": "location"},
            {"id": "market", "name": "Market", "type": "location"}
        ]
        
        # Demo characters
        characters = [
            {
                "id": "clara_001", 
                "name": "Clara",
                "type": "character",
                "location": "village_square",
                "mood": "curious",
                "personality_traits": {
                    "openness": 0.8,
                    "conscientiousness": 0.6,
                    "extraversion": 0.7,
                    "agreeableness": 0.9,
                    "neuroticism": 0.3
                }
            },
            {
                "id": "elder_001",
                "name": "Village Elder",
                "type": "character", 
                "location": "elder_tree",
                "mood": "wise",
                "personality_traits": {
                    "openness": 0.6,
                    "conscientiousness": 0.9,
                    "extraversion": 0.4,
                    "agreeableness": 0.8,
                    "neuroticism": 0.2
                }
            },
            {
                "id": "merchant_001",
                "name": "Traveling Merchant",
                "type": "character",
                "location": "market",
                "mood": "cheerful",
                "personality_traits": {
                    "openness": 0.7,
                    "conscientiousness": 0.7,
                    "extraversion": 0.9,
                    "agreeableness": 0.7,
                    "neuroticism": 0.4
                }
            }
        ]
        
        # Create entities
        for loc in locations:
            entity = EntityState(
                entity_id=loc["id"],
                entity_type=loc["type"],
                location="world",
                custom_data={"name": loc["name"]}
            )
            self.state_manager.create_entity(entity)
        
        for char in characters:
            entity = EntityState(
                entity_id=char["id"],
                entity_type=char["type"],
                location=char["location"],
                custom_data={
                    "name": char["name"],
                    "mood": char["mood"],
                    "personality_traits": char["personality_traits"]
                }
            )
            self.state_manager.create_entity(entity)
            
            # Add some demo subtext
            self.state_manager.add_subtext(
                char["id"],
                f"*{char['name']} is here, feeling {char['mood']}*"
            )
    
    async def get_world_snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of the world state"""
        if not self._initialized:
            return {"error": "State service not initialized"}
        
        # Get all entities
        all_entities = self.state_manager.backend.entities
        
        # Organize by type
        locations = []
        characters = []
        
        for entity_id, entity in all_entities.items():
            entity_dict = entity.to_dict()
            
            if entity.entity_type == "location":
                locations.append({
                    "id": entity_id,
                    "name": entity.custom_data.get("name", entity_id),
                    "type": "location",
                    "custom_data": entity.custom_data
                })
            elif entity.entity_type == "character":
                characters.append({
                    "id": entity_id,
                    "name": entity.custom_data.get("name", entity_id),
                    "type": "character",
                    "location": entity.location,
                    "custom_data": entity.custom_data,
                    "memory_count": len(entity.memories),
                    "relationship_count": len(entity.relationships)
                })
        
        # Get recent subtext
        recent_subtext = self.state_manager.get_subtext(limit=20)
        
        return {
            "locations": locations,
            "characters": characters,
            "total_entities": len(all_entities),
            "recent_subtext": [
                {
                    "agent_id": entry.agent_id,
                    "text": entry.subtext,
                    "timestamp": entry.timestamp.isoformat()
                }
                for entry in recent_subtext
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        entity = self.state_manager.get_entity(entity_id)
        
        if not entity:
            return {"error": f"Entity {entity_id} not found"}
        
        details = entity.to_dict()
        
        # Add subtext history
        subtext_history = self.state_manager.get_subtext(agent_id=entity_id, limit=10)
        details["subtext_history"] = [
            {
                "text": entry.subtext,
                "timestamp": entry.timestamp.isoformat()
            }
            for entry in subtext_history
        ]
        
        # Add recent events
        recent_events = self.state_manager.get_recent_events(limit=10)
        details["recent_events"] = [
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "changes": event.changes
            }
            for event in recent_events
        ]
        
        return details
    
    async def update_entity_state(self, entity_id: str, changes: Dict[str, Any]):
        """Update an entity's state and broadcast the change"""
        try:
            # Create StateUpdate object
            update = StateUpdate(entity_id=entity_id, changes=changes)
            
            # Update in state manager
            self.state_manager.update_entity(update)
            
            # Publish event
            event = StateUpdateEvent(
                source="state_service",
                entity_id=entity_id,
                changes=changes
            )
            await event_bus.publish(event)
            
            logger.info("Entity state updated", entity_id=entity_id, changes=changes)
            
        except Exception as e:
            logger.error("Failed to update entity state", 
                        entity_id=entity_id, error=str(e))
            raise
    
    async def simulate_world_activity(self):
        """Simulate world activity for demo purposes"""
        characters = ["clara_001", "elder_001", "merchant_001"]
        locations = ["village_square", "forest_path", "elder_tree", "market"]
        moods = ["happy", "curious", "contemplative", "excited", "worried"]
        
        while True:
            try:
                # Random character movement
                import random
                if random.random() < 0.3:  # 30% chance
                    char_id = random.choice(characters)
                    new_location = random.choice(locations)
                    
                    await self.update_entity_state(
                        char_id,
                        {"location": new_location}
                    )
                    
                    # Add subtext about movement
                    entity = self.state_manager.get_entity(char_id)
                    if entity:
                        name = entity.custom_data.get("name", char_id)
                        self.state_manager.add_subtext(
                            char_id,
                            f"*{name} walks to the {new_location.replace('_', ' ')}*"
                        )
                
                # Random mood changes
                if random.random() < 0.2:  # 20% chance
                    char_id = random.choice(characters)
                    new_mood = random.choice(moods)
                    
                    entity = self.state_manager.get_entity(char_id)
                    if entity:
                        attributes = entity.custom_data.copy()
                        attributes["mood"] = new_mood
                        await self.update_entity_state(
                            char_id,
                            {"custom_data": attributes}
                        )
                
                await asyncio.sleep(5)  # Activity every 5 seconds
                
            except Exception as e:
                logger.error("Error in world simulation", error=str(e))
                await asyncio.sleep(10)


# Global instance
state_service = StateService() 