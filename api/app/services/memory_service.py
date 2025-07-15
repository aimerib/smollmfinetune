"""
Memory Service

Manages character memories and memory formation events.
"""

import asyncio
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from app.services.event_bus import event_bus, MemoryFormationEvent

logger = structlog.get_logger()


class MemoryService:
    """Service for managing character memories"""
    
    def __init__(self):
        self.memories: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the memory service"""
        try:
            # Create demo memories
            await self._create_demo_memories()
            
            # Start memory simulation
            asyncio.create_task(self.simulate_memory_formation())
            
            self._initialized = True
            logger.info("Memory service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize memory service", error=str(e))
            raise
    
    async def _create_demo_memories(self):
        """Create demo memories for testing"""
        demo_memories = {
            "clara_001": [
                {
                    "id": "mem_001",
                    "content": "The Elder Tree whispered my name when I first approached it",
                    "importance": 0.9,
                    "emotional_valence": 0.7,
                    "memory_type": "episodic",
                    "timestamp": datetime.utcnow().isoformat(),
                    "surprise_score": 0.8
                },
                {
                    "id": "mem_002",
                    "content": "The merchant told me about lands beyond the mountains",
                    "importance": 0.6,
                    "emotional_valence": 0.5,
                    "memory_type": "semantic",
                    "timestamp": datetime.utcnow().isoformat(),
                    "surprise_score": 0.4
                }
            ],
            "elder_001": [
                {
                    "id": "mem_003",
                    "content": "Clara reminds me of myself when I was young",
                    "importance": 0.7,
                    "emotional_valence": 0.6,
                    "memory_type": "emotional",
                    "timestamp": datetime.utcnow().isoformat(),
                    "surprise_score": 0.3
                }
            ]
        }
        
        self.memories = demo_memories
    
    async def get_character_memories(self, character_id: str, 
                                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get memories for a specific character"""
        character_memories = self.memories.get(character_id, [])
        
        # Sort by timestamp (newest first)
        sorted_memories = sorted(
            character_memories,
            key=lambda m: m.get("timestamp", ""),
            reverse=True
        )
        
        if limit:
            sorted_memories = sorted_memories[:limit]
        
        return sorted_memories
    
    async def add_memory(self, character_id: str, memory: Dict[str, Any]):
        """Add a new memory for a character"""
        if character_id not in self.memories:
            self.memories[character_id] = []
        
        # Add ID and timestamp if not present
        if "id" not in memory:
            memory["id"] = f"mem_{datetime.utcnow().timestamp()}"
        if "timestamp" not in memory:
            memory["timestamp"] = datetime.utcnow().isoformat()
        
        self.memories[character_id].append(memory)
        
        # Publish memory formation event
        event = MemoryFormationEvent(
            source="memory_service",
            character_id=character_id,
            memory_content=memory.get("content", ""),
            importance=memory.get("importance", 0.5),
            emotional_valence=memory.get("emotional_valence", 0.0),
            memory_type=memory.get("memory_type", "episodic"),
            bubble_size=50 + (memory.get("importance", 0.5) * 50)  # Size based on importance
        )
        await event_bus.publish(event)
        
        logger.info("Memory added", 
                   character_id=character_id,
                   memory_id=memory["id"])
    
    async def simulate_memory_formation(self):
        """Simulate memory formation for demo purposes"""
        characters = ["clara_001", "elder_001", "merchant_001"]
        
        memory_templates = [
            {
                "content": "I noticed something strange about the {location}",
                "type": "episodic",
                "valence_range": (-0.2, 0.2),
                "importance_range": (0.3, 0.6)
            },
            {
                "content": "I learned that {fact}",
                "type": "semantic",
                "valence_range": (0.0, 0.3),
                "importance_range": (0.4, 0.7)
            },
            {
                "content": "I felt {emotion} when {event}",
                "type": "emotional",
                "valence_range": (-0.8, 0.8),
                "importance_range": (0.6, 0.9)
            },
            {
                "content": "I remembered how to {action}",
                "type": "procedural",
                "valence_range": (0.1, 0.4),
                "importance_range": (0.3, 0.5)
            }
        ]
        
        locations = ["village square", "forest path", "elder tree", "market"]
        facts = ["the stars guide travelers", "herbs can heal wounds", "music soothes souls"]
        emotions = ["peaceful", "curious", "nostalgic", "hopeful", "uncertain"]
        events = ["the sun set", "birds sang", "wind whispered", "someone smiled"]
        actions = ["navigate by stars", "brew healing tea", "calm anxiety"]
        
        while True:
            try:
                await asyncio.sleep(random.uniform(10, 30))  # Random interval
                
                # Pick random character and template
                character_id = random.choice(characters)
                template = random.choice(memory_templates)
                
                # Generate memory content
                content = template["content"]
                content = content.replace("{location}", random.choice(locations))
                content = content.replace("{fact}", random.choice(facts))
                content = content.replace("{emotion}", random.choice(emotions))
                content = content.replace("{event}", random.choice(events))
                content = content.replace("{action}", random.choice(actions))
                
                # Generate memory properties
                importance = random.uniform(*template["importance_range"])
                valence = random.uniform(*template["valence_range"])
                surprise = random.uniform(0.1, 0.9)
                
                # Create memory
                memory = {
                    "content": content,
                    "importance": importance,
                    "emotional_valence": valence,
                    "memory_type": template["type"],
                    "surprise_score": surprise
                }
                
                await self.add_memory(character_id, memory)
                
            except Exception as e:
                logger.error("Error in memory simulation", error=str(e))
                await asyncio.sleep(30)


# Global instance
memory_service = MemoryService() 