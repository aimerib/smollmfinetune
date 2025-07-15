"""
Emotion Service

Manages character emotional states and emotion change events.
"""

import asyncio
import random
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from app.services.event_bus import event_bus, EmotionChangeEvent

logger = structlog.get_logger()


class EmotionService:
    """Service for managing character emotions"""
    
    def __init__(self):
        self.emotional_states: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the emotion service"""
        try:
            # Create initial emotional states
            await self._create_initial_states()
            
            # Start emotion simulation
            asyncio.create_task(self.simulate_emotional_changes())
            
            self._initialized = True
            logger.info("Emotion service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize emotion service", error=str(e))
            raise
    
    async def _create_initial_states(self):
        """Create initial emotional states for demo characters"""
        self.emotional_states = {
            "clara_001": {
                "active_emotions": {
                    "curious": 0.8,
                    "hopeful": 0.6
                },
                "surprise_history": [0.3, 0.5, 0.4, 0.8, 0.2],
                "momentum": 0.7,
                "decay_states": {
                    "curious": {"strength": 0.8, "decay_rate": 0.05, "turns_remaining": 5},
                    "hopeful": {"strength": 0.6, "decay_rate": 0.08, "turns_remaining": 4}
                }
            },
            "elder_001": {
                "active_emotions": {
                    "wise": 0.9,
                    "contemplative": 0.7
                },
                "surprise_history": [0.1, 0.2, 0.1, 0.3, 0.2],
                "momentum": 0.3,
                "decay_states": {
                    "wise": {"strength": 0.9, "decay_rate": 0.02, "turns_remaining": 10},
                    "contemplative": {"strength": 0.7, "decay_rate": 0.06, "turns_remaining": 6}
                }
            },
            "merchant_001": {
                "active_emotions": {
                    "cheerful": 0.8,
                    "ambitious": 0.5
                },
                "surprise_history": [0.4, 0.6, 0.5, 0.3, 0.7],
                "momentum": 0.6,
                "decay_states": {
                    "cheerful": {"strength": 0.8, "decay_rate": 0.04, "turns_remaining": 7},
                    "ambitious": {"strength": 0.5, "decay_rate": 0.07, "turns_remaining": 5}
                }
            }
        }
    
    async def get_character_emotions(self, character_id: str) -> Dict[str, Any]:
        """Get current emotional state for a character"""
        state = self.emotional_states.get(character_id, {})
        
        if not state:
            # Return default state if character not found
            return {
                "active_emotions": {},
                "surprise_score": 0.5,
                "momentum": 0.5,
                "emotional_volatility": 0.0
            }
        
        # Calculate current surprise score
        surprise_history = state.get("surprise_history", [])
        avg_surprise = sum(surprise_history) / len(surprise_history) if surprise_history else 0.5
        
        # Calculate emotional volatility (standard deviation of surprise)
        if len(surprise_history) > 1:
            mean = avg_surprise
            variance = sum((x - mean) ** 2 for x in surprise_history) / len(surprise_history)
            volatility = math.sqrt(variance)
        else:
            volatility = 0.0
        
        return {
            "active_emotions": state.get("active_emotions", {}),
            "surprise_score": surprise_history[-1] if surprise_history else 0.5,
            "average_surprise": avg_surprise,
            "momentum": state.get("momentum", 0.5),
            "emotional_volatility": volatility,
            "decay_states": state.get("decay_states", {})
        }
    
    async def update_emotion(self, character_id: str, 
                           emotion: str, strength: float,
                           decay_rate: float = 0.05):
        """Update a specific emotion for a character"""
        if character_id not in self.emotional_states:
            self.emotional_states[character_id] = {
                "active_emotions": {},
                "surprise_history": [],
                "momentum": 0.5,
                "decay_states": {}
            }
        
        state = self.emotional_states[character_id]
        
        # Update active emotions
        state["active_emotions"][emotion] = strength
        
        # Update decay state
        state["decay_states"][emotion] = {
            "strength": strength,
            "decay_rate": decay_rate,
            "turns_remaining": int(strength / decay_rate)
        }
        
        # Add surprise based on change magnitude
        old_strength = state["active_emotions"].get(emotion, 0.0)
        surprise = abs(strength - old_strength)
        state["surprise_history"].append(surprise)
        
        # Keep history limited
        if len(state["surprise_history"]) > 10:
            state["surprise_history"].pop(0)
        
        # Update momentum
        state["momentum"] = 0.7 * state["momentum"] + 0.3 * surprise
        
        # Publish event
        event = EmotionChangeEvent(
            source="emotion_service",
            character_id=character_id,
            active_emotions=state["active_emotions"],
            surprise_score=surprise,
            momentum=state["momentum"]
        )
        await event_bus.publish(event)
    
    async def decay_emotions(self, character_id: str):
        """Apply decay to all emotions for a character"""
        if character_id not in self.emotional_states:
            return
        
        state = self.emotional_states[character_id]
        decay_states = state.get("decay_states", {})
        active_emotions = state.get("active_emotions", {})
        
        emotions_to_remove = []
        
        for emotion, decay_info in decay_states.items():
            # Apply decay
            new_strength = decay_info["strength"] - decay_info["decay_rate"]
            
            if new_strength <= 0.1:  # Remove emotion if too weak
                emotions_to_remove.append(emotion)
            else:
                # Update strength
                decay_info["strength"] = new_strength
                decay_info["turns_remaining"] -= 1
                active_emotions[emotion] = new_strength
        
        # Remove decayed emotions
        for emotion in emotions_to_remove:
            decay_states.pop(emotion, None)
            active_emotions.pop(emotion, None)
    
    async def simulate_emotional_changes(self):
        """Simulate emotional changes for demo purposes"""
        characters = ["clara_001", "elder_001", "merchant_001"]
        
        emotions = {
            "positive": ["happy", "excited", "peaceful", "confident", "grateful"],
            "negative": ["worried", "sad", "frustrated", "nervous", "confused"],
            "neutral": ["curious", "contemplative", "focused", "patient", "observant"]
        }
        
        while True:
            try:
                await asyncio.sleep(random.uniform(8, 20))  # Random interval
                
                # Pick random character
                character_id = random.choice(characters)
                
                # Decay existing emotions
                await self.decay_emotions(character_id)
                
                # Sometimes add new emotion
                if random.random() < 0.6:  # 60% chance
                    # Pick emotion category based on current state
                    state = self.emotional_states.get(character_id, {})
                    current_momentum = state.get("momentum", 0.5)
                    
                    # Higher momentum = more likely to have strong emotions
                    if current_momentum > 0.7:
                        category = random.choice(["positive", "negative"])
                    else:
                        category = "neutral"
                    
                    emotion = random.choice(emotions[category])
                    strength = random.uniform(0.3, 0.9)
                    decay_rate = random.uniform(0.03, 0.1)
                    
                    await self.update_emotion(character_id, emotion, strength, decay_rate)
                
            except Exception as e:
                logger.error("Error in emotion simulation", error=str(e))
                await asyncio.sleep(30)


# Global instance
emotion_service = EmotionService() 