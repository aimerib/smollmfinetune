"""
Digital Ecology Engine - Relationship Manager

Implements the RelationshipManager that observes agent interactions and updates
relationship dynamics using the triple-head architecture for sophisticated analysis.
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pydantic import BaseModel, Field

from .state_manager import StateManager, EntityState, StateUpdate, EventLog
from .agent import Action, SpeakToAction

logger = logging.getLogger(__name__)


class RelationshipStatus(str, Enum):
    """Possible relationship statuses"""
    STRANGER = "Stranger"
    ACQUAINTANCE = "Acquaintance"
    FRIEND = "Friend"
    CLOSE_FRIEND = "Close_Friend"
    RIVAL = "Rival"
    ENEMY = "Enemy"
    NEUTRAL = "Neutral"
    ALLY = "Ally"
    ROMANTIC = "Romantic"
    FAMILY = "Family"
    MENTOR = "Mentor"
    STUDENT = "Student"
    COLLEAGUE = "Colleague"


@dataclass
class EnhancedRelationship:
    """
    Enhanced relationship model with emotional history and memory significance.
    Tracks the dynamic evolution of relationships between agents.
    """
    affinity: float = 0.0  # -1.0 to 1.0 scale
    status: str = RelationshipStatus.STRANGER
    emotional_history: List[str] = field(default_factory=list)
    memory_significance: float = 0.0  # 0.0 to 1.0 scale
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.last_interaction:
            data['last_interaction'] = self.last_interaction.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedRelationship':
        """Create from dictionary"""
        if 'last_interaction' in data and isinstance(data['last_interaction'], str):
            data['last_interaction'] = datetime.fromisoformat(data['last_interaction'])
        return cls(**data)
    
    def update_from_analysis(self, analysis: Dict[str, Any]) -> None:
        """Update relationship based on triple-head analysis"""
        # Update from generation analysis
        if 'generation_analysis' in analysis:
            gen_analysis = analysis['generation_analysis']
            if 'affinity_change' in gen_analysis:
                self.affinity += gen_analysis['affinity_change']
                # Clamp affinity to valid range
                self.affinity = max(-1.0, min(1.0, self.affinity))
            
            if 'status_change' in gen_analysis:
                self.status = gen_analysis['status_change']
        
        # Update from control analysis
        if 'control_analysis' in analysis:
            control_analysis = analysis['control_analysis']
            if 'emotional_state' in control_analysis:
                # Add emotions to history (keep last 10)
                new_emotions = list(control_analysis['emotional_state'].keys())
                self.emotional_history.extend(new_emotions)
                self.emotional_history = self.emotional_history[-10:]  # Keep last 10
        
        # Update from memory analysis
        if 'memory_analysis' in analysis:
            memory_analysis = analysis['memory_analysis']
            if 'significance_score' in memory_analysis:
                # For new relationships, use the significance directly
                # For existing relationships, use weighted average
                if self.interaction_count == 0:
                    self.memory_significance = memory_analysis['significance_score']
                else:
                    # Update memory significance using weighted average
                    current_weight = 0.3  # Weight for new significance
                    self.memory_significance = (
                        current_weight * memory_analysis['significance_score'] +
                        (1 - current_weight) * self.memory_significance
                    )
        
        # Update interaction metadata
        self.last_interaction = datetime.now()
        self.interaction_count += 1


class RelationshipManager:
    """
    Manages dynamic relationships between agents using triple-head analysis.
    Subscribes to StateManager events and updates relationship states accordingly.
    """
    
    def __init__(self, state_manager: StateManager, narrative_model=None):
        """
        Initialize RelationshipManager.
        
        Args:
            state_manager: The runtime state manager
            narrative_model: Triple-head NarrativeLLM model for analysis
        """
        self.state_manager = state_manager
        self.narrative_model = narrative_model
        self.is_monitoring = False
        self._event_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized RelationshipManager")
    
    async def start_monitoring(self) -> None:
        """Start monitoring StateManager events for relationship updates"""
        if self.is_monitoring:
            logger.warning("RelationshipManager is already monitoring")
            return
        
        self.is_monitoring = True
        
        # Subscribe to state manager events
        self.state_manager.subscribe_to_events(self._handle_state_event)
        
        logger.info("RelationshipManager started monitoring events")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring events"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
        
        logger.info("RelationshipManager stopped monitoring")
    
    async def _handle_state_event(self, event_log: EventLog) -> None:
        """Handle incoming state events and process relationship changes"""
        try:
            # Only process SpeakToAction events for now
            if event_log.event_type == "SpeakToAction":
                await self._process_speak_action_event(event_log)
        except Exception as e:
            logger.error(f"Error handling state event: {e}")
    
    async def _process_speak_action_event(self, event_log: EventLog) -> None:
        """Process a SpeakToAction event for relationship analysis"""
        details = event_log.details
        if not details:
            return
        
        speaker_id = event_log.entity_id
        target_id = details.get("target")
        message = details.get("message")
        
        if not all([speaker_id, target_id, message]):
            logger.warning(f"Incomplete SpeakToAction event: {details}")
            return
        
        # Get speaker and target entities
        speaker_entity = self.state_manager.get_entity(speaker_id)
        target_entity = self.state_manager.get_entity(target_id)
        
        if not speaker_entity or not target_entity:
            logger.warning(f"Could not find entities: speaker={speaker_id}, target={target_id}")
            return
        
        # Build interaction event for analysis
        interaction_event = {
            'speaker_id': speaker_id,
            'target_id': target_id,
            'message': message,
            'speaker_personality': speaker_entity.custom_data.get('personality', {}),
            'target_personality': target_entity.custom_data.get('personality', {})
        }
        
        # Process the interaction
        await self.process_interaction(interaction_event)
    
    async def analyze_interaction_triple_head(self, interaction_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze interaction using triple-head architecture.
        
        Args:
            interaction_event: Dict containing speaker_id, target_id, message, personalities
            
        Returns:
            Dict with analysis from all three heads
        """
        if not self.narrative_model:
            # Fallback analysis without model
            return self._fallback_analysis(interaction_event)
        
        try:
            # Call narrative model's analyze_interaction method
            analysis = await self.narrative_model.analyze_interaction(interaction_event)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in triple-head analysis: {e}")
            return self._fallback_analysis(interaction_event)
    
    def _fallback_analysis(self, interaction_event: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when no model is available"""
        message = interaction_event.get('message', '').lower()
        
        # Simple heuristic analysis
        affinity_change = 0.0
        emotions = []
        significance = 0.3
        
        # Positive keywords
        if any(word in message for word in ['thank', 'great', 'good', 'help', 'nice']):
            affinity_change = 0.1
            emotions = ['pleased', 'grateful']
            significance = 0.5
        
        # Negative keywords
        elif any(word in message for word in ['fool', 'wrong', 'bad', 'hate', 'stupid']):
            affinity_change = -0.2
            emotions = ['annoyed', 'frustrated']
            significance = 0.7
        
        return {
            'generation_analysis': {
                'semantic_content': f"Simple analysis of: {message[:50]}...",
                'social_intent': 'communication',
                'affinity_change': affinity_change,
                'status_change': 'acquaintance'
            },
            'control_analysis': {
                'emotional_tokens': [f'<mood_{emotion}>' for emotion in emotions],
                'emotional_state': {emotion: 0.6 for emotion in emotions},
                'mood_indicators': emotions
            },
            'memory_analysis': {
                'significance_score': significance,
                'importance': significance,
                'emotional_impact': abs(affinity_change),
                'memory_formation': {
                    'content': f"Interaction: {message[:30]}...",
                    'tags': ['conversation'],
                    'persistence': significance
                }
            }
        }
    
    async def process_interaction(self, interaction_event: Dict[str, Any]) -> None:
        """
        Process an interaction event and update relationships.
        
        Args:
            interaction_event: Dict containing interaction details
        """
        speaker_id = interaction_event['speaker_id']
        target_id = interaction_event['target_id']
        
        # Analyze the interaction using triple-head architecture
        analysis = await self.analyze_interaction_triple_head(interaction_event)
        
        # Update relationships for both entities
        await self._update_entity_relationship(speaker_id, target_id, analysis)
        await self._update_entity_relationship(target_id, speaker_id, analysis, reverse=True)
        
        # Add memory formation if significant
        if analysis.get('memory_analysis', {}).get('memory_formation'):
            memory_data = analysis['memory_analysis']['memory_formation'].copy()
            memory_data['timestamp'] = datetime.now().isoformat()
            
            # Add memory to target (recipient of interaction)
            self.state_manager.add_memory_to_character(target_id, memory_data)
        
        logger.debug(f"Processed interaction between {speaker_id} and {target_id}")
    
    async def _update_entity_relationship(
        self, 
        entity_id: str, 
        other_id: str, 
        analysis: Dict[str, Any],
        reverse: bool = False
    ) -> None:
        """Update relationship data for an entity"""
        entity = self.state_manager.get_entity(entity_id)
        if not entity:
            return
        
        # Get current relationships
        relationships = entity.relationships.copy()
        
        # Get or create relationship
        if other_id in relationships:
            # Load existing relationship
            rel_data = relationships[other_id]
            if isinstance(rel_data, dict):
                # Convert from dict if needed
                relationship = EnhancedRelationship.from_dict(rel_data)
            else:
                # Create new enhanced relationship
                relationship = EnhancedRelationship()
        else:
            # Create new relationship
            relationship = EnhancedRelationship()
        
        # Apply analysis to relationship (adjust for reverse perspective)
        analysis_copy = analysis.copy()
        if reverse and 'generation_analysis' in analysis_copy:
            # Reverse perspective: reduce affinity change magnitude for observer
            gen_analysis = analysis_copy['generation_analysis']
            if 'affinity_change' in gen_analysis:
                gen_analysis['affinity_change'] *= 0.5  # Observer gets half the impact
        
        relationship.update_from_analysis(analysis_copy)
        
        # Update relationships dict
        relationships[other_id] = relationship.to_dict()
        
        # Update entity in state manager
        update = StateUpdate(
            entity_id=entity_id,
            changes={"relationships": relationships}
        )
        self.state_manager.update_entity(update) 