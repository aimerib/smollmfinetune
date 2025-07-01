"""
Event Bus for Inter-Service Communication

Provides a publish-subscribe mechanism for decoupled service communication.
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    """Types of events in the system"""
    STATE_UPDATE = "state_update"
    MEMORY_FORMED = "memory_formed"
    EMOTION_CHANGED = "emotion_changed"
    SUBTEXT_ADDED = "subtext_added"
    TRIPLE_HEAD_METRICS = "triple_head_metrics"
    CHARACTER_ACTION = "character_action"
    WORLD_EVENT = "world_event"
    MODEL_INFERENCE = "model_inference"


@dataclass
class Event:
    """Base event class"""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data
        }


@dataclass
class StateUpdateEvent(Event):
    """Entity state change event"""
    event_type: EventType = EventType.STATE_UPDATE
    entity_id: str = ""
    changes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.data = {
            "entity_id": self.entity_id,
            "changes": self.changes
        }


@dataclass
class MemoryFormationEvent(Event):
    """Memory formation event with visualization hints"""
    event_type: EventType = EventType.MEMORY_FORMED
    character_id: str = ""
    memory_content: str = ""
    importance: float = 0.5
    emotional_valence: float = 0.0
    memory_type: str = "episodic"
    bubble_color: str = ""
    bubble_size: float = 50.0
    
    def __post_init__(self):
        # Auto-generate bubble color from valence if not provided
        if not self.bubble_color:
            # Map valence to hue: -1=0 (red), 0=60 (yellow), 1=120 (green)
            hue = 60 * (self.emotional_valence + 1)
            self.bubble_color = f"hsl({hue}, 70%, 50%)"
        
        self.data = {
            "character_id": self.character_id,
            "memory_content": self.memory_content,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "memory_type": self.memory_type,
            "visualization": {
                "bubble_color": self.bubble_color,
                "bubble_size": self.bubble_size
            }
        }


@dataclass
class EmotionChangeEvent(Event):
    """Emotional state change event"""
    event_type: EventType = EventType.EMOTION_CHANGED
    character_id: str = ""
    active_emotions: Dict[str, float] = field(default_factory=dict)
    surprise_score: float = 0.0
    momentum: float = 0.0
    
    def __post_init__(self):
        self.data = {
            "character_id": self.character_id,
            "active_emotions": self.active_emotions,
            "surprise_score": self.surprise_score,
            "momentum": self.momentum
        }


@dataclass
class TripleHeadMetricsEvent(Event):
    """Triple-head model performance metrics"""
    event_type: EventType = EventType.TRIPLE_HEAD_METRICS
    character_id: str = ""
    generation_quality: float = 0.0
    control_effectiveness: float = 0.0
    memory_coherence: float = 0.0
    coordination_score: float = 0.0
    
    def __post_init__(self):
        self.data = {
            "character_id": self.character_id,
            "metrics": {
                "generation_quality": self.generation_quality,
                "control_effectiveness": self.control_effectiveness,
                "memory_coherence": self.memory_coherence,
                "coordination_score": self.coordination_score
            }
        }


class EventBus:
    """Central event bus for publish-subscribe communication"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._processor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the event processor"""
        self.running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        
    async def stop(self):
        """Stop the event processor"""
        self.running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.debug("Handler subscribed", event_type=event_type.value)
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
            logger.debug("Handler unsubscribed", event_type=event_type.value)
    
    async def publish(self, event: Event):
        """Publish an event"""
        await self.event_queue.put(event)
        logger.debug("Event published", 
                    event_type=event.event_type.value,
                    source=event.source)
    
    async def _process_events(self):
        """Process events from the queue"""
        while self.running:
            try:
                # Wait for event with timeout to allow cancellation
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                # Get subscribers for this event type
                handlers = self.subscribers.get(event.event_type, [])
                
                # Call all handlers
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error("Event handler error",
                                   event_type=event.event_type.value,
                                   error=str(e))
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Event processing error", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "running": self.running,
            "queue_size": self.event_queue.qsize(),
            "subscribers": {
                event_type.value: len(handlers)
                for event_type, handlers in self.subscribers.items()
            }
        } 