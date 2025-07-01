"""
Director's View WebSocket Endpoint

Main WebSocket endpoint for the Director's View real-time interface.
"""

import json
from datetime import datetime
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import structlog

from app.websocket.manager import websocket_manager
from app.services.event_bus import event_bus, EventType
from app.services.state_service import state_service
from app.services.memory_service import memory_service
from app.services.emotion_service import emotion_service

logger = structlog.get_logger()


async def director_websocket(websocket: WebSocket):
    """Handle Director's View WebSocket connections"""
    client_id = None
    
    try:
        # Accept connection
        client_id = await websocket_manager.connect(websocket)
        logger.info("Director's View client connected", client_id=client_id)
        
        # Subscribe to all event types by default
        await websocket_manager.subscribe_to_topic(client_id, "state_updates")
        await websocket_manager.subscribe_to_topic(client_id, "memory_events")
        await websocket_manager.subscribe_to_topic(client_id, "emotion_events")
        await websocket_manager.subscribe_to_topic(client_id, "metrics")
        
        # Send initial state snapshot
        await send_initial_state(client_id)
        
        # Event handlers
        async def handle_state_update(event):
            """Forward state updates to WebSocket"""
            await websocket_manager.broadcast_to_topic(
                "state_updates",
                {
                    "type": "state_update",
                    **event.to_dict()
                }
            )
        
        async def handle_memory_event(event):
            """Forward memory events to WebSocket"""
            await websocket_manager.broadcast_to_topic(
                "memory_events",
                {
                    "type": "memory_formed",
                    **event.to_dict()
                }
            )
        
        async def handle_emotion_event(event):
            """Forward emotion events to WebSocket"""
            await websocket_manager.broadcast_to_topic(
                "emotion_events",
                {
                    "type": "emotion_changed",
                    **event.to_dict()
                }
            )
        
        async def handle_metrics_event(event):
            """Forward metrics events to WebSocket"""
            await websocket_manager.broadcast_to_topic(
                "metrics",
                {
                    "type": "triple_head_metrics",
                    **event.to_dict()
                }
            )
        
        # Subscribe event handlers
        event_bus.subscribe(EventType.STATE_UPDATE, handle_state_update)
        event_bus.subscribe(EventType.MEMORY_FORMED, handle_memory_event)
        event_bus.subscribe(EventType.EMOTION_CHANGED, handle_emotion_event)
        event_bus.subscribe(EventType.TRIPLE_HEAD_METRICS, handle_metrics_event)
        
        # Main message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()
                
                # Handle message
                await handle_director_message(client_id, data)
                
            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(
                    client_id,
                    {
                        "type": "error",
                        "message": "Invalid JSON"
                    }
                )
                
    except WebSocketDisconnect:
        logger.info("Director's View client disconnected", client_id=client_id)
        
    except Exception as e:
        logger.error("WebSocket error", client_id=client_id, error=str(e))
        
    finally:
        # Clean up
        if client_id:
            await websocket_manager.disconnect(client_id)
        
        # Unsubscribe event handlers
        try:
            event_bus.unsubscribe(EventType.STATE_UPDATE, handle_state_update)
            event_bus.unsubscribe(EventType.MEMORY_FORMED, handle_memory_event)
            event_bus.unsubscribe(EventType.EMOTION_CHANGED, handle_emotion_event)
            event_bus.unsubscribe(EventType.TRIPLE_HEAD_METRICS, handle_metrics_event)
        except:
            pass


async def send_initial_state(client_id: str):
    """Send initial world state snapshot to client"""
    try:
        # Get current state
        world_state = await state_service.get_world_snapshot()
        
        # Send snapshot
        await websocket_manager.send_personal_message(
            client_id,
            {
                "type": "initial_state",
                "data": {
                    "world": world_state,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        
        logger.info("Sent initial state to client", client_id=client_id)
        
    except Exception as e:
        logger.error("Failed to send initial state", 
                    client_id=client_id, error=str(e))


async def handle_director_message(client_id: str, message: Dict[str, Any]):
    """Handle incoming message from Director's View client"""
    message_type = message.get("type")
    
    if message_type == "get_entity_details":
        # Get detailed entity information
        entity_id = message.get("entity_id")
        if entity_id:
            entity_details = await state_service.get_entity_details(entity_id)
            await websocket_manager.send_personal_message(
                client_id,
                {
                    "type": "entity_details",
                    "data": entity_details
                }
            )
    
    elif message_type == "get_memory_timeline":
        # Get memory timeline for character
        character_id = message.get("character_id")
        limit = message.get("limit", 100)
        if character_id:
            memories = await memory_service.get_character_memories(
                character_id, limit=limit
            )
            await websocket_manager.send_personal_message(
                client_id,
                {
                    "type": "memory_timeline",
                    "data": {
                        "character_id": character_id,
                        "memories": memories
                    }
                }
            )
    
    elif message_type == "get_emotion_state":
        # Get current emotional state
        character_id = message.get("character_id")
        if character_id:
            emotion_state = await emotion_service.get_character_emotions(character_id)
            await websocket_manager.send_personal_message(
                client_id,
                {
                    "type": "emotion_state",
                    "data": {
                        "character_id": character_id,
                        "emotions": emotion_state
                    }
                }
            )
    
    elif message_type == "set_focus":
        # Set focus on specific entities
        entity_ids = message.get("entity_ids", [])
        logger.info("Client set focus", client_id=client_id, entities=entity_ids)
        # Store focus in connection metadata for filtering
        if client_id in websocket_manager.active_connections:
            websocket_manager.active_connections[client_id].metadata["focus"] = entity_ids
    
    elif message_type == "filter_update":
        # Update event filters
        filters = message.get("filters", {})
        logger.info("Client updated filters", client_id=client_id, filters=filters)
        # Store filters in connection metadata
        if client_id in websocket_manager.active_connections:
            websocket_manager.active_connections[client_id].metadata["filters"] = filters
    
    else:
        # Let the connection manager handle standard messages
        await websocket_manager.handle_message(client_id, message) 