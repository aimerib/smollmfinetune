"""
WebSocket Relationship Director

Manages real-time relationship updates and visualization state for Director's View
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio
import logging
import json

from fastapi import WebSocket, WebSocketDisconnect
from api.app.services.directors_view_service import DirectorsViewRelationshipService
from api.app.services.event_bus import EventBus
from narrative_engine.state_manager import StateManager
from narrative_engine.relationship_manager import RelationshipManager


logger = logging.getLogger(__name__)


class RelationshipDirectorWebSocket:
    """WebSocket handler for relationship visualization"""
    
    def __init__(
        self,
        state_manager: StateManager,
        relationship_manager: RelationshipManager,
        event_bus: EventBus
    ):
        self.state_manager = state_manager
        self.relationship_manager = relationship_manager
        self.event_bus = event_bus
        self.service = DirectorsViewRelationshipService(
            state_manager, 
            relationship_manager
        )
        self.active_connections: Set[WebSocket] = set()
        self._subscription_id: Optional[str] = None
        
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Send initial relationship graph data
        initial_data = await self.service.get_relationship_graph_data()
        await self._send_to_client(websocket, {
            "type": "relationship_graph_init",
            "data": initial_data
        })
        
        # Subscribe to relationship events if not already subscribed
        if not self._subscription_id:
            self._subscription_id = await self.event_bus.subscribe(
                "relationship_update",
                self._handle_relationship_event
            )
            
        logger.info("Relationship WebSocket connected")
        
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.discard(websocket)
        
        # Unsubscribe if no more connections
        if not self.active_connections and self._subscription_id:
            await self.event_bus.unsubscribe(self._subscription_id)
            self._subscription_id = None
            
        logger.info("Relationship WebSocket disconnected")
        
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            msg_type = message.get("type")
            data = message.get("data", {})
            
            if msg_type == "get_relationship_timeline":
                # Get timeline for specific entities
                entity_ids = data.get("entity_ids", [])
                limit = data.get("limit", 50)
                
                timeline = await self.service.get_relationship_timeline(
                    entity_ids, limit
                )
                
                await self._send_to_client(websocket, {
                    "type": "relationship_timeline",
                    "data": timeline
                })
                
            elif msg_type == "get_social_metrics":
                # Get overall social ecosystem metrics
                metrics = await self.service.get_social_metrics()
                
                await self._send_to_client(websocket, {
                    "type": "social_metrics",
                    "data": metrics
                })
                
            elif msg_type == "update_node_position":
                # Update node position from frontend drag
                entity_id = data.get("entity_id")
                position = data.get("position", {})
                
                if entity_id and position:
                    self.service._node_positions[entity_id] = position
                    
                    # Broadcast position update to all clients
                    await self._broadcast({
                        "type": "node_position_update",
                        "data": {
                            "entity_id": entity_id,
                            "position": position
                        }
                    })
                    
            elif msg_type == "filter_relationships":
                # Apply affinity filter and resend graph
                affinity_filter = data.get("affinity_filter", {"min": -1, "max": 1})
                
                # Get filtered graph data
                graph_data = await self.service.get_relationship_graph_data()
                
                # Apply filter client-side for now
                # In production, filter in service
                filtered_edges = [
                    edge for edge in graph_data["edges"]
                    if affinity_filter["min"] <= edge["affinity"] <= affinity_filter["max"]
                ]
                
                graph_data["edges"] = filtered_edges
                
                await self._send_to_client(websocket, {
                    "type": "relationship_graph_update", 
                    "data": graph_data
                })
                
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, str(e))
            
    async def _handle_relationship_event(self, event: Dict[str, Any]):
        """Handle relationship events from event bus"""
        try:
            # Process event through service
            update_event = await self.service.handle_relationship_update(event)
            
            if "error" not in update_event:
                # Broadcast to all connected clients
                await self._broadcast({
                    "type": "relationship_event",
                    "data": update_event
                })
                
                # Also send updated graph data periodically
                # (In production, batch updates for efficiency)
                if hasattr(self, '_last_graph_update'):
                    time_since_update = (
                        datetime.utcnow() - self._last_graph_update
                    ).total_seconds()
                    
                    if time_since_update > 2.0:  # Update every 2 seconds max
                        graph_data = await self.service.get_relationship_graph_data()
                        await self._broadcast({
                            "type": "relationship_graph_update",
                            "data": graph_data
                        })
                        self._last_graph_update = datetime.utcnow()
                else:
                    self._last_graph_update = datetime.utcnow()
                    
        except Exception as e:
            logger.error(f"Error handling relationship event: {e}")
            
    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = set()
        
        for websocket in self.active_connections:
            try:
                await self._send_to_client(websocket, message)
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)
                
        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)
            
    async def _send_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            raise
            
    async def _send_error(self, websocket: WebSocket, error: str):
        """Send error message to client"""
        await self._send_to_client(websocket, {
            "type": "error",
            "data": {"message": error}
        })
        

# Global instance for the WebSocket handler
relationship_director_ws: Optional[RelationshipDirectorWebSocket] = None


def get_relationship_director(
    state_manager: StateManager,
    relationship_manager: RelationshipManager,
    event_bus: EventBus
) -> RelationshipDirectorWebSocket:
    """Get or create relationship director WebSocket handler"""
    global relationship_director_ws
    
    if relationship_director_ws is None:
        relationship_director_ws = RelationshipDirectorWebSocket(
            state_manager,
            relationship_manager,
            event_bus
        )
        
    return relationship_director_ws 