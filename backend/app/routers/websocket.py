from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}  # session_id -> [connection_ids]

    async def connect(self, websocket: WebSocket, connection_id: str, session_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(connection_id)
        
        logger.info(f"WebSocket connection {connection_id} established for session {session_id}")

    def disconnect(self, connection_id: str, session_id: str = None):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if session_id and session_id in self.session_connections:
            if connection_id in self.session_connections[session_id]:
                self.session_connections[session_id].remove(connection_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        logger.info(f"WebSocket connection {connection_id} disconnected from session {session_id}")

    async def send_personal_message(self, message: str, connection_id: str):
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message)

    async def send_to_session(self, message: str, session_id: str):
        """Send a message to all connections in a session"""
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                await self.send_personal_message(message, connection_id)

    async def broadcast(self, message: str):
        """Send a message to all active connections"""
        for connection_id in self.active_connections:
            await self.send_personal_message(message, connection_id)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/chat/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    connection_id = f"chat_{session_id}_{id(websocket)}"
    await manager.connect(websocket, connection_id, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "chat")
                
                if message_type == "chat":
                    # Handle chat message
                    response = {
                        "type": "chat_response",
                        "message": f"Echo: {message.get('message', '')}",
                        "session_id": session_id,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await manager.send_to_session(json.dumps(response), session_id)
                
                elif message_type == "typing":
                    # Handle typing indicators
                    response = {
                        "type": "typing",
                        "user": message.get("user", "unknown"),
                        "session_id": session_id
                    }
                    await manager.send_to_session(json.dumps(response), session_id)
                
            except json.JSONDecodeError:
                # Handle plain text messages
                response = {
                    "type": "echo",
                    "message": data,
                    "session_id": session_id
                }
                await manager.send_to_session(json.dumps(response), session_id)
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id, session_id)
        logger.info(f"Client {connection_id} disconnected from chat session {session_id}")

@router.websocket("/ws/directors-chair/{session_id}")
async def websocket_directors_chair_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for Director's Chair monitoring"""
    connection_id = f"directors_{session_id}_{id(websocket)}"
    await manager.connect(websocket, connection_id, session_id)
    
    try:
        while True:
            # Receive commands from Director's Chair client
            data = await websocket.receive_text()
            
            try:
                command = json.loads(data)
                command_type = command.get("type", "monitor")
                
                if command_type == "monitor":
                    # Send monitoring data
                    monitoring_data = {
                        "type": "monitoring_update",
                        "emotional_state": {
                            "valence": 0.6,
                            "arousal": 0.4,
                            "dominance": 0.5
                        },
                        "relationships": [
                            {"entity": "Player", "relationship": "friendly", "strength": 0.7}
                        ],
                        "memory_formation": {
                            "recent_memories": 3,
                            "importance_threshold": 0.6
                        },
                        "session_id": session_id,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await manager.send_personal_message(json.dumps(monitoring_data), connection_id)
                
                elif command_type == "intervention":
                    # Handle Director's Chair interventions
                    intervention = {
                        "type": "intervention_applied",
                        "intervention": command.get("intervention", {}),
                        "session_id": session_id,
                        "success": True
                    }
                    await manager.send_personal_message(json.dumps(intervention), connection_id)
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from Directors Chair: {data}")
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id, session_id)
        logger.info(f"Directors Chair {connection_id} disconnected from session {session_id}")

@router.websocket("/ws/training/{job_id}")
async def websocket_training_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training updates"""
    connection_id = f"training_{job_id}_{id(websocket)}"
    await manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Send training progress updates
            await asyncio.sleep(1)  # Simulate training progress
            
            progress_update = {
                "type": "training_progress",
                "job_id": job_id,
                "progress": 0.1,  # Mock progress
                "loss": 2.5,
                "step": 100,
                "eta": "5 minutes",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await manager.send_personal_message(json.dumps(progress_update), connection_id)
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        logger.info(f"Training monitor {connection_id} disconnected from job {job_id}")

# Health endpoint for WebSocket service
@router.get("/websocket/health")
async def websocket_health():
    """Get WebSocket service health status"""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "active_sessions": len(manager.session_connections),
        "service": "websocket"
    } 