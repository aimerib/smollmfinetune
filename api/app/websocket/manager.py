"""
WebSocket Connection Manager

Handles WebSocket connections, message broadcasting, and connection lifecycle.
"""

import asyncio
import json
import uuid
from typing import Dict, Set, Optional, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
import structlog

logger = structlog.get_logger()


class ConnectionInfo:
    """Information about a WebSocket connection"""
    
    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.subscriptions: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
    
    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if connection is stale"""
        return (datetime.utcnow() - self.last_heartbeat).total_seconds() > timeout_seconds


class WebSocketManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        client_id = str(uuid.uuid4())
        
        async with self._lock:
            connection = ConnectionInfo(websocket, client_id)
            self.active_connections[client_id] = connection
            
        logger.info("WebSocket connected", client_id=client_id)
        
        # Send welcome message
        await self.send_personal_message(
            client_id,
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return client_id
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket connection"""
        async with self._lock:
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                
                # Remove from all topic subscriptions
                for topic in connection.subscriptions:
                    if topic in self.topic_subscribers:
                        self.topic_subscribers[topic].discard(client_id)
                
                # Remove connection
                del self.active_connections[client_id]
                
        logger.info("WebSocket disconnected", client_id=client_id)
    
    async def disconnect_all(self):
        """Disconnect all active connections"""
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            await self.disconnect(client_id)
    
    async def send_personal_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.error("Failed to send message", client_id=client_id, error=str(e))
                await self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        disconnected_clients = []
        
        for client_id, connection in self.active_connections.items():
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.error("Failed to broadcast", client_id=client_id, error=str(e))
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast a message to all clients subscribed to a topic"""
        if topic not in self.topic_subscribers:
            return
        
        disconnected_clients = []
        
        for client_id in self.topic_subscribers[topic]:
            if client_id in self.active_connections:
                try:
                    await self.send_personal_message(client_id, message)
                except Exception as e:
                    logger.error("Failed to send to topic subscriber", 
                               client_id=client_id, topic=topic, error=str(e))
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def subscribe_to_topic(self, client_id: str, topic: str):
        """Subscribe a client to a topic"""
        async with self._lock:
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                connection.subscriptions.add(topic)
                
                if topic not in self.topic_subscribers:
                    self.topic_subscribers[topic] = set()
                self.topic_subscribers[topic].add(client_id)
                
                logger.info("Client subscribed to topic", client_id=client_id, topic=topic)
    
    async def unsubscribe_from_topic(self, client_id: str, topic: str):
        """Unsubscribe a client from a topic"""
        async with self._lock:
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                connection.subscriptions.discard(topic)
                
                if topic in self.topic_subscribers:
                    self.topic_subscribers[topic].discard(client_id)
                
                logger.info("Client unsubscribed from topic", client_id=client_id, topic=topic)
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "ping":
            # Handle heartbeat
            if client_id in self.active_connections:
                self.active_connections[client_id].update_heartbeat()
            await self.send_personal_message(client_id, {"type": "pong"})
            
        elif message_type == "subscribe":
            # Subscribe to topics
            topics = message.get("topics", [])
            for topic in topics:
                await self.subscribe_to_topic(client_id, topic)
            
        elif message_type == "unsubscribe":
            # Unsubscribe from topics
            topics = message.get("topics", [])
            for topic in topics:
                await self.unsubscribe_from_topic(client_id, topic)
        
        else:
            logger.warning("Unknown message type", client_id=client_id, type=message_type)
    
    async def heartbeat_loop(self):
        """Periodic heartbeat check to clean up stale connections"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                stale_clients = []
                for client_id, connection in self.active_connections.items():
                    if connection.is_stale():
                        stale_clients.append(client_id)
                
                for client_id in stale_clients:
                    logger.info("Removing stale connection", client_id=client_id)
                    await self.disconnect(client_id)
                    
            except Exception as e:
                logger.error("Heartbeat loop error", error=str(e))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections"""
        return {
            "total_connections": len(self.active_connections),
            "topics": {
                topic: len(subscribers) 
                for topic, subscribers in self.topic_subscribers.items()
            },
            "connections": [
                {
                    "client_id": client_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "last_heartbeat": conn.last_heartbeat.isoformat(),
                    "subscriptions": list(conn.subscriptions)
                }
                for client_id, conn in self.active_connections.items()
            ]
        }

# Global instance
websocket_manager = WebSocketManager() 