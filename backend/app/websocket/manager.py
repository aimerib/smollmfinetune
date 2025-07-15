"""
Generic WebSocket Manager for Backend Services

Provides a reusable WebSocket connection manager that can be used across
different services for real-time communication.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Generic WebSocket connection manager
    
    Manages WebSocket connections for real-time communication across services.
    Supports session-based grouping and broadcast messaging.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}  # session_id -> [connection_ids]
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}  # connection_id -> metadata
    
    async def connect(self, websocket: WebSocket, connection_id: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Accept a new WebSocket connection
        
        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
            session_id: Optional session identifier for grouping connections
            metadata: Optional metadata about the connection
        """
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if metadata:
            self.connection_metadata[connection_id] = metadata
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(connection_id)
        
        logger.info(f"WebSocket connection {connection_id} established for session {session_id}")
    
    def disconnect(self, connection_id: str, session_id: Optional[str] = None):
        """
        Remove a WebSocket connection
        
        Args:
            connection_id: The connection to remove
            session_id: Optional session identifier for cleanup
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        if session_id and session_id in self.session_connections:
            if connection_id in self.session_connections[session_id]:
                self.session_connections[session_id].remove(connection_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        logger.info(f"WebSocket connection {connection_id} disconnected from session {session_id}")
    
    async def send_personal_message(self, message: Any, connection_id: str):
        """
        Send a message to a specific connection
        
        Args:
            message: The message to send (dict, str, or bytes)
            connection_id: Target connection identifier
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
            return False
        
        websocket = self.active_connections[connection_id]
        
        try:
            if isinstance(message, dict):
                await websocket.send_json(message)
            elif isinstance(message, str):
                await websocket.send_text(message)
            elif isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                # Convert to JSON string if it's not a supported type
                await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            # Clean up the broken connection
            self.disconnect(connection_id)
            return False
    
    async def send_to_session(self, message: Any, session_id: str):
        """
        Send a message to all connections in a session
        
        Args:
            message: The message to send
            session_id: Target session identifier
        """
        if session_id not in self.session_connections:
            logger.warning(f"Attempted to send message to non-existent session: {session_id}")
            return
        
        connection_ids = self.session_connections[session_id].copy()
        successful_sends = 0
        
        for connection_id in connection_ids:
            success = await self.send_personal_message(message, connection_id)
            if success:
                successful_sends += 1
        
        logger.debug(f"Sent message to {successful_sends}/{len(connection_ids)} connections in session {session_id}")
    
    async def broadcast(self, message: Any, exclude_connections: Optional[Set[str]] = None):
        """
        Send a message to all active connections
        
        Args:
            message: The message to send
            exclude_connections: Set of connection IDs to exclude from broadcast
        """
        exclude_connections = exclude_connections or set()
        connection_ids = [cid for cid in self.active_connections.keys() if cid not in exclude_connections]
        successful_sends = 0
        
        for connection_id in connection_ids:
            success = await self.send_personal_message(message, connection_id)
            if success:
                successful_sends += 1
        
        logger.debug(f"Broadcast message to {successful_sends}/{len(connection_ids)} connections")
    
    async def send_audio_chunk(self, data: Dict[str, Any]):
        """
        Send audio chunk data (compatibility method for conversation manager)
        
        Args:
            data: Audio chunk data with metadata
        """
        # This method provides compatibility with the conversation manager interface
        message = {
            "type": "audio_chunk",
            "data": data
        }
        
        # If there's a session context in the data, send to that session
        session_id = data.get("session_id")
        if session_id:
            await self.send_to_session(message, session_id)
        else:
            # Otherwise broadcast to all connections
            await self.broadcast(message)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_session_count(self) -> int:
        """Get total number of active sessions"""
        return len(self.session_connections)
    
    def get_session_connections(self, session_id: str) -> List[str]:
        """Get connection IDs for a specific session"""
        return self.session_connections.get(session_id, [])
    
    def get_connection_metadata(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific connection"""
        return self.connection_metadata.get(connection_id)
    
    def is_connection_active(self, connection_id: str) -> bool:
        """Check if a connection is active"""
        return connection_id in self.active_connections
    
    def cleanup_session(self, session_id: str):
        """
        Clean up all connections for a session
        
        Args:
            session_id: Session to clean up
        """
        if session_id in self.session_connections:
            connection_ids = self.session_connections[session_id].copy()
            for connection_id in connection_ids:
                self.disconnect(connection_id, session_id)
            logger.info(f"Cleaned up session {session_id} with {len(connection_ids)} connections")
    
    async def send_heartbeat(self, connection_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Send heartbeat/ping messages to maintain connections
        
        Args:
            connection_id: Send to specific connection, or None for all
            session_id: Send to specific session, or None for all
        """
        heartbeat_message = {
            "type": "heartbeat",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        if connection_id:
            await self.send_personal_message(heartbeat_message, connection_id)
        elif session_id:
            await self.send_to_session(heartbeat_message, session_id)
        else:
            await self.broadcast(heartbeat_message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status information"""
        return {
            "total_connections": self.get_connection_count(),
            "total_sessions": self.get_session_count(),
            "active_sessions": list(self.session_connections.keys()),
            "connections_per_session": {
                session_id: len(connection_ids) 
                for session_id, connection_ids in self.session_connections.items()
            }
        } 