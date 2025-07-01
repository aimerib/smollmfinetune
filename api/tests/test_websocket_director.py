"""
Tests for Director's View WebSocket Endpoint

Following TDD principles to ensure the WebSocket connection and
real-time event streaming work correctly.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from websocket import create_connection

from app.main import app
from app.services.event_bus import EventBus, EventType, MemoryFormationEvent
from app.websocket.manager import WebSocketManager


@pytest.fixture
def test_client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def mock_state_service():
    """Mock state service"""
    with patch('app.websocket.director.state_service') as mock:
        mock.get_world_snapshot = AsyncMock(return_value={
            "locations": [
                {"id": "loc1", "name": "Test Location", "type": "location"}
            ],
            "characters": [
                {"id": "char1", "name": "Test Character", "type": "character", 
                 "location": "loc1", "memory_count": 0, "relationship_count": 0}
            ],
            "total_entities": 2,
            "recent_subtext": [],
            "timestamp": "2025-01-17T00:00:00"
        })
        yield mock


class TestWebSocketConnection:
    """Test WebSocket connection and basic functionality"""
    
    def test_websocket_connect(self, test_client):
        """Test that we can connect to the WebSocket endpoint"""
        with test_client.websocket_connect("/ws/director") as websocket:
            # Should receive connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "client_id" in data
    
    def test_websocket_initial_state(self, test_client, mock_state_service):
        """Test that initial state is sent on connection"""
        with test_client.websocket_connect("/ws/director") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Should receive initial state
            data = websocket.receive_json()
            assert data["type"] == "initial_state"
            assert "world" in data["data"]
            assert len(data["data"]["world"]["locations"]) == 1
            assert len(data["data"]["world"]["characters"]) == 1
    
    def test_websocket_ping_pong(self, test_client):
        """Test ping/pong heartbeat"""
        with test_client.websocket_connect("/ws/director") as websocket:
            # Skip initial messages
            websocket.receive_json()  # connection
            websocket.receive_json()  # initial state
            
            # Send ping
            websocket.send_json({"type": "ping"})
            
            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
    
    def test_websocket_subscribe(self, test_client):
        """Test topic subscription"""
        with test_client.websocket_connect("/ws/director") as websocket:
            # Skip initial messages
            websocket.receive_json()
            websocket.receive_json()
            
            # Subscribe to topics
            websocket.send_json({
                "type": "subscribe",
                "topics": ["test_topic"]
            })
            
            # No error should occur (subscription is handled internally)
            # In a real test, we'd verify the subscription was stored


class TestWebSocketEvents:
    """Test WebSocket event broadcasting"""
    
    @pytest.mark.asyncio
    async def test_memory_formation_event(self, test_client):
        """Test that memory formation events are broadcast"""
        # This test would require more complex setup with event bus integration
        # For now, we test the event structure
        event = MemoryFormationEvent(
            source="test",
            character_id="char1",
            memory_content="Test memory",
            importance=0.8,
            emotional_valence=0.5,
            memory_type="episodic"
        )
        
        assert event.event_type == EventType.MEMORY_FORMED
        assert event.data["character_id"] == "char1"
        assert "visualization" in event.data
        assert "bubble_color" in event.data["visualization"]
    
    def test_entity_details_request(self, test_client):
        """Test requesting entity details"""
        with test_client.websocket_connect("/ws/director") as websocket:
            # Skip initial messages
            websocket.receive_json()
            websocket.receive_json()
            
            # Request entity details
            websocket.send_json({
                "type": "get_entity_details",
                "entity_id": "char1"
            })
            
            # In a real implementation, we'd mock the response
            # For now, we just verify the request doesn't error


class TestWebSocketManager:
    """Test the WebSocket manager functionality"""
    
    @pytest.mark.asyncio
    async def test_connection_management(self):
        """Test WebSocket connection management"""
        manager = WebSocketManager()
        
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        # Connect
        client_id = await manager.connect(mock_ws)
        assert client_id is not None
        assert client_id in manager.active_connections
        
        # Disconnect
        await manager.disconnect(client_id)
        assert client_id not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test message broadcasting"""
        manager = WebSocketManager()
        
        # Create mock connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        client1 = await manager.connect(mock_ws1)
        client2 = await manager.connect(mock_ws2)
        
        # Broadcast message
        message = {"type": "test", "data": "broadcast"}
        await manager.broadcast(message)
        
        # Both clients should receive the message
        mock_ws1.send_json.assert_called_with(message)
        mock_ws2.send_json.assert_called_with(message)
    
    @pytest.mark.asyncio
    async def test_topic_subscription(self):
        """Test topic-based message routing"""
        manager = WebSocketManager()
        
        # Create connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        client1 = await manager.connect(mock_ws1)
        client2 = await manager.connect(mock_ws2)
        
        # Subscribe client1 to a topic
        await manager.subscribe_to_topic(client1, "test_topic")
        
        # Broadcast to topic
        message = {"type": "test", "data": "topic_message"}
        await manager.broadcast_to_topic("test_topic", message)
        
        # Only client1 should receive the message
        mock_ws1.send_json.assert_called_with(message)
        mock_ws2.send_json.assert_not_called() 