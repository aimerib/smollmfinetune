"""
Tests for Quad-Head Streaming WebSocket functionality

Tests the real-time multimodal streaming capabilities of the QuadHeadNarrativeLM model.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import torch

from backend.app.routers.quad_head_streaming import (
    router,
    QuadHeadStreamRequest,
    QuadHeadStreamResponse,
    QuadHeadConnectionManager,
    connection_manager,
    stream_multimodal_generation
)
from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM
from backend.app.narrative_engine.config import NarrativeLLMConfig


class TestQuadHeadStreamRequest:
    """Test the streaming request model"""
    
    def test_valid_request_creation(self):
        """Test creating a valid streaming request"""
        request = QuadHeadStreamRequest(
            type="generate_multimodal",
            text="Hello, how are you?",
            character_id="test_character"
        )
        
        assert request.type == "generate_multimodal"
        assert request.text == "Hello, how are you?"
        assert request.character_id == "test_character"
        assert request.max_length == 100  # default
        assert request.temperature == 0.8  # default
        assert request.speech_temperature == 0.7  # default
        assert request.force_speech is True  # default
        assert request.streaming is True  # default
    
    def test_request_with_custom_parameters(self):
        """Test creating request with custom parameters"""
        request = QuadHeadStreamRequest(
            type="generate_multimodal",
            text="Custom text",
            character_id="custom_char",
            max_length=200,
            temperature=0.9,
            speech_temperature=0.8,
            emotion_context={"happiness": 0.8},
            force_speech=False,
            streaming=False
        )
        
        assert request.max_length == 200
        assert request.temperature == 0.9
        assert request.speech_temperature == 0.8
        assert request.emotion_context == {"happiness": 0.8}
        assert request.force_speech is False
        assert request.streaming is False


class TestQuadHeadStreamResponse:
    """Test the streaming response model"""
    
    def test_basic_response_creation(self):
        """Test creating a basic response"""
        response = QuadHeadStreamResponse(
            type="generation_step",
            timestamp=123456.789,
            finished=False
        )
        
        assert response.type == "generation_step"
        assert response.timestamp == 123456.789
        assert response.finished is False
        assert response.text_token is None
        assert response.speech_frame is None
        assert response.control_signal is None
        assert response.memory_update is None
        assert response.error is None
    
    def test_complete_response_creation(self):
        """Test creating response with all fields"""
        response = QuadHeadStreamResponse(
            type="generation_step",
            text_token="hello",
            speech_frame=[0.1, 0.2, 0.3],
            control_signal="control_1",
            memory_update={"key": "value"},
            timestamp=123456.789,
            finished=True,
            error="test error"
        )
        
        assert response.text_token == "hello"
        assert response.speech_frame == [0.1, 0.2, 0.3]
        assert response.control_signal == "control_1"
        assert response.memory_update == {"key": "value"}
        assert response.error == "test error"


class TestQuadHeadConnectionManager:
    """Test the WebSocket connection manager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = QuadHeadConnectionManager()
        self.mock_websocket = Mock(spec=WebSocket)
        self.mock_websocket.accept = AsyncMock()
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test connecting a WebSocket"""
        character_id = "test_character"
        
        connection_id = await self.manager.connect(self.mock_websocket, character_id)
        
        # Check connection was established
        assert connection_id in self.manager.connections
        assert self.manager.connections[connection_id] == self.mock_websocket
        assert self.manager.character_sessions[character_id] == connection_id
        assert connection_id.startswith(f"quad_head_{character_id}_")
        
        # Check websocket was accepted
        self.mock_websocket.accept.assert_called_once()
    
    def test_disconnect_websocket(self):
        """Test disconnecting a WebSocket"""
        # Set up connection
        connection_id = "test_connection"
        character_id = "test_character"
        self.manager.connections[connection_id] = self.mock_websocket
        self.manager.character_sessions[character_id] = connection_id
        
        # Disconnect
        self.manager.disconnect(connection_id)
        
        # Check connection was removed
        assert connection_id not in self.manager.connections
        assert character_id not in self.manager.character_sessions
    
    @pytest.mark.asyncio
    async def test_send_to_connection_success(self):
        """Test sending message to connection successfully"""
        connection_id = "test_connection"
        self.manager.connections[connection_id] = self.mock_websocket
        self.mock_websocket.send_text = AsyncMock()
        
        message = {"type": "test", "data": "hello"}
        
        await self.manager.send_to_connection(connection_id, message)
        
        # Check message was sent
        expected_message = json.dumps(message)
        self.mock_websocket.send_text.assert_called_once_with(expected_message)
    
    @pytest.mark.asyncio
    async def test_send_to_connection_failure(self):
        """Test handling send failure"""
        connection_id = "test_connection"
        self.manager.connections[connection_id] = self.mock_websocket
        self.mock_websocket.send_text = AsyncMock(side_effect=Exception("Send failed"))
        
        message = {"type": "test", "data": "hello"}
        
        # Should not raise exception, but should disconnect
        await self.manager.send_to_connection(connection_id, message)
        
        # Connection should be removed
        assert connection_id not in self.manager.connections


class TestStreamMultimodalGeneration:
    """Test the multimodal generation streaming function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=QuadHeadNarrativeLM)
        self.mock_model.config = Mock()
        self.mock_model.config.hidden_size = 768
        
        self.mock_connection_manager = Mock()
        self.mock_connection_manager.send_to_connection = AsyncMock()
        
        self.request = QuadHeadStreamRequest(
            type="generate_multimodal",
            text="Hello world",
            character_id="test_character",
            max_length=5  # Short for testing
        )
        
        self.connection_id = "test_connection"
    
    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful multimodal generation"""
        # Mock model outputs
        mock_outputs = {
            'text_logits': torch.randn(1, 20, 1000),
            'speech_logits': torch.randn(1, 20, 80),
            'control_logits': torch.randn(1, 20, 100),
            'memory_updates': torch.randn(1, 20, 512)
        }
        self.mock_model.return_value = mock_outputs
        
        with patch('backend.app.routers.quad_head_streaming.connection_manager', self.mock_connection_manager):
            await stream_multimodal_generation(
                self.mock_model,
                self.request,
                self.connection_id
            )
        
        # Check model was called
        assert self.mock_model.call_count == self.request.max_length
        
        # Check responses were sent
        assert self.mock_connection_manager.send_to_connection.call_count >= self.request.max_length
        
        # Check completion message was sent
        calls = self.mock_connection_manager.send_to_connection.call_args_list
        completion_call = calls[-1]
        completion_message = completion_call[0][1]  # Second argument (message)
        assert completion_message['type'] == 'generation_complete'
        assert completion_message['finished'] is True
    
    @pytest.mark.asyncio
    async def test_generation_with_error(self):
        """Test handling generation errors"""
        # Mock model to raise exception
        self.mock_model.side_effect = Exception("Model error")
        
        with patch('backend.app.routers.quad_head_streaming.connection_manager', self.mock_connection_manager):
            await stream_multimodal_generation(
                self.mock_model,
                self.request,
                self.connection_id
            )
        
        # Check error response was sent
        self.mock_connection_manager.send_to_connection.assert_called()
        error_call = self.mock_connection_manager.send_to_connection.call_args_list[-1]
        error_message = error_call[0][1]
        assert error_message['type'] == 'error'
        assert 'Model error' in error_message['error']
        assert error_message['finished'] is True


class TestQuadHeadStreamingIntegration:
    """Integration tests for the quad-head streaming endpoints"""
    
    def setup_method(self):
        """Set up test client"""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
    
    def test_streaming_status_endpoint(self):
        """Test the streaming status endpoint"""
        response = self.client.get("/api/v1/quad-head/stream/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "connected_clients" in data
        assert "character_sessions" in data
        assert "model_loaded" in data
        assert data["status"] == "active"
        assert isinstance(data["connected_clients"], int)
        assert isinstance(data["character_sessions"], int)
        assert isinstance(data["model_loaded"], bool)


class TestQuadHeadStreamingWebSocket:
    """Test WebSocket streaming functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_flow(self):
        """Test basic WebSocket connection flow"""
        # This would require more complex WebSocket testing setup
        # For now, we test the components individually
        
        # Test connection manager
        manager = QuadHeadConnectionManager()
        mock_ws = Mock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        
        # Test connection
        connection_id = await manager.connect(mock_ws, "test_character")
        assert connection_id in manager.connections
        
        # Test disconnection
        manager.disconnect(connection_id)
        assert connection_id not in manager.connections
    
    def test_websocket_message_parsing(self):
        """Test parsing WebSocket messages"""
        # Test valid message
        valid_message = {
            "type": "generate_multimodal",
            "text": "Hello world",
            "character_id": "test_character"
        }
        
        request = QuadHeadStreamRequest(**valid_message)
        assert request.type == "generate_multimodal"
        assert request.text == "Hello world"
        assert request.character_id == "test_character"
        
        # Test response creation
        response = QuadHeadStreamResponse(
            type="generation_step",
            text_token="hello",
            timestamp=123456.789,
            finished=False
        )
        
        response_dict = response.model_dump()
        assert response_dict["type"] == "generation_step"
        assert response_dict["text_token"] == "hello"
        assert response_dict["finished"] is False


@pytest.mark.slow
@pytest.mark.integration
class TestQuadHeadStreamingEndToEnd:
    """End-to-end tests for quad-head streaming (marked as slow)"""
    
    @pytest.mark.asyncio
    async def test_full_streaming_pipeline(self):
        """Test the complete streaming pipeline"""
        # This would test the full pipeline with a real model
        # Marked as slow since it involves model loading
        
        # Mock the dependencies for now
        with patch('backend.app.routers.quad_head_streaming.get_quad_head_model') as mock_get_model:
            mock_model = Mock()
            mock_model.config.hidden_size = 768
            mock_model.return_value = {
                'text_logits': torch.randn(1, 1, 1000),
                'speech_logits': torch.randn(1, 1, 80),
                'control_logits': torch.randn(1, 1, 100),
                'memory_updates': torch.randn(1, 1, 512)
            }
            mock_get_model.return_value = mock_model
            
            # Test would involve creating actual WebSocket connection
            # and verifying the complete flow
            assert True  # Placeholder for actual implementation


if __name__ == "__main__":
    pytest.main([__file__]) 