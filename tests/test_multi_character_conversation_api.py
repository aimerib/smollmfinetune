"""
Tests for Multi-Character Conversation FastAPI Router
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import websocket

from backend.app.routers.multi_character_conversation import router
from backend.app.services.conversation.multi_character_manager import (
    MultiCharacterConversationManager,
    DialogueTurn,
    Vector3D
)


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestMultiCharacterConversationAPI:
    """Test the FastAPI endpoints for multi-character conversations"""
    
    @pytest.fixture
    def mock_conversation_manager(self):
        """Mock conversation manager"""
        manager = Mock(spec=MultiCharacterConversationManager)
        manager.conversation_state = Mock()
        manager.conversation_state.get_active_speakers.return_value = ['char1', 'char2']
        manager.conversation_state.get_context.return_value = {'test': 'context'}
        manager.conversation_state.add_speaker = Mock()
        manager.conversation_state.set_context = Mock()
        
        manager.spatial_audio = Mock()
        manager.spatial_audio.get_position_info.return_value = {'x': 0, 'y': 0, 'z': 0}
        manager.spatial_audio.update_character_position.return_value = {
            'character_id': 'char1',
            'position': {'x': 1, 'y': 0, 'z': 1},
            'distance': 1.414
        }
        
        manager.generate_multi_character_dialogue = AsyncMock()
        manager.handle_conversation_interruption = AsyncMock()
        
        return manager
    
    @pytest.fixture(autouse=True)
    def setup_test_managers(self, mock_conversation_manager):
        """Setup test conversation managers"""
        with patch.dict('backend.app.routers.multi_character_conversation.conversation_managers', 
                       {'test-session': mock_conversation_manager}, clear=True):
            yield
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/multi-character/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_sessions" in data
        assert data["service"] == "multi-character-conversation"
    
    def test_create_conversation(self, mock_conversation_manager):
        """Test creating a new conversation session"""
        request_data = {
            "session_id": "new-session-123",
            "character_ids": ["char1", "char2"],
            "initial_context": {"test": "data"}
        }
        
        with patch('backend.app.routers.multi_character_conversation.MultiCharacterConversationManager') as MockManager:
            MockManager.return_value = mock_conversation_manager
            with patch('backend.app.routers.multi_character_conversation.conversation_managers') as mock_managers:
                mock_managers.__setitem__ = Mock()
                
                response = client.post("/api/v1/multi-character/conversations", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new-session-123"
        assert data["active_characters"] == ['char1', 'char2']
        assert data["conversation_context"] == {'test': 'context'}
        assert "spatial_positions" in data
    
    def test_get_conversation_status(self, mock_conversation_manager):
        """Test getting conversation status"""
        response = client.get("/api/v1/multi-character/conversations/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["active_characters"] == ['char1', 'char2']
        assert data["conversation_context"] == {'test': 'context'}
        assert "spatial_positions" in data
    
    def test_get_conversation_status_not_found(self):
        """Test getting status for non-existent conversation"""
        response = client.get("/api/v1/multi-character/conversations/non-existent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_add_dialogue_turn(self, mock_conversation_manager):
        """Test adding a dialogue turn to conversation"""
        request_data = {
            "character_id": "char1",
            "text": "Hello world",
            "emotion_context": {"happiness": 0.8},
            "interrupts_previous": False,
            "urgency_level": 0.5
        }
        
        response = client.post("/api/v1/multi-character/conversations/test-session/dialogue", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Verify conversation manager was called
        mock_conversation_manager.generate_multi_character_dialogue.assert_called_once()
        call_args = mock_conversation_manager.generate_multi_character_dialogue.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].character_id == "char1"
        assert call_args[0].text == "Hello world"
    
    def test_add_dialogue_turn_with_interruption(self, mock_conversation_manager):
        """Test adding a dialogue turn that interrupts previous speaker"""
        request_data = {
            "character_id": "char1",
            "text": "Wait, let me interrupt!",
            "interrupts_previous": True,
            "urgency_level": 0.8
        }
        
        response = client.post("/api/v1/multi-character/conversations/test-session/dialogue", json=request_data)
        
        assert response.status_code == 200
        
        # Verify interruption handler was called
        mock_conversation_manager.handle_conversation_interruption.assert_called_once()
    
    def test_add_dialogue_turn_session_not_found(self):
        """Test adding dialogue to non-existent session"""
        request_data = {
            "character_id": "char1",
            "text": "Hello world"
        }
        
        response = client.post("/api/v1/multi-character/conversations/non-existent/dialogue", json=request_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_update_spatial_position(self, mock_conversation_manager):
        """Test updating character spatial position"""
        request_data = {
            "character_id": "char1",
            "position": {"x": 2.0, "y": 0.5, "z": -1.0}
        }
        
        response = client.post("/api/v1/multi-character/conversations/test-session/spatial-position", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["character_id"] == "char1"
        assert data["position"] == {"x": 2.0, "y": 0.5, "z": -1.0}
        assert "distance" in data
        
        # Verify spatial audio update was called
        mock_conversation_manager.spatial_audio.update_character_position.assert_called_once()
        call_args = mock_conversation_manager.spatial_audio.update_character_position.call_args
        assert call_args[0][0] == "char1"
        position = call_args[0][1]
        assert position.x == 2.0
        assert position.y == 0.5
        assert position.z == -1.0
    
    def test_update_spatial_position_session_not_found(self):
        """Test updating spatial position for non-existent session"""
        request_data = {
            "character_id": "char1",
            "position": {"x": 1.0, "y": 0.0, "z": 1.0}
        }
        
        response = client.post("/api/v1/multi-character/conversations/non-existent/spatial-position", json=request_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_end_conversation(self):
        """Test ending a conversation session"""
        with patch('backend.app.routers.multi_character_conversation.conversation_managers') as mock_managers:
            mock_managers.__contains__ = Mock(return_value=True)
            mock_managers.__delitem__ = Mock()
            
            response = client.delete("/api/v1/multi-character/conversations/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "ended" in data["message"]
    
    def test_end_conversation_not_found(self):
        """Test ending non-existent conversation"""
        response = client.delete("/api/v1/multi-character/conversations/non-existent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestMultiCharacterWebSocket:
    """Test WebSocket functionality for multi-character conversations"""
    
    def test_websocket_connection_wrapper(self):
        """Test WebSocketConnectionWrapper functionality"""
        from backend.app.routers.multi_character_conversation import WebSocketConnectionWrapper
        
        mock_websocket = Mock()
        mock_websocket.send_json = AsyncMock()
        
        wrapper = WebSocketConnectionWrapper(mock_websocket)
        
        # Test send_audio_chunk
        test_data = {
            'character_id': 'char1',
            'audio_data': b'test_audio',
            'spatial_info': {'x': 1, 'y': 0, 'z': 1}
        }
        
        asyncio.run(wrapper.send_audio_chunk(test_data))
        
        mock_websocket.send_json.assert_called_once_with({
            'type': 'audio_chunk',
            'data': test_data
        })


class TestDialogueTurnModel:
    """Test DialogueTurn model creation and validation"""
    
    def test_dialogue_turn_creation(self):
        """Test creating a DialogueTurn object"""
        turn = DialogueTurn(
            character_id="char1",
            text="Hello world",
            emotion_context={"happiness": 0.8},
            interrupts_previous=False,
            urgency_level=0.5
        )
        
        assert turn.character_id == "char1"
        assert turn.text == "Hello world"
        assert turn.emotion_context == {"happiness": 0.8}
        assert turn.interrupts_previous == False
        assert turn.urgency_level == 0.5
    
    def test_dialogue_turn_optional_fields(self):
        """Test DialogueTurn with only required fields"""
        turn = DialogueTurn(
            character_id="char1",
            text="Hello world"
        )
        
        assert turn.character_id == "char1"
        assert turn.text == "Hello world"
        assert turn.emotion_context is None
        assert turn.interrupts_previous == False  # Default value
        assert turn.urgency_level == 0.5  # Default value


class TestVector3DModel:
    """Test Vector3D model"""
    
    def test_vector3d_creation(self):
        """Test creating a Vector3D object"""
        vector = Vector3D(x=1.0, y=2.0, z=3.0)
        
        assert vector.x == 1.0
        assert vector.y == 2.0
        assert vector.z == 3.0
    
    def test_vector3d_equality(self):
        """Test Vector3D equality comparison"""
        v1 = Vector3D(x=1.0, y=2.0, z=3.0)
        v2 = Vector3D(x=1.0, y=2.0, z=3.0)
        v3 = Vector3D(x=2.0, y=2.0, z=3.0)
        
        assert v1 == v2
        assert v1 != v3
        assert v1 != "not a vector"


class TestPydanticModels:
    """Test Pydantic models used in API"""
    
    def test_conversation_request_model(self):
        """Test ConversationRequest model validation"""
        from backend.app.routers.multi_character_conversation import ConversationRequest
        
        # Valid request
        valid_data = {
            "session_id": "test-session",
            "character_ids": ["char1", "char2"],
            "initial_context": {"key": "value"}
        }
        
        request = ConversationRequest(**valid_data)
        assert request.session_id == "test-session"
        assert request.character_ids == ["char1", "char2"]
        assert request.initial_context == {"key": "value"}
        
        # Request without optional field
        minimal_data = {
            "session_id": "test-session",
            "character_ids": ["char1"]
        }
        
        request = ConversationRequest(**minimal_data)
        assert request.initial_context is None
    
    def test_dialogue_turn_request_model(self):
        """Test DialogueTurnRequest model validation"""
        from backend.app.routers.multi_character_conversation import DialogueTurnRequest
        
        # Full request
        full_data = {
            "character_id": "char1",
            "text": "Hello world",
            "emotion_context": {"happiness": 0.8},
            "interrupts_previous": True,
            "urgency_level": 0.9
        }
        
        request = DialogueTurnRequest(**full_data)
        assert request.character_id == "char1"
        assert request.text == "Hello world"
        assert request.emotion_context == {"happiness": 0.8}
        assert request.interrupts_previous == True
        assert request.urgency_level == 0.9
        
        # Minimal request
        minimal_data = {
            "character_id": "char1",
            "text": "Hello"
        }
        
        request = DialogueTurnRequest(**minimal_data)
        assert request.emotion_context is None
        assert request.interrupts_previous == False
        assert request.urgency_level == 0.5
    
    def test_spatial_position_request_model(self):
        """Test SpatialPositionRequest model validation"""
        from backend.app.routers.multi_character_conversation import SpatialPositionRequest
        
        data = {
            "character_id": "char1",
            "position": {"x": 1.0, "y": 0.5, "z": -2.0}
        }
        
        request = SpatialPositionRequest(**data)
        assert request.character_id == "char1"
        assert request.position == {"x": 1.0, "y": 0.5, "z": -2.0}
    
    def test_conversation_status_response_model(self):
        """Test ConversationStatusResponse model"""
        from backend.app.routers.multi_character_conversation import ConversationStatusResponse
        
        data = {
            "session_id": "test-session",
            "active_characters": ["char1", "char2"],
            "conversation_context": {"key": "value"},
            "spatial_positions": {
                "char1": {"x": 1.0, "y": 0.0, "z": 1.0},
                "char2": {"x": -1.0, "y": 0.0, "z": -1.0}
            }
        }
        
        response = ConversationStatusResponse(**data)
        assert response.session_id == "test-session"
        assert response.active_characters == ["char1", "char2"]
        assert response.conversation_context == {"key": "value"}
        assert len(response.spatial_positions) == 2


class TestIntegrationScenarios:
    """Test integrated scenarios across multiple endpoints"""
    
    @pytest.fixture
    def mock_conversation_manager(self):
        """Mock conversation manager for integration tests"""
        manager = Mock(spec=MultiCharacterConversationManager)
        manager.conversation_state = Mock()
        manager.conversation_state.get_active_speakers.return_value = ['char1', 'char2']
        manager.conversation_state.get_context.return_value = {'test': 'context'}
        manager.conversation_state.add_speaker = Mock()
        manager.conversation_state.set_context = Mock()
        
        manager.spatial_audio = Mock()
        manager.spatial_audio.get_position_info.return_value = {'x': 0, 'y': 0, 'z': 0}
        manager.spatial_audio.update_character_position.return_value = {
            'character_id': 'char1',
            'position': {'x': 1, 'y': 0, 'z': 1},
            'distance': 1.414
        }
        
        manager.generate_multi_character_dialogue = AsyncMock()
        manager.handle_conversation_interruption = AsyncMock()
        
        return manager
    
    def test_complete_conversation_workflow(self, mock_conversation_manager):
        """Test complete workflow from creation to deletion"""
        # 1. Create conversation
        create_data = {
            "session_id": "workflow-test",
            "character_ids": ["alice", "bob"]
        }
        
        with patch('backend.app.routers.multi_character_conversation.MultiCharacterConversationManager') as MockManager:
            MockManager.return_value = mock_conversation_manager
            with patch('backend.app.routers.multi_character_conversation.conversation_managers') as mock_managers:
                mock_managers.__setitem__ = Mock()
                mock_managers.__contains__ = Mock(return_value=True)
                mock_managers.__getitem__ = Mock(return_value=mock_conversation_manager)
                mock_managers.__delitem__ = Mock()
                
                # Create
                response = client.post("/api/v1/multi-character/conversations", json=create_data)
                assert response.status_code == 200
                
                # Get status
                response = client.get("/api/v1/multi-character/conversations/workflow-test")
                assert response.status_code == 200
                
                # Add dialogue
                dialogue_data = {
                    "character_id": "alice",
                    "text": "Hello Bob!"
                }
                response = client.post("/api/v1/multi-character/conversations/workflow-test/dialogue", json=dialogue_data)
                assert response.status_code == 200
                
                # Update spatial position
                position_data = {
                    "character_id": "alice",
                    "position": {"x": 1.0, "y": 0.0, "z": 0.0}
                }
                response = client.post("/api/v1/multi-character/conversations/workflow-test/spatial-position", json=position_data)
                assert response.status_code == 200
                
                # End conversation
                response = client.delete("/api/v1/multi-character/conversations/workflow-test")
                assert response.status_code == 200
    
    def test_error_handling_across_endpoints(self):
        """Test error handling consistency across endpoints"""
        non_existent_session = "does-not-exist"
        
        # All endpoints should return 404 for non-existent session
        endpoints_to_test = [
            ("GET", f"/api/v1/multi-character/conversations/{non_existent_session}"),
            ("POST", f"/api/v1/multi-character/conversations/{non_existent_session}/dialogue"),
            ("POST", f"/api/v1/multi-character/conversations/{non_existent_session}/spatial-position"),
            ("DELETE", f"/api/v1/multi-character/conversations/{non_existent_session}")
        ]
        
        for method, url in endpoints_to_test:
            if method == "GET":
                response = client.get(url)
            elif method == "POST":
                if "dialogue" in url:
                    # Provide valid dialogue data
                    test_data = {
                        "character_id": "char1",
                        "text": "test message"
                    }
                elif "spatial-position" in url:
                    # Provide valid spatial position data
                    test_data = {
                        "character_id": "char1",
                        "position": {"x": 1.0, "y": 0.0, "z": 1.0}
                    }
                else:
                    test_data = {"test": "data"}
                response = client.post(url, json=test_data)
            elif method == "DELETE":
                response = client.delete(url)
            
            assert response.status_code == 404, f"Failed for {method} {url}"
            assert "not found" in response.json()["detail"].lower()
    
    def test_concurrent_session_management(self, mock_conversation_manager):
        """Test handling multiple concurrent sessions"""
        sessions = ["session1", "session2", "session3"]
        
        with patch('backend.app.routers.multi_character_conversation.MultiCharacterConversationManager') as MockManager:
            MockManager.return_value = mock_conversation_manager
            with patch('backend.app.routers.multi_character_conversation.conversation_managers') as mock_managers:
                mock_managers.__setitem__ = Mock()
                mock_managers.__contains__ = Mock(side_effect=lambda x: x in sessions)
                mock_managers.__getitem__ = Mock(return_value=mock_conversation_manager)
                
                # Create multiple sessions
                for session_id in sessions:
                    create_data = {
                        "session_id": session_id,
                        "character_ids": ["char1", "char2"]
                    }
                    response = client.post("/api/v1/multi-character/conversations", json=create_data)
                    assert response.status_code == 200
                
                # Verify each session can be accessed independently
                for session_id in sessions:
                    response = client.get(f"/api/v1/multi-character/conversations/{session_id}")
                    assert response.status_code == 200
                    assert response.json()["session_id"] == session_id 