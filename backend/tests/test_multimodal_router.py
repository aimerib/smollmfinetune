"""Tests for multimodal dataset generation API endpoints"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import WebSocket
from pathlib import Path

from backend.app.main import app
from backend.app.routers import multimodal
from backend.app.models import User, MultimodalDataset


class TestMultimodalAPI:
    """Test multimodal dataset generation API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return User(
            id="test-user-123",
            username="testuser",
            email="test@example.com"
        )
    
    @pytest.fixture
    def multimodal_config(self):
        """Sample multimodal generation config"""
        return {
            "name": "Test Character Voice Dataset",
            "sampleCount": 1000,
            "characterCount": 5,
            "narrativeTypes": ["dialogue", "emotional_moment"],
            "useMockTTS": True,
            "ttsProvider": "orpheus",
            "outputDir": "test_multimodal_output",
            "batchSize": 100,
            "temperature": 0.8
        }
    
    def test_start_multimodal_generation_success(self, client, mock_user, multimodal_config):
        """Test starting multimodal generation successfully"""
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.routers.multimodal.celery_app.send_task') as mock_celery:
                mock_celery.return_value.id = "test-task-123"
                
                response = client.post(
                    "/api/multimodal/generate",
                    json=multimodal_config
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Check response structure
                assert "id" in data
                assert data["name"] == multimodal_config["name"]
                assert data["status"] == "pending"
                assert data["progress"] == 0.0
                assert data["config"] == multimodal_config
                assert "createdAt" in data
                
                # Verify Celery task was triggered
                mock_celery.assert_called_once()
    
    def test_start_multimodal_generation_unauthorized(self, client, multimodal_config):
        """Test unauthorized access to generation endpoint"""
        response = client.post(
            "/api/multimodal/generate",
            json=multimodal_config
        )
        
        assert response.status_code == 401
    
    def test_start_multimodal_generation_invalid_config(self, client, mock_user):
        """Test starting generation with invalid config"""
        invalid_config = {
            "sampleCount": -1,  # Invalid negative count
            "characterCount": 0  # Invalid zero count
        }
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/multimodal/generate",
                json=invalid_config
            )
            
            assert response.status_code == 422  # Validation error
    
    def test_get_multimodal_job_success(self, client, mock_user):
        """Test retrieving multimodal job successfully"""
        job_id = "test-job-123"
        mock_job = MultimodalDataset(
            id=job_id,
            name="Test Job",
            status="generating",
            progress=0.45,
            samples_generated=450,
            total_samples=1000,
            current_step="Synthesizing speech for character 3/5"
        )
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                mock_session = MagicMock()
                mock_session.query.return_value.filter.return_value.first.return_value = mock_job
                mock_db.return_value = mock_session
                
                response = client.get(f"/api/multimodal/jobs/{job_id}")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["id"] == job_id
                assert data["status"] == "generating"
                assert data["progress"] == 0.45
                assert data["samplesGenerated"] == 450
                assert data["currentStep"] == "Synthesizing speech for character 3/5"
    
    def test_get_multimodal_job_not_found(self, client, mock_user):
        """Test retrieving non-existent job"""
        job_id = "nonexistent-job"
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                mock_session = MagicMock()
                mock_session.query.return_value.filter.return_value.first.return_value = None
                mock_db.return_value = mock_session
                
                response = client.get(f"/api/multimodal/jobs/{job_id}")
                
                assert response.status_code == 404
    
    def test_get_generation_progress(self, client, mock_user):
        """Test getting generation progress"""
        job_id = "test-job-123"
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                mock_session = MagicMock()
                mock_job = MultimodalDataset(
                    id=job_id,
                    progress=0.75,
                    current_step="Processing final samples",
                    samples_generated=750,
                    total_samples=1000
                )
                mock_session.query.return_value.filter.return_value.first.return_value = mock_job
                mock_db.return_value = mock_session
                
                response = client.get(f"/api/multimodal/jobs/{job_id}/progress")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["progress"] == 0.75
                assert data["currentStep"] == "Processing final samples"
                assert data["samplesGenerated"] == 750
                assert data["totalSamples"] == 1000
    
    def test_cancel_generation(self, client, mock_user):
        """Test canceling a generation job"""
        job_id = "test-job-123"
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                with patch('backend.app.routers.multimodal.celery_app.control.revoke') as mock_revoke:
                    mock_session = MagicMock()
                    mock_job = MultimodalDataset(
                        id=job_id,
                        status="generating",
                        celery_task_id="celery-task-123"
                    )
                    mock_session.query.return_value.filter.return_value.first.return_value = mock_job
                    mock_db.return_value = mock_session
                    
                    response = client.post(f"/api/multimodal/jobs/{job_id}/cancel")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "cancelled"
                    
                    # Verify Celery task was revoked
                    mock_revoke.assert_called_once_with("celery-task-123", terminate=True)
    
    def test_cancel_completed_job(self, client, mock_user):
        """Test canceling already completed job"""
        job_id = "test-job-123"
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                mock_session = MagicMock()
                mock_job = MultimodalDataset(
                    id=job_id,
                    status="completed"
                )
                mock_session.query.return_value.filter.return_value.first.return_value = mock_job
                mock_db.return_value = mock_session
                
                response = client.post(f"/api/multimodal/jobs/{job_id}/cancel")
                
                assert response.status_code == 400
                data = response.json()
                assert "Cannot cancel" in data["detail"]
    
    def test_download_dataset_success(self, client, mock_user):
        """Test downloading completed dataset"""
        job_id = "test-job-123"
        output_path = "/path/to/dataset"
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('fastapi.responses.FileResponse') as mock_file_response:
                        mock_session = MagicMock()
                        mock_job = MultimodalDataset(
                            id=job_id,
                            status="completed",
                            output_path=output_path
                        )
                        mock_session.query.return_value.filter.return_value.first.return_value = mock_job
                        mock_db.return_value = mock_session
                        
                        response = client.get(f"/api/multimodal/jobs/{job_id}/download")
                        
                        # Since we're mocking FileResponse, check that it was called
                        mock_file_response.assert_called_once()
    
    def test_download_incomplete_dataset(self, client, mock_user):
        """Test downloading incomplete dataset"""
        job_id = "test-job-123"
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.database.get_db') as mock_db:
                mock_session = MagicMock()
                mock_job = MultimodalDataset(
                    id=job_id,
                    status="generating"
                )
                mock_session.query.return_value.filter.return_value.first.return_value = mock_job
                mock_db.return_value = mock_session
                
                response = client.get(f"/api/multimodal/jobs/{job_id}/download")
                
                assert response.status_code == 400
                data = response.json()
                assert "not completed" in data["detail"]


class TestMultimodalWebSocket:
    """Test multimodal WebSocket functionality"""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection"""
        websocket = AsyncMock(spec=WebSocket)
        return websocket
    
    @pytest.mark.asyncio
    async def test_websocket_connection_and_updates(self, mock_websocket):
        """Test WebSocket connection and progress updates"""
        job_id = "test-job-123"
        
        # Mock the websocket manager
        with patch('backend.app.routers.multimodal.multimodal_manager') as mock_manager:
            # Simulate connecting to websocket
            await multimodal.multimodal_websocket(mock_websocket, job_id)
            
            # Verify connection was established
            mock_manager.connect.assert_called_once_with(mock_websocket, job_id)
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_progress(self):
        """Test broadcasting progress updates via WebSocket"""
        job_id = "test-job-123"
        progress_update = {
            "progress": 0.5,
            "currentStep": "Generating speech",
            "samplesGenerated": 500
        }
        
        # Create websocket manager
        manager = multimodal.MultimodalWebSocketManager()
        
        # Mock websocket
        mock_websocket = AsyncMock(spec=WebSocket)
        
        # Connect websocket
        await manager.connect(mock_websocket, job_id)
        
        # Broadcast update
        await manager.broadcast_progress(job_id, progress_update)
        
        # Verify update was sent
        mock_websocket.send_json.assert_called_once_with(progress_update)
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection_cleanup(self):
        """Test proper cleanup when websocket disconnects"""
        job_id = "test-job-123"
        
        # Create websocket manager
        manager = multimodal.MultimodalWebSocketManager()
        
        # Mock websocket that will fail on send
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.send_json.side_effect = Exception("Connection closed")
        
        # Connect websocket
        await manager.connect(mock_websocket, job_id)
        
        # Try to broadcast (should handle disconnection)
        await manager.broadcast_progress(job_id, {"progress": 0.5})
        
        # Verify websocket was removed from connections
        assert job_id not in manager.connections or len(manager.connections[job_id]) == 0


class TestMultimodalCeleryIntegration:
    """Test integration with Celery tasks"""
    
    def test_celery_task_signature(self):
        """Test that Celery task is properly configured"""
        from backend.app.tasks.multimodal_generation import generate_multimodal_dataset
        
        # Verify task is registered
        assert hasattr(generate_multimodal_dataset, 'delay')
        assert hasattr(generate_multimodal_dataset, 'apply_async')
    
    @patch('backend.app.tasks.multimodal_generation.MultimodalDatasetGenerator')
    def test_celery_task_execution(self, mock_generator_class):
        """Test Celery task execution flow"""
        from backend.app.tasks.multimodal_generation import generate_multimodal_dataset
        
        # Mock the generator
        mock_generator = AsyncMock()
        mock_generator.generate_dataset.return_value = [
            {"text": "Hello world", "mel_frames": []},
            {"text": "Test sample", "mel_frames": []}
        ]
        mock_generator_class.return_value = mock_generator
        
        job_id = "test-job-123"
        config = {
            "sampleCount": 100,
            "characterCount": 5,
            "narrativeTypes": ["dialogue"]
        }
        
        # Execute task
        result = generate_multimodal_dataset(job_id, config)
        
        # Verify generator was called
        mock_generator_class.assert_called_once()
        
        # Verify result structure
        assert result["job_id"] == job_id
        assert "samples_generated" in result
        assert "output_path" in result


class TestMultimodalErrorHandling:
    """Test error handling in multimodal API"""
    
    def test_tts_service_unavailable_error(self, client, mock_user, multimodal_config):
        """Test graceful handling when TTS service is unavailable"""
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.routers.multimodal.celery_app.send_task') as mock_celery:
                # Simulate TTS service error
                mock_celery.side_effect = Exception("TTS service unavailable")
                
                response = client.post(
                    "/api/multimodal/generate",
                    json=multimodal_config
                )
                
                assert response.status_code == 503  # Service unavailable
                data = response.json()
                assert "TTS service" in data["detail"]
    
    def test_disk_space_error(self, client, mock_user, multimodal_config):
        """Test handling insufficient disk space"""
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            with patch('backend.app.routers.multimodal.check_disk_space', return_value=False):
                response = client.post(
                    "/api/multimodal/generate",
                    json=multimodal_config
                )
                
                assert response.status_code == 507  # Insufficient storage
                data = response.json()
                assert "disk space" in data["detail"]
    
    def test_large_dataset_memory_warning(self, client, mock_user):
        """Test warning for very large dataset requests"""
        large_config = {
            "name": "Huge Dataset",
            "sampleCount": 100000,  # Very large
            "characterCount": 100,
            "narrativeTypes": ["dialogue"],
            "useMockTTS": False  # Real TTS increases memory usage
        }
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/multimodal/generate",
                json=large_config
            )
            
            # Should still succeed but with warnings
            assert response.status_code == 200
            data = response.json()
            
            # Check for warning in response
            assert "warnings" in data
            assert any("memory" in warning.lower() for warning in data["warnings"])


class TestMultimodalValidation:
    """Test input validation for multimodal endpoints"""
    
    def test_sample_count_validation(self, client, mock_user):
        """Test sample count validation"""
        test_cases = [
            (0, False),      # Zero samples
            (-1, False),     # Negative samples
            (1, True),       # Minimum valid
            (10000, True),   # Normal size
            (1000000, False) # Too large
        ]
        
        for sample_count, should_succeed in test_cases:
            config = {
                "name": "Test Dataset",
                "sampleCount": sample_count,
                "characterCount": 5,
                "narrativeTypes": ["dialogue"]
            }
            
            with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
                response = client.post(
                    "/api/multimodal/generate",
                    json=config
                )
                
                if should_succeed:
                    assert response.status_code in [200, 202]
                else:
                    assert response.status_code in [400, 422]
    
    def test_narrative_types_validation(self, client, mock_user):
        """Test narrative types validation"""
        valid_types = ["dialogue", "monologue", "action_scene", "emotional_moment", "memory_recall", "world_description"]
        
        # Test valid types
        config = {
            "name": "Test Dataset",
            "sampleCount": 100,
            "characterCount": 5,
            "narrativeTypes": valid_types[:3]
        }
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/multimodal/generate",
                json=config
            )
            
            assert response.status_code in [200, 202]
        
        # Test invalid types
        invalid_config = {
            "name": "Test Dataset", 
            "sampleCount": 100,
            "characterCount": 5,
            "narrativeTypes": ["invalid_type", "another_invalid"]
        }
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/multimodal/generate",
                json=invalid_config
            )
            
            assert response.status_code == 422
    
    def test_tts_provider_validation(self, client, mock_user):
        """Test TTS provider validation"""
        valid_providers = ["orpheus", "kokoro", "xtts", "bark"]
        
        for provider in valid_providers:
            config = {
                "name": "Test Dataset",
                "sampleCount": 100,
                "characterCount": 5,
                "narrativeTypes": ["dialogue"],
                "ttsProvider": provider
            }
            
            with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
                response = client.post(
                    "/api/multimodal/generate",
                    json=config
                )
                
                assert response.status_code in [200, 202]
        
        # Test invalid provider
        invalid_config = {
            "name": "Test Dataset",
            "sampleCount": 100,
            "characterCount": 5,
            "narrativeTypes": ["dialogue"],
            "ttsProvider": "invalid_provider"
        }
        
        with patch('backend.app.routers.multimodal.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/multimodal/generate",
                json=invalid_config
            )
            
            assert response.status_code == 422 