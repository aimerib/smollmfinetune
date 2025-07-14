"""
Tests for Quad-Head Training API

Comprehensive test suite for the FastAPI training endpoints that handle:
- Training job lifecycle (start, monitor, cancel)
- Real-time metrics streaming via WebSocket
- Model download functionality
- Multi-task loss configuration
- Background task integration
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import WebSocket
from datetime import datetime
from pathlib import Path
import tempfile

# Test imports
from backend.app.routers.quad_head_training import (
    QuadHeadTrainingConfig,
    TrainingStatus,
    TrainingMetrics,
    TrainingWebSocketManager,
    router
)


class TestQuadHeadTrainingConfig:
    """Test training configuration validation"""
    
    def test_training_config_defaults(self):
        """Test default training configuration values"""
        config = QuadHeadTrainingConfig(dataset_path="/tmp/test_data")
        
        assert config.base_model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert config.enable_speech_head is True
        assert config.speech_mel_bins == 80
        assert config.speech_quantization_bits == 4
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.num_train_epochs == 3
        assert config.text_weight == 1.0
        assert config.control_weight == 0.8
        assert config.memory_weight == 0.6
        assert config.speech_weight == 0.5
    
    def test_training_config_custom_values(self):
        """Test custom training configuration"""
        config = QuadHeadTrainingConfig(
            dataset_path="/custom/path",
            base_model_name="custom/model",
            learning_rate=5e-5,
            batch_size=8,
            num_train_epochs=5,
            speech_weight=0.8,
            enable_speech_head=False
        )
        
        assert config.dataset_path == "/custom/path"
        assert config.base_model_name == "custom/model"
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8
        assert config.num_train_epochs == 5
        assert config.speech_weight == 0.8
        assert config.enable_speech_head is False
    
    def test_training_config_validation(self):
        """Test training configuration validation"""
        # Valid config
        config = QuadHeadTrainingConfig(
            dataset_path="/tmp/test",
            learning_rate=1e-4,
            batch_size=4
        )
        
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_train_epochs > 0


class TestTrainingStatus:
    """Test training status model"""
    
    def test_training_status_creation(self):
        """Test training status model creation"""
        status = TrainingStatus(
            job_id="test-job-123",
            status="running",
            created_at=datetime.utcnow(),
            current_step=50,
            total_steps=1000,
            progress=5.0
        )
        
        assert status.job_id == "test-job-123"
        assert status.status == "running"
        assert status.current_step == 50
        assert status.total_steps == 1000
        assert status.progress == 5.0
    
    def test_training_status_with_metrics(self):
        """Test training status with last metrics"""
        metrics_dict = {
            "step": 100,
            "epoch": 1,
            "total_loss": 0.5,
            "learning_rate": 1e-4
        }
        
        status = TrainingStatus(
            job_id="test-job-456",
            status="running",
            created_at=datetime.utcnow(),
            last_metrics=metrics_dict
        )
        
        assert status.last_metrics == metrics_dict
        assert status.last_metrics["total_loss"] == 0.5


class TestTrainingMetrics:
    """Test training metrics model"""
    
    def test_training_metrics_creation(self):
        """Test training metrics model creation"""
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            total_loss=0.8,
            generation_loss=0.4,
            control_loss=0.2,
            memory_loss=0.1,
            speech_loss=0.1,
            learning_rate=1e-4,
            grad_norm=1.2,
            samples_per_second=8.5,
            timestamp=datetime.utcnow()
        )
        
        assert metrics.step == 100
        assert metrics.epoch == 1
        assert metrics.total_loss == 0.8
        assert metrics.generation_loss == 0.4
        assert metrics.control_loss == 0.2
        assert metrics.memory_loss == 0.1
        assert metrics.speech_loss == 0.1
        assert metrics.learning_rate == 1e-4
        assert metrics.grad_norm == 1.2
        assert metrics.samples_per_second == 8.5
    
    def test_training_metrics_optional_fields(self):
        """Test training metrics with optional fields"""
        metrics = TrainingMetrics(
            step=50,
            epoch=1,
            total_loss=1.2,
            learning_rate=5e-5,
            timestamp=datetime.utcnow()
        )
        
        assert metrics.step == 50
        assert metrics.total_loss == 1.2
        assert metrics.generation_loss is None
        assert metrics.grad_norm is None


class TestTrainingWebSocketManager:
    """Test WebSocket manager for training updates"""
    
    def test_websocket_manager_initialization(self):
        """Test WebSocket manager initialization"""
        manager = TrainingWebSocketManager()
        assert manager.connections == {}
    
    @pytest.mark.asyncio
    async def test_websocket_connect(self):
        """Test WebSocket connection management"""
        manager = TrainingWebSocketManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        
        await manager.connect(mock_websocket, "job-123")
        
        assert "job-123" in manager.connections
        assert mock_websocket in manager.connections["job-123"]
        mock_websocket.accept.assert_called_once()
    
    def test_websocket_disconnect(self):
        """Test WebSocket disconnection"""
        manager = TrainingWebSocketManager()
        mock_websocket = Mock(spec=WebSocket)
        
        # Add connection first
        manager.connections["job-123"] = [mock_websocket]
        
        # Disconnect
        manager.disconnect(mock_websocket, "job-123")
        
        assert "job-123" not in manager.connections
    
    @pytest.mark.asyncio
    async def test_broadcast_metrics(self):
        """Test broadcasting metrics to connected clients"""
        manager = TrainingWebSocketManager()
        mock_websocket1 = AsyncMock(spec=WebSocket)
        mock_websocket2 = AsyncMock(spec=WebSocket)
        
        # Setup connections
        manager.connections["job-123"] = [mock_websocket1, mock_websocket2]
        
        # Create test metrics
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            total_loss=0.5,
            learning_rate=1e-4,
            timestamp=datetime.utcnow()
        )
        
        # Broadcast metrics
        await manager.broadcast_metrics("job-123", metrics)
        
        # Verify both websockets received the message
        expected_message = {
            "type": "metrics",
            "data": metrics.model_dump()
        }
        
        mock_websocket1.send_json.assert_called_once()
        mock_websocket2.send_json.assert_called_once()
        
        # Check message content
        call_args = mock_websocket1.send_json.call_args[0][0]
        sent_message = call_args
        assert sent_message["type"] == "metrics"
        assert "data" in sent_message


class TestTrainingAPIEndpoints:
    """Test FastAPI training endpoints"""
    
    @pytest.fixture
    def training_client(self):
        """Create test client for training API"""
        from fastapi import FastAPI
        from backend.app.routers.quad_head_training import get_current_user, get_db
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock dependencies
        def mock_get_current_user():
            mock_user = Mock()
            mock_user.id = 1
            mock_user.username = "test_user"
            return mock_user
        
        def mock_get_db():
            return Mock()
        
        # Override dependencies
        app.dependency_overrides[get_current_user] = mock_get_current_user
        app.dependency_overrides[get_db] = mock_get_db
        
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth(self):
        """Mock authentication dependency - kept for compatibility"""
        return Mock()
    
    @pytest.fixture
    def mock_db(self):
        """Mock database dependency - kept for compatibility"""
        return Mock()
    
    def test_start_training_endpoint(self, training_client, mock_auth, mock_db):
        """Test starting a training job"""
        with patch('backend.app.routers.quad_head_training.QUAD_HEAD_AVAILABLE', True), \
             patch('pathlib.Path.exists', return_value=True):
            training_config = {
                "dataset_path": "/tmp/test_dataset",
                "base_model_name": "test/model",
                "learning_rate": 1e-4,
                "batch_size": 4,
                "num_train_epochs": 2
            }
            
            response = training_client.post("/api/training/quad-head/start", json=training_config)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "job_id" in data
            assert data["status"] == "pending"
            assert "created_at" in data
    
    def test_start_training_unavailable(self, training_client, mock_auth, mock_db):
        """Test starting training when quad-head model unavailable"""
        with patch('backend.app.routers.quad_head_training.QUAD_HEAD_AVAILABLE', False):
            training_config = {
                "dataset_path": "/tmp/test_dataset"
            }
            
            response = training_client.post("/api/training/quad-head/start", json=training_config)
            
            assert response.status_code == 501
            assert "QuadHeadNarrativeLM not available" in response.json()["detail"]
    
    def test_get_training_job_status(self, training_client, mock_auth, mock_db):
        """Test getting training job status"""
        # First create a job
        with patch('backend.app.routers.quad_head_training.QUAD_HEAD_AVAILABLE', True), \
             patch('pathlib.Path.exists', return_value=True):
            training_config = {"dataset_path": "/tmp/test"}
            
            create_response = training_client.post("/api/training/quad-head/start", json=training_config)
            job_id = create_response.json()["job_id"]
            
            # Get job status
            status_response = training_client.get(f"/api/training/quad-head/jobs/{job_id}")
            
            assert status_response.status_code == 200
            data = status_response.json()
            
            assert data["job_id"] == job_id
            assert "status" in data
            assert "created_at" in data
    
    def test_get_nonexistent_job(self, training_client, mock_auth, mock_db):
        """Test getting status of nonexistent job"""
        response = training_client.get("/api/training/quad-head/jobs/nonexistent-job")
        
        assert response.status_code == 404
        assert "Training job nonexistent-job not found" in response.json()["detail"]
    
    def test_list_training_jobs(self, training_client, mock_auth, mock_db):
        """Test listing all training jobs"""
        response = training_client.get("/api/training/quad-head/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_cancel_training_job(self, training_client, mock_auth, mock_db):
        """Test cancelling a training job"""
        # First create a job
        with patch('backend.app.routers.quad_head_training.QUAD_HEAD_AVAILABLE', True), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('asyncio.sleep'):  # Speed up the mock training loop
            training_config = {"dataset_path": "/tmp/test"}
            
            create_response = training_client.post("/api/training/quad-head/start", json=training_config)
            job_id = create_response.json()["job_id"]
            
            # Cancel the job immediately (before it completes)
            cancel_response = training_client.post(f"/api/training/quad-head/jobs/{job_id}/cancel")
            
            assert cancel_response.status_code == 200
            data = cancel_response.json()
            
            assert data["message"] == f"Training job {job_id} cancelled"
    
    def test_cancel_nonexistent_job(self, training_client, mock_auth, mock_db):
        """Test cancelling nonexistent job"""
        response = training_client.post("/api/training/quad-head/jobs/nonexistent-job/cancel")
        
        assert response.status_code == 404
        assert "Training job nonexistent-job not found" in response.json()["detail"]


class TestTrainingIntegration:
    """Integration tests for training functionality"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_training_workflow(self):
        """Test complete training workflow"""
        from backend.app.routers.quad_head_training import (
            training_jobs, training_ws_manager, run_quad_head_training
        )
        
        # Clear any existing jobs
        training_jobs.clear()
        
        # Create training config
        config = QuadHeadTrainingConfig(
            dataset_path="/tmp/test_dataset",
            num_train_epochs=1,
            batch_size=2
        )
        
        job_id = "test-workflow-job"
        
        # Create mock job
        job = TrainingStatus(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow()
        )
        training_jobs[job_id] = job
        
        # Mock the model creation and training
        with patch('backend.app.routers.quad_head_training.create_quad_head_model') as mock_create, \
             patch('backend.app.routers.quad_head_training.QuadHeadLoss') as mock_loss:
            
            mock_model = Mock()
            mock_create.return_value = mock_model
            mock_loss.return_value = Mock()
            
            # Start background training
            task = asyncio.create_task(run_quad_head_training(job_id, config, user_id=1))
            
            # Wait a bit for training to start
            await asyncio.sleep(0.2)
            
            # Check job progressed
            assert job.status == "running"
            assert job.progress > 0
            
            # Wait for completion
            await task
            
            # Verify final state
            assert job.status == "completed"
            assert job.progress == 100.0
            assert job.completed_at is not None
            assert job.model_path is not None
    
    @pytest.mark.asyncio 
    async def test_training_failure_handling(self):
        """Test training failure handling"""
        from backend.app.routers.quad_head_training import (
            training_jobs, run_quad_head_training
        )
        
        # Clear any existing jobs
        training_jobs.clear()
        
        config = QuadHeadTrainingConfig(dataset_path="/tmp/test")
        job_id = "test-failure-job"
        
        # Create job
        job = TrainingStatus(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow()
        )
        training_jobs[job_id] = job
        
        # Mock model creation to fail
        with patch('backend.app.routers.quad_head_training.create_quad_head_model') as mock_create:
            mock_create.side_effect = Exception("Model creation failed")
            
            # Start training (should fail)
            await run_quad_head_training(job_id, config, user_id=1)
            
            # Check failure state
            assert job.status == "failed"
            assert job.error_message == "Model creation failed"
            assert job.completed_at is not None
    
    def test_model_download_endpoint(self):
        """Test model download functionality"""
        from backend.app.routers.quad_head_training import training_jobs, get_current_user, get_db
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Setup test app with dependency overrides
        app = FastAPI()
        app.include_router(router)
        
        # Mock dependencies
        def mock_get_current_user():
            mock_user = Mock()
            mock_user.id = 1
            mock_user.username = "test_user"
            return mock_user
        
        def mock_get_db():
            return Mock()
        
        # Override dependencies
        app.dependency_overrides[get_current_user] = mock_get_current_user
        app.dependency_overrides[get_db] = mock_get_db
        
        client = TestClient(app)
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_file.write(b"mock model data")
            model_path = tmp_file.name
        
        try:
            # Setup completed job
            job_id = "download-test-job"
            job = TrainingStatus(
                job_id=job_id,
                status="completed",
                created_at=datetime.utcnow(),
                model_path=model_path
            )
            training_jobs[job_id] = job
            
            # Test download (auth is handled by dependency override)
            response = client.get(f"/api/training/quad-head/models/{job_id}/download")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"
            assert "attachment" in response.headers["content-disposition"]
            assert response.content == b"mock model data"
        
        finally:
            # Cleanup
            Path(model_path).unlink(missing_ok=True)
            if job_id in training_jobs:
                del training_jobs[job_id]
    
    def test_download_nonexistent_model(self):
        """Test downloading nonexistent model"""
        from backend.app.routers.quad_head_training import get_current_user, get_db
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock dependencies
        def mock_get_current_user():
            mock_user = Mock()
            mock_user.id = 1
            return mock_user
        
        def mock_get_db():
            return Mock()
        
        # Override dependencies
        app.dependency_overrides[get_current_user] = mock_get_current_user
        app.dependency_overrides[get_db] = mock_get_db
        
        client = TestClient(app)
        
        response = client.get("/api/training/quad-head/models/nonexistent-job/download")
        
        assert response.status_code == 404
        assert "Training job nonexistent-job not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__]) 