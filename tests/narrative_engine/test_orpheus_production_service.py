"""
Tests for Orpheus Production Service

Tests the dual-mode OrpheusProductionService for both platform TTS
and data generation workloads, including integration with TTSOrchestrator.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import time as dt_time
from typing import Dict, List, Any

# Mock dependencies that may not be available
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies for testing"""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.cuda.empty_cache'), \
         patch('psutil.virtual_memory') as mock_psutil:
        
        # Mock psutil memory info
        mock_memory = Mock()
        mock_memory.used = 1024 * 1024 * 1024  # 1GB
        mock_psutil.return_value = mock_memory
        
        yield


class TestServiceMode:
    """Test ServiceMode enum"""
    
    def test_service_modes(self):
        """Test ServiceMode enum values"""
        from narrative_engine.orpheus_production_service import ServiceMode
        
        assert ServiceMode.PLATFORM.value == "platform"
        assert ServiceMode.DATA_GENERATION.value == "data_generation"


class TestWorkloadScheduler:
    """Test WorkloadScheduler functionality"""
    
    def test_scheduler_init(self):
        """Test WorkloadScheduler initialization"""
        from narrative_engine.orpheus_production_service import WorkloadScheduler
        
        scheduler = WorkloadScheduler()
        assert scheduler.platform_start == dt_time(8, 0)
        assert scheduler.platform_end == dt_time(22, 0)
        assert scheduler.manual_override is None
    
    def test_platform_hours(self):
        """Test platform mode during business hours"""
        from narrative_engine.orpheus_production_service import WorkloadScheduler, ServiceMode
        
        scheduler = WorkloadScheduler()
        
        # Mock current time to 10 AM
        with patch('narrative_engine.orpheus_production_service.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(10, 0)
            mode = scheduler.get_current_mode()
            assert mode == ServiceMode.PLATFORM
    
    def test_data_generation_hours(self):
        """Test data generation mode during night hours"""
        from narrative_engine.orpheus_production_service import WorkloadScheduler, ServiceMode
        
        scheduler = WorkloadScheduler()
        
        # Mock current time to 2 AM
        with patch('narrative_engine.orpheus_production_service.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = dt_time(2, 0)
            mode = scheduler.get_current_mode()
            assert mode == ServiceMode.DATA_GENERATION
    
    def test_manual_override(self):
        """Test manual override functionality"""
        from narrative_engine.orpheus_production_service import WorkloadScheduler, ServiceMode
        
        scheduler = WorkloadScheduler()
        
        # Set manual override
        scheduler.set_manual_override(ServiceMode.DATA_GENERATION)
        assert scheduler.get_current_mode() == ServiceMode.DATA_GENERATION
        
        # Clear override
        scheduler.clear_manual_override()
        assert scheduler.manual_override is None


class TestSynthesisRequest:
    """Test SynthesisRequest dataclass"""
    
    def test_request_creation(self):
        """Test SynthesisRequest creation"""
        from narrative_engine.orpheus_production_service import SynthesisRequest
        
        request = SynthesisRequest(
            text="Hello world",
            emotion_tags=["laugh"],
            voice_id="test_voice"
        )
        
        assert request.text == "Hello world"
        assert request.emotion_tags == ["laugh"]
        assert request.voice_id == "test_voice"
        assert request.priority == 0
        assert request.request_id.startswith("req_")
    
    def test_request_auto_id(self):
        """Test automatic request ID generation"""
        from narrative_engine.orpheus_production_service import SynthesisRequest
        
        request1 = SynthesisRequest(text="Test 1")
        request2 = SynthesisRequest(text="Test 2")
        
        assert request1.request_id != request2.request_id
        assert request1.request_id.startswith("req_")
        assert request2.request_id.startswith("req_")


class TestOrpheusProductionService:
    """Test OrpheusProductionService functionality"""
    
    @pytest.fixture
    def service(self):
        """Create test service instance"""
        from narrative_engine.orpheus_production_service import OrpheusProductionService
        return OrpheusProductionService()
    
    def test_service_init(self, service):
        """Test service initialization"""
        assert service.model_path == "canopylabs/orpheus-3b-0.1-ft"
        assert service.current_mode.value == "platform"
        assert service.is_running is False
        assert service.model is None
        assert service.model_loaded is False
        assert len(service.emotion_tags) == 8
    
    def test_emotion_tag_mapping(self, service):
        """Test emotion tag processing"""
        # Test with emotion tags
        tagged = service._add_emotion_tags("Hello world", ["laugh"])
        assert tagged == "<laugh> Hello world"
        
        # Test with multiple tags (should use first)
        tagged = service._add_emotion_tags("Hello", ["laugh", "sigh"])
        assert tagged == "<laugh> Hello"
        
        # Test without emotion tags
        tagged = service._add_emotion_tags("Hello", [])
        assert tagged == "Hello"
        
        # Test with unknown tag
        tagged = service._add_emotion_tags("Hello", ["unknown"])
        assert tagged == "Hello"
    
    def test_mock_audio_generation(self, service):
        """Test mock audio generation"""
        audio, sr = service._generate_mock_audio("Test text")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 22050
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0  # Normalized
    
    @pytest.mark.asyncio
    async def test_load_model_fallback(self, service):
        """Test model loading with fallback"""
        # Mock transformers import to fail
        with patch('transformers.AutoModelForCausalLM') as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Model not found")
            
            result = await service._load_model()
            assert result is False
            assert service.model is None
            assert service.model_loaded is False
    
    @pytest.mark.asyncio
    async def test_synthesize_request(self, service):
        """Test synthesis request processing"""
        from narrative_engine.orpheus_production_service import SynthesisRequest
        
        # Mock model loading to avoid actual model
        service.model = None
        service.model_loaded = False
        
        request = SynthesisRequest(
            text="Hello world",
            emotion_tags=["laugh"]
        )
        
        result = await service._process_request(request)
        
        assert result.request_id == request.request_id
        assert result.model_used == "orpheus-production"
        assert isinstance(result.audio, np.ndarray)
        assert result.sample_rate == 22050
        assert result.duration_seconds > 0
        assert result.metadata["tagged_text"] == "<laugh> Hello world"
    
    def test_get_status(self, service):
        """Test service status reporting"""
        status = service.get_status()
        
        assert "running" in status
        assert "mode" in status
        assert "model_loaded" in status
        assert "quantization_enabled" in status
        assert "metrics" in status
        assert "queues" in status
        assert "cache" in status
        
        assert status["running"] is False
        assert status["mode"] == "platform"
        assert status["model_loaded"] is False


class TestIntegrationWithTTSOrchestrator:
    """Test integration with existing TTS system"""
    
    @pytest.mark.asyncio
    async def test_orpheus_tts_production_service_integration(self):
        """Test OrpheusTTS provider with production service"""
        # Mock the production service
        with patch('narrative_engine.orpheus_production_service.get_orpheus_service') as mock_get_service:
            mock_service = Mock()
            mock_result = Mock()
            mock_result.audio = np.random.randn(1000).astype(np.float32)
            mock_result.sample_rate = 22050
            mock_result.duration_seconds = 0.5
            
            mock_service.synthesize = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service
            
            from narrative_engine.tts_integration import OrpheusTTS
            
            # Create provider with production service enabled
            orpheus = OrpheusTTS(use_production_service=True)
            
            audio, sr = await orpheus.synthesize(
                text="Hello there",
                emotion_tags=["laugh"]
            )
            
            assert isinstance(audio, np.ndarray)
            assert sr == 22050
            assert len(audio) == 1000
    
    @pytest.mark.asyncio
    async def test_orpheus_tts_fallback_to_direct(self):
        """Test OrpheusTTS fallback to direct model when production service fails"""
        # Mock production service to fail
        with patch('narrative_engine.orpheus_production_service.get_orpheus_service') as mock_get_service:
            mock_get_service.side_effect = Exception("Service unavailable")
            
            from narrative_engine.tts_integration import OrpheusTTS
            
            # Create provider with production service enabled
            orpheus = OrpheusTTS(use_production_service=True)
            
            # Mock direct model loading
            with patch.object(orpheus, '_load_model'), \
                 patch.object(orpheus, '_generate_mock_audio') as mock_generate:
                
                mock_generate.return_value = (np.zeros(500, dtype=np.float32), 22050)
                
                audio, sr = await orpheus.synthesize(
                    text="Hello there",
                    emotion_tags=["laugh"]
                )
                
                assert isinstance(audio, np.ndarray)
                assert sr == 22050
                assert len(audio) == 500
                
                # Should have called mock generation
                mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tts_orchestrator_with_production_orpheus(self):
        """Test TTSOrchestrator using production Orpheus service"""
        from narrative_engine.tts_integration import TTSOrchestrator
        
        orchestrator = TTSOrchestrator()
        
        # Mock the Orpheus provider
        mock_audio = np.random.randn(1500).astype(np.float32)
        mock_orpheus = Mock()
        mock_orpheus.synthesize = AsyncMock(return_value=(mock_audio, 22050))
        orchestrator.providers["orpheus"] = mock_orpheus
        
        character = {"id": "test_char", "gender": "female", "name": "Alice"}
        
        # Should select Orpheus for emotion tags
        audio, sr = await orchestrator.synthesize_character_voice(
            text="Oh wow!",
            character=character,
            emotion_tags=["gasp", "surprise"]
        )
        
        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert len(audio) == 1500
        
        # Verify Orpheus was used with correct parameters
        mock_orpheus.synthesize.assert_called_once_with(
            text="Oh wow!",
            voice_id="expressive_female",
            emotion_tags=["gasp", "surprise"]
        )


class TestDualModeWorkflow:
    """Test dual-mode workflow scenarios"""
    
    @pytest.mark.asyncio
    async def test_mode_switching_workflow(self):
        """Test the complete mode switching workflow"""
        from narrative_engine.orpheus_production_service import (
            OrpheusProductionService, 
            ServiceMode,
            SynthesisRequest
        )
        
        service = OrpheusProductionService()
        
        # Mock model loading
        with patch.object(service, '_load_model', return_value=True), \
             patch.object(service, '_unload_model'):
            
            # Test platform mode
            service.current_mode = ServiceMode.PLATFORM
            
            request = SynthesisRequest(text="User query", emotion_tags=[])
            result = await service._process_request(request)
            
            assert result.metadata["mode"] == "platform"
            
            # Switch to data generation mode
            await service._switch_mode(ServiceMode.DATA_GENERATION)
            assert service.current_mode == ServiceMode.DATA_GENERATION
            
            # Test batch processing
            requests = [
                SynthesisRequest(text=f"Sample {i}", emotion_tags=["laugh"])
                for i in range(5)
            ]
            
            results = await service.synthesize_batch(requests)
            assert len(results) == 5
            
            for result in results:
                assert result.metadata["mode"] == "data_generation"
                assert result.metadata["tagged_text"].startswith("<laugh>")


# Performance tests (marked as slow)
@pytest.mark.slow
class TestPerformanceMetrics:
    """Test performance monitoring and metrics"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test performance metrics collection"""
        from narrative_engine.orpheus_production_service import (
            OrpheusProductionService,
            SynthesisRequest
        )
        
        service = OrpheusProductionService()
        
        # Process multiple requests
        for i in range(5):
            request = SynthesisRequest(text=f"Test {i}")
            await service._process_request(request)
        
        # Check metrics
        assert service.metrics.requests_processed == 5
        assert service.metrics.avg_latency > 0
        assert service.metrics.total_latency > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """Test memory usage monitoring"""
        from narrative_engine.orpheus_production_service import OrpheusProductionService
        
        service = OrpheusProductionService()
        memory_stats = service._get_memory_usage()
        
        assert "system_mb" in memory_stats
        assert "gpu_allocated_mb" in memory_stats
        assert "gpu_reserved_mb" in memory_stats
        
        assert isinstance(memory_stats["system_mb"], (int, float))
        assert memory_stats["system_mb"] >= 0 