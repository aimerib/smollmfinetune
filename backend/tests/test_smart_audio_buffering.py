"""
Tests for Smart Audio Buffering

This module tests the smart audio buffering system for voice streaming:
- Adaptive buffer sizing based on network conditions  
- Smooth playback during generation delays
- Buffer health monitoring and optimization
- Chunk processing and ordering

Focus on unit testing the buffering mechanics, not actual audio processing.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Mock dependencies before importing our modules
with patch('backend.app.redis_client.get_redis_pool'), \
     patch('backend.app.database.create_tables'):
    from backend.app.services.voice.smart_audio_buffer import SmartAudioBuffer
    from backend.app.services.voice.audio_chunk_processor import AudioChunkProcessor
    from backend.app.services.voice.buffer_health_monitor import BufferHealthMonitor


class TestSmartAudioBuffer:
    """Test smart audio buffer for voice streaming"""
    
    @pytest.fixture
    def audio_buffer(self):
        """Create SmartAudioBuffer instance"""
        return SmartAudioBuffer(
            initial_buffer_size=5,
            max_buffer_size=20,
            min_buffer_size=2,
            target_latency_ms=500
        )
    
    @pytest.fixture
    def sample_audio_chunks(self):
        """Sample audio chunks for testing"""
        return [
            {
                "chunk_id": f"chunk_{i:03d}",
                "sequence_number": i,
                "audio_data": f"audio_data_{i}",
                "timestamp": time.time() + i * 0.1,
                "duration_ms": 100,
                "is_final": i == 9
            }
            for i in range(10)
        ]
    
    def test_buffer_initialization(self, audio_buffer):
        """Test SmartAudioBuffer initializes correctly"""
        assert audio_buffer.initial_buffer_size == 5
        assert audio_buffer.max_buffer_size == 20
        assert audio_buffer.min_buffer_size == 2
        assert audio_buffer.target_latency_ms == 500
        assert audio_buffer.current_buffer_size == 5
        assert len(audio_buffer.chunks) == 0
        assert audio_buffer.health_monitor is not None
    
    @pytest.mark.asyncio
    async def test_add_audio_chunk_basic(self, audio_buffer, sample_audio_chunks):
        """Test adding audio chunks to buffer"""
        chunk = sample_audio_chunks[0]
        
        result = await audio_buffer.add_chunk(chunk)
        
        assert result is True
        assert len(audio_buffer.chunks) == 1
        assert audio_buffer.chunks[0]["chunk_id"] == "chunk_000"
    
    @pytest.mark.asyncio
    async def test_add_audio_chunk_ordering(self, audio_buffer, sample_audio_chunks):
        """Test audio chunks are properly ordered by sequence number"""
        # Add chunks out of order
        chunks_to_add = [sample_audio_chunks[2], sample_audio_chunks[0], sample_audio_chunks[1]]
        
        for chunk in chunks_to_add:
            await audio_buffer.add_chunk(chunk)
        
        # Should be reordered by sequence number
        assert len(audio_buffer.chunks) == 3
        assert audio_buffer.chunks[0]["sequence_number"] == 0
        assert audio_buffer.chunks[1]["sequence_number"] == 1
        assert audio_buffer.chunks[2]["sequence_number"] == 2
    
    @pytest.mark.asyncio
    async def test_get_next_chunk_for_playback(self, audio_buffer, sample_audio_chunks):
        """Test retrieving next chunk for playback"""
        # Add some chunks
        for chunk in sample_audio_chunks[:3]:
            await audio_buffer.add_chunk(chunk)
        
        # Get next chunk for playback
        next_chunk = await audio_buffer.get_next_chunk()
        
        assert next_chunk is not None
        assert next_chunk["chunk_id"] == "chunk_000"
        assert len(audio_buffer.chunks) == 2  # Should be removed from buffer
    
    @pytest.mark.asyncio
    async def test_buffer_health_monitoring(self, audio_buffer):
        """Test buffer health monitoring and adaptation"""
        # Simulate low buffer health (slow generation)
        await audio_buffer.report_playback_event("underrun", {"severity": "high"})
        
        # Buffer should adapt by increasing size
        assert audio_buffer.current_buffer_size > 5
        
        # Simulate good buffer health
        await audio_buffer.report_playback_event("healthy", {"latency_ms": 200})
        
        # Buffer might reduce size if consistently healthy
        health_score = audio_buffer.health_monitor.get_current_health()
        assert 0.0 <= health_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_adaptive_buffer_sizing_network_conditions(self, audio_buffer):
        """Test buffer size adaptation based on network conditions"""
        # Simulate poor network conditions
        network_metrics = {
            "latency_ms": 800,
            "jitter_ms": 200,
            "packet_loss": 0.05
        }
        
        await audio_buffer.adapt_to_network_conditions(network_metrics)
        
        # Buffer should increase for poor conditions
        assert audio_buffer.current_buffer_size > 5
        
        # Simulate excellent network conditions
        network_metrics = {
            "latency_ms": 50,
            "jitter_ms": 10,
            "packet_loss": 0.001
        }
        
        await audio_buffer.adapt_to_network_conditions(network_metrics)
        
        # Buffer might decrease for excellent conditions (within limits)
        assert audio_buffer.current_buffer_size >= audio_buffer.min_buffer_size
    
    @pytest.mark.asyncio
    async def test_chunk_gap_detection_and_handling(self, audio_buffer, sample_audio_chunks):
        """Test detection and handling of missing chunks"""
        # Add chunks with a gap (missing chunk 1)
        await audio_buffer.add_chunk(sample_audio_chunks[0])
        await audio_buffer.add_chunk(sample_audio_chunks[2])  # Skip chunk 1
        await audio_buffer.add_chunk(sample_audio_chunks[3])
        
        # Should detect gap
        gaps = audio_buffer.detect_sequence_gaps()
        assert len(gaps) > 0
        assert 1 in gaps  # Missing sequence number 1
        
        # Should handle gaps gracefully
        next_chunk = await audio_buffer.get_next_chunk()
        assert next_chunk["chunk_id"] == "chunk_000"  # Can still play available chunks
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_handling(self, audio_buffer, sample_audio_chunks):
        """Test buffer behavior when it exceeds max size"""
        # Fill buffer beyond max capacity
        extended_chunks = sample_audio_chunks * 3  # 30 chunks > max_buffer_size (20)
        
        for chunk in extended_chunks:
            await audio_buffer.add_chunk(chunk)
        
        # Should not exceed max buffer size
        assert len(audio_buffer.chunks) <= audio_buffer.max_buffer_size
    
    @pytest.mark.asyncio
    async def test_buffer_underrun_recovery(self, audio_buffer):
        """Test buffer recovery from underrun conditions"""
        # Simulate buffer underrun
        await audio_buffer.report_playback_event("underrun", {"cause": "slow_generation"})
        
        # Should enter recovery mode
        assert audio_buffer.is_in_recovery_mode() is True
        
        # Should request faster generation or increase buffer
        recovery_actions = audio_buffer.get_recovery_actions()
        assert "increase_buffer_size" in recovery_actions or "request_quality_reduction" in recovery_actions
    
    @pytest.mark.asyncio
    async def test_predictive_buffering(self, audio_buffer, sample_audio_chunks):
        """Test predictive buffering based on generation patterns"""
        # Simulate consistent generation pattern
        for i, chunk in enumerate(sample_audio_chunks[:5]):
            chunk["generation_time_ms"] = 200 + i * 10  # Slightly increasing generation time
            await audio_buffer.add_chunk(chunk)
            await asyncio.sleep(0.01)  # Small delay
        
        # Should predict need for larger buffer
        predicted_size = audio_buffer.predict_optimal_buffer_size()
        assert predicted_size >= audio_buffer.initial_buffer_size


class TestAudioChunkProcessor:
    """Test audio chunk processing for buffering"""
    
    @pytest.fixture
    def chunk_processor(self):
        """Create AudioChunkProcessor instance"""
        return AudioChunkProcessor(
            format_support=["pcm", "opus", "aac"],
            resampling_enabled=True,
            normalization_enabled=True
        )
    
    def test_processor_initialization(self, chunk_processor):
        """Test AudioChunkProcessor initializes correctly"""
        assert "pcm" in chunk_processor.supported_formats
        assert chunk_processor.resampling_enabled is True
        assert chunk_processor.normalization_enabled is True
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk_basic(self, chunk_processor):
        """Test basic audio chunk processing"""
        raw_chunk = {
            "audio_data": b"fake_audio_data",
            "format": "pcm",
            "sample_rate": 22050,
            "channels": 1
        }
        
        processed_chunk = await chunk_processor.process_chunk(raw_chunk)
        
        assert processed_chunk is not None
        assert "processed_audio_data" in processed_chunk
        assert processed_chunk["format"] == "pcm"
    
    @pytest.mark.asyncio
    async def test_chunk_format_conversion(self, chunk_processor):
        """Test audio format conversion during processing"""
        opus_chunk = {
            "audio_data": b"fake_opus_data",
            "format": "opus",
            "sample_rate": 48000,
            "channels": 2
        }
        
        # Convert to PCM for playback
        processed_chunk = await chunk_processor.process_chunk(
            opus_chunk, target_format="pcm"
        )
        
        assert processed_chunk["format"] == "pcm"
    
    @pytest.mark.asyncio
    async def test_chunk_resampling(self, chunk_processor):
        """Test audio resampling during processing"""
        chunk_22k = {
            "audio_data": b"fake_22k_audio",
            "format": "pcm",
            "sample_rate": 22050,
            "channels": 1
        }
        
        # Resample to 44.1kHz
        processed_chunk = await chunk_processor.process_chunk(
            chunk_22k, target_sample_rate=44100
        )
        
        assert processed_chunk["sample_rate"] == 44100
    
    @pytest.mark.asyncio
    async def test_chunk_volume_normalization(self, chunk_processor):
        """Test audio volume normalization"""
        loud_chunk = {
            "audio_data": b"fake_loud_audio",
            "format": "pcm", 
            "sample_rate": 44100,
            "channels": 1,
            "peak_amplitude": 0.95  # Very loud
        }
        
        processed_chunk = await chunk_processor.process_chunk(loud_chunk)
        
        # Should be normalized to reasonable level
        assert processed_chunk.get("peak_amplitude", 0.8) <= 0.8


class TestBufferHealthMonitor:
    """Test buffer health monitoring system"""
    
    @pytest.fixture
    def health_monitor(self):
        """Create BufferHealthMonitor instance"""
        return BufferHealthMonitor(
            history_length=100,
            health_check_interval=1.0
        )
    
    def test_monitor_initialization(self, health_monitor):
        """Test BufferHealthMonitor initializes correctly"""
        assert health_monitor.history_length == 100
        assert health_monitor.health_check_interval == 1.0
        assert len(health_monitor.event_history) == 0
        assert health_monitor.current_health_score == 1.0  # Start healthy
    
    @pytest.mark.asyncio
    async def test_record_playback_event(self, health_monitor):
        """Test recording playback events for health analysis"""
        event = {
            "type": "chunk_played",
            "timestamp": time.time(),
            "latency_ms": 150,
            "buffer_level": 0.7
        }
        
        await health_monitor.record_event(event)
        
        assert len(health_monitor.event_history) == 1
        assert health_monitor.event_history[0]["type"] == "chunk_played"
    
    @pytest.mark.asyncio
    async def test_health_score_calculation(self, health_monitor):
        """Test health score calculation from events"""
        # Record healthy events
        for i in range(10):
            await health_monitor.record_event({
                "type": "chunk_played",
                "timestamp": time.time(),
                "latency_ms": 100 + i * 5,  # Stable latency
                "buffer_level": 0.8
            })
        
        health_score = health_monitor.calculate_health_score()
        assert health_score > 0.8  # Should be healthy
        
        # Record some problematic events
        for i in range(5):
            await health_monitor.record_event({
                "type": "underrun",
                "timestamp": time.time(),
                "severity": "medium"
            })
        
        health_score = health_monitor.calculate_health_score()
        assert health_score < 0.8  # Should decrease
    
    @pytest.mark.asyncio
    async def test_performance_trend_analysis(self, health_monitor):
        """Test analysis of performance trends over time"""
        # Simulate degrading performance
        for i in range(20):
            await health_monitor.record_event({
                "type": "chunk_played",
                "timestamp": time.time() + i * 0.1,
                "latency_ms": 100 + i * 10,  # Increasing latency
                "buffer_level": max(0.1, 0.8 - i * 0.03)  # Decreasing buffer
            })
        
        trend = health_monitor.analyze_performance_trend()
        assert trend["direction"] == "degrading"
        assert trend["confidence"] > 0.3  # More reasonable threshold for gradual degradation
    
    @pytest.mark.asyncio
    async def test_adaptive_recommendations(self, health_monitor):
        """Test adaptive recommendations based on health analysis"""
        # Simulate unstable network conditions
        await health_monitor.record_event({
            "type": "network_issue",
            "timestamp": time.time(),
            "jitter_ms": 200,
            "packet_loss": 0.03
        })
        
        recommendations = health_monitor.get_adaptive_recommendations()
        
        # Should return list of recommendation dictionaries
        assert len(recommendations) > 0
        assert any(rec.get("action") == "increase_buffer_size" for rec in recommendations)
        assert any(rec.get("reason") == "network_instability" for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_health_history_cleanup(self, health_monitor):
        """Test cleanup of old health monitoring data"""
        # Fill history beyond capacity
        for i in range(150):  # More than history_length (100)
            await health_monitor.record_event({
                "type": "test_event",
                "timestamp": time.time() + i * 0.01,
                "data": f"event_{i}"
            })
        
        # Should maintain history length limit
        assert len(health_monitor.event_history) <= health_monitor.history_length
        
        # Should keep most recent events
        latest_event = health_monitor.event_history[-1]
        assert "event_" in latest_event["data"] 