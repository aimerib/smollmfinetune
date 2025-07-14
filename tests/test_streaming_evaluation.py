"""
Tests for Streaming Evaluation Service

Tests the comprehensive multimodal evaluation capabilities for quad-head architecture.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from backend.app.services.evaluation.streaming_evaluation import (
    StreamingEvaluationService,
    StreamingGenerationSample,
    SpeechQualityMetrics,
    TextSpeechAlignmentMetrics,
    RealTimePerformanceMetrics,
    VoiceConsistencyMetrics,
    CrossModalCoherenceMetrics
)


class TestStreamingEvaluationService:
    """Test the streaming evaluation service functionality"""
    
    @pytest.fixture
    def evaluation_service(self):
        """Create evaluation service instance for testing"""
        return StreamingEvaluationService(
            sample_rate=22050,
            mel_bins=80,
            hop_length=256,
            evaluation_window_size=10
        )
    
    @pytest.fixture
    def sample_generation(self):
        """Create sample generation data for testing"""
        # Generate realistic mel-spectrogram frames
        speech_frames = [
            np.random.rand(80, 4) for _ in range(10)  # 10 frames of 80-dimensional mel
        ]
        
        return StreamingGenerationSample(
            text_tokens=['Hello', 'world', 'this', 'is', 'a', 'test'],
            speech_frames=speech_frames,
            control_signals=['emotion_happy', 'pace_normal', 'tone_friendly'],
            memory_updates=[{'key': 'memory_item', 'value': 'test_memory'}],
            character_id='test_character',
            generation_timestamp=datetime.now(),
            total_generation_time_ms=150.0
        )

    def test_service_initialization(self, evaluation_service):
        """Test that service initializes correctly"""
        assert evaluation_service.sample_rate == 22050
        assert evaluation_service.mel_bins == 80
        assert evaluation_service.hop_length == 256
        assert evaluation_service.evaluation_window_size == 10
        assert len(evaluation_service.speech_quality_history) == 0
        assert len(evaluation_service.alignment_history) == 0
        assert len(evaluation_service.performance_history) == 0

    @pytest.mark.asyncio
    async def test_evaluate_streaming_sample_complete(self, evaluation_service, sample_generation):
        """Test complete evaluation of a streaming sample"""
        results = await evaluation_service.evaluate_streaming_sample(sample_generation)
        
        # Check that all evaluation components are present
        assert 'speech_quality' in results
        assert 'text_speech_alignment' in results
        assert 'realtime_performance' in results
        assert 'voice_consistency' in results
        assert 'cross_modal_coherence' in results
        assert 'overall_quality_score' in results
        
        # Check that metrics are of correct type
        assert isinstance(results['speech_quality'], SpeechQualityMetrics)
        assert isinstance(results['text_speech_alignment'], TextSpeechAlignmentMetrics)
        assert isinstance(results['realtime_performance'], RealTimePerformanceMetrics)
        assert isinstance(results['voice_consistency'], VoiceConsistencyMetrics)
        assert isinstance(results['cross_modal_coherence'], CrossModalCoherenceMetrics)
        assert isinstance(results['overall_quality_score'], float)
        
        # Check that overall quality score is in valid range
        assert 0.0 <= results['overall_quality_score'] <= 1.0

    @pytest.mark.asyncio
    async def test_speech_quality_evaluation(self, evaluation_service, sample_generation):
        """Test speech quality evaluation metrics"""
        speech_quality = await evaluation_service._evaluate_speech_quality(
            sample_generation.speech_frames
        )
        
        assert isinstance(speech_quality, SpeechQualityMetrics)
        assert isinstance(speech_quality.mel_spectral_distance, float)
        assert isinstance(speech_quality.mel_cepstral_distortion, float)
        assert isinstance(speech_quality.spectral_convergence, float)
        assert isinstance(speech_quality.log_magnitude_error, float)
        assert isinstance(speech_quality.signal_to_noise_ratio, float)
        assert isinstance(speech_quality.timestamp, datetime)
        
        # Check reasonable ranges for metrics
        assert speech_quality.spectral_convergence > 0
        assert speech_quality.signal_to_noise_ratio > -50  # Reasonable SNR lower bound

    @pytest.mark.asyncio
    async def test_text_speech_alignment_evaluation(self, evaluation_service, sample_generation):
        """Test text-speech alignment evaluation"""
        alignment = await evaluation_service._evaluate_text_speech_alignment(
            sample_generation.text_tokens,
            sample_generation.speech_frames
        )
        
        assert isinstance(alignment, TextSpeechAlignmentMetrics)
        assert 0.0 <= alignment.alignment_score <= 1.0
        assert alignment.temporal_drift >= 0.0
        assert 0.0 <= alignment.phoneme_duration_accuracy <= 1.0
        assert 0.0 <= alignment.cross_modal_attention_score <= 1.0

    @pytest.mark.asyncio
    async def test_realtime_performance_evaluation(self, evaluation_service, sample_generation):
        """Test real-time performance evaluation"""
        performance = await evaluation_service._evaluate_realtime_performance(sample_generation)
        
        assert isinstance(performance, RealTimePerformanceMetrics)
        assert performance.generation_latency_ms >= 0.0
        assert performance.streaming_throughput_fps >= 0.0
        assert performance.memory_usage_mb >= 0.0
        assert 0.0 <= performance.cpu_utilization_percent <= 100.0
        assert 0.0 <= performance.gpu_utilization_percent <= 100.0
        assert performance.network_latency_ms >= 0.0
        assert performance.buffer_underruns >= 0
        assert performance.frame_drops >= 0

    @pytest.mark.asyncio
    async def test_voice_consistency_evaluation(self, evaluation_service, sample_generation):
        """Test voice consistency evaluation"""
        voice_consistency = await evaluation_service._evaluate_voice_consistency(
            sample_generation.character_id,
            sample_generation.speech_frames
        )
        
        assert isinstance(voice_consistency, VoiceConsistencyMetrics)
        assert 0.0 <= voice_consistency.voice_similarity_score <= 1.0
        assert 0.0 <= voice_consistency.prosody_consistency <= 1.0
        assert 0.0 <= voice_consistency.emotional_consistency <= 1.0
        assert voice_consistency.character_voice_deviation >= 0.0

    @pytest.mark.asyncio
    async def test_cross_modal_coherence_evaluation(self, evaluation_service, sample_generation):
        """Test cross-modal coherence evaluation"""
        coherence = await evaluation_service._evaluate_cross_modal_coherence(sample_generation)
        
        assert isinstance(coherence, CrossModalCoherenceMetrics)
        assert 0.0 <= coherence.semantic_alignment <= 1.0
        assert 0.0 <= coherence.emotional_alignment <= 1.0
        assert 0.0 <= coherence.timing_synchronization <= 1.0
        assert 0.0 <= coherence.control_signal_compliance <= 1.0
        assert 0.0 <= coherence.overall_coherence_score <= 1.0

    def test_overall_quality_calculation(self, evaluation_service):
        """Test overall quality score calculation"""
        mock_results = {
            'speech_quality': SpeechQualityMetrics(0.1, 0.2, 0.8, 0.3, 20.0),
            'text_speech_alignment': TextSpeechAlignmentMetrics(0.85, 50.0, 0.9, 0.8),
            'realtime_performance': RealTimePerformanceMetrics(100.0, 30.0, 512.0, 45.0, 78.0, 25.0, 0, 0),
            'voice_consistency': VoiceConsistencyMetrics(0.9, 0.85, 0.88, 0.1),
            'cross_modal_coherence': CrossModalCoherenceMetrics(0.85, 0.82, 0.9, 0.88, 0.86)
        }
        
        quality_score = evaluation_service._calculate_overall_quality(mock_results)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

    def test_history_tracking(self, evaluation_service, sample_generation):
        """Test that evaluation history is properly tracked"""
        initial_lengths = {
            'speech_quality': len(evaluation_service.speech_quality_history),
            'alignment': len(evaluation_service.alignment_history),
            'performance': len(evaluation_service.performance_history),
            'voice_consistency': len(evaluation_service.voice_consistency_history),
            'coherence': len(evaluation_service.coherence_history)
        }
        
        # Run evaluation (this is a coroutine, so we need to handle it properly)
        import asyncio
        asyncio.run(evaluation_service.evaluate_streaming_sample(sample_generation))
        
        # Check that history lengths increased
        assert len(evaluation_service.speech_quality_history) == initial_lengths['speech_quality'] + 1
        assert len(evaluation_service.alignment_history) == initial_lengths['alignment'] + 1
        assert len(evaluation_service.performance_history) == initial_lengths['performance'] + 1
        assert len(evaluation_service.voice_consistency_history) == initial_lengths['voice_consistency'] + 1
        assert len(evaluation_service.coherence_history) == initial_lengths['coherence'] + 1

    @pytest.mark.asyncio
    async def test_evaluation_summary(self, evaluation_service, sample_generation):
        """Test evaluation summary generation"""
        # Add some sample evaluations
        for _ in range(5):
            await evaluation_service.evaluate_streaming_sample(sample_generation)
        
        summary = await evaluation_service.get_evaluation_summary(window_size=3)
        
        assert 'evaluation_window_size' in summary
        assert 'total_samples_evaluated' in summary
        assert 'timestamp' in summary
        assert summary['evaluation_window_size'] == 3
        assert summary['total_samples_evaluated'] >= 5
        
        # Check component summaries
        if 'speech_quality' in summary:
            assert 'avg_snr' in summary['speech_quality']
            assert 'avg_spectral_distance' in summary['speech_quality']
            assert 'trend_improving' in summary['speech_quality']
        
        if 'performance' in summary:
            assert 'avg_latency_ms' in summary['performance']
            assert 'avg_throughput_fps' in summary['performance']
            assert 'avg_memory_usage_mb' in summary['performance']

    def test_mel_spectral_distance_calculation(self, evaluation_service):
        """Test mel spectral distance calculation"""
        mel_spec = np.random.rand(80, 100)
        distance = evaluation_service._calculate_mel_spectral_distance(mel_spec, None)
        
        assert isinstance(distance, float)
        assert distance >= 0.0

    def test_signal_to_noise_ratio_calculation(self, evaluation_service):
        """Test SNR calculation"""
        # Create a signal with known properties
        signal = np.ones((80, 100)) + 0.1 * np.random.randn(80, 100)  # Signal + noise
        snr = evaluation_service._calculate_signal_to_noise_ratio(signal)
        
        assert isinstance(snr, float)
        assert snr > 0  # Should be positive for signal > noise

    def test_voice_feature_extraction(self, evaluation_service):
        """Test voice feature extraction"""
        mel_spec = np.random.rand(80, 100)
        features = evaluation_service._extract_voice_features(mel_spec)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (240,)  # 80 + 80 + 80 features
        assert not np.any(np.isnan(features))

    def test_cosine_similarity_calculation(self, evaluation_service):
        """Test cosine similarity calculation"""
        # Test with identical vectors
        vec_a = np.array([1, 2, 3, 4, 5])
        vec_b = np.array([1, 2, 3, 4, 5])
        similarity = evaluation_service._calculate_cosine_similarity(vec_a, vec_b)
        
        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical vectors
        
        # Test with orthogonal vectors
        vec_c = np.array([1, 0, 0])
        vec_d = np.array([0, 1, 0])
        similarity_ortho = evaluation_service._calculate_cosine_similarity(vec_c, vec_d)
        
        assert abs(similarity_ortho) < 1e-6  # Should be 0.0 for orthogonal vectors

    def test_prosody_consistency_calculation(self, evaluation_service):
        """Test prosody consistency calculation"""
        # Create mel spectrogram with consistent prosody
        mel_spec = np.random.rand(80, 100)
        consistency = evaluation_service._calculate_prosody_consistency(mel_spec)
        
        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0

    @pytest.mark.asyncio
    async def test_evaluation_with_empty_speech_frames(self, evaluation_service):
        """Test evaluation handles empty speech frames gracefully"""
        sample = StreamingGenerationSample(
            text_tokens=['hello'],
            speech_frames=[],  # Empty speech frames
            control_signals=[],
            memory_updates=[],
            character_id='test_character',
            generation_timestamp=datetime.now(),
            total_generation_time_ms=50.0
        )
        
        results = await evaluation_service.evaluate_streaming_sample(sample)
        
        # Should still return results without crashing
        assert 'realtime_performance' in results
        assert 'overall_quality_score' in results

    @pytest.mark.asyncio
    async def test_evaluation_error_handling(self, evaluation_service):
        """Test that evaluation handles errors gracefully"""
        # Create a sample that might cause issues
        sample = StreamingGenerationSample(
            text_tokens=[],
            speech_frames=[np.array([])],  # Malformed speech frame
            control_signals=[],
            memory_updates=[],
            character_id='',
            generation_timestamp=datetime.now(),
            total_generation_time_ms=0.0
        )
        
        # Should not raise exception
        results = await evaluation_service.evaluate_streaming_sample(sample)
        
        # Should handle error gracefully - either have overall score or error message
        assert 'overall_quality_score' in results or 'error' in results

    def test_reference_voice_profile_creation(self, evaluation_service, sample_generation):
        """Test that reference voice profiles are created and stored"""
        character_id = sample_generation.character_id
        
        # Initially no reference profile
        assert character_id not in evaluation_service.reference_voice_profiles
        
        # Run evaluation (this should create a reference profile)
        import asyncio
        asyncio.run(evaluation_service._evaluate_voice_consistency(
            character_id, sample_generation.speech_frames
        ))
        
        # Reference profile should now exist
        assert character_id in evaluation_service.reference_voice_profiles
        assert isinstance(evaluation_service.reference_voice_profiles[character_id], np.ndarray)

    def test_timing_synchronization_calculation(self, evaluation_service):
        """Test timing synchronization calculation"""
        text_tokens = ['hello', 'world', 'test']
        speech_frames = [np.random.rand(80, 4) for _ in range(8)]  # Close to expected length
        
        sync_score = evaluation_service._calculate_timing_synchronization(text_tokens, speech_frames)
        
        assert isinstance(sync_score, float)
        assert 0.0 <= sync_score <= 1.0


class TestSpeechQualityMetrics:
    """Test speech quality metrics dataclass"""
    
    def test_speech_quality_metrics_creation(self):
        """Test creating speech quality metrics"""
        metrics = SpeechQualityMetrics(
            mel_spectral_distance=0.1,
            mel_cepstral_distortion=0.2,
            spectral_convergence=0.8,
            log_magnitude_error=0.3,
            signal_to_noise_ratio=15.5
        )
        
        assert metrics.mel_spectral_distance == 0.1
        assert metrics.mel_cepstral_distortion == 0.2
        assert metrics.spectral_convergence == 0.8
        assert metrics.log_magnitude_error == 0.3
        assert metrics.signal_to_noise_ratio == 15.5
        assert isinstance(metrics.timestamp, datetime)


class TestStreamingGenerationSample:
    """Test streaming generation sample dataclass"""
    
    def test_sample_creation(self):
        """Test creating a streaming generation sample"""
        speech_frames = [np.random.rand(80, 4) for _ in range(5)]
        
        sample = StreamingGenerationSample(
            text_tokens=['hello', 'world'],
            speech_frames=speech_frames,
            control_signals=['emotion_happy'],
            memory_updates=[{'key': 'value'}],
            character_id='test_char',
            generation_timestamp=datetime.now(),
            total_generation_time_ms=120.0
        )
        
        assert len(sample.text_tokens) == 2
        assert len(sample.speech_frames) == 5
        assert len(sample.control_signals) == 1
        assert len(sample.memory_updates) == 1
        assert sample.character_id == 'test_char'
        assert sample.total_generation_time_ms == 120.0
        assert isinstance(sample.generation_timestamp, datetime) 