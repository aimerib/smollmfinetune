"""
Streaming Evaluation Service for Quad-Head Multimodal Architecture

This module provides comprehensive evaluation metrics for real-time multimodal generation:
- Speech quality evaluation (mel-spectrogram analysis)
- Text-speech alignment metrics
- Real-time performance monitoring
- Character voice consistency evaluation
- Cross-modal coherence assessment
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class SpeechQualityMetrics:
    """Metrics for evaluating speech generation quality"""
    mel_spectral_distance: float
    mel_cepstral_distortion: float
    spectral_convergence: float
    log_magnitude_error: float
    signal_to_noise_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TextSpeechAlignmentMetrics:
    """Metrics for evaluating text-speech temporal alignment"""
    alignment_score: float  # 0-1, higher is better
    temporal_drift: float   # ms of drift from expected timing
    phoneme_duration_accuracy: float  # How well speech matches expected phoneme timing
    cross_modal_attention_score: float  # Attention alignment between text and speech
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RealTimePerformanceMetrics:
    """Metrics for real-time streaming performance"""
    generation_latency_ms: float
    streaming_throughput_fps: float  # Frames per second
    memory_usage_mb: float
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    network_latency_ms: float
    buffer_underruns: int
    frame_drops: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VoiceConsistencyMetrics:
    """Metrics for character voice consistency evaluation"""
    voice_similarity_score: float  # 0-1, similarity to target voice profile
    prosody_consistency: float     # Consistency of pitch, rhythm, stress
    emotional_consistency: float   # Consistency with intended emotion
    character_voice_deviation: float  # Deviation from character's established voice
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrossModalCoherenceMetrics:
    """Metrics for evaluating coherence between modalities"""
    semantic_alignment: float       # How well speech content matches text
    emotional_alignment: float      # How well speech emotion matches text emotion  
    timing_synchronization: float   # How well speech timing matches text pacing
    control_signal_compliance: float # How well speech follows control signals
    overall_coherence_score: float  # Combined coherence metric
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StreamingGenerationSample:
    """Single sample from streaming generation for evaluation"""
    text_tokens: List[str]
    speech_frames: List[np.ndarray]  # Mel-spectrogram frames
    control_signals: List[str]
    memory_updates: List[Dict[str, Any]]
    character_id: str
    generation_timestamp: datetime
    total_generation_time_ms: float


class StreamingEvaluationService:
    """Service for evaluating multimodal streaming generation quality"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 mel_bins: int = 80,
                 hop_length: int = 256,
                 evaluation_window_size: int = 100):
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.hop_length = hop_length
        self.evaluation_window_size = evaluation_window_size
        
        # Evaluation history for trend analysis
        self.speech_quality_history: List[SpeechQualityMetrics] = []
        self.alignment_history: List[TextSpeechAlignmentMetrics] = []
        self.performance_history: List[RealTimePerformanceMetrics] = []
        self.voice_consistency_history: List[VoiceConsistencyMetrics] = []
        self.coherence_history: List[CrossModalCoherenceMetrics] = []
        
        # Reference models for comparison
        self.reference_voice_profiles: Dict[str, np.ndarray] = {}
        self.baseline_performance_metrics: Optional[RealTimePerformanceMetrics] = None
        
        logger.info("StreamingEvaluationService initialized")

    async def evaluate_streaming_sample(self, 
                                       sample: StreamingGenerationSample,
                                       reference_audio: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a streaming generation sample
        
        Args:
            sample: Generated sample to evaluate
            reference_audio: Optional reference audio for comparison
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        evaluation_results = {}
        
        try:
            # Evaluate speech quality
            if sample.speech_frames:
                speech_quality = await self._evaluate_speech_quality(
                    sample.speech_frames, reference_audio
                )
                evaluation_results['speech_quality'] = speech_quality
                self.speech_quality_history.append(speech_quality)
            
            # Evaluate text-speech alignment
            if sample.text_tokens and sample.speech_frames:
                alignment_metrics = await self._evaluate_text_speech_alignment(
                    sample.text_tokens, sample.speech_frames
                )
                evaluation_results['text_speech_alignment'] = alignment_metrics
                self.alignment_history.append(alignment_metrics)
            
            # Evaluate real-time performance
            performance_metrics = await self._evaluate_realtime_performance(sample)
            evaluation_results['realtime_performance'] = performance_metrics
            self.performance_history.append(performance_metrics)
            
            # Evaluate voice consistency
            if sample.character_id and sample.speech_frames:
                voice_consistency = await self._evaluate_voice_consistency(
                    sample.character_id, sample.speech_frames
                )
                evaluation_results['voice_consistency'] = voice_consistency
                self.voice_consistency_history.append(voice_consistency)
            
            # Evaluate cross-modal coherence
            coherence_metrics = await self._evaluate_cross_modal_coherence(sample)
            evaluation_results['cross_modal_coherence'] = coherence_metrics
            self.coherence_history.append(coherence_metrics)
            
            # Calculate overall quality score
            evaluation_results['overall_quality_score'] = self._calculate_overall_quality(
                evaluation_results
            )
            
            logger.info(f"Completed evaluation for sample from {sample.character_id}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            evaluation_results['error'] = str(e)
        
        return evaluation_results

    async def _evaluate_speech_quality(self, 
                                     speech_frames: List[np.ndarray],
                                     reference_audio: Optional[np.ndarray] = None) -> SpeechQualityMetrics:
        """Evaluate speech generation quality using spectral analysis"""
        
        if not speech_frames:
            return SpeechQualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Concatenate frames into full mel-spectrogram
        mel_spec = np.concatenate(speech_frames, axis=-1)  # Shape: (mel_bins, time_frames)
        
        # Calculate spectral metrics
        mel_spectral_distance = self._calculate_mel_spectral_distance(mel_spec, reference_audio)
        mel_cepstral_distortion = self._calculate_mel_cepstral_distortion(mel_spec)
        spectral_convergence = self._calculate_spectral_convergence(mel_spec)
        log_magnitude_error = self._calculate_log_magnitude_error(mel_spec)
        snr = self._calculate_signal_to_noise_ratio(mel_spec)
        
        return SpeechQualityMetrics(
            mel_spectral_distance=mel_spectral_distance,
            mel_cepstral_distortion=mel_cepstral_distortion,
            spectral_convergence=spectral_convergence,
            log_magnitude_error=log_magnitude_error,
            signal_to_noise_ratio=snr
        )

    async def _evaluate_text_speech_alignment(self, 
                                            text_tokens: List[str],
                                            speech_frames: List[np.ndarray]) -> TextSpeechAlignmentMetrics:
        """Evaluate temporal alignment between text and speech"""
        
        # Estimate expected speech duration based on text
        expected_duration_frames = len(text_tokens) * 2.5  # ~2.5 frames per token average
        actual_duration_frames = len(speech_frames)
        
        # Calculate alignment score
        duration_ratio = min(actual_duration_frames, expected_duration_frames) / max(actual_duration_frames, expected_duration_frames)
        alignment_score = duration_ratio
        
        # Calculate temporal drift
        temporal_drift = abs(actual_duration_frames - expected_duration_frames) * (self.hop_length / self.sample_rate) * 1000
        
        # Estimate phoneme duration accuracy (simplified)
        phoneme_duration_accuracy = self._estimate_phoneme_duration_accuracy(text_tokens, speech_frames)
        
        # Placeholder for cross-modal attention (would require attention weights from model)
        cross_modal_attention_score = 0.8  # Placeholder
        
        return TextSpeechAlignmentMetrics(
            alignment_score=alignment_score,
            temporal_drift=temporal_drift,
            phoneme_duration_accuracy=phoneme_duration_accuracy,
            cross_modal_attention_score=cross_modal_attention_score
        )

    async def _evaluate_realtime_performance(self, 
                                           sample: StreamingGenerationSample) -> RealTimePerformanceMetrics:
        """Evaluate real-time streaming performance metrics"""
        
        # Calculate generation latency
        generation_latency_ms = sample.total_generation_time_ms
        
        # Calculate streaming throughput
        if sample.speech_frames and generation_latency_ms > 0:
            throughput_fps = len(sample.speech_frames) / (generation_latency_ms / 1000.0)
        else:
            throughput_fps = 0.0
        
        # Get system metrics (simplified - would use psutil in production)
        memory_usage_mb = 512.0  # Placeholder
        cpu_utilization = 45.0   # Placeholder
        gpu_utilization = 78.0   # Placeholder
        network_latency = 25.0   # Placeholder
        
        return RealTimePerformanceMetrics(
            generation_latency_ms=generation_latency_ms,
            streaming_throughput_fps=throughput_fps,
            memory_usage_mb=memory_usage_mb,
            cpu_utilization_percent=cpu_utilization,
            gpu_utilization_percent=gpu_utilization,
            network_latency_ms=network_latency,
            buffer_underruns=0,
            frame_drops=0
        )

    async def _evaluate_voice_consistency(self, 
                                        character_id: str,
                                        speech_frames: List[np.ndarray]) -> VoiceConsistencyMetrics:
        """Evaluate consistency with character voice profile"""
        
        if not speech_frames:
            return VoiceConsistencyMetrics(0.0, 0.0, 0.0, 0.0)
        
        # Get reference voice profile for character
        reference_profile = self.reference_voice_profiles.get(character_id)
        
        if reference_profile is None:
            # Create reference profile from current sample
            mel_spec = np.concatenate(speech_frames, axis=-1)
            self.reference_voice_profiles[character_id] = self._extract_voice_features(mel_spec)
            reference_profile = self.reference_voice_profiles[character_id]
        
        # Extract features from current sample
        mel_spec = np.concatenate(speech_frames, axis=-1)
        current_features = self._extract_voice_features(mel_spec)
        
        # Calculate similarity scores
        voice_similarity = self._calculate_cosine_similarity(current_features, reference_profile)
        prosody_consistency = self._calculate_prosody_consistency(mel_spec)
        emotional_consistency = 0.85  # Placeholder - would analyze emotional markers
        voice_deviation = 1.0 - voice_similarity
        
        return VoiceConsistencyMetrics(
            voice_similarity_score=voice_similarity,
            prosody_consistency=prosody_consistency,
            emotional_consistency=emotional_consistency,
            character_voice_deviation=voice_deviation
        )

    async def _evaluate_cross_modal_coherence(self, 
                                            sample: StreamingGenerationSample) -> CrossModalCoherenceMetrics:
        """Evaluate coherence between different modalities"""
        
        # Semantic alignment between text and speech
        semantic_alignment = self._calculate_semantic_alignment(
            sample.text_tokens, sample.speech_frames
        )
        
        # Emotional alignment
        emotional_alignment = self._calculate_emotional_alignment(
            sample.text_tokens, sample.speech_frames, sample.control_signals
        )
        
        # Timing synchronization
        timing_sync = self._calculate_timing_synchronization(
            sample.text_tokens, sample.speech_frames
        )
        
        # Control signal compliance
        control_compliance = self._calculate_control_signal_compliance(
            sample.control_signals, sample.speech_frames
        )
        
        # Overall coherence (weighted combination)
        overall_coherence = (
            semantic_alignment * 0.3 +
            emotional_alignment * 0.25 +
            timing_sync * 0.25 +
            control_compliance * 0.2
        )
        
        return CrossModalCoherenceMetrics(
            semantic_alignment=semantic_alignment,
            emotional_alignment=emotional_alignment,
            timing_synchronization=timing_sync,
            control_signal_compliance=control_compliance,
            overall_coherence_score=overall_coherence
        )

    def _calculate_overall_quality(self, evaluation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from all metrics"""
        
        weights = {
            'speech_quality': 0.25,
            'text_speech_alignment': 0.20,
            'realtime_performance': 0.15,
            'voice_consistency': 0.20,
            'cross_modal_coherence': 0.20
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_type, weight in weights.items():
            if metric_type in evaluation_results:
                metric_data = evaluation_results[metric_type]
                
                if metric_type == 'speech_quality':
                    # Normalize speech quality metrics (higher SNR is better, lower distortion is better)
                    score = metric_data.signal_to_noise_ratio / 30.0  # Normalize assuming max SNR ~30dB
                    score = min(1.0, max(0.0, score))
                    
                elif metric_type == 'text_speech_alignment':
                    score = metric_data.alignment_score
                    
                elif metric_type == 'realtime_performance':
                    # Performance score based on throughput and latency
                    throughput_score = min(1.0, metric_data.streaming_throughput_fps / 50.0)  # Target 50 FPS
                    latency_score = max(0.0, 1.0 - metric_data.generation_latency_ms / 1000.0)  # Target <1s
                    score = (throughput_score + latency_score) / 2
                    
                elif metric_type == 'voice_consistency':
                    score = metric_data.voice_similarity_score
                    
                elif metric_type == 'cross_modal_coherence':
                    score = metric_data.overall_coherence_score
                
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    # Helper methods for metric calculations
    def _calculate_mel_spectral_distance(self, mel_spec: np.ndarray, reference: Optional[np.ndarray]) -> float:
        """Calculate spectral distance metric"""
        if reference is None:
            # Use variance as quality indicator
            return float(np.var(mel_spec))
        # Would implement actual spectral distance calculation
        return 0.1

    def _calculate_mel_cepstral_distortion(self, mel_spec: np.ndarray) -> float:
        """Calculate mel-cepstral distortion"""
        # Simplified implementation
        mel_cepstrum = np.fft.fft(mel_spec, axis=0)
        return float(np.mean(np.abs(mel_cepstrum)))

    def _calculate_spectral_convergence(self, mel_spec: np.ndarray) -> float:
        """Calculate spectral convergence metric"""
        # Measure how well the spectrum converges
        spectral_diff = np.diff(mel_spec, axis=-1)
        return 1.0 / (1.0 + float(np.mean(np.abs(spectral_diff))))

    def _calculate_log_magnitude_error(self, mel_spec: np.ndarray) -> float:
        """Calculate log magnitude error"""
        log_mel = np.log(mel_spec + 1e-8)
        return float(np.mean(np.abs(log_mel)))

    def _calculate_signal_to_noise_ratio(self, mel_spec: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        signal_power = np.mean(mel_spec ** 2)
        noise_power = np.var(mel_spec)
        return 10 * np.log10(signal_power / (noise_power + 1e-8))

    def _estimate_phoneme_duration_accuracy(self, text_tokens: List[str], speech_frames: List[np.ndarray]) -> float:
        """Estimate how well phoneme durations match expected values"""
        # Simplified implementation
        expected_phonemes = len(''.join(text_tokens)) * 0.7  # Rough phoneme estimate
        actual_frames = len(speech_frames)
        expected_frames = expected_phonemes * 3  # ~3 frames per phoneme
        
        accuracy = min(actual_frames, expected_frames) / max(actual_frames, expected_frames)
        return float(accuracy)

    def _extract_voice_features(self, mel_spec: np.ndarray) -> np.ndarray:
        """Extract voice characteristic features from mel-spectrogram"""
        # Extract statistical features across time
        features = np.concatenate([
            np.mean(mel_spec, axis=-1),      # Mean mel coefficients
            np.std(mel_spec, axis=-1),       # Std mel coefficients  
            np.mean(np.diff(mel_spec), axis=-1),  # Dynamic features
        ])
        return features

    def _calculate_cosine_similarity(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        dot_product = np.dot(features_a, features_b)
        norm_a = np.linalg.norm(features_a)
        norm_b = np.linalg.norm(features_b)
        return float(dot_product / (norm_a * norm_b + 1e-8))

    def _calculate_prosody_consistency(self, mel_spec: np.ndarray) -> float:
        """Calculate prosody consistency score"""
        # Analyze pitch and rhythm patterns
        pitch_contour = np.mean(mel_spec[40:60], axis=0)  # Focus on pitch-relevant bins
        pitch_variance = np.var(pitch_contour)
        rhythm_consistency = 1.0 / (1.0 + pitch_variance)
        return float(rhythm_consistency)

    def _calculate_semantic_alignment(self, text_tokens: List[str], speech_frames: List[np.ndarray]) -> float:
        """Calculate semantic alignment between text and speech"""
        # Placeholder - would use semantic similarity models
        return 0.85

    def _calculate_emotional_alignment(self, text_tokens: List[str], speech_frames: List[np.ndarray], control_signals: List[str]) -> float:
        """Calculate emotional alignment across modalities"""
        # Placeholder - would analyze emotional markers
        return 0.82

    def _calculate_timing_synchronization(self, text_tokens: List[str], speech_frames: List[np.ndarray]) -> float:
        """Calculate timing synchronization score"""
        expected_duration = len(text_tokens) * 2.5
        actual_duration = len(speech_frames)
        sync_score = min(actual_duration, expected_duration) / max(actual_duration, expected_duration)
        return float(sync_score)

    def _calculate_control_signal_compliance(self, control_signals: List[str], speech_frames: List[np.ndarray]) -> float:
        """Calculate how well speech follows control signals"""
        # Placeholder - would analyze control signal implementation
        return 0.88

    async def get_evaluation_summary(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of recent evaluation metrics"""
        
        window = window_size or self.evaluation_window_size
        
        summary = {
            'evaluation_window_size': window,
            'total_samples_evaluated': len(self.performance_history),
            'timestamp': datetime.now()
        }
        
        # Speech quality summary
        if self.speech_quality_history:
            recent_speech = self.speech_quality_history[-window:]
            summary['speech_quality'] = {
                'avg_snr': np.mean([m.signal_to_noise_ratio for m in recent_speech]),
                'avg_spectral_distance': np.mean([m.mel_spectral_distance for m in recent_speech]),
                'trend_improving': len(recent_speech) > 1 and recent_speech[-1].signal_to_noise_ratio > recent_speech[0].signal_to_noise_ratio
            }
        
        # Performance summary
        if self.performance_history:
            recent_perf = self.performance_history[-window:]
            summary['performance'] = {
                'avg_latency_ms': np.mean([m.generation_latency_ms for m in recent_perf]),
                'avg_throughput_fps': np.mean([m.streaming_throughput_fps for m in recent_perf]),
                'avg_memory_usage_mb': np.mean([m.memory_usage_mb for m in recent_perf])
            }
        
        # Voice consistency summary
        if self.voice_consistency_history:
            recent_voice = self.voice_consistency_history[-window:]
            summary['voice_consistency'] = {
                'avg_similarity': np.mean([m.voice_similarity_score for m in recent_voice]),
                'avg_prosody_consistency': np.mean([m.prosody_consistency for m in recent_voice])
            }
        
        return summary 