"""
Audio Chunk Processor

Processes audio chunks for smart buffering including format conversion,
resampling, and volume normalization.
"""

import asyncio
from typing import Dict, Any, List, Optional


class AudioChunkProcessor:
    """Process audio chunks for buffering and playback"""
    
    def __init__(
        self,
        format_support: List[str] = None,
        resampling_enabled: bool = True,
        normalization_enabled: bool = True
    ):
        """
        Initialize audio chunk processor
        
        Args:
            format_support: List of supported audio formats
            resampling_enabled: Enable audio resampling
            normalization_enabled: Enable volume normalization
        """
        self.supported_formats = format_support or ["pcm", "opus", "aac", "mp3"]
        self.resampling_enabled = resampling_enabled
        self.normalization_enabled = normalization_enabled
        
        # Default audio parameters
        self.default_sample_rate = 44100
        self.default_channels = 1
        self.target_peak_amplitude = 0.8
    
    async def process_chunk(
        self,
        chunk: Dict[str, Any],
        target_format: Optional[str] = None,
        target_sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process audio chunk with format conversion and normalization
        
        Args:
            chunk: Raw audio chunk data
            target_format: Target audio format (optional)
            target_sample_rate: Target sample rate (optional)
            
        Returns:
            Processed audio chunk
        """
        processed_chunk = chunk.copy()
        
        # Format conversion
        if target_format and target_format != chunk.get("format"):
            processed_chunk = await self._convert_format(processed_chunk, target_format)
        
        # Resampling
        if (target_sample_rate and self.resampling_enabled and 
            target_sample_rate != chunk.get("sample_rate")):
            processed_chunk = await self._resample_audio(processed_chunk, target_sample_rate)
        
        # Volume normalization
        if self.normalization_enabled:
            processed_chunk = await self._normalize_volume(processed_chunk)
        
        # Add processing metadata
        processed_chunk["processed_audio_data"] = processed_chunk.get("audio_data")
        processed_chunk["processing_timestamp"] = asyncio.get_event_loop().time()
        
        return processed_chunk
    
    async def _convert_format(self, chunk: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """
        Convert audio chunk to target format
        
        Args:
            chunk: Audio chunk to convert
            target_format: Target audio format
            
        Returns:
            Converted audio chunk
        """
        # Mock format conversion (in real implementation, would use audio libraries)
        converted_chunk = chunk.copy()
        
        if target_format in self.supported_formats:
            converted_chunk["format"] = target_format
            # Simulate format conversion by updating metadata
            if target_format == "pcm":
                converted_chunk["sample_rate"] = converted_chunk.get("sample_rate", self.default_sample_rate)
                converted_chunk["channels"] = converted_chunk.get("channels", self.default_channels)
            elif target_format == "opus":
                # Opus typically uses 48kHz
                converted_chunk["sample_rate"] = 48000
            elif target_format == "aac":
                # AAC can use various sample rates
                converted_chunk["sample_rate"] = converted_chunk.get("sample_rate", 44100)
        
        return converted_chunk
    
    async def _resample_audio(self, chunk: Dict[str, Any], target_sample_rate: int) -> Dict[str, Any]:
        """
        Resample audio chunk to target sample rate
        
        Args:
            chunk: Audio chunk to resample
            target_sample_rate: Target sample rate in Hz
            
        Returns:
            Resampled audio chunk
        """
        # Mock resampling (in real implementation, would use audio libraries like librosa)
        resampled_chunk = chunk.copy()
        
        current_rate = chunk.get("sample_rate", self.default_sample_rate)
        if current_rate != target_sample_rate:
            # Simulate resampling by updating sample rate
            resampled_chunk["sample_rate"] = target_sample_rate
            
            # Adjust duration proportionally
            if "duration_ms" in chunk:
                # Duration stays the same in real resampling, but data length changes
                resampled_chunk["duration_ms"] = chunk["duration_ms"]
            
            # Add resampling metadata
            resampled_chunk["resampling_applied"] = True
            resampled_chunk["original_sample_rate"] = current_rate
        
        return resampled_chunk
    
    async def _normalize_volume(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize audio volume to target level
        
        Args:
            chunk: Audio chunk to normalize
            
        Returns:
            Volume-normalized audio chunk
        """
        # Mock volume normalization (in real implementation, would analyze and adjust amplitude)
        normalized_chunk = chunk.copy()
        
        current_amplitude = chunk.get("peak_amplitude", 0.5)
        
        if current_amplitude > self.target_peak_amplitude:
            # Reduce volume
            normalization_factor = self.target_peak_amplitude / current_amplitude
            normalized_chunk["peak_amplitude"] = self.target_peak_amplitude
            normalized_chunk["normalization_applied"] = True
            normalized_chunk["normalization_factor"] = normalization_factor
        elif current_amplitude < self.target_peak_amplitude * 0.5:
            # Boost volume if too quiet
            normalization_factor = self.target_peak_amplitude / current_amplitude
            normalized_chunk["peak_amplitude"] = min(self.target_peak_amplitude, current_amplitude * normalization_factor)
            normalized_chunk["normalization_applied"] = True
            normalized_chunk["normalization_factor"] = normalization_factor
        else:
            # Volume is acceptable
            normalized_chunk["peak_amplitude"] = current_amplitude
            normalized_chunk["normalization_applied"] = False
        
        return normalized_chunk
    
    def is_format_supported(self, format_name: str) -> bool:
        """Check if audio format is supported"""
        return format_name.lower() in self.supported_formats
    
    def get_recommended_format(self, use_case: str = "streaming") -> str:
        """Get recommended audio format for use case"""
        format_recommendations = {
            "streaming": "opus",  # Good compression for streaming
            "playback": "pcm",    # Uncompressed for immediate playback
            "storage": "aac",     # Good compression for storage
            "realtime": "pcm"     # Low latency for real-time
        }
        
        recommended = format_recommendations.get(use_case, "pcm")
        
        # Fall back to PCM if recommended format not supported
        if recommended not in self.supported_formats:
            return "pcm" if "pcm" in self.supported_formats else self.supported_formats[0]
        
        return recommended
    
    def estimate_processing_time(self, chunk: Dict[str, Any]) -> float:
        """
        Estimate processing time for audio chunk
        
        Args:
            chunk: Audio chunk to estimate processing time for
            
        Returns:
            Estimated processing time in milliseconds
        """
        base_time = 5.0  # Base processing overhead
        
        # Add time based on operations needed
        format_conversion_time = 10.0 if chunk.get("format") != "pcm" else 0.0
        resampling_time = 15.0 if self.resampling_enabled else 0.0
        normalization_time = 5.0 if self.normalization_enabled else 0.0
        
        # Scale by chunk duration
        duration_ms = chunk.get("duration_ms", 100)
        duration_factor = duration_ms / 100  # Normalize to 100ms chunks
        
        total_time = (base_time + format_conversion_time + resampling_time + normalization_time) * duration_factor
        
        return total_time 