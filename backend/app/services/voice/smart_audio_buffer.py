"""
Smart Audio Buffer

Intelligent audio buffering system for voice streaming that adapts to
network conditions, generation speed, and playback health.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from collections import deque

from backend.app.services.voice.buffer_health_monitor import BufferHealthMonitor


class SmartAudioBuffer:
    """Smart audio buffer with adaptive sizing and health monitoring"""
    
    def __init__(
        self,
        initial_buffer_size: int = 5,
        max_buffer_size: int = 20,
        min_buffer_size: int = 2,
        target_latency_ms: int = 500
    ):
        """
        Initialize smart audio buffer
        
        Args:
            initial_buffer_size: Starting buffer size in chunks
            max_buffer_size: Maximum allowed buffer size
            min_buffer_size: Minimum required buffer size
            target_latency_ms: Target playback latency in milliseconds
        """
        self.initial_buffer_size = initial_buffer_size
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.target_latency_ms = target_latency_ms
        
        # Current buffer state
        self.current_buffer_size = initial_buffer_size
        self.chunks: List[Dict[str, Any]] = []
        self.next_sequence_number = 0
        
        # Buffer management
        self.health_monitor = BufferHealthMonitor()
        self.is_recovery_mode = False
        self.generation_times = deque(maxlen=20)  # Track generation patterns
        
        # Performance tracking
        self.total_chunks_processed = 0
        self.last_adaptation_time = time.time()
    
    async def add_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Add audio chunk to buffer with intelligent ordering
        
        Args:
            chunk: Audio chunk data with sequence_number
            
        Returns:
            True if chunk was added successfully
        """
        # Track generation timing if available
        if "generation_time_ms" in chunk:
            self.generation_times.append(chunk["generation_time_ms"])
        
        # Insert chunk in correct sequence order
        inserted = False
        for i, existing_chunk in enumerate(self.chunks):
            if chunk["sequence_number"] < existing_chunk["sequence_number"]:
                self.chunks.insert(i, chunk)
                inserted = True
                break
        
        if not inserted:
            self.chunks.append(chunk)
        
        # Enforce max buffer size
        if len(self.chunks) > self.max_buffer_size:
            # Remove oldest chunks beyond buffer size
            self.chunks = self.chunks[-self.max_buffer_size:]
        
        self.total_chunks_processed += 1
        
        # Update buffer health based on current state
        await self._update_buffer_health()
        
        return True
    
    async def get_next_chunk(self) -> Optional[Dict[str, Any]]:
        """
        Get next chunk for playback, removing it from buffer
        
        Returns:
            Next audio chunk or None if buffer is empty
        """
        if not self.chunks:
            # Buffer underrun - report to health monitor
            await self.report_playback_event("underrun", {"cause": "empty_buffer"})
            return None
        
        # Get first chunk (lowest sequence number)
        chunk = self.chunks.pop(0)
        
        # Update next expected sequence number
        self.next_sequence_number = chunk["sequence_number"] + 1
        
        # Report successful chunk playback
        await self.report_playback_event("chunk_played", {
            "sequence_number": chunk["sequence_number"],
            "buffer_level": len(self.chunks) / self.current_buffer_size,
            "latency_ms": chunk.get("latency_ms", 100)
        })
        
        return chunk
    
    def detect_sequence_gaps(self) -> List[int]:
        """
        Detect missing sequence numbers in current buffer
        
        Returns:
            List of missing sequence numbers
        """
        if not self.chunks:
            return []
        
        sequence_numbers = [chunk["sequence_number"] for chunk in self.chunks]
        sequence_numbers.sort()
        
        gaps = []
        for i in range(len(sequence_numbers) - 1):
            current = sequence_numbers[i]
            next_seq = sequence_numbers[i + 1]
            
            # Check for gaps
            for missing in range(current + 1, next_seq):
                gaps.append(missing)
        
        return gaps
    
    async def report_playback_event(self, event_type: str, data: Dict[str, Any]):
        """
        Report playback event for health monitoring and adaptation
        
        Args:
            event_type: Type of event (underrun, chunk_played, etc.)
            data: Event-specific data
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            **data
        }
        
        await self.health_monitor.record_event(event)
        
        # Adapt buffer size based on event
        if event_type == "underrun":
            severity = data.get("severity", "medium")
            if severity == "high":
                await self._increase_buffer_size(factor=1.5)
                self.is_recovery_mode = True
            else:
                await self._increase_buffer_size(factor=1.2)
                self.is_recovery_mode = True  # Any underrun triggers recovery mode
        
        elif event_type == "healthy":
            # Gradually reduce buffer if consistently healthy
            if self.health_monitor.get_current_health() > 0.9:
                await self._decrease_buffer_size(factor=0.95)
                self.is_recovery_mode = False
    
    async def adapt_to_network_conditions(self, metrics: Dict[str, Any]):
        """
        Adapt buffer size based on network conditions
        
        Args:
            metrics: Network metrics (latency_ms, jitter_ms, packet_loss)
        """
        latency = metrics.get("latency_ms", 100)
        jitter = metrics.get("jitter_ms", 10)
        packet_loss = metrics.get("packet_loss", 0.0)
        
        # Calculate network quality score
        quality_score = self._calculate_network_quality(latency, jitter, packet_loss)
        
        # Adapt buffer based on network quality
        if quality_score < 0.3:  # Poor network
            target_size = min(self.max_buffer_size, self.current_buffer_size * 1.4)
        elif quality_score < 0.7:  # Fair network
            target_size = self.current_buffer_size * 1.1
        else:  # Good network
            target_size = max(self.min_buffer_size, self.current_buffer_size * 0.95)
        
        self.current_buffer_size = int(target_size)
    
    def is_in_recovery_mode(self) -> bool:
        """Check if buffer is in recovery mode"""
        return self.is_recovery_mode
    
    def get_recovery_actions(self) -> List[str]:
        """Get recommended recovery actions"""
        actions = []
        
        if self.is_recovery_mode:
            actions.append("increase_buffer_size")
            
            # Check if generation is slow
            if self.generation_times and sum(self.generation_times) / len(self.generation_times) > 300:
                actions.append("request_quality_reduction")
            
            # Check buffer level
            buffer_level = len(self.chunks) / self.current_buffer_size
            if buffer_level < 0.3:
                actions.append("request_faster_generation")
        
        return actions
    
    def predict_optimal_buffer_size(self) -> int:
        """
        Predict optimal buffer size based on generation patterns
        
        Returns:
            Predicted optimal buffer size
        """
        if not self.generation_times:
            return self.initial_buffer_size
        
        # Calculate average generation time
        avg_generation_time = sum(self.generation_times) / len(self.generation_times)
        
        # Predict buffer size needed to maintain smooth playback
        # Assume 100ms chunks, target 500ms buffer
        chunks_per_second = 1000 / 100  # 10 chunks per second
        generation_chunks_per_second = 1000 / avg_generation_time
        
        # Need buffer to cover generation delay
        buffer_ratio = chunks_per_second / generation_chunks_per_second
        predicted_size = int(self.initial_buffer_size * buffer_ratio)
        
        # Clamp to valid range
        return max(self.min_buffer_size, min(self.max_buffer_size, predicted_size))
    
    def _calculate_network_quality(self, latency: float, jitter: float, packet_loss: float) -> float:
        """Calculate network quality score (0.0 to 1.0)"""
        # Normalize metrics to 0-1 scale
        latency_score = max(0, 1 - (latency - 50) / 500)  # Good: 50ms, Bad: 550ms+
        jitter_score = max(0, 1 - jitter / 100)  # Good: 0ms, Bad: 100ms+
        loss_score = max(0, 1 - packet_loss / 0.05)  # Good: 0%, Bad: 5%+
        
        # Weighted average
        return (latency_score * 0.4 + jitter_score * 0.3 + loss_score * 0.3)
    
    async def _increase_buffer_size(self, factor: float = 1.2):
        """Increase buffer size by factor"""
        new_size = min(self.max_buffer_size, int(self.current_buffer_size * factor))
        self.current_buffer_size = new_size
        self.last_adaptation_time = time.time()
    
    async def _decrease_buffer_size(self, factor: float = 0.95):
        """Decrease buffer size by factor"""
        new_size = max(self.min_buffer_size, int(self.current_buffer_size * factor))
        self.current_buffer_size = new_size
        self.last_adaptation_time = time.time()
    
    async def _update_buffer_health(self):
        """Update buffer health based on current state"""
        buffer_level = len(self.chunks) / self.current_buffer_size
        
        await self.health_monitor.record_event({
            "type": "buffer_update",
            "timestamp": time.time(),
            "buffer_level": buffer_level,
            "buffer_size": self.current_buffer_size,
            "chunk_count": len(self.chunks)
        }) 