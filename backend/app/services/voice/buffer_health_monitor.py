"""
Buffer Health Monitor

Monitors buffer performance and provides health analytics for
smart audio buffering system.
"""

import time
import statistics
from typing import Dict, Any, List
from collections import deque


class BufferHealthMonitor:
    """Monitor and analyze buffer health for adaptive optimization"""
    
    def __init__(
        self,
        history_length: int = 100,
        health_check_interval: float = 1.0
    ):
        """
        Initialize buffer health monitor
        
        Args:
            history_length: Maximum number of events to track
            health_check_interval: Interval for health checks in seconds
        """
        self.history_length = history_length
        self.health_check_interval = health_check_interval
        
        # Event tracking
        self.event_history: deque = deque(maxlen=history_length)
        self.current_health_score = 1.0  # Start with perfect health
        
        # Performance metrics
        self.underrun_count = 0
        self.total_chunks_played = 0
        self.average_latency = 100.0
        self.buffer_level_history = deque(maxlen=50)
        
        # Trend analysis
        self.last_health_check = time.time()
        self.health_trend = "stable"
    
    async def record_event(self, event: Dict[str, Any]):
        """
        Record a buffer-related event for health analysis
        
        Args:
            event: Event data including type, timestamp, and metrics
        """
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        
        # Store event
        self.event_history.append(event)
        
        # Update metrics based on event type
        if event["type"] == "underrun":
            self.underrun_count += 1
        elif event["type"] == "chunk_played":
            self.total_chunks_played += 1
            if "latency_ms" in event:
                self._update_average_latency(event["latency_ms"])
        elif event["type"] == "buffer_update":
            if "buffer_level" in event:
                self.buffer_level_history.append(event["buffer_level"])
        
        # Recalculate health score
        self.current_health_score = self.calculate_health_score()
    
    def calculate_health_score(self) -> float:
        """
        Calculate overall buffer health score (0.0 to 1.0)
        
        Returns:
            Health score where 1.0 is perfect, 0.0 is critical
        """
        if not self.event_history:
            return 1.0
        
        # Analyze recent events (last 50)
        recent_events = list(self.event_history)[-50:]
        
        # Count different event types
        underruns = sum(1 for e in recent_events if e["type"] == "underrun")
        total_events = len(recent_events)
        
        # Base health from underrun rate
        underrun_rate = underruns / max(1, total_events)
        underrun_health = max(0.0, 1.0 - underrun_rate * 5)  # Penalize underruns heavily
        
        # Buffer level stability
        buffer_stability = self._calculate_buffer_stability()
        
        # Latency consistency
        latency_consistency = self._calculate_latency_consistency()
        
        # Weighted combination
        health_score = (
            underrun_health * 0.5 +
            buffer_stability * 0.3 +
            latency_consistency * 0.2
        )
        
        return max(0.0, min(1.0, health_score))
    
    def get_current_health(self) -> float:
        """Get current health score"""
        return self.current_health_score
    
    def analyze_performance_trend(self) -> Dict[str, Any]:
        """
        Analyze performance trend over time
        
        Returns:
            Trend analysis including direction and confidence
        """
        if len(self.event_history) < 10:
            return {"direction": "unknown", "confidence": 0.0}
        
        # Analyze health scores over time windows
        recent_events = list(self.event_history)
        window_size = min(10, len(recent_events) // 3)
        
        if window_size < 3:
            return {"direction": "stable", "confidence": 0.0}
        
        # Calculate health for different time windows
        early_events = recent_events[:window_size]
        middle_events = recent_events[window_size:window_size*2]
        late_events = recent_events[-window_size:]
        
        early_health = self._calculate_window_health(early_events)
        middle_health = self._calculate_window_health(middle_events)
        late_health = self._calculate_window_health(late_events)
        
        # Determine trend
        if late_health > middle_health > early_health:
            direction = "improving"
            confidence = min(1.0, (late_health - early_health) * 2)
        elif late_health < middle_health < early_health:
            direction = "degrading"
            confidence = min(1.0, (early_health - late_health) * 2)
        else:
            direction = "stable"
            confidence = max(0.0, 1.0 - abs(late_health - early_health) * 2)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "early_health": early_health,
            "middle_health": middle_health,
            "late_health": late_health
        }
    
    def get_adaptive_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get adaptive recommendations based on health analysis
        
        Returns:
            List of recommended actions with reasoning
        """
        recommendations = []
        
        # Check for network issues
        recent_events = list(self.event_history)[-20:]
        network_issues = [e for e in recent_events if e["type"] == "network_issue"]
        
        if network_issues:
            # Analyze network metrics
            avg_jitter = statistics.mean([e.get("jitter_ms", 0) for e in network_issues])
            avg_loss = statistics.mean([e.get("packet_loss", 0) for e in network_issues])
            
            if avg_jitter > 100 or avg_loss > 0.02:
                recommendations.append({
                    "action": "increase_buffer_size",
                    "reason": "network_instability",
                    "priority": "high",
                    "factor": 1.3
                })
        
        # Check underrun patterns
        underruns = [e for e in recent_events if e["type"] == "underrun"]
        if len(underruns) > 2:
            recommendations.append({
                "action": "increase_buffer_size",
                "reason": "frequent_underruns", 
                "priority": "high",
                "factor": 1.5
            })
        
        # Check buffer level trends
        if len(self.buffer_level_history) > 10:
            recent_levels = list(self.buffer_level_history)[-10:]
            avg_level = statistics.mean(recent_levels)
            
            if avg_level < 0.3:
                recommendations.append({
                    "action": "increase_buffer_size",
                    "reason": "low_buffer_levels",
                    "priority": "medium",
                    "factor": 1.2
                })
            elif avg_level > 0.9 and self.current_health_score > 0.9:
                recommendations.append({
                    "action": "decrease_buffer_size", 
                    "reason": "excess_buffering",
                    "priority": "low",
                    "factor": 0.9
                })
        
        return recommendations
    
    def _update_average_latency(self, latency: float):
        """Update running average latency"""
        # Simple exponential moving average
        alpha = 0.1
        self.average_latency = (alpha * latency + (1 - alpha) * self.average_latency)
    
    def _calculate_buffer_stability(self) -> float:
        """Calculate buffer level stability score"""
        if len(self.buffer_level_history) < 5:
            return 1.0
        
        # Calculate variance in buffer levels
        levels = list(self.buffer_level_history)
        try:
            variance = statistics.variance(levels)
            # Lower variance = higher stability
            stability = max(0.0, 1.0 - variance * 4)
            return stability
        except statistics.StatisticsError:
            return 1.0
    
    def _calculate_latency_consistency(self) -> float:
        """Calculate latency consistency score"""
        recent_events = list(self.event_history)[-20:]
        latencies = [e.get("latency_ms") for e in recent_events 
                    if e["type"] == "chunk_played" and "latency_ms" in e]
        
        if len(latencies) < 3:
            return 1.0
        
        try:
            variance = statistics.variance(latencies)
            # Lower variance = higher consistency
            consistency = max(0.0, 1.0 - variance / 10000)  # Scale appropriately
            return consistency
        except statistics.StatisticsError:
            return 1.0
    
    def _calculate_window_health(self, events: List[Dict[str, Any]]) -> float:
        """Calculate health score for a specific time window"""
        if not events:
            return 1.0
        
        # Calculate health based on various factors
        underruns = sum(1 for e in events if e["type"] == "underrun")
        chunk_events = [e for e in events if e["type"] == "chunk_played"]
        
        # Base health from underrun rate
        underrun_rate = underruns / max(1, len(events))
        underrun_health = max(0.0, 1.0 - underrun_rate * 3)
        
        # Factor in latency trends for chunk events
        if chunk_events:
            latencies = [e.get("latency_ms", 100) for e in chunk_events]
            avg_latency = sum(latencies) / len(latencies)
            # Normalize latency (100ms = good, 500ms+ = bad)
            latency_health = max(0.0, 1.0 - (avg_latency - 100) / 400)
        else:
            latency_health = 1.0
        
        # Factor in buffer levels
        buffer_events = [e for e in events if "buffer_level" in e]
        if buffer_events:
            buffer_levels = [e["buffer_level"] for e in buffer_events]
            avg_buffer_level = sum(buffer_levels) / len(buffer_levels)
            buffer_health = avg_buffer_level  # 0.0 to 1.0
        else:
            buffer_health = 1.0
        
        # Weighted combination
        return (underrun_health * 0.5 + latency_health * 0.3 + buffer_health * 0.2) 