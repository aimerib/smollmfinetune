"""
Telemetry logging for Judge Service

Simple telemetry system to track request counts, latency, and cache hit ratios.
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """Simple telemetry logging for judge service metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency = 0.0
    
    def log_request(self, request_type: str, latency: float, cache_hit: bool, **kwargs):
        """
        Log a request with telemetry data
        
        Args:
            request_type: Type of request (personality_alignment, lore_adherence)
            latency: Request latency in seconds
            cache_hit: Whether this was a cache hit
            **kwargs: Additional telemetry data
        """
        self.request_count += 1
        self.total_latency += latency
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Log the request details
        logger.info(f"Judge request: {request_type}, latency={latency:.3f}s, cache_hit={cache_hit}, {kwargs}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current telemetry metrics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_ratio = self.cache_hits / total_requests if total_requests > 0 else 0.0
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0.0
        
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": cache_hit_ratio,
            "average_latency": avg_latency,
            "total_latency": self.total_latency
        }
    
    def reset_metrics(self):
        """Reset all metrics to zero"""
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency = 0.0 