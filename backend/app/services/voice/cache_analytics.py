"""
Cache Analytics and Performance Monitoring

Tracks cache performance metrics including:
- Hit/miss ratios and latency
- Character-specific performance
- Top cached phrases and trends
- Performance optimization insights
"""

import time
from typing import Dict, Any, List
from collections import defaultdict, deque


class CacheAnalytics:
    """Analytics system for phrase-level caching performance"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize analytics system
        
        Args:
            max_history: Maximum number of requests to track in history
        """
        self.max_history = max_history
        
        # Core metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Latency tracking
        self.hit_latencies: deque = deque(maxlen=max_history)
        self.miss_latencies: deque = deque(maxlen=max_history)
        
        # Character-specific tracking
        self.character_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "requests": 0}
        )
        
        # Phrase popularity tracking
        self.phrase_frequency: Dict[str, int] = defaultdict(int)
        self.phrase_hit_rate: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "total": 0}
        )
        
        # Time series data
        self.request_history: deque = deque(maxlen=max_history)
    
    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def record_cache_hit(self, character_id: str, phrase: str, latency_ms: float):
        """
        Record a cache hit
        
        Args:
            character_id: Character identifier
            phrase: The phrase that was cached
            latency_ms: Latency in milliseconds
        """
        # Update core metrics
        self.total_requests += 1
        self.cache_hits += 1
        
        # Track latency
        self.hit_latencies.append(latency_ms)
        
        # Update character stats
        self.character_stats[character_id]["hits"] += 1
        self.character_stats[character_id]["requests"] += 1
        
        # Update phrase stats
        self.phrase_frequency[phrase] += 1
        self.phrase_hit_rate[phrase]["hits"] += 1
        self.phrase_hit_rate[phrase]["total"] += 1
        
        # Track in history
        self.request_history.append({
            "timestamp": time.time(),
            "type": "hit",
            "character_id": character_id,
            "phrase": phrase,
            "latency_ms": latency_ms
        })
    
    def record_cache_miss(self, character_id: str, phrase: str, latency_ms: float):
        """
        Record a cache miss
        
        Args:
            character_id: Character identifier
            phrase: The phrase that was missed
            latency_ms: Latency in milliseconds
        """
        # Update core metrics
        self.total_requests += 1
        self.cache_misses += 1
        
        # Track latency
        self.miss_latencies.append(latency_ms)
        
        # Update character stats
        self.character_stats[character_id]["misses"] += 1
        self.character_stats[character_id]["requests"] += 1
        
        # Update phrase stats
        self.phrase_frequency[phrase] += 1
        self.phrase_hit_rate[phrase]["total"] += 1
        
        # Track in history
        self.request_history.append({
            "timestamp": time.time(),
            "type": "miss",
            "character_id": character_id,
            "phrase": phrase,
            "latency_ms": latency_ms
        })
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        
        Returns:
            Dictionary of performance analytics
        """
        # Calculate average latencies
        avg_hit_latency = (
            sum(self.hit_latencies) / len(self.hit_latencies)
            if self.hit_latencies else 0.0
        )
        
        avg_miss_latency = (
            sum(self.miss_latencies) / len(self.miss_latencies)
            if self.miss_latencies else 0.0
        )
        
        # Get top cached phrases
        top_phrases = sorted(
            self.phrase_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate character hit rates
        character_hit_rates = {}
        for char_id, stats in self.character_stats.items():
            if stats["requests"] > 0:
                hit_rate = stats["hits"] / stats["requests"]
                character_hit_rates[char_id] = {
                    "hit_rate": hit_rate,
                    "total_requests": stats["requests"]
                }
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "avg_hit_latency_ms": avg_hit_latency,
            "avg_miss_latency_ms": avg_miss_latency,
            "top_cached_phrases": top_phrases,
            "character_hit_rates": character_hit_rates,
            "recent_activity": list(self.request_history)[-10:]  # Last 10 requests
        }
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Get insights for cache optimization
        
        Returns:
            Dictionary of optimization recommendations
        """
        insights = {
            "recommendations": [],
            "performance_issues": [],
            "optimization_opportunities": []
        }
        
        # Check hit rate
        if self.hit_rate < 0.3:
            insights["performance_issues"].append(
                "Low cache hit rate - consider warming more common phrases"
            )
        
        # Check for frequently missed phrases
        frequent_misses = []
        for phrase, stats in self.phrase_hit_rate.items():
            if stats["total"] >= 5 and stats["hits"] == 0:
                frequent_misses.append(phrase)
        
        if frequent_misses:
            insights["optimization_opportunities"].append({
                "type": "cache_warming",
                "phrases": frequent_misses[:5],
                "reason": "Frequently requested but never cached"
            })
        
        # Check character performance
        low_performing_characters = []
        for char_id, stats in self.character_stats.items():
            if stats["requests"] >= 10:
                hit_rate = stats["hits"] / stats["requests"]
                if hit_rate < 0.2:
                    low_performing_characters.append(char_id)
        
        if low_performing_characters:
            insights["optimization_opportunities"].append({
                "type": "character_optimization",
                "characters": low_performing_characters,
                "reason": "Low hit rate for these characters"
            })
        
        return insights 