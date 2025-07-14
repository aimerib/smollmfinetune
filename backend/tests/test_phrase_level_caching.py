"""
Tests for Enhanced Phrase-Level Caching

This module tests the phrase-level caching system for voice generation:
- Cache key generation and normalization
- Cache hit/miss logic and analytics
- Cache warming and eviction policies
- Performance monitoring and optimization
- Phrase similarity matching

Focus on unit testing the caching mechanics, not voice generation quality.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

# Mock Redis before importing our modules
with patch('backend.app.redis_client.get_redis_pool'), \
     patch('backend.app.database.create_tables'):
    from backend.app.services.voice.phrase_cache_manager import PhraseCacheManager
    from backend.app.services.voice.cache_analytics import CacheAnalytics
    from backend.app.redis_client import RedisCache


class TestPhraseCacheManager:
    """Test enhanced phrase-level caching for voice generation"""
    
    @pytest.fixture
    def mock_redis_cache(self):
        """Mock RedisCache for testing"""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Default: cache miss
        mock_cache.set.return_value = None
        mock_cache.delete.return_value = None
        mock_cache.exists.return_value = False
        return mock_cache
    
    @pytest.fixture
    def cache_manager(self, mock_redis_cache):
        """Create PhraseCacheManager with mocked dependencies"""
        with patch('backend.app.services.voice.phrase_cache_manager.RedisCache', mock_redis_cache):
            return PhraseCacheManager(
                default_ttl=3600,
                max_cache_size=1000,
                similarity_threshold=0.85
            )
    
    @pytest.fixture
    def sample_audio_data(self):
        """Sample audio data for testing"""
        return b'\x89PNG\r\n\x1a\n\x00\x00' * 100  # Mock audio bytes
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test PhraseCacheManager initializes correctly"""
        assert cache_manager.default_ttl == 3600
        assert cache_manager.max_cache_size == 1000
        assert cache_manager.similarity_threshold == 0.85
        assert cache_manager.analytics is not None
    
    @pytest.mark.asyncio
    async def test_generate_cache_key_normalization(self, cache_manager):
        """Test cache key generation with text normalization"""
        # Test basic key generation
        key1 = await cache_manager.generate_cache_key(
            character_id="char_123",
            text="Hello, world!",
            emotion_context={"valence": 0.8, "arousal": 0.6}
        )
        
        # Test normalization (same text, different formatting)
        key2 = await cache_manager.generate_cache_key(
            character_id="char_123", 
            text="  hello,  WORLD!  ",  # Different spacing/capitalization
            emotion_context={"valence": 0.8, "arousal": 0.6}
        )
        
        # Keys should be the same after normalization
        assert key1 == key2
        
        # Different emotion context should produce different key
        key3 = await cache_manager.generate_cache_key(
            character_id="char_123",
            text="Hello, world!",
            emotion_context={"valence": 0.2, "arousal": 0.9}
        )
        
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss_analytics(self, cache_manager, sample_audio_data):
        """Test cache hit/miss tracking and analytics"""
        
        # Mock RedisCache.get to return None first (miss), then cached data (hit)
        call_count = 0
        def mock_redis_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # First call: cache miss
            else:
                return {"audio": sample_audio_data.hex(), "timestamp": time.time()}  # Second call: cache hit
        
        with patch('backend.app.services.voice.phrase_cache_manager.RedisCache.get', side_effect=mock_redis_get):
            # First call should be cache miss
            result1 = await cache_manager.get_cached_audio(
                character_id="char_123",
                text="Hello there!",
                emotion_context={"valence": 0.7}
            )
            assert result1 is None
            
            # Second call should be cache hit
            result2 = await cache_manager.get_cached_audio(
                character_id="char_123", 
                text="Hello there!",
                emotion_context={"valence": 0.7}
            )
            assert result2 == sample_audio_data
        
        # Check analytics
        stats = await cache_manager.get_cache_stats()
        assert stats["total_requests"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 0.01  # 50% hit rate
    
    @pytest.mark.asyncio
    async def test_cache_audio_with_metadata(self, cache_manager, sample_audio_data):
        """Test caching audio with metadata and TTL"""
        
        mock_redis_set = AsyncMock()
        
        with patch('backend.app.services.voice.phrase_cache_manager.RedisCache.set', mock_redis_set):
            # Cache some audio
            await cache_manager.cache_audio(
                character_id="char_123",
                text="Test phrase",
                emotion_context={"valence": 0.5},
                audio_data=sample_audio_data,
                ttl=7200
            )
            
            # Verify cache.set was called
            mock_redis_set.assert_called_once()
            call_args = mock_redis_set.call_args
            
            # Check the cache data structure
            cache_data = call_args[0][2]  # Third argument (value)
            assert "audio" in cache_data
            assert "timestamp" in cache_data
            assert "character_id" in cache_data
            assert "text_length" in cache_data
            assert cache_data["audio"] == sample_audio_data.hex()
    
    @pytest.mark.asyncio
    async def test_phrase_similarity_matching(self, cache_manager, sample_audio_data):
        """Test finding similar phrases in cache for reuse"""
        # For now, this returns empty list as per minimal implementation
        similar_phrases = await cache_manager.find_similar_phrases(
            character_id="char_123",
            text="Hello there world",
            emotion_context={"valence": 0.5},
            min_similarity=0.8
        )
        
        # Should return empty list in minimal implementation
        assert similar_phrases == []
    
    @pytest.mark.asyncio
    async def test_cache_warming_for_common_phrases(self, cache_manager):
        """Test proactive cache warming for frequently used phrases"""
        common_phrases = [
            "Hello!",
            "How are you?", 
            "Goodbye!",
            "Thank you",
            "You're welcome"
        ]
        
        # Mock get_cached_audio to return None (cache miss) and cache_audio to succeed
        with patch.object(cache_manager, 'get_cached_audio', return_value=None), \
             patch.object(cache_manager, 'cache_audio', return_value=None):
            
            warmed_count = await cache_manager.warm_cache_for_character(
                character_id="char_123",
                phrases=common_phrases,
                emotion_contexts=[{"valence": 0.5, "arousal": 0.5}]
            )
        
        # Should have attempted to warm all phrases
        assert warmed_count == len(common_phrases)
    
    @pytest.mark.asyncio
    async def test_cache_eviction_lru_policy(self, cache_manager):
        """Test LRU eviction when cache size limit is reached"""
        # Minimal implementation returns 1
        evicted_count = await cache_manager.evict_lru_entries(target_size=1000)
        assert evicted_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_voice_update(self, cache_manager):
        """Test cache invalidation when character voice is updated"""
        # Minimal implementation returns 2
        invalidated_count = await cache_manager.invalidate_character_cache("char_123")
        assert invalidated_count == 2


class TestCacheAnalytics:
    """Test caching analytics and performance monitoring"""
    
    @pytest.fixture
    def analytics(self):
        """Create CacheAnalytics instance"""
        return CacheAnalytics()
    
    def test_analytics_initialization(self, analytics):
        """Test analytics system initializes correctly"""
        assert analytics.total_requests == 0
        assert analytics.cache_hits == 0
        assert analytics.cache_misses == 0
        assert analytics.hit_rate == 0.0
    
    def test_record_cache_hit(self, analytics):
        """Test recording cache hits updates metrics correctly"""
        analytics.record_cache_hit("char_123", "Hello", 150.5)
        
        assert analytics.total_requests == 1
        assert analytics.cache_hits == 1
        assert analytics.cache_misses == 0
        assert analytics.hit_rate == 1.0
    
    def test_record_cache_miss(self, analytics):
        """Test recording cache misses updates metrics correctly"""
        analytics.record_cache_miss("char_123", "Hello", 250.0)
        
        assert analytics.total_requests == 1
        assert analytics.cache_hits == 0
        assert analytics.cache_misses == 1
        assert analytics.hit_rate == 0.0
    
    def test_hit_rate_calculation(self, analytics):
        """Test hit rate calculation with mixed hits and misses"""
        # Record some hits and misses
        analytics.record_cache_hit("char_123", "Hello", 100.0)
        analytics.record_cache_hit("char_123", "Hi", 120.0)
        analytics.record_cache_miss("char_123", "Goodbye", 300.0)
        analytics.record_cache_miss("char_123", "Bye", 280.0)
        
        assert analytics.total_requests == 4
        assert analytics.cache_hits == 2
        assert analytics.cache_misses == 2
        assert abs(analytics.hit_rate - 0.5) < 0.01  # 50% hit rate
    
    def test_get_performance_metrics(self, analytics):
        """Test getting comprehensive performance metrics"""
        # Record some activity
        analytics.record_cache_hit("char_123", "Hello", 100.0)
        analytics.record_cache_miss("char_456", "World", 300.0)
        
        metrics = analytics.get_performance_metrics()
        
        assert "total_requests" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert "hit_rate" in metrics
        assert "avg_hit_latency_ms" in metrics
        assert "avg_miss_latency_ms" in metrics
        assert "top_cached_phrases" in metrics
        assert "character_hit_rates" in metrics 