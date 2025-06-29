"""
Cache management for Judge Service

SQLite-based caching system with SHA256 keys and TTL expiry.
"""

import hashlib
import json
import sqlite3
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """SQLite-based cache with SHA256 keys and TTL expiry"""
    
    def __init__(self, db_path: str = "judge_cache.db", ttl_seconds: int = 3600):
        """
        Initialize cache manager
        
        Args:
            db_path: Path to SQLite database file
            ttl_seconds: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database and create tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    result TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            # Create index for expiry cleanup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON judge_cache(expires_at)
            """)
            conn.commit()
    
    def _generate_cache_key(self, text: str, target: str) -> str:
        """
        Generate SHA256 cache key from text and target
        
        Args:
            text: The text to evaluate
            target: The target (Big-Five scores or lore fact)
        
        Returns:
            64-character SHA256 hash
        """
        # Combine text and target into a single string for hashing
        combined = f"{text}||{target}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result by key
        
        Args:
            cache_key: SHA256 cache key
        
        Returns:
            Cached result dict or None if not found/expired
        """
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT result, expires_at FROM judge_cache 
                WHERE cache_key = ? AND expires_at > ?
            """, (cache_key, current_time))
            
            row = cursor.fetchone()
            if row:
                result_json, expires_at = row
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                return json.loads(result_json)
            else:
                logger.debug(f"Cache miss for key {cache_key[:8]}...")
                return None
    
    def set(self, cache_key: str, result: Dict[str, Any]):
        """
        Store result in cache
        
        Args:
            cache_key: SHA256 cache key
            result: Result dictionary to cache
        """
        current_time = time.time()
        expires_at = current_time + self.ttl_seconds
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO judge_cache 
                (cache_key, result, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (cache_key, json.dumps(result), current_time, expires_at))
            conn.commit()
        
        logger.debug(f"Cached result for key {cache_key[:8]}...")
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM judge_cache WHERE expires_at <= ?
            """, (current_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM judge_cache")
            total_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT COUNT(*) FROM judge_cache WHERE expires_at > ?
            """, (current_time,))
            active_entries = cursor.fetchone()[0]
            
            expired_entries = total_entries - active_entries
        
        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries
        } 