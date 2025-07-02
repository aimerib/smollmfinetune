import redis.asyncio as redis
import json
from typing import Optional, Any, Dict
from app.config import settings

# Redis connection pool
redis_pool = None

async def get_redis_pool():
    """Get or create Redis connection pool."""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
    return redis_pool

async def get_redis():
    """Get Redis client."""
    pool = await get_redis_pool()
    return redis.Redis(connection_pool=pool)

class RedisCache:
    """Redis cache utilities."""
    
    @staticmethod
    def _key(prefix: str, identifier: str) -> str:
        """Generate cache key."""
        return f"devkit:{prefix}:{identifier}"
    
    @staticmethod
    async def get(prefix: str, identifier: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        r = await get_redis()
        key = RedisCache._key(prefix, identifier)
        value = await r.get(key)
        if value:
            return json.loads(value)
        return None
    
    @staticmethod
    async def set(
        prefix: str, 
        identifier: str, 
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        r = await get_redis()
        key = RedisCache._key(prefix, identifier)
        serialized = json.dumps(value)
        
        if ttl is None:
            ttl = settings.REDIS_CACHE_TTL
            
        await r.setex(key, ttl, serialized)
    
    @staticmethod
    async def delete(prefix: str, identifier: str):
        """Delete value from cache."""
        r = await get_redis()
        key = RedisCache._key(prefix, identifier)
        await r.delete(key)
    
    @staticmethod
    async def exists(prefix: str, identifier: str) -> bool:
        """Check if key exists in cache."""
        r = await get_redis()
        key = RedisCache._key(prefix, identifier)
        return await r.exists(key) > 0

class RedisPubSub:
    """Redis pub/sub utilities for real-time updates."""
    
    @staticmethod
    async def publish(channel: str, message: Dict[str, Any]):
        """Publish message to channel."""
        r = await get_redis()
        serialized = json.dumps(message)
        await r.publish(channel, serialized)
    
    @staticmethod
    async def subscribe(channel: str):
        """Subscribe to channel."""
        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        return pubsub
    
    @staticmethod
    async def listen(pubsub):
        """Listen for messages."""
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                yield data

# Training status tracking
class TrainingStatusTracker:
    """Track training job status in Redis."""
    
    @staticmethod
    async def update_status(
        job_id: str, 
        status: str, 
        progress: float,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Update training job status."""
        update = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "metrics": metrics
        }
        
        # Store in cache
        await RedisCache.set(f"training_status", job_id, update, ttl=86400)  # 24 hours
        
        # Publish update
        await RedisPubSub.publish(f"training:{job_id}", update)
    
    @staticmethod
    async def get_status(job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status."""
        return await RedisCache.get("training_status", job_id)

# Session management
class SessionManager:
    """Manage user sessions in Redis."""
    
    @staticmethod
    async def create_session(user_id: str, session_data: Dict[str, Any]) -> str:
        """Create a new session."""
        import uuid
        session_id = str(uuid.uuid4())
        
        await RedisCache.set(
            "session",
            session_id,
            {"user_id": user_id, **session_data},
            ttl=86400  # 24 hours
        )
        
        return session_id
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return await RedisCache.get("session", session_id)
    
    @staticmethod
    async def delete_session(session_id: str):
        """Delete session."""
        await RedisCache.delete("session", session_id)

# Rate limiting
class RateLimiter:
    """Simple rate limiter using Redis."""
    
    @staticmethod
    async def check_rate_limit(
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60
    ) -> bool:
        """Check if rate limit is exceeded."""
        r = await get_redis()
        key = f"rate_limit:{identifier}"
        
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        results = await pipe.execute()
        
        current_requests = results[0]
        return current_requests <= max_requests 