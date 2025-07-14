import asyncio
import json
import logging
import os
import random
import time
from typing import Dict, Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])

# Cache management
class CacheManager:
    """Simple cache manager for evaluation results"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
        
    def _generate_cache_key(self, text: str, target: str) -> str:
        """Generate cache key from text and target"""
        return f"{hash(text)}:{hash(target)}"
    
    def get(self, key: str) -> Dict[str, Any]:
        """Get item from cache if not expired"""
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl:
                return item["data"]
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict[str, Any]):
        """Set item in cache with timestamp"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def cleanup_expired(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item["timestamp"] >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "total": len(self.cache),
            "hits": getattr(self, "_hits", 0),
            "misses": getattr(self, "_misses", 0)
        }

# Telemetry logging
class TelemetryLogger:
    """Simple telemetry logger for request metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "personality_alignment_requests": 0,
            "lore_adherence_requests": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0
        }
    
    def log_request(self, request_type: str, latency: float, cache_hit: bool = False):
        """Log a request with its metrics"""
        self.metrics["total_requests"] += 1
        self.metrics[f"{request_type}_requests"] += 1
        self.metrics["total_latency_ms"] += latency * 1000
        
        if cache_hit:
            self.metrics["cache_hits"] += 1
        
        # Update average latency
        if self.metrics["total_requests"] > 0:
            self.metrics["avg_latency_ms"] = (
                self.metrics["total_latency_ms"] / self.metrics["total_requests"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

# Prompt formatting functions
def format_personality_prompt(text: str, target_personality: Dict[str, float]) -> str:
    """Format prompt for personality alignment evaluation"""
    return f"""
Evaluate how well the following text aligns with the target Big-Five personality traits:

Target Personality (0.0 = low, 1.0 = high):
- Openness: {target_personality.get('openness', 0.5):.2f}
- Conscientiousness: {target_personality.get('conscientiousness', 0.5):.2f}
- Extraversion: {target_personality.get('extraversion', 0.5):.2f}
- Agreeableness: {target_personality.get('agreeableness', 0.5):.2f}
- Neuroticism: {target_personality.get('neuroticism', 0.5):.2f}

Text to evaluate:
"{text}"

Rate the alignment between the text and target personality on a scale of 0.0 to 1.0.
Consider word choice, communication style, emotional expression, and behavioral indicators.

Respond with only a JSON object:
{{"alignment_score": 0.XX}}
"""

def format_lore_prompt(text: str, lore_fact: str) -> str:
    """Format prompt for lore adherence evaluation"""
    return f"""
Evaluate how well the following text adheres to the given lore fact:

Lore Fact:
"{lore_fact}"

Text to evaluate:
"{text}"

Rate the adherence on a scale of 0.0 to 1.0:
- 1.0: Perfect adherence, text fully aligns with lore
- 0.8-0.9: Good adherence with minor deviations
- 0.5-0.7: Moderate adherence, some inconsistencies
- 0.2-0.4: Poor adherence, significant contradictions
- 0.0: Complete contradiction of lore

Respond with only a JSON object:
{{"lore_score": 0.XX}}
"""

# Helper functions
def _get_api_key():
    """Get API key for LLM service"""
    return os.getenv("JUDGE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

def _is_dev_mode():
    """Check if running in dev mode (no API key)"""
    return _get_api_key() is None

# Initialize components
cache_manager = CacheManager()
telemetry_logger = TelemetryLogger()

# Log dev mode status at startup
if _is_dev_mode():
    logger.warning("Running in DEV MODE - no LLM API key found. Will return random scores.")

# Request/Response models
class PersonalityAlignmentRequest(BaseModel):
    text: str
    target: Dict[str, float]  # Big-Five scores

class LoreAdherenceRequest(BaseModel):
    text: str
    target: str  # Lore fact

class JudgeResponse(BaseModel):
    score: float
    dev_mode: bool = False
    cache_hit: bool = False

class HealthResponse(BaseModel):
    status: str
    dev_mode: bool = False
    cache_stats: Dict[str, int] = {}
    telemetry: Dict[str, Any] = {}

# LLM calling function
async def call_llm_judge(prompt: str, task_type: str) -> Dict[str, Any]:
    """
    Call the LLM for judge evaluation
    
    Args:
        prompt: Formatted prompt for the LLM
        task_type: Type of task (personality_alignment, lore_adherence)
    
    Returns:
        Dict containing the score
    """
    # Make actual API call
    async with httpx.AsyncClient() as client:
        api_key = _get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "seed": 42
        }
        
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        
        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=502, detail="LLM API error")
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        try:
            parsed = json.loads(content)
            if task_type == "personality_alignment":
                score = parsed.get("alignment_score", 0.5)
            else:  # lore_adherence
                score = parsed.get("lore_score", 0.5)
            
            return {"score": max(0.0, min(1.0, float(score)))}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {content}. Error: {e}")
            return {"score": 0.5}  # Default neutral score

# Router endpoints
@router.get("/evaluation/health", response_model=HealthResponse)
async def evaluation_health():
    """Get evaluation service health status"""
    # Clean up expired cache entries
    cache_manager.cleanup_expired()
    
    return HealthResponse(
        status="healthy",
        dev_mode=_is_dev_mode(),
        cache_stats=cache_manager.get_cache_stats(),
        telemetry=telemetry_logger.get_metrics()
    )

@router.post("/evaluation/personality_alignment", response_model=JudgeResponse)
async def evaluation_personality_alignment(request: PersonalityAlignmentRequest):
    """Evaluate how closely a text aligns with a target Big-Five personality using LLM-as-judge"""
    start_time = time.time()
    
    # Validate inputs
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Generate cache key
    cache_key = cache_manager._generate_cache_key(
        request.text, 
        json.dumps(request.target, sort_keys=True)
    )
    
    # Check cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        latency = time.time() - start_time
        telemetry_logger.log_request(
            request_type="personality_alignment",
            latency=latency,
            cache_hit=True
        )
        return JudgeResponse(
            score=cached_result["score"],
            dev_mode=_is_dev_mode(),
            cache_hit=True
        )
    
    # Format prompt and call LLM
    try:
        if _is_dev_mode():
            # Return random score in dev mode
            await asyncio.sleep(0.1)  # Simulate API latency
            result = {"score": random.uniform(0.0, 1.0)}
        else:
            prompt = format_personality_prompt(request.text, request.target)
            result = await call_llm_judge(prompt=prompt, task_type="personality_alignment")
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        latency = time.time() - start_time
        telemetry_logger.log_request(
            request_type="personality_alignment",
            latency=latency,
            cache_hit=False
        )
        
        return JudgeResponse(
            score=result["score"],
            dev_mode=_is_dev_mode(),
            cache_hit=False
        )
        
    except Exception as e:
        logger.error(f"Error in personality alignment evaluation: {e}")
        raise HTTPException(status_code=500, detail="Evaluation failed")

@router.post("/evaluation/lore_adherence", response_model=JudgeResponse)
async def evaluation_lore_adherence(request: LoreAdherenceRequest):
    """Judge the adherence of a text sample to a given lore fact using LLM-as-judge"""
    start_time = time.time()
    
    # Validate inputs
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not request.target or not request.target.strip():
        raise HTTPException(status_code=400, detail="Lore fact cannot be empty")
    
    # Generate cache key
    cache_key = cache_manager._generate_cache_key(request.text, request.target)
    
    # Check cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        latency = time.time() - start_time
        telemetry_logger.log_request(
            request_type="lore_adherence",
            latency=latency,
            cache_hit=True
        )
        return JudgeResponse(
            score=cached_result["score"],
            dev_mode=_is_dev_mode(),
            cache_hit=True
        )
    
    # Format prompt and call LLM
    try:
        if _is_dev_mode():
            # Return random score in dev mode
            await asyncio.sleep(0.1)  # Simulate API latency
            result = {"score": random.uniform(0.0, 1.0)}
        else:
            prompt = format_lore_prompt(request.text, request.target)
            result = await call_llm_judge(prompt=prompt, task_type="lore_adherence")
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        latency = time.time() - start_time
        telemetry_logger.log_request(
            request_type="lore_adherence",
            latency=latency,
            cache_hit=False
        )
        
        return JudgeResponse(
            score=result["score"],
            dev_mode=_is_dev_mode(),
            cache_hit=False
        )
        
    except Exception as e:
        logger.error(f"Error in lore adherence evaluation: {e}")
        raise HTTPException(status_code=500, detail="Evaluation failed") 