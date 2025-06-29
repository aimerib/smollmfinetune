"""
Judge Model Micro-Service (R4-1.2)

FastAPI service for centralized LLM-as-judge evaluation of personality alignment and lore adherence.
Provides caching, telemetry, and dev mode fallback.
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Dict, Any, Union

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .cache import CacheManager
from .prompts import format_personality_prompt, format_lore_prompt
from .telemetry import TelemetryLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Judge Model Service",
    description="Centralized LLM-as-judge service for personality alignment and lore adherence evaluation",
    version="1.0.0"
)

# Initialize components
cache_manager = CacheManager()
telemetry_logger = TelemetryLogger()

# Check for API key
def _get_api_key():
    return os.getenv("JUDGE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

def _is_dev_mode():
    return _get_api_key() is None

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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with service status"""
    # Clean up expired cache entries
    cache_manager.cleanup_expired()
    
    return HealthResponse(
        status="healthy",
        dev_mode=_is_dev_mode(),
        cache_stats=cache_manager.get_cache_stats(),
        telemetry=telemetry_logger.get_metrics()
    )


@app.post("/personality_alignment", response_model=JudgeResponse)
async def evaluate_personality_alignment(request: PersonalityAlignmentRequest):
    """Evaluate personality alignment using LLM-as-judge"""
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


@app.post("/lore_adherence", response_model=JudgeResponse)
async def evaluate_lore_adherence(request: LoreAdherenceRequest):
    """Evaluate lore adherence using LLM-as-judge"""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 