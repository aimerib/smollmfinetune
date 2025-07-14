"""
Core Production Inference Engine with FastAPI integration.

Provides high-performance async inference with triple-head outputs,
GPU optimization, and request queueing.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from transformers import AutoTokenizer
from peft import PeftModel

# Import existing infrastructure
from ..utils.inference import InferenceManager
from ..utils.vllm_optimized_client import VLLMOptimizedClient, BatchConfig

# Try to import NarrativeLLM for triple-head support
try:
    from ..narrative_engine.model import NarrativeLLM, create_narrative_model
    from ..narrative_engine.config import NarrativeLLMConfig
    NARRATIVE_ENGINE_AVAILABLE = True
except ImportError:
    NARRATIVE_ENGINE_AVAILABLE = False
    logging.warning("NarrativeLLM not available - triple-head features disabled")

logger = logging.getLogger(__name__)


# Data Models
@dataclass
class ControlToken:
    """Control token with probability and metadata"""
    token: str
    probability: float
    token_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryMetadata:
    """Metadata for memory vectors"""
    importance: float = 0.5
    emotional_valence: float = 0.0
    recency: float = 1.0
    coherence: float = 0.5
    access_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class TripleHeadOutput:
    """Output from triple-head model"""
    generation_text: str
    control_tokens: List[Dict[str, Any]]
    memory_vector: List[float]
    memory_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "generation_text": self.generation_text,
            "control_tokens": self.control_tokens,
            "memory_vector": self.memory_vector,
            "memory_metadata": self.memory_metadata
        }


class InferenceRequest(BaseModel):
    """Request model for inference API"""
    session_id: str
    character_id: str
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    use_cache: bool = True
    priority: int = 0
    context_window: Optional[List[Dict[str, str]]] = None
    forced_control_tokens: Optional[List[str]] = None
    memory_context_ids: Optional[List[str]] = None


class InferenceResponse(BaseModel):
    """Response model for inference API"""
    session_id: str
    character_id: str
    generation_text: str
    control_tokens: List[Dict[str, Any]]
    memory_vector: List[float]
    memory_metadata: Dict[str, Any]
    inference_time_ms: float
    tokens_generated: int
    cache_hit: bool = False
    
    @classmethod
    def from_triple_head(cls, request: InferenceRequest, output: TripleHeadOutput, 
                         inference_time_ms: float, cache_hit: bool = False) -> 'InferenceResponse':
        """Create response from triple-head output"""
        return cls(
            session_id=request.session_id,
            character_id=request.character_id,
            generation_text=output.generation_text,
            control_tokens=output.control_tokens,
            memory_vector=output.memory_vector,
            memory_metadata=output.memory_metadata,
            inference_time_ms=inference_time_ms,
            tokens_generated=len(output.generation_text.split()),  # Approximate
            cache_hit=cache_hit
        )


class ProductionInferenceEngine:
    """
    High-performance inference engine optimized for production.
    
    Features:
    - Async request processing with FastAPI
    - GPU memory optimization with attention caching
    - Request queueing and prioritization
    - Health monitoring and automatic recovery
    - Triple-head model support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the inference engine"""
        self.config = config or self._default_config()
        
        # Core components
        self.app = FastAPI(title="Character Inference Engine", version="1.0.0")
        self.inference_manager = InferenceManager()
        
        # Model cache
        self.loaded_models: Dict[str, Any] = {}
        self.model_cache_size = self.config.get("model_cache_size", 3)
        
        # Request queue
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.processing_semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 10))
        
        # Attention cache for optimization
        self.attention_cache: Dict[str, Any] = {}
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        
        # Health monitoring
        self.health_status = {
            "status": "initializing",
            "gpu_utilization": 0.0,
            "memory_usage_gb": 0.0,
            "active_requests": 0,
            "queued_requests": 0,
            "last_check": datetime.now()
        }
        
        # Performance tracking
        self.request_metrics: List[Dict[str, Any]] = []
        self.max_metrics_history = 1000
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self.processor_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        logger.info("Production Inference Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "max_concurrent": 10,
            "model_cache_size": 3,
            "cache_ttl": 300,
            "gpu_memory_threshold": 0.85,
            "health_check_interval": 30,
            "enable_vllm": True,
            "enable_triple_head": NARRATIVE_ENGINE_AVAILABLE,
            "memory_vector_dim": 768,
            "control_token_vocab_size": 1000
        }
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/generate", response_model=InferenceResponse)
        async def generate(request: InferenceRequest):
            """Generate response with triple-head model"""
            return await self.generate(request)
        
        @self.app.get("/health")
        async def health():
            """Get health status"""
            return self.health_check()
        
        @self.app.get("/metrics")
        async def metrics():
            """Get performance metrics"""
            return self.get_metrics()
        
        @self.app.get("/queue/status")
        async def queue_status():
            """Get queue status"""
            return self.get_queue_status()
        
        @self.app.post("/cache/clear")
        async def clear_cache():
            """Clear attention cache"""
            self.clear_attention_cache()
            return {"status": "cache_cleared"}
    
    async def initialize(self):
        """Initialize the engine and start background tasks"""
        try:
            # Start request processor
            self.processor_task = asyncio.create_task(self._request_processor())
            
            # Start health monitor
            self.health_monitor_task = asyncio.create_task(self._health_monitor())
            
            # Update status
            self.health_status["status"] = "healthy"
            self.health_status["last_check"] = datetime.now()
            
            logger.info("Inference engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            self.health_status["status"] = "error"
            self.health_status["error"] = str(e)
            raise
    
    async def shutdown(self):
        """Shutdown the engine and cleanup resources"""
        logger.info("Shutting down inference engine...")
        
        # Cancel background tasks
        if self.processor_task:
            self.processor_task.cancel()
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        
        # Clear caches
        self.loaded_models.clear()
        self.attention_cache.clear()
        
        # Update status
        self.health_status["status"] = "shutdown"
        
        logger.info("Inference engine shutdown complete")
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response with triple-head model"""
        start_time = time.time()
        
        try:
            # Add to queue with priority
            queue_item = {
                "request": request,
                "future": asyncio.Future(),
                "priority": request.priority,
                "timestamp": time.time()
            }
            
            await self.request_queue.put(queue_item)
            
            # Wait for result
            result = await queue_item["future"]
            
            # Track metrics
            inference_time_ms = (time.time() - start_time) * 1000
            self._track_request_metrics(request, inference_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def generate_with_metrics(self, request: InferenceRequest) -> Dict[str, Any]:
        """Generate response with detailed metrics"""
        response = await self.generate(request)
        
        return {
            "response": response,
            "metrics": {
                "inference_time_ms": response.inference_time_ms,
                "tokens_generated": response.tokens_generated,
                "cache_hit": response.cache_hit,
                "queue_depth": self.request_queue.qsize(),
                "gpu_utilization": self.health_status["gpu_utilization"]
            }
        }
    
    async def _request_processor(self):
        """Background task to process requests from queue"""
        logger.info("Request processor started")
        
        while True:
            try:
                # Get request from queue
                queue_item = await self.request_queue.get()
                
                # Process with semaphore for concurrency control
                async with self.processing_semaphore:
                    try:
                        result = await self._process_request(queue_item["request"])
                        queue_item["future"].set_result(result)
                    except Exception as e:
                        queue_item["future"].set_exception(e)
                
            except asyncio.CancelledError:
                logger.info("Request processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in request processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request"""
        start_time = time.time()
        cache_hit = False
        
        try:
            # Check attention cache
            cache_key = self._get_cache_key(request)
            cached_attention = None
            
            if request.use_cache and cache_key in self.attention_cache:
                cached_item = self.attention_cache[cache_key]
                if time.time() - cached_item["timestamp"] < self.cache_ttl:
                    cached_attention = cached_item["attention"]
                    cache_hit = True
            
            # Load model for character
            model = await self._load_model(request.character_id)
            
            # Generate with triple-head model
            if self.config["enable_triple_head"] and hasattr(model, 'generate_triple_head'):
                output = await self._generate_triple_head(
                    model, request, cached_attention
                )
            else:
                # Fallback to standard generation
                output = await self._generate_standard(
                    model, request, cached_attention
                )
            
            # Update attention cache
            if request.use_cache and not cache_hit:
                # TripleHeadOutput doesn't have attention_weights, skip caching for now
                self.attention_cache[cache_key] = {
                    "attention": None,  # Could be added to TripleHeadOutput if needed
                    "timestamp": time.time()
                }
            
            # Create response
            inference_time_ms = (time.time() - start_time) * 1000
            return InferenceResponse.from_triple_head(
                request, output, inference_time_ms, cache_hit
            )
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            raise
    
    async def _load_model(self, character_id: str) -> Any:
        """Load model for character with caching"""
        if character_id in self.loaded_models:
            return self.loaded_models[character_id]
        
        # Check cache size
        if len(self.loaded_models) >= self.model_cache_size:
            # Evict least recently used
            oldest_id = min(self.loaded_models.keys(), 
                          key=lambda k: self.loaded_models[k].get("last_used", 0))
            del self.loaded_models[oldest_id]
        
        # Load model
        logger.info(f"Loading model for character: {character_id}")
        
        if self.config["enable_triple_head"] and NARRATIVE_ENGINE_AVAILABLE:
            # Load NarrativeLLM
            config = NarrativeLLMConfig(
                base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                enable_control_head=True,
                enable_memory_head=True,
                memory_vector_dim=self.config["memory_vector_dim"]
            )
            model = create_narrative_model(config)
            
            # Load character adapter if exists
            adapter_path = f"training_output/adapters/{character_id}"
            if Path(adapter_path).exists():
                model.load_adapter(adapter_path)
        else:
            # Use standard inference manager
            model = self.inference_manager
        
        self.loaded_models[character_id] = {
            "model": model,
            "last_used": time.time()
        }
        
        return model
    
    async def _generate_triple_head(self, model: Any, request: InferenceRequest, 
                                   cached_attention: Optional[Any]) -> TripleHeadOutput:
        """Generate with triple-head model"""
        # Prepare inputs
        inputs = {
            "prompt": request.prompt,
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty
        }
        
        if cached_attention is not None:
            inputs["past_key_values"] = cached_attention
        
        if request.forced_control_tokens:
            inputs["forced_control_tokens"] = request.forced_control_tokens
        
        # Generate
        result = await asyncio.to_thread(
            model.generate_with_control,
            **inputs
        )
        
        # Extract outputs
        return TripleHeadOutput(
            generation_text=result["generated_text"],
            control_tokens=result.get("control_tokens", []),
            memory_vector=result.get("memory_vector", [0.0] * self.config["memory_vector_dim"]),
            memory_metadata=result.get("memory_metadata", {})
        )
    
    async def _generate_standard(self, model: Any, request: InferenceRequest,
                               cached_attention: Optional[Any]) -> TripleHeadOutput:
        """Generate with standard model (fallback)"""
        # Use inference manager (synchronous, so wrap in thread)
        response = await asyncio.to_thread(
            self.inference_manager.generate_response,
            model_path=f"character-{request.character_id}",
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        
        # Create fake triple-head output
        return TripleHeadOutput(
            generation_text=response,
            control_tokens=[],
            memory_vector=[0.0] * self.config["memory_vector_dim"],
            memory_metadata={}
        )
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        # Include session, character, and recent context
        context_str = ""
        if request.context_window:
            context_str = str(request.context_window[-3:])  # Last 3 messages
        
        return f"{request.session_id}:{request.character_id}:{hash(context_str)}"
    
    def _track_request_metrics(self, request: InferenceRequest, inference_time_ms: float):
        """Track request metrics for monitoring"""
        metric = {
            "timestamp": time.time(),
            "session_id": request.session_id,
            "character_id": request.character_id,
            "inference_time_ms": inference_time_ms,
            "queue_depth": self.request_queue.qsize(),
            "priority": request.priority
        }
        
        self.request_metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.request_metrics) > self.max_metrics_history:
            self.request_metrics = self.request_metrics[-self.max_metrics_history:]
    
    async def _health_monitor(self):
        """Monitor system health and trigger recovery if needed"""
        logger.info("Health monitor started")
        
        while True:
            try:
                await asyncio.sleep(self.config["health_check_interval"])
                
                # Update health metrics
                gpu_util = self._get_gpu_utilization()
                memory_usage = self._get_memory_usage()
                
                self.health_status.update({
                    "gpu_utilization": gpu_util,
                    "memory_usage_gb": memory_usage,
                    "active_requests": self.processing_semaphore._value,
                    "queued_requests": self.request_queue.qsize(),
                    "last_check": datetime.now()
                })
                
                # Check if recovery needed
                if gpu_util > self.config["gpu_memory_threshold"]:
                    logger.warning(f"High GPU utilization: {gpu_util}")
                    await self.check_and_recover()
                
            except asyncio.CancelledError:
                logger.info("Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            elif torch.backends.mps.is_available():
                # MPS doesn't provide memory stats, return reasonable default
                return 0.5
            return 0.0
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            elif torch.backends.mps.is_available():
                # MPS doesn't provide memory stats, return estimate
                return 4.0  # Assume 4GB usage
            return 0.0
        except:
            return 0.0
    
    async def check_and_recover(self) -> bool:
        """Check system health and trigger recovery if needed"""
        recovery_triggered = False
        
        # Clear attention cache if memory pressure
        if self.health_status["gpu_utilization"] > self.config["gpu_memory_threshold"]:
            self.clear_attention_cache()
            recovery_triggered = True
            logger.info("Cleared attention cache due to memory pressure")
        
        # Clear old models from cache
        if len(self.loaded_models) > 1:
            # Keep only most recently used model
            most_recent = max(self.loaded_models.items(), 
                            key=lambda x: x[1].get("last_used", 0))
            self.loaded_models = {most_recent[0]: most_recent[1]}
            recovery_triggered = True
            logger.info("Cleared model cache due to memory pressure")
        
        return recovery_triggered
    
    def clear_attention_cache(self):
        """Clear the attention cache"""
        self.attention_cache.clear()
        logger.info("Attention cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Get current health status"""
        status = self.health_status.copy()
        
        # Determine overall status
        if status["gpu_utilization"] > self.config["gpu_memory_threshold"]:
            status["status"] = "warning"
            status["warnings"] = ["High GPU memory usage"]
        elif status["queued_requests"] > 100:
            status["status"] = "warning" 
            status["warnings"] = ["High request queue depth"]
        else:
            status["status"] = "healthy"
            status["warnings"] = []
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.request_metrics:
            return {
                "avg_inference_time_ms": 0,
                "p95_inference_time_ms": 0,
                "requests_per_minute": 0,
                "total_requests": 0
            }
        
        # Calculate metrics
        inference_times = [m["inference_time_ms"] for m in self.request_metrics]
        recent_metrics = [m for m in self.request_metrics 
                         if time.time() - m["timestamp"] < 60]
        
        return {
            "avg_inference_time_ms": np.mean(inference_times),
            "p95_inference_time_ms": np.percentile(inference_times, 95),
            "requests_per_minute": len(recent_metrics),
            "total_requests": len(self.request_metrics),
            "avg_queue_depth": np.mean([m["queue_depth"] for m in self.request_metrics])
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queued_requests": self.request_queue.qsize(),
            "processing_requests": self.config["max_concurrent"] - self.processing_semaphore._value,
            "max_queue_size": self.request_queue.maxsize
        }
    
    def is_ready(self) -> bool:
        """Check if engine is ready to serve requests"""
        return self.health_status["status"] in ["healthy", "warning"]
    
    def gpu_memory_available(self) -> float:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                used = torch.cuda.memory_allocated() / (1024 ** 3)
                return total - used
            elif torch.backends.mps.is_available():
                # MPS doesn't provide memory stats, return estimate
                return 8.0  # Assume 8GB available
            return 0.0
        except:
            return 0.0
    
    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent requests"""
        return self.config["max_concurrent"]


# FastAPI app instance for uvicorn
app = ProductionInferenceEngine().app

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000) 